//! Game-audio and mix check: is a recorded stretch of gameplay (a WAV
//! capture from offscreen rendering) or a rendered/captured scene fit
//! to ship?
//!
//! Four dimensions are measured and gated:
//! 1. **Loudness** (ITU-R BS.1770-4 / EBU Tech 3342): integrated LUFS,
//!    short-term (3 s) maximum, momentary (400 ms) maximum, and
//!    loudness range (LRA).
//! 2. **True peaks & clipping**: 4x-oversampled BS.1770 true peak,
//!    clipped samples and contiguous clipped frame runs.
//! 3. **Silence & dropouts**: BS.1770's `-70 LUFS` and `-10 LU` gates
//!    ignore dead air — a 30 s recording whose audio dies after 2 s
//!    still reads `-18 LUFS` integrated. `SilenceReport` measures every
//!    100 ms step directly against `silence_floor_lufs`, separating
//!    overall silent share and longest silent stretch from interior
//!    dropouts strictly between active audio.
//! 4. **Limiter engagement & ducking clarity**:
//!    - On a bare WAV (`MixTelemetry` absent, or when the mix runs no
//!      master limiter), the audio is replayed through [`Limiter`] at
//!      `ceiling_dbtp` (latency-aligned) to measure what share of 10 ms
//!      blocks would engage (`>= 0.5 dB` GR), how many separate runs
//!      that is, and the deepest/mean reduction.
//!    - When [`MixTelemetry`] is attached (from `af-render` or headless
//!      `af-runtime` / `af-bevy`) and `limiter_configured` is true,
//!      engagement comes from the **actual** master limiter — catching
//!      an over-driven mix that is crushing 14 dB behind a clean
//!      `-1.03 dBTP` output ceiling.
//!    - When [`MixTelemetry`] includes sidechain/ducking windows,
//!      [`DuckCheck`] verifies that marked `sidechain` cues have a
//!      ducked bus wired, actually pull the bed down on sustained cues,
//!      stay audible above the ducked bed (`clarity_db`), do not hold
//!      the bed down continuously (a loop marked `sidechain` by
//!      mistake), and let the bed recover after cues end.

use serde::{Deserialize, Serialize};

use crate::dynamics::{Limiter, LimiterSpec};
use crate::loudness::{self, LoudnessReport};

/// A ~10 ms block counts as limiter-engaged at this much gain reduction.
pub const ENGAGE_THRESHOLD_DB: f64 = 0.5;
/// Limiter replay block length, seconds.
pub const BLOCK_SECS: f64 = 0.01;
/// Unengaged time needed between two runs for them to count as separate
/// engagements (50 ms).
pub const GAP_SECS: f64 = 0.05;
/// Shortest silent run strictly between active steps counted as an
/// interior dropout (300 ms).
pub const MIN_INTERIOR_GAP_SECS: f64 = 0.30;
/// 100 ms window threshold (dBFS RMS at master input) where a
/// `sidechain` cue counts as active.
pub const CUE_ACTIVE_DB: f64 = -55.0;
/// 100 ms window threshold (dBFS RMS at master input) where a ducked
/// bed bus counts as active for cue-to-bed clarity.
pub const BED_ACTIVE_DB: f64 = -70.0;
/// Minimum duck drop (dB) for a 100 ms window to count toward
/// `ducked_percent`.
pub const DUCK_ACTIVE_DB: f64 = 1.0;

/// Shipping gates for [`check`] and [`check_with_telemetry`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlaythroughLimits {
    /// Limiter replay target (and true-peak gate), dBTP.
    pub ceiling_dbtp: f32,
    /// Most clipped samples allowed.
    pub max_clipped: u64,
    /// Lowest allowed integrated loudness, LUFS.
    pub min_lufs: f64,
    /// Highest allowed integrated loudness, LUFS.
    pub max_lufs: f64,
    /// Optional ceiling on 3 s short-term loudness, LUFS.
    pub max_short_term_lufs: Option<f64>,
    /// Highest allowed share of time the limiter is engaged, percent (0..100).
    pub max_engaged_percent: f64,
    /// Deepest allowed limiter gain reduction, dB.
    pub max_reduction_db: f64,
    /// K-weighted step level below which a 100 ms step counts as silent, LUFS.
    pub silence_floor_lufs: f64,
    /// Longest contiguous silent stretch allowed, seconds.
    pub max_silence_secs: f64,
    /// Highest share of 100 ms steps allowed below `silence_floor_lufs`, percent.
    pub max_silent_percent: f64,
    /// Minimum mean duck drop (dB) expected during sustained sidechain cues
    /// (clamped to 40% of the bus's `amount_db` when smaller).
    pub min_cue_drop_db: f64,
    /// Lowest allowed cue-minus-bed level (dB) during sustained sidechain cues.
    pub min_clarity_db: f64,
    /// Highest allowed share of windows a ducked bus stays ducked >= 1 dB, percent.
    pub max_ducked_percent: f64,
    /// Most dropped voices allowed (when telemetry is present).
    pub max_voices_dropped: usize,
}

impl Default for PlaythroughLimits {
    fn default() -> Self {
        Self {
            ceiling_dbtp: -1.0,
            max_clipped: 0,
            min_lufs: -32.0,
            max_lufs: -12.0,
            max_short_term_lufs: None,
            max_engaged_percent: 25.0,
            max_reduction_db: 12.0,
            silence_floor_lufs: -60.0,
            max_silence_secs: 2.5,
            max_silent_percent: 40.0,
            min_cue_drop_db: 3.0,
            min_clarity_db: -3.0,
            max_ducked_percent: 65.0,
            max_voices_dropped: 0,
        }
    }
}

/// Where [`Engagement`] came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngagementSource {
    /// Audio replayed through a limiter at `ceiling_dbtp` (bare WAV or
    /// mix without an active master limiter).
    Replay,
    /// Ground-truth gain reduction recorded from the active master limiter.
    Telemetry,
}

/// How hard the master limiter worked (or would work at `ceiling_dbtp`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Engagement {
    pub source: EngagementSource,
    /// Whether the mix itself had a master limiter enabled (`None` on bare WAV).
    pub limiter_configured: Option<bool>,
    /// Share of blocks engaged (`>= 0.5 dB` GR), percent (0..100).
    pub engaged_percent: f64,
    /// Separate engagements (runs split by `>= 50 ms` unengaged).
    pub engagements: u32,
    /// Deepest gain reduction in any block, dB (`0.0` = never engaged).
    pub max_reduction_db: f64,
    /// Mean gain reduction over engaged blocks, dB.
    pub mean_reduction_db: f64,
    /// `(end_secs, gain_reduction_db)` per block, for charts.
    pub series: Vec<(f64, f64)>,
}

/// Silence and dropout analysis across 100 ms K-weighted steps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SilenceReport {
    pub floor_lufs: f64,
    /// Share of steps below `floor_lufs`, percent (0..100).
    pub silent_percent: f64,
    /// Longest contiguous silent stretch anywhere in the recording, seconds.
    pub longest_silence_secs: f64,
    /// Start time of `longest_silence_secs`, seconds.
    pub longest_silence_at_secs: f64,
    /// Silent stretches `>= 300 ms` strictly between active audio steps.
    pub interior_gaps: u32,
    /// Longest interior dropout, seconds.
    pub longest_interior_gap_secs: f64,
    /// Start time of `longest_interior_gap_secs`, seconds.
    pub longest_interior_gap_at_secs: f64,
    /// Silent stretches `(start_secs, end_secs)` at least 200 ms long, for charts.
    pub intervals: Vec<(f64, f64)>,
}

/// Per-ducked-bus telemetry over consecutive windows (`window_secs`, 100 ms).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BusDuckTelemetry {
    pub bus: String,
    pub amount_db: f32,
    pub hold_ms: f32,
    pub release_ms: f32,
    /// Peak duck drop (dB, `>= 0`) per window.
    pub drop_db: Vec<f64>,
    /// Post-duck, post-gain RMS (dBFS at master summing junction) per window.
    pub bed_rms_db: Vec<f64>,
}

/// Ground-truth mix-stage telemetry captured alongside rendered/pumped audio.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MixTelemetry {
    pub limiter_configured: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limiter_ceiling_dbtp: Option<f32>,
    /// `(end_secs, gr_db)` recorded directly from the master limiter.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub limiter_series: Vec<(f64, f64)>,
    /// Whether any event in the mix is marked `sidechain: true`.
    #[serde(default)]
    pub sidechain_configured: bool,
    /// Window length for `cue_rms_db` and `ducks`, seconds (`0.1`).
    #[serde(default)]
    pub window_secs: f64,
    /// Heard RMS (dBFS at master summing junction) of `sidechain` voices per window.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cue_rms_db: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ducks: Vec<BusDuckTelemetry>,
    #[serde(default)]
    pub voices_dropped: usize,
    #[serde(default)]
    pub voices_stolen: usize,
    #[serde(default)]
    pub triggers_unknown_event: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub missing_sources: Vec<String>,
}

/// Ducking clarity and recovery report for one ducked bus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BusDuckCheck {
    pub bus: String,
    pub target_amount_db: f32,
    pub max_drop_db: f64,
    /// Mean duck drop (dB) across sustained sidechain cue windows.
    pub mean_drop_under_cue_db: f64,
    /// Share of windows ducked `>= 1.0 dB`, percent (0..100).
    pub ducked_percent: f64,
    /// Mean `(cue_rms_db - bed_rms_db)` while both cue and bed sound, dB.
    pub clarity_db: Option<f64>,
    /// Estimated cue-minus-bed level without ducking, dB.
    pub open_clarity_db: Option<f64>,
    /// Whether the bus returned within `0.5 dB` of open after cues ended
    /// (true when no post-release tail exists in the recording).
    pub recovered: bool,
}

/// Sidechain ducking verdict across the mix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DuckCheck {
    pub sidechain_configured: bool,
    pub active_cue_windows: usize,
    pub sustained_cue_windows: usize,
    pub window_secs: f64,
    pub cue_rms_db: Vec<f64>,
    pub buses: Vec<BusDuckCheck>,
    /// `(bus_name, drop_db_per_window)` kept for charts.
    pub drop_series: Vec<(String, Vec<f64>)>,
}

/// Complete verdict on a playthrough WAV or rendered/captured scene.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlaythroughReport {
    pub loudness: LoudnessReport,
    pub silence: SilenceReport,
    pub engagement: Engagement,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ducking: Option<DuckCheck>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voices_dropped: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub triggers_unknown_event: Option<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub missing_sources: Vec<String>,
    /// One entry per broken gate; empty means PASS.
    pub breaches: Vec<String>,
}

impl PlaythroughReport {
    pub fn passed(&self) -> bool {
        self.breaches.is_empty()
    }

    pub fn duration_secs(&self) -> f64 {
        self.loudness.frames as f64 / self.loudness.sample_rate.max(1) as f64
    }
}

/// Waveform envelope for charts: `(min, max)` per column over `columns`
/// equal slices of the file (all channels together).
pub fn envelope(samples: &[f32], channels: u16, columns: usize) -> Vec<(f32, f32)> {
    let ch = channels.max(1) as usize;
    let frames = samples.len() / ch;
    if frames == 0 || columns == 0 {
        return Vec::new();
    }
    (0..columns)
        .map(|c| {
            let a = (c as u128 * frames as u128 / columns as u128) as usize;
            let b = ((c as u128 + 1) * frames as u128 / columns as u128)
                .max(a as u128 + 1)
                .min(frames as u128) as usize;
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for f in a..b {
                for k in 0..ch {
                    let x = samples[f * ch + k];
                    if x.is_finite() {
                        lo = lo.min(x);
                        hi = hi.max(x);
                    }
                }
            }
            if lo <= hi {
                (lo, hi)
            } else {
                (0.0, 0.0)
            }
        })
        .collect()
}

/// Compute [`Engagement`] summary statistics from `(end_secs, gr_db)` blocks.
pub fn engagement_from_series(
    series: Vec<(f64, f64)>,
    source: EngagementSource,
    limiter_configured: Option<bool>,
) -> Engagement {
    if series.is_empty() {
        return Engagement {
            source,
            limiter_configured,
            engaged_percent: 0.0,
            engagements: 0,
            max_reduction_db: 0.0,
            mean_reduction_db: 0.0,
            series,
        };
    }
    let mut engaged = 0usize;
    let mut runs = 0u32;
    let mut unengaged_secs = GAP_SECS;
    let mut max_gr = 0.0f64;
    let mut mean_gr = 0.0f64;
    let mut prev_t = 0.0f64;
    for &(t, gr) in &series {
        let dt = if t.is_finite() {
            let d = (t - prev_t).max(0.0);
            prev_t = t;
            d
        } else {
            0.0
        };
        let gr = if gr.is_finite() { gr.max(0.0) } else { 0.0 };
        if gr >= ENGAGE_THRESHOLD_DB {
            engaged += 1;
            mean_gr += (gr - mean_gr) / engaged as f64;
            if unengaged_secs + 1e-9 >= GAP_SECS {
                runs += 1;
            }
            unengaged_secs = 0.0;
        } else {
            unengaged_secs += dt;
        }
        max_gr = max_gr.max(gr);
    }
    let total = series.len() as f64;
    Engagement {
        source,
        limiter_configured,
        engaged_percent: 100.0 * engaged as f64 / total,
        engagements: runs,
        max_reduction_db: max_gr,
        mean_reduction_db: mean_gr,
        series,
    }
}

/// Replay `samples` through the look-ahead true-peak limiter at
/// `ceiling_dbtp` in 10 ms blocks (latency-aligned so block timestamps
/// match the input waveform and peaks in the final 5 ms are included).
pub fn engagement_from_samples(
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
    ceiling_dbtp: f32,
    limiter_configured: Option<bool>,
) -> Engagement {
    let ch = channels.max(1) as usize;
    let frames = samples.len() / ch;
    let sr = sample_rate.max(1);
    if frames == 0 {
        return engagement_from_series(Vec::new(), EngagementSource::Replay, limiter_configured);
    }
    let block = ((sr as f64 * BLOCK_SECS).round() as usize).max(1);
    let mut lim = Limiter::new(
        LimiterSpec {
            ceiling_dbtp,
            ..Default::default()
        },
        sr,
    );
    let lat = lim.latency();
    let sample_at = |idx: usize| -> (f32, f32) {
        if idx < frames {
            let l = samples[idx * ch];
            let r = if ch > 1 { samples[idx * ch + 1] } else { l };
            (l, r)
        } else {
            (0.0, 0.0)
        }
    };
    // Prime the look-ahead delay line so output frame 0 corresponds to input frame 0.
    for i in 0..lat {
        let (l, r) = sample_at(i);
        let _ = lim.process_frame(l, r);
    }
    let _ = lim.take_min_gain();

    let mut series: Vec<(f64, f64)> = Vec::with_capacity(frames.div_ceil(block));
    let mut at = 0usize;
    while at < frames {
        let n = block.min(frames - at);
        for k in at..at + n {
            let (l, r) = sample_at(k + lat);
            let _ = lim.process_frame(l, r);
        }
        let gr = -20.0 * (lim.take_min_gain() as f64).max(1e-9).log10();
        let gr = if gr.is_finite() { gr.max(0.0) } else { 0.0 };
        series.push((((at + n) as f64) / sr as f64, gr));
        at += n;
    }
    engagement_from_series(series, EngagementSource::Replay, limiter_configured)
}

/// Measure silence share, longest silence, and interior dropouts across
/// 100 ms K-weighted steps.
pub fn measure_silence(
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
    floor_lufs: f64,
) -> SilenceReport {
    let ch = channels.max(1) as usize;
    let frames = samples.len() / ch;
    let sr = sample_rate.max(1) as f64;
    if frames == 0 {
        return SilenceReport {
            floor_lufs,
            silent_percent: 100.0,
            longest_silence_secs: 0.0,
            longest_silence_at_secs: 0.0,
            interior_gaps: 0,
            longest_interior_gap_secs: 0.0,
            longest_interior_gap_at_secs: 0.0,
            intervals: Vec::new(),
        };
    }
    let steps = loudness::step_lufs_series(samples, channels, sample_rate);
    if steps.is_empty() {
        let secs = frames as f64 / sr;
        return SilenceReport {
            floor_lufs,
            silent_percent: 100.0,
            longest_silence_secs: secs,
            longest_silence_at_secs: 0.0,
            interior_gaps: 0,
            longest_interior_gap_secs: 0.0,
            longest_interior_gap_at_secs: 0.0,
            intervals: vec![(0.0, secs)],
        };
    }
    let is_silent: Vec<bool> = steps
        .iter()
        .map(|&(_, _, v)| !v.is_finite() || v < floor_lufs)
        .collect();
    let silent_count = is_silent.iter().filter(|&&s| s).count();
    let first_active = is_silent.iter().position(|&s| !s);
    let last_active = is_silent.iter().rposition(|&s| !s);

    let mut longest_secs = 0.0f64;
    let mut longest_at = 0.0f64;
    let mut interior_gaps = 0u32;
    let mut longest_interior_secs = 0.0f64;
    let mut longest_interior_at = 0.0f64;
    let mut intervals = Vec::new();

    let mut i = 0usize;
    while i < steps.len() {
        if !is_silent[i] {
            i += 1;
            continue;
        }
        let start_idx = i;
        while i < steps.len() && is_silent[i] {
            i += 1;
        }
        let end_idx = i - 1;
        let a = steps[start_idx].0;
        let b = steps[end_idx].1;
        let len = (b - a).max(0.0);
        if len > longest_secs {
            longest_secs = len;
            longest_at = a;
        }
        if len + 1e-9 >= 0.20 {
            intervals.push((a, b));
        }
        if let (Some(fa), Some(la)) = (first_active, last_active) {
            if start_idx > fa && end_idx < la && len + 1e-9 >= MIN_INTERIOR_GAP_SECS {
                interior_gaps += 1;
                if len > longest_interior_secs {
                    longest_interior_secs = len;
                    longest_interior_at = a;
                }
            }
        }
    }

    SilenceReport {
        floor_lufs,
        silent_percent: 100.0 * silent_count as f64 / steps.len() as f64,
        longest_silence_secs: longest_secs,
        longest_silence_at_secs: longest_at,
        interior_gaps,
        longest_interior_gap_secs: longest_interior_secs,
        longest_interior_gap_at_secs: longest_interior_at,
        intervals,
    }
}

fn evaluate_ducking(
    tel: &MixTelemetry,
    duration_secs: f64,
    limits: &PlaythroughLimits,
    breaches: &mut Vec<String>,
) -> Option<DuckCheck> {
    if !tel.sidechain_configured && tel.ducks.is_empty() && tel.cue_rms_db.is_empty() {
        return None;
    }
    let w_secs = if tel.window_secs.is_finite() && tel.window_secs > 0.0 {
        tel.window_secs
    } else {
        0.1
    };
    let active_cue: Vec<bool> = tel
        .cue_rms_db
        .iter()
        .map(|&db| db.is_finite() && db >= CUE_ACTIVE_DB)
        .collect();
    let sustained_cue: Vec<bool> = (0..active_cue.len())
        .map(|w| active_cue[w] && w > 0 && active_cue[w - 1])
        .collect();
    let active_cue_windows = active_cue.iter().filter(|&&x| x).count();
    let sustained_cue_windows = sustained_cue.iter().filter(|&&x| x).count();
    let last_cue_end_secs = active_cue
        .iter()
        .rposition(|&x| x)
        .map(|idx| (idx + 1) as f64 * w_secs);

    if active_cue_windows > 0 && tel.ducks.is_empty() {
        breaches.push("sidechain events played, but no bus has duck configured".to_string());
    }

    let mut buses = Vec::with_capacity(tel.ducks.len());
    let mut drop_series = Vec::with_capacity(tel.ducks.len());
    for d in &tel.ducks {
        let n = d.drop_db.len().max(1);
        let max_drop =
            d.drop_db
                .iter()
                .copied()
                .fold(0.0f64, |m, v| if v.is_finite() { m.max(v) } else { m });
        let ducked_wins = d
            .drop_db
            .iter()
            .filter(|&&v| v.is_finite() && v >= DUCK_ACTIVE_DB)
            .count();
        let ducked_percent = 100.0 * ducked_wins as f64 / n as f64;

        // Evaluate duck depth & cue-to-bed clarity over sustained cue windows
        // (past the 80 ms attack ramp); fall back to all active cue windows for
        // clarity if only sub-200 ms cues occurred.
        let use_sustained = sustained_cue_windows > 0;
        let mask = if use_sustained {
            &sustained_cue
        } else {
            &active_cue
        };
        let mut mean_drop = 0.0f64;
        let mut drop_cnt = 0usize;
        let mut mean_clar = 0.0f64;
        let mut clar_cnt = 0usize;
        for (w, (&raw_drop, &is_active)) in d.drop_db.iter().zip(mask.iter()).enumerate() {
            if !is_active {
                continue;
            }
            let dr = if raw_drop.is_finite() {
                raw_drop.max(0.0)
            } else {
                0.0
            };
            drop_cnt += 1;
            mean_drop += (dr - mean_drop) / drop_cnt as f64;
            if let (Some(&cue), Some(&bed)) = (tel.cue_rms_db.get(w), d.bed_rms_db.get(w)) {
                if cue.is_finite() && bed.is_finite() && bed >= BED_ACTIVE_DB {
                    let diff = cue - bed;
                    if diff.is_finite() {
                        clar_cnt += 1;
                        mean_clar += (diff - mean_clar) / clar_cnt as f64;
                    }
                }
            }
        }
        let mean_drop_under_cue = mean_drop;
        let clarity_db = (clar_cnt > 0).then_some(mean_clar);
        let open_clarity_db = clarity_db.map(|c| c - mean_drop_under_cue);

        let hold_ms = if d.hold_ms.is_finite() {
            d.hold_ms.max(0.0) as f64
        } else {
            0.0
        };
        let release_ms = if d.release_ms.is_finite() {
            d.release_ms.max(0.0) as f64
        } else {
            0.0
        };
        let amount_db = if d.amount_db.is_finite() {
            d.amount_db.max(0.0)
        } else {
            0.0
        };
        let settle_needed = (hold_ms + release_ms) * 1e-3 + 0.20;
        let recovered = match last_cue_end_secs {
            Some(end_t) if duration_secs >= end_t + settle_needed => {
                let last = d.drop_db.last().copied().unwrap_or(0.0);
                last.is_finite() && last <= 0.5
            }
            _ => true,
        };

        let expected_drop = limits.min_cue_drop_db.min(amount_db as f64 * 0.4);
        if use_sustained && (expected_drop.is_nan() || mean_drop_under_cue < expected_drop) {
            breaches.push(format!(
                "bus {:?} ducked only {:.1} dB under sustained sidechain cues < {:.1} dB",
                d.bus, mean_drop_under_cue, expected_drop
            ));
        }
        if let Some(clar) = clarity_db {
            if use_sustained && (limits.min_clarity_db.is_nan() || clar < limits.min_clarity_db) {
                breaches.push(format!(
                    "bus {:?} drowns sidechain cues (clarity {:+.1} dB < {:+.1} dB)",
                    d.bus, clar, limits.min_clarity_db
                ));
            }
        }
        if limits.max_ducked_percent.is_nan() || ducked_percent > limits.max_ducked_percent {
            breaches.push(format!(
                "bus {:?} ducked {:.1}% of the time > {:.1}%",
                d.bus, ducked_percent, limits.max_ducked_percent
            ));
        }
        if !recovered {
            breaches.push(format!(
                "bus {:?} did not recover after sidechain cues ended",
                d.bus
            ));
        }

        drop_series.push((d.bus.clone(), d.drop_db.clone()));
        buses.push(BusDuckCheck {
            bus: d.bus.clone(),
            target_amount_db: amount_db,
            max_drop_db: max_drop,
            mean_drop_under_cue_db: mean_drop_under_cue,
            ducked_percent,
            clarity_db,
            open_clarity_db,
            recovered,
        });
    }

    Some(DuckCheck {
        sidechain_configured: tel.sidechain_configured || active_cue_windows > 0,
        active_cue_windows,
        sustained_cue_windows,
        window_secs: w_secs,
        cue_rms_db: tel.cue_rms_db.clone(),
        buses,
        drop_series,
    })
}

/// Measure a bare WAV recording and lay it against [`PlaythroughLimits`].
pub fn check(
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
    limits: &PlaythroughLimits,
) -> PlaythroughReport {
    check_with_telemetry(samples, channels, sample_rate, limits, None)
}

/// Measure audio plus optional [`MixTelemetry`] (ground-truth limiter
/// engagement, sidechain ducking clarity, and voice drop counts).
pub fn check_with_telemetry(
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
    limits: &PlaythroughLimits,
    telemetry: Option<&MixTelemetry>,
) -> PlaythroughReport {
    let sr = sample_rate.max(1);
    let loud = loudness::measure(samples, channels, sr);
    let silence = measure_silence(samples, channels, sr, limits.silence_floor_lufs);
    let eng = match telemetry {
        Some(tel) if tel.limiter_configured && !tel.limiter_series.is_empty() => {
            engagement_from_series(
                tel.limiter_series.clone(),
                EngagementSource::Telemetry,
                Some(true),
            )
        }
        Some(tel) => engagement_from_samples(
            samples,
            channels,
            sr,
            limits.ceiling_dbtp,
            Some(tel.limiter_configured),
        ),
        None => engagement_from_samples(samples, channels, sr, limits.ceiling_dbtp, None),
    };

    let mut breaches = Vec::new();
    // Guard NaN thresholds so malformed limits can never silently pass.
    if limits.ceiling_dbtp.is_nan()
        || limits.min_lufs.is_nan()
        || limits.max_lufs.is_nan()
        || limits.max_short_term_lufs.is_some_and(f64::is_nan)
        || limits.max_engaged_percent.is_nan()
        || limits.max_reduction_db.is_nan()
        || limits.silence_floor_lufs.is_nan()
        || limits.max_silence_secs.is_nan()
        || limits.max_silent_percent.is_nan()
        || limits.min_cue_drop_db.is_nan()
        || limits.min_clarity_db.is_nan()
        || limits.max_ducked_percent.is_nan()
    {
        breaches.push("invalid NaN threshold in PlaythroughLimits".to_string());
    }
    // Allow 0.02 dB float/WAV rounding tolerance around ceiling_dbtp.
    if loud.true_peak_dbtp > limits.ceiling_dbtp as f64 + 0.02 {
        breaches.push(format!(
            "true peak {:+.2} dBTP > ceiling {:+.2}",
            loud.true_peak_dbtp, limits.ceiling_dbtp
        ));
    }
    if loud.clipped_samples > limits.max_clipped {
        breaches.push(format!(
            "{} clipped samples ({} runs, longest {} frames) > {}",
            loud.clipped_samples, loud.clipped_runs, loud.max_clipped_run, limits.max_clipped
        ));
    }
    if !loud.integrated_lufs.is_finite() || loud.integrated_lufs < limits.min_lufs {
        breaches.push(format!(
            "integrated {:.2} LUFS < {:.2}",
            loud.integrated_lufs, limits.min_lufs
        ));
    }
    if loud.integrated_lufs > limits.max_lufs {
        breaches.push(format!(
            "integrated {:.2} LUFS > {:.2}",
            loud.integrated_lufs, limits.max_lufs
        ));
    }
    if let Some(st_max) = limits.max_short_term_lufs {
        if loud.short_term_max_lufs > st_max {
            breaches.push(format!(
                "short-term max {:.2} LUFS > {:.2}",
                loud.short_term_max_lufs, st_max
            ));
        }
    }
    if silence.longest_silence_secs > limits.max_silence_secs {
        breaches.push(format!(
            "longest silence {:.2} s (at {:.2} s) > {:.2} s",
            silence.longest_silence_secs, silence.longest_silence_at_secs, limits.max_silence_secs
        ));
    }
    if silence.silent_percent > limits.max_silent_percent {
        breaches.push(format!(
            "silent {:.1}% of steps (< {:.0} LUFS) > {:.1}%",
            silence.silent_percent, silence.floor_lufs, limits.max_silent_percent
        ));
    }
    if eng.engaged_percent > limits.max_engaged_percent {
        breaches.push(format!(
            "limiter engaged {:.1}% of the time > {:.1}%",
            eng.engaged_percent, limits.max_engaged_percent
        ));
    }
    if eng.max_reduction_db > limits.max_reduction_db {
        breaches.push(format!(
            "limiter pulled {:.1} dB > {:.1} dB",
            eng.max_reduction_db, limits.max_reduction_db
        ));
    }

    let duration_secs = loud.frames as f64 / sr as f64;
    let mut ducking = None;
    let mut voices_dropped = None;
    let mut triggers_unknown = None;
    let mut missing_sources = Vec::new();
    if let Some(tel) = telemetry {
        ducking = evaluate_ducking(tel, duration_secs, limits, &mut breaches);
        voices_dropped = Some(tel.voices_dropped);
        triggers_unknown = Some(tel.triggers_unknown_event);
        missing_sources = tel.missing_sources.clone();
        if tel.voices_dropped > limits.max_voices_dropped {
            breaches.push(format!(
                "{} voices dropped > {}",
                tel.voices_dropped, limits.max_voices_dropped
            ));
        }
        if tel.triggers_unknown_event > 0 {
            breaches.push(format!(
                "{} unknown event triggers",
                tel.triggers_unknown_event
            ));
        }
        if !tel.missing_sources.is_empty() {
            breaches.push(format!(
                "{} missing audio source(s): {}",
                tel.missing_sources.len(),
                tel.missing_sources.join(", ")
            ));
        }
    }

    PlaythroughReport {
        loudness: loud,
        silence,
        engagement: eng,
        ducking,
        voices_dropped,
        triggers_unknown_event: triggers_unknown,
        missing_sources,
        breaches,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;

    fn sine(amp: f32, secs: f32) -> Vec<f32> {
        (0..(secs * SR as f32) as usize)
            .flat_map(|i| {
                let v = amp * (std::f32::consts::TAU * 1000.0 * i as f32 / SR as f32).sin();
                [v, v]
            })
            .collect()
    }

    #[test]
    fn quiet_playthrough_passes_and_never_engages() {
        // 1 kHz at -23 dBFS (EBU reference level, peak 0.071).
        let r = check(&sine(0.071, 5.0), 2, SR, &PlaythroughLimits::default());
        assert!(r.passed(), "{:?}", r.breaches);
        assert_eq!(r.engagement.engaged_percent, 0.0);
        assert_eq!(r.engagement.engagements, 0);
        assert_eq!(r.silence.silent_percent, 0.0);
        assert_eq!(r.silence.interior_gaps, 0);
    }

    #[test]
    fn planted_clipping_fails() {
        // Planted bad case 1: -6 dBFS signal driven +12 dB (+6 dBFS) and hard-clipped at +-1.
        let hot: Vec<f32> = sine(0.5, 5.0)
            .iter()
            .map(|x| (x * 4.0).clamp(-1.0, 1.0))
            .collect();
        let r = check(&hot, 2, SR, &PlaythroughLimits::default());
        assert!(!r.passed(), "planted clipping must fail");
        assert!(r.loudness.clipped_samples > 1000, "{:?}", r.loudness);
        assert!(r.loudness.max_clipped_run > 5, "{:?}", r.loudness);
        assert!(
            r.breaches.iter().any(|b| b.contains("clipped")),
            "{:?}",
            r.breaches
        );
        assert!(
            r.breaches.iter().any(|b| b.contains("true peak")),
            "{:?}",
            r.breaches
        );
    }

    #[test]
    fn planted_dropout_caught_even_when_bs1770_gates_silence_out() {
        // Planted bad case 2: 2 s at -20 LUFS, 4 s dead dropout, 2 s at -20 LUFS.
        // BS.1770 integrated loudness gates the silence out (~-20.6 LUFS)!
        let mut x = sine(0.1, 2.0);
        x.extend(vec![0.0; SR as usize * 2 * 4]);
        x.extend(sine(0.1, 2.0));
        let r = check(&x, 2, SR, &PlaythroughLimits::default());
        assert!(
            (-22.0..-19.5).contains(&r.loudness.integrated_lufs),
            "{}",
            r.loudness.integrated_lufs
        );
        assert!(!r.passed(), "4 s dropout must fail: {:?}", r);
        assert_eq!(r.silence.interior_gaps, 1);
        assert!((r.silence.longest_interior_gap_secs - 4.0).abs() < 0.15);
        assert!(r.breaches.iter().any(|b| b.contains("longest silence")));
    }

    #[test]
    fn telemetry_catches_overdriven_post_limiter_mix_that_bare_wav_misses() {
        // Planted bad case 3: audio already limited to -1.03 dBTP (amp ~0.88),
        // so bare WAV replay sees 0% engagement, while ground-truth telemetry
        // shows the game's master limiter was slamming 15 dB 80% of the time.
        let post_lim = sine(0.2, 4.0);
        let bare = check(&post_lim, 2, SR, &PlaythroughLimits::default());
        assert!(bare.passed());
        assert_eq!(bare.engagement.engaged_percent, 0.0);

        let tel = MixTelemetry {
            limiter_configured: true,
            limiter_ceiling_dbtp: Some(-1.0),
            limiter_series: (1..=400)
                .map(|i| (i as f64 * 0.01, if i <= 320 { 15.0 } else { 0.0 }))
                .collect(),
            ..Default::default()
        };
        let with_tel =
            check_with_telemetry(&post_lim, 2, SR, &PlaythroughLimits::default(), Some(&tel));
        assert!(!with_tel.passed(), "over-limited mix must fail");
        assert_eq!(with_tel.engagement.source, EngagementSource::Telemetry);
        assert!((with_tel.engagement.engaged_percent - 80.0).abs() < 0.1);
        assert!(with_tel
            .breaches
            .iter()
            .any(|b| b.contains("limiter engaged")));
        assert!(with_tel
            .breaches
            .iter()
            .any(|b| b.contains("limiter pulled")));
    }

    #[test]
    fn duck_clarity_catches_unwired_buried_and_stuck_sidechains() {
        let audio = sine(0.1, 4.0);
        // 1) Sidechain cues played, no bus ducked -> fail.
        let unwired = MixTelemetry {
            sidechain_configured: true,
            window_secs: 0.1,
            cue_rms_db: vec![-12.0; 10],
            ..Default::default()
        };
        let r = check_with_telemetry(&audio, 2, SR, &PlaythroughLimits::default(), Some(&unwired));
        assert!(r.breaches.iter().any(|b| b.contains("no bus has duck")));

        // 2) Cue buried (-18 dB) under loud bed (-6 dB) -> clarity -12 dB -> fail.
        let buried = MixTelemetry {
            sidechain_configured: true,
            window_secs: 0.1,
            cue_rms_db: vec![-18.0; 20],
            ducks: vec![BusDuckTelemetry {
                bus: "music".into(),
                amount_db: 9.0,
                hold_ms: 200.0,
                release_ms: 600.0,
                drop_db: vec![6.0; 20],
                bed_rms_db: vec![-6.0; 20],
            }],
            ..Default::default()
        };
        let r = check_with_telemetry(&audio, 2, SR, &PlaythroughLimits::default(), Some(&buried));
        assert!(r
            .breaches
            .iter()
            .any(|b| b.contains("drowns sidechain cues")));

        // 3) Clean cue (-10 dB) over ducked bed (-22 dB) for 1 s, then recovers -> pass!
        let mut cue = vec![-120.0; 40];
        let mut drop = vec![0.0; 40];
        let mut bed = vec![-13.0; 40];
        for i in 5..15 {
            cue[i] = -10.0;
            drop[i] = 9.0;
            bed[i] = -22.0;
        }
        let clean = MixTelemetry {
            sidechain_configured: true,
            window_secs: 0.1,
            cue_rms_db: cue,
            ducks: vec![BusDuckTelemetry {
                bus: "music".into(),
                amount_db: 9.0,
                hold_ms: 200.0,
                release_ms: 600.0,
                drop_db: drop,
                bed_rms_db: bed,
            }],
            ..Default::default()
        };
        let r = check_with_telemetry(&audio, 2, SR, &PlaythroughLimits::default(), Some(&clean));
        assert!(r.passed(), "{:?}", r.breaches);
        let d = &r.ducking.unwrap().buses[0];
        assert!((d.clarity_db.unwrap() - 12.0).abs() < 0.01);
        assert!(d.recovered);
    }

    #[test]
    fn cant_tell_control_same_file_same_verdict() {
        let x = sine(0.25, 5.0);
        let a = check(&x, 2, SR, &PlaythroughLimits::default());
        let b = check(&x, 2, SR, &PlaythroughLimits::default());
        assert_eq!(a, b);
    }

    #[test]
    fn one_burst_is_one_engagement_including_near_file_end() {
        let mut x = sine(0.05, 10.0);
        let burst = sine(1.8, 1.0);
        let at = 5 * SR as usize * 2;
        x[at..at + burst.len()].copy_from_slice(&burst);
        let r = check(&x, 2, SR, &PlaythroughLimits::default());
        assert_eq!(r.engagement.engagements, 1, "{:?}", r.engagement);
        assert!(
            (8.0..15.0).contains(&r.engagement.engaged_percent),
            "{:?}",
            r.engagement
        );
        assert!(r.engagement.max_reduction_db > 6.0);

        // A peak in the final 4 ms (inside the limiter's 5.2 ms look-ahead) is still caught.
        let mut tail = sine(0.05, 4.0);
        let n = tail.len();
        for s in &mut tail[n - 300..] {
            *s *= 36.0;
        }
        let rt = check(&tail, 2, SR, &PlaythroughLimits::default());
        assert!(rt.engagement.max_reduction_db > 4.0, "{:?}", rt.engagement);
    }

    #[test]
    fn silence_fails_loudness_and_silence_gates() {
        let r = check(
            &vec![0.0; SR as usize * 2 * 4],
            2,
            SR,
            &PlaythroughLimits::default(),
        );
        assert!(!r.passed());
        assert!(r.breaches.iter().any(|b| b.contains("LUFS <")));
        assert!(r.breaches.iter().any(|b| b.contains("longest silence")));
    }

    #[test]
    fn envelope_covers_the_file() {
        let e = envelope(&sine(0.5, 1.0), 2, 48);
        assert_eq!(e.len(), 48);
        assert!(e.iter().all(|(lo, hi)| *lo >= -0.51 && *hi <= 0.51));
        assert!(e.iter().any(|(lo, hi)| hi - lo > 0.9));
    }

    #[test]
    fn handles_nonfinite_envelope_and_telemetry() {
        let e = envelope(&[f32::NAN; 8], 1, 2);
        assert!(e
            .iter()
            .all(|(lo, hi)| lo.is_finite() && hi.is_finite() && lo <= hi));

        let eng = engagement_from_series(
            vec![(0.5, 12.0), (f64::NAN, 0.0), (0.8, 0.0), (1.0, 12.0)],
            EngagementSource::Telemetry,
            Some(true),
        );
        assert_eq!(eng.engagements, 2);

        let audio = sine(0.1, 2.0);
        let tel = MixTelemetry {
            window_secs: f64::NAN,
            sidechain_configured: true,
            cue_rms_db: vec![-12.0, -12.0, -40.0],
            ducks: vec![BusDuckTelemetry {
                bus: "bed".into(),
                amount_db: 12.0,
                hold_ms: 80.0,
                release_ms: 250.0,
                drop_db: vec![0.0, f64::INFINITY, 0.0],
                bed_rms_db: vec![-24.0, -24.0, -24.0],
            }],
            ..Default::default()
        };
        let r = check_with_telemetry(&audio, 2, SR, &PlaythroughLimits::default(), Some(&tel));
        let d = &r.ducking.unwrap().buses[0];
        assert!(d.mean_drop_under_cue_db.is_finite());
    }
}
