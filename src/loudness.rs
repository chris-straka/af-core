//! Loudness and peak measurement of a finished mix (ITU-R BS.1770-4 /
//! EBU R128 / Tech 3342): integrated loudness, short-term maximum,
//! loudness range (LRA), true peak and clipped-sample / clipped-run
//! counts. Offline only (allocates).
//!
//! - K-weighting: the two BS.1770 biquads, derived for any sample rate
//!   (same formulas as libebur128, which ffmpeg's `ebur128` follows).
//! - Integrated: 400 ms blocks every 100 ms, absolute gate -70 LUFS,
//!   relative gate -10 LU.
//! - Short-term: 3 s windows every 100 ms (full windows only; a file
//!   shorter than 3 s reports its whole length as one window).
//! - Loudness range (LRA): EBU Tech 3342 10th-to-95th percentile of
//!   3 s short-term blocks after -70 LUFS absolute and -20 LU relative
//!   gating (`0.0` when fewer than two 3 s blocks pass gating).
//! - True peak: 4x oversampled with the BS.1770 Annex 2 interpolator
//!   (inside the file: the edges are not padded with silence).
//! - Clipped: samples at or past 16-bit full scale (|x| >= 32767/32768),
//!   plus contiguous clipped frame runs (`max_clipped_run` distinguishes
//!   a grazed peak from flat-topped clamping).

use serde::{Deserialize, Serialize};

use crate::dynamics::{TP_TAPS, TRUE_PEAK_PHASES};

/// |x| at or above this counts as clipped (16-bit full scale).
pub const CLIP_LEVEL: f32 = 32_767.0 / 32_768.0;

/// ITU-R BS.1770-4 loudness calibration offset (dB).
pub const CALIBRATION_OFFSET_DB: f64 = -0.691;
/// Absolute gate threshold (LUFS): 400 ms blocks quieter than this are excluded.
pub const ABSOLUTE_GATE_LUFS: f64 = -70.0;
/// Relative gate offset (LU) below the absolute-gated mean.
pub const RELATIVE_GATE_LU: f64 = -10.0;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LoudnessReport {
    pub sample_rate: u32,
    pub channels: u16,
    pub frames: usize,
    /// Integrated loudness, LUFS (`-inf` when every block is gated out).
    pub integrated_lufs: f64,
    /// Loudest 3 s window, LUFS.
    pub short_term_max_lufs: f64,
    /// Loudest 400 ms window, LUFS.
    pub momentary_max_lufs: f64,
    /// EBU Tech 3342 loudness range across 3 s windows, LU.
    pub loudness_range_lu: f64,
    /// dBTP.
    pub true_peak_dbtp: f64,
    /// dBFS.
    pub sample_peak_dbfs: f64,
    pub clipped_samples: u64,
    /// Contiguous runs of frames where at least one channel clipped.
    pub clipped_runs: u32,
    /// Longest contiguous run of clipped frames.
    pub max_clipped_run: usize,
}

#[derive(Debug, Clone, Copy)]
struct Biquad {
    b: [f64; 3],
    a: [f64; 2],
    z: [f64; 2],
}

impl Biquad {
    #[inline]
    fn run(&mut self, x: f64) -> f64 {
        // Transposed direct form II.
        let y = self.b[0] * x + self.z[0];
        self.z[0] = self.b[1] * x - self.a[0] * y + self.z[1];
        self.z[1] = self.b[2] * x - self.a[1] * y;
        y
    }

    fn reset(&mut self) {
        self.z = [0.0; 2];
    }
}

/// Two-stage ITU-R BS.1770 K-weighting filter (head-model high-shelf
/// followed by the 38.135 Hz RLB high-pass), derived for any sample rate.
/// Allocation-free on [`KWeighting::process`] for real-time metering.
#[derive(Debug, Clone, Copy)]
pub struct KWeighting {
    sample_rate: f64,
    shelf: Biquad,
    hp: Biquad,
}

impl KWeighting {
    pub fn new(sample_rate: f64) -> Self {
        let fs = if sample_rate.is_finite() {
            sample_rate.max(4_000.0)
        } else {
            48_000.0
        };
        let (shelf, hp) = k_weighting(fs);
        Self {
            sample_rate: fs,
            shelf,
            hp,
        }
    }

    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    pub fn reset(&mut self) {
        self.shelf.reset();
        self.hp.reset();
    }

    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        let clean = if x.is_finite() { x } else { 0.0 };
        self.hp.run(self.shelf.run(clean))
    }

    #[inline]
    pub fn process_f32(&mut self, x: f32) -> f64 {
        self.process(x as f64)
    }
}

/// The BS.1770 pre-filter (high shelf) and RLB high-pass at `fs`.
fn k_weighting(fs: f64) -> (Biquad, Biquad) {
    use std::f64::consts::PI;
    let f0 = 1681.974450955533;
    let g = 3.999843853973347;
    let q = 0.7071752369554196;
    let k = (PI * f0 / fs).tan();
    let vh = 10f64.powf(g / 20.0);
    let vb = vh.powf(0.4996667741545416);
    let a0 = 1.0 + k / q + k * k;
    let shelf = Biquad {
        b: [
            (vh + vb * k / q + k * k) / a0,
            2.0 * (k * k - vh) / a0,
            (vh - vb * k / q + k * k) / a0,
        ],
        a: [2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
        z: [0.0; 2],
    };
    let f0 = 38.13547087602444;
    let q = 0.5003270373238773;
    let k = (PI * f0 / fs).tan();
    let a0 = 1.0 + k / q + k * k;
    let hp = Biquad {
        b: [1.0, -2.0, 1.0],
        a: [2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
        z: [0.0; 2],
    };
    (shelf, hp)
}

/// Convert mean K-weighted square power to LUFS (`-inf` when `mean_power <= 0`).
#[inline]
pub fn energy_to_lufs(mean_power: f64) -> f64 {
    if mean_power <= 0.0 {
        f64::NEG_INFINITY
    } else {
        CALIBRATION_OFFSET_DB + 10.0 * mean_power.log10()
    }
}

/// BS.1770 two-stage gated integrated loudness from 400 ms block mean
/// powers (`None` when every block falls below `-70.0` LUFS).
pub fn integrated_from_block_powers(blocks: &[f64]) -> Option<f64> {
    let abs: Vec<f64> = blocks
        .iter()
        .copied()
        .filter(|&z| energy_to_lufs(z) >= ABSOLUTE_GATE_LUFS)
        .collect();
    if abs.is_empty() {
        return None;
    }
    let rel = energy_to_lufs(abs.iter().sum::<f64>() / abs.len() as f64) + RELATIVE_GATE_LU;
    let gated: Vec<f64> = abs
        .into_iter()
        .filter(|&z| energy_to_lufs(z) >= rel)
        .collect();
    if gated.is_empty() {
        None
    } else {
        Some(energy_to_lufs(
            gated.iter().sum::<f64>() / gated.len() as f64,
        ))
    }
}

fn to_db(x: f64) -> f64 {
    if x <= 0.0 {
        f64::NEG_INFINITY
    } else {
        20.0 * x.log10()
    }
}

/// True peak (linear) of interleaved audio: max |x| over the samples
/// and the 4x-interpolated points between them. Only points whose 12
/// taps lie inside the file count: a file cut off mid-waveform does not
/// read the cut itself as an over.
pub fn true_peak(samples: &[f32], channels: u16) -> f32 {
    let ch = channels.max(1) as usize;
    let mut peak = 0.0f32;
    for c in 0..ch {
        let mut hist = [0.0f32; TP_TAPS];
        for (i, raw) in samples.iter().skip(c).step_by(ch).copied().enumerate() {
            let x = if raw.is_finite() { raw } else { 0.0 };
            hist.copy_within(0..TP_TAPS - 1, 1);
            hist[0] = x;
            peak = peak.max(x.abs());
            if i + 1 < TP_TAPS {
                continue;
            }
            for ph in &TRUE_PEAK_PHASES {
                let y: f32 = ph.iter().zip(hist.iter()).map(|(a, b)| a * b).sum();
                peak = peak.max(y.abs());
            }
        }
    }
    peak
}

/// One K-weighted pass over the file: per-100 ms sums of squares (all
/// channels together), optional trailing partial step, sample peak and
/// clipping statistics.
struct KPass {
    steps: Vec<f64>,
    tail: Option<(usize, f64)>,
    step_frames: usize,
    peak: f32,
    clipped: u64,
    clipped_runs: u32,
    max_clipped_run: usize,
}

fn k_pass(samples: &[f32], channels: u16, sample_rate: u32) -> KPass {
    let ch = channels.max(1) as usize;
    let frames = samples.len() / ch;
    let fs = sample_rate.max(1) as f64;
    // K-weighted power per 100 ms step (sum over channels, G = 1 for
    // L/R/mono per BS.1770).
    let step = ((fs * 0.1).round() as usize).max(1);
    let mut steps: Vec<f64> = Vec::with_capacity(frames / step + 1);
    let mut filters: Vec<KWeighting> = (0..ch).map(|_| KWeighting::new(fs)).collect();
    let mut acc = 0.0f64;
    let mut in_step = 0usize;
    let mut peak = 0.0f32;
    let mut clipped = 0u64;
    let mut clipped_runs = 0u32;
    let mut cur_run = 0usize;
    let mut max_clipped_run = 0usize;
    for f in 0..frames {
        let mut frame_clipped = false;
        for (c, k) in filters.iter_mut().enumerate() {
            let raw = samples[f * ch + c];
            let x = if raw.is_finite() { raw } else { 0.0 };
            let a = x.abs();
            peak = peak.max(a);
            if a >= CLIP_LEVEL {
                clipped += 1;
                frame_clipped = true;
            }
            let y = k.process_f32(x);
            acc += y * y;
        }
        if frame_clipped {
            if cur_run == 0 {
                clipped_runs += 1;
            }
            cur_run += 1;
            max_clipped_run = max_clipped_run.max(cur_run);
        } else {
            cur_run = 0;
        }
        in_step += 1;
        if in_step == step {
            steps.push(acc);
            acc = 0.0;
            in_step = 0;
        }
    }
    KPass {
        steps,
        tail: (in_step > 0).then_some((in_step, acc)),
        step_frames: step,
        peak,
        clipped,
        clipped_runs,
        max_clipped_run,
    }
}

fn prefix_sums(steps: &[f64]) -> Vec<f64> {
    let mut prefix = Vec::with_capacity(steps.len() + 1);
    prefix.push(0.0);
    let mut sum = 0.0;
    for &s in steps {
        sum += s;
        prefix.push(sum);
    }
    prefix
}

/// Short-term loudness over time, for charts: one `(end_secs, LUFS)`
/// point per 100 ms step, each the loudness of the up-to-3 s window
/// ending at that step (shorter at the start of the file). Silence
/// reads as `-inf`.
pub fn short_term_series(samples: &[f32], channels: u16, sample_rate: u32) -> Vec<(f64, f64)> {
    let p = k_pass(samples, channels, sample_rate);
    let sr = sample_rate.max(1) as f64;
    let prefix = prefix_sums(&p.steps);
    p.steps
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let start = (i + 1).saturating_sub(30);
            let k = i + 1 - start;
            let mean = (prefix[i + 1] - prefix[start]) / (k * p.step_frames) as f64;
            (((i + 1) * p.step_frames) as f64 / sr, energy_to_lufs(mean))
        })
        .collect()
}

/// Per-100 ms step K-weighted loudness `(start_secs, end_secs, LUFS)`,
/// plus any trailing partial step at least 20 ms long. Used by silence
/// and dropout detection so stretches gated out by BS.1770 are still
/// measured directly.
pub fn step_lufs_series(samples: &[f32], channels: u16, sample_rate: u32) -> Vec<(f64, f64, f64)> {
    let p = k_pass(samples, channels, sample_rate);
    let sr = sample_rate.max(1) as f64;
    let mut out: Vec<(f64, f64, f64)> = p
        .steps
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let a = (i * p.step_frames) as f64 / sr;
            let b = ((i + 1) * p.step_frames) as f64 / sr;
            (a, b, energy_to_lufs(s / p.step_frames as f64))
        })
        .collect();
    if let Some((n, s)) = p.tail {
        if n * 5 >= p.step_frames || p.steps.is_empty() {
            let a = (p.steps.len() * p.step_frames) as f64 / sr;
            let b = (p.steps.len() * p.step_frames + n) as f64 / sr;
            out.push((a, b, energy_to_lufs(s / n as f64)));
        }
    }
    out
}

/// EBU Tech 3342 loudness range from 3 s short-term block mean powers.
fn loudness_range(st_powers: &[f64]) -> f64 {
    let abs: Vec<f64> = st_powers
        .iter()
        .copied()
        .filter(|&z| z.is_finite() && energy_to_lufs(z) >= ABSOLUTE_GATE_LUFS)
        .collect();
    if abs.len() < 2 {
        return 0.0;
    }
    let rel = energy_to_lufs(abs.iter().sum::<f64>() / abs.len() as f64) - 20.0;
    let mut gated: Vec<f64> = abs
        .into_iter()
        .map(energy_to_lufs)
        .filter(|&v| v >= rel)
        .collect();
    if gated.len() < 2 {
        return 0.0;
    }
    gated.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f64| -> f64 {
        let pos = p * (gated.len() - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let frac = pos - lo as f64;
        gated[lo] + (gated[hi] - gated[lo]) * frac
    };
    (pct(0.95) - pct(0.10)).max(0.0)
}

/// Measure interleaved audio (1 or 2 channels, any rate).
pub fn measure(samples: &[f32], channels: u16, sample_rate: u32) -> LoudnessReport {
    let ch = channels.max(1) as usize;
    let frames = samples.len() / ch;
    let pass = k_pass(samples, channels, sample_rate);
    let prefix = prefix_sums(&pass.steps);
    let step = pass.step_frames;
    let window = |i: usize, k: usize| -> f64 { (prefix[i + k] - prefix[i]) / (k * step) as f64 };
    let blocks: Vec<f64> = if pass.steps.len() >= 4 {
        (0..=pass.steps.len() - 4).map(|i| window(i, 4)).collect()
    } else if !pass.steps.is_empty() {
        vec![window(0, pass.steps.len())]
    } else {
        vec![]
    };
    let integrated = integrated_from_block_powers(&blocks).unwrap_or(f64::NEG_INFINITY);
    let momentary = blocks
        .iter()
        .map(|&z| energy_to_lufs(z))
        .fold(f64::NEG_INFINITY, f64::max);
    let (short, lra) = if pass.steps.len() >= 30 {
        let st: Vec<f64> = (0..=pass.steps.len() - 30).map(|i| window(i, 30)).collect();
        let smax = st
            .iter()
            .map(|&z| energy_to_lufs(z))
            .fold(f64::NEG_INFINITY, f64::max);
        (smax, loudness_range(&st))
    } else if !pass.steps.is_empty() {
        (energy_to_lufs(window(0, pass.steps.len())), 0.0)
    } else {
        (f64::NEG_INFINITY, 0.0)
    };
    let tp = true_peak(&samples[..frames * ch], channels);
    LoudnessReport {
        sample_rate,
        channels,
        frames,
        integrated_lufs: integrated,
        short_term_max_lufs: short,
        momentary_max_lufs: momentary,
        loudness_range_lu: lra,
        true_peak_dbtp: to_db(tp as f64),
        sample_peak_dbfs: to_db(pass.peak as f64),
        clipped_samples: pass.clipped,
        clipped_runs: pass.clipped_runs,
        max_clipped_run: pass.max_clipped_run,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, amp: f32, secs: f32, sr: u32, phase: f32) -> Vec<f32> {
        let n = (secs * sr as f32) as usize;
        (0..n)
            .flat_map(|i| {
                let v = amp * (std::f32::consts::TAU * freq * i as f32 / sr as f32 + phase).sin();
                [v, v]
            })
            .collect()
    }

    #[test]
    fn reference_tone_reads_as_bs1770_says() {
        // EBU Tech 3341 case 1: a stereo 1 kHz sine at -23 dBFS reads
        // -23 LUFS; here at -20 dBFS (peak 0.1) it must read -20.
        let x = sine(1000.0, 0.1, 10.0, 48_000, 0.0);
        let r = measure(&x, 2, 48_000);
        assert!((r.integrated_lufs - -20.0).abs() < 0.1, "{r:?}");
        assert!((r.short_term_max_lufs - -20.0).abs() < 0.1, "{r:?}");
        assert!(r.loudness_range_lu < 0.1, "{r:?}");
        assert_eq!(r.clipped_samples, 0);
        assert_eq!(r.clipped_runs, 0);
        assert_eq!(r.max_clipped_run, 0);
        assert!((r.sample_peak_dbfs - -20.0).abs() < 0.01);
        // Same at 44.1 kHz (coefficients derived per rate).
        let x = sine(1000.0, 0.1, 10.0, 44_100, 0.0);
        let r = measure(&x, 2, 44_100);
        assert!((r.integrated_lufs - -20.0).abs() < 0.1, "{r:?}");
    }

    #[test]
    fn loudness_range_reflects_step_change() {
        // 10 s at -26 dBFS followed by 10 s at -14 dBFS: LRA ~ 12 LU.
        let mut x = sine(1000.0, 0.05, 10.0, 48_000, 0.0);
        x.extend(sine(1000.0, 0.2, 10.0, 48_000, 0.0));
        let r = measure(&x, 2, 48_000);
        assert!(
            (11.0..12.6).contains(&r.loudness_range_lu),
            "expected ~12 LU, got {:?}",
            r.loudness_range_lu
        );
    }

    /// EBU Tech 3342's test signals 1-4 (stereo 1 kHz sine, 20 s per
    /// level, levels in dBFS) and the loudness range the standard expects
    /// from each, within its +-1 LU. The oracle for LRA: the standard
    /// itself, not another tool.
    #[test]
    fn loudness_range_matches_ebu_tech_3342_test_signals() {
        let cases: [(&[f32], f64); 4] = [
            (&[-20.0, -30.0], 10.0),
            (&[-20.0, -15.0], 5.0),
            (&[-40.0, -20.0], 20.0),
            (&[-50.0, -35.0, -20.0, -35.0, -50.0], 15.0),
        ];
        for (levels, want) in cases {
            let mut x = Vec::new();
            for db in levels {
                x.extend(sine(1000.0, 10f32.powf(db / 20.0), 20.0, 48_000, 0.0));
            }
            let lra = measure(&x, 2, 48_000).loudness_range_lu;
            assert!(
                (lra - want).abs() <= 1.0,
                "levels {levels:?}: LRA {lra:.2}, Tech 3342 expects {want} +-1"
            );
        }
    }

    #[test]
    fn true_peak_and_clipped_runs_distinguish_grazes_from_flat_tops() {
        // fs/4 sine sampled at +-45 degrees: samples read 0.707 of the
        // real peak, the interpolator recovers ~1.0 (+3 dB).
        let x = sine(12_000.0, 1.0, 1.0, 48_000, std::f32::consts::FRAC_PI_4);
        let r = measure(&x, 2, 48_000);
        assert!((-3.1..-2.5).contains(&r.sample_peak_dbfs), "{r:?}");
        assert!(r.true_peak_dbtp > -0.3, "{r:?}");
        assert_eq!(r.clipped_samples, 0);
        // Two separate clipped runs: 1 frame, then 3 consecutive frames.
        let clip: Vec<f32> = vec![
            1.0, -1.0, // frame 0: clipped
            0.5, 0.2, // frame 1: clean
            1.0, 0.0, // frame 2: clipped (L)
            0.0, -1.0, // frame 3: clipped (R)
            1.0, 1.0, // frame 4: clipped (L+R)
            0.1, 0.1, // frame 5: clean
        ];
        let cr = measure(&clip, 2, 48_000);
        assert_eq!(cr.clipped_samples, 6);
        assert_eq!(cr.clipped_runs, 2);
        assert_eq!(cr.max_clipped_run, 3);
    }

    #[test]
    fn series_tracks_the_tone_and_ends_at_the_short_term_max() {
        let x = sine(1000.0, 0.1, 10.0, 48_000, 0.0);
        let s = short_term_series(&x, 2, 48_000);
        assert_eq!(s.len(), 100);
        let (t, last) = *s.last().unwrap();
        assert!((t - 10.0).abs() < 0.11, "{t}");
        assert!((last - -20.0).abs() < 0.1, "{last}");
        assert!((s[0].1 - -20.0).abs() < 0.15, "{:?}", s[0]);
    }

    #[test]
    fn silence_is_gated_out() {
        let r = measure(&vec![0.0; 96_000], 2, 48_000);
        assert_eq!(r.integrated_lufs, f64::NEG_INFINITY);
        assert_eq!(r.loudness_range_lu, 0.0);
        assert_eq!(integrated_from_block_powers(&[]), None);
        assert_eq!(integrated_from_block_powers(&[0.0, 1e-9]), None);
    }

    #[test]
    fn k_weighting_matches_bs1770_at_50hz_and_reset_clears_state() {
        // BS.1770 2nd-order RLB high-pass (f0 = 38.135 Hz, Q = 0.5003)
        // attenuates 50 Hz by -3.94 dB (contrast a 60 Hz Q=0.5 HPF at
        // -7.75 dB). A 50 Hz mono sine at amplitude 0.1 (RMS -23.01 dBFS)
        // reads -23.01 - 0.691 - 3.94 = -27.64 LUFS.
        let n = 48_000 * 4;
        let x: Vec<f32> = (0..n)
            .map(|i| 0.1 * (std::f32::consts::TAU * 50.0 * i as f32 / 48_000.0).sin())
            .collect();
        let r = measure(&x, 1, 48_000);
        assert!(
            (-28.0..-27.2).contains(&r.integrated_lufs),
            "50 Hz mono sine read {} LUFS, expected ~-27.64",
            r.integrated_lufs
        );

        let mut kw = KWeighting::new(48_000.0);
        let mut sum_sq = 0.0f64;
        for &s in &x[48_000..] {
            let y = kw.process_f32(s);
            sum_sq += y * y;
        }
        let streaming = energy_to_lufs(sum_sq / (3 * 48_000) as f64);
        assert!((streaming - r.integrated_lufs).abs() < 0.05);
        kw.reset();
        assert_eq!(kw.process(0.0), 0.0);
    }

    #[test]
    fn handles_extreme_sample_rates_and_nonfinite_inputs() {
        let mut kw = KWeighting::new(0.0);
        for _ in 0..100 {
            let y = kw.process(1.0);
            assert!(y.is_finite() && y.abs() < 10.0, "unstable k-weighting: {y}");
        }

        let mut samples = vec![0.1f32; 48_000 * 2];
        samples[100] = f32::NAN;
        samples[200] = f32::INFINITY;
        let r = measure(&samples, 1, 48_000);
        assert!(r.integrated_lufs.is_finite(), "{:?}", r);
        assert!(r.sample_peak_dbfs.is_finite() && r.true_peak_dbtp.is_finite());
        assert!(short_term_series(&samples, 1, 48_000)
            .iter()
            .all(|(_, l)| l.is_finite()));

        let lra = loudness_range(&[1.0, 2.0, 10.0, f64::NAN, f64::INFINITY]);
        assert!(lra.is_finite() && lra > 0.0, "got lra={lra}");
    }
}
