//! Blind audio A/B pack builder and scorer (`gameaudio-4`).
//!
//! Generates 8 shuffled clean-vs-planted fault pairs (2x music +6 dB over talk,
//! 2x clipped hit, 2x 3 s dropout, 2x pumping limiter) plus 1 identical
//! same-file control, verifies every clip against [`crate::playthrough`],
//! keeps the listening folder free of any key or fault-label leaks, and writes
//! the answer key strictly outside the listening folder.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::dynamics::{DuckSpec, Ducker, KeyDetector, Limiter, LimiterSpec};
use crate::error::{AfError, Result};
use crate::playthrough::{
    check_with_telemetry, BusDuckTelemetry, MixTelemetry, PlaythroughLimits, BLOCK_SECS,
};
use crate::rng::SeededRng;
use crate::voice::{synthesize_utterance, VoiceMood, VoiceRegister};
use crate::wav::{read_wav_any, write_wav, WavSpec};

/// Minimum excerpt duration (seconds) so a 3 s dropout and a sustained talk phrase fit cleanly.
pub const MIN_CLIP_SECS: f32 = 6.0;
/// Required number of fault-pair agreements out of 8 for the gate to pass.
pub const REQUIRED_AGREEMENTS: usize = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlindFault {
    MusicOverTalk,
    ClippedHit,
    #[serde(rename = "dropout_3s")]
    Dropout3s,
    PumpingLimiter,
    SameControl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    A,
    B,
    Same,
}

impl Side {
    fn as_str(self) -> &'static str {
        match self {
            Side::A => "a",
            Side::B => "b",
            Side::Same => "same",
        }
    }
}

#[derive(Debug, Clone)]
pub struct BlindPackConfig {
    pub seed: u64,
    pub clip_secs: f32,
    pub limits: PlaythroughLimits,
}

impl Default for BlindPackConfig {
    fn default() -> Self {
        Self {
            seed: 2026_1119,
            clip_secs: 12.0,
            limits: PlaythroughLimits::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindPairKey {
    pub pair_id: String,
    pub source_stem: String,
    pub fault: BlindFault,
    pub clean_side: Side,
    pub gate_choice: Side,
    pub control: bool,
    pub a_gate_pass: bool,
    pub a_breaches: Vec<String>,
    pub b_gate_pass: bool,
    pub b_breaches: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlindKeyFile {
    pub seed: u64,
    pub clip_secs: f32,
    pub pairs: Vec<BlindPairKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindPairScore {
    pub pair_id: String,
    pub fault: BlindFault,
    pub control: bool,
    pub listener: String,
    pub gate_choice: Side,
    pub clean_side: Side,
    pub agreed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindScoreReport {
    pub total_fault_pairs: usize,
    pub agreements: usize,
    pub required_agreements: usize,
    pub control_pair: String,
    pub control_answer: String,
    pub control_ok: bool,
    pub pass: bool,
    pub per_pair: Vec<BlindPairScore>,
}

fn normalize_path(path: &Path) -> PathBuf {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    let mut out = PathBuf::new();
    for c in abs.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn key_is_inside_out_dir(out_dir: &Path, key_path: &Path) -> bool {
    let out_norm = normalize_path(out_dir);
    let key_norm = normalize_path(key_path);
    key_norm == out_norm || key_norm.starts_with(&out_norm)
}

/// Default sibling answer-key path (`<out>.key.json`), guaranteed outside `out_dir`.
pub fn default_key_path(out_dir: &Path) -> PathBuf {
    let trimmed = out_dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "blind_pack".to_string());
    let parent = out_dir.parent().unwrap_or_else(|| Path::new("."));
    let parent = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    parent.join(format!("{trimmed}.key.json"))
}

struct LoadedSource {
    stem: String,
    sample_rate: u32,
    /// Interleaved stereo samples (`frames * 2`).
    stereo: Vec<f32>,
}

fn load_stereo_wav(path: &Path) -> Result<LoadedSource> {
    let bytes =
        std::fs::read(path).map_err(|e| AfError::BadFormat(format!("{}: {e}", path.display())))?;
    let wav =
        read_wav_any(&bytes).map_err(|e| AfError::BadFormat(format!("{}: {e}", path.display())))?;
    let sr = wav.spec.sample_rate;
    let ch = wav.spec.channels;
    if !(8_000..=768_000).contains(&sr) {
        return Err(AfError::BadFormat(format!(
            "{}: sample rate {sr} Hz out of range (8000..=768000)",
            path.display()
        )));
    }
    if wav.samples.iter().any(|s| !s.is_finite()) {
        return Err(AfError::BadFormat(format!(
            "{}: WAV contains non-finite samples",
            path.display()
        )));
    }
    let stereo = match ch {
        1 => {
            let mut out = Vec::with_capacity(wav.samples.len() * 2);
            for &s in &wav.samples {
                out.push(s);
                out.push(s);
            }
            out
        }
        2 => wav.samples,
        other => {
            return Err(AfError::BadFormat(format!(
                "{}: expected mono or stereo WAV, got {other} channels",
                path.display()
            )));
        }
    };
    let frames = stereo.len() / 2;
    let dur = frames as f32 / sr as f32;
    if dur < MIN_CLIP_SECS {
        return Err(AfError::BadArg(format!(
            "{}: duration {dur:.2} s is shorter than minimum {MIN_CLIP_SECS:.1} s",
            path.display()
        )));
    }
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "capture".into());
    Ok(LoadedSource {
        stem,
        sample_rate: sr,
        stereo,
    })
}

fn stereo_rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

fn rms_db(samples: &[f32]) -> f64 {
    let r = stereo_rms(samples);
    if r <= 1e-12 {
        -120.0
    } else {
        20.0 * r.log10()
    }
}

fn db_to_lin(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

fn extract_excerpt(src: &LoadedSource, clip_secs: f32, slot_idx: usize) -> Vec<f32> {
    let sr = src.sample_rate as usize;
    let total_frames = src.stereo.len() / 2;
    let target_frames = ((clip_secs as f64) * sr as f64)
        .round()
        .clamp((MIN_CLIP_SECS as f64) * sr as f64, total_frames as f64)
        as usize;

    let max_start = total_frames.saturating_sub(target_frames);
    let skip_head = (sr / 2).min(max_start);
    let start_frame = if max_start <= skip_head {
        max_start
    } else {
        let span = max_start - skip_head;
        let step = (span / 9).max(1);
        skip_head + ((slot_idx * step) % (span + 1))
    };

    let mut out = src.stereo[start_frame * 2..(start_frame + target_frames) * 2].to_vec();

    // Gentle 30 ms edge fade so every clip starts and ends click-free.
    let fade = ((sr as f64 * 0.03).round() as usize).min(target_frames / 10);
    for i in 0..fade {
        let g = (i as f32) / (fade.max(1) as f32);
        out[i * 2] *= g;
        out[i * 2 + 1] *= g;
        let j = target_frames - 1 - i;
        out[j * 2] *= g;
        out[j * 2 + 1] *= g;
    }
    out
}

struct RenderedClip {
    samples: Vec<f32>,
    telemetry: Option<MixTelemetry>,
}

/// Build a sustained ~2.4 s wordless talk cue using `af_core::voice`.
fn build_talk_cue(sr: u32, variant: usize, seed: u32) -> Result<Vec<f32>> {
    let (reg, mood_a, mood_b) = if variant == 0 {
        (VoiceRegister::Mid, VoiceMood::Plain, VoiceMood::Glad)
    } else {
        (VoiceRegister::High, VoiceMood::PlainB, VoiceMood::Sly)
    };
    let u1 = synthesize_utterance(reg, mood_a, seed.wrapping_add(11), 6, sr)?;
    let u2 = synthesize_utterance(reg, mood_b, seed.wrapping_add(29), 6, sr)?;
    let pause = (sr as f64 * 0.08).round() as usize;
    let mut cue = Vec::with_capacity(u1.len() + pause + u2.len());
    cue.extend_from_slice(&u1);
    cue.resize(cue.len() + pause, 0.0);
    cue.extend_from_slice(&u2);
    Ok(cue)
}

fn build_music_over_talk_pair(
    bed: &[f32],
    sr: u32,
    variant: usize,
    seed: u32,
) -> Result<(RenderedClip, RenderedClip)> {
    let frames = bed.len() / 2;
    let cue_mono = build_talk_cue(sr, variant, seed)?;
    let cue_len = cue_mono.len().min((frames * 4) / 10);
    let cue_start = (frames.saturating_sub(cue_len)) / 2;
    let cue_end = cue_start + cue_len;

    let bed_slice = &bed[cue_start * 2..cue_end * 2];
    let bed_rms = stereo_rms(bed_slice).max(1e-3);
    let bed_peak = bed.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-4) as f64;

    let cue_raw_rms = {
        let sum_sq: f64 = cue_mono[..cue_len]
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum();
        (sum_sq / cue_len.max(1) as f64).sqrt().max(1e-5)
    };
    let cue_raw_peak = cue_mono[..cue_len]
        .iter()
        .fold(0.0f32, |m, &x| m.max(x.abs()))
        .max(1e-4) as f64;

    let duck_spec = DuckSpec {
        amount_db: 9.0,
        threshold_db: -48.0,
        attack_ms: 15.0,
        hold_ms: 120.0,
        release_ms: 220.0,
    };

    // Clean side: talk cue sits +3 dB above un-ducked bed, and real Ducker drops bed 9 dB
    // underneath talk so dialogue is crystal clear (+10 dB clarity) and bed recovers smoothly.
    let clean_cue_scale =
        ((bed_rms * db_to_lin(3.0)) / cue_raw_rms).min(0.45 / cue_raw_peak) as f32;
    let win_frames = ((sr as f64) * 0.1).round().max(1.0) as usize;
    let n_windows = frames.div_ceil(win_frames);

    let mut clean_samples = Vec::with_capacity(bed.len());
    let mut clean_cue_rms_db = Vec::with_capacity(n_windows);
    let mut clean_bed_rms_db = Vec::with_capacity(n_windows);
    let mut clean_drop_db = Vec::with_capacity(n_windows);

    let mut detector = KeyDetector::new(sr);
    let mut ducker = Ducker::new(duck_spec, sr);

    for w in 0..n_windows {
        let f0 = w * win_frames;
        let f1 = (f0 + win_frames).min(frames);
        let mut cue_sq = 0.0f64;
        let mut bed_sq = 0.0f64;
        let mut max_drop = 0.0f64;
        for f in f0..f1 {
            let c = if f >= cue_start && f < cue_end {
                cue_mono[f - cue_start] * clean_cue_scale
            } else {
                0.0
            };
            let key_db = detector.push(c, c);
            let gain = ducker.step(key_db);
            max_drop = max_drop.max(f64::from(ducker.drop_db()));
            let bl = bed[f * 2] * gain;
            let br = bed[f * 2 + 1] * gain;
            clean_samples.push(bl + c);
            clean_samples.push(br + c);
            cue_sq += f64::from(c) * f64::from(c);
            bed_sq += 0.5 * (f64::from(bl) * f64::from(bl) + f64::from(br) * f64::from(br));
        }
        let n = (f1 - f0).max(1) as f64;
        clean_cue_rms_db.push(10.0 * (cue_sq / n + 1e-12).log10());
        clean_bed_rms_db.push(10.0 * (bed_sq / n + 1e-12).log10());
        clean_drop_db.push(max_drop);
    }

    let clean_tel = MixTelemetry {
        sidechain_configured: true,
        window_secs: 0.1,
        cue_rms_db: clean_cue_rms_db,
        ducks: vec![BusDuckTelemetry {
            bus: "music".into(),
            amount_db: duck_spec.amount_db,
            hold_ms: duck_spec.hold_ms,
            release_ms: duck_spec.release_ms,
            drop_db: clean_drop_db,
            bed_rms_db: clean_bed_rms_db,
        }],
        ..MixTelemetry::default()
    };

    // Planted side: music bed sits +6.5 dB OVER the talk cue with zero ducking.
    // Boost bed modestly (bounded below -3 dBFS so no clipping/limiter interference) and place
    // talk 6.5 dB below the boosted bed (still above -55 dBFS active cue floor).
    let bed_boost = db_to_lin(4.0).min(0.65 / bed_peak);
    let boosted_bed_rms = bed_rms * bed_boost;
    let planted_cue_target_rms = (boosted_bed_rms * db_to_lin(-6.5)).max(db_to_lin(-46.0));
    // Re-ensure bed stays at least +6.2 dB above planted cue even on quiet beds.
    let final_bed_boost = bed_boost.max((planted_cue_target_rms * db_to_lin(6.5)) / bed_rms);
    let final_bed_boost = final_bed_boost.min(0.75 / bed_peak);
    let final_cue_rms = (bed_rms * final_bed_boost) * db_to_lin(-6.5);
    let planted_cue_scale = (final_cue_rms / cue_raw_rms) as f32;

    let mut planted_samples = Vec::with_capacity(bed.len());
    let mut planted_cue_rms_db = Vec::with_capacity(n_windows);
    let mut planted_bed_rms_db = Vec::with_capacity(n_windows);
    let mut planted_drop_db = Vec::with_capacity(n_windows);

    for w in 0..n_windows {
        let f0 = w * win_frames;
        let f1 = (f0 + win_frames).min(frames);
        let mut cue_sq = 0.0f64;
        let mut bed_sq = 0.0f64;
        for f in f0..f1 {
            let c = if f >= cue_start && f < cue_end {
                cue_mono[f - cue_start] * planted_cue_scale
            } else {
                0.0
            };
            let bl = bed[f * 2] * final_bed_boost as f32;
            let br = bed[f * 2 + 1] * final_bed_boost as f32;
            planted_samples.push(bl + c);
            planted_samples.push(br + c);
            cue_sq += f64::from(c) * f64::from(c);
            bed_sq += 0.5 * (f64::from(bl) * f64::from(bl) + f64::from(br) * f64::from(br));
        }
        let n = (f1 - f0).max(1) as f64;
        planted_cue_rms_db.push(10.0 * (cue_sq / n + 1e-12).log10());
        planted_bed_rms_db.push(10.0 * (bed_sq / n + 1e-12).log10());
        planted_drop_db.push(0.0);
    }

    let planted_tel = MixTelemetry {
        sidechain_configured: true,
        window_secs: 0.1,
        cue_rms_db: planted_cue_rms_db,
        ducks: vec![BusDuckTelemetry {
            bus: "music".into(),
            amount_db: duck_spec.amount_db,
            hold_ms: duck_spec.hold_ms,
            release_ms: duck_spec.release_ms,
            drop_db: planted_drop_db,
            bed_rms_db: planted_bed_rms_db,
        }],
        ..MixTelemetry::default()
    };

    Ok((
        RenderedClip {
            samples: clean_samples,
            telemetry: Some(clean_tel),
        },
        RenderedClip {
            samples: planted_samples,
            telemetry: Some(planted_tel),
        },
    ))
}

fn build_clipped_hit_pair(bed: &[f32], sr: u32, seed: u32) -> (RenderedClip, RenderedClip) {
    let frames = bed.len() / 2;
    let mut planted = bed.to_vec();
    let mut rng = SeededRng::new(seed.wrapping_add(303));
    let hit_len = ((sr as f64) * 0.14).round() as usize;

    for &frac in &[0.24f64, 0.53, 0.78] {
        let start = ((frames as f64) * frac).round() as usize;
        let end = (start + hit_len).min(frames);
        for (k, f) in (start..end).enumerate() {
            let t = k as f64 / sr as f64;
            let env = (-t * 22.0).exp();
            let body = (2.0 * std::f64::consts::PI * (145.0 * (-t * 4.0).exp() + 55.0) * t).sin();
            let crack = (2.0 * std::f64::consts::PI * 1_650.0 * t).sin() * (-t * 45.0).exp();
            let noise = (rng.next_f32() * 2.0 - 1.0) as f64 * 0.45 * (-t * 35.0).exp();
            // Overdrive hard (+16 dB) and hard-clip at +-1.0 full scale.
            let hit = (env * (body * 3.8 + crack * 2.2) + noise * 3.0) as f32;
            planted[f * 2] = (planted[f * 2] + hit).clamp(-1.0, 1.0);
            planted[f * 2 + 1] = (planted[f * 2 + 1] + hit).clamp(-1.0, 1.0);
        }
    }

    (
        RenderedClip {
            samples: bed.to_vec(),
            telemetry: None,
        },
        RenderedClip {
            samples: planted,
            telemetry: None,
        },
    )
}

fn build_dropout_3s_pair(bed: &[f32], sr: u32) -> (RenderedClip, RenderedClip) {
    let frames = bed.len() / 2;
    let mut planted = bed.to_vec();
    let drop_frames = ((sr as f64) * 3.10).round() as usize;
    let ramp = ((sr as f64) * 0.005).round().max(1.0) as usize;
    let start = frames.saturating_sub(drop_frames) / 2;
    let end = (start + drop_frames).min(frames);

    for f in start..end {
        planted[f * 2] = 0.0;
        planted[f * 2 + 1] = 0.0;
    }
    for i in 0..ramp {
        let g = 1.0 - (i as f32 / ramp as f32);
        if start > i {
            let f = start - 1 - i;
            planted[f * 2] *= 1.0 - g;
            planted[f * 2 + 1] *= 1.0 - g;
        }
        if end + i < frames {
            let f = end + i;
            let g_in = i as f32 / ramp as f32;
            planted[f * 2] *= g_in;
            planted[f * 2 + 1] *= g_in;
        }
    }

    (
        RenderedClip {
            samples: bed.to_vec(),
            telemetry: None,
        },
        RenderedClip {
            samples: planted,
            telemetry: None,
        },
    )
}

fn build_pumping_limiter_pair(bed: &[f32], sr: u32) -> (RenderedClip, RenderedClip) {
    let frames = bed.len() / 2;
    let bed_rms = stereo_rms(bed).max(1e-3);

    // Build an over-driven feed with periodic heavy low-end thumps every 1.1 s
    // so the slow-release limiter visibly squashes and pumps the music bed.
    let pulse_period = ((sr as f64) * 1.1).round().max(1.0) as usize;
    let pulse_len = ((sr as f64) * 0.24).round() as usize;
    let drive = (2.8 / bed_rms).clamp(8.0, 60.0) as f32;
    let pulse_amp = 4.2f32;

    let spec = LimiterSpec {
        ceiling_dbtp: -1.0,
        lookahead_ms: 5.0,
        release_ms: 300.0,
    };
    let mut limiter = Limiter::new(spec, sr);
    let latency = limiter.latency();
    let block_frames = ((sr as f64) * BLOCK_SECS).round().max(1.0) as usize;

    let mut limited = vec![0.0f32; frames * 2];
    let mut limiter_series = Vec::with_capacity(frames.div_ceil(block_frames));
    let mut in_block = 0usize;

    for f in 0..(frames + latency) {
        let (il, ir) = if f < frames {
            let pos_in_bar = f % pulse_period;
            let pulse = if pos_in_bar < pulse_len {
                let t = pos_in_bar as f64 / sr as f64;
                let atk = (t / 0.012).min(1.0);
                let dec = (-t * 11.0).exp();
                (atk * dec * (2.0 * std::f64::consts::PI * 58.0 * t).sin()) as f32 * pulse_amp
            } else {
                0.0
            };
            (bed[f * 2] * drive + pulse, bed[f * 2 + 1] * drive + pulse)
        } else {
            (0.0, 0.0)
        };

        let (ol, or) = limiter.process_frame(il, ir);
        if f >= latency {
            let out_f = f - latency;
            limited[out_f * 2] = ol;
            limited[out_f * 2 + 1] = or;
            in_block += 1;
            if in_block >= block_frames || out_f + 1 == frames {
                let min_g = limiter.take_min_gain().clamp(1e-6, 1.0);
                let gr_db = (-20.0 * f64::from(min_g).log10()).max(0.0);
                limiter_series.push(((out_f + 1) as f64 / sr as f64, gr_db));
                in_block = 0;
            }
        } else if f + 1 == latency {
            let _ = limiter.take_min_gain();
        }
    }

    // Bring post-limiter loudness back near the clean bed's RMS so the listener hears
    // pumping/breathing rather than just a louder clip, while telemetry keeps the true GR.
    let lim_rms = stereo_rms(&limited).max(1e-4);
    let lim_peak = limited
        .iter()
        .fold(0.0f32, |m, &x| m.max(x.abs()))
        .max(1e-4) as f64;
    let match_gain = (bed_rms / lim_rms).min(0.85 / lim_peak) as f32;
    for s in &mut limited {
        *s *= match_gain;
    }

    let tel = MixTelemetry {
        limiter_configured: true,
        limiter_ceiling_dbtp: Some(spec.ceiling_dbtp),
        limiter_series,
        window_secs: 0.1,
        ..MixTelemetry::default()
    };

    (
        RenderedClip {
            samples: bed.to_vec(),
            telemetry: None,
        },
        RenderedClip {
            samples: limited,
            telemetry: Some(tel),
        },
    )
}

fn shuffle<T>(items: &mut [T], rng: &mut SeededRng) {
    let n = items.len();
    for i in (1..n).rev() {
        let j = (rng.next_u32() as usize) % (i + 1);
        items.swap(i, j);
    }
}

fn write_stereo_wav(path: &Path, sr: u32, samples: &[f32]) -> Result<()> {
    let spec = WavSpec {
        channels: 2,
        sample_rate: sr,
    };
    let mut bytes = Vec::new();
    write_wav(&mut bytes, &spec, samples)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| AfError::BadFormat(format!("{}: {e}", parent.display())))?;
    }
    std::fs::write(path, bytes).map_err(|e| AfError::BadFormat(format!("{}: {e}", path.display())))
}

/// Build a 9-pair (8 fault + 1 same-file control) blind listening pack and its
/// external answer key.
pub fn build_blind_pack(
    sources: &[PathBuf],
    out_dir: &Path,
    key_path: &Path,
    cfg: &BlindPackConfig,
) -> Result<BlindKeyFile> {
    if sources.is_empty() {
        return Err(AfError::BadArg(
            "blind-pack requires at least one input WAV file".into(),
        ));
    }
    if !(cfg.clip_secs.is_finite() && cfg.clip_secs >= MIN_CLIP_SECS) {
        return Err(AfError::BadArg(format!(
            "clip_secs must be >= {MIN_CLIP_SECS}, got {}",
            cfg.clip_secs
        )));
    }
    if key_is_inside_out_dir(out_dir, key_path) {
        return Err(AfError::BadArg(format!(
            "key path {} must sit outside blind listening directory {}",
            key_path.display(),
            out_dir.display()
        )));
    }

    let mut loaded = Vec::with_capacity(sources.len());
    for p in sources {
        loaded.push(load_stereo_wav(p)?);
    }

    // 4 fault families x 2 + 1 same-file control = 9 pairs.
    let mut plan: Vec<(BlindFault, usize)> = vec![
        (BlindFault::MusicOverTalk, 0),
        (BlindFault::MusicOverTalk, 1),
        (BlindFault::ClippedHit, 0),
        (BlindFault::ClippedHit, 1),
        (BlindFault::Dropout3s, 0),
        (BlindFault::Dropout3s, 1),
        (BlindFault::PumpingLimiter, 0),
        (BlindFault::PumpingLimiter, 1),
        (BlindFault::SameControl, 0),
    ];

    let seed32 = (cfg.seed as u32) ^ ((cfg.seed >> 32) as u32);
    let mut rng = SeededRng::new(seed32);
    shuffle(&mut plan, &mut rng);

    // Balanced A/B clean-side assignment across the 8 fault pairs, shuffled by seed.
    let mut fault_sides = [
        Side::A,
        Side::A,
        Side::A,
        Side::A,
        Side::B,
        Side::B,
        Side::B,
        Side::B,
    ];
    shuffle(&mut fault_sides, &mut rng);

    std::fs::create_dir_all(out_dir)
        .map_err(|e| AfError::BadFormat(format!("{}: {e}", out_dir.display())))?;

    let mut pairs = Vec::with_capacity(plan.len());
    let mut fault_side_idx = 0usize;

    for (idx, &(fault, variant)) in plan.iter().enumerate() {
        let pair_id = format!("pair_{:02}", idx + 1);
        let src = &loaded[idx % loaded.len()];
        let sr = src.sample_rate;
        let excerpt = extract_excerpt(src, cfg.clip_secs, idx);
        let pair_seed = rng.next_u32();

        let (clean_clip, other_clip, clean_side, control) = match fault {
            BlindFault::MusicOverTalk => {
                let (c, p) = build_music_over_talk_pair(&excerpt, sr, variant, pair_seed)?;
                let side = fault_sides[fault_side_idx];
                fault_side_idx += 1;
                (c, p, side, false)
            }
            BlindFault::ClippedHit => {
                let (c, p) = build_clipped_hit_pair(&excerpt, sr, pair_seed);
                let side = fault_sides[fault_side_idx];
                fault_side_idx += 1;
                (c, p, side, false)
            }
            BlindFault::Dropout3s => {
                let (c, p) = build_dropout_3s_pair(&excerpt, sr);
                let side = fault_sides[fault_side_idx];
                fault_side_idx += 1;
                (c, p, side, false)
            }
            BlindFault::PumpingLimiter => {
                let (c, p) = build_pumping_limiter_pair(&excerpt, sr);
                let side = fault_sides[fault_side_idx];
                fault_side_idx += 1;
                (c, p, side, false)
            }
            BlindFault::SameControl => (
                RenderedClip {
                    samples: excerpt.clone(),
                    telemetry: None,
                },
                RenderedClip {
                    samples: excerpt,
                    telemetry: None,
                },
                Side::Same,
                true,
            ),
        };

        let (clip_a, clip_b) = match clean_side {
            Side::A | Side::Same => (clean_clip, other_clip),
            Side::B => (other_clip, clean_clip),
        };

        let rep_a = check_with_telemetry(
            &clip_a.samples,
            2,
            sr,
            &cfg.limits,
            clip_a.telemetry.as_ref(),
        );
        let rep_b = check_with_telemetry(
            &clip_b.samples,
            2,
            sr,
            &cfg.limits,
            clip_b.telemetry.as_ref(),
        );
        let a_pass = rep_a.passed();
        let b_pass = rep_b.passed();

        let gate_choice = match (a_pass, b_pass) {
            (true, false) => Side::A,
            (false, true) => Side::B,
            (true, true) => Side::Same,
            (false, false) => {
                return Err(AfError::BadFormat(format!(
                    "{pair_id} ({fault:?}, source {}): both A and B failed the gate (A: {:?}; B: {:?})",
                    src.stem, rep_a.breaches, rep_b.breaches
                )));
            }
        };

        if gate_choice != clean_side {
            return Err(AfError::BadFormat(format!(
                "{pair_id} ({fault:?}, source {}): gate chose {:?} but expected clean side {:?} (A pass={a_pass} {:?}, B pass={b_pass} {:?}, rms={:.1} dB)",
                src.stem,
                gate_choice,
                clean_side,
                rep_a.breaches,
                rep_b.breaches,
                rms_db(&clip_a.samples)
            )));
        }

        let pair_dir = out_dir.join(&pair_id);
        write_stereo_wav(&pair_dir.join("a.wav"), sr, &clip_a.samples)?;
        write_stereo_wav(&pair_dir.join("b.wav"), sr, &clip_b.samples)?;

        pairs.push(BlindPairKey {
            pair_id,
            source_stem: src.stem.clone(),
            fault,
            clean_side,
            gate_choice,
            control,
            a_gate_pass: a_pass,
            a_breaches: rep_a.breaches,
            b_gate_pass: b_pass,
            b_breaches: rep_b.breaches,
        });
    }

    // Write clean blind sheet + blank answer template (zero leaks inside out_dir).
    let mut sheet = String::from(
        "# Blind Audio Listening Sheet\n\n\
         Listen to `a.wav` and `b.wav` in each pair below.\n\
         For each pair, pick which clip sounds cleaner (`a` or `b`), or `same` if you cannot hear a difference.\n\
         Fill in `answers_template.json` (or save a copy as `answers.json`) and score with `af blind-score`.\n\n\
         | Pair | Files | Your pick (`a` / `b` / `same`) | Notes |\n\
         |---|---|---|---|\n",
    );
    let mut template = BTreeMap::new();
    for p in &pairs {
        sheet.push_str(&format!(
            "| {} | `{}/a.wav`, `{}/b.wav` | | |\n",
            p.pair_id, p.pair_id, p.pair_id
        ));
        template.insert(p.pair_id.clone(), String::new());
    }
    std::fs::write(out_dir.join("sheet.md"), sheet)
        .map_err(|e| AfError::BadFormat(format!("{}: {e}", out_dir.join("sheet.md").display())))?;
    let tmpl_json = format_template_json(&template);
    std::fs::write(out_dir.join("answers_template.json"), tmpl_json).map_err(|e| {
        AfError::BadFormat(format!(
            "{}: {e}",
            out_dir.join("answers_template.json").display()
        ))
    })?;

    let key = BlindKeyFile {
        seed: cfg.seed,
        clip_secs: cfg.clip_secs,
        pairs,
    };
    if let Some(parent) = key_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| AfError::BadFormat(format!("{}: {e}", parent.display())))?;
        }
    }
    let key_json = format_key_json(&key);
    std::fs::write(key_path, key_json)
        .map_err(|e| AfError::BadFormat(format!("{}: {e}", key_path.display())))?;

    Ok(key)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn fault_as_str(f: BlindFault) -> &'static str {
    match f {
        BlindFault::MusicOverTalk => "music_over_talk",
        BlindFault::ClippedHit => "clipped_hit",
        BlindFault::Dropout3s => "dropout_3s",
        BlindFault::PumpingLimiter => "pumping_limiter",
        BlindFault::SameControl => "same_control",
    }
}

fn format_template_json(map: &BTreeMap<String, String>) -> String {
    let mut out = String::from("{\n");
    for (i, (k, v)) in map.iter().enumerate() {
        let comma = if i + 1 < map.len() { "," } else { "" };
        out.push_str(&format!(
            "  {}: {}{comma}\n",
            json_escape(k),
            json_escape(v)
        ));
    }
    out.push_str("}\n");
    out
}

fn format_str_array(items: &[String]) -> String {
    if items.is_empty() {
        return "[]".into();
    }
    let mut out = String::from("[");
    for (i, s) in items.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&json_escape(s));
    }
    out.push(']');
    out
}

fn format_key_json(key: &BlindKeyFile) -> String {
    let mut out = format!(
        "{{\n  \"seed\": {},\n  \"clip_secs\": {},\n  \"pairs\": [\n",
        key.seed, key.clip_secs
    );
    for (i, p) in key.pairs.iter().enumerate() {
        let comma = if i + 1 < key.pairs.len() { "," } else { "" };
        out.push_str(&format!(
            "    {{\n\
             \x20     \"pair_id\": {},\n\
             \x20     \"source_stem\": {},\n\
             \x20     \"fault\": {},\n\
             \x20     \"clean_side\": {},\n\
             \x20     \"gate_choice\": {},\n\
             \x20     \"control\": {},\n\
             \x20     \"a_gate_pass\": {},\n\
             \x20     \"a_breaches\": {},\n\
             \x20     \"b_gate_pass\": {},\n\
             \x20     \"b_breaches\": {}\n\
             \x20   }}{comma}\n",
            json_escape(&p.pair_id),
            json_escape(&p.source_stem),
            json_escape(fault_as_str(p.fault)),
            json_escape(p.clean_side.as_str()),
            json_escape(p.gate_choice.as_str()),
            p.control,
            p.a_gate_pass,
            format_str_array(&p.a_breaches),
            p.b_gate_pass,
            format_str_array(&p.b_breaches),
        ));
    }
    out.push_str("  ]\n}\n");
    out
}

fn parse_listener_choice(raw: &str, pair_id: &str) -> Result<Side> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "a" => Ok(Side::A),
        "b" => Ok(Side::B),
        "same" | "cant_tell" | "can't tell" | "cant tell" => Ok(Side::Same),
        other => Err(AfError::BadArg(format!(
            "{pair_id}: expected 'a', 'b', or 'same'/'cant_tell', got {other:?}"
        ))),
    }
}

/// Score listener answers against a [`BlindKeyFile`].
///
/// Passes iff the same-file control reads `same`/`cant_tell` AND the listener
/// agrees with the gate on at least 7 of the 8 planted-fault pairs.
pub fn score_blind_pack(
    key: &BlindKeyFile,
    answers: &BTreeMap<String, String>,
) -> Result<BlindScoreReport> {
    let mut total_fault_pairs = 0usize;
    let mut agreements = 0usize;
    let mut control_pair = String::new();
    let mut control_answer = String::new();
    let mut control_ok = false;
    let mut per_pair = Vec::with_capacity(key.pairs.len());
    let mut seen_ids = std::collections::BTreeSet::new();

    for p in &key.pairs {
        if !seen_ids.insert(p.pair_id.as_str()) {
            return Err(AfError::BadFormat(format!(
                "duplicate pair_id {} in key file",
                p.pair_id
            )));
        }
        let raw = answers
            .get(&p.pair_id)
            .ok_or_else(|| AfError::BadArg(format!("answers missing entry for {}", p.pair_id)))?;
        let choice = parse_listener_choice(raw, &p.pair_id)?;
        let agreed = choice == p.gate_choice;

        if p.control {
            control_pair = p.pair_id.clone();
            control_answer = choice.as_str().to_string();
            control_ok = choice == Side::Same;
        } else {
            total_fault_pairs += 1;
            if agreed {
                agreements += 1;
            }
        }

        per_pair.push(BlindPairScore {
            pair_id: p.pair_id.clone(),
            fault: p.fault,
            control: p.control,
            listener: choice.as_str().to_string(),
            gate_choice: p.gate_choice,
            clean_side: p.clean_side,
            agreed,
        });
    }

    if control_pair.is_empty() {
        return Err(AfError::BadFormat(
            "key file does not contain a same-file control pair".into(),
        ));
    }

    let pass = control_ok && agreements >= REQUIRED_AGREEMENTS;
    Ok(BlindScoreReport {
        total_fault_pairs,
        agreements,
        required_agreements: REQUIRED_AGREEMENTS,
        control_pair,
        control_answer,
        control_ok,
        pass,
        per_pair,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_float32_wav(path: &Path, sr: u32, samples: &[f32]) {
        let data_bytes = (samples.len() * 4) as u32;
        let mut out = Vec::with_capacity(44 + data_bytes as usize);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&sr.to_le_bytes());
        out.extend_from_slice(&(sr.saturating_mul(4)).to_le_bytes());
        out.extend_from_slice(&4u16.to_le_bytes());
        out.extend_from_slice(&32u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_bytes.to_le_bytes());
        for &s in samples {
            out.extend_from_slice(&s.to_le_bytes());
        }
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn rejects_excessive_rate_nonfinite_wav_and_duplicate_pair_ids() {
        let dir = std::env::temp_dir().join(format!("af_blind_red_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let hi_sr = dir.join("hi_sr.wav");
        write_float32_wav(&hi_sr, 2_000_000, &[0.1, 0.2]);
        assert!(load_stereo_wav(&hi_sr).is_err());

        let nan_wav = dir.join("nan.wav");
        write_float32_wav(&nan_wav, 48_000, &[0.1, f32::NAN]);
        assert!(load_stereo_wav(&nan_wav).is_err());
        let _ = std::fs::remove_dir_all(&dir);

        let key = BlindKeyFile {
            seed: 1,
            clip_secs: 6.0,
            pairs: vec![
                BlindPairKey {
                    pair_id: "p1".into(),
                    source_stem: "s".into(),
                    fault: BlindFault::SameControl,
                    clean_side: Side::Same,
                    gate_choice: Side::Same,
                    control: true,
                    a_gate_pass: true,
                    a_breaches: vec![],
                    b_gate_pass: true,
                    b_breaches: vec![],
                },
                BlindPairKey {
                    pair_id: "p1".into(),
                    source_stem: "s".into(),
                    fault: BlindFault::ClippedHit,
                    clean_side: Side::A,
                    gate_choice: Side::B,
                    control: false,
                    a_gate_pass: true,
                    a_breaches: vec![],
                    b_gate_pass: false,
                    b_breaches: vec!["clip".into()],
                },
            ],
        };
        let mut answers = BTreeMap::new();
        answers.insert("p1".to_string(), "same".to_string());
        assert!(score_blind_pack(&key, &answers).is_err());
    }
}
