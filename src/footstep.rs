//! Procedural footstep/material synth + contact-lane step detection.
//!
//! Each step is synth-from-seed — a one-pole-filtered noise transient plus
//! three decaying resonant body modes per surface, parameterized by sample rate.
//!
//! Step detection consumes a `contact_lane` (1.0 in contact,
//! 0.0 airborne, per anim frame) and `[{"frame": N, "surface": "gravel"}]`
//! annotations.
//!
//! Synthesis runs on the control thread into owned buffers; the mixer
//! borrows them like any other voice.

use serde::{Deserialize, Serialize};

use crate::error::{AfError, Result};

/// Surface material tags with a procedural recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Surface {
    Concrete,
    Gravel,
    Grass,
    Wood,
    Metal,
}

pub const ALL_SURFACES: [Surface; 5] = [
    Surface::Concrete,
    Surface::Gravel,
    Surface::Grass,
    Surface::Wood,
    Surface::Metal,
];

/// Default step length (C++ `kFootstepFrames` = 0.18 s).
pub const FOOTSTEP_SECS: f32 = 0.18;
const SEED_SALT: u32 = 0x9E37_79B9;

impl Surface {
    /// Case-sensitive tag lookup (tags are machine-written).
    pub fn from_tag(tag: &str) -> Option<Surface> {
        ALL_SURFACES.into_iter().find(|s| s.name() == tag)
    }

    pub fn name(self) -> &'static str {
        match self {
            Surface::Concrete => "concrete",
            Surface::Gravel => "gravel",
            Surface::Grass => "grass",
            Surface::Wood => "wood",
            Surface::Metal => "metal",
        }
    }

    fn index(self) -> u32 {
        self as u32
    }
}

/// Resonant-body + noise recipe. Frequencies Hz, decays 1/s, levels
/// linear; `noise_lp` is the one-pole coefficient (larger = brighter).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceVoice {
    pub body_freq: [f32; 3],
    pub body_decay: [f32; 3],
    pub body_level: [f32; 3],
    pub noise_decay: f32,
    pub noise_lp: f32,
    pub noise_level: f32,
}

pub fn surface_voice(s: Surface) -> SurfaceVoice {
    let v = |f: [f32; 3], d: [f32; 3], l: [f32; 3], nd: f32, lp: f32, nl: f32| SurfaceVoice {
        body_freq: f,
        body_decay: d,
        body_level: l,
        noise_decay: nd,
        noise_lp: lp,
        noise_level: nl,
    };
    match s {
        // bright short slap
        Surface::Concrete => v(
            [180.0, 340.0, 520.0],
            [55.0, 70.0, 90.0],
            [0.35, 0.22, 0.12],
            60.0,
            0.55,
            0.8,
        ),
        // long bright crunch, scattered body
        Surface::Gravel => v(
            [420.0, 810.0, 1450.0],
            [40.0, 48.0, 55.0],
            [0.16, 0.12, 0.08],
            22.0,
            0.9,
            0.7,
        ),
        // soft dark thud
        Surface::Grass => v(
            [90.0, 150.0, 230.0],
            [35.0, 45.0, 60.0],
            [0.42, 0.18, 0.04],
            70.0,
            0.025,
            0.3,
        ),
        // hollow knock
        Surface::Wood => v(
            [220.0, 440.0, 660.0],
            [30.0, 38.0, 50.0],
            [0.4, 0.28, 0.14],
            80.0,
            0.38,
            0.45,
        ),
        // ringing plate
        Surface::Metal => v(
            [620.0, 930.0, 1240.0],
            [9.0, 12.0, 16.0],
            [0.3, 0.22, 0.15],
            90.0,
            0.9,
            0.6,
        ),
    }
}

fn xrand(s: &mut u32) -> u32 {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    *s
}

fn urand(s: &mut u32) -> f32 {
    (xrand(s) >> 8) as f32 * (2.0 / 16_777_216.0) - 1.0
}

/// Renders one step: deterministic in (surface, seed, gain, rate, frames),
/// clamped to [-1, 1].
pub fn synthesize_footstep(
    surface: Surface,
    seed: u32,
    gain: f32,
    sample_rate: u32,
    frames: usize,
) -> Result<Vec<f32>> {
    if !gain.is_finite() || gain < 0.0 {
        return Err(AfError::BadGain(format!(
            "footstep gain must be finite and >= 0, got {gain}"
        )));
    }
    if sample_rate == 0 {
        return Err(AfError::BadArg("footstep sample_rate must be > 0".into()));
    }
    let v = surface_voice(surface);
    let mut rng = seed ^ SEED_SALT ^ surface.index().wrapping_mul(0x85EB_CA6B);
    if rng == 0 {
        rng = 0x243F_6A88; // xorshift must not sit on zero
    }
    let pi = std::f32::consts::PI;
    let mut phase = [0.0f32; 3];
    for p in phase.iter_mut() {
        *p = urand(&mut rng) * pi;
    }
    let inv_rate = 1.0 / sample_rate as f32;
    let mut lp = 0.0f32;
    let mut out = Vec::with_capacity(frames);
    for i in 0..frames {
        let t = i as f32 * inv_rate;
        let w = urand(&mut rng);
        lp += v.noise_lp * (w - lp);
        let noise = lp * v.noise_level * (-t * v.noise_decay).exp();
        let mut body = 0.0f32;
        for (k, ph) in phase.iter().enumerate() {
            body += v.body_level[k]
                * (2.0 * pi * v.body_freq[k] * t + ph).sin()
                * (-t * v.body_decay[k]).exp();
        }
        out.push(((noise + body) * gain).clamp(-1.0, 1.0));
    }
    Ok(out)
}

/// One detected step: anim frame of the contact rising edge + surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepEvent {
    pub anim_frame: u64,
    pub surface: Surface,
}

/// Rising-edge detector: a step for every frame `i` with
/// `lane[i] > threshold` while `lane[i-1] <= threshold` (the lane starts
/// airborne, so frame 0 counts when already in contact).
pub fn contacts_to_steps(
    lane: &[f32],
    default_surface: Surface,
    threshold: f32,
) -> Result<Vec<StepEvent>> {
    if !threshold.is_finite() {
        return Err(AfError::BadArg(format!(
            "contact threshold must be finite, got {threshold}"
        )));
    }
    let mut out = Vec::new();
    let mut was = false;
    for (i, &c) in lane.iter().enumerate() {
        if !c.is_finite() {
            return Err(AfError::BadArg(format!(
                "contact_lane[{i}] must be finite, got {c}"
            )));
        }
        let now = c > threshold;
        if now && !was {
            out.push(StepEvent {
                anim_frame: i as u64,
                surface: default_surface,
            });
        }
        was = now;
    }
    Ok(out)
}

/// Surface annotation: from `frame` onward the ground is `surface`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurfaceAnnotation {
    pub frame: u64,
    pub surface: Surface,
}

/// Each step takes the annotation with the greatest `frame <= step frame`
/// (ties: the later entry wins, like the C++ duplicate rule), else keeps
/// its surface. Annotations need not be sorted.
pub fn resolve_step_surfaces(steps: &mut [StepEvent], annotations: &[SurfaceAnnotation]) {
    for s in steps.iter_mut() {
        let mut best: Option<&SurfaceAnnotation> = None;
        for a in annotations {
            if a.frame <= s.anim_frame && best.map(|b| a.frame >= b.frame).unwrap_or(true) {
                best = Some(a);
            }
        }
        if let Some(b) = best {
            s.surface = b.surface;
        }
    }
}

/// Anim frame -> audio frame at `sample_rate`, rounded.
pub fn anim_frame_to_audio_frame(anim_frame: u64, anim_fps: f32, sample_rate: u32) -> Result<u64> {
    if !anim_fps.is_finite() || anim_fps <= 0.0 {
        return Err(AfError::BadArg(format!(
            "anim_fps must be finite and > 0, got {anim_fps}"
        )));
    }
    Ok((anim_frame as f64 * sample_rate as f64 / anim_fps as f64).round() as u64)
}

/// Output statistics for surface A/B differentiation.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SignalStats {
    pub rms: f32,
    pub peak: f32,
    /// Fraction of samples changing sign.
    pub zero_cross_rate: f32,
    /// High-frequency energy proxy in [0, 1] (mean |diff| / mean |x| / 4).
    pub brightness: f32,
}

pub fn signal_stats(data: &[f32]) -> SignalStats {
    let mut st = SignalStats::default();
    if data.is_empty() {
        return st;
    }
    let (mut sum_sq, mut sum_abs, mut sum_diff) = (0.0f64, 0.0f64, 0.0f64);
    let mut crossings = 0u64;
    for (i, &v) in data.iter().enumerate() {
        st.peak = st.peak.max(v.abs());
        sum_sq += v as f64 * v as f64;
        sum_abs += v.abs() as f64;
        if i > 0 {
            let prev = data[i - 1];
            if (v >= 0.0) != (prev >= 0.0) {
                crossings += 1;
            }
            sum_diff += (v - prev).abs() as f64;
        }
    }
    let n = data.len() as f64;
    st.rms = (sum_sq / n).sqrt() as f32;
    st.zero_cross_rate = (crossings as f64 / n) as f32;
    st.brightness = ((sum_diff / (sum_abs + 1e-9)) * 0.25).clamp(0.0, 1.0) as f32;
    st
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(s: Surface) -> Vec<f32> {
        synthesize_footstep(s, 42, 1.0, 48_000, 8_640).unwrap()
    }

    #[test]
    fn deterministic_and_bounded() {
        for s in ALL_SURFACES {
            let a = step(s);
            assert_eq!(a, step(s));
            assert_eq!(a.len(), 8_640);
            assert!(a.iter().all(|v| (-1.0..=1.0).contains(v)));
            assert!(signal_stats(&a).peak > 0.05, "{s:?} must sound");
        }
        assert_ne!(
            synthesize_footstep(Surface::Wood, 1, 1.0, 48_000, 512).unwrap(),
            synthesize_footstep(Surface::Wood, 2, 1.0, 48_000, 512).unwrap()
        );
        assert!(synthesize_footstep(Surface::Wood, 1, -1.0, 48_000, 8).is_err());
    }

    /// Port of the C++ blind A/B: surfaces differ in output stats.
    #[test]
    fn surfaces_are_distinguishable() {
        let gravel = signal_stats(&step(Surface::Gravel));
        let grass = signal_stats(&step(Surface::Grass));
        let metal = signal_stats(&step(Surface::Metal));
        let concrete = signal_stats(&step(Surface::Concrete));
        assert!(
            gravel.brightness > grass.brightness * 2.0,
            "{gravel:?} vs {grass:?}"
        );
        assert!(gravel.zero_cross_rate > grass.zero_cross_rate * 2.0);
        // Metal rings: more late energy than concrete's short slap.
        let tail_rms = |s: Surface| signal_stats(&step(s)[4_320..]).rms;
        assert!(tail_rms(Surface::Metal) > tail_rms(Surface::Concrete) * 2.0);
        assert!(metal.rms > 0.0 && concrete.rms > 0.0);
    }

    #[test]
    fn contact_lane_rising_edges_and_annotations() {
        let lane = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let mut steps = contacts_to_steps(&lane, Surface::Concrete, 0.5).unwrap();
        let frames: Vec<u64> = steps.iter().map(|s| s.anim_frame).collect();
        assert_eq!(frames, vec![0, 4, 7]);
        let ann: Vec<SurfaceAnnotation> = serde_json::from_str(
            r#"[{"frame":5,"surface":"gravel"},{"frame":3,"surface":"wood"}]"#,
        )
        .unwrap();
        resolve_step_surfaces(&mut steps, &ann);
        let surf: Vec<Surface> = steps.iter().map(|s| s.surface).collect();
        assert_eq!(
            surf,
            vec![Surface::Concrete, Surface::Wood, Surface::Gravel]
        );
        assert!(contacts_to_steps(&[f32::NAN], Surface::Wood, 0.5).is_err());
        assert!(serde_json::from_str::<Vec<SurfaceAnnotation>>(
            r#"[{"frame":1,"surface":"lava"}]"#
        )
        .is_err());
        assert_eq!(anim_frame_to_audio_frame(30, 30.0, 48_000).unwrap(), 48_000);
        assert!(anim_frame_to_audio_frame(1, 0.0, 48_000).is_err());
        assert_eq!(Surface::from_tag("metal"), Some(Surface::Metal));
        assert_eq!(Surface::from_tag("Metal"), None);
    }
}
