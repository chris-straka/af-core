//! Procedural wordless pulse-and-formant voice synthesizer.
//!
//! Generates deterministic wordless dialogue utterances varied by speaker
//! register, mood, seed, and syllable count. Runs on the control thread
//! into caller-owned mono PCM buffers, identical offline and in real time.

use serde::{Deserialize, Serialize};

use crate::error::{AfError, Result};
use crate::rng::SeededRng;

/// Speaker vocal register (fundamental pitch + vocal-tract scaling).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoiceRegister {
    Low,
    Mid,
    High,
}

pub const ALL_REGISTERS: [VoiceRegister; 3] =
    [VoiceRegister::Low, VoiceRegister::Mid, VoiceRegister::High];

impl VoiceRegister {
    pub fn base_pitch_hz(self) -> f32 {
        match self {
            VoiceRegister::Low => 104.0,
            VoiceRegister::Mid => 152.0,
            VoiceRegister::High => 228.0,
        }
    }

    pub fn mouth_scale(self) -> f32 {
        match self {
            VoiceRegister::Low => 0.94,
            VoiceRegister::Mid => 1.0,
            VoiceRegister::High => 1.12,
        }
    }
}

/// Emotional inflection preset for wordless utterances.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoiceMood {
    Plain,
    PlainB,
    Glad,
    Afraid,
    Angry,
    Sad,
    Sly,
    Tired,
}

pub const ALL_MOODS: [VoiceMood; 8] = [
    VoiceMood::Plain,
    VoiceMood::PlainB,
    VoiceMood::Glad,
    VoiceMood::Afraid,
    VoiceMood::Angry,
    VoiceMood::Sad,
    VoiceMood::Sly,
    VoiceMood::Tired,
];

/// Vowel / closed-mouth formant target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MouthShape {
    M,
    A,
    O,
    E,
    U,
    Uh,
}

impl MouthShape {
    /// Returns `(freqs_hz, widths_hz, gains)` for the three vocal formants.
    pub fn formants(self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        match self {
            MouthShape::M => (
                [250.0, 1000.0, 2200.0],
                [70.0, 140.0, 220.0],
                [1.0, 0.12, 0.03],
            ),
            MouthShape::A => (
                [730.0, 1090.0, 2440.0],
                [90.0, 110.0, 170.0],
                [1.0, 0.55, 0.20],
            ),
            MouthShape::O => (
                [500.0, 850.0, 2500.0],
                [80.0, 100.0, 170.0],
                [1.0, 0.45, 0.10],
            ),
            MouthShape::E => (
                [530.0, 1840.0, 2480.0],
                [80.0, 120.0, 170.0],
                [1.0, 0.40, 0.20],
            ),
            MouthShape::U => (
                [320.0, 800.0, 2240.0],
                [70.0, 100.0, 170.0],
                [1.0, 0.30, 0.06],
            ),
            MouthShape::Uh => (
                [600.0, 1200.0, 2500.0],
                [90.0, 120.0, 170.0],
                [1.0, 0.40, 0.12],
            ),
        }
    }
}

/// Full low-level specification for a single pulse-and-formant voice gesture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoiceSpec {
    pub register: VoiceRegister,
    pub duration_secs: f32,
    /// `(time_secs, semitones_relative_to_register)`.
    pub pitch_keys: Vec<(f32, f32)>,
    /// `(time_secs, linear_amplitude)`.
    pub amp_keys: Vec<(f32, f32)>,
    /// `(time_secs, mouth_shape)`.
    pub shape_keys: Vec<(f32, MouthShape)>,
    /// Breath noise mix in `[0.0, 1.0]`.
    pub breath: f32,
    /// Growl roughness flutter in `[0.0, 1.0]`.
    pub rough: f32,
    /// Harmonic roll-off exponent (`> 0.0`, default `1.6`).
    pub tilt: f32,
    /// Optional `(time_secs, center_hz)` for the air band-pass; when empty,
    /// uses `1800.0 * register.mouth_scale()`.
    pub breath_band_keys: Vec<(f32, f32)>,
    /// Target peak amplitude in `(0.0, 1.0]`.
    pub peak: f32,
    pub seed: u32,
}

/// RBJ constant-0-dB-peak-gain band-pass biquad (Direct Form I for smooth retuning).
struct BandPass {
    b0: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl BandPass {
    fn new() -> Self {
        Self {
            b0: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn retune(&mut self, freq_hz: f32, width_hz: f32, sample_rate: f64) {
        let freq = (freq_hz as f64).min(sample_rate * 0.45).max(1.0);
        let width = (width_hz as f64).max(1.0);
        let w = std::f64::consts::TAU * freq / sample_rate;
        let q = (freq / width).max(0.5);
        let alpha = w.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        self.b0 = alpha / a0;
        self.a1 = -2.0 * w.cos() / a0;
        self.a2 = (1.0 - alpha) / a0;
    }

    #[inline]
    fn step(&mut self, x: f32) -> f32 {
        let x = x as f64;
        let y = self.b0 * (x - self.x2) - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y as f32
    }
}

fn interp_f32(keys: &[(f32, f32)], t: f32) -> f32 {
    if keys.len() == 1 || t <= keys[0].0 {
        return keys[0].1;
    }
    let last = keys.len() - 1;
    if t >= keys[last].0 {
        return keys[last].1;
    }
    for w in keys.windows(2) {
        let (t0, v0) = w[0];
        let (t1, v1) = w[1];
        if t <= t1 {
            let dt = t1 - t0;
            if dt <= 1e-9 {
                return v1;
            }
            let a = ((t - t0) / dt).clamp(0.0, 1.0);
            return v0 + (v1 - v0) * a;
        }
    }
    keys[last].1
}

fn interp_formants(
    keys: &[(f32, MouthShape)],
    t: f32,
    mouth_scale: f32,
) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let scaled = |s: MouthShape| {
        let (mut f, w, g) = s.formants();
        for freq in &mut f {
            *freq *= mouth_scale;
        }
        (f, w, g)
    };
    if keys.len() == 1 || t <= keys[0].0 {
        return scaled(keys[0].1);
    }
    let last = keys.len() - 1;
    if t >= keys[last].0 {
        return scaled(keys[last].1);
    }
    for w in keys.windows(2) {
        let (t0, s0) = w[0];
        let (t1, s1) = w[1];
        if t <= t1 {
            let dt = t1 - t0;
            if dt <= 1e-9 {
                return scaled(s1);
            }
            let a = ((t - t0) / dt).clamp(0.0, 1.0);
            let (f0, w0, g0) = scaled(s0);
            let (f1, w1, g1) = scaled(s1);
            let mut f = [0.0; 3];
            let mut bw = [0.0; 3];
            let mut g = [0.0; 3];
            for j in 0..3 {
                f[j] = f0[j] + (f1[j] - f0[j]) * a;
                bw[j] = w0[j] + (w1[j] - w0[j]) * a;
                g[j] = g0[j] + (g1[j] - g0[j]) * a;
            }
            return (f, bw, g);
        }
    }
    scaled(keys[last].1)
}

fn normalize_fade_clamp(samples: &mut [f32], peak: f32, sample_rate: u32) {
    for s in samples.iter_mut() {
        if !s.is_finite() {
            *s = 0.0;
        }
    }
    let max_abs = samples.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if max_abs > 0.0 {
        let scale = peak / max_abs;
        for s in samples.iter_mut() {
            *s *= scale;
        }
    }
    let n = samples.len();
    let fade = ((0.004 * sample_rate as f64).round() as usize).min(n / 2);
    if fade > 0 {
        let inv = 1.0 / fade as f32;
        for i in 0..fade {
            let g = i as f32 * inv;
            samples[i] *= g;
            samples[n - 1 - i] *= g;
        }
    }
    for s in samples.iter_mut() {
        *s = s.clamp(-1.0, 1.0);
    }
}

fn validate_spec(spec: &VoiceSpec, sample_rate: u32) -> Result<()> {
    if !(8000..=768_000).contains(&sample_rate) {
        return Err(AfError::BadArg(format!(
            "voice sample_rate must be in 8000..=768000, got {sample_rate}"
        )));
    }
    if !spec.duration_secs.is_finite() || spec.duration_secs <= 0.0 || spec.duration_secs > 10.0 {
        return Err(AfError::BadArg(format!(
            "voice duration_secs must be finite and in (0, 10], got {}",
            spec.duration_secs
        )));
    }
    if !spec.breath.is_finite() || !(0.0..=1.0).contains(&spec.breath) {
        return Err(AfError::BadArg(format!(
            "voice breath must be finite and in [0, 1], got {}",
            spec.breath
        )));
    }
    if !spec.rough.is_finite() || !(0.0..=1.0).contains(&spec.rough) {
        return Err(AfError::BadArg(format!(
            "voice rough must be finite and in [0, 1], got {}",
            spec.rough
        )));
    }
    if !spec.tilt.is_finite() || spec.tilt <= 0.0 {
        return Err(AfError::BadArg(format!(
            "voice tilt must be finite and > 0, got {}",
            spec.tilt
        )));
    }
    if !spec.peak.is_finite() || spec.peak <= 0.0 || spec.peak > 1.0 {
        return Err(AfError::BadArg(format!(
            "voice peak must be finite and in (0, 1], got {}",
            spec.peak
        )));
    }
    if spec.pitch_keys.is_empty() {
        return Err(AfError::BadArg("voice pitch_keys must be non-empty".into()));
    }
    if spec.amp_keys.is_empty() {
        return Err(AfError::BadArg("voice amp_keys must be non-empty".into()));
    }
    if spec.shape_keys.is_empty() {
        return Err(AfError::BadArg("voice shape_keys must be non-empty".into()));
    }
    for &(t, st) in &spec.pitch_keys {
        if !t.is_finite() || t < 0.0 || !st.is_finite() {
            return Err(AfError::BadArg(format!(
                "voice pitch_keys entry ({t}, {st}) must be finite with time >= 0"
            )));
        }
    }
    for &(t, amp) in &spec.amp_keys {
        if !t.is_finite() || t < 0.0 || !amp.is_finite() || amp < 0.0 {
            return Err(AfError::BadArg(format!(
                "voice amp_keys entry ({t}, {amp}) must be finite and >= 0"
            )));
        }
    }
    for &(t, _) in &spec.shape_keys {
        if !t.is_finite() || t < 0.0 {
            return Err(AfError::BadArg(format!(
                "voice shape_keys time {t} must be finite and >= 0"
            )));
        }
    }
    for &(t, hz) in &spec.breath_band_keys {
        if !t.is_finite() || t < 0.0 || !hz.is_finite() || hz <= 0.0 {
            return Err(AfError::BadArg(format!(
                "voice breath_band_keys entry ({t}, {hz}) must be finite with time >= 0 and hz > 0"
            )));
        }
    }
    Ok(())
}

fn sorted_keys<T: Copy>(keys: &[(f32, T)]) -> std::borrow::Cow<'_, [(f32, T)]> {
    if keys.windows(2).any(|w| w[0].0 > w[1].0) {
        let mut sorted = keys.to_vec();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        std::borrow::Cow::Owned(sorted)
    } else {
        std::borrow::Cow::Borrowed(keys)
    }
}

/// Synthesize a single pulse-and-formant voice gesture to mono PCM in `[-1, 1]`.
pub fn synthesize_voice(spec: &VoiceSpec, sample_rate: u32) -> Result<Vec<f32>> {
    validate_spec(spec, sample_rate)?;

    let pitch_keys = sorted_keys(&spec.pitch_keys);
    let amp_keys = sorted_keys(&spec.amp_keys);
    let shape_keys = sorted_keys(&spec.shape_keys);
    let breath_band_keys = sorted_keys(&spec.breath_band_keys);

    let sr_f = sample_rate as f32;
    let sr_f64 = sample_rate as f64;
    let n = (spec.duration_secs as f64 * sr_f64).round() as usize;
    if n == 0 {
        return Err(AfError::BadArg(
            "voice duration_secs is too short for sample_rate".into(),
        ));
    }

    let f0 = spec.register.base_pitch_hz();
    let mouth_scale = spec.register.mouth_scale();
    let ctrl_step = (sample_rate / 500).max(1) as usize;

    let mut inv_h_tilt = [0.0f32; 41];
    for (h, slot) in inv_h_tilt.iter_mut().enumerate().skip(1) {
        *slot = (h as f32).powf(-spec.tilt);
    }

    let mut rng = SeededRng::new(spec.seed);
    let mut res = [BandPass::new(), BandPass::new(), BandPass::new()];
    let mut air = BandPass::new();
    let mut gains = [0.0f32; 3];
    let mut drift: f32 = 0.0;
    let mut phase: f64 = 0.0;
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|_| AfError::BadArg(format!("failed to allocate {n} voice frames")))?;

    for i in 0..n {
        let t = i as f32 / sr_f;
        if i % ctrl_step == 0 {
            let (freqs, widths, g) = interp_formants(&shape_keys, t, mouth_scale);
            gains = g;
            for j in 0..3 {
                res[j].retune(freqs[j], widths[j], sr_f64);
            }
            let band = if breath_band_keys.is_empty() {
                1800.0 * mouth_scale
            } else {
                interp_f32(&breath_band_keys, t)
            };
            air.retune(band, band * 0.4, sr_f64);
            let step_noise = rng.next_f32() * 2.0 - 1.0;
            drift = (drift * 0.92 + step_noise * 0.003).clamp(-0.015, 0.015);
        }

        let st = interp_f32(&pitch_keys, t);
        let vib = 0.012 * (std::f32::consts::TAU * 5.3 * t).sin();
        let f = (f0 * (st / 12.0).exp2() * (1.0 + vib + drift)).max(20.0);

        let top = ((5200.0 / f).min(sr_f * 0.45 / f) as usize).min(40);
        let mut src = 0.0f32;
        if top > 0 && spec.breath < 1.0 {
            let (s1, c1) = phase.sin_cos();
            let mut sh = s1;
            let mut ch = c1;
            for &inv_ht in &inv_h_tilt[1..=top] {
                src += (sh as f32) * inv_ht;
                let next_s = sh * c1 + ch * s1;
                let next_c = ch * c1 - sh * s1;
                sh = next_s;
                ch = next_c;
            }
            if spec.rough > 0.0 {
                let flutter =
                    1.0 - spec.rough * 0.5 * (1.0 + (std::f32::consts::TAU * 37.0 * t).sin());
                src *= flutter;
            }
        }

        phase += std::f64::consts::TAU * (f as f64) / sr_f64;
        if phase >= std::f64::consts::TAU {
            phase -= std::f64::consts::TAU * (phase / std::f64::consts::TAU).floor();
        }

        let noise = rng.next_f32() * 2.0 - 1.0;
        let mouth_in = src * (1.0 - spec.breath) + noise * spec.breath * 5.0;
        let mut sample = 0.0f32;
        for j in 0..3 {
            sample += gains[j] * res[j].step(mouth_in);
        }
        sample += air.step(noise) * spec.breath * 0.7;

        let amp = interp_f32(&amp_keys, t);
        out.push(sample * amp);
    }

    normalize_fade_clamp(&mut out, spec.peak, sample_rate);
    Ok(out)
}

fn mood_preset(register: VoiceRegister, mood: VoiceMood, seed: u32) -> VoiceSpec {
    match mood {
        VoiceMood::Plain => VoiceSpec {
            register,
            duration_secs: 0.26,
            pitch_keys: vec![(0.0, 0.5), (0.10, 0.2), (0.26, -1.5)],
            amp_keys: vec![(0.0, 0.0), (0.03, 1.0), (0.18, 0.85), (0.26, 0.0)],
            shape_keys: vec![(0.0, MouthShape::M)],
            breath: 0.04,
            rough: 0.0,
            tilt: 1.6,
            breath_band_keys: vec![],
            peak: 0.5,
            seed,
        },
        VoiceMood::PlainB => VoiceSpec {
            register,
            duration_secs: 0.30,
            pitch_keys: vec![(0.0, -1.0), (0.16, 0.0), (0.30, 1.2)],
            amp_keys: vec![(0.0, 0.0), (0.035, 1.0), (0.22, 0.85), (0.30, 0.0)],
            shape_keys: vec![(0.0, MouthShape::M)],
            breath: 0.04,
            rough: 0.0,
            tilt: 1.6,
            breath_band_keys: vec![],
            peak: 0.5,
            seed,
        },
        VoiceMood::Glad => VoiceSpec {
            register,
            duration_secs: 0.44,
            pitch_keys: vec![(0.0, 0.0), (0.16, 0.5), (0.24, 4.0), (0.44, 5.0)],
            amp_keys: vec![
                (0.0, 0.0),
                (0.03, 0.9),
                (0.15, 0.65),
                (0.20, 0.35),
                (0.26, 1.0),
                (0.36, 0.85),
                (0.44, 0.0),
            ],
            shape_keys: vec![
                (0.0, MouthShape::M),
                (0.18, MouthShape::M),
                (0.24, MouthShape::Uh),
                (0.44, MouthShape::Uh),
            ],
            breath: 0.05,
            rough: 0.0,
            tilt: 1.6,
            breath_band_keys: vec![],
            peak: 0.5,
            seed,
        },
        VoiceMood::Afraid => VoiceSpec {
            register,
            duration_secs: 0.34,
            pitch_keys: vec![(0.0, 6.0), (0.12, 10.0), (0.34, 9.0)],
            amp_keys: vec![(0.0, 0.0), (0.025, 1.0), (0.14, 0.9), (0.34, 0.0)],
            shape_keys: vec![
                (0.0, MouthShape::Uh),
                (0.10, MouthShape::A),
                (0.34, MouthShape::A),
            ],
            breath: 0.80,
            rough: 0.0,
            tilt: 1.6,
            breath_band_keys: vec![(0.0, 900.0), (0.12, 2600.0), (0.34, 2200.0)],
            peak: 0.5,
            seed,
        },
        VoiceMood::Angry => VoiceSpec {
            register,
            duration_secs: 0.32,
            pitch_keys: vec![(0.0, -2.0), (0.12, -3.5), (0.32, -6.0)],
            amp_keys: vec![(0.0, 0.0), (0.025, 1.0), (0.20, 0.9), (0.32, 0.0)],
            shape_keys: vec![
                (0.0, MouthShape::Uh),
                (0.16, MouthShape::M),
                (0.32, MouthShape::M),
            ],
            breath: 0.10,
            rough: 0.60,
            tilt: 1.25,
            breath_band_keys: vec![],
            peak: 0.5,
            seed,
        },
        VoiceMood::Sad => VoiceSpec {
            register,
            duration_secs: 0.78,
            pitch_keys: vec![(0.0, 2.0), (0.24, 1.0), (0.78, -5.0)],
            amp_keys: vec![
                (0.0, 0.0),
                (0.08, 0.9),
                (0.32, 1.0),
                (0.62, 0.45),
                (0.78, 0.0),
            ],
            shape_keys: vec![
                (0.0, MouthShape::O),
                (0.30, MouthShape::O),
                (0.55, MouthShape::U),
                (0.78, MouthShape::U),
            ],
            breath: 0.45,
            rough: 0.0,
            tilt: 1.6,
            breath_band_keys: vec![(0.0, 1500.0), (0.78, 800.0)],
            peak: 0.5,
            seed,
        },
        VoiceMood::Sly => VoiceSpec {
            register,
            duration_secs: 0.34,
            pitch_keys: vec![(0.0, 1.0), (0.15, -0.5), (0.20, 0.0), (0.34, -3.5)],
            amp_keys: vec![
                (0.0, 0.0),
                (0.025, 0.95),
                (0.11, 0.6),
                (0.16, 0.15),
                (0.21, 1.0),
                (0.28, 0.7),
                (0.34, 0.0),
            ],
            shape_keys: vec![
                (0.0, MouthShape::E),
                (0.16, MouthShape::E),
                (0.22, MouthShape::Uh),
                (0.34, MouthShape::Uh),
            ],
            breath: 0.35,
            rough: 0.0,
            tilt: 1.6,
            breath_band_keys: vec![],
            peak: 0.5,
            seed,
        },
        VoiceMood::Tired => VoiceSpec {
            register,
            duration_secs: 0.86,
            pitch_keys: vec![(0.0, -3.0), (0.30, -4.5), (0.86, -8.0)],
            amp_keys: vec![
                (0.0, 0.0),
                (0.09, 0.85),
                (0.32, 1.0),
                (0.68, 0.4),
                (0.86, 0.0),
            ],
            shape_keys: vec![
                (0.0, MouthShape::U),
                (0.36, MouthShape::U),
                (0.64, MouthShape::M),
                (0.86, MouthShape::M),
            ],
            breath: 0.80,
            rough: 0.0,
            tilt: 1.6,
            breath_band_keys: vec![(0.0, 1400.0), (0.86, 650.0)],
            peak: 0.5,
            seed,
        },
    }
}

fn scale_spec_timing(spec: &mut VoiceSpec, dur_scale: f32, pitch_offset: f32) {
    spec.duration_secs *= dur_scale;
    for (t, st) in &mut spec.pitch_keys {
        *t *= dur_scale;
        *st += pitch_offset;
    }
    for (t, _) in &mut spec.amp_keys {
        *t *= dur_scale;
    }
    for (t, _) in &mut spec.shape_keys {
        *t *= dur_scale;
    }
    for (t, _) in &mut spec.breath_band_keys {
        *t *= dur_scale;
    }
}

fn alternate_mouth(mood: VoiceMood, idx: u8, roll: u32) -> MouthShape {
    const PALETTE_CLOSED: [MouthShape; 3] = [MouthShape::M, MouthShape::Uh, MouthShape::O];
    const PALETTE_BRIGHT: [MouthShape; 3] = [MouthShape::Uh, MouthShape::E, MouthShape::A];
    const PALETTE_DARK: [MouthShape; 3] = [MouthShape::O, MouthShape::U, MouthShape::M];
    let palette = match mood {
        VoiceMood::Plain | VoiceMood::PlainB => &PALETTE_CLOSED,
        VoiceMood::Glad | VoiceMood::Afraid | VoiceMood::Sly => &PALETTE_BRIGHT,
        VoiceMood::Angry | VoiceMood::Sad | VoiceMood::Tired => &PALETTE_DARK,
    };
    palette[((idx as usize).wrapping_add(roll as usize)) % palette.len()]
}

/// Synthesize a deterministic wordless utterance for `(register, mood, seed, syllables)`.
pub fn synthesize_utterance(
    register: VoiceRegister,
    mood: VoiceMood,
    seed: u32,
    syllables: u8,
    sample_rate: u32,
) -> Result<Vec<f32>> {
    if syllables == 0 || syllables > 8 {
        return Err(AfError::BadArg(format!(
            "voice syllables must be in 1..=8, got {syllables}"
        )));
    }
    if sample_rate < 8000 {
        return Err(AfError::BadArg(format!(
            "voice sample_rate must be >= 8000, got {sample_rate}"
        )));
    }

    let mut rng = SeededRng::new(seed);
    let base_pitch_offset = rng.range_f32(-1.0, 1.0);

    if syllables == 1 {
        let dur_scale = rng.range_f32(0.92, 1.08);
        let syl_seed = rng.next_u32();
        let mut spec = mood_preset(register, mood, syl_seed);
        scale_spec_timing(&mut spec, dur_scale, base_pitch_offset);
        return synthesize_voice(&spec, sample_rate);
    }

    let gap_samples = (0.040 * sample_rate as f64).round() as usize;
    let mut out = Vec::new();
    for idx in 0..syllables {
        let is_last = idx + 1 == syllables;
        let dur_scale = if is_last {
            rng.range_f32(0.84, 1.00)
        } else {
            rng.range_f32(0.72, 0.88)
        };
        let syl_pitch =
            base_pitch_offset + rng.range_f32(-1.4, 1.4) + if is_last { -0.5 } else { 0.35 };
        let syl_gain = if is_last {
            1.0
        } else {
            rng.range_f32(0.82, 1.0)
        };
        let shape_roll = rng.next_u32();
        let syl_seed = rng.next_u32();

        let mut spec = mood_preset(register, mood, syl_seed);
        if !is_last {
            let alt = alternate_mouth(mood, idx, shape_roll);
            if spec.shape_keys.len() == 1 {
                let mid_t = spec.duration_secs * 0.45;
                let end_t = spec.duration_secs;
                let orig = spec.shape_keys[0].1;
                spec.shape_keys = vec![(0.0, alt), (mid_t, orig), (end_t, orig)];
            } else {
                spec.shape_keys[0].1 = alt;
            }
        }
        scale_spec_timing(&mut spec, dur_scale, syl_pitch);
        spec.peak = 0.5 * syl_gain;

        let syl_pcm = synthesize_voice(&spec, sample_rate)?;
        if idx > 0 {
            out.resize(out.len() + gap_samples, 0.0);
        }
        out.extend_from_slice(&syl_pcm);
    }

    normalize_fade_clamp(&mut out, 0.5, sample_rate);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_round_trips_enums_snake_case() {
        for r in ALL_REGISTERS {
            let s = serde_json::to_string(&r).unwrap();
            let back: VoiceRegister = serde_json::from_str(&s).unwrap();
            assert_eq!(r, back);
        }
        for m in ALL_MOODS {
            let s = serde_json::to_string(&m).unwrap();
            let back: VoiceMood = serde_json::from_str(&s).unwrap();
            assert_eq!(m, back);
        }
        assert_eq!(
            serde_json::to_string(&VoiceMood::PlainB).unwrap(),
            "\"plain_b\""
        );
        for sh in [
            MouthShape::M,
            MouthShape::A,
            MouthShape::O,
            MouthShape::E,
            MouthShape::U,
            MouthShape::Uh,
        ] {
            let s = serde_json::to_string(&sh).unwrap();
            let back: MouthShape = serde_json::from_str(&s).unwrap();
            assert_eq!(sh, back);
        }
    }

    #[test]
    fn synthesize_voice_rejects_bad_arguments() {
        let base = VoiceSpec {
            register: VoiceRegister::Mid,
            duration_secs: 0.25,
            pitch_keys: vec![(0.0, 0.0)],
            amp_keys: vec![(0.0, 1.0)],
            shape_keys: vec![(0.0, MouthShape::A)],
            breath: 0.1,
            rough: 0.0,
            tilt: 1.6,
            breath_band_keys: vec![],
            peak: 0.5,
            seed: 1,
        };
        assert!(synthesize_voice(&base, 24_000).is_ok());
        assert!(synthesize_voice(&base, 7_999).is_err());

        let mut bad = base.clone();
        bad.duration_secs = 0.0;
        assert!(synthesize_voice(&bad, 24_000).is_err());
        bad.duration_secs = 10.5;
        assert!(synthesize_voice(&bad, 24_000).is_err());

        let mut bad = base.clone();
        bad.breath = -0.1;
        assert!(synthesize_voice(&bad, 24_000).is_err());
        bad.breath = 1.1;
        assert!(synthesize_voice(&bad, 24_000).is_err());

        let mut bad = base.clone();
        bad.rough = 1.2;
        assert!(synthesize_voice(&bad, 24_000).is_err());

        let mut bad = base.clone();
        bad.tilt = 0.0;
        assert!(synthesize_voice(&bad, 24_000).is_err());

        let mut bad = base.clone();
        bad.peak = 0.0;
        assert!(synthesize_voice(&bad, 24_000).is_err());
        bad.peak = 1.1;
        assert!(synthesize_voice(&bad, 24_000).is_err());

        let mut bad = base.clone();
        bad.pitch_keys.clear();
        assert!(synthesize_voice(&bad, 24_000).is_err());

        let mut bad = base.clone();
        bad.amp_keys.clear();
        assert!(synthesize_voice(&bad, 24_000).is_err());

        let mut bad = base.clone();
        bad.shape_keys.clear();
        assert!(synthesize_voice(&bad, 24_000).is_err());
    }

    #[test]
    fn unsorted_keyframes_match_sorted_output_bit_for_bit() {
        let sorted = VoiceSpec {
            register: VoiceRegister::Mid,
            duration_secs: 0.30,
            pitch_keys: vec![(0.0, -1.0), (0.15, 1.0), (0.30, 0.0)],
            amp_keys: vec![(0.0, 0.0), (0.05, 1.0), (0.30, 0.0)],
            shape_keys: vec![(0.0, MouthShape::M), (0.15, MouthShape::A)],
            breath: 0.1,
            rough: 0.1,
            tilt: 1.6,
            breath_band_keys: vec![(0.0, 1200.0), (0.30, 2200.0)],
            peak: 0.5,
            seed: 42,
        };
        let mut unsorted = sorted.clone();
        unsorted.pitch_keys.reverse();
        unsorted.amp_keys.reverse();
        unsorted.shape_keys.reverse();
        unsorted.breath_band_keys.reverse();
        let a = synthesize_voice(&sorted, 48_000).unwrap();
        let b = synthesize_voice(&unsorted, 48_000).unwrap();
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    #[test]
    fn rejects_excessive_rate_and_sanitizes_nonfinite_normalization() {
        let spec = VoiceSpec {
            register: VoiceRegister::Mid,
            duration_secs: 0.2,
            pitch_keys: vec![(0.0, 0.0)],
            amp_keys: vec![(0.0, 1.0)],
            shape_keys: vec![(0.0, MouthShape::A)],
            breath: 0.1,
            rough: 0.1,
            tilt: 1.5,
            breath_band_keys: vec![(0.0, 1500.0)],
            peak: 0.5,
            seed: 7,
        };
        assert!(synthesize_voice(&spec, u32::MAX).is_err());

        let mut buf = [0.25f32, f32::INFINITY, f32::NAN, -0.5];
        normalize_fade_clamp(&mut buf, 0.5, 48_000);
        assert!(buf
            .iter()
            .all(|v| v.is_finite() && (-1.0..=1.0).contains(v)));
    }
}
