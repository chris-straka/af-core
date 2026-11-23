//! Head-related binaural rendering: mono in, stereo out, for one direction.
//!
//! A spherical-head model (interaural time and level differences) plus a
//! pinna notch that separates front from back, so a sound ahead and the
//! same sound behind read differently on headphones. Cheap enough per
//! voice; deterministic (no randomness, no tables).
//!
//! Listener space: azimuth 0 straight ahead, + to the right, in [-PI, PI];
//! elevation 0 level, + above. Out-of-range inputs are wrapped/clamped.

use std::f32::consts::PI;

pub const HEAD_RADIUS_M: f32 = 0.0875;
pub const SPEED_OF_SOUND_M_S: f32 = 343.0;
/// The longest interaural delay, at a source hard to one side.
pub const MAX_ITD_SECS: f32 = HEAD_RADIUS_M / SPEED_OF_SOUND_M_S * (PI / 2.0 + 1.0);

/// Second-order IIR section (RBJ cookbook), direct form I.
#[derive(Debug, Clone, Copy)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    fn new(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Self {
        let mut q = Biquad {
            b0: 0.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        };
        q.set(b0, b1, b2, a0, a1, a2);
        q
    }
    /// New coefficients, keeping the running state (no clicks on retune).
    fn set(&mut self, b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) {
        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;
    }
    fn lowpass(freq_hz: f32, q: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * PI * (freq_hz / sample_rate).clamp(0.001, 0.49);
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q.max(0.05));
        Biquad::new(
            (1.0 - cos) / 2.0,
            1.0 - cos,
            (1.0 - cos) / 2.0,
            1.0 + alpha,
            -2.0 * cos,
            1.0 - alpha,
        )
    }
    fn notch(freq_hz: f32, q: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * PI * (freq_hz / sample_rate).clamp(0.001, 0.49);
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q.max(0.05));
        Biquad::new(1.0, -2.0 * cos, 1.0, 1.0 + alpha, -2.0 * cos, 1.0 - alpha)
    }
    fn peaking(freq_hz: f32, q: f32, gain_db: f32, sample_rate: f32) -> Self {
        let a = 10.0f32.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * (freq_hz / sample_rate).clamp(0.001, 0.49);
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q.max(0.05));
        Biquad::new(
            1.0 + alpha * a,
            -2.0 * cos,
            1.0 - alpha * a,
            1.0 + alpha / a,
            -2.0 * cos,
            1.0 - alpha / a,
        )
    }
    fn run(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Persistent per-voice state for block-rate rendering ([`crate::mixer`]).
#[derive(Debug, Clone)]
pub struct BinauralState {
    sample_rate: f32,
    /// Fractional-delay lines (both fed the same input).
    delay_l: Vec<f32>,
    delay_r: Vec<f32>,
    pos: usize,
    shadow_l: Biquad,
    shadow_r: Biquad,
    pinna_l: Biquad,
    pinna_r: Biquad,
    rear_l: Biquad,
    rear_r: Biquad,
    last_az: f32,
    last_el: f32,
    last_dark: f32,
}

impl BinauralState {
    pub fn new(sample_rate: u32) -> Self {
        let sr = sample_rate.max(1) as f32;
        let max_delay = (MAX_ITD_SECS * sr).ceil() as usize + 2;
        BinauralState {
            sample_rate: sr,
            delay_l: vec![0.0; max_delay],
            delay_r: vec![0.0; max_delay],
            pos: 0,
            shadow_l: Biquad::lowpass(18_000.0, 0.7, sr),
            shadow_r: Biquad::lowpass(18_000.0, 0.7, sr),
            pinna_l: Biquad::notch(7_500.0, 1.8, sr),
            pinna_r: Biquad::notch(7_500.0, 1.8, sr),
            rear_l: Biquad::peaking(2_800.0, 1.0, 0.0, sr),
            rear_r: Biquad::peaking(2_800.0, 1.0, 0.0, sr),
            // Infinity retunes on the first block, then only on change.
            last_az: f32::INFINITY,
            last_el: f32::INFINITY,
            last_dark: f32::INFINITY,
        }
    }

    fn coefs_lowpass(freq_hz: f32, q: f32, sr: f32) -> (f32, f32, f32, f32, f32, f32) {
        let w0 = 2.0 * PI * (freq_hz / sr).clamp(0.001, 0.49);
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q.max(0.05));
        (
            (1.0 - cos) / 2.0,
            1.0 - cos,
            (1.0 - cos) / 2.0,
            1.0 + alpha,
            -2.0 * cos,
            1.0 - alpha,
        )
    }
    fn coefs_notch(freq_hz: f32, q: f32, sr: f32) -> (f32, f32, f32, f32, f32, f32) {
        let w0 = 2.0 * PI * (freq_hz / sr).clamp(0.001, 0.49);
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q.max(0.05));
        (1.0, -2.0 * cos, 1.0, 1.0 + alpha, -2.0 * cos, 1.0 - alpha)
    }
    fn coefs_peak(freq_hz: f32, q: f32, gain_db: f32, sr: f32) -> (f32, f32, f32, f32, f32, f32) {
        let a = 10.0f32.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * (freq_hz / sr).clamp(0.001, 0.49);
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q.max(0.05));
        (
            1.0 + alpha * a,
            -2.0 * cos,
            1.0 - alpha * a,
            1.0 + alpha / a,
            -2.0 * cos,
            1.0 - alpha / a,
        )
    }

    fn retune(&mut self, azimuth: f32, elevation: f32) {
        let el = elevation.clamp(-PI / 2.0, PI / 2.0);
        if (azimuth - self.last_az).abs() > 1e-4 || (el - self.last_el).abs() > 1e-4 {
            let (b0, b1, b2, a0, a1, a2) =
                Self::coefs_notch(6_000.0 + 3_000.0 * el.sin(), 1.8, self.sample_rate);
            self.pinna_l.set(b0, b1, b2, a0, a1, a2);
            self.pinna_r.set(b0, b1, b2, a0, a1, a2);
            let rear = smoothstep(azimuth.abs(), PI / 2.0, PI);
            let (b0, b1, b2, a0, a1, a2) =
                Self::coefs_peak(2_800.0, 1.0, -3.5 * rear, self.sample_rate);
            self.rear_l.set(b0, b1, b2, a0, a1, a2);
            self.rear_r.set(b0, b1, b2, a0, a1, a2);
            self.last_az = azimuth;
            self.last_el = el;
        }
    }

    /// Render one block of mono, adding the HRTF result into `left`/`right`
    /// (which must hold at least `mono.len()` frames each). State persists
    /// across blocks; retunes are coefficient-only.
    pub fn process_into(
        &mut self,
        mono: &[f32],
        azimuth: f32,
        elevation: f32,
        left: &mut [f32],
        right: &mut [f32],
    ) {
        let az = wrap_pi(azimuth);
        self.retune(az, elevation);
        let s = az.sin();
        let side = s.abs().min(1.0);
        // Woodworth-ish split: the far ear waits, and hears less highs.
        let itd_l = (-s).max(0.0) * MAX_ITD_SECS * self.sample_rate;
        let itd_r = s.max(0.0) * MAX_ITD_SECS * self.sample_rate;
        let dark = 18_000.0_f32.lerp(900.0, side.powf(1.5)).max(120.0);
        if (dark - self.last_dark).abs() > 1.0 {
            let (b0, b1, b2, a0, a1, a2) = Self::coefs_lowpass(dark, 0.7, self.sample_rate);
            self.shadow_l.set(b0, b1, b2, a0, a1, a2);
            self.shadow_r.set(b0, b1, b2, a0, a1, a2);
            self.last_dark = dark;
        }
        let g_far = 1.0 - 0.55 * side.powf(1.5);
        let (gl, gr) = if s >= 0.0 { (g_far, 1.0) } else { (1.0, g_far) };
        let n = self.delay_l.len();
        for (k, &x) in mono.iter().enumerate() {
            let p = self.pos;
            self.delay_l[p] = x;
            self.delay_r[p] = x;
            self.pos = (p + 1) % n;
            let l = read_delay(&self.delay_l, self.pos, itd_l) * gl;
            let r = read_delay(&self.delay_r, self.pos, itd_r) * gr;
            let l = self.rear_l.run(self.pinna_l.run(self.shadow_l.run(l)));
            let r = self.rear_r.run(self.pinna_r.run(self.shadow_r.run(r)));
            left[k] += l;
            right[k] += r;
        }
    }
}

fn read_delay(line: &[f32], pos: usize, delay: f32) -> f32 {
    let n = line.len() as f32;
    let at = (pos as f32 - delay).rem_euclid(n);
    let i0 = at.floor() as usize % line.len();
    let i1 = (i0 + 1) % line.len();
    let f = at - at.floor();
    line[i0] * (1.0 - f) + line[i1] * f
}

fn wrap_pi(a: f32) -> f32 {
    let mut a = a % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    }
    if a < -PI {
        a += 2.0 * PI;
    }
    a
}

fn smoothstep(x: f32, e0: f32, e1: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

trait Lerp {
    fn lerp(self, other: f32, t: f32) -> f32;
}
impl Lerp for f32 {
    fn lerp(self, other: f32, t: f32) -> f32 {
        self + (other - self) * t
    }
}

/// Render `mono` (at `sample_rate` Hz) for a source at `azimuth`/`elevation`
/// radians in listener space. Returns (left, right).
pub fn render(
    mono: &[f32],
    azimuth: f32,
    elevation: f32,
    sample_rate: u32,
) -> (Vec<f32>, Vec<f32>) {
    let mut st = BinauralState::new(sample_rate);
    let mut l = vec![0.0; mono.len()];
    let mut r = vec![0.0; mono.len()];
    st.process_into(mono, azimuth, elevation, &mut l, &mut r);
    (l, r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(len: usize, freq: f32, sr: f32) -> Vec<f32> {
        (0..len)
            .map(|i| (2.0 * PI * freq * i as f32 / sr).sin())
            .collect()
    }
    /// Deterministic broadband burst (a sum of sines to 9 kHz).
    fn burst(len: usize, sr: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / sr;
                (2.0 * PI * 500.0 * t).sin() * 0.5
                    + (2.0 * PI * 1500.0 * t).sin() * 0.35
                    + (2.0 * PI * 3200.0 * t).sin() * 0.3
                    + (2.0 * PI * 5600.0 * t).sin() * 0.22
                    + (2.0 * PI * 8200.0 * t).sin() * 0.15
            })
            .collect()
    }
    fn hf_energy(v: &[f32]) -> f32 {
        // High-pass-ish: energy of the sample-to-sample difference.
        v.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum::<f32>() / v.len().max(1) as f32
    }

    #[test]
    fn silence_stays_silence() {
        let (l, r) = render(&vec![0.0; 512], 0.7, 0.2, 48_000);
        assert_eq!(l.len(), 512);
        assert!(l.iter().all(|&x| x == 0.0) && r.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn centered_is_symmetric() {
        let (l, r) = render(&tone(2048, 440.0, 48_000.0), 0.0, 0.0, 48_000);
        let diff: f32 = l.iter().zip(&r).map(|(a, b)| (a - b).abs()).sum::<f32>() / l.len() as f32;
        assert!(diff < 1e-4, "L/R differ by {diff}");
    }

    #[test]
    fn hard_right_arrives_right_first_and_louder() {
        let (l, r) = render(&tone(2048, 880.0, 48_000.0), PI / 2.0, 0.0, 48_000);
        // Right leads: the right channel's first big peak comes earlier.
        let peak = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        assert!(
            peak(&r) < peak(&l),
            "right should lead ({} vs {})",
            peak(&r),
            peak(&l)
        );
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        assert!(rms(&r) > rms(&l) * 1.2, "right should be louder");
        assert!(hf_energy(&r) > hf_energy(&l) * 2.0, "left should be darker");
    }

    #[test]
    fn front_and_back_differ() {
        let (fl, _fr) = render(&burst(2048, 48_000.0), 0.6, 0.0, 48_000);
        let (bl, _br) = render(&burst(2048, 48_000.0), PI - 0.6, 0.0, 48_000);
        // Same loudness either side, but the rear's pinna dip changes colour.
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        assert!((rms(&fl) - rms(&bl)).abs() / rms(&fl) < 0.2);
        let colour = hf_energy(&fl) / hf_energy(&bl).max(1e-9);
        assert!(
            colour > 1.1 || colour < 1.0 / 1.1,
            "front/back should colour differently ({colour})"
        );
    }

    #[test]
    fn elevation_changes_the_sound() {
        let (a, _) = render(&burst(2048, 48_000.0), 0.3, 0.0, 48_000);
        let (b, _) = render(&burst(2048, 48_000.0), 0.3, 0.9, 48_000);
        let diff: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
        assert!(diff > 1e-5, "elevation should matter");
    }

    #[test]
    fn deterministic_and_finite() {
        let m = tone(1024, 300.0, 48_000.0);
        let (l1, r1) = render(&m, 10.0, -3.0, 48_000);
        let (l2, r2) = render(&m, 10.0, -3.0, 48_000);
        assert_eq!(l1, l2);
        assert_eq!(r1, r2);
        assert!(l1.iter().chain(&r1).all(|x| x.is_finite()));
    }
}
