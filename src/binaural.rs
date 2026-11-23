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
/// Internal state uses `f64` so extreme finite inputs cannot overflow into
/// `Inf - Inf = NaN` wedges; subnormals are flushed to zero.
#[derive(Debug, Clone, Copy)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Biquad {
    fn new_zero() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// New coefficients, keeping the running state (no clicks on retune).
    fn set(&mut self, b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) {
        if !a0.is_finite() || a0.abs() < 1e-12 {
            return;
        }
        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;
    }

    fn set_lowpass(&mut self, freq_hz: f32, q: f32, sample_rate: f32) {
        let w0 = 2.0 * std::f64::consts::PI * ((freq_hz / sample_rate).clamp(0.001, 0.49) as f64);
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * (q.max(0.05) as f64));
        self.set(
            (1.0 - cos) / 2.0,
            1.0 - cos,
            (1.0 - cos) / 2.0,
            1.0 + alpha,
            -2.0 * cos,
            1.0 - alpha,
        );
    }

    fn set_notch(&mut self, freq_hz: f32, q: f32, sample_rate: f32) {
        let w0 = 2.0 * std::f64::consts::PI * ((freq_hz / sample_rate).clamp(0.001, 0.49) as f64);
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * (q.max(0.05) as f64));
        self.set(1.0, -2.0 * cos, 1.0, 1.0 + alpha, -2.0 * cos, 1.0 - alpha);
    }

    fn set_peaking(&mut self, freq_hz: f32, q: f32, gain_db: f32, sample_rate: f32) {
        let a = 10.0f64.powf(gain_db as f64 / 40.0);
        let w0 = 2.0 * std::f64::consts::PI * ((freq_hz / sample_rate).clamp(0.001, 0.49) as f64);
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * (q.max(0.05) as f64));
        self.set(
            1.0 + alpha * a,
            -2.0 * cos,
            1.0 - alpha * a,
            1.0 + alpha / a,
            -2.0 * cos,
            1.0 - alpha / a,
        );
    }

    fn lowpass(freq_hz: f32, q: f32, sample_rate: f32) -> Self {
        let mut q_filt = Self::new_zero();
        q_filt.set_lowpass(freq_hz, q, sample_rate);
        q_filt
    }

    fn notch(freq_hz: f32, q: f32, sample_rate: f32) -> Self {
        let mut q_filt = Self::new_zero();
        q_filt.set_notch(freq_hz, q, sample_rate);
        q_filt
    }

    fn peaking(freq_hz: f32, q: f32, gain_db: f32, sample_rate: f32) -> Self {
        let mut q_filt = Self::new_zero();
        q_filt.set_peaking(freq_hz, q, gain_db, sample_rate);
        q_filt
    }

    fn run(&mut self, x: f32) -> f32 {
        let x = if x.is_finite() { x as f64 } else { 0.0 };
        let mut y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        if !y.is_finite() {
            self.x1 = 0.0;
            self.x2 = 0.0;
            self.y1 = 0.0;
            self.y2 = 0.0;
            return 0.0;
        }
        if y.abs() < 1e-30 {
            y = 0.0;
        }
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y.clamp(-f32::MAX as f64, f32::MAX as f64) as f32
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
        let sr = sample_rate.clamp(1, 768_000) as f32;
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

    fn retune(&mut self, azimuth: f32, elevation: f32) {
        let el = elevation.clamp(-PI / 2.0, PI / 2.0);
        if (azimuth - self.last_az).abs() > 1e-4 || (el - self.last_el).abs() > 1e-4 {
            let notch_hz = 6_000.0 + 3_000.0 * el.sin();
            self.pinna_l.set_notch(notch_hz, 1.8, self.sample_rate);
            self.pinna_r.set_notch(notch_hz, 1.8, self.sample_rate);
            let rear = smoothstep(azimuth.abs(), PI / 2.0, PI);
            self.rear_l
                .set_peaking(2_800.0, 1.0, -3.5 * rear, self.sample_rate);
            self.rear_r
                .set_peaking(2_800.0, 1.0, -3.5 * rear, self.sample_rate);
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
        let frames = mono.len().min(left.len()).min(right.len());
        if frames == 0 {
            return;
        }
        let az = if azimuth.is_finite() {
            wrap_pi(azimuth)
        } else {
            0.0
        };
        let el = if elevation.is_finite() {
            elevation.clamp(-PI / 2.0, PI / 2.0)
        } else {
            0.0
        };
        self.retune(az, el);
        let s = az.sin();
        let side = s.abs().min(1.0);
        // Woodworth-ish split: the far ear waits, and hears less highs.
        let itd_l = (-s).max(0.0) * MAX_ITD_SECS * self.sample_rate;
        let itd_r = s.max(0.0) * MAX_ITD_SECS * self.sample_rate;
        let dark = 18_000.0_f32.lerp(900.0, side.powf(1.5)).max(120.0);
        if (dark - self.last_dark).abs() > 1.0 {
            self.shadow_l.set_lowpass(dark, 0.7, self.sample_rate);
            self.shadow_r.set_lowpass(dark, 0.7, self.sample_rate);
            self.last_dark = dark;
        }
        let g_far = 1.0 - 0.55 * side.powf(1.5);
        let (gl, gr) = if s >= 0.0 { (g_far, 1.0) } else { (1.0, g_far) };
        let n = self.delay_l.len();
        for (k, &raw_x) in mono[..frames].iter().enumerate() {
            let x = if raw_x.is_finite() { raw_x } else { 0.0 };
            let p = self.pos;
            self.delay_l[p] = x;
            self.delay_r[p] = x;
            self.pos = (p + 1) % n;
            let l = read_delay(&self.delay_l, self.pos, itd_l) * gl;
            let r = read_delay(&self.delay_r, self.pos, itd_r) * gr;
            let l = self.rear_l.run(self.pinna_l.run(self.shadow_l.run(l)));
            let r = self.rear_r.run(self.pinna_r.run(self.shadow_r.run(r)));
            add_finite(&mut left[k], l);
            add_finite(&mut right[k], r);
        }
    }
}

#[inline]
fn add_finite(dst: &mut f32, delta: f32) {
    let cur = if dst.is_finite() { *dst } else { 0.0 };
    let d = if delta.is_finite() { delta } else { 0.0 };
    let sum = cur + d;
    *dst = if sum.is_finite() {
        sum
    } else {
        (cur as f64 + d as f64).clamp(-f32::MAX as f64, f32::MAX as f64) as f32
    };
}

fn read_delay(line: &[f32], pos: usize, delay: f32) -> f32 {
    let n = line.len() as f32;
    let at = (pos as f32 - delay).rem_euclid(n);
    let i0 = at.floor() as usize % line.len();
    let i1 = (i0 + 1) % line.len();
    let f = (at - at.floor()) as f64;
    let a = line[i0] as f64;
    let b = line[i1] as f64;
    (a + (b - a) * f).clamp(-f32::MAX as f64, f32::MAX as f64) as f32
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

    #[test]
    fn handles_mismatched_buffers_and_nonfinite_inputs() {
        let mut st = BinauralState::new(48_000);
        let mut l = vec![0.0; 16];
        let mut r = vec![0.0; 4];
        st.process_into(&[0.1; 16], 0.5, 0.2, &mut l, &mut r);

        let mut st2 = BinauralState::new(48_000);
        let mut l2 = vec![0.0; 8];
        let mut r2 = vec![0.0; 8];
        st2.process_into(
            &[f32::NAN, 0.2, 0.3],
            f32::NAN,
            f32::INFINITY,
            &mut l2[..3],
            &mut r2[..3],
        );
        st2.process_into(&[0.5; 8], 0.1, 0.0, &mut l2, &mut r2);
        assert!(l2.iter().chain(&r2).all(|x| x.is_finite()));
    }
}
