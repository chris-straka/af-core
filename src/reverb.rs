//! Reverb for zone sends: a Freeverb-style Schroeder network
//! (8 damped combs + 4 allpasses per channel, stereo spread).
//!
//! Instead of a direct-form 8192-tap FIR impulse (which makes debug-build
//! scene renders crawl), the recursive network gives a dense tail for ~24
//! multiplies per sample and is equally deterministic (pure f32 recurrences,
//! no RNG). Delay storage is allocated once in [`Reverb::new`]; processing
//! never allocates (audio-thread safe).

use crate::error::{AfError, Result};

const COMB_TUNING: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
const ALLPASS_TUNING: [usize; 4] = [556, 441, 341, 225];
const STEREO_SPREAD: usize = 23;
const FIXED_GAIN: f32 = 0.015;

/// User-facing reverb parameters (scene bus `reverb` block).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReverbParams {
    /// 0..=1, larger = longer tail.
    pub room_size: f32,
    /// 0..=1, larger = darker tail.
    pub damping: f32,
}

impl Default for ReverbParams {
    fn default() -> Self {
        Self {
            room_size: 0.7,
            damping: 0.4,
        }
    }
}

impl ReverbParams {
    pub fn validate(&self) -> Result<()> {
        for (name, v) in [("room_size", self.room_size), ("damping", self.damping)] {
            if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                return Err(AfError::BadArg(format!(
                    "reverb {name} must be finite and in [0, 1], got {v}"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Comb {
    buf: Vec<f32>,
    pos: usize,
    store: f32,
}

impl Comb {
    fn new(len: usize) -> Self {
        Self {
            buf: vec![0.0; len.max(1)],
            pos: 0,
            store: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32, feedback: f32, damp: f32) -> f32 {
        let y = self.buf[self.pos];
        self.store = y * (1.0 - damp) + self.store * damp;
        self.buf[self.pos] = x + self.store * feedback;
        self.pos += 1;
        if self.pos == self.buf.len() {
            self.pos = 0;
        }
        y
    }
}

#[derive(Debug, Clone)]
struct Allpass {
    buf: Vec<f32>,
    pos: usize,
}

impl Allpass {
    fn new(len: usize) -> Self {
        Self {
            buf: vec![0.0; len.max(1)],
            pos: 0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let b = self.buf[self.pos];
        self.buf[self.pos] = x + b * 0.5;
        self.pos += 1;
        if self.pos == self.buf.len() {
            self.pos = 0;
        }
        b - x
    }
}

#[derive(Debug, Clone)]
pub struct Reverb {
    combs: [Vec<Comb>; 2],
    allpasses: [Vec<Allpass>; 2],
    feedback: f32,
    damp: f32,
}

impl Reverb {
    /// Allocates all delay lines (control thread).
    pub fn new(params: ReverbParams, sample_rate: u32) -> Result<Self> {
        params.validate()?;
        if sample_rate == 0 || sample_rate > 768_000 {
            return Err(AfError::BadArg(format!(
                "reverb sample_rate must be in 1..=768_000, got {sample_rate}"
            )));
        }
        let scale = sample_rate as f64 / 44_100.0;
        let len = |n: usize, spread: usize| ((n + spread) as f64 * scale).round() as usize;
        let mk = |spread: usize| {
            (
                COMB_TUNING
                    .iter()
                    .map(|&n| Comb::new(len(n, spread)))
                    .collect(),
                ALLPASS_TUNING
                    .iter()
                    .map(|&n| Allpass::new(len(n, spread)))
                    .collect(),
            )
        };
        let (cl, al) = mk(0);
        let (cr, ar) = mk(STEREO_SPREAD);
        Ok(Self {
            combs: [cl, cr],
            allpasses: [al, ar],
            feedback: params.room_size * 0.28 + 0.7,
            damp: params.damping * 0.4,
        })
    }

    /// In-place, 100% wet: input is the mono sum of `l`/`r`; outputs the
    /// stereo tail. No allocation.
    pub fn process(&mut self, l: &mut [f32], r: &mut [f32]) {
        let n = l.len().min(r.len());
        let (fb, damp) = (self.feedback, self.damp);
        for i in 0..n {
            let li = if l[i].is_finite() { l[i] } else { 0.0 };
            let ri = if r[i].is_finite() { r[i] } else { 0.0 };
            let x = (li + ri) * FIXED_GAIN;
            for ch in 0..2 {
                let mut acc = 0.0f32;
                for c in self.combs[ch].iter_mut() {
                    acc += c.process(x, fb, damp);
                }
                for a in self.allpasses[ch].iter_mut() {
                    acc = a.process(acc);
                }
                if ch == 0 {
                    l[i] = acc;
                } else {
                    r[i] = acc;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn impulse_gives_decaying_deterministic_tail() {
        let run = || {
            let mut rv = Reverb::new(ReverbParams::default(), 48_000).unwrap();
            let n = 48_000;
            let mut l = vec![0.0f32; n];
            let mut r = vec![0.0f32; n];
            l[0] = 1.0;
            r[0] = 1.0;
            for (cl, cr) in l.chunks_mut(512).zip(r.chunks_mut(512)) {
                rv.process(cl, cr);
            }
            (l, r)
        };
        let (l, r) = run();
        assert_eq!((l.clone(), r.clone()), run());
        let energy = |s: &[f32]| s.iter().map(|v| v * v).sum::<f32>();
        let early = energy(&l[..12_000]);
        let late = energy(&l[36_000..]);
        assert!(
            early > 0.0 && late < early * 0.1,
            "early {early} late {late}"
        );
        assert_ne!(l, r, "stereo spread decorrelates channels");
        assert!(l.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rejects_bad_params() {
        assert!(Reverb::new(
            ReverbParams {
                room_size: 1.5,
                damping: 0.0
            },
            48_000
        )
        .is_err());
        assert!(Reverb::new(ReverbParams::default(), 0).is_err());
    }

    #[test]
    fn rejects_excessive_rate_and_handles_mismatched_and_nan() {
        assert!(Reverb::new(ReverbParams::default(), u32::MAX).is_err());

        let mut rv = Reverb::new(ReverbParams::default(), 48_000).unwrap();
        let mut l = vec![0.0; 16];
        let mut r = vec![0.0; 4];
        rv.process(&mut l, &mut r);

        let mut l_nan = vec![f32::NAN, 0.2, f32::INFINITY, 0.0];
        let mut r_nan = vec![0.0; 4];
        rv.process(&mut l_nan, &mut r_nan);
        let mut l2 = vec![0.1; 64];
        let mut r2 = vec![0.1; 64];
        rv.process(&mut l2, &mut r2);
        assert!(l2.iter().chain(&r2).all(|v| v.is_finite()));
    }
}
