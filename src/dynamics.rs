//! Mix-stage dynamics shared by the offline mixer and the realtime Kira
//! effects: a look-ahead true-peak limiter for the master and a
//! sidechain ducker for buses (music, ambience) keyed by marked events.
//!
//! Both are opt-in through the mix data (a bus's `limiter` / `duck`
//! block, an event's `sidechain` flag); with neither set nothing here
//! runs and renders stay bit-identical.
//!
//! Real-time rules: every buffer is allocated in `new` (control thread),
//! sized for the largest sample rate and look-ahead, so
//! `set_sample_rate` and every `process*` call are allocation-free.

use serde::{Deserialize, Serialize};

/// Highest sample rate the preallocated buffers cover.
pub const MAX_RATE: u32 = 192_000;
/// Longest look-ahead a limiter accepts (milliseconds).
pub const MAX_LOOKAHEAD_MS: f32 = 20.0;

/// The master limiter, from the `master` bus's `limiter` block.
///
/// ```json
/// { "name": "master", "limiter": { "ceiling_dbtp": -1.0 } }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LimiterSpec {
    /// Output ceiling, dB true peak (4x oversampled), in [-24, 0].
    #[serde(default = "default_ceiling")]
    pub ceiling_dbtp: f32,
    /// Look-ahead (and attack) in ms, in [0.5, 20].
    #[serde(default = "default_lookahead")]
    pub lookahead_ms: f32,
    /// Release time constant in ms, in [10, 2000].
    #[serde(default = "default_limiter_release")]
    pub release_ms: f32,
}

fn default_ceiling() -> f32 {
    -1.0
}
fn default_lookahead() -> f32 {
    5.0
}
fn default_limiter_release() -> f32 {
    100.0
}

impl Default for LimiterSpec {
    fn default() -> Self {
        Self {
            ceiling_dbtp: default_ceiling(),
            lookahead_ms: default_lookahead(),
            release_ms: default_limiter_release(),
        }
    }
}

impl LimiterSpec {
    pub fn validate(&self) -> Result<(), String> {
        let ok = |v: f32, lo: f32, hi: f32| v.is_finite() && v >= lo && v <= hi;
        if !ok(self.ceiling_dbtp, -24.0, 0.0) {
            return Err(format!(
                "limiter.ceiling_dbtp must be in [-24, 0], got {}",
                self.ceiling_dbtp
            ));
        }
        if !ok(self.lookahead_ms, 0.5, MAX_LOOKAHEAD_MS) {
            return Err(format!(
                "limiter.lookahead_ms must be in [0.5, {MAX_LOOKAHEAD_MS}], got {}",
                self.lookahead_ms
            ));
        }
        if !ok(self.release_ms, 10.0, 2000.0) {
            return Err(format!(
                "limiter.release_ms must be in [10, 2000], got {}",
                self.release_ms
            ));
        }
        Ok(())
    }
}

/// A ducked bus, from the bus's `duck` block. The key is every voice of
/// an event marked `"sidechain": true` (its level after event volume and
/// distance, before bus gains).
///
/// The bus drops by up to `amount_db`: by how far the key's level
/// (2 ms RMS) is over `threshold_db`, capped at `amount_db`. The drop
/// grows at `amount_db` per `attack_ms`, so a key must sound for about
/// `attack_ms` to reach full depth (a short tick barely moves it); it
/// holds `hold_ms` after the key falls, then recovers fully within
/// `release_ms`.
///
/// ```json
/// { "name": "music", "gain": 0.8, "duck": { "amount_db": 9 } }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DuckSpec {
    /// Deepest drop, dB, in (0, 48].
    #[serde(default = "default_amount")]
    pub amount_db: f32,
    /// Key level (dBFS RMS) where ducking starts, in [-80, 0].
    #[serde(default = "default_threshold")]
    pub threshold_db: f32,
    /// Time for the full drop, ms, in [1, 2000].
    #[serde(default = "default_attack")]
    pub attack_ms: f32,
    /// Time held after the key falls, ms, in [0, 5000].
    #[serde(default = "default_hold")]
    pub hold_ms: f32,
    /// Time for a full recovery, ms, in [10, 10000].
    #[serde(default = "default_duck_release")]
    pub release_ms: f32,
}

fn default_amount() -> f32 {
    9.0
}
fn default_threshold() -> f32 {
    -40.0
}
fn default_attack() -> f32 {
    80.0
}
fn default_hold() -> f32 {
    200.0
}
fn default_duck_release() -> f32 {
    600.0
}

impl Default for DuckSpec {
    fn default() -> Self {
        Self {
            amount_db: default_amount(),
            threshold_db: default_threshold(),
            attack_ms: default_attack(),
            hold_ms: default_hold(),
            release_ms: default_duck_release(),
        }
    }
}

impl DuckSpec {
    pub fn validate(&self) -> Result<(), String> {
        let ok = |v: f32, lo: f32, hi: f32| v.is_finite() && v >= lo && v <= hi;
        if !(self.amount_db.is_finite() && self.amount_db > 0.0 && self.amount_db <= 48.0) {
            return Err(format!(
                "duck.amount_db must be in (0, 48], got {}",
                self.amount_db
            ));
        }
        for (name, v, lo, hi) in [
            ("threshold_db", self.threshold_db, -80.0, 0.0),
            ("attack_ms", self.attack_ms, 1.0, 2000.0),
            ("hold_ms", self.hold_ms, 0.0, 5000.0),
            ("release_ms", self.release_ms, 10.0, 10_000.0),
        ] {
            if !ok(v, lo, hi) {
                return Err(format!("duck.{name} must be in [{lo}, {hi}], got {v}"));
            }
        }
        Ok(())
    }
}

/// ITU-R BS.1770-4 Annex 2 true-peak interpolator: 48 taps as 4 phases
/// of 12 (4x oversampling). Shared with [`crate::loudness`].
pub const TRUE_PEAK_PHASES: [[f32; 12]; 4] = [
    [
        0.001_708_984_4,
        0.010_986_328,
        -0.019_653_32,
        0.033_203_125,
        -0.059_448_242,
        0.137_329_1,
        0.972_167_97,
        -0.102_294_92,
        0.047_607_422,
        -0.026_611_328,
        0.014_892_578,
        -0.008_300_781,
    ],
    [
        -0.029_174_805,
        0.029_296_875,
        -0.051_757_813,
        0.089_111_33,
        -0.166_503_9,
        0.465_087_9,
        0.779_785_16,
        -0.200_317_38,
        0.101_562_5,
        -0.058_227_54,
        0.033_081_055,
        -0.018_920_898,
    ],
    [
        -0.018_920_898,
        0.033_081_055,
        -0.058_227_54,
        0.101_562_5,
        -0.200_317_38,
        0.779_785_16,
        0.465_087_9,
        -0.166_503_9,
        0.089_111_33,
        -0.051_757_813,
        0.029_296_875,
        -0.029_174_805,
    ],
    [
        -0.008_300_781,
        0.014_892_578,
        -0.026_611_328,
        0.047_607_422,
        -0.102_294_92,
        0.972_167_97,
        0.137_329_1,
        -0.059_448_242,
        0.033_203_125,
        -0.019_653_32,
        0.010_986_328,
        0.001_708_984_4,
    ],
];

/// Interpolator history length (input samples).
pub const TP_TAPS: usize = 12;
/// The interpolated points between `hist[6]` and `hist[5]` (newest at
/// 0): the detector looks this many samples behind the input.
const TP_DELAY: usize = 6;
/// The limiter holds each gain requirement over every input sample its
/// interpolation reads (the whole tap span), not just the two around it.
const TP_SPAN: usize = TP_TAPS - 1;
/// The limiter aims this far under the ceiling: the gain still moves a
/// little across an interpolator's 12 taps, which can lift the true peak
/// by a few thousandths of a dB on steep attacks.
pub const LIMITER_MARGIN_DB: f32 = 0.03;

/// Largest |x| over the samples `hist[5]`, `hist[6]` and the four
/// interpolated points around them (`hist[0]` newest).
#[inline]
pub fn interval_peak(hist: &[f32; TP_TAPS]) -> f32 {
    let mut p = hist[TP_DELAY].abs().max(hist[TP_DELAY - 1].abs());
    for ph in &TRUE_PEAK_PHASES {
        let mut y = 0.0f32;
        for (c, h) in ph.iter().zip(hist.iter()) {
            y += c * h;
        }
        p = p.max(y.abs());
    }
    p
}

fn db_to_lin(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

/// Look-ahead true-peak limiter (stereo-linked).
///
/// Per sample: the required gain `target / peak` (peak over two samples
/// and the 4x-interpolated points between them; target = ceiling - 0.03
/// dB) -> sliding minimum over the look-ahead + 11 samples ->
/// instant-attack / exponential-release envelope -> moving average over
/// the look-ahead. The audio is delayed by look-ahead + 11 samples, so
/// every input sample an interpolated peak reads is scaled by a gain at
/// or below what that peak needs; the gain reaches its floor smoothly
/// over the look-ahead (attack) instead of jumping. A final clamp at the
/// ceiling catches float rounding.
pub struct Limiter {
    spec: LimiterSpec,
    sample_rate: u32,
    ceiling: f32,
    /// Ceiling less the margin: what the gain aims for.
    target: f32,
    /// Look-ahead in samples (box length).
    look: usize,
    /// Hold window = look + TP_SPAN.
    win: usize,
    rel_coef: f32,
    // Audio delay line (look + TP_SPAN samples).
    dl: Box<[f32]>,
    dr: Box<[f32]>,
    dlen: usize,
    dpos: usize,
    // Interpolator history, newest first.
    hl: [f32; TP_TAPS],
    hr: [f32; TP_TAPS],
    // Sliding-minimum deque (ring of (index, value)).
    q_idx: Box<[u64]>,
    q_val: Box<[f32]>,
    q_head: usize,
    q_len: usize,
    // Release envelope.
    env: f32,
    // Moving average.
    boxbuf: Box<[f32]>,
    box_pos: usize,
    box_sum: f64,
    n: u64,
    // Input gain (the master bus gain, ahead of the limiter).
    in_gain: f32,
    in_target: f32,
    in_coef: f32,
    /// Lowest gain applied since the last `take_min_gain`.
    min_gain: f32,
}

impl Limiter {
    /// Allocates every buffer (control thread). `spec` must validate.
    pub fn new(spec: LimiterSpec, sample_rate: u32) -> Self {
        let cap = (MAX_LOOKAHEAD_MS as f64 * 1e-3 * MAX_RATE as f64).ceil() as usize + 8;
        let mut l = Self {
            spec,
            sample_rate: 0,
            ceiling: 1.0,
            target: 1.0,
            look: 1,
            win: 3,
            rel_coef: 0.0,
            dl: vec![0.0; cap + TP_SPAN].into_boxed_slice(),
            dr: vec![0.0; cap + TP_SPAN].into_boxed_slice(),
            dlen: 1,
            dpos: 0,
            hl: [0.0; TP_TAPS],
            hr: [0.0; TP_TAPS],
            q_idx: vec![0; cap + TP_SPAN + 2].into_boxed_slice(),
            q_val: vec![0.0; cap + TP_SPAN + 2].into_boxed_slice(),
            q_head: 0,
            q_len: 0,
            env: 1.0,
            boxbuf: vec![1.0; cap].into_boxed_slice(),
            box_pos: 0,
            box_sum: 0.0,
            n: 0,
            in_gain: 1.0,
            in_target: 1.0,
            in_coef: 0.0,
            min_gain: 1.0,
        };
        l.set_sample_rate(sample_rate);
        l
    }

    pub fn spec(&self) -> LimiterSpec {
        self.spec
    }

    /// Recompute lengths and clear state (no allocation).
    pub fn set_sample_rate(&mut self, sample_rate: u32) {
        let sr = sample_rate.clamp(1, MAX_RATE);
        self.sample_rate = sr;
        let ceil = if self.spec.ceiling_dbtp.is_finite() {
            self.spec.ceiling_dbtp.min(0.0)
        } else {
            -1.0
        };
        self.ceiling = db_to_lin(ceil);
        self.target = db_to_lin(ceil - LIMITER_MARGIN_DB);
        let ms = if self.spec.lookahead_ms.is_finite() {
            self.spec.lookahead_ms.clamp(0.0, MAX_LOOKAHEAD_MS) as f64
        } else {
            0.0
        };
        self.look = ((ms * 1e-3 * sr as f64).round() as usize).clamp(1, self.boxbuf.len());
        self.win = self.look + TP_SPAN;
        self.dlen = self.look + TP_SPAN;
        let rel_ms = if self.spec.release_ms.is_finite() {
            self.spec.release_ms.max(1.0)
        } else {
            50.0
        };
        let rel = (rel_ms as f64 * 1e-3 * sr as f64).max(1.0);
        self.rel_coef = (1.0 - (-1.0 / rel).exp()) as f32;
        self.in_coef = (1.0 - (-1.0 / (0.02 * sr as f64)).exp()) as f32;
        self.reset();
    }

    /// Clear the audio and gain state (silence in, unity gain).
    pub fn reset(&mut self) {
        self.dl[..self.dlen].fill(0.0);
        self.dr[..self.dlen].fill(0.0);
        self.dpos = 0;
        self.hl = [0.0; TP_TAPS];
        self.hr = [0.0; TP_TAPS];
        self.q_head = 0;
        self.q_len = 0;
        self.env = 1.0;
        self.boxbuf[..self.look].fill(1.0);
        self.box_pos = 0;
        self.box_sum = self.look as f64;
        self.n = 0;
        self.in_gain = self.in_target;
    }

    /// Delay from input to output, in samples.
    pub fn latency(&self) -> usize {
        self.dlen
    }

    /// Gain ahead of the limiter (the master bus gain). Ramps over
    /// ~20 ms; `immediate` jumps (start of an offline render).
    pub fn set_input_gain(&mut self, gain: f32, immediate: bool) {
        let g = if gain.is_finite() { gain.max(0.0) } else { 0.0 };
        self.in_target = g;
        if immediate {
            self.in_gain = g;
        }
    }

    /// Lowest gain applied since the last call (1 = never limited).
    pub fn take_min_gain(&mut self) -> f32 {
        std::mem::replace(&mut self.min_gain, 1.0)
    }

    /// Limit one stereo frame (output is `latency()` samples late).
    #[inline]
    pub fn process_frame(&mut self, l: f32, r: f32) -> (f32, f32) {
        self.in_gain += (self.in_target - self.in_gain) * self.in_coef;
        let l = if l.is_finite() { l * self.in_gain } else { 0.0 };
        let r = if r.is_finite() { r * self.in_gain } else { 0.0 };
        let l = if l.is_finite() { l } else { 0.0 };
        let r = if r.is_finite() { r } else { 0.0 };
        // Interpolator history (newest first).
        self.hl.copy_within(0..TP_TAPS - 1, 1);
        self.hr.copy_within(0..TP_TAPS - 1, 1);
        self.hl[0] = l;
        self.hr[0] = r;
        let peak = interval_peak(&self.hl).max(interval_peak(&self.hr));
        let req = if peak > self.target {
            self.target / peak
        } else {
            1.0
        };
        // Sliding minimum over the last `win` requirements.
        let cap = self.q_idx.len();
        while self.q_len > 0 {
            let back = (self.q_head + self.q_len - 1) % cap;
            if self.q_val[back] >= req {
                self.q_len -= 1;
            } else {
                break;
            }
        }
        let slot = (self.q_head + self.q_len) % cap;
        self.q_idx[slot] = self.n;
        self.q_val[slot] = req;
        self.q_len += 1;
        while self.q_idx[self.q_head] + (self.win as u64) <= self.n {
            self.q_head = (self.q_head + 1) % cap;
            self.q_len -= 1;
        }
        let hold = self.q_val[self.q_head];
        // Instant attack, exponential release.
        self.env = if hold < self.env {
            hold
        } else {
            self.env + (hold - self.env) * self.rel_coef
        };
        // Moving average over the look-ahead.
        self.box_sum += self.env as f64 - self.boxbuf[self.box_pos] as f64;
        self.boxbuf[self.box_pos] = self.env;
        self.box_pos += 1;
        if self.box_pos == self.look {
            self.box_pos = 0;
        }
        let g = ((self.box_sum / self.look as f64) as f32).min(1.0);
        self.min_gain = self.min_gain.min(g);
        // Delay line.
        let (ol, or) = (self.dl[self.dpos], self.dr[self.dpos]);
        self.dl[self.dpos] = l;
        self.dr[self.dpos] = r;
        self.dpos += 1;
        if self.dpos == self.dlen {
            self.dpos = 0;
        }
        self.n += 1;
        let c = self.ceiling;
        ((ol * g).clamp(-c, c), (or * g).clamp(-c, c))
    }

    /// Limit a block in place.
    pub fn process(&mut self, l: &mut [f32], r: &mut [f32]) {
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let (x, y) = self.process_frame(*a, *b);
            *a = x;
            *b = y;
        }
    }
}

/// Key level follower: 2 ms RMS of the summed key voices, in dBFS.
#[derive(Debug, Clone, Copy)]
pub struct KeyDetector {
    coef: f32,
    power: f32,
}

/// RMS time constant of the key detector.
pub const KEY_RMS_MS: f32 = 2.0;

impl KeyDetector {
    pub fn new(sample_rate: u32) -> Self {
        let mut k = Self {
            coef: 0.0,
            power: 0.0,
        };
        k.set_sample_rate(sample_rate);
        k
    }

    pub fn set_sample_rate(&mut self, sample_rate: u32) {
        let n = (KEY_RMS_MS as f64 * 1e-3 * sample_rate.max(1) as f64).max(1.0);
        self.coef = (1.0 - (-1.0 / n).exp()) as f32;
    }

    /// Feed one key frame; returns the level in dBFS.
    #[inline]
    pub fn push(&mut self, l: f32, r: f32) -> f32 {
        let lf = if l.is_finite() { l as f64 } else { 0.0 };
        let rf = if r.is_finite() { r as f64 } else { 0.0 };
        let p = (0.5 * (lf * lf + rf * rf)).min(f32::MAX as f64) as f32;
        self.power += (p - self.power) * self.coef;
        if !self.power.is_finite() {
            self.power = f32::MAX;
        }
        self.level_db()
    }

    pub fn level_db(&self) -> f32 {
        10.0 * (self.power + 1e-12).log10()
    }
}

/// Gain envelope of one ducked bus (see [`DuckSpec`]).
#[derive(Debug, Clone, Copy)]
pub struct Ducker {
    spec: DuckSpec,
    att_step: f32,
    rel_step: f32,
    hold_frames: u32,
    hold: u32,
    /// Current drop, dB (>= 0).
    gr: f32,
    /// Deepest drop since the last `take_max_drop`.
    max_gr: f32,
}

impl Ducker {
    pub fn new(spec: DuckSpec, sample_rate: u32) -> Self {
        let mut d = Self {
            spec,
            att_step: 0.0,
            rel_step: 0.0,
            hold_frames: 0,
            hold: 0,
            gr: 0.0,
            max_gr: 0.0,
        };
        d.set_sample_rate(sample_rate);
        d
    }

    pub fn spec(&self) -> DuckSpec {
        self.spec
    }

    pub fn set_sample_rate(&mut self, sample_rate: u32) {
        let fs = sample_rate.max(1) as f32 * 1e-3;
        self.att_step = self.spec.amount_db / (self.spec.attack_ms * fs).max(1.0);
        self.rel_step = self.spec.amount_db / (self.spec.release_ms * fs).max(1.0);
        self.hold_frames = (self.spec.hold_ms * fs).round() as u32;
    }

    /// Advance one frame with the key level (dBFS); returns the bus gain.
    #[inline]
    pub fn step(&mut self, key_db: f32) -> f32 {
        let key = if key_db.is_nan() {
            f32::NEG_INFINITY
        } else {
            key_db
        };
        let target = (key - self.spec.threshold_db).clamp(0.0, self.spec.amount_db);
        if target >= self.gr {
            self.gr = (self.gr + self.att_step).min(target);
            self.hold = self.hold_frames;
        } else if self.hold > 0 {
            self.hold -= 1;
        } else {
            self.gr = (self.gr - self.rel_step).max(target);
        }
        self.max_gr = self.max_gr.max(self.gr);
        if self.gr <= 0.0 {
            1.0
        } else {
            db_to_lin(-self.gr)
        }
    }

    /// Current drop in dB.
    pub fn drop_db(&self) -> f32 {
        self.gr
    }

    pub fn take_max_drop(&mut self) -> f32 {
        std::mem::replace(&mut self.max_gr, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loudness::true_peak;

    fn noise(n: usize, seed: u32, amp: f32) -> Vec<f32> {
        let mut s = seed.max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                amp * ((s as f32 / u32::MAX as f32) * 2.0 - 1.0)
            })
            .collect()
    }

    #[test]
    fn specs_validate_ranges() {
        assert!(LimiterSpec::default().validate().is_ok());
        assert!(DuckSpec::default().validate().is_ok());
        let bad = LimiterSpec {
            ceiling_dbtp: 1.0,
            ..Default::default()
        };
        assert!(bad.validate().unwrap_err().contains("ceiling_dbtp"));
        let bad = DuckSpec {
            amount_db: 0.0,
            ..Default::default()
        };
        assert!(bad.validate().unwrap_err().contains("amount_db"));
        let j: LimiterSpec = serde_json::from_str("{}").unwrap();
        assert_eq!(j, LimiterSpec::default());
        assert!(serde_json::from_str::<DuckSpec>(r#"{"bogus":1}"#).is_err());
    }

    #[test]
    fn limiter_holds_true_peak_under_ceiling_and_passes_quiet_audio() {
        let sr = 48_000;
        let mut lim = Limiter::new(LimiterSpec::default(), sr);
        let lat = lim.latency();
        // Quiet: 0.25 peak tone passes unchanged, only delayed.
        let tone: Vec<f32> = (0..4800).map(|i| 0.25 * (i as f32 * 0.05).sin()).collect();
        let (mut l, mut r) = (tone.clone(), tone.clone());
        lim.process(&mut l, &mut r);
        for i in lat..tone.len() {
            assert!((l[i] - tone[i - lat]).abs() < 1e-6, "sample {i}");
        }
        // Loud: +12 dB noise bursts and a near-Nyquist square wave.
        lim.reset();
        let mut x = noise(48_000, 3, 4.0);
        for (i, v) in x.iter_mut().enumerate().skip(24_000) {
            *v = if (i / 2) % 2 == 0 { 3.0 } else { -3.0 };
        }
        let (mut l, mut r) = (x.clone(), x.clone());
        lim.process(&mut l, &mut r);
        let ceiling = db_to_lin(-1.0);
        let inter: Vec<f32> = l.iter().zip(&r).flat_map(|(a, b)| [*a, *b]).collect();
        let tp = true_peak(&inter, 2);
        assert!(tp <= ceiling, "true peak {tp} over ceiling {ceiling}");
        assert!(lim.take_min_gain() < 0.5);
    }

    #[test]
    fn ducker_follows_key_energy_not_a_timer() {
        let sr = 48_000;
        let spec = DuckSpec::default();
        let run = |key_secs: f32| {
            let mut k = KeyDetector::new(sr);
            let mut d = Ducker::new(spec, sr);
            let mut deepest = 0.0f32;
            let n = (2.0 * sr as f32) as usize;
            let key_n = (key_secs * sr as f32) as usize;
            let src = noise(n, 9, 0.3);
            for (i, &s) in src.iter().enumerate() {
                let v = if i < key_n { s } else { 0.0 };
                d.step(k.push(v, v));
                deepest = deepest.max(d.drop_db());
            }
            (deepest, d.drop_db())
        };
        let (long, long_end) = run(1.0);
        let (tick, _) = run(0.005);
        assert!(long > spec.amount_db - 0.01, "long key drop {long}");
        assert!(tick < spec.amount_db / 3.0, "tick drop {tick}");
        assert_eq!(long_end, 0.0, "recovered by the end");
    }

    #[test]
    fn sanitizes_nonfinite_inputs_and_recovers() {
        let spec = LimiterSpec {
            lookahead_ms: f32::NAN,
            ..Default::default()
        };
        let mut lim = Limiter::new(spec, 48_000);
        let _ = lim.process_frame(f32::NAN, f32::INFINITY);
        for _ in 0..(lim.latency() + 16) {
            let (a, b) = lim.process_frame(0.1, 0.1);
            assert!(
                a.is_finite() && b.is_finite(),
                "limiter poisoned: ({a}, {b})"
            );
        }
        assert!(lim.take_min_gain().is_finite());

        let mut kd = KeyDetector::new(48_000);
        assert!(kd.push(1e20, 1e20).is_finite());
        assert!(kd.push(f32::NAN, 0.0).is_finite());
        assert!(kd.push(0.1, 0.1).is_finite());

        let mut d = Ducker::new(DuckSpec::default(), 48_000);
        assert!(d.step(f32::NAN).is_finite());
        assert!(d.step(-10.0).is_finite());
    }
}
