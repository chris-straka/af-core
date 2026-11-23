//! Deterministic software mixer: voice pool + bus graph.
//!
//! This is the offline/headless render path (Kira owns the realtime
//! device path; see `crate::kira_bridge`).
//!
//! Rules (frozen):
//! - 64 voices, 8 buses. Bus 0 is the master: only bus 0 reaches output.
//! - Voices borrow shared mono PCM (`Arc<[f32]>`); the mixer never copies
//!   source data at schedule time beyond the `Arc` clone (no alloc/logic
//!   inside `render_block` itself).
//! - Pan law (frozen): linear constant-unity. `pan = 0` => unity both;
//!   `-1` => left only; `+1` => right only.
//! - `render_block` performs no allocation (member scratch only).
//! - P2 (spatial): a voice's gain/pan can be retargeted per block
//!   ([`Mixer::set_voice_targets`]); the change ramps linearly across the
//!   next block (no zipper). Optional one-pole low-pass (occlusion) and a
//!   mono send to a reverb bus. Voices that use none of these run the
//!   exact P0 arithmetic, so P0/P1 renders stay bit-identical.

use std::sync::Arc;

use crate::binaural::BinauralState;
use crate::dynamics::{DuckSpec, Ducker, KeyDetector, Limiter, LimiterSpec};
use crate::error::{AfError, Result};
use crate::reverb::{Reverb, ReverbParams};

pub const MAX_VOICES: usize = 64;
pub const MAX_BUSES: usize = 8;
/// Largest single render_block call (member scratch is sized by this).
pub const MAX_BLOCK: usize = 4096;

fn valid_gain(g: f32) -> bool {
    g.is_finite() && g >= 0.0
}

/// Placement parameters for one voice instance.
#[derive(Debug, Clone)]
pub struct VoiceParams {
    /// Destination bus index, < MAX_BUSES.
    pub bus: u32,
    /// Linear gain, finite, >= 0 (may exceed 1 for headroom).
    pub gain: f32,
    /// -1 (hard left) .. +1 (hard right).
    pub pan: f32,
    /// Absolute mix frame of the first sample.
    pub start_frame: u64,
    /// Playback rate (pitch randomization), finite and > 0. Linear
    /// interpolation; 1.0 = original pitch.
    pub rate: f64,
    /// When `Some(n)`, the first `n` source frames loop with wraparound.
    pub loop_frames: Option<usize>,
    /// One-pole low-pass coefficient in (0, 1]; 1.0 bypasses the filter.
    pub lowpass: f32,
    /// Optional send (post-filter, pre-pan, scaled by voice gain) into a
    /// reverb bus.
    pub send_bus: Option<u32>,
    /// Linear send level, finite, >= 0.
    pub send_gain: f32,
    /// Binaural direction (azimuth, elevation, radians, listener space):
    /// render through the HRTF instead of the pan law. `None` pans.
    pub binaural: Option<(f32, f32)>,
    /// Feeds the ducking key (an event marked `sidechain`): buses with a
    /// ducker drop while key voices sound. Audio is unchanged.
    pub sidechain: bool,
}

impl Default for VoiceParams {
    fn default() -> Self {
        Self {
            bus: 0,
            gain: 1.0,
            pan: 0.0,
            start_frame: 0,
            rate: 1.0,
            loop_frames: None,
            lowpass: 1.0,
            send_bus: None,
            send_gain: 0.0,
            binaural: None,
            sidechain: false,
        }
    }
}

#[derive(Debug, Clone)]
struct Voice {
    active: bool,
    data: Arc<[f32]>,
    start: u64,
    bus: u32,
    gain: f32,
    pan: f32,
    rate: f64,
    loop_frames: Option<usize>,
    target_gain: f32,
    target_pan: f32,
    lowpass: f32,
    lp_state: f32,
    /// -1 = no send.
    send_bus: i32,
    send_gain: f32,
    target_send: f32,
    /// Binaural target (azimuth, elevation): HRTF instead of pan law.
    t_binaural: Option<(f32, f32)>,
    /// Binaural state (allocated at voice start, control thread only).
    bin_state: Option<BinauralState>,
    /// Adds its output to the ducking key.
    sidechain: bool,
}

impl Default for Voice {
    fn default() -> Self {
        Self {
            active: false,
            data: Arc::from([]),
            start: 0,
            bus: 0,
            gain: 1.0,
            pan: 0.0,
            rate: 1.0,
            loop_frames: None,
            target_gain: 1.0,
            target_pan: 0.0,
            lowpass: 1.0,
            lp_state: 0.0,
            send_bus: -1,
            send_gain: 0.0,
            target_send: 0.0,
            t_binaural: None,
            bin_state: None,
            sidechain: false,
        }
    }
}

/// Linear constant-unity pan law (frozen).
#[inline]
fn pan_gains(gain: f32, pan: f32) -> (f32, f32) {
    if pan <= 0.0 {
        (gain, gain * (1.0 + pan))
    } else {
        (gain * (1.0 - pan), gain)
    }
}

#[derive(Debug, Clone, Copy)]
struct Bus {
    gain: f32,
    /// Parent bus index, or -1. Bus 0 is pinned to -1 (master out).
    /// A non-master bus with parent -1 is a dead send (discarded).
    parent: i32,
}

impl Default for Bus {
    fn default() -> Self {
        Self {
            gain: 1.0,
            parent: 0,
        }
    }
}

#[inline]
fn add_finite(dst: &mut f32, delta: f32) {
    let sum = *dst + delta;
    if sum.is_finite() {
        *dst = sum;
    } else {
        let cur = if dst.is_finite() { *dst as f64 } else { 0.0 };
        let d = if delta.is_finite() {
            delta as f64
        } else if delta.is_nan() {
            0.0
        } else {
            delta.signum() as f64 * f32::MAX as f64
        };
        *dst = (cur + d).clamp(-f32::MAX as f64, f32::MAX as f64) as f32;
    }
}

#[inline]
fn scale_finite(val: f32, gain: f32) -> f32 {
    let prod = val * gain;
    if prod.is_finite() {
        prod
    } else if val.is_nan() || gain.is_nan() {
        0.0
    } else {
        (val as f64 * gain as f64).clamp(-f32::MAX as f64, f32::MAX as f64) as f32
    }
}

fn fold_bus_channel(acc: &mut [f32], child: usize, parent: usize, n: usize, gain: f32) {
    let c_off = child * MAX_BLOCK;
    let p_off = parent * MAX_BLOCK;
    for i in 0..n {
        let delta = scale_finite(acc[c_off + i], gain);
        add_finite(&mut acc[p_off + i], delta);
    }
}

#[inline]
fn sample_voice_frame(data: &[f32], pos: f64, is_loop: bool, src_len: usize) -> Option<f32> {
    if !pos.is_finite() || pos < 0.0 || data.is_empty() {
        return None;
    }
    let clean = |v: f32| if v.is_finite() { v } else { 0.0 };
    if is_loop {
        if src_len == 0 {
            return None;
        }
        let wrapped = if pos >= (usize::MAX / 2) as f64 {
            pos % src_len as f64
        } else {
            pos
        };
        let idx = (wrapped.floor() as usize) % src_len;
        let frac = (wrapped - wrapped.floor()) as f32;
        let s0 = clean(data[idx]);
        let s1 = clean(data[(idx + 1) % src_len]);
        let diff = s1 - s0;
        Some(if diff.is_finite() {
            s0 + diff * frac
        } else {
            (s0 as f64 + (s1 as f64 - s0 as f64) * frac as f64) as f32
        })
    } else {
        if pos >= data.len() as f64 {
            return None;
        }
        let idx = pos.floor() as usize;
        let frac = (pos - idx as f64) as f32;
        let s0 = clean(data[idx]);
        let s1 = data.get(idx + 1).copied().map(clean).unwrap_or(0.0);
        let diff = s1 - s0;
        Some(if diff.is_finite() {
            s0 + diff * frac
        } else {
            (s0 as f64 + (s1 as f64 - s0 as f64) * frac as f64) as f32
        })
    }
}

pub struct Mixer {
    voices: [Voice; MAX_VOICES],
    buses: [Bus; MAX_BUSES],
    /// Optional 100%-wet reverb insert per bus (zone returns).
    reverbs: [Option<Reverb>; MAX_BUSES],
    acc_l: Box<[f32]>,
    acc_r: Box<[f32]>,
    /// Mono scratch for the binaural path (control: never reallocates).
    scratch: Box<[f32]>,
    /// Per-bus post-gain stats (do not affect audio; for `af render` reports).
    stat_peak: [f32; MAX_BUSES],
    stat_sum_sq: [f64; MAX_BUSES],
    stat_frames: u64,
    /// Sample rate for binaural HRTF state (control thread only).
    sample_rate: u32,
    /// Ducking key: summed output of sidechain voices (pre bus gain).
    key_l: Box<[f32]>,
    key_r: Box<[f32]>,
    /// Key level per frame (dBFS), then per-bus duck gain scratch.
    key_db: Box<[f32]>,
    duck_gain: Box<[f32]>,
    /// Sidechain voices' signal at the master summing junction (telemetry
    /// only; its own buffers, so ducking never shares scratch with it).
    heard_key_l: Box<[f32]>,
    heard_key_r: Box<[f32]>,
    key: KeyDetector,
    /// Optional ducker per bus (never on the master).
    duckers: [Option<Ducker>; MAX_BUSES],
    /// Optional master limiter (after the master gain).
    limiter: Option<Box<Limiter>>,
    /// Per-bus post-gain energy of the last rendered block (L+R squares).
    block_sum_sq: [f64; MAX_BUSES],
    /// Sidechain voices' energy at the master summing junction in the last block.
    block_heard_key_sum_sq: f64,
}

/// Post-gain peak/RMS snapshot for one bus.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BusStats {
    pub peak: f32,
    pub rms: f32,
}

impl Default for Mixer {
    fn default() -> Self {
        let mut buses = [Bus::default(); MAX_BUSES];
        buses[0].parent = -1;
        Self {
            voices: std::array::from_fn(|_| Voice::default()),
            buses,
            reverbs: std::array::from_fn(|_| None),
            acc_l: vec![0.0; MAX_BUSES * MAX_BLOCK].into_boxed_slice(),
            acc_r: vec![0.0; MAX_BUSES * MAX_BLOCK].into_boxed_slice(),
            scratch: vec![0.0; MAX_BLOCK].into_boxed_slice(),
            stat_peak: [0.0; MAX_BUSES],
            stat_sum_sq: [0.0; MAX_BUSES],
            stat_frames: 0,
            sample_rate: 48_000,
            key_l: vec![0.0; MAX_BLOCK].into_boxed_slice(),
            key_r: vec![0.0; MAX_BLOCK].into_boxed_slice(),
            key_db: vec![0.0; MAX_BLOCK].into_boxed_slice(),
            duck_gain: vec![0.0; MAX_BLOCK].into_boxed_slice(),
            heard_key_l: vec![0.0; MAX_BLOCK].into_boxed_slice(),
            heard_key_r: vec![0.0; MAX_BLOCK].into_boxed_slice(),
            key: KeyDetector::new(48_000),
            duckers: [None; MAX_BUSES],
            limiter: None,
            block_sum_sq: [0.0; MAX_BUSES],
            block_heard_key_sum_sq: 0.0,
        }
    }
}

impl Mixer {
    /// Sample rate the binaural HRTF renders at (control thread only).
    pub fn set_sample_rate(&mut self, sample_rate: u32) {
        self.sample_rate = sample_rate.max(1);
        self.key.set_sample_rate(self.sample_rate);
        for d in self.duckers.iter_mut().flatten() {
            d.set_sample_rate(self.sample_rate);
        }
        if let Some(l) = self.limiter.as_mut() {
            l.set_sample_rate(self.sample_rate);
        }
    }

    /// Put a look-ahead limiter after the master gain (`None` removes
    /// it). Control thread: allocates. The output is then
    /// [`Mixer::latency`] frames late.
    pub fn set_master_limiter(&mut self, spec: Option<LimiterSpec>) -> Result<()> {
        if let Some(s) = &spec {
            s.validate().map_err(AfError::BadArg)?;
        }
        self.limiter = spec.map(|s| Box::new(Limiter::new(s, self.sample_rate)));
        Ok(())
    }

    /// Whether a master limiter is installed.
    pub fn has_limiter(&self) -> bool {
        self.limiter.is_some()
    }

    /// Ceiling of the installed master limiter (dBTP), if any.
    pub fn limiter_ceiling_dbtp(&self) -> Option<f32> {
        self.limiter.as_ref().map(|l| l.spec().ceiling_dbtp)
    }

    /// Output delay added by the master limiter, in frames.
    pub fn latency(&self) -> usize {
        self.limiter.as_ref().map(|l| l.latency()).unwrap_or(0)
    }

    /// Lowest master-limiter gain since the last call (1 = untouched).
    pub fn take_limiter_min_gain(&mut self) -> f32 {
        self.limiter
            .as_mut()
            .map(|l| l.take_min_gain())
            .unwrap_or(1.0)
    }

    /// Duck `bus` while sidechain voices sound (`None` removes it).
    pub fn set_bus_duck(&mut self, bus: u32, spec: Option<DuckSpec>) -> Result<()> {
        self.bus_ref(bus)?;
        if bus == 0 && spec.is_some() {
            return Err(AfError::BadBus(
                "master bus cannot be ducked (duck its children)".to_string(),
            ));
        }
        if let Some(s) = &spec {
            s.validate().map_err(AfError::BadArg)?;
        }
        self.duckers[bus as usize] = spec.map(|s| Ducker::new(s, self.sample_rate));
        Ok(())
    }

    /// Current duck drop of `bus` in dB (0 when not ducked).
    pub fn bus_duck_db(&self, bus: u32) -> f32 {
        self.duckers
            .get(bus as usize)
            .and_then(|d| d.as_ref())
            .map(|d| d.drop_db())
            .unwrap_or(0.0)
    }

    /// Deepest duck drop of `bus` (dB) since the last call.
    pub fn take_bus_duck_max_db(&mut self, bus: u32) -> f32 {
        self.duckers
            .get_mut(bus as usize)
            .and_then(|d| d.as_mut())
            .map(|d| d.take_max_drop())
            .unwrap_or(0.0)
    }

    /// Post-gain energy (sum of L and R squares) of `bus` in the last
    /// rendered block.
    pub fn last_block_energy(&self, bus: u32) -> f64 {
        self.block_sum_sq.get(bus as usize).copied().unwrap_or(0.0)
    }

    /// Coherent stereo energy (sum of L and R squares) of all `sidechain`
    /// voices at the master summing junction (pre-master-gain) in the last
    /// rendered block.
    pub fn last_block_heard_key_energy(&self) -> f64 {
        self.block_heard_key_sum_sq
    }

    /// Post-duck, post-gain energy of `bus` scaled up to the master summing
    /// junction (pre-master-gain) in the last rendered block.
    pub fn last_block_bus_master_energy(&self, bus: u32) -> f64 {
        let b = bus as usize;
        if b == 0 || b >= MAX_BUSES {
            return self.last_block_energy(bus);
        }
        let parent = self.buses[b].parent;
        if parent < 0 {
            return 0.0;
        }
        let pg = self.chain_gain_to_master_f64(parent as usize);
        let energy = self.block_sum_sq[b] * pg * pg;
        if energy.is_finite() {
            energy
        } else {
            f64::MAX
        }
    }

    fn chain_gain_to_master_f64(&self, mut bus: usize) -> f64 {
        let mut g = 1.0f64;
        let mut hops = 0usize;
        while bus > 0 && hops < MAX_BUSES {
            let b = &self.buses[bus];
            if b.parent < 0 {
                return 0.0;
            }
            g *= b.gain as f64;
            bus = b.parent as usize;
            hops += 1;
        }
        if bus == 0 {
            g
        } else {
            0.0
        }
    }

    /// Clear per-bus stats (call once before a render run).
    pub fn reset_stats(&mut self) {
        self.stat_peak = [0.0; MAX_BUSES];
        self.stat_sum_sq = [0.0; MAX_BUSES];
        self.stat_frames = 0;
    }

    /// Post-gain peak/RMS per bus since the last [`Mixer::reset_stats`].
    pub fn bus_stats(&self, bus: u32) -> Result<BusStats> {
        self.bus_ref(bus)?;
        let b = bus as usize;
        let rms = if self.stat_frames == 0 {
            0.0
        } else {
            (self.stat_sum_sq[b] / (2.0 * self.stat_frames as f64))
                .sqrt()
                .min(f32::MAX as f64) as f32
        };
        Ok(BusStats {
            peak: self.stat_peak[b],
            rms,
        })
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_bus_gain(&mut self, bus: u32, gain: f32) -> Result<()> {
        let b = self.bus_mut(bus)?;
        if !valid_gain(gain) {
            return Err(AfError::BadGain(format!(
                "bus {bus} gain must be finite and >= 0, got {gain}"
            )));
        }
        b.gain = gain;
        Ok(())
    }

    pub fn set_bus_parent(&mut self, bus: u32, parent: i32) -> Result<()> {
        if bus as usize >= MAX_BUSES {
            return Err(AfError::BadBus(format!(
                "bus index {bus} out of range (max {})",
                MAX_BUSES - 1
            )));
        }
        if bus == 0 {
            return Err(AfError::BadBus(
                "bus 0 (master) parent is pinned to -1".to_string(),
            ));
        }
        if parent < -1 || parent >= MAX_BUSES as i32 {
            return Err(AfError::BadBus(format!(
                "parent {parent} out of range [-1, {})",
                MAX_BUSES - 1
            )));
        }
        if parent == bus as i32 {
            return Err(AfError::BadBus(format!("bus {bus} cannot parent itself")));
        }
        // Cycle check: walk up from the proposed parent.
        let mut p = parent;
        while p != -1 {
            if p == bus as i32 {
                return Err(AfError::BadBus(format!(
                    "parenting bus {bus} to {parent} would create a cycle"
                )));
            }
            p = self.buses[p as usize].parent;
        }
        self.buses[bus as usize].parent = parent;
        Ok(())
    }

    pub fn bus_gain(&self, bus: u32) -> Result<f32> {
        Ok(self.bus_ref(bus)?.gain)
    }

    /// Start a voice on shared mono PCM. Returns the voice slot index.
    pub fn start_voice(&mut self, data: Arc<[f32]>, p: &VoiceParams) -> Result<usize> {
        if p.bus as usize >= MAX_BUSES {
            return Err(AfError::BadBus(format!(
                "bus index {} out of range (max {})",
                p.bus,
                MAX_BUSES - 1
            )));
        }
        if !valid_gain(p.gain) {
            return Err(AfError::BadGain(format!(
                "voice gain must be finite and >= 0, got {}",
                p.gain
            )));
        }
        if !p.pan.is_finite() || p.pan < -1.0 || p.pan > 1.0 {
            return Err(AfError::BadPan(format!(
                "voice pan must be in [-1, 1], got {}",
                p.pan
            )));
        }
        if !p.rate.is_finite() || p.rate <= 0.0 {
            return Err(AfError::BadArg(format!(
                "playback rate must be finite and > 0, got {}",
                p.rate
            )));
        }
        if let Some(n) = p.loop_frames {
            if n == 0 || n > data.len() {
                return Err(AfError::BadArg(format!(
                    "loop_frames {n} out of range for buffer of {} frames",
                    data.len()
                )));
            }
        }
        if !p.lowpass.is_finite() || p.lowpass <= 0.0 || p.lowpass > 1.0 {
            return Err(AfError::BadArg(format!(
                "lowpass coefficient must be in (0, 1], got {}",
                p.lowpass
            )));
        }
        if !valid_gain(p.send_gain) {
            return Err(AfError::BadGain(format!(
                "send gain must be finite and >= 0, got {}",
                p.send_gain
            )));
        }
        if let Some(sb) = p.send_bus {
            if sb as usize >= MAX_BUSES {
                return Err(AfError::BadBus(format!(
                    "send bus index {sb} out of range (max {})",
                    MAX_BUSES - 1
                )));
            }
        }
        if let Some((az, el)) = p.binaural {
            if !az.is_finite() || !el.is_finite() {
                return Err(AfError::BadArg(format!(
                    "binaural direction must be finite, got ({az}, {el})"
                )));
            }
        }
        if data.is_empty() {
            // Zero-length voices never activate (contract §1); still ok.
            return Err(AfError::BadArg(
                "cannot start a voice on empty PCM".to_string(),
            ));
        }
        let slot = self
            .voices
            .iter()
            .position(|v| !v.active)
            .ok_or(AfError::NoFreeVoice)?;
        self.voices[slot] = Voice {
            active: true,
            data,
            start: p.start_frame,
            bus: p.bus,
            gain: p.gain,
            pan: p.pan,
            rate: p.rate,
            loop_frames: p.loop_frames,
            target_gain: p.gain,
            target_pan: p.pan,
            lowpass: p.lowpass,
            lp_state: 0.0,
            send_bus: p.send_bus.map(|b| b as i32).unwrap_or(-1),
            send_gain: p.send_gain,
            target_send: p.send_gain,
            t_binaural: p.binaural,
            // Control thread only (never inside render_block).
            bin_state: p.binaural.map(|_| BinauralState::new(self.sample_rate)),
            sidechain: p.sidechain,
        };
        Ok(slot)
    }

    /// Retarget a live voice (spatial update). Gain, pan and send ramp
    /// linearly over the next rendered block; the low-pass coefficient
    /// applies immediately. Inactive/unknown slots are a silent no-op.
    pub fn set_voice_targets(
        &mut self,
        slot: usize,
        gain: f32,
        pan: f32,
        lowpass: f32,
        send_gain: f32,
        binaural: Option<(f32, f32)>,
    ) -> Result<()> {
        if !valid_gain(gain) || !valid_gain(send_gain) {
            return Err(AfError::BadGain(format!(
                "voice gain/send must be finite and >= 0, got {gain}/{send_gain}"
            )));
        }
        if !pan.is_finite() || !(-1.0..=1.0).contains(&pan) {
            return Err(AfError::BadPan(format!(
                "voice pan must be in [-1, 1], got {pan}"
            )));
        }
        if let Some(b) = binaural {
            let (az, el) = b;
            if !az.is_finite() || !el.is_finite() {
                return Err(AfError::BadArg(format!(
                    "binaural direction must be finite, got ({az}, {el})"
                )));
            }
        }
        if !lowpass.is_finite() || lowpass <= 0.0 || lowpass > 1.0 {
            return Err(AfError::BadArg(format!(
                "lowpass coefficient must be in (0, 1], got {lowpass}"
            )));
        }
        if let Some(v) = self.voices.get_mut(slot).filter(|v| v.active) {
            v.target_gain = gain;
            v.target_pan = pan;
            v.lowpass = lowpass;
            v.target_send = send_gain;
            v.t_binaural = binaural;
            if binaural.is_some() && v.bin_state.is_none() {
                // Control thread only (never inside render_block).
                v.bin_state = Some(BinauralState::new(self.sample_rate));
            }
        }
        Ok(())
    }

    /// Re-route a live voice's send (listener moved into another reverb
    /// zone). Takes effect at the next block; `None` disables the send.
    pub fn set_voice_send_bus(&mut self, slot: usize, bus: Option<u32>) -> Result<()> {
        if let Some(b) = bus {
            self.bus_ref(b)?;
        }
        if let Some(v) = self.voices.get_mut(slot).filter(|v| v.active) {
            v.send_bus = bus.map(|b| b as i32).unwrap_or(-1);
        }
        Ok(())
    }

    /// Install (or clear with `None`) a 100%-wet reverb on a bus. The
    /// bus's accumulated input (voice sends + routed children) is replaced
    /// by the reverb tail before the bus gain/fold. Allocates (control
    /// thread only).
    pub fn set_bus_reverb(
        &mut self,
        bus: u32,
        params: Option<ReverbParams>,
        sample_rate: u32,
    ) -> Result<()> {
        self.bus_ref(bus)?;
        if bus == 0 && params.is_some() {
            return Err(AfError::BadBus(
                "master bus cannot host a reverb insert".to_string(),
            ));
        }
        self.reverbs[bus as usize] = match params {
            Some(p) => Some(Reverb::new(p, sample_rate)?),
            None => None,
        };
        Ok(())
    }

    pub fn active_voices(&self) -> usize {
        self.voices.iter().filter(|v| v.active).count()
    }

    /// Deactivate a voice immediately (priority steal, loop stop).
    /// Unknown or already-inactive slots are a silent no-op.
    pub fn stop_voice(&mut self, slot: usize) {
        if let Some(v) = self.voices.get_mut(slot) {
            v.active = false;
        }
    }

    /// Absolute start frame of an active voice, else `None`.
    pub fn voice_start(&self, slot: usize) -> Option<u64> {
        self.voices.get(slot).filter(|v| v.active).map(|v| v.start)
    }

    fn bus_fold_order(&self) -> [usize; MAX_BUSES] {
        let mut child_count = [0u8; MAX_BUSES];
        for bus in self.buses.iter().skip(1) {
            let p = bus.parent;
            if p >= 0 && (p as usize) < MAX_BUSES {
                child_count[p as usize] += 1;
            }
        }
        let mut done = [false; MAX_BUSES];
        let mut order = [0usize; MAX_BUSES];
        for slot in &mut order {
            // Highest-index ready bus first keeps standard 1..7->0 in 7..=0 order.
            let next = (0..MAX_BUSES)
                .rev()
                .find(|&b| !done[b] && child_count[b] == 0)
                .unwrap_or(0);
            done[next] = true;
            let p = self.buses[next].parent;
            if p >= 0 && (p as usize) < MAX_BUSES {
                child_count[p as usize] = child_count[p as usize].saturating_sub(1);
            }
            *slot = next;
        }
        order
    }

    /// Render `out.len()` frames starting at absolute frame `start_frame`.
    /// Deterministic: same voices/params/frames => bit-identical floats.
    /// No allocation (member scratch only). `out_l`/`out_r` must match len.
    pub fn render_block(&mut self, out_l: &mut [f32], out_r: &mut [f32], start_frame: u64) {
        assert_eq!(out_l.len(), out_r.len());
        let n = out_l.len();
        assert!(n <= MAX_BLOCK, "block of {n} exceeds MAX_BLOCK {MAX_BLOCK}");
        if n == 0 {
            return;
        }
        for b in 0..MAX_BUSES {
            let off = b * MAX_BLOCK;
            self.acc_l[off..off + n].fill(0.0);
            self.acc_r[off..off + n].fill(0.0);
        }
        self.block_sum_sq = [0.0; MAX_BUSES];
        let ducking = self.duckers.iter().any(Option::is_some);
        let sidechain_active = self.voices.iter().any(|v| v.active && v.sidechain);
        let mut bus_chain_gains = [0.0f32; MAX_BUSES];
        if sidechain_active {
            bus_chain_gains[0] = 1.0;
            for (b, slot) in bus_chain_gains.iter_mut().enumerate().skip(1) {
                *slot = self.chain_gain_to_master_f64(b).min(f32::MAX as f64) as f32;
            }
            self.heard_key_l[..n].fill(0.0);
            self.heard_key_r[..n].fill(0.0);
        }
        if ducking {
            self.key_l[..n].fill(0.0);
            self.key_r[..n].fill(0.0);
        }
        for v in self.voices.iter_mut() {
            if !v.active {
                continue;
            }
            let src_len = v.loop_frames.unwrap_or(v.data.len()) as f64;
            // Output length of a one-shot at this rate (loops run forever).
            let is_loop = v.loop_frames.is_some();
            let out_len = match v.loop_frames {
                Some(_) => u64::MAX,
                None => (v.data.len() as f64 / v.rate).ceil() as u64,
            };
            let end = v.start.saturating_add(out_len);
            if start_frame >= end || start_frame.saturating_add(n as u64) <= v.start {
                if start_frame >= end {
                    v.active = false;
                }
                continue;
            }
            // Binaural path: mono render with gain ramp, then the HRTF at
            // the block's direction. Sends are not supported here (the
            // HRTF voices are point sources; verbs stay mono in).
            if v.t_binaural.is_some() {
                let (az, el) = v.t_binaural.unwrap_or((0.0, 0.0));
                let bus = v.bus as usize;
                let bus_off = bus * MAX_BLOCK;
                let rate = v.rate;
                let filt = v.lowpass < 1.0;
                let inv_n = 1.0 / n as f32;
                let scratch = &mut self.scratch[..n];
                for (i, slot) in scratch.iter_mut().enumerate() {
                    let f = start_frame.saturating_add(i as u64);
                    if f < v.start || f >= end {
                        *slot = 0.0;
                        continue;
                    }
                    let pos = (f - v.start) as f64 * rate;
                    let Some(mut src) = sample_voice_frame(&v.data, pos, is_loop, src_len as usize)
                    else {
                        *slot = 0.0;
                        continue;
                    };
                    if filt {
                        let next =
                            v.lp_state as f64 + v.lowpass as f64 * (src as f64 - v.lp_state as f64);
                        v.lp_state = if !next.is_finite() || next.abs() < 1e-30 {
                            0.0
                        } else {
                            next.clamp(-f32::MAX as f64, f32::MAX as f64) as f32
                        };
                        src = v.lp_state;
                    }
                    let t = (i + 1) as f32 * inv_n;
                    let g = v.gain + (v.target_gain - v.gain) * t;
                    *slot = scale_finite(src, g);
                }
                v.gain = v.target_gain;
                v.target_pan = v.pan;
                let st = v.bin_state.as_mut().expect("binaural voice without state");
                st.process_into(
                    &self.scratch[..n],
                    az,
                    el,
                    &mut self.acc_l[bus_off..bus_off + n],
                    &mut self.acc_r[bus_off..bus_off + n],
                );
                if v.sidechain {
                    let cg = bus_chain_gains[bus];
                    for (i, &x) in self.scratch[..n].iter().enumerate() {
                        let scaled = scale_finite(x, cg);
                        add_finite(&mut self.heard_key_l[i], scaled);
                        add_finite(&mut self.heard_key_r[i], scaled);
                        if ducking {
                            // Key from the mono voice (pre-HRTF): its level, not
                            // its direction, drives the ducking.
                            add_finite(&mut self.key_l[i], x);
                            add_finite(&mut self.key_r[i], x);
                        }
                    }
                }
                if !is_loop && start_frame.saturating_add(n as u64) >= end {
                    v.active = false;
                }
                continue;
            }
            // Linear constant-unity pan law (frozen).
            let (gl, gr) = pan_gains(v.gain, v.pan);
            let (gl1, gr1) = pan_gains(v.target_gain, v.target_pan);
            let ramp = gl1 != gl || gr1 != gr || v.target_send != v.send_gain;
            let filt = v.lowpass < 1.0;
            let send = v.send_bus >= 0 && (v.send_gain > 0.0 || v.target_send > 0.0);
            let bus = v.bus as usize;
            let bus_off = bus * MAX_BLOCK;
            let rate = v.rate;
            let key = v.sidechain;
            if ramp || filt || send || key {
                // Spatial path (P2): per-sample ramp, filter, send.
                let inv_n = 1.0 / n as f32;
                let sbus_off = v.send_bus.max(0) as usize * MAX_BLOCK;
                let cg = bus_chain_gains[bus];
                for i in 0..n {
                    let f = start_frame.saturating_add(i as u64);
                    if f < v.start || f >= end {
                        continue;
                    }
                    let pos = (f - v.start) as f64 * rate;
                    let Some(mut src) = sample_voice_frame(&v.data, pos, is_loop, src_len as usize)
                    else {
                        continue;
                    };
                    if filt {
                        let next =
                            v.lp_state as f64 + v.lowpass as f64 * (src as f64 - v.lp_state as f64);
                        v.lp_state = if !next.is_finite() || next.abs() < 1e-30 {
                            0.0
                        } else {
                            next.clamp(-f32::MAX as f64, f32::MAX as f64) as f32
                        };
                        src = v.lp_state;
                    }
                    let t = (i + 1) as f32 * inv_n;
                    let (l, r) = if ramp {
                        (gl + (gl1 - gl) * t, gr + (gr1 - gr) * t)
                    } else {
                        (gl, gr)
                    };
                    let vl = scale_finite(src, l);
                    let vr = scale_finite(src, r);
                    add_finite(&mut self.acc_l[bus_off + i], vl);
                    add_finite(&mut self.acc_r[bus_off + i], vr);
                    if key {
                        add_finite(&mut self.heard_key_l[i], scale_finite(vl, cg));
                        add_finite(&mut self.heard_key_r[i], scale_finite(vr, cg));
                        if ducking {
                            add_finite(&mut self.key_l[i], vl);
                            add_finite(&mut self.key_r[i], vr);
                        }
                    }
                    if send {
                        let g = v.gain + (v.target_gain - v.gain) * t;
                        let sg = v.send_gain + (v.target_send - v.send_gain) * t;
                        let x = scale_finite(scale_finite(src, g), sg);
                        add_finite(&mut self.acc_l[sbus_off + i], x);
                        add_finite(&mut self.acc_r[sbus_off + i], x);
                    }
                }
                v.gain = v.target_gain;
                v.pan = v.target_pan;
                v.send_gain = v.target_send;
                if !is_loop && start_frame.saturating_add(n as u64) >= end {
                    v.active = false;
                }
                continue;
            }
            for i in 0..n {
                let f = start_frame.saturating_add(i as u64);
                if f < v.start || f >= end {
                    continue;
                }
                let pos = (f - v.start) as f64 * rate;
                let Some(src) = sample_voice_frame(&v.data, pos, is_loop, src_len as usize) else {
                    continue;
                };
                add_finite(&mut self.acc_l[bus_off + i], scale_finite(src, gl));
                add_finite(&mut self.acc_r[bus_off + i], scale_finite(src, gr));
            }
            if !is_loop && start_frame.saturating_add(n as u64) >= end {
                v.active = false;
            }
        }
        self.block_heard_key_sum_sq = if sidechain_active {
            let mut sum = 0.0f64;
            for i in 0..n {
                let l = self.heard_key_l[i] as f64;
                let r = self.heard_key_r[i] as f64;
                sum += l * l + r * r;
            }
            sum
        } else {
            0.0
        };
        if ducking {
            for i in 0..n {
                self.key_db[i] = self.key.push(self.key_l[i], self.key_r[i]);
            }
        }
        // Fold children into parents in topological order, scaled by own gain.
        for b in self.bus_fold_order() {
            let off = b * MAX_BLOCK;
            if let Some(rv) = self.reverbs[b].as_mut() {
                rv.process(&mut self.acc_l[off..off + n], &mut self.acc_r[off..off + n]);
            }
            if let Some(d) = self.duckers[b].as_mut() {
                for i in 0..n {
                    self.duck_gain[i] = d.step(self.key_db[i]);
                }
                for i in 0..n {
                    self.acc_l[off + i] = scale_finite(self.acc_l[off + i], self.duck_gain[i]);
                    self.acc_r[off + i] = scale_finite(self.acc_r[off + i], self.duck_gain[i]);
                }
            }
            let (gain, parent) = (self.buses[b].gain, self.buses[b].parent);
            if b == 0 {
                for i in 0..n {
                    out_l[i] = scale_finite(self.acc_l[i], gain);
                    out_r[i] = scale_finite(self.acc_r[i], gain);
                }
                if let Some(lim) = self.limiter.as_mut() {
                    lim.process(&mut out_l[..n], &mut out_r[..n]);
                    self.acc_l[..n].copy_from_slice(&out_l[..n]);
                    self.acc_r[..n].copy_from_slice(&out_r[..n]);
                    self.accumulate_stats(0, 1.0, n);
                } else {
                    self.accumulate_stats(0, gain, n);
                }
            } else if parent >= 0 {
                self.accumulate_stats(b, gain, n);
                let p = parent as usize;
                fold_bus_channel(&mut self.acc_l, b, p, n, gain);
                fold_bus_channel(&mut self.acc_r, b, p, n, gain);
            }
            // parent == -1 on non-master: dead send, discarded.
        }
        self.stat_frames += n as u64;
    }

    /// Fold one bus's post-gain contribution into its stats.
    fn accumulate_stats(&mut self, bus: usize, gain: f32, n: usize) {
        let mut peak = self.stat_peak[bus];
        let mut sum = 0.0f64;
        let off = bus * MAX_BLOCK;
        let g = gain as f64;
        for i in 0..n {
            let l = (self.acc_l[off + i] as f64) * g;
            let r = (self.acc_r[off + i] as f64) * g;
            let l = if l.is_finite() { l } else { 0.0 };
            let r = if r.is_finite() { r } else { 0.0 };
            let a = l.abs().max(r.abs()).min(f32::MAX as f64) as f32;
            if a > peak {
                peak = a;
            }
            sum += l * l + r * r;
        }
        self.stat_peak[bus] = peak;
        if sum.is_finite() {
            self.stat_sum_sq[bus] += sum;
            self.block_sum_sq[bus] = sum;
        }
    }

    fn bus_ref(&self, bus: u32) -> Result<&Bus> {
        self.buses.get(bus as usize).ok_or_else(|| {
            AfError::BadBus(format!(
                "bus index {bus} out of range (max {})",
                MAX_BUSES - 1
            ))
        })
    }

    fn bus_mut(&mut self, bus: u32) -> Result<&mut Bus> {
        self.buses.get_mut(bus as usize).ok_or_else(|| {
            AfError::BadBus(format!(
                "bus index {bus} out of range (max {})",
                MAX_BUSES - 1
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mono(frames: usize, val: f32) -> Arc<[f32]> {
        Arc::from(vec![val; frames])
    }

    #[test]
    fn center_pan_is_unity_both() {
        let mut m = Mixer::new();
        m.start_voice(
            mono(8, 0.5),
            &VoiceParams {
                start_frame: 0,
                ..Default::default()
            },
        )
        .unwrap();
        let (mut l, mut r) = (vec![0.0; 8], vec![0.0; 8]);
        m.render_block(&mut l, &mut r, 0);
        assert_eq!(l, vec![0.5; 8]);
        assert_eq!(r, vec![0.5; 8]);
    }

    #[test]
    fn hard_pans_route_single_channel() {
        for (pan, expect_l, expect_r) in [(-1.0, 0.5, 0.0), (1.0, 0.0, 0.5)] {
            let mut m = Mixer::new();
            m.start_voice(
                mono(4, 0.5),
                &VoiceParams {
                    pan,
                    ..Default::default()
                },
            )
            .unwrap();
            let (mut l, mut r) = (vec![0.0; 4], vec![0.0; 4]);
            m.render_block(&mut l, &mut r, 0);
            assert_eq!((l, r), (vec![expect_l; 4], vec![expect_r; 4]), "pan={pan}");
        }
    }

    #[test]
    fn voice_deactivates_after_end() {
        let mut m = Mixer::new();
        m.start_voice(mono(4, 1.0), &VoiceParams::default())
            .unwrap();
        let (mut l, mut r) = (vec![0.0; 8], vec![0.0; 8]);
        m.render_block(&mut l, &mut r, 0);
        assert_eq!(&l[..4], &[1.0; 4]);
        assert_eq!(&l[4..], &[0.0; 4]);
        assert_eq!(m.active_voices(), 0);
    }

    #[test]
    fn loop_wraps_source() {
        let mut m = Mixer::new();
        let data: Arc<[f32]> = Arc::from(vec![1.0, 2.0]);
        m.start_voice(
            data,
            &VoiceParams {
                loop_frames: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        let (mut l, mut r) = (vec![0.0; 6], vec![0.0; 6]);
        m.render_block(&mut l, &mut r, 0);
        assert_eq!(l, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        assert_eq!(m.active_voices(), 1); // loops never self-deactivate
    }

    #[test]
    fn sub_bus_folds_into_master_scaled() {
        let mut m = Mixer::new();
        m.set_bus_gain(1, 0.5).unwrap();
        m.start_voice(
            mono(4, 1.0),
            &VoiceParams {
                bus: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let (mut l, mut r) = (vec![0.0; 4], vec![0.0; 4]);
        m.render_block(&mut l, &mut r, 0);
        assert_eq!(l, vec![0.5; 4]);
        assert_eq!(r, vec![0.5; 4]);
    }

    #[test]
    fn dead_send_is_discarded() {
        let mut m = Mixer::new();
        m.set_bus_parent(2, -1).unwrap();
        m.start_voice(
            mono(4, 1.0),
            &VoiceParams {
                bus: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let (mut l, mut r) = (vec![9.0; 4], vec![9.0; 4]);
        m.render_block(&mut l, &mut r, 0);
        assert_eq!(l, vec![0.0; 4]);
        assert_eq!(r, vec![0.0; 4]);
    }

    #[test]
    fn rejects_bad_params() {
        let mut m = Mixer::new();
        assert!(m.set_bus_gain(0, f32::NAN).is_err());
        assert!(m.set_bus_gain(0, -1.0).is_err());
        assert!(m.set_bus_gain(99, 1.0).is_err());
        assert!(m.set_bus_parent(0, 1).is_err());
        assert!(m.set_bus_parent(1, 1).is_err());
        assert!(m.set_bus_parent(1, 2).is_ok());
        assert!(m.set_bus_parent(2, 1).is_err()); // would cycle
        let p = VoiceParams {
            pan: 2.0,
            ..Default::default()
        };
        assert!(m.start_voice(mono(4, 1.0), &p).is_err());
        let p = VoiceParams {
            gain: f32::INFINITY,
            ..Default::default()
        };
        assert!(m.start_voice(mono(4, 1.0), &p).is_err());
    }

    #[test]
    fn rate_resamples_duration() {
        // 4 frames of ramp at 2x rate => 2 output frames, then silence.
        let mut m = Mixer::new();
        let data: Arc<[f32]> = Arc::from(vec![0.0, 1.0, 2.0, 3.0]);
        m.start_voice(
            data,
            &VoiceParams {
                rate: 2.0,
                ..Default::default()
            },
        )
        .unwrap();
        let (mut l, mut r) = (vec![0.0; 4], vec![0.0; 4]);
        m.render_block(&mut l, &mut r, 0);
        assert_eq!(&l[..2], &[0.0, 2.0]);
        assert_eq!(&l[2..], &[0.0; 2]);
        assert_eq!(m.active_voices(), 0);
    }

    #[test]
    fn bus_stats_track_post_gain_peak_rms() {
        let mut m = Mixer::new();
        m.reset_stats();
        m.set_bus_gain(1, 0.5).unwrap();
        m.start_voice(
            mono(4, 1.0),
            &VoiceParams {
                bus: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let (mut l, mut r) = (vec![0.0; 4], vec![0.0; 4]);
        m.render_block(&mut l, &mut r, 0);
        let s = m.bus_stats(1).unwrap();
        assert!((s.peak - 0.5).abs() < 1e-6, "peak {}", s.peak);
        assert!((s.rms - 0.5).abs() < 1e-6, "rms {}", s.rms);
        let master = m.bus_stats(0).unwrap();
        assert!((master.peak - 0.5).abs() < 1e-6);
    }

    #[test]
    fn retarget_ramps_across_one_block() {
        let mut m = Mixer::new();
        let slot = m
            .start_voice(mono(16, 1.0), &VoiceParams::default())
            .unwrap();
        m.set_voice_targets(slot, 0.0, 0.0, 1.0, 0.0, None).unwrap();
        let (mut l, mut r) = (vec![0.0; 4], vec![0.0; 4]);
        m.render_block(&mut l, &mut r, 0);
        assert_eq!(l, vec![0.75, 0.5, 0.25, 0.0]);
        assert_eq!(l, r);
        m.render_block(&mut l, &mut r, 4);
        assert_eq!(l, vec![0.0; 4], "target held after the ramp");
        // Pan retarget: hard right at the end of the block.
        let mut m = Mixer::new();
        let slot = m
            .start_voice(mono(16, 1.0), &VoiceParams::default())
            .unwrap();
        m.set_voice_targets(slot, 1.0, 1.0, 1.0, 0.0, None).unwrap();
        m.render_block(&mut l, &mut r, 0);
        assert_eq!(l[3], 0.0);
        assert_eq!(r, vec![1.0; 4]);
        assert!(m.set_voice_targets(slot, 1.0, 2.0, 1.0, 0.0, None).is_err());
        assert!(m.set_voice_targets(slot, 1.0, 0.0, 0.0, 0.0, None).is_err());
    }

    #[test]
    fn binaural_voice_renders_directionally() {
        let mut m = Mixer::new();
        m.set_sample_rate(48_000);
        let pcm: Arc<[f32]> = (0..4096).map(|i| ((i as f32) * 0.05).sin()).collect();
        m.start_voice(
            pcm,
            &VoiceParams {
                binaural: Some((std::f32::consts::FRAC_PI_2, 0.0)),
                ..Default::default()
            },
        )
        .unwrap();
        let (mut l, mut r) = (vec![0.0; 512], vec![0.0; 512]);
        m.render_block(&mut l, &mut r, 0);
        assert!(l.iter().any(|&v| v != 0.0) && r.iter().any(|&v| v != 0.0));
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        assert!(rms(&r) > rms(&l), "hard right should favour the right ear");
    }

    #[test]
    fn lowpass_smooths_and_send_feeds_reverb() {
        let mut m = Mixer::new();
        // Alternating +-1 (Nyquist) is crushed by a dark one-pole.
        let data: Arc<[f32]> = (0..64)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        m.start_voice(
            data,
            &VoiceParams {
                lowpass: 0.05,
                ..Default::default()
            },
        )
        .unwrap();
        let (mut l, mut r) = (vec![0.0; 64], vec![0.0; 64]);
        m.render_block(&mut l, &mut r, 0);
        assert!(l[32..].iter().all(|v| v.abs() < 0.1), "{:?}", &l[32..40]);

        // Send into a reverb bus that is otherwise silent.
        let mut m = Mixer::new();
        m.set_bus_reverb(3, Some(ReverbParams::default()), 48_000)
            .unwrap();
        assert!(m
            .set_bus_reverb(0, Some(ReverbParams::default()), 48_000)
            .is_err());
        m.start_voice(
            mono(256, 1.0),
            &VoiceParams {
                bus: 1,
                send_bus: Some(3),
                send_gain: 1.0,
                ..Default::default()
            },
        )
        .unwrap();
        m.set_bus_gain(1, 0.0).unwrap(); // mute the dry path
        m.reset_stats();
        let mut out_l = vec![0.0; 4096];
        let mut out_r = vec![0.0; 4096];
        m.render_block(&mut out_l, &mut out_r, 0);
        assert!(
            m.bus_stats(3).unwrap().peak > 0.0,
            "reverb return must sound"
        );
        assert!(
            out_l[2000..].iter().any(|&v| v != 0.0),
            "tail outlives the dry voice"
        );
    }

    #[test]
    fn pool_exhaustion_reports() {
        let mut m = Mixer::new();
        for _ in 0..MAX_VOICES {
            m.start_voice(mono(64, 0.1), &VoiceParams::default())
                .unwrap();
        }
        assert!(matches!(
            m.start_voice(mono(64, 0.1), &VoiceParams::default()),
            Err(AfError::NoFreeVoice)
        ));
    }

    #[test]
    fn render_block_handles_max_frame_and_loop_position_without_overflow() {
        let mut m = Mixer::new();
        m.start_voice(
            mono(8, 0.25),
            &VoiceParams {
                loop_frames: Some(4),
                start_frame: 0,
                ..Default::default()
            },
        )
        .unwrap();
        m.start_voice(
            mono(8, 0.25),
            &VoiceParams {
                loop_frames: Some(4),
                start_frame: 0,
                lowpass: 0.5,
                ..Default::default()
            },
        )
        .unwrap();
        m.start_voice(
            mono(8, 0.25),
            &VoiceParams {
                loop_frames: Some(4),
                start_frame: 0,
                binaural: Some((0.2, 0.0)),
                ..Default::default()
            },
        )
        .unwrap();
        let (mut l, mut r) = (vec![0.0; 8], vec![0.0; 8]);
        m.render_block(&mut l, &mut r, u64::MAX - 2);
        assert!(l.iter().chain(&r).all(|v| v.is_finite()));
    }
}
