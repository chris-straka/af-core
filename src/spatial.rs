//! 3D spatializer math: distance attenuation, panning, occlusion hooks.
//!
//! Pure arithmetic, no allocation: safe on the audio thread. Output feeds
//! [`crate::mixer::VoiceParams`] gain + pan; the mixer's frozen linear
//! constant-unity pan law is unchanged.
//!
//! Listener convention: right-handed, +X right, +Y up, -Z forward at
//! yaw 0. Yaw rotates counter-clockwise seen from above (radians).

use serde::{Deserialize, Serialize};

use crate::error::{AfError, Result};

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(from = "[f32; 3]", into = "[f32; 3]")]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    pub fn scale(self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }

    pub fn dot(self, o: Vec3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    pub fn length(self) -> f32 {
        let (x, y, z) = (self.x as f64, self.y as f64, self.z as f64);
        (x * x + y * y + z * z).sqrt() as f32
    }

    /// Component-wise linear interpolation, `t` in [0, 1].
    pub fn lerp(self, o: Vec3, t: f32) -> Vec3 {
        self + (o - self).scale(t)
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl From<[f32; 3]> for Vec3 {
    fn from(a: [f32; 3]) -> Self {
        Vec3::new(a[0], a[1], a[2])
    }
}

impl From<Vec3> for [f32; 3] {
    fn from(v: Vec3) -> Self {
        [v.x, v.y, v.z]
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttenuationCurve {
    /// 1 inside `ref_distance`, straight line to 0 at `max_distance`.
    Linear,
    /// `ref / (ref + rolloff * (d - ref))`, OpenAL-style.
    #[default]
    Inverse,
    /// Square of `Inverse` (faster falloff).
    InverseSquare,
    /// `exp(-rolloff * (d - ref) / ref)`.
    Exponential,
}

/// Distance model for one emitter (event `spatial` block).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Attenuation {
    pub curve: AttenuationCurve,
    pub ref_distance: f32,
    pub max_distance: f32,
    pub rolloff: f32,
}

impl Default for Attenuation {
    fn default() -> Self {
        Self {
            curve: AttenuationCurve::Inverse,
            ref_distance: 1.0,
            max_distance: 50.0,
            rolloff: 1.0,
        }
    }
}

impl Attenuation {
    pub fn validate(&self) -> Result<()> {
        if !self.ref_distance.is_finite() || self.ref_distance <= 0.0 {
            return Err(AfError::BadArg(format!(
                "ref_distance must be finite and > 0, got {}",
                self.ref_distance
            )));
        }
        if !self.max_distance.is_finite() || self.max_distance <= self.ref_distance {
            return Err(AfError::BadArg(format!(
                "max_distance must be finite and > ref_distance {}, got {}",
                self.ref_distance, self.max_distance
            )));
        }
        if !self.rolloff.is_finite() || self.rolloff <= 0.0 {
            return Err(AfError::BadArg(format!(
                "rolloff must be finite and > 0, got {}",
                self.rolloff
            )));
        }
        Ok(())
    }

    /// Gain in [0, 1] at `distance` metres.
    pub fn gain(&self, distance: f32) -> Result<f32> {
        self.validate()?;
        if !distance.is_finite() || distance < 0.0 {
            return Err(AfError::BadArg(format!(
                "distance must be finite and >= 0, got {distance}"
            )));
        }
        Ok(self.gain_unchecked(distance))
    }

    fn gain_unchecked(&self, distance: f32) -> f32 {
        let (r, m, k) = (self.ref_distance, self.max_distance, self.rolloff);
        if distance <= r {
            return 1.0;
        }
        let over = distance - r;
        match self.curve {
            AttenuationCurve::Linear => {
                if distance >= m {
                    0.0
                } else {
                    1.0 - over / (m - r)
                }
            }
            AttenuationCurve::Inverse => (r / (r + k * over)).max(0.0),
            AttenuationCurve::InverseSquare => {
                let g = r / (r + k * over);
                g * g
            }
            AttenuationCurve::Exponential => (-k * over / r).exp(),
        }
    }
}

/// Stereo pan in [-1, 1] from a listener-relative azimuth (radians,
/// 0 = ahead, + = right): `sin(azimuth)`. Known answers: 0 -> 0,
/// +pi/2 -> +1, -pi/2 -> -1, pi -> 0.
pub fn pan_from_azimuth(azimuth_rad: f32) -> Result<f32> {
    if !azimuth_rad.is_finite() {
        return Err(AfError::BadArg(format!(
            "azimuth must be finite, got {azimuth_rad}"
        )));
    }
    Ok(azimuth_rad.sin().clamp(-1.0, 1.0))
}

/// Occlusion hook: `occlusion01` (0 = clear, 1 = fully blocked; the
/// engine's raycast or [`crate::occlusion`] supplies it) maps linearly
/// to [1, floor_gain].
pub fn occlusion_gain(occlusion01: f32, floor_gain: f32) -> Result<f32> {
    if !occlusion01.is_finite() || !(0.0..=1.0).contains(&occlusion01) {
        return Err(AfError::BadArg(format!(
            "occlusion must be finite and in [0, 1], got {occlusion01}"
        )));
    }
    if !floor_gain.is_finite() || !(0.0..=1.0).contains(&floor_gain) {
        return Err(AfError::BadArg(format!(
            "occlusion floor gain must be finite and in [0, 1], got {floor_gain}"
        )));
    }
    Ok(1.0 - occlusion01 * (1.0 - floor_gain))
}

/// Clear-path cutoff and fully-occluded cutoff for the occlusion /
/// obstruction low-pass (log-interpolated between them).
pub const LOWPASS_OPEN_HZ: f32 = 20_000.0;
pub const LOWPASS_BLOCKED_HZ: f32 = 800.0;

/// Low-pass corner (Hz) for an obstruction amount in (0, 1]: log sweep
/// from [`LOWPASS_OPEN_HZ`] to [`LOWPASS_BLOCKED_HZ`]. `None` at 0 (open).
pub fn lowpass_cutoff_hz(amount01: f32) -> Option<f32> {
    if !amount01.is_finite() || amount01 <= 0.0 {
        return None;
    }
    let a = amount01.min(1.0);
    Some(LOWPASS_OPEN_HZ * (LOWPASS_BLOCKED_HZ / LOWPASS_OPEN_HZ).powf(a))
}

/// One-pole low-pass coefficient for an obstruction amount in [0, 1].
/// Exactly `1.0` (filter bypassed, bit-exact passthrough) at 0.
pub fn lowpass_coeff(amount01: f32, sample_rate: u32) -> f32 {
    let Some(fc) = lowpass_cutoff_hz(amount01) else {
        return 1.0;
    };
    if sample_rate == 0 {
        return 1.0;
    }
    let fc = fc.min(0.45 * sample_rate as f32);
    1.0 - (-2.0 * std::f32::consts::PI * fc / sample_rate as f32).exp()
}

/// Strength of an occlusion/obstruction pair for the low-pass.
pub fn lowpass_amount(occ: Occlusion) -> f32 {
    occ.occlusion.max(occ.obstruction)
}

/// Listener pose: position + yaw (radians, see module convention).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Listener {
    pub position: Vec3,
    pub yaw: f32,
}

/// Fully placed emitter, ready for the mixer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement {
    /// Distance attenuation * occlusion gain, in [0, 1].
    pub gain: f32,
    /// [-1, 1].
    pub pan: f32,
    /// Listener-space azimuth (0 ahead, + right) and elevation (0 level,
    /// + above), radians. Feeds binaural rendering when the event asks.
    pub azimuth: f32,
    pub elevation: f32,
    /// Metres, emitter to listener.
    pub distance: f32,
    /// One-pole coefficient (1.0 = bypass) for occlusion/obstruction.
    pub lowpass: f32,
}

/// Engine-supplied raycast results for one emitter (the occlusion hook).
/// `occlusion` attenuates and darkens; `obstruction` (partial blockage,
/// e.g. a pillar) only darkens.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Occlusion {
    pub occlusion: f32,
    pub obstruction: f32,
}

/// Places one emitter for one listener.
pub fn place_emitter(
    emitter: Vec3,
    listener: Listener,
    att: &Attenuation,
    occ: Occlusion,
    occlusion_floor: f32,
    sample_rate: u32,
) -> Result<Placement> {
    if !emitter.is_finite() || !listener.position.is_finite() || !listener.yaw.is_finite() {
        return Err(AfError::BadArg(
            "emitter/listener position and yaw must be finite".to_string(),
        ));
    }
    let d = emitter - listener.position;
    let distance = d.length();
    let dist_gain = att.gain(distance)?;
    let occ_gain = occlusion_gain(occ.occlusion, occlusion_floor)?;
    if !occ.obstruction.is_finite() || !(0.0..=1.0).contains(&occ.obstruction) {
        return Err(AfError::BadArg(format!(
            "obstruction must be finite and in [0, 1], got {}",
            occ.obstruction
        )));
    }
    // Rotate the offset into listener space (yaw about +Y). At yaw 0 the
    // listener faces -Z: ahead is -Z, right is +X.
    let (s, c) = listener.yaw.sin_cos();
    let rx = c * d.x - s * d.z;
    let rz = s * d.x + c * d.z;
    let ry = d.y;
    let horizontal = (rx as f64).hypot(rz as f64) as f32;
    let (pan, azimuth, elevation) = if distance <= 1e-6 {
        (0.0, 0.0, 0.0)
    } else {
        let az = rx.atan2(-rz);
        (pan_from_azimuth(az)?, az, ry.atan2(horizontal))
    };
    Ok(Placement {
        gain: dist_gain * occ_gain,
        pan,
        azimuth,
        elevation,
        distance,
        lowpass: lowpass_coeff(lowpass_amount(occ), sample_rate),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI};

    fn att(curve: AttenuationCurve) -> Attenuation {
        Attenuation {
            curve,
            ref_distance: 1.0,
            max_distance: 11.0,
            rolloff: 1.0,
        }
    }

    #[test]
    fn curves_known_answers() {
        // Inside ref => unity for every curve.
        for c in [
            AttenuationCurve::Linear,
            AttenuationCurve::Inverse,
            AttenuationCurve::InverseSquare,
            AttenuationCurve::Exponential,
        ] {
            assert_eq!(att(c).gain(0.5).unwrap(), 1.0);
            assert_eq!(att(c).gain(1.0).unwrap(), 1.0);
        }
        let l = att(AttenuationCurve::Linear);
        assert!((l.gain(6.0).unwrap() - 0.5).abs() < 1e-6);
        assert_eq!(l.gain(11.0).unwrap(), 0.0);
        assert_eq!(l.gain(99.0).unwrap(), 0.0);
        let i = att(AttenuationCurve::Inverse);
        assert!((i.gain(2.0).unwrap() - 0.5).abs() < 1e-6);
        let i2 = att(AttenuationCurve::InverseSquare);
        assert!((i2.gain(2.0).unwrap() - 0.25).abs() < 1e-6);
        let e = att(AttenuationCurve::Exponential);
        assert!((e.gain(2.0).unwrap() - (-1.0f32).exp()).abs() < 1e-6);
    }

    #[test]
    fn rejects_bad_attenuation() {
        let mut a = att(AttenuationCurve::Inverse);
        assert!(a.gain(-1.0).is_err());
        assert!(a.gain(f32::NAN).is_err());
        a.max_distance = 0.5;
        assert!(a.gain(2.0).is_err());
        let mut a = att(AttenuationCurve::Inverse);
        a.rolloff = 0.0;
        assert!(a.gain(2.0).is_err());
    }

    #[test]
    fn pan_known_answers() {
        assert_eq!(pan_from_azimuth(0.0).unwrap(), 0.0);
        assert!((pan_from_azimuth(FRAC_PI_2).unwrap() - 1.0).abs() < 1e-6);
        assert!((pan_from_azimuth(-FRAC_PI_2).unwrap() + 1.0).abs() < 1e-6);
        assert!(pan_from_azimuth(PI).unwrap().abs() < 1e-6);
        assert!(pan_from_azimuth(f32::INFINITY).is_err());
    }

    #[test]
    fn placement_pans_by_yaw() {
        let a = att(AttenuationCurve::Inverse);
        let l = Listener::default();
        let right = place_emitter(
            Vec3::new(2.0, 0.0, 0.0),
            l,
            &a,
            Occlusion::default(),
            0.05,
            48_000,
        )
        .unwrap();
        assert!((right.pan - 1.0).abs() < 1e-6);
        assert!((right.gain - 0.5).abs() < 1e-6);
        assert_eq!(right.lowpass, 1.0);
        let ahead = place_emitter(
            Vec3::new(0.0, 0.0, -3.0),
            l,
            &a,
            Occlusion::default(),
            0.05,
            48_000,
        )
        .unwrap();
        assert!(ahead.pan.abs() < 1e-6);
        // Turn left 90 degrees (CCW): the +X emitter is now behind-right... ahead
        // was -Z; after +pi/2 yaw, ahead is -X, so +X is behind (pan 0).
        let turned = Listener {
            position: Vec3::default(),
            yaw: FRAC_PI_2,
        };
        let p = place_emitter(
            Vec3::new(2.0, 0.0, 0.0),
            turned,
            &a,
            Occlusion::default(),
            0.05,
            48_000,
        )
        .unwrap();
        assert!(p.pan.abs() < 1e-5, "pan {}", p.pan);
        // ...and the -Z emitter is now on the right.
        let p = place_emitter(
            Vec3::new(0.0, 0.0, -2.0),
            turned,
            &a,
            Occlusion::default(),
            0.05,
            48_000,
        )
        .unwrap();
        assert!((p.pan - 1.0).abs() < 1e-5, "pan {}", p.pan);
    }

    #[test]
    fn occlusion_hook_scales_and_darkens() {
        assert_eq!(occlusion_gain(0.0, 0.05).unwrap(), 1.0);
        assert!((occlusion_gain(1.0, 0.05).unwrap() - 0.05).abs() < 1e-6);
        assert!(occlusion_gain(1.5, 0.05).is_err());
        assert!(occlusion_gain(0.5, 2.0).is_err());
        let a = att(AttenuationCurve::Inverse);
        let p = place_emitter(
            Vec3::new(0.0, 0.0, -1.0),
            Listener::default(),
            &a,
            Occlusion {
                occlusion: 1.0,
                obstruction: 0.0,
            },
            0.1,
            48_000,
        )
        .unwrap();
        assert!((p.gain - 0.1).abs() < 1e-6);
        assert!(
            p.lowpass < 0.2,
            "fully occluded => dark, coeff {}",
            p.lowpass
        );
        // Obstruction darkens without attenuating.
        let p = place_emitter(
            Vec3::new(0.0, 0.0, -1.0),
            Listener::default(),
            &a,
            Occlusion {
                occlusion: 0.0,
                obstruction: 0.5,
            },
            0.1,
            48_000,
        )
        .unwrap();
        assert_eq!(p.gain, 1.0);
        assert!(p.lowpass < 1.0);
    }

    #[test]
    fn large_finite_coordinates_do_not_overflow_distance() {
        let a = att(AttenuationCurve::Inverse);
        let p = place_emitter(
            Vec3::new(1e20, 0.0, 1e20),
            Listener::default(),
            &a,
            Occlusion::default(),
            0.1,
            48_000,
        )
        .unwrap();
        assert!(p.distance.is_finite());
        assert!(p.pan.is_finite() && p.azimuth.is_finite() && p.elevation.is_finite());
    }
}
