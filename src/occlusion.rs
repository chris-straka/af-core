//! Deterministic occlusion estimator (offline stand-in for the engine
//! raycast that feeds [`crate::spatial::Occlusion`]).
//!
//! The listener->source segment is tested against boxes + triangles; 16
//! fixed-hash jittered rays around the source turn hit/miss into a fractional
//! `occlusion01`. Pure arithmetic, no allocation in [`OcclusionScene::occlusion01`].
//!
//! JSON shape (scene `occluders`, both keys optional):
//! `{"boxes": [{"min": [x,y,z], "max": [x,y,z]}], "tris": [[[x,y,z],[..],[..]]]}`

use serde::{Deserialize, Serialize};

use crate::error::{AfError, Result};
use crate::spatial::Vec3;

pub const MAX_OCCLUDER_BOXES: usize = 64;
pub const MAX_OCCLUDER_TRIS: usize = 256;
/// Sample rays per query (ray 0 is always the direct ray).
pub const OCCLUSION_RAYS: usize = 16;
const OCCLUSION_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const SEG_EPS: f32 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OcclusionScene {
    #[serde(default)]
    pub boxes: Vec<Aabb>,
    #[serde(default)]
    pub tris: Vec<[Vec3; 3]>,
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Jitter offset for ray `i` (i >= 1), in [-1, 1]^2. Pure function of `i`.
fn jitter(i: usize) -> (f32, f32) {
    let i = i as u64;
    let ox = (splitmix64(OCCLUSION_SEED ^ (i * 2)) % 2001) as f32 / 1000.0 - 1.0;
    let oy = (splitmix64(OCCLUSION_SEED ^ (i * 2 + 1)) % 2001) as f32 / 1000.0 - 1.0;
    (ox, oy)
}

/// Segment `o + d*s`, s in [0, 1], vs an AABB. Starting inside counts.
fn ray_hits_box(o: Vec3, d: Vec3, b: &Aabb) -> bool {
    let (mut tmin, mut tmax) = (0.0f32, 1.0f32);
    let oo = [o.x, o.y, o.z];
    let dd = [d.x, d.y, d.z];
    let mn = [b.min.x, b.min.y, b.min.z];
    let mx = [b.max.x, b.max.y, b.max.z];
    for a in 0..3 {
        if dd[a].abs() < 1e-12 {
            if oo[a] < mn[a] || oo[a] > mx[a] {
                return false;
            }
        } else {
            let inv = 1.0 / dd[a];
            let mut t0 = (mn[a] - oo[a]) * inv;
            let mut t1 = (mx[a] - oo[a]) * inv;
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            tmin = tmin.max(t0);
            tmax = tmax.min(t1);
            if tmin > tmax {
                return false;
            }
        }
    }
    tmax >= SEG_EPS && tmin <= 1.0 - SEG_EPS
}

/// Moller-Trumbore, segment parameterized on [0, 1].
fn ray_hits_tri(o: Vec3, d: Vec3, t: &[Vec3; 3]) -> bool {
    let e1 = t[1] - t[0];
    let e2 = t[2] - t[0];
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return false;
    }
    let inv = 1.0 / det;
    let s = o - t[0];
    let u = s.dot(p) * inv;
    if !(-SEG_EPS..=1.0 + SEG_EPS).contains(&u) {
        return false;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < -SEG_EPS || u + v > 1.0 + SEG_EPS {
        return false;
    }
    let seg = e2.dot(q) * inv;
    (SEG_EPS..=1.0 - SEG_EPS).contains(&seg)
}

impl OcclusionScene {
    pub fn is_empty(&self) -> bool {
        self.boxes.is_empty() && self.tris.is_empty()
    }

    /// Caps, finiteness, non-degenerate boxes/triangles.
    pub fn validate(&self) -> Result<()> {
        if self.boxes.len() > MAX_OCCLUDER_BOXES {
            return Err(AfError::BadArg(format!(
                "occluders.boxes: at most {MAX_OCCLUDER_BOXES}, got {}",
                self.boxes.len()
            )));
        }
        if self.tris.len() > MAX_OCCLUDER_TRIS {
            return Err(AfError::BadArg(format!(
                "occluders.tris: at most {MAX_OCCLUDER_TRIS}, got {}",
                self.tris.len()
            )));
        }
        for (i, b) in self.boxes.iter().enumerate() {
            if !b.min.is_finite() || !b.max.is_finite() {
                return Err(AfError::BadArg(format!(
                    "occluders.boxes[{i}]: corners must be finite"
                )));
            }
            if !(b.min.x < b.max.x && b.min.y < b.max.y && b.min.z < b.max.z) {
                return Err(AfError::BadArg(format!(
                    "occluders.boxes[{i}]: min must be < max on every axis (zero-volume box)"
                )));
            }
        }
        for (i, t) in self.tris.iter().enumerate() {
            if !t.iter().all(|v| v.is_finite()) {
                return Err(AfError::BadArg(format!(
                    "occluders.tris[{i}]: vertices must be finite"
                )));
            }
            if (t[1] - t[0]).cross(t[2] - t[0]).length() <= 1e-9 {
                return Err(AfError::BadArg(format!(
                    "occluders.tris[{i}]: zero-area triangle"
                )));
            }
        }
        Ok(())
    }

    fn segment_blocked(&self, from: Vec3, to: Vec3) -> bool {
        let d = to - from;
        self.boxes.iter().any(|b| ray_hits_box(from, d, b))
            || self.tris.iter().any(|t| ray_hits_tri(from, d, t))
    }

    /// Fraction of [`OCCLUSION_RAYS`] listener->source segments blocked,
    /// in [0, 1]. `source_radius == 0` casts only the direct ray.
    pub fn occlusion01(&self, listener: Vec3, source: Vec3, source_radius: f32) -> Result<f32> {
        if !listener.is_finite() || !source.is_finite() {
            return Err(AfError::BadArg("occlusion endpoints must be finite".into()));
        }
        if !source_radius.is_finite() || source_radius < 0.0 {
            return Err(AfError::BadArg(format!(
                "source_radius must be finite and >= 0, got {source_radius}"
            )));
        }
        let axis = source - listener;
        let dist = axis.length();
        if dist <= 1e-9 || self.is_empty() {
            return Ok(0.0);
        }
        if source_radius == 0.0 {
            return Ok(if self.segment_blocked(listener, source) {
                1.0
            } else {
                0.0
            });
        }
        let w = axis.scale(1.0 / dist);
        let mut u = w.cross(Vec3::new(0.0, 1.0, 0.0));
        if u.length() < 1e-6 {
            u = w.cross(Vec3::new(1.0, 0.0, 0.0));
        }
        let u = u.scale(1.0 / u.length());
        let v = w.cross(u);
        let mut blocked = 0usize;
        for i in 0..OCCLUSION_RAYS {
            let target = if i == 0 {
                source
            } else {
                let (ox, oy) = jitter(i);
                source + u.scale(ox * source_radius) + v.scale(oy * source_radius)
            };
            if self.segment_blocked(listener, target) {
                blocked += 1;
            }
        }
        Ok(blocked as f32 / OCCLUSION_RAYS as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wall() -> OcclusionScene {
        // A 2 m wide wall across the -Z axis at z = -5.
        OcclusionScene {
            boxes: vec![Aabb {
                min: Vec3::new(-1.0, -1.0, -5.1),
                max: Vec3::new(1.0, 1.0, -4.9),
            }],
            tris: vec![],
        }
    }

    #[test]
    fn direct_ray_hit_and_miss() {
        let s = wall();
        let l = Vec3::default();
        assert_eq!(
            s.occlusion01(l, Vec3::new(0.0, 0.0, -10.0), 0.0).unwrap(),
            1.0
        );
        assert_eq!(
            s.occlusion01(l, Vec3::new(5.0, 0.0, -10.0), 0.0).unwrap(),
            0.0
        );
        assert_eq!(
            s.occlusion01(l, Vec3::new(0.0, 0.0, -3.0), 0.0).unwrap(),
            0.0
        );
    }

    #[test]
    fn jitter_gives_partial_occlusion_deterministically() {
        let s = wall();
        // Source right at the wall's edge: some rays pass, some do not.
        let src = Vec3::new(2.0, 0.0, -10.0); // ray crosses the wall edge
        let a = s.occlusion01(Vec3::default(), src, 1.0).unwrap();
        let b = s.occlusion01(Vec3::default(), src, 1.0).unwrap();
        assert_eq!(a, b);
        assert!(a > 0.0 && a < 1.0, "partial occlusion expected, got {a}");
    }

    #[test]
    fn triangles_block() {
        let s = OcclusionScene {
            boxes: vec![],
            tris: vec![[
                Vec3::new(-5.0, -5.0, -2.0),
                Vec3::new(5.0, -5.0, -2.0),
                Vec3::new(0.0, 5.0, -2.0),
            ]],
        };
        assert!(s.validate().is_ok());
        assert_eq!(
            s.occlusion01(Vec3::default(), Vec3::new(0.0, 0.0, -4.0), 0.0)
                .unwrap(),
            1.0
        );
    }

    #[test]
    fn rejects_degenerate_geometry() {
        let mut s = wall();
        s.boxes[0].max.x = s.boxes[0].min.x;
        assert!(s.validate().is_err());
        let s = OcclusionScene {
            boxes: vec![],
            tris: vec![[Vec3::default(); 3]],
        };
        assert!(s.validate().is_err());
        let json = r#"{"boxes":[{"min":[0,0,0],"max":[1,1,1]}],"tris":[]}"#;
        let parsed: OcclusionScene = serde_json::from_str(json).unwrap();
        assert!(parsed.validate().is_ok());
        assert!(serde_json::from_str::<OcclusionScene>(r#"{"walls":[]}"#).is_err());
    }
}
