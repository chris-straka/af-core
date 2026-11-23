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
fn ray_hits_box(o: [f64; 3], d: [f64; 3], eps: f64, b: &Aabb) -> bool {
    if !b.min.is_finite() || !b.max.is_finite() {
        return false;
    }
    let (mut tmin, mut tmax) = (0.0f64, 1.0f64);
    let mn = [b.min.x as f64, b.min.y as f64, b.min.z as f64];
    let mx = [b.max.x as f64, b.max.y as f64, b.max.z as f64];
    for a in 0..3 {
        if mn[a] > mx[a] {
            return false;
        }
        if d[a].abs() < 1e-12 {
            if o[a] < mn[a] || o[a] > mx[a] {
                return false;
            }
        } else {
            let inv = 1.0 / d[a];
            let mut t0 = (mn[a] - o[a]) * inv;
            let mut t1 = (mx[a] - o[a]) * inv;
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
    tmax >= eps && tmin <= 1.0 - eps
}

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Moller-Trumbore, segment parameterized on [0, 1].
fn ray_hits_tri(o: [f64; 3], d: [f64; 3], eps: f64, tri: &[Vec3; 3]) -> bool {
    if !tri.iter().all(|v| v.is_finite()) {
        return false;
    }
    let t0 = [tri[0].x as f64, tri[0].y as f64, tri[0].z as f64];
    let t1 = [tri[1].x as f64, tri[1].y as f64, tri[1].z as f64];
    let t2 = [tri[2].x as f64, tri[2].y as f64, tri[2].z as f64];
    let e1 = sub3(t1, t0);
    let e2 = sub3(t2, t0);
    let p = cross3(d, e2);
    let det = dot3(e1, p);
    if det.abs() < 1e-12 {
        return false;
    }
    let inv = 1.0 / det;
    let s = sub3(o, t0);
    let u = dot3(s, p) * inv;
    let tri_eps = SEG_EPS as f64;
    if !(-tri_eps..=1.0 + tri_eps).contains(&u) {
        return false;
    }
    let q = cross3(s, e1);
    let v = dot3(d, q) * inv;
    if v < -tri_eps || u + v > 1.0 + tri_eps {
        return false;
    }
    let seg = dot3(e2, q) * inv;
    (eps..=1.0 - eps).contains(&seg)
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
            let p0 = [t[0].x as f64, t[0].y as f64, t[0].z as f64];
            let p1 = [t[1].x as f64, t[1].y as f64, t[1].z as f64];
            let p2 = [t[2].x as f64, t[2].y as f64, t[2].z as f64];
            let cr = cross3(sub3(p1, p0), sub3(p2, p0));
            let area2 = dot3(cr, cr).sqrt();
            if !area2.is_finite() || area2 <= 1e-9 {
                return Err(AfError::BadArg(format!(
                    "occluders.tris[{i}]: zero-area triangle"
                )));
            }
        }
        Ok(())
    }

    fn segment_blocked_f64(&self, o: [f64; 3], to: [f64; 3]) -> bool {
        let d = [to[0] - o[0], to[1] - o[1], to[2] - o[2]];
        let dist = dot3(d, d).sqrt();
        let eps = (SEG_EPS as f64 / dist.max(1.0)).min(SEG_EPS as f64);
        self.boxes.iter().any(|b| ray_hits_box(o, d, eps, b))
            || self.tris.iter().any(|t| ray_hits_tri(o, d, eps, t))
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
        let o = [listener.x as f64, listener.y as f64, listener.z as f64];
        let s = [source.x as f64, source.y as f64, source.z as f64];
        let dx = s[0] - o[0];
        let dy = s[1] - o[1];
        let dz = s[2] - o[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist <= 1e-9 || !dist.is_finite() || self.is_empty() {
            return Ok(0.0);
        }
        if source_radius == 0.0 {
            return Ok(if self.segment_blocked_f64(o, s) {
                1.0
            } else {
                0.0
            });
        }
        let w = [dx / dist, dy / dist, dz / dist];
        let mut u = cross3(w, [0.0, 1.0, 0.0]);
        let mut u_len = dot3(u, u).sqrt();
        if u_len < 1e-6 {
            u = cross3(w, [1.0, 0.0, 0.0]);
            u_len = dot3(u, u).sqrt();
        }
        let u = [u[0] / u_len, u[1] / u_len, u[2] / u_len];
        let v = cross3(w, u);
        let rad = source_radius as f64;
        let mut blocked = 0usize;
        for i in 0..OCCLUSION_RAYS {
            let target = if i == 0 {
                s
            } else {
                let (ox, oy) = jitter(i);
                let ox = ox as f64 * rad;
                let oy = oy as f64 * rad;
                [
                    s[0] + u[0] * ox + v[0] * oy,
                    s[1] + u[1] * ox + v[1] * oy,
                    s[2] + u[2] * ox + v[2] * oy,
                ]
            };
            if self.segment_blocked_f64(o, target) {
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

    #[test]
    fn large_finite_distances_preserve_occlusion() {
        let s = wall();
        let occ = s
            .occlusion01(Vec3::default(), Vec3::new(0.0, 0.0, -1e20), 0.5)
            .unwrap();
        assert_eq!(occ, 1.0);
    }
}
