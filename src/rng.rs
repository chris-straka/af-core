//! Deterministic seeded RNG (mulberry32) for event randomization.
//!
//! The offline render path must be bit-identical across runs, so event
//! randomization (pitch/volume/pick variation) cannot use thread RNG.
//! Every random draw is a pure function of (seed, draw index).

/// Minimal fast seeded RNG. Deterministic across platforms for the same
/// `u32` stream (wrapping arithmetic only, no floats in the core step).
#[derive(Debug, Clone)]
pub struct SeededRng {
    state: u32,
    draws: u64,
}

impl SeededRng {
    pub fn new(seed: u32) -> Self {
        Self {
            state: seed,
            draws: 0,
        }
    }

    /// Next `u32` in [0, 2^32).
    pub fn next_u32(&mut self) -> u32 {
        // mulberry32
        self.state = self.state.wrapping_add(0x6D2B_79F5);
        let mut z = self.state;
        z = (z ^ (z >> 15)).wrapping_mul(z | 1);
        z ^= z.wrapping_add((z ^ (z >> 7)).wrapping_mul(z | 61));
        self.draws += 1;
        z ^ (z >> 14)
    }

    /// Next `f32` in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        // Top 24 bits -> [0,1). Exact same bits on every platform.
        const SCALE: f32 = 1.0 / 16_777_216.0;
        ((self.next_u32() >> 8) as f32) * SCALE
    }

    /// Next `f32` in [lo, hi].
    pub fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    pub fn draws(&self) -> u64 {
        self.draws
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = SeededRng::new(1234);
        let mut b = SeededRng::new(1234);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = SeededRng::new(1);
        let mut b = SeededRng::new(2);
        let va: Vec<u32> = (0..16).map(|_| a.next_u32()).collect();
        let vb: Vec<u32> = (0..16).map(|_| b.next_u32()).collect();
        assert_ne!(va, vb);
    }

    #[test]
    fn f32_in_unit_range() {
        let mut r = SeededRng::new(7);
        for _ in 0..1000 {
            let v = r.next_f32();
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }
    }
}
