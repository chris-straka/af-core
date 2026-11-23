//! RT-safety audit: `Mixer::render_block` must not touch the heap,
//! with and without the mix stage (master limiter, sidechain ducker).
//!
//! A counting global allocator wraps the system allocator; the test drives
//! every render path (P0 constant voices, P2 ramps, low-pass, sends, bus
//! reverb, loops, resampling) and asserts zero allocations and frees
//! inside the render calls. Failing-if-violated, per PLAN.md rules.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use af_core::dynamics::{DuckSpec, LimiterSpec};
use af_core::mixer::{Mixer, VoiceParams};
use af_core::reverb::ReverbParams;

struct Counting;

static ARMED: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static A: Counting = Counting;

#[test]
fn render_block_never_allocates() {
    let mut m = Box::new(Mixer::new());
    m.set_bus_reverb(2, Some(ReverbParams::default()), 48_000)
        .unwrap();
    let pcm: Arc<[f32]> = (0..9_000).map(|i| ((i as f32) * 0.01).sin()).collect();
    let mut slots = Vec::new();
    for k in 0..48u32 {
        let p = VoiceParams {
            bus: k % 2,
            gain: 0.1,
            pan: (k as f32 / 48.0) * 2.0 - 1.0,
            start_frame: (k as u64) * 300,
            rate: if k % 3 == 0 { 1.25 } else { 1.0 },
            loop_frames: if k % 4 == 0 { Some(9_000) } else { None },
            lowpass: if k % 5 == 0 { 0.3 } else { 1.0 },
            send_bus: if k % 2 == 0 { Some(2) } else { None },
            send_gain: 0.5,
            // Every sixth voice renders binaurally, so the audit covers it.
            binaural: if k % 6 == 0 {
                Some((k as f32 * 0.13, 0.1))
            } else {
                None
            },
            // Every fifth voice keys the ducker (exercised in pass 2).
            sidechain: k % 5 == 0,
        };
        slots.push(m.start_voice(pcm.clone(), &p).unwrap());
        if k % 6 == 0 {
            m.set_voice_targets(slots[k as usize], 0.1, 0.0, 1.0, 0.5, Some((0.5, -0.1)))
                .unwrap();
        }
    }
    let mut l = vec![0.0f32; 512];
    let mut r = vec![0.0f32; 512];
    let mut frame = 0u64;
    let mut allocs = 0usize;
    // Pass 1: plain mixer. Pass 2 (blocks 200..400): the mix stage on
    // (master limiter, bus 1 ducked by the sidechain voices), switched
    // on from the control side between blocks.
    for block in 0..400 {
        if block == 200 {
            m.set_master_limiter(Some(LimiterSpec::default())).unwrap();
            m.set_bus_duck(1, Some(DuckSpec::default())).unwrap();
            m.set_bus_gain(0, 4.0).unwrap(); // drive the limiter hard
        }
        // Spatial retarget between blocks (control side, also alloc-free).
        for (i, &s) in slots.iter().enumerate() {
            let g = 0.05 + 0.05 * (((block + i) % 7) as f32 / 7.0);
            m.set_voice_targets(s, g, 0.0, 1.0, 0.25, None).unwrap();
        }
        ALLOCS.store(0, Ordering::SeqCst);
        ARMED.store(true, Ordering::SeqCst);
        m.render_block(&mut l, &mut r, frame);
        ARMED.store(false, Ordering::SeqCst);
        allocs += ALLOCS.load(Ordering::SeqCst);
        frame += 512;
    }
    assert!(
        l.iter().any(|&v| v != 0.0),
        "audit must exercise audible voices"
    );
    assert_eq!(allocs, 0, "render_block allocated {allocs} times");
    assert!(
        m.take_limiter_min_gain() < 0.9,
        "pass 2 must drive the limiter"
    );
    assert!(m.take_bus_duck_max_db(1) > 0.0, "pass 2 must duck bus 1");
}
