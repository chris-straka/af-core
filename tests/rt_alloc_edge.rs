//! RT-safety audit under adversarial / edge inputs:
//! zero heap allocations inside real-time audio processing paths even when
//! fed NaN/Inf/subnormals/0-length/1-sample/huge inputs or bus topology chains.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use af_core::binaural::BinauralState;
use af_core::dynamics::{DuckSpec, Ducker, KeyDetector, Limiter, LimiterSpec};
use af_core::loudness::KWeighting;
use af_core::mixer::{Mixer, VoiceParams};
use af_core::occlusion::{Aabb, OcclusionScene};
use af_core::reverb::{Reverb, ReverbParams};
use af_core::spatial::Vec3;

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
fn rt_paths_never_allocate_under_edge_inputs() {
    let mut m = Box::new(Mixer::new());
    m.set_bus_parent(1, 2).unwrap();
    m.set_bus_reverb(2, Some(ReverbParams::default()), 48_000)
        .unwrap();
    m.set_bus_duck(3, Some(DuckSpec::default())).unwrap();
    m.set_master_limiter(Some(LimiterSpec::default())).unwrap();

    let bad_pcm: Arc<[f32]> = Arc::from(vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        -f32::MAX,
        1e-42,
        0.25,
        -0.25,
    ]);
    let s0 = m
        .start_voice(
            bad_pcm.clone(),
            &VoiceParams {
                bus: 1,
                loop_frames: Some(8),
                lowpass: 0.3,
                send_bus: Some(2),
                send_gain: 0.5,
                sidechain: true,
                ..Default::default()
            },
        )
        .unwrap();
    let s1 = m
        .start_voice(
            bad_pcm,
            &VoiceParams {
                bus: 3,
                loop_frames: Some(4),
                binaural: Some((0.8, -0.2)),
                ..Default::default()
            },
        )
        .unwrap();

    let mut bin = BinauralState::new(48_000);
    let mut rv = Reverb::new(ReverbParams::default(), 48_000).unwrap();
    let mut lim = Limiter::new(LimiterSpec::default(), 48_000);
    let mut kd = KeyDetector::new(48_000);
    let mut ducker = Ducker::new(DuckSpec::default(), 48_000);
    let mut kw = KWeighting::new(48_000.0);
    let occ = OcclusionScene {
        boxes: vec![Aabb {
            min: Vec3::new(-1.0, -1.0, -5.0),
            max: Vec3::new(1.0, 1.0, -3.0),
        }],
        tris: vec![[
            Vec3::new(-2.0, -2.0, -4.0),
            Vec3::new(2.0, -2.0, -4.0),
            Vec3::new(0.0, 2.0, -4.0),
        ]],
    };

    let mut l = vec![0.0f32; 64];
    let mut r = vec![0.0f32; 64];
    let mut one_l = [0.0f32; 1];
    let mut one_r = [0.0f32; 1];
    let mono_bad = [
        f32::NAN,
        f32::INFINITY,
        f32::MAX,
        -f32::MAX,
        1e-42,
        0.1,
        -0.1,
        0.0,
    ];

    ALLOCS.store(0, Ordering::SeqCst);
    ARMED.store(true, Ordering::SeqCst);

    m.render_block(&mut [], &mut [], 0);
    m.render_block(&mut one_l, &mut one_r, 0);
    m.render_block(&mut l, &mut r, u64::MAX - 32);
    let _ = m.set_voice_targets(s0, 0.5, 0.2, 0.4, 0.1, None);
    let _ = m.set_voice_targets(s1, 0.5, 0.0, 1.0, 0.0, Some((-0.4, 0.3)));
    m.render_block(&mut l, &mut r, 64);

    bin.process_into(&[], f32::NAN, 0.0, &mut [], &mut []);
    bin.process_into(&mono_bad, f32::NAN, f32::INFINITY, &mut l[..8], &mut r[..8]);
    rv.process(&mut [], &mut []);
    rv.process(&mut l, &mut r);
    lim.process(&mut [], &mut []);
    lim.process(&mut l, &mut r);
    let db = kd.push(f32::NAN, f32::MAX);
    let _ = ducker.step(db);
    let _ = kw.process(f64::MAX);
    let _ = kw.process(f64::NAN);
    let _ = occ.occlusion01(Vec3::default(), Vec3::new(0.0, 0.0, -10.0), 0.5);

    ARMED.store(false, Ordering::SeqCst);
    let allocs = ALLOCS.load(Ordering::SeqCst);
    assert_eq!(
        allocs, 0,
        "RT paths allocated {allocs} times under edge inputs"
    );
}
