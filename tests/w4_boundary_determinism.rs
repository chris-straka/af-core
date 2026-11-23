//! Wave 4 boundary, NaN/Inf/subnormal recovery, and repeat-run bit-exact
//! determinism tests across all `af-core` modules.

use std::collections::BTreeMap;
use std::f32::consts::{FRAC_PI_2, PI};
use std::path::PathBuf;
use std::sync::Arc;

use af_core::binaural::{self, BinauralState};
use af_core::blind::{build_blind_pack, score_blind_pack, BlindPackConfig};
use af_core::dynamics::{DuckSpec, Ducker, KeyDetector, Limiter, LimiterSpec, MAX_RATE};
use af_core::footstep::{
    anim_frame_to_audio_frame, contacts_to_steps, resolve_step_surfaces, signal_stats,
    synthesize_footstep, Surface, SurfaceAnnotation, ALL_SURFACES,
};
use af_core::loudness::{
    self, energy_to_lufs, integrated_from_block_powers, short_term_series, step_lufs_series,
    true_peak, KWeighting,
};
use af_core::mixer::{Mixer, VoiceParams, MAX_BLOCK};
use af_core::occlusion::{Aabb, OcclusionScene};
use af_core::playthrough::{
    check, check_with_telemetry, engagement_from_samples, engagement_from_series, envelope,
    measure_silence, BusDuckTelemetry, EngagementSource, MixTelemetry, PlaythroughLimits,
};
use af_core::reverb::{Reverb, ReverbParams};
use af_core::rng::SeededRng;
use af_core::scheduler::{Scheduler, SchedulerEvent};
use af_core::spatial::{
    lowpass_coeff, place_emitter, Attenuation, AttenuationCurve, Listener, Occlusion, Vec3,
};
use af_core::voice::{
    synthesize_utterance, synthesize_voice, MouthShape, VoiceMood, VoiceRegister, VoiceSpec,
};
use af_core::wav::{read_wav, read_wav_any, write_wav, WavSpec};
use af_core::AfError;

fn assert_bits_eq(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}: sample {i} bits differ ({x} vs {y})"
        );
    }
}

#[test]
fn repeat_run_bit_exact_determinism_across_all_dsp() {
    // 1. RNG
    let mut r1 = SeededRng::new(0x1234_5678);
    let mut r2 = SeededRng::new(0x1234_5678);
    for _ in 0..128 {
        assert_eq!(r1.next_u32(), r2.next_u32());
        assert_eq!(
            r1.range_f32(-3.5, 7.25).to_bits(),
            r2.range_f32(-3.5, 7.25).to_bits()
        );
    }

    // 2. Voice
    let v_spec = VoiceSpec {
        register: VoiceRegister::Mid,
        duration_secs: 0.35,
        pitch_keys: vec![(0.0, -2.0), (0.2, 3.0), (0.35, 0.5)],
        amp_keys: vec![(0.0, 0.0), (0.04, 1.0), (0.35, 0.0)],
        shape_keys: vec![(0.0, MouthShape::A), (0.2, MouthShape::O)],
        breath: 0.25,
        rough: 0.4,
        tilt: 1.5,
        breath_band_keys: vec![(0.0, 1200.0), (0.35, 2400.0)],
        peak: 0.6,
        seed: 77,
    };
    assert_bits_eq(
        &synthesize_voice(&v_spec, 48_000).unwrap(),
        &synthesize_voice(&v_spec, 48_000).unwrap(),
        "voice",
    );
    assert_bits_eq(
        &synthesize_utterance(VoiceRegister::Low, VoiceMood::Angry, 9, 3, 48_000).unwrap(),
        &synthesize_utterance(VoiceRegister::Low, VoiceMood::Angry, 9, 3, 48_000).unwrap(),
        "utterance",
    );

    // 3. Footstep
    for s in ALL_SURFACES {
        let f1 = synthesize_footstep(s, 313, 0.85, 48_000, 4096).unwrap();
        let f2 = synthesize_footstep(s, 313, 0.85, 48_000, 4096).unwrap();
        assert_bits_eq(&f1, &f2, s.name());
    }

    // 4. Spatial & Occlusion
    let att = Attenuation::default();
    let listener = Listener {
        position: Vec3::new(1.0, 2.0, 3.0),
        yaw: 0.75,
    };
    let occ_scene = OcclusionScene {
        boxes: vec![Aabb {
            min: Vec3::new(-1.0, -1.0, -4.0),
            max: Vec3::new(1.0, 3.0, -2.0),
        }],
        tris: vec![[
            Vec3::new(-2.0, 0.0, -5.0),
            Vec3::new(2.0, 0.0, -5.0),
            Vec3::new(0.0, 4.0, -5.0),
        ]],
    };
    let o1 = occ_scene
        .occlusion01(listener.position, Vec3::new(0.8, 1.5, -8.0), 0.9)
        .unwrap();
    let o2 = occ_scene
        .occlusion01(listener.position, Vec3::new(0.8, 1.5, -8.0), 0.9)
        .unwrap();
    assert_eq!(o1.to_bits(), o2.to_bits());
    let p1 = place_emitter(
        Vec3::new(0.8, 1.5, -8.0),
        listener,
        &att,
        Occlusion {
            occlusion: o1,
            obstruction: 0.3,
        },
        0.1,
        48_000,
    )
    .unwrap();
    let p2 = place_emitter(
        Vec3::new(0.8, 1.5, -8.0),
        listener,
        &att,
        Occlusion {
            occlusion: o2,
            obstruction: 0.3,
        },
        0.1,
        48_000,
    )
    .unwrap();
    assert_eq!(p1.gain.to_bits(), p2.gain.to_bits());
    assert_eq!(p1.pan.to_bits(), p2.pan.to_bits());
    assert_eq!(p1.azimuth.to_bits(), p2.azimuth.to_bits());
    assert_eq!(p1.elevation.to_bits(), p2.elevation.to_bits());
    assert_eq!(p1.lowpass.to_bits(), p2.lowpass.to_bits());

    // 5. Binaural & Reverb & Dynamics
    let mono: Vec<f32> = (0..1024)
        .map(|i| 0.7 * (i as f32 * 0.09).sin() + 0.2 * (i as f32 * 0.31).cos())
        .collect();
    let (bl1, br1) = binaural::render(&mono, 1.1, -0.35, 48_000);
    let (bl2, br2) = binaural::render(&mono, 1.1, -0.35, 48_000);
    assert_bits_eq(&bl1, &bl2, "binaural L");
    assert_bits_eq(&br1, &br2, "binaural R");

    let run_reverb = || {
        let mut rv = Reverb::new(ReverbParams::default(), 48_000).unwrap();
        let mut l = mono.clone();
        let mut r = mono.clone();
        rv.process(&mut l, &mut r);
        (l, r)
    };
    let (rl1, rr1) = run_reverb();
    let (rl2, rr2) = run_reverb();
    assert_bits_eq(&rl1, &rl2, "reverb L");
    assert_bits_eq(&rr1, &rr2, "reverb R");

    let run_dyn = || {
        let mut lim = Limiter::new(LimiterSpec::default(), 48_000);
        let mut kd = KeyDetector::new(48_000);
        let mut ducker = Ducker::new(DuckSpec::default(), 48_000);
        let mut l: Vec<f32> = mono.iter().map(|&x| x * 2.5).collect();
        let mut r: Vec<f32> = mono.iter().map(|&x| -x * 2.5).collect();
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            let db = kd.push(*a, *b);
            let dg = ducker.step(db);
            let (ol, or) = lim.process_frame(*a * dg, *b * dg);
            *a = ol;
            *b = or;
        }
        (l, r)
    };
    let (dl1, dr1) = run_dyn();
    let (dl2, dr2) = run_dyn();
    assert_bits_eq(&dl1, &dl2, "dynamics L");
    assert_bits_eq(&dr1, &dr2, "dynamics R");

    // 6. Loudness
    let stereo: Vec<f32> = dl1.iter().zip(&dr1).flat_map(|(&a, &b)| [a, b]).collect();
    let m1 = loudness::measure(&stereo, 2, 48_000);
    let m2 = loudness::measure(&stereo, 2, 48_000);
    assert_eq!(m1, m2);
    assert_eq!(
        true_peak(&stereo, 2).to_bits(),
        true_peak(&stereo, 2).to_bits()
    );

    // 7. Mixer full graph (binaural, pan, ramp, lowpass, send+reverb, sidechain, limiter)
    let run_mixer = || {
        let mut m = Mixer::new();
        m.set_sample_rate(48_000);
        m.set_bus_reverb(2, Some(ReverbParams::default()), 48_000)
            .unwrap();
        m.set_bus_duck(1, Some(DuckSpec::default())).unwrap();
        m.set_master_limiter(Some(LimiterSpec::default())).unwrap();
        let pcm: Arc<[f32]> = Arc::from(mono.clone());
        let s0 = m
            .start_voice(
                pcm.clone(),
                &VoiceParams {
                    bus: 1,
                    gain: 0.9,
                    pan: -0.4,
                    lowpass: 0.4,
                    send_bus: Some(2),
                    send_gain: 0.3,
                    ..Default::default()
                },
            )
            .unwrap();
        let s1 = m
            .start_voice(
                pcm.clone(),
                &VoiceParams {
                    bus: 3,
                    gain: 0.8,
                    binaural: Some((0.7, 0.2)),
                    sidechain: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let mut out_l = vec![0.0; 2048];
        let mut out_r = vec![0.0; 2048];
        for blk in 0..4 {
            let off = blk * 512;
            m.set_voice_targets(s0, 0.4 + 0.1 * blk as f32, 0.5, 0.7, 0.2, None)
                .unwrap();
            m.set_voice_targets(s1, 0.6, 0.0, 1.0, 0.0, Some((-0.5 + 0.2 * blk as f32, 0.1)))
                .unwrap();
            m.render_block(
                &mut out_l[off..off + 512],
                &mut out_r[off..off + 512],
                off as u64,
            );
        }
        (
            out_l,
            out_r,
            m.bus_stats(0).unwrap(),
            m.bus_stats(1).unwrap(),
        )
    };
    let (ml1, mr1, bs0_1, bs1_1) = run_mixer();
    let (ml2, mr2, bs0_2, bs1_2) = run_mixer();
    assert_bits_eq(&ml1, &ml2, "mixer L");
    assert_bits_eq(&mr1, &mr2, "mixer R");
    assert_eq!(bs0_1.peak.to_bits(), bs0_2.peak.to_bits());
    assert_eq!(bs0_1.rms.to_bits(), bs0_2.rms.to_bits());
    assert_eq!(bs1_1.peak.to_bits(), bs1_2.peak.to_bits());
    assert_eq!(bs1_1.rms.to_bits(), bs1_2.rms.to_bits());
}

#[test]
fn binaural_huge_and_nonfinite_inputs_recover_cleanly() {
    let mut st = BinauralState::new(48_000);
    let mut l = vec![0.0; 16];
    let mut r = vec![0.0; 16];
    // Alternating +-f32::MAX triggers Biquad overflow -> Inf - Inf = NaN if unguarded.
    let mut poison = vec![f32::MAX, -f32::MAX, f32::MAX, -f32::MAX];
    poison.extend_from_slice(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e-42]);
    poison.resize(16, 1e38);
    st.process_into(&poison, 1.2, -0.5, &mut l, &mut r);
    // Centered (18 kHz shadow filter) + sustained f32::MAX triggers biquad overshoot overflow.
    st.process_into(&[f32::MAX; 16], 0.0, 0.0, &mut l, &mut r);
    assert!(
        l.iter().chain(&r).all(|v| v.is_finite()),
        "binaural output poisoned during huge input: L={l:?}"
    );

    let mut l2 = vec![0.0; 64];
    let mut r2 = vec![0.0; 64];
    st.process_into(&[0.25; 64], 0.3, 0.1, &mut l2, &mut r2);
    assert!(
        l2.iter().chain(&r2).all(|v| v.is_finite()),
        "binaural state wedged into NaN after recovery: L={l2:?}"
    );
    assert!(
        l2.iter().any(|&v| v.abs() > 1e-4),
        "binaural failed to recover signal"
    );

    // Zero/1/max sample rate and 0/1-sample buffers.
    for sr in [0, 1, 768_000, u32::MAX] {
        let (el, er) = binaural::render(&[], 0.0, 0.0, sr);
        assert!(el.is_empty() && er.is_empty());
        let (sl, sr_out) = binaural::render(&[0.5], FRAC_PI_2, PI, sr);
        assert_eq!(sl.len(), 1);
        assert!(sl[0].is_finite() && sr_out[0].is_finite());
    }
}

#[test]
fn reverb_huge_and_nonfinite_inputs_recover_cleanly() {
    let mut rv = Reverb::new(
        ReverbParams {
            room_size: 1.0,
            damping: 0.5,
        },
        48_000,
    )
    .unwrap();
    let mut l = vec![f32::MAX; 128];
    let mut r = vec![f32::MAX; 128];
    for i in (0..128).step_by(2) {
        r[i] = -f32::MAX;
    }
    l[0] = f32::NAN;
    r[1] = f32::INFINITY;
    rv.process(&mut l, &mut r);
    assert!(
        l.iter().chain(&r).all(|v| v.is_finite()),
        "reverb output non-finite on extreme input"
    );

    // Must recover cleanly on normal audio (no permanent NaN wedge).
    for _ in 0..40 {
        let mut l2 = vec![0.1; 256];
        let mut r2 = vec![0.1; 256];
        rv.process(&mut l2, &mut r2);
        assert!(
            l2.iter().chain(&r2).all(|v| v.is_finite()),
            "reverb wedged into NaN after recovery"
        );
    }

    // 0-length, 1-sample, subnormal, and boundary sample rates.
    for sr in [1, 768_000] {
        let mut rvb = Reverb::new(ReverbParams::default(), sr).unwrap();
        rvb.process(&mut [], &mut []);
        let mut a = [1e-42f32];
        let mut b = [-1e-42f32];
        rvb.process(&mut a, &mut b);
        assert!(a[0].is_finite() && b[0].is_finite());
    }
    assert!(Reverb::new(
        ReverbParams {
            room_size: f32::NAN,
            damping: 0.5
        },
        48_000
    )
    .is_err());
    assert!(Reverb::new(
        ReverbParams {
            room_size: 0.5,
            damping: -0.1
        },
        48_000
    )
    .is_err());
}

#[test]
fn dynamics_unvalidated_specs_extreme_inputs_and_recovery() {
    // Ducker must NEVER panic or output NaN/Inf even with unvalidated extreme/negative/NaN DuckSpec.
    for bad_spec in [
        DuckSpec {
            amount_db: -5.0,
            threshold_db: -20.0,
            attack_ms: 0.0,
            hold_ms: -10.0,
            release_ms: 0.0,
        },
        DuckSpec {
            amount_db: f32::NAN,
            threshold_db: f32::NAN,
            attack_ms: f32::NAN,
            hold_ms: f32::INFINITY,
            release_ms: f32::NEG_INFINITY,
        },
    ] {
        for sr in [0, 1, MAX_RATE, u32::MAX] {
            let mut d = Ducker::new(bad_spec, sr);
            for k in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -10.0, 0.0, 1e20] {
                let g = d.step(k);
                assert!(g.is_finite() && g >= 0.0, "bad ducker gain {g}");
                assert!(d.drop_db().is_finite() && d.drop_db() >= 0.0);
            }
        }
    }

    // Limiter interval_peak & process_frame on +-f32::MAX and extreme input gain.
    let mut lim = Limiter::new(LimiterSpec::default(), 48_000);
    lim.set_input_gain(10.0, true);
    let mut hist = [0.0f32; 12];
    for (i, h) in hist.iter_mut().enumerate() {
        *h = if i % 2 == 0 { f32::MAX } else { -f32::MAX };
    }
    let ip = af_core::dynamics::interval_peak(&hist);
    assert!(ip.is_finite() && ip > 0.0, "interval_peak={ip}");

    for i in 0..512 {
        let s = if i % 2 == 0 { f32::MAX } else { -f32::MAX };
        let (ol, or) = lim.process_frame(s, -s);
        assert!(
            ol.is_finite() && or.is_finite(),
            "limiter non-finite at {i}"
        );
        assert!(ol.abs() <= 1.0 && or.abs() <= 1.0);
    }
    let min_g = lim.take_min_gain();
    assert!(
        min_g.is_finite() && (0.0..=1.0).contains(&min_g),
        "min_g={min_g}"
    );

    // Limiter zero/1-sample and extreme sample rate.
    for sr in [0, 1, MAX_RATE, u32::MAX] {
        let mut l2 = Limiter::new(
            LimiterSpec {
                ceiling_dbtp: -1000.0,
                lookahead_ms: -1.0,
                release_ms: 0.0,
            },
            sr,
        );
        l2.process(&mut [], &mut []);
        let mut a = [1e-42f32];
        let mut b = [0.5f32];
        l2.process(&mut a, &mut b);
        assert!(a[0].is_finite() && b[0].is_finite());
    }

    // KeyDetector with 0/u32::MAX sample rate & subnormals.
    let mut kd = KeyDetector::new(0);
    assert!(kd.push(1e-42, -1e-42).is_finite());
    kd.set_sample_rate(u32::MAX);
    assert!(kd.push(f32::MAX, f32::MAX).is_finite());
}

#[test]
fn loudness_extreme_sample_rates_max_inputs_and_odd_channels() {
    // Huge sample rate must not destabilize K-weighting high-pass poles.
    let mut kw_huge = KWeighting::new(1e20);
    for _ in 0..1000 {
        let y = kw_huge.process(1.0);
        assert!(
            y.is_finite() && y.abs() < 10.0,
            "KWeighting unstable at huge sample rate: {y}"
        );
    }

    // f64::MAX must not wedge KWeighting into permanent NaN.
    let mut kw = KWeighting::new(48_000.0);
    let _ = kw.process(f64::MAX);
    let _ = kw.process(-f64::MAX);
    for _ in 0..16 {
        let y = kw.process(0.1);
        assert!(y.is_finite(), "KWeighting wedged into NaN: {y}");
    }
    assert!(kw.process(0.1).abs() > 0.01);

    // energy_to_lufs & integrated_from_block_powers edge cases.
    assert_eq!(energy_to_lufs(f64::NAN), f64::NEG_INFINITY);
    assert_eq!(energy_to_lufs(-5.0), f64::NEG_INFINITY);
    let huge_blocks = integrated_from_block_powers(&[1e308, 1e308]).unwrap();
    assert!(huge_blocks.is_finite(), "huge blocks gave {huge_blocks}");
    assert_eq!(
        integrated_from_block_powers(&[f64::NAN, f64::INFINITY, -1.0]),
        None
    );

    // true_peak with +-f32::MAX and odd-length stereo slice.
    let max_alt: Vec<f32> = (0..32)
        .map(|i| if i % 2 == 0 { f32::MAX } else { -f32::MAX })
        .collect();
    let tp = true_peak(&max_alt, 1);
    assert!(tp.is_finite() && tp > 0.0, "true_peak={tp}");
    assert_eq!(true_peak(&[0.9], 2), 0.0); // partial frame < 1 stereo frame

    // Empty, 1-sample, odd-channel, 0-rate, u32::MAX-rate measure.
    for (samples, ch, sr) in [
        (&[][..], 0, 0),
        (&[0.5f32][..], 2, 48_000),
        (&[0.5f32][..], 1, 1),
        (&[1e-42f32, -1e-42f32, 0.2f32][..], 3, u32::MAX),
    ] {
        let rep = loudness::measure(samples, ch, sr);
        assert!(!rep.integrated_lufs.is_nan());
        assert!(!rep.short_term_max_lufs.is_nan());
        assert!(!rep.momentary_max_lufs.is_nan());
        assert!(rep.loudness_range_lu.is_finite() && rep.loudness_range_lu >= 0.0);
        assert!(!rep.true_peak_dbtp.is_nan());
        assert!(!rep.sample_peak_dbfs.is_nan());
        let _ = short_term_series(samples, ch, sr);
        let _ = step_lufs_series(samples, ch, sr);
    }
}

#[test]
fn mixer_edge_cases_nan_pcm_zero_block_and_bus_parent_order() {
    // 1. Non-finite PCM & +-f32::MAX must not wedge lp_state, acc, or bus_stats.
    let mut m = Mixer::new();
    m.set_bus_reverb(2, Some(ReverbParams::default()), 48_000)
        .unwrap();
    m.set_bus_duck(1, Some(DuckSpec::default())).unwrap();
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
    m.start_voice(
        bad_pcm,
        &VoiceParams {
            bus: 1,
            gain: 2.0,
            lowpass: 0.4,
            send_bus: Some(2),
            send_gain: 1.0,
            sidechain: true,
            ..Default::default()
        },
    )
    .unwrap();
    let mut l = vec![0.0; 16];
    let mut r = vec![0.0; 16];
    m.render_block(&mut l, &mut r, 0);
    assert!(
        l.iter().chain(&r).all(|v| v.is_finite()),
        "mixer non-finite output: L={l:?}"
    );
    let bs1 = m.bus_stats(1).unwrap();
    let bs0 = m.bus_stats(0).unwrap();
    assert!(bs1.peak.is_finite() && bs1.rms.is_finite());
    assert!(bs0.peak.is_finite() && bs0.rms.is_finite());
    assert!(m.last_block_energy(1).is_finite());
    assert!(m.last_block_heard_key_energy().is_finite());

    // Subsequent clean voice renders cleanly.
    m.start_voice(Arc::from(vec![0.5; 16]), &VoiceParams::default())
        .unwrap();
    m.render_block(&mut l, &mut r, 16);
    assert!(l.iter().chain(&r).all(|v| v.is_finite()));
    assert!(m.bus_stats(0).unwrap().rms.is_finite());

    // 2. start_voice rejects non-finite binaural (matching set_voice_targets).
    assert!(m
        .start_voice(
            Arc::from(vec![0.5; 8]),
            &VoiceParams {
                binaural: Some((f32::NAN, 0.0)),
                ..Default::default()
            }
        )
        .is_err());

    // 3. Zero-length block render_block(&mut [], &mut [], 0) must NOT clobber pending ramp targets.
    let mut m_zero = Mixer::new();
    let slot = m_zero
        .start_voice(Arc::from(vec![1.0; 8]), &VoiceParams::default())
        .unwrap();
    m_zero
        .set_voice_targets(slot, 0.0, 1.0, 1.0, 0.0, None)
        .unwrap();
    m_zero.render_block(&mut [], &mut [], 0);
    let mut zl = vec![0.0; 4];
    let mut zr = vec![0.0; 4];
    m_zero.render_block(&mut zl, &mut zr, 0);
    assert_eq!(
        zr,
        vec![0.75, 0.5, 0.25, 0.0],
        "zero-length block clobbered pending gain ramp"
    );

    // 4. Child bus index < parent bus index (bus 1 -> bus 2 -> master 0) must reach master!
    let mut m_topo = Mixer::new();
    m_topo.set_bus_parent(1, 2).unwrap();
    m_topo.set_bus_gain(1, 0.5).unwrap();
    m_topo.set_bus_gain(2, 0.5).unwrap();
    m_topo
        .start_voice(
            Arc::from(vec![1.0; 4]),
            &VoiceParams {
                bus: 1,
                ..Default::default()
            },
        )
        .unwrap();
    let mut tl = vec![0.0; 4];
    let mut tr = vec![0.0; 4];
    m_topo.render_block(&mut tl, &mut tr, 0);
    assert_eq!(
        tl,
        vec![0.25; 4],
        "bus 1 -> bus 2 -> bus 0 audio was dropped"
    );

    // 5. 1-sample and MAX_BLOCK renders.
    let mut m_max = Mixer::new();
    m_max
        .start_voice(Arc::from(vec![0.25; MAX_BLOCK]), &VoiceParams::default())
        .unwrap();
    let mut max_l = vec![0.0; MAX_BLOCK];
    let mut max_r = vec![0.0; MAX_BLOCK];
    m_max.render_block(&mut max_l, &mut max_r, 0);
    assert!(max_l.iter().all(|&v| v == 0.25));
    let mut one_l = [0.0f32; 1];
    let mut one_r = [0.0f32; 1];
    m_max.render_block(&mut one_l, &mut one_r, MAX_BLOCK as u64);
    assert_eq!(one_l, [0.0]);
}

#[test]
fn spatial_and_occlusion_extreme_coordinates_and_rates() {
    let att = Attenuation::default();
    // Opposite extreme finite positions whose difference overflows f32::MAX.
    let p = place_emitter(
        Vec3::new(3e38, 0.0, 0.0),
        Listener {
            position: Vec3::new(-3e38, 0.0, 0.0),
            yaw: 0.0,
        },
        &att,
        Occlusion::default(),
        0.1,
        u32::MAX,
    )
    .expect("finite coordinates must not overflow distance");
    assert!(p.distance.is_finite() && p.gain.is_finite());
    assert!(p.pan.is_finite() && p.azimuth.is_finite() && p.elevation.is_finite());
    assert!(p.lowpass.is_finite() && p.lowpass > 0.0);

    // Coincident emitter/listener and extreme sample rates for lowpass_coeff.
    let p0 = place_emitter(
        Vec3::default(),
        Listener::default(),
        &att,
        Occlusion {
            occlusion: 0.5,
            obstruction: 0.5,
        },
        0.1,
        0,
    )
    .unwrap();
    assert_eq!(p0.distance, 0.0);
    assert_eq!(p0.lowpass, 1.0);
    assert!(lowpass_coeff(1.0, u32::MAX) > 0.0);
    assert!(lowpass_coeff(f32::NAN, 48_000) == 1.0);

    // Linear attenuation at huge distance.
    let lin = Attenuation {
        curve: AttenuationCurve::Linear,
        ref_distance: 1.0,
        max_distance: f32::MAX,
        rolloff: 1.0,
    };
    assert!(lin.gain(f32::MAX).unwrap().is_finite());

    // Occlusion at huge finite coordinates & huge radius must not overflow f32 to Inf.
    let wall = OcclusionScene {
        boxes: vec![Aabb {
            min: Vec3::new(-10.0, -10.0, -6.0),
            max: Vec3::new(10.0, 10.0, -4.0),
        }],
        tris: vec![],
    };
    let occ = wall
        .occlusion01(Vec3::default(), Vec3::new(0.0, 0.0, -3e38), 3e38)
        .unwrap();
    assert!(occ.is_finite() && (0.0..=1.0).contains(&occ));

    // Huge collinear zero-area triangle must be rejected by validate().
    let bad_tri = OcclusionScene {
        boxes: vec![],
        tris: vec![[
            Vec3::new(1e38, 0.0, 0.0),
            Vec3::new(-1e38, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
        ]],
    };
    assert!(
        bad_tri.validate().is_err(),
        "huge collinear triangle should fail validation"
    );
}

#[test]
fn voice_and_footstep_boundary_and_extreme_inputs() {
    // Voice extreme semitone pitch shift and extreme key values must not produce NaN/Inf.
    let spec = VoiceSpec {
        register: VoiceRegister::High,
        duration_secs: 0.05,
        pitch_keys: vec![(0.0, 2000.0), (0.05, -2000.0)],
        amp_keys: vec![(0.0, 1.0), (0.05, 1.0)],
        shape_keys: vec![(0.0, MouthShape::A)],
        breath: 0.2,
        rough: 0.5,
        tilt: 1.6,
        breath_band_keys: vec![(0.0, f32::MAX)],
        peak: 0.5,
        seed: 1,
    };
    let pcm = synthesize_voice(&spec, 8_000).unwrap();
    assert!(pcm
        .iter()
        .all(|v| v.is_finite() && (-1.0..=1.0).contains(v)));

    // Single-sample voice at 8 kHz (1/8000 s).
    let one_sample = VoiceSpec {
        duration_secs: 1.0 / 8000.0,
        ..spec.clone()
    };
    let pcm1 = synthesize_voice(&one_sample, 8_000).unwrap();
    assert_eq!(pcm1.len(), 1);
    assert!(pcm1[0].is_finite());

    // Opposite-sign huge pitch keys and huge amp keys must not produce NaN or zero-out signal.
    let huge_keys = VoiceSpec {
        duration_secs: 0.05,
        pitch_keys: vec![(0.0, -3e38), (0.05, 3e38)],
        amp_keys: vec![(0.0, 1e38), (0.05, 1e38)],
        ..spec
    };
    let pcm_hk = synthesize_voice(&huge_keys, 8_000).unwrap();
    assert!(pcm_hk.iter().all(|v| v.is_finite()));
    assert!(
        pcm_hk.iter().any(|&v| v.abs() > 0.1),
        "huge amp keys should normalize cleanly instead of overflowing to 0"
    );

    let mut rng = SeededRng::new(u32::MAX);
    for _ in 0..16 {
        assert!(rng.range_f32(-f32::MAX, f32::MAX).is_finite());
        assert!(rng.range_f32(f32::NAN, 1.0).is_finite());
    }

    // Footstep: 0 frames, 1 frame, huge gain, signal_stats with NaN/Inf.
    assert!(synthesize_footstep(Surface::Concrete, 0, 1.0, 48_000, 0)
        .unwrap()
        .is_empty());
    assert_eq!(
        synthesize_footstep(Surface::Metal, 0, f32::MAX, 1, 1)
            .unwrap()
            .len(),
        1
    );
    assert!(synthesize_footstep(Surface::Grass, 1, 1.0, 0, 8).is_err());
    assert!(synthesize_footstep(Surface::Grass, 1, 1.0, u32::MAX, 8).is_err());

    let stats = signal_stats(&[f32::NAN, 0.5, f32::INFINITY, -f32::MAX, f32::MAX]);
    assert!(
        stats.rms.is_finite()
            && stats.peak.is_finite()
            && stats.zero_cross_rate.is_finite()
            && stats.brightness.is_finite(),
        "footstep signal_stats poisoned: {stats:?}"
    );
    assert_eq!(signal_stats(&[]), Default::default());
    assert_eq!(contacts_to_steps(&[], Surface::Wood, 0.5).unwrap(), vec![]);
    let mut steps = contacts_to_steps(&[1.0], Surface::Wood, 0.5).unwrap();
    resolve_step_surfaces(
        &mut steps,
        &[SurfaceAnnotation {
            frame: 0,
            surface: Surface::Metal,
        }],
    );
    assert_eq!(steps[0].surface, Surface::Metal);
    assert_eq!(anim_frame_to_audio_frame(0, 60.0, 48_000).unwrap(), 0);
}

#[test]
fn playthrough_nan_limits_and_extreme_telemetry_do_not_silently_pass() {
    let tone: Vec<f32> = (0..48_000)
        .flat_map(|i| {
            let v = 0.071 * (std::f32::consts::TAU * 1000.0 * i as f32 / 48_000.0).sin();
            [v, v]
        })
        .collect();

    // NaN limits must NEVER silently pass!
    for bad_limits in [
        PlaythroughLimits {
            ceiling_dbtp: f32::NAN,
            ..Default::default()
        },
        PlaythroughLimits {
            min_lufs: f64::NAN,
            ..Default::default()
        },
        PlaythroughLimits {
            max_lufs: f64::NAN,
            ..Default::default()
        },
        PlaythroughLimits {
            max_short_term_lufs: Some(f64::NAN),
            ..Default::default()
        },
        PlaythroughLimits {
            max_silence_secs: f64::NAN,
            ..Default::default()
        },
        PlaythroughLimits {
            max_engaged_percent: f64::NAN,
            ..Default::default()
        },
    ] {
        let rep = check(&tone, 2, 48_000, &bad_limits);
        assert!(
            !rep.passed(),
            "NaN PlaythroughLimits silently passed: {bad_limits:?}"
        );
    }

    // Extreme telemetry engagement_from_series must not overflow sum_gr to Inf.
    let eng = engagement_from_series(
        vec![(0.1, 1e308), (0.2, 1e308)],
        EngagementSource::Telemetry,
        Some(true),
    );
    assert!(
        eng.mean_reduction_db.is_finite(),
        "mean_reduction_db overflowed: {}",
        eng.mean_reduction_db
    );

    // Empty & 1-sample playthrough checks never panic.
    let rep0 = check(&[], 2, 0, &PlaythroughLimits::default());
    assert!(!rep0.passed());
    let rep1 = check(&[0.5], 1, 1, &PlaythroughLimits::default());
    assert!(!rep1.passed());
    assert!(envelope(&[], 2, 10).is_empty());
    assert_eq!(measure_silence(&[], 1, 0, -60.0).silent_percent, 100.0);
    assert_eq!(
        engagement_from_samples(&[], 2, 0, -1.0, None).engaged_percent,
        0.0
    );

    // NaN/negative hold/release/amount in BusDuckTelemetry must stay finite & not skip recovery.
    let tel = MixTelemetry {
        sidechain_configured: true,
        window_secs: 0.1,
        cue_rms_db: vec![-12.0, -12.0, -120.0, -120.0, -120.0, -120.0],
        ducks: vec![BusDuckTelemetry {
            bus: "music".into(),
            amount_db: f32::NAN,
            hold_ms: f32::NAN,
            release_ms: f32::NAN,
            drop_db: vec![0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
            bed_rms_db: vec![-20.0; 6],
        }],
        ..Default::default()
    };
    let r = check_with_telemetry(&tone, 2, 48_000, &PlaythroughLimits::default(), Some(&tel));
    assert!(!r.passed());
    assert!(!r.ducking.unwrap().buses[0].recovered);
}

#[test]
fn wav_and_scheduler_and_blind_boundary_cases() {
    // 0-sample WAV roundtrip.
    let spec = WavSpec {
        channels: 2,
        sample_rate: 48_000,
    };
    let mut zero_wav = Vec::new();
    write_wav(&mut zero_wav, &spec, &[]).unwrap();
    let back0 = read_wav(&zero_wav[..]).unwrap();
    assert_eq!(back0.frames(), 0);
    assert_eq!(read_wav_any(&zero_wav).unwrap(), back0);

    // write_wav sanitizes NaN/Inf to 0.0.
    let mut nan_wav = Vec::new();
    write_wav(
        &mut nan_wav,
        &WavSpec {
            channels: 1,
            sample_rate: 48_000,
        },
        &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5],
    )
    .unwrap();
    let decoded = read_wav(&nan_wav[..]).unwrap();
    assert_eq!(decoded.samples[0], 0.0);
    assert_eq!(decoded.samples[1], 0.0);
    assert_eq!(decoded.samples[2], 0.0);
    assert!((decoded.samples[3] - 0.5).abs() < 1e-4);

    // Duplicate `fmt ` chunk changing channels after `data` must be rejected cleanly.
    let mut dup_fmt = Vec::new();
    write_wav(
        &mut dup_fmt,
        &WavSpec {
            channels: 1,
            sample_rate: 48_000,
        },
        &[0.25],
    )
    .unwrap();
    dup_fmt.extend_from_slice(b"fmt ");
    dup_fmt.extend_from_slice(&16u32.to_le_bytes());
    let mut fmt2 = dup_fmt[20..36].to_vec();
    fmt2[2..4].copy_from_slice(&2u16.to_le_bytes()); // mutate to stereo
    dup_fmt.extend_from_slice(&fmt2);
    assert!(
        read_wav(&dup_fmt[..]).is_err(),
        "duplicate fmt chunk after data corrupted WavData channels"
    );

    // Malformed read_wav_any cases: 0 channels, 3 channels, 0 rate, huge chunks.
    let mut bad_any = zero_wav.clone();
    bad_any[22..24].copy_from_slice(&0u16.to_le_bytes());
    assert!(matches!(read_wav_any(&bad_any), Err(AfError::BadFormat(_))));
    bad_any[22..24].copy_from_slice(&3u16.to_le_bytes());
    assert!(matches!(read_wav_any(&bad_any), Err(AfError::BadFormat(_))));
    let mut zero_rate_any = zero_wav.clone();
    zero_rate_any[24..28].copy_from_slice(&0u32.to_le_bytes());
    assert!(matches!(
        read_wav_any(&zero_rate_any),
        Err(AfError::BadFormat(_))
    ));
    let mut huge_fmt_any = b"RIFF\x24\x00\x00\x00WAVEfmt ".to_vec();
    huge_fmt_any.extend_from_slice(&u32::MAX.to_le_bytes());
    huge_fmt_any.extend_from_slice(&zero_wav[20..36]);
    assert!(read_wav_any(&huge_fmt_any).is_err());

    // Multiple control pairs in BlindKeyFile must be rejected.
    let mut two_controls = af_core::blind::BlindKeyFile {
        seed: 1,
        clip_secs: 6.0,
        pairs: vec![
            af_core::blind::BlindPairKey {
                pair_id: "p1".into(),
                source_stem: "s".into(),
                fault: af_core::blind::BlindFault::SameControl,
                clean_side: af_core::blind::Side::Same,
                gate_choice: af_core::blind::Side::Same,
                control: true,
                a_gate_pass: true,
                a_breaches: vec![],
                b_gate_pass: true,
                b_breaches: vec![],
            },
            af_core::blind::BlindPairKey {
                pair_id: "p2".into(),
                source_stem: "s".into(),
                fault: af_core::blind::BlindFault::SameControl,
                clean_side: af_core::blind::Side::Same,
                gate_choice: af_core::blind::Side::Same,
                control: true,
                a_gate_pass: true,
                a_breaches: vec![],
                b_gate_pass: true,
                b_breaches: vec![],
            },
        ],
    };
    let mut ctrl_ans = BTreeMap::new();
    ctrl_ans.insert("p1".to_string(), "same".to_string());
    ctrl_ans.insert("p2".to_string(), "same".to_string());
    assert!(score_blind_pack(&two_controls, &ctrl_ans).is_err());
    two_controls.pairs.pop();
    let _ = score_blind_pack(&two_controls, &ctrl_ans);

    // Scheduler pop_due_indexed deterministic sequence at extreme frame u64::MAX.
    let mut sched = Scheduler::new();
    sched
        .push(SchedulerEvent::SetBusGain {
            frame: u64::MAX,
            bus: 0,
            gain: 0.5,
        })
        .unwrap();
    assert!(sched.pop_due_indexed(u64::MAX).is_empty());
    assert_eq!(sched.len(), 1);

    // Blind pack repeatability (two packs from identical input must produce bit-exact WAVs & key).
    let dir = std::env::temp_dir().join(format!("af_core_w4_blind_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src_path = dir.join("bed.wav");
    let sr = 24_000u32;
    let bed: Vec<f32> = (0..(sr as usize * 8))
        .flat_map(|i| {
            let t = i as f32 / sr as f32;
            let s = 0.08 * (2.0 * PI * 220.0 * t).sin() + 0.05 * (2.0 * PI * 330.0 * t).sin();
            [s, s]
        })
        .collect();
    let mut bed_bytes = Vec::new();
    write_wav(
        &mut bed_bytes,
        &WavSpec {
            channels: 2,
            sample_rate: sr,
        },
        &bed,
    )
    .unwrap();
    std::fs::write(&src_path, bed_bytes).unwrap();

    let out_a = dir.join("pack_a");
    let key_a = dir.join("pack_a.key.json");
    let out_b = dir.join("pack_b");
    let key_b = dir.join("pack_b.key.json");
    let cfg = BlindPackConfig {
        seed: 987654321,
        clip_secs: 6.5,
        limits: PlaythroughLimits::default(),
    };
    let ka = build_blind_pack(&[PathBuf::from(&src_path)], &out_a, &key_a, &cfg).unwrap();
    let kb = build_blind_pack(&[PathBuf::from(&src_path)], &out_b, &key_b, &cfg).unwrap();
    assert_eq!(ka, kb);
    assert_eq!(
        std::fs::read(&key_a).unwrap(),
        std::fs::read(&key_b).unwrap()
    );
    assert_eq!(
        std::fs::read(out_a.join("pair_01/a.wav")).unwrap(),
        std::fs::read(out_b.join("pair_01/a.wav")).unwrap()
    );

    let mut answers = BTreeMap::new();
    for p in &ka.pairs {
        let ans = match p.clean_side {
            af_core::blind::Side::A => "a",
            af_core::blind::Side::B => "b",
            af_core::blind::Side::Same => "same",
        };
        answers.insert(p.pair_id.clone(), ans.to_string());
    }
    let score = score_blind_pack(&ka, &answers).unwrap();
    assert!(score.pass);
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(feature = "kira")]
#[test]
fn kira_bridge_gain_to_db_handles_nonfinite_and_extreme_values() {
    use af_core::kira_bridge::{gain_to_db, RealtimePlayer};
    use kira::backend::mock::MockBackend;
    use kira::AudioManagerSettings;

    assert!(gain_to_db(0.0).is_finite());
    assert!(gain_to_db(-1.0).is_finite());
    assert!(gain_to_db(f32::NAN).is_finite());
    assert!(gain_to_db(f32::INFINITY).is_finite());
    assert!(gain_to_db(f32::NEG_INFINITY).is_finite());

    let mut player =
        RealtimePlayer::<MockBackend>::new(AudioManagerSettings::default(), 1).unwrap();
    player
        .play_binaural(0, &[0.5; 16], 1e38, 0.2, 0.0, 1.0, false, 48_000)
        .unwrap();
}
