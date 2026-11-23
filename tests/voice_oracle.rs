//! Protected oracle tests for gameaudio-10: wordless pulse-and-formant voice
//! synthesizer in `af_core::voice`.

use af_core::voice::{
    synthesize_utterance, synthesize_voice, MouthShape, VoiceMood, VoiceRegister, VoiceSpec,
    ALL_MOODS, ALL_REGISTERS,
};

/// Welch-averaged DFT magnitude at frequency `freq_hz` over the steady middle
/// of `samples` (Hann-windowed 2048-sample frames, 50% hop).
fn welch_power_at(samples: &[f32], sr: u32, freq_hz: f32) -> f64 {
    let win = 2048usize;
    let hop = 1024usize;
    let start = samples.len() / 8;
    let end = (samples.len() * 7) / 8;
    assert!(end > start + win, "signal too short for Welch window");
    let mut sum = 0.0f64;
    let mut frames = 0usize;
    let omega = std::f64::consts::TAU * freq_hz as f64 / sr as f64;
    let mut pos = start;
    while pos + win <= end {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for i in 0..win {
            let w = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / win as f64).cos();
            let x = samples[pos + i] as f64 * w;
            let ang = omega * i as f64;
            re += x * ang.cos();
            im -= x * ang.sin();
        }
        sum += re * re + im * im;
        frames += 1;
        pos += hop;
    }
    sum / frames.max(1) as f64
}

/// Frequency of maximum Welch power in `[lo_hz, hi_hz]` sampled every `step_hz`.
fn peak_freq_in(samples: &[f32], sr: u32, lo_hz: f32, hi_hz: f32, step_hz: f32) -> f32 {
    let mut best_f = lo_hz;
    let mut best_p = -1.0f64;
    let mut f = lo_hz;
    while f <= hi_hz {
        let p = welch_power_at(samples, sr, f);
        if p > best_p {
            best_p = p;
            best_f = f;
        }
        f += step_hz;
    }
    best_f
}

#[test]
fn formant_peaks_match_vowel_resonances_within_five_percent() {
    let sr = 48_000u32;
    // Broadband excitation (`breath = 1.0`) with the air band parked high at
    // 8 kHz isolates the three mouth formants below 3.5 kHz so peaks are not
    // quantized to harmonic spacing.
    let cases = [
        (
            MouthShape::A,
            [(500.0, 900.0), (920.0, 1400.0), (2000.0, 2900.0)],
        ),
        (
            MouthShape::E,
            [(350.0, 750.0), (1450.0, 2150.0), (2200.0, 2900.0)],
        ),
        (
            MouthShape::U,
            [(200.0, 520.0), (600.0, 1050.0), (1850.0, 2650.0)],
        ),
    ];
    for (shape, windows) in cases {
        let (expected, _, _) = shape.formants();
        let spec = VoiceSpec {
            register: VoiceRegister::Mid,
            duration_secs: 1.5,
            pitch_keys: vec![(0.0, 0.0)],
            amp_keys: vec![(0.0, 1.0), (1.5, 1.0)],
            shape_keys: vec![(0.0, shape)],
            breath: 1.0,
            rough: 0.0,
            tilt: 1.6,
            breath_band_keys: vec![(0.0, 8000.0)],
            peak: 0.5,
            seed: 42,
        };
        let pcm = synthesize_voice(&spec, sr).expect("voice spec renders");
        for j in 0..3 {
            let (lo, hi) = windows[j];
            let measured = peak_freq_in(&pcm, sr, lo, hi, 5.0);
            let target = expected[j];
            let rel_err = (measured - target).abs() / target;
            assert!(
                rel_err <= 0.05,
                "{shape:?} F{}: measured {measured:.1} Hz vs target {target:.1} Hz (err {:.1}%)",
                j + 1,
                rel_err * 100.0
            );
        }
    }
}

#[test]
fn registers_sit_at_their_fundamental_pitches_in_order() {
    let sr = 24_000u32;
    let mut measured = Vec::new();
    for reg in ALL_REGISTERS {
        let spec = VoiceSpec {
            register: reg,
            duration_secs: 0.6,
            pitch_keys: vec![(0.0, 0.0)],
            amp_keys: vec![(0.0, 1.0), (0.6, 1.0)],
            shape_keys: vec![(0.0, MouthShape::M)],
            breath: 0.0,
            rough: 0.0,
            tilt: 1.6,
            breath_band_keys: vec![],
            peak: 0.5,
            seed: 7,
        };
        let pcm = synthesize_voice(&spec, sr).expect("steady voice renders");
        let target = reg.base_pitch_hz();
        let f = peak_freq_in(&pcm, sr, target * 0.75, target * 1.25, 1.0);
        let rel_err = (f - target).abs() / target;
        assert!(
            rel_err <= 0.05,
            "{reg:?}: measured f0 {f:.1} Hz vs target {target:.1} Hz"
        );
        measured.push(f);
    }
    assert!(measured[0] < measured[1] && measured[1] < measured[2]);
}

#[test]
fn utterances_are_deterministic_varied_by_seed_and_syllables_and_click_free() {
    let sr = 24_000u32;
    for reg in ALL_REGISTERS {
        for mood in ALL_MOODS {
            let a = synthesize_utterance(reg, mood, 19, 1, sr).expect("utterance renders");
            let b = synthesize_utterance(reg, mood, 19, 1, sr).expect("utterance renders");
            assert_eq!(a.len(), b.len());
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "same seed must be bit-identical for {reg:?}/{mood:?}"
            );
            assert!(a.len() >= (sr as usize) / 5, "{reg:?}/{mood:?} too short");
            assert!(a.iter().all(|s| s.is_finite() && (-1.0..=1.0).contains(s)));
            let peak = a.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            assert!(
                (0.35..=0.60).contains(&peak),
                "{reg:?}/{mood:?} unexpected peak {peak}"
            );
            // Edge fade prevents clicks.
            assert!(a[0].abs() < 1e-4 && a[a.len() - 1].abs() < 1e-4);
        }
    }
    // Different seeds vary the sound.
    let s1 = synthesize_utterance(VoiceRegister::Mid, VoiceMood::Plain, 1, 1, sr).unwrap();
    let s2 = synthesize_utterance(VoiceRegister::Mid, VoiceMood::Plain, 99, 1, sr).unwrap();
    assert_ne!(s1, s2, "different seeds must vary per-page utterance");

    // Multi-syllable utterances are longer and deterministic.
    let one = synthesize_utterance(VoiceRegister::Mid, VoiceMood::Glad, 13, 1, sr).unwrap();
    let three_a = synthesize_utterance(VoiceRegister::Mid, VoiceMood::Glad, 13, 3, sr).unwrap();
    let three_b = synthesize_utterance(VoiceRegister::Mid, VoiceMood::Glad, 13, 3, sr).unwrap();
    assert!(three_a
        .iter()
        .zip(&three_b)
        .all(|(x, y)| x.to_bits() == y.to_bits()));
    assert!(
        three_a.len() > one.len() + sr as usize / 4,
        "3 syllables ({}) must be meaningfully longer than 1 ({})",
        three_a.len(),
        one.len()
    );

    // Invalid arguments return Err.
    assert!(synthesize_utterance(VoiceRegister::Mid, VoiceMood::Plain, 1, 0, sr).is_err());
    assert!(synthesize_utterance(VoiceRegister::Mid, VoiceMood::Plain, 1, 9, sr).is_err());
    assert!(synthesize_utterance(VoiceRegister::Mid, VoiceMood::Plain, 1, 1, 0).is_err());
}
