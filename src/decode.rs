//! Compressed-audio decode via Kira (no new deps).
//!
//! Bank clips may be WAV or OGG (whatever the synthforge kit ships);
//! [`load_audio_file`] decodes to mono f32 through Kira's
//! `StaticSoundData::from_file` (Symphonia under the hood). Pure decode —
//! no audio device involved. Sample-rate policy stays P0: the caller
//! (renderer) requires the file rate to equal the scene rate.

use std::path::Path;

use kira::sound::static_sound::StaticSoundData;

use crate::error::{AfError, Result};

#[derive(Debug, Clone)]
pub struct DecodedAudio {
    pub sample_rate: u32,
    /// Mono, (L+R)/2.
    pub mono: Vec<f32>,
}

pub fn load_audio_file(path: &Path) -> Result<DecodedAudio> {
    let data = StaticSoundData::from_file(path).map_err(|e| {
        AfError::BadFormat(format!("{}: cannot decode audio: {e:?}", path.display()))
    })?;
    if data.sample_rate == 0 || data.frames.is_empty() {
        return Err(AfError::BadFormat(format!(
            "{}: decoded to zero frames",
            path.display()
        )));
    }
    let mono: Vec<f32> = data
        .frames
        .iter()
        .map(|f| ((f.left as f64 + f.right as f64) * 0.5) as f32)
        .collect();
    if mono.iter().any(|s| !s.is_finite()) {
        return Err(AfError::BadFormat(format!(
            "{}: decoded audio contains non-finite samples",
            path.display()
        )));
    }
    Ok(DecodedAudio {
        sample_rate: data.sample_rate,
        mono,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav::{write_wav, WavSpec};

    #[test]
    fn decodes_wav_roundtrip() {
        let dir = std::env::temp_dir().join("af_decode_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tone.wav");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 48_000,
        };
        let samples: Vec<f32> = (0..480)
            .map(|i| 0.5 * (i as f32 / 48_000.0 * 440.0 * 2.0 * std::f32::consts::PI).sin())
            .collect();
        let mut buf = Vec::new();
        write_wav(&mut buf, &spec, &samples).unwrap();
        std::fs::write(&path, &buf).unwrap();

        let dec = load_audio_file(&path).unwrap();
        assert_eq!(dec.sample_rate, 48_000);
        assert_eq!(dec.mono.len(), 480);
        let peak = dec.mono.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!((peak - 0.5).abs() < 0.01, "peak {peak}");
    }

    #[test]
    fn missing_file_is_bad_format() {
        let r = load_audio_file(Path::new("/nonexistent/af_nope.wav"));
        assert!(matches!(r, Err(AfError::BadFormat(_))), "{r:?}");
    }
}
