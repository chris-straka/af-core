//! Realtime playback bridge onto Kira.
//!
//! The deterministic [`crate::mixer`] owns the offline/headless render
//! path (bit-identical WAVs). This module maps the *same* bus/event data
//! onto a Kira `AudioManager` for realtime playback:
//! one Kira sub-track per af bus, sounds played on their bus track with
//! per-event volume/pitch/pan baked into Kira sound settings.
//!
//! Headless tests use Kira's `MockBackend` (no audio device needed).

use std::sync::Arc;

use kira::{
    backend::Backend,
    sound::{
        static_sound::{StaticSoundData, StaticSoundSettings},
        EndPosition, PlaybackPosition, Region,
    },
    track::{TrackBuilder, TrackHandle},
    AudioManager, AudioManagerSettings, Decibels, Frame, Panning, PlaybackRate, Tween,
};

use crate::error::{AfError, Result};

/// Linear amplitude -> dB, floored at -120 dB so gain 0 stays finite.
pub fn gain_to_db(gain: f32) -> f32 {
    20.0 * gain.max(1e-6).log10()
}

fn db(gain: f32) -> Decibels {
    Decibels(gain_to_db(gain))
}

pub struct RealtimePlayer<B: Backend> {
    manager: AudioManager<B>,
    /// tracks[bus] — index mirrors the af bus graph (bus 0 = master).
    tracks: Vec<TrackHandle>,
}

impl<B: Backend> RealtimePlayer<B> {
    /// Build a manager with `bus_count` sub-tracks (bus 0 = master).
    pub fn new(settings: AudioManagerSettings<B>, bus_count: usize) -> Result<Self>
    where
        B::Error: std::fmt::Debug,
    {
        let mut manager =
            AudioManager::<B>::new(settings).map_err(|e| AfError::Kira(format!("{e:?}")))?;
        let mut tracks = Vec::with_capacity(bus_count);
        for i in 0..bus_count {
            tracks.push(
                manager
                    .add_sub_track(TrackBuilder::new())
                    .map_err(|e| AfError::Kira(format!("sub-track {i} (bus capacity): {e:?}")))?,
            );
        }
        Ok(Self { manager, tracks })
    }

    pub fn bus_count(&self) -> usize {
        self.tracks.len()
    }

    /// Mirror an af bus gain onto the Kira sub-track (immediate).
    pub fn set_bus_gain(&mut self, bus: usize, gain: f32) -> Result<()> {
        if !gain.is_finite() || gain < 0.0 {
            return Err(AfError::BadGain(format!(
                "bus {bus} gain must be finite and >= 0, got {gain}"
            )));
        }
        let n = self.tracks.len();
        let t = self
            .tracks
            .get_mut(bus)
            .ok_or_else(|| AfError::BadBus(format!("bus index {bus} out of range (have {n})")))?;
        t.set_volume(db(gain), Tween::default());
        Ok(())
    }

    /// Play mono PCM on a bus track. `gain`/`pan`/`rate` mirror
    /// [`crate::mixer::VoiceParams`] (+ playback rate for pitch
    /// randomization). `loop_whole` loops the full buffer.
    pub fn play(
        &mut self,
        bus: usize,
        pcm: &[f32],
        gain: f32,
        pan: f32,
        rate: f64,
        loop_whole: bool,
    ) -> Result<()> {
        if pcm.is_empty() {
            return Err(AfError::BadArg("cannot play empty PCM".to_string()));
        }
        if !gain.is_finite() || gain < 0.0 {
            return Err(AfError::BadGain(format!(
                "voice gain must be finite and >= 0, got {gain}"
            )));
        }
        if !pan.is_finite() || !(-1.0..=1.0).contains(&pan) {
            return Err(AfError::BadPan(format!(
                "voice pan must be in [-1, 1], got {pan}"
            )));
        }
        if !rate.is_finite() || rate <= 0.0 {
            return Err(AfError::BadArg(format!(
                "playback rate must be finite and > 0, got {rate}"
            )));
        }
        let n = self.tracks.len();
        let track = self
            .tracks
            .get_mut(bus)
            .ok_or_else(|| AfError::BadBus(format!("bus index {bus} out of range (have {n})")))?;
        let frames: Arc<[Frame]> = pcm
            .iter()
            .map(|&s| Frame::from_mono(s))
            .collect::<Vec<_>>()
            .into();
        let mut settings = StaticSoundSettings::new()
            .volume(db(gain))
            .playback_rate(PlaybackRate(rate))
            .panning(Panning(pan.clamp(-1.0, 1.0)));
        if loop_whole {
            settings = settings.loop_region(Region {
                start: PlaybackPosition::Samples(0),
                end: EndPosition::Custom(PlaybackPosition::Samples(frames.len())),
            });
        }
        let data = StaticSoundData {
            sample_rate: 48_000,
            frames,
            settings,
            slice: None,
        };
        track
            .play(data)
            .map(|_| ())
            .map_err(|e| AfError::Kira(format!("play failed: {e:?}")))
    }

    /// Play mono PCM rendered through the HRTF for `azimuth`/`elevation`
    /// (listener space, radians) at `sample_rate` Hz: true stereo, so the
    /// direction reads on headphones. Kira-side pan stays centred.
    #[allow(clippy::too_many_arguments)]
    pub fn play_binaural(
        &mut self,
        bus: usize,
        pcm: &[f32],
        gain: f32,
        azimuth: f32,
        elevation: f32,
        rate: f64,
        loop_whole: bool,
        sample_rate: u32,
    ) -> Result<()> {
        if pcm.is_empty() {
            return Err(AfError::BadArg("cannot play empty PCM".to_string()));
        }
        if !gain.is_finite() || gain < 0.0 {
            return Err(AfError::BadGain(format!(
                "voice gain must be finite and >= 0, got {gain}"
            )));
        }
        if !azimuth.is_finite() || !elevation.is_finite() {
            return Err(AfError::BadArg(format!(
                "binaural direction must be finite, got ({azimuth}, {elevation})"
            )));
        }
        if !rate.is_finite() || rate <= 0.0 {
            return Err(AfError::BadArg(format!(
                "playback rate must be finite and > 0, got {rate}"
            )));
        }
        let (l, r) = crate::binaural::render(pcm, azimuth, elevation, sample_rate);
        let n = self.tracks.len();
        let track = self
            .tracks
            .get_mut(bus)
            .ok_or_else(|| AfError::BadBus(format!("bus index {bus} out of range (have {n})")))?;
        let frames: Arc<[Frame]> = l
            .iter()
            .zip(&r)
            .map(|(&a, &b)| Frame::new(a * gain, b * gain))
            .collect::<Vec<_>>()
            .into();
        let mut settings = StaticSoundSettings::new()
            .volume(Decibels(0.0))
            .playback_rate(PlaybackRate(rate))
            .panning(Panning(0.0));
        if loop_whole {
            settings = settings.loop_region(Region {
                start: PlaybackPosition::Samples(0),
                end: EndPosition::Custom(PlaybackPosition::Samples(frames.len())),
            });
        }
        let data = StaticSoundData {
            sample_rate,
            frames,
            settings,
            slice: None,
        };
        track
            .play(data)
            .map(|_| ())
            .map_err(|e| AfError::Kira(format!("play failed: {e:?}")))
    }

    pub fn manager_mut(&mut self) -> &mut AudioManager<B> {
        &mut self.manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kira::backend::mock::MockBackend;

    #[test]
    fn mock_backend_wiring() {
        let mut player = RealtimePlayer::<MockBackend>::new(AudioManagerSettings::default(), 2)
            .expect("mock backend manager");
        assert_eq!(player.bus_count(), 2);
        player.set_bus_gain(0, 0.8).unwrap();
        player.set_bus_gain(1, 0.5).unwrap();
        assert!(player.set_bus_gain(9, 1.0).is_err());
        assert!(player.set_bus_gain(0, f32::NAN).is_err());
        let pcm = vec![0.25; 1024];
        player.play(1, &pcm, 0.9, -0.2, 1.0, false).unwrap();
        player.play(0, &pcm, 1.0, 0.0, 0.98, true).unwrap();
        assert!(player.play(1, &[], 1.0, 0.0, 1.0, false).is_err());
        assert!(player.play(1, &pcm, -1.0, 0.0, 1.0, false).is_err());
        assert!(player.play(1, &pcm, 1.0, 2.0, 1.0, false).is_err());
        assert!(player.play(1, &pcm, 1.0, 0.0, 0.0, false).is_err());
        player
            .play_binaural(1, &pcm, 0.9, 0.7, 0.1, 1.0, false, 48_000)
            .unwrap();
        assert!(player
            .play_binaural(1, &[], 1.0, 0.0, 0.0, 1.0, false, 48_000)
            .is_err());
        assert!(player
            .play_binaural(1, &pcm, 1.0, f32::NAN, 0.0, 1.0, false, 48_000)
            .is_err());
    }

    #[test]
    fn gain_to_db_known_answers() {
        assert!((gain_to_db(1.0)).abs() < 1e-6);
        assert!((gain_to_db(0.5) + 6.0206).abs() < 1e-3);
        assert!(gain_to_db(0.0).is_finite());
    }
}
