#![forbid(unsafe_code)]

//! af-core: engine-agnostic runtime.
//!
//! Two paths share one contract:
//! - [`mixer`]: deterministic software mixer (voice pool + bus graph).
//!   This is the offline/headless render path: same setup + same events
//!   => bit-identical floats, no audio device needed.
//! - `kira_bridge` (feature `kira`): maps the same buses/events onto a Kira
//!   `AudioManager` for realtime playback. Tested with Kira's
//!   `MockBackend` so CI stays headless.
//!
//! Shared mixer voice/bus contracts, [`spatial`], [`occlusion`], and
//! [`footstep`] recipes are deterministic and engine-agnostic.

pub mod binaural;
pub mod blind;
#[cfg(feature = "kira")]
pub mod decode;
pub mod dynamics;
pub mod error;
pub mod footstep;
#[cfg(feature = "kira")]
pub mod kira_bridge;
pub mod loudness;
pub mod mixer;
pub mod occlusion;
pub mod playthrough;
pub mod reverb;
pub mod rng;
pub mod scheduler;
pub mod spatial;
pub mod voice;
pub mod wav;

pub use error::{AfError, Result};
