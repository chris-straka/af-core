# af-core

Engine-agnostic deterministic audio DSP, ITU-R BS.1770-4 loudness metering,
and wordless pulse-and-formant voice synthesis (`MIT OR Apache-2.0`).

## Modules

- `loudness`: ITU-R BS.1770-4 / EBU R128 / Tech 3342 (`measure`, `true_peak`,
  `short_term_series`, `step_lufs_series`, allocation-free `KWeighting`, and
  `integrated_from_block_powers`).
- `voice`: deterministic pulse-and-formant wordless character voice
  synthesizer (`VoiceRegister`, `VoiceMood`, `MouthShape`, `VoiceSpec`,
  `synthesize_voice`, `synthesize_utterance`).
- `mixer`, `dynamics`, `spatial`, `binaural`, `occlusion`, `reverb`,
  `footstep`, `playthrough`, `blind`, `wav`, `rng`, `scheduler`.
- `decode` and `kira_bridge` (behind the default `kira` feature).

## Features

- `default = ["kira"]`: enables Kira decoding and realtime bridge.
- `default-features = false`: pure DSP + `serde` only (zero audio driver or
  decoder dependencies).

## Checks

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --locked
cargo clippy --no-default-features --all-targets -- -D warnings
cargo test --locked --no-default-features
```
