//! 16-bit PCM WAV reader/writer (contract §4 port).
//!
//! Layout: RIFF/WAVE, `fmt ` PCM (tag 1), 16-bit, 1-2 channels, any
//! nonzero rate. Writer emits exactly this (44-byte header, one `data`
//! chunk). Float mapping: write clamps to [-1, 1] and stores
//! `round(s * 32767)`; read returns `q / 32768`.

use std::io::{Read, Write};

use crate::error::{AfError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct WavSpec {
    pub channels: u16,
    pub sample_rate: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WavData {
    pub spec: WavSpec,
    /// Interleaved samples in [-1, 1].
    pub samples: Vec<f32>,
}

impl WavData {
    pub fn frames(&self) -> usize {
        if self.spec.channels == 0 {
            0
        } else {
            self.samples.len() / self.spec.channels as usize
        }
    }

    /// Mono view: channel 0 when stereo, the only channel when mono.
    pub fn mono(&self) -> Vec<f32> {
        let ch = self.spec.channels as usize;
        if ch <= 1 {
            return self.samples.clone();
        }
        self.samples.iter().step_by(ch).copied().collect()
    }
}

pub fn write_wav<W: Write>(mut w: W, spec: &WavSpec, samples: &[f32]) -> Result<()> {
    if spec.channels == 0 || spec.channels > 2 {
        return Err(AfError::BadFormat(format!(
            "channel count {} unsupported (need 1-2)",
            spec.channels
        )));
    }
    if spec.sample_rate == 0 {
        return Err(AfError::BadFormat(
            "sample rate must be nonzero".to_string(),
        ));
    }
    if !samples.len().is_multiple_of(spec.channels as usize) {
        return Err(AfError::BadFormat("partial frame in samples".to_string()));
    }
    let data_bytes = (samples.len() * 2) as u32;
    let byte_rate = spec.sample_rate * spec.channels as u32 * 2;
    let block_align = spec.channels * 2;

    let mut hdr = [0u8; 44];
    hdr[0..4].copy_from_slice(b"RIFF");
    hdr[4..8].copy_from_slice(&(36 + data_bytes).to_le_bytes());
    hdr[8..12].copy_from_slice(b"WAVE");
    hdr[12..16].copy_from_slice(b"fmt ");
    hdr[16..20].copy_from_slice(&16u32.to_le_bytes());
    hdr[20..22].copy_from_slice(&1u16.to_le_bytes()); // PCM
    hdr[22..24].copy_from_slice(&spec.channels.to_le_bytes());
    hdr[24..28].copy_from_slice(&spec.sample_rate.to_le_bytes());
    hdr[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    hdr[32..34].copy_from_slice(&block_align.to_le_bytes());
    hdr[34..36].copy_from_slice(&16u16.to_le_bytes());
    hdr[36..40].copy_from_slice(b"data");
    hdr[40..44].copy_from_slice(&data_bytes.to_le_bytes());
    w.write_all(&hdr)?;

    let mut buf = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let c = s.clamp(-1.0, 1.0);
        let q = (c * 32767.0).round() as i16;
        buf.extend_from_slice(&q.to_le_bytes());
    }
    w.write_all(&buf)?;
    Ok(())
}

fn skip_exact<R: Read>(r: &mut R, len: u64, msg: &str) -> Result<()> {
    let n = std::io::copy(&mut r.by_ref().take(len), &mut std::io::sink())
        .map_err(|_| AfError::Truncated(msg.to_string()))?;
    if n != len {
        return Err(AfError::Truncated(msg.to_string()));
    }
    Ok(())
}

fn read_exact_bounded<R: Read>(r: &mut R, len: u64, msg: &str) -> Result<Vec<u8>> {
    // Cap upfront allocation so corrupt/truncated headers cannot force a 4 GiB alloc.
    let cap = usize::try_from(len).unwrap_or(usize::MAX).min(64 * 1024);
    let mut buf = Vec::with_capacity(cap);
    let n = r
        .by_ref()
        .take(len)
        .read_to_end(&mut buf)
        .map_err(|_| AfError::Truncated(msg.to_string()))?;
    if (n as u64) != len {
        return Err(AfError::Truncated(msg.to_string()));
    }
    Ok(buf)
}

pub fn read_wav<R: Read>(mut r: R) -> Result<WavData> {
    let mut hdr = [0u8; 12];
    r.read_exact(&mut hdr)
        .map_err(|_| AfError::Truncated("short RIFF header (need 12 bytes)".to_string()))?;
    if &hdr[0..4] != b"RIFF" || &hdr[8..12] != b"WAVE" {
        return Err(AfError::BadFormat("not RIFF/WAVE".to_string()));
    }

    let mut fmt_seen = false;
    let mut channels: u16 = 0;
    let mut sample_rate: u32 = 0;
    let mut payload: Option<Vec<f32>> = None;

    loop {
        let mut chunk_hdr = [0u8; 8];
        match r.read_exact(&mut chunk_hdr) {
            Ok(()) => {}
            Err(_) => break, // EOF: stop scanning
        }
        let size_u64 = u32::from_le_bytes(chunk_hdr[4..8].try_into().unwrap()) as u64;
        let padded = size_u64 + (size_u64 & 1);
        if &chunk_hdr[0..4] == b"fmt " {
            if size_u64 < 16 {
                return Err(AfError::BadFormat("fmt chunk < 16 bytes".to_string()));
            }
            let mut fmt = [0u8; 16];
            r.read_exact(&mut fmt)
                .map_err(|_| AfError::Truncated("truncated fmt chunk".to_string()))?;
            if padded > 16 {
                skip_exact(&mut r, padded - 16, "truncated fmt chunk")?;
            }
            let tag = u16::from_le_bytes([fmt[0], fmt[1]]);
            channels = u16::from_le_bytes([fmt[2], fmt[3]]);
            sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
            let bits = u16::from_le_bytes([fmt[14], fmt[15]]);
            if tag != 1 || bits != 16 {
                return Err(AfError::UnsupportedFormat(format!(
                    "need PCM 16-bit, got tag {tag} bits {bits}"
                )));
            }
            if channels == 0 || channels > 2 {
                return Err(AfError::BadFormat(format!(
                    "channel count {channels} unsupported (need 1-2)"
                )));
            }
            if sample_rate == 0 {
                return Err(AfError::BadFormat("zero sample rate".to_string()));
            }
            fmt_seen = true;
        } else if &chunk_hdr[0..4] == b"data" {
            if !fmt_seen {
                return Err(AfError::BadFormat("data chunk before fmt".to_string()));
            }
            if payload.is_some() {
                // First data chunk wins; skip the rest without allocating chunk size.
                skip_exact(&mut r, padded, "truncated data chunk")?;
                continue;
            }
            let raw = read_exact_bounded(&mut r, padded, "truncated data chunk")?;
            let size = usize::try_from(size_u64)
                .map_err(|_| AfError::BadFormat("data chunk exceeds address space".to_string()))?;
            if !size.is_multiple_of(2) {
                return Err(AfError::BadFormat("odd data size".to_string()));
            }
            let frame_bytes = channels as usize * 2;
            if !size.is_multiple_of(frame_bytes) {
                return Err(AfError::BadFormat("partial frame in data".to_string()));
            }
            let mut samples = Vec::with_capacity(size / 2);
            for i in 0..(size / 2) {
                let q = i16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
                samples.push((q as f32) / 32768.0);
            }
            payload = Some(samples);
        } else {
            // Unknown chunk: skip word-aligned without allocating chunk size.
            skip_exact(&mut r, padded, "truncated chunk payload")?;
        }
    }

    match payload {
        Some(samples) => Ok(WavData {
            spec: WavSpec {
                channels,
                sample_rate,
            },
            samples,
        }),
        None => {
            if fmt_seen {
                Err(AfError::Truncated("fmt seen but no data chunk".to_string()))
            } else {
                Err(AfError::BadFormat("no fmt/data chunks".to_string()))
            }
        }
    }
}

/// Read any common WAV for measuring: PCM 8/16/24/32-bit or IEEE float
/// 32/64-bit (plain or `WAVE_FORMAT_EXTENSIBLE`), 1-2 channels. Samples
/// map to [-1, 1) as `q / 2^(bits-1)` (16-bit: same as [`read_wav`]);
/// floats pass through unclamped.
pub fn read_wav_any(bytes: &[u8]) -> Result<WavData> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(AfError::BadFormat("not RIFF/WAVE".to_string()));
    }
    let mut pos = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // tag, channels, rate, bits
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = pos + 8;
        let end = body.saturating_add(size).min(bytes.len());
        if id == b"fmt " {
            if size < 16 || body + 16 > bytes.len() {
                return Err(AfError::BadFormat("fmt chunk < 16 bytes".to_string()));
            }
            let b = &bytes[body..];
            let mut tag = u16::from_le_bytes([b[0], b[1]]);
            let channels = u16::from_le_bytes([b[2], b[3]]);
            let rate = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
            let bits = u16::from_le_bytes([b[14], b[15]]);
            if tag == 0xFFFE && size >= 26 && body + 26 <= bytes.len() {
                tag = u16::from_le_bytes([b[24], b[25]]);
            }
            fmt = Some((tag, channels, rate, bits));
        } else if id == b"data" {
            let (tag, channels, sample_rate, bits) =
                fmt.ok_or_else(|| AfError::BadFormat("data chunk before fmt".to_string()))?;
            if channels == 0 || channels > 2 {
                return Err(AfError::BadFormat(format!(
                    "channel count {channels} unsupported (need 1-2)"
                )));
            }
            if sample_rate == 0 {
                return Err(AfError::BadFormat("zero sample rate".to_string()));
            }
            let data = &bytes[body..end];
            let samples: Vec<f32> = match (tag, bits) {
                (1, 8) => data.iter().map(|&v| (v as f32 - 128.0) / 128.0).collect(),
                (1, 16) => data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&c| i16::from_le_bytes(c) as f32 / 32_768.0)
                    .collect(),
                (1, 24) => data
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|c| (i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8) as f32 / 8_388_608.0)
                    .collect(),
                (1, 32) => data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&c| (i32::from_le_bytes(c) as f64 / 2_147_483_648.0) as f32)
                    .collect(),
                (3, 32) => data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&c| f32::from_le_bytes(c))
                    .collect(),
                (3, 64) => data
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|&c| f64::from_le_bytes(c) as f32)
                    .collect(),
                _ => {
                    return Err(AfError::UnsupportedFormat(format!(
                        "WAV format tag {tag} with {bits}-bit samples"
                    )))
                }
            };
            let whole = samples.len() - samples.len() % channels as usize;
            let mut samples = samples;
            samples.truncate(whole);
            return Ok(WavData {
                spec: WavSpec {
                    channels,
                    sample_rate,
                },
                samples,
            });
        }
        pos = body.saturating_add(size + (size & 1));
    }
    Err(AfError::BadFormat("no data chunk".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_any_takes_pcm16_and_float32() {
        let spec = WavSpec {
            channels: 2,
            sample_rate: 44_100,
        };
        let samples = vec![0.5f32, -0.25, 1.0, -1.0];
        let mut bytes = Vec::new();
        write_wav(&mut bytes, &spec, &samples).unwrap();
        let a = read_wav_any(&bytes).unwrap();
        assert_eq!(a, read_wav(&bytes[..]).unwrap());
        // Same audio as 32-bit float (tag 3), values past full scale kept.
        let mut f = bytes[..44].to_vec();
        f[20..22].copy_from_slice(&3u16.to_le_bytes());
        f[28..32].copy_from_slice(&(44_100u32 * 8).to_le_bytes());
        f[32..34].copy_from_slice(&8u16.to_le_bytes());
        f[34..36].copy_from_slice(&32u16.to_le_bytes());
        f[40..44].copy_from_slice(&16u32.to_le_bytes());
        for v in [0.5f32, -0.25, 1.5, -1.0] {
            f.extend_from_slice(&v.to_le_bytes());
        }
        let b = read_wav_any(&f).unwrap();
        assert_eq!(b.spec, spec);
        assert_eq!(b.samples, vec![0.5, -0.25, 1.5, -1.0]);
    }

    #[test]
    fn roundtrip_stereo() {
        let spec = WavSpec {
            channels: 2,
            sample_rate: 48_000,
        };
        let samples = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.25];
        let mut buf = Vec::new();
        write_wav(&mut buf, &spec, &samples).unwrap();
        assert_eq!(buf.len(), 44 + samples.len() * 2);
        let back = read_wav(&buf[..]).unwrap();
        assert_eq!(back.spec, spec);
        // Full-scale positive reads back as 32767/32768.
        assert!((back.samples[3] - 32767.0 / 32768.0).abs() < 1e-7);
        for (a, b) in samples.iter().zip(back.samples.iter()).take(3) {
            let q = (a.clamp(-1.0, 1.0) * 32767.0).round() / 32768.0;
            assert!((b - q).abs() < 1e-7, "{a} vs {b}");
        }
    }

    #[test]
    fn clamp_and_header_size() {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 8000,
        };
        let mut buf = Vec::new();
        write_wav(&mut buf, &spec, &[2.0, -2.0]).unwrap();
        let back = read_wav(&buf[..]).unwrap();
        assert!((back.samples[0] - 32767.0 / 32768.0).abs() < 1e-7);
        // -1.0 stores as -32767 and reads back as -32767/32768 (contract §4).
        assert!((back.samples[1] + 32767.0 / 32768.0).abs() < 1e-7);
    }

    #[test]
    fn rejection_table() {
        assert!(matches!(read_wav(&[][..]), Err(AfError::Truncated(_))));
        assert!(matches!(
            read_wav(&b"NOT RIFF AT ALL........".to_vec()[..]),
            Err(AfError::Truncated(_) | AfError::BadFormat(_))
        ));
        // fmt tag != PCM => UnsupportedFormat
        let mut buf = Vec::new();
        write_wav(
            &mut buf,
            &WavSpec {
                channels: 1,
                sample_rate: 8000,
            },
            &[0.0],
        )
        .unwrap();
        buf[20] = 3; // IEEE float tag
        assert!(matches!(
            read_wav(&buf[..]),
            Err(AfError::UnsupportedFormat(_))
        ));
        // data before fmt
        let mut bad = b"RIFF____WAVEdata\x04\x00\x00\x00\x00\x00\x00\x00".to_vec();
        let riff_len = (bad.len() - 8) as u32;
        bad[4..8].copy_from_slice(&riff_len.to_le_bytes());
        assert!(matches!(
            read_wav(&bad[..]),
            Err(AfError::BadFormat(_)) | Err(AfError::Truncated(_))
        ));
    }

    #[test]
    fn truncated_huge_chunk_sizes_return_truncated_without_oom() {
        // Construct minimal valid fmt chunk (16 bytes) followed by huge bogus chunk sizes.
        let mut base = Vec::new();
        write_wav(
            &mut base,
            &WavSpec {
                channels: 1,
                sample_rate: 48_000,
            },
            &[0.0],
        )
        .unwrap();

        // 1. Huge `data` chunk claiming u32::MAX bytes with 0 payload bytes.
        let mut huge_data = base[..36].to_vec();
        huge_data.extend_from_slice(b"data");
        huge_data.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            read_wav(&huge_data[..]),
            Err(AfError::Truncated(_))
        ));

        // 2. Huge unknown `JUNK` chunk claiming u32::MAX bytes before `data`.
        let mut huge_junk = base[..36].to_vec();
        huge_junk.extend_from_slice(b"JUNK");
        huge_junk.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            read_wav(&huge_junk[..]),
            Err(AfError::Truncated(_))
        ));

        // 3. Huge `fmt ` chunk claiming u32::MAX bytes with only 16 bytes present.
        let mut huge_fmt = b"RIFF\x24\x00\x00\x00WAVEfmt ".to_vec();
        huge_fmt.extend_from_slice(&u32::MAX.to_le_bytes());
        huge_fmt.extend_from_slice(&base[20..36]);
        assert!(matches!(
            read_wav(&huge_fmt[..]),
            Err(AfError::Truncated(_))
        ));

        // 4. Huge duplicate `data` chunk claiming u32::MAX bytes after valid first `data`.
        let mut dup_data = base.clone();
        dup_data.extend_from_slice(b"data");
        dup_data.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            read_wav(&dup_data[..]),
            Err(AfError::Truncated(_))
        ));
    }
}
