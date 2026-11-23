//! Error type for the whole workspace (no exceptions across boundaries,
//! mirroring the C++ `std::expected` policy).

use std::{fmt, io};

#[derive(Debug)]
pub enum AfError {
    BadBus(String),
    BadGain(String),
    BadPan(String),
    BadArg(String),
    BadFormat(String),
    UnsupportedFormat(String),
    Truncated(String),
    IoError(String),
    NoFreeVoice,
    QueueFull,
    Kira(String),
}

impl fmt::Display for AfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AfError::BadBus(m) => write!(f, "BadBus: {m}"),
            AfError::BadGain(m) => write!(f, "BadGain: {m}"),
            AfError::BadPan(m) => write!(f, "BadPan: {m}"),
            AfError::BadArg(m) => write!(f, "BadArg: {m}"),
            AfError::BadFormat(m) => write!(f, "BadFormat: {m}"),
            AfError::UnsupportedFormat(m) => write!(f, "UnsupportedFormat: {m}"),
            AfError::Truncated(m) => write!(f, "Truncated: {m}"),
            AfError::IoError(m) => write!(f, "IoError: {m}"),
            AfError::NoFreeVoice => write!(f, "NoFreeVoice: voice pool exhausted"),
            AfError::QueueFull => write!(f, "QueueFull: event queue is full"),
            AfError::Kira(m) => write!(f, "Kira: {m}"),
        }
    }
}

impl std::error::Error for AfError {}

impl From<io::Error> for AfError {
    fn from(e: io::Error) -> Self {
        AfError::IoError(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, AfError>;
