//! Sample-accurate event scheduler (contract §3 port).
//!
//! - `SchedulerEvent`: absolute `frame`, `StartVoice` or `SetBusGain`.
//! - Drain order is total: ascending frame, FIFO within a frame.
//! - `pop_due(end_exclusive)` moves every event with `frame < end`.
//! - Capacity 4096; overflow returns `QueueFull`.

use std::sync::Arc;

use crate::error::{AfError, Result};
use crate::mixer::VoiceParams;

pub const MAX_EVENTS: usize = 4096;

#[derive(Debug, Clone)]
pub enum SchedulerEvent {
    StartVoice {
        frame: u64,
        data: Arc<[f32]>,
        params: VoiceParams,
    },
    SetBusGain {
        frame: u64,
        bus: u32,
        gain: f32,
    },
}

impl SchedulerEvent {
    pub fn frame(&self) -> u64 {
        match self {
            SchedulerEvent::StartVoice { frame, .. } => *frame,
            SchedulerEvent::SetBusGain { frame, .. } => *frame,
        }
    }
}

#[derive(Debug, Default)]
pub struct Scheduler {
    /// Sorted by (frame, insertion order). `stable` insertion keeps FIFO.
    events: Vec<(u64, u64, SchedulerEvent)>,
    seq: u64,
}

impl Scheduler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, ev: SchedulerEvent) -> Result<()> {
        if self.events.len() >= MAX_EVENTS {
            return Err(AfError::QueueFull);
        }
        let frame = ev.frame();
        // Stable insertion: after all existing events with frame <= new.
        let pos = self
            .events
            .iter()
            .position(|(f, _, _)| *f > frame)
            .unwrap_or(self.events.len());
        self.events.insert(pos, (frame, self.seq, ev));
        self.seq = self.seq.wrapping_add(1);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Move every event with `frame < end_exclusive` into a returned vec
    /// (ascending frame, FIFO within a frame). The queue keeps the rest.
    pub fn pop_due(&mut self, end_exclusive: u64) -> Vec<SchedulerEvent> {
        let split = self
            .events
            .iter()
            .position(|(f, _, _)| *f >= end_exclusive)
            .unwrap_or(self.events.len());
        self.events.drain(..split).map(|(_, _, ev)| ev).collect()
    }

    /// [`Scheduler::pop_due`] plus each event's push sequence number
    /// (0-based since construction), so callers can pair drained events
    /// with side data kept in push order.
    pub fn pop_due_indexed(&mut self, end_exclusive: u64) -> Vec<(usize, SchedulerEvent)> {
        let split = self
            .events
            .iter()
            .position(|(f, _, _)| *f >= end_exclusive)
            .unwrap_or(self.events.len());
        self.events
            .drain(..split)
            .map(|(_, seq, ev)| (seq as usize, ev))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voice_ev(frame: u64) -> SchedulerEvent {
        SchedulerEvent::StartVoice {
            frame,
            data: Arc::from(vec![0.0; 4]),
            params: VoiceParams::default(),
        }
    }

    #[test]
    fn total_order_frame_then_fifo() {
        let mut s = Scheduler::new();
        s.push(voice_ev(30)).unwrap();
        s.push(voice_ev(10)).unwrap();
        s.push(voice_ev(10)).unwrap();
        s.push(SchedulerEvent::SetBusGain {
            frame: 20,
            bus: 0,
            gain: 0.5,
        })
        .unwrap();
        let due = s.pop_due(31);
        let frames: Vec<u64> = due.iter().map(|e| e.frame()).collect();
        assert_eq!(frames, vec![10, 10, 20, 30]);
        assert!(s.is_empty());
    }

    #[test]
    fn pop_due_leaves_future_events() {
        let mut s = Scheduler::new();
        s.push(voice_ev(5)).unwrap();
        s.push(voice_ev(100)).unwrap();
        let due = s.pop_due(10);
        assert_eq!(due.len(), 1);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn overflow_reports_queue_full() {
        let mut s = Scheduler::new();
        for _ in 0..MAX_EVENTS {
            s.push(voice_ev(0)).unwrap();
        }
        assert!(matches!(s.push(voice_ev(0)), Err(AfError::QueueFull)));
    }
}
