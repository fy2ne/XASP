// Copyright (c) 2026 fy2ne (https://xasp.fy2ne.me). All rights reserved.
// Dual-Licensed:
//   1. Noncommercial Evaluation & Academic Research: PolyForm-Noncommercial-1.0.0
//   2. Commercial / Production Deployment: Requires a paid Commercial License from fy2ne.
//      Source-available on https://github.com/fy2ne/xasp (not distributed via crates.io).

use crate::frame::{FLAG_IS_VALUE_TENSOR, XaspTcpFrameHeader};
use crate::pool::{StripedTcpRxPool, StripedTcpTxPool, rearm_quickack};
use crate::stripe_mux::{MuxedFrame, NUM_STREAMS, decode_stratum0_payload, stream_coord_range};
use std::io::{self, ErrorKind, Read, Write};
use std::net::TcpStream;
use std::thread;
use std::time::{Duration, Instant};
use xasp_math::polar_qjl::unpack_bulk_stripe_into;
use xasp_math::{Bf16, EncodedLayerHalf, decode_layer_half};

const RING_CAP: usize = 256;

#[derive(Clone, Debug)]
pub struct RetiredLayerFilter {
    ring: [(u64, u16); RING_CAP],
    occupied: [bool; RING_CAP],
    cursor: usize,
}

impl Default for RetiredLayerFilter {
    fn default() -> Self {
        Self {
            ring: [(0, 0); RING_CAP],
            occupied: [false; RING_CAP],
            cursor: 0,
        }
    }
}

impl RetiredLayerFilter {
    pub fn mark_retired(&mut self, request_id_lo: u64, layer_id: u16) {
        self.ring[self.cursor] = (request_id_lo, layer_id);
        self.occupied[self.cursor] = true;
        self.cursor = (self.cursor + 1) % RING_CAP;
    }

    pub fn is_retired(&self, request_id_lo: u64, layer_id: u16) -> bool {
        for i in 0..RING_CAP {
            if self.occupied[i] && self.ring[i] == (request_id_lo, layer_id) {
                return true;
            }
        }
        false
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DrainStats {
    pub late_drained_frames: u64,
    pub late_drained_bytes: u64,
}

pub struct StreamReaderState {
    hdr_buf: [u8; 64],
    hdr_pos: usize,
    active_hdr: Option<XaspTcpFrameHeader>,
    body_buf: Vec<u8>,
    body_pos: usize,
}

impl StreamReaderState {
    fn new() -> Self {
        Self {
            hdr_buf: [0u8; 64],
            hdr_pos: 0,
            active_hdr: None,
            body_buf: Vec::new(),
            body_pos: 0,
        }
    }

    pub fn poll_frame(
        &mut self,
        stream: &mut TcpStream,
        filter: &RetiredLayerFilter,
        stats: &mut DrainStats,
    ) -> io::Result<Option<(XaspTcpFrameHeader, Vec<u8>)>> {
        loop {
            if self.active_hdr.is_none() {
                while self.hdr_pos < 64 {
                    match stream.read(&mut self.hdr_buf[self.hdr_pos..64]) {
                        Ok(0) => return Err(io::Error::new(ErrorKind::UnexpectedEof, "eof")),
                        Ok(n) => self.hdr_pos += n,
                        Err(e) if e.kind() == ErrorKind::WouldBlock => return Ok(None),
                        Err(e) => return Err(e),
                    }
                }
                let hdr = XaspTcpFrameHeader::from_bytes(&self.hdr_buf)
                    .map_err(|msg| io::Error::new(ErrorKind::InvalidData, msg))?;
                let len = hdr.payload_len as usize;
                self.body_buf.resize(len, 0);
                self.body_pos = 0;
                self.active_hdr = Some(hdr);
            }

            let hdr = self.active_hdr.unwrap();
            let need = hdr.payload_len as usize;
            while self.body_pos < need {
                match stream.read(&mut self.body_buf[self.body_pos..need]) {
                    Ok(0) => return Err(io::Error::new(ErrorKind::UnexpectedEof, "eof")),
                    Ok(n) => self.body_pos += n,
                    Err(e) if e.kind() == ErrorKind::WouldBlock => return Ok(None),
                    Err(e) => return Err(e),
                }
            }

            let body = std::mem::take(&mut self.body_buf);
            self.hdr_pos = 0;
            self.body_pos = 0;
            self.active_hdr = None;
            rearm_quickack(stream);

            if filter.is_retired(hdr.request_id_lo, hdr.layer_id) {
                stats.late_drained_frames += 1;
                stats.late_drained_bytes += 64 + need as u64;
                continue;
            }
            return Ok(Some((hdr, body)));
        }
    }
}

pub struct DeadlineAssembler {
    readers: Vec<StreamReaderState>,
    pub filter: RetiredLayerFilter,
    pub stats: DrainStats,
}

impl Default for DeadlineAssembler {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct AssembledTile {
    pub tile_bf16: Vec<Bf16>,
    pub arrived_streams: u16,
    pub arrived_mask: u128,
    pub delta_hat: f32,
}

impl DeadlineAssembler {
    pub fn new() -> Self {
        let mut readers = Vec::with_capacity(NUM_STREAMS);
        for _ in 0..NUM_STREAMS {
            readers.push(StreamReaderState::new());
        }
        Self {
            readers,
            filter: RetiredLayerFilter::default(),
            stats: DrainStats::default(),
        }
    }

    pub fn send_frames(tx: &mut StripedTcpTxPool, frames: &[MuxedFrame]) -> io::Result<()> {
        for frame in frames {
            let sid = frame.header.stream_id as usize;
            let hdr_bytes = frame.header.to_bytes();
            tx.streams[sid].write_all(&hdr_bytes)?;
            tx.streams[sid].write_all(&frame.payload)?;
        }
        Ok(())
    }

    pub fn send_frames_with_stalls(
        tx: &mut StripedTcpTxPool,
        frames: &[MuxedFrame],
        stalled_mask: u16,
        stall_duration: Duration,
    ) -> io::Result<Option<thread::JoinHandle<()>>> {
        let mut delayed: Vec<(usize, [u8; 64], Vec<u8>)> = Vec::new();
        for frame in frames {
            let sid = frame.header.stream_id as usize;
            let hdr_bytes = frame.header.to_bytes();
            if ((stalled_mask >> sid) & 1) != 0 {
                delayed.push((sid, hdr_bytes, frame.payload.clone()));
            } else {
                tx.streams[sid].write_all(&hdr_bytes)?;
                tx.streams[sid].write_all(&frame.payload)?;
            }
        }

        if delayed.is_empty() {
            return Ok(None);
        }

        // clone stalled sockets so background thread releases them after deadline
        let mut owned = Vec::with_capacity(delayed.len());
        for (sid, hdr, body) in delayed {
            let dup = tx.streams[sid].try_clone()?;
            owned.push((dup, hdr, body));
        }

        let handle = thread::spawn(move || {
            thread::sleep(stall_duration);
            for (mut sock, hdr, body) in owned {
                let _ = sock.write_all(&hdr);
                let _ = sock.write_all(&body);
            }
        });
        Ok(Some(handle))
    }

    pub fn collect_layer_until_deadline(
        &mut self,
        rx: &mut StripedTcpRxPool,
        request_id_lo: u64,
        layer_id: u16,
        deadline: Instant,
    ) -> io::Result<AssembledTile> {
        let mut slots: Vec<Option<(XaspTcpFrameHeader, Vec<u8>)>> = vec![None; NUM_STREAMS];
        let mut arrived_streams = 0u16;

        loop {
            for sid in 0..NUM_STREAMS {
                if slots[sid].is_some() {
                    continue;
                }
                if let Some((hdr, body)) =
                    self.readers[sid].poll_frame(&mut rx.streams[sid], &self.filter, &mut self.stats)?
                {
                    if hdr.request_id_lo == request_id_lo && hdr.layer_id == layer_id {
                        slots[sid] = Some((hdr, body));
                        arrived_streams += 1;
                    }
                }
            }

            let vip_ready = slots[0].is_some() || slots[1].is_some();
            if arrived_streams == NUM_STREAMS as u16 {
                break;
            }
            if vip_ready && Instant::now() >= deadline {
                break;
            }
            if Instant::now() >= deadline + Duration::from_millis(200) && !vip_ready {
                return Err(io::Error::new(ErrorKind::TimedOut, "stratum0 vip missed deadline"));
            }
            thread::sleep(Duration::from_micros(50));
        }

        self.filter.mark_retired(request_id_lo, layer_id);

        let (vip_hdr, vip_body) = slots[0]
            .as_ref()
            .or(slots[1].as_ref())
            .ok_or_else(|| io::Error::new(ErrorKind::TimedOut, "missing stratum0"))?;

        let is_value = (vip_hdr.flags & FLAG_IS_VALUE_TENSOR) != 0;
        let mut quant_bits = 2u8;
        for sid in 2..NUM_STREAMS {
            if let Some((hdr, _)) = &slots[sid] {
                quant_bits = hdr.quant_bits_br;
                break;
            }
        }

        let mut half: EncodedLayerHalf = decode_stratum0_payload(vip_body, quant_bits)
            .map_err(|msg| io::Error::new(ErrorKind::InvalidData, msg))?;
        let mut arrived_mask = 0u128;

        for sid in 2..NUM_STREAMS {
            if let Some((hdr, body)) = &slots[sid] {
                let (c_start, c_count) = stream_coord_range(sid).unwrap();
                unpack_bulk_stripe_into(body, &mut half.bulk_rows, c_start, c_count, hdr.quant_bits_br);
                let bits = ((1u128 << c_count) - 1) << c_start;
                arrived_mask |= bits;
            }
        }

        let tile_bf16 = decode_layer_half(&half, arrived_mask, request_id_lo, layer_id, is_value);
        let delta_hat = 1.0 - (arrived_mask.count_ones() as f32) / 128.0;

        Ok(AssembledTile {
            tile_bf16,
            arrived_streams,
            arrived_mask,
            delta_hat,
        })
    }

    pub fn drain_late_frames(&mut self, rx: &mut StripedTcpRxPool) -> io::Result<()> {
        for sid in 0..NUM_STREAMS {
            while let Some((_hdr, _body)) =
                self.readers[sid].poll_frame(&mut rx.streams[sid], &self.filter, &mut self.stats)?
            {
                // unexpected non-retired frame dropped in drain pass
            }
        }
        Ok(())
    }
}
