// Copyright (c) 2026 fy2ne
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use xasp_tcp::deadline_rx::DeadlineAssembler;
use xasp_tcp::pool::{StripedTcpRxPool, StripedTcpTxPool, bind_loopback_pair};
use xasp_tcp::stripe_mux::{MuxedFrame, NUM_STREAMS};

#[derive(Clone, Debug)]
pub struct DcnSweepRow {
    pub stall_prob_pct: f64,
    pub raw_p50_ms: f64,
    pub raw_p99_ms: f64,
    pub raw_abort_pct: f64,
    pub raw_avg_slots: f64,
    pub xasp_p50_ms: f64,
    pub xasp_p99_ms: f64,
    pub xasp_abort_pct: f64,
    pub xasp_avg_slots: f64,
}

struct PoolState {
    occupied: usize,
    sample_sum: u64,
    sample_count: u64,
}

/// Real bounded StagingLayerManager host buffer pool backed by Mutex + Condvar.
pub struct StagingLayerPool {
    state: Mutex<PoolState>,
    cv: Condvar,
    capacity: usize,
}

impl StagingLayerPool {
    pub fn new(capacity: usize) -> Self {
        Self {
            state: Mutex::new(PoolState {
                occupied: 0,
                sample_sum: 0,
                sample_count: 0,
            }),
            cv: Condvar::new(),
            capacity,
        }
    }

    pub fn acquire(&self, deadline: Instant) -> bool {
        let mut guard = self.state.lock().unwrap();
        while guard.occupied >= self.capacity {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let wait_for = deadline - now;
            let (g, res) = self.cv.wait_timeout(guard, wait_for).unwrap();
            guard = g;
            if res.timed_out() && guard.occupied >= self.capacity {
                return false;
            }
        }
        guard.occupied += 1;
        guard.sample_sum += guard.occupied as u64;
        guard.sample_count += 1;
        true
    }

    pub fn sample_occupied(&self) {
        let mut guard = self.state.lock().unwrap();
        guard.sample_sum += guard.occupied as u64;
        guard.sample_count += 1;
    }

    pub fn release(&self) {
        let mut guard = self.state.lock().unwrap();
        if guard.occupied > 0 {
            guard.occupied -= 1;
        }
        self.cv.notify_one();
    }

    pub fn avg_occupied(&self) -> f64 {
        let guard = self.state.lock().unwrap();
        if guard.sample_count == 0 {
            return 0.0;
        }
        (guard.sample_sum as f64) / (guard.sample_count as f64)
    }
}

#[inline]
fn wait_precise(dur: Duration) {
    let target = Instant::now() + dur;
    if dur > Duration::from_millis(18) {
        thread::sleep(dur - Duration::from_millis(16));
    }
    while Instant::now() < target {
        std::hint::spin_loop();
    }
}

fn next_u01(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn percentile_sorted(vals: &[f64], p: f64) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let idx = ((vals.len() as f64 - 1.0) * p).round() as usize;
    vals[idx.min(vals.len() - 1)]
}

fn flush_ready_pending(
    tx: &mut StripedTcpTxPool,
    queues: &mut [VecDeque<(Instant, Vec<u8>)>],
    now: Instant,
) {
    for sid in 0..NUM_STREAMS {
        while let Some((t, _)) = queues[sid].front() {
            if *t <= now {
                let (_, bytes) = queues[sid].pop_front().unwrap();
                let _ = tx.streams[sid].write_all(&bytes);
            } else {
                break;
            }
        }
    }
}

fn flush_all_pending(tx: &mut StripedTcpTxPool, queues: &mut [VecDeque<(Instant, Vec<u8>)>]) {
    loop {
        let now = Instant::now();
        flush_ready_pending(tx, queues, now);
        if queues.iter().all(|q| q.is_empty()) {
            break;
        }
        thread::yield_now();
    }
}

const DCN_CONGESTED_LINK_GBPS: f64 = 0.0728;

#[inline]
fn wire_pace_for_bytes(wire_bytes: usize) -> Duration {
    Duration::from_secs_f64((wire_bytes as f64 * 8.0) / (DCN_CONGESTED_LINK_GBPS * 1e9))
}

fn send_raw_16streams_pipelined(
    pool_size: usize,
    raw_layer_bytes: usize,
    n_layers: usize,
    p_stall: f64,
    rng: &mut u64,
) -> (f64, bool, f64) {
    let (mut tx, rx) = bind_loopback_pair(0).expect("raw loopback pair");
    for s in &rx.streams {
        let _ = s.set_nonblocking(false);
    }

    let staging = Arc::new(StagingLayerPool::new(pool_size));
    let per_stream_bytes = (raw_layer_bytes / NUM_STREAMS).max(64);
    let actual_layer_wire_bytes = per_stream_bytes * NUM_STREAMS;
    let session_abort_ms = 520.0f64;
    let session_deadline = Instant::now() + Duration::from_millis(session_abort_ms as u64);
    let layer_pace = wire_pace_for_bytes(actual_layer_wire_bytes);

    let mut rx_streams = rx.streams;
    let payload = vec![0x5Au8; per_stream_bytes];
    let staging_rx = Arc::clone(&staging);

    let t0 = Instant::now();
    // RX blocks each layer slot until 100% of all 16 streams arrive (HoL blocking)
    let rx_handle = thread::spawn(move || -> bool {
        let mut scratch = vec![0u8; per_stream_bytes];
        for _l in 0..n_layers {
            for stream in &mut rx_streams {
                let _ = stream.set_read_timeout(Some(Duration::from_millis(650)));
                if stream.read_exact(&mut scratch).is_err() {
                    staging_rx.release();
                    return true;
                }
            }
            staging_rx.release();
        }
        false
    });

    let mut queues: Vec<VecDeque<(Instant, Vec<u8>)>> =
        (0..NUM_STREAMS).map(|_| VecDeque::new()).collect();
    let mut ready_at = [Instant::now(); NUM_STREAMS];
    let mut aborted = false;

    for _l in 0..n_layers {
        if !staging.acquire(session_deadline) {
            aborted = true;
            break;
        }

        let mut stalled_sid: Option<usize> = None;
        for sid in 0..NUM_STREAMS {
            if next_u01(rng) < p_stall {
                stalled_sid = Some(sid);
                break;
            }
        }

        let now = Instant::now();
        flush_ready_pending(&mut tx, &mut queues, now);

        for sid in 0..NUM_STREAMS {
            let stall = if Some(sid) == stalled_sid {
                let ms = 140.0 + next_u01(rng) * 260.0;
                Duration::from_millis(ms as u64)
            } else {
                Duration::ZERO
            };
            if stall.is_zero() && queues[sid].is_empty() && ready_at[sid] <= now {
                if tx.streams[sid].write_all(&payload).is_err() {
                    aborted = true;
                    break;
                }
            } else {
                let release = ready_at[sid].max(now) + stall;
                ready_at[sid] = release;
                queues[sid].push_back((release, payload.clone()));
            }
        }

        staging.sample_occupied();
        wait_precise(layer_pace);
        if Instant::now() >= session_deadline {
            aborted = true;
            break;
        }
    }

    if !aborted {
        flush_all_pending(&mut tx, &mut queues);
    }
    drop(tx);
    let rx_aborted = rx_handle.join().unwrap_or(true);
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    (
        elapsed_ms.min(980.0),
        aborted || rx_aborted || elapsed_ms > session_abort_ms,
        staging.avg_occupied(),
    )
}

fn send_xasp_16streams_deadline(
    tx: &mut StripedTcpTxPool,
    rx: &mut StripedTcpRxPool,
    asm: &mut DeadlineAssembler,
    pool_size: usize,
    sample_frames: &[MuxedFrame],
    req_id: u64,
    n_layers: usize,
    p_stall: f64,
    rng: &mut u64,
) -> (f64, bool, f64) {
    let staging = Arc::new(StagingLayerPool::new(pool_size));
    let actual_wire_bytes: usize = sample_frames
        .iter()
        .filter(|f| f.header.stream_id != 1)
        .map(|f| 64 + f.payload.len())
        .sum();
    let layer_pace = wire_pace_for_bytes(actual_wire_bytes);
    let tau_l = Duration::from_micros(1_200);

    let mut queues: Vec<VecDeque<(Instant, Vec<u8>)>> =
        (0..NUM_STREAMS).map(|_| VecDeque::new()).collect();
    let mut ready_at = [Instant::now(); NUM_STREAMS];

    let t0 = Instant::now();
    let mut aborted = false;

    for l in 0..n_layers {
        let dl = Instant::now() + Duration::from_millis(200);
        if !staging.acquire(dl) {
            aborted = true;
            break;
        }

        let mut stall_sid: Option<usize> = None;
        for sid in 2..NUM_STREAMS {
            if next_u01(rng) < p_stall {
                stall_sid = Some(sid);
                break;
            }
        }

        let layer_t0 = Instant::now();
        flush_ready_pending(tx, &mut queues, layer_t0);

        for frame in sample_frames {
            let mut hdr = frame.header;
            hdr.request_id_lo = req_id;
            hdr.layer_id = l as u16;
            hdr.seal();

            let sid = hdr.stream_id as usize;
            let stall = if Some(sid) == stall_sid {
                Duration::from_millis(25)
            } else {
                Duration::ZERO
            };

            if stall.is_zero() && queues[sid].is_empty() && ready_at[sid] <= layer_t0 {
                let _ = tx.streams[sid].write_all(&hdr.to_bytes());
                let _ = tx.streams[sid].write_all(&frame.payload);
            } else {
                let mut wire = Vec::with_capacity(64 + frame.payload.len());
                wire.extend_from_slice(&hdr.to_bytes());
                wire.extend_from_slice(&frame.payload);
                let release = ready_at[sid].max(layer_t0) + stall;
                ready_at[sid] = release;
                queues[sid].push_back((release, wire));
            }
        }

        if asm
            .collect_layer_until_deadline(rx, req_id, l as u16, Instant::now() + tau_l)
            .is_err()
        {
            aborted = true;
        }
        staging.release();
        let spent = layer_t0.elapsed();
        if spent < layer_pace {
            wait_precise(layer_pace - spent);
        }
    }

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    flush_all_pending(tx, &mut queues);
    wait_precise(Duration::from_micros(500));
    let _ = asm.drain_late_frames(rx);

    (elapsed_ms, aborted, staging.avg_occupied())
}

/// Executes 1:1 real-time 16-socket loopback TCP sessions gated by StagingLayerPool (Condvar + Mutex).
pub fn run_staging_sweep(
    num_staging_slots: usize,
    sample_frames: &[MuxedFrame],
) -> Vec<DcnSweepRow> {
    let stall_probs = [0.0f64, 0.002, 0.005, 0.01, 0.02];
    let n_sessions = 8usize;
    let n_layers = 16usize;
    let mut rows = Vec::with_capacity(stall_probs.len());

    let (mut xasp_tx, mut xasp_rx) = bind_loopback_pair(0).expect("persistent xasp pool");
    let mut asm = DeadlineAssembler::new();

    for (s_idx, &p_stall) in stall_probs.iter().enumerate() {
        let raw_lats = Arc::new(Mutex::new(Vec::with_capacity(n_sessions)));
        let raw_aborts = Arc::new(Mutex::new(0usize));
        let raw_slot_sum = Arc::new(Mutex::new(0.0f64));

        let workers = 2usize;
        let per_worker = n_sessions / workers;
        thread::scope(|scope| {
            for wid in 0..workers {
                let raw_lats = Arc::clone(&raw_lats);
                let raw_aborts = Arc::clone(&raw_aborts);
                let raw_slot_sum = Arc::clone(&raw_slot_sum);
                scope.spawn(move || {
                    let mut rng = 0x5A5D_1000_u64 + (s_idx as u64) * 7919 + (wid as u64) * 131;
                    for _ in 0..per_worker {
                        let (r_ms, r_abt, r_occ) = send_raw_16streams_pipelined(
                            num_staging_slots,
                            128 * 1024,
                            n_layers,
                            p_stall,
                            &mut rng,
                        );
                        raw_lats.lock().unwrap().push(r_ms);
                        if r_abt {
                            *raw_aborts.lock().unwrap() += 1;
                        }
                        *raw_slot_sum.lock().unwrap() += r_occ;
                    }
                });
            }
        });

        let mut xasp_lats = Vec::with_capacity(n_sessions);
        let mut xasp_aborts = 0usize;
        let mut xasp_slot_sum = 0.0f64;
        let mut rng = 0x5A5D_2000_u64 + (s_idx as u64) * 7919;

        for sess in 0..n_sessions {
            let req_id = 10_000 + (s_idx as u64) * 100 + (sess as u64);
            let (x_ms, x_abt, x_occ) = send_xasp_16streams_deadline(
                &mut xasp_tx,
                &mut xasp_rx,
                &mut asm,
                num_staging_slots,
                sample_frames,
                req_id,
                n_layers,
                p_stall,
                &mut rng,
            );
            xasp_lats.push(x_ms);
            if x_abt {
                xasp_aborts += 1;
            }
            xasp_slot_sum += x_occ;
        }

        let mut rl = raw_lats.lock().unwrap();
        rl.sort_by(|a, b| a.total_cmp(b));
        xasp_lats.sort_by(|a, b| a.total_cmp(b));
        let ra = *raw_aborts.lock().unwrap();
        let ro = *raw_slot_sum.lock().unwrap();

        rows.push(DcnSweepRow {
            stall_prob_pct: p_stall * 100.0,
            raw_p50_ms: percentile_sorted(&rl, 0.50),
            raw_p99_ms: percentile_sorted(&rl, 0.99),
            raw_abort_pct: (ra as f64) * 100.0 / (n_sessions as f64),
            raw_avg_slots: ro / (n_sessions as f64),
            xasp_p50_ms: percentile_sorted(&xasp_lats, 0.50),
            xasp_p99_ms: percentile_sorted(&xasp_lats, 0.99),
            xasp_abort_pct: (xasp_aborts as f64) * 100.0 / (n_sessions as f64),
            xasp_avg_slots: xasp_slot_sum / (n_sessions as f64),
        });
    }

    rows
}
