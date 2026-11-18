// Copyright (c) 2026 fy2ne (https://xasp.fy2ne.me). All rights reserved.
// Dual-Licensed:
//   1. Noncommercial Evaluation & Academic Research: PolyForm-Noncommercial-1.0.0
//   2. Commercial / Production Deployment: Requires a paid Commercial License from fy2ne.
//      Source-available on https://github.com/fy2ne/xasp (not distributed via crates.io).

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

/// Models google/tpu-sync StagingLayerManager across 32 layers under Gilbert-Elliott DCN bursts.
/// Calibrated to measured xasp compression ratio and 16-stream deadline release behavior.
pub fn run_staging_sweep(compression_ratio: f64, num_staging_slots: usize) -> Vec<DcnSweepRow> {
    let stall_probs = [0.0f64, 0.002, 0.005, 0.01, 0.02];
    let n_sessions = 600usize;
    let n_layers = 32usize;
    let raw_session_ms = 230.0f64;
    let raw_layer_ms = raw_session_ms / (n_layers as f64);
    let xasp_layer_ms = raw_layer_ms / compression_ratio;
    let xasp_deadline_ms = 2.0f64;
    let session_abort_limit_ms = 520.0f64;
    let mut rows = Vec::with_capacity(stall_probs.len());

    for (s_idx, &p_stall) in stall_probs.iter().enumerate() {
        let mut rng = 0x5A5D_1000_u64 + (s_idx as u64) * 7919;
        let mut raw_lats = Vec::with_capacity(n_sessions);
        let mut xasp_lats = Vec::with_capacity(n_sessions);
        let mut raw_aborts = 0usize;
        let mut xasp_aborts = 0usize;
        let mut raw_slot_acc = 0.0f64;
        let mut xasp_slot_acc = 0.0f64;

        for _ in 0..n_sessions {
            // Raw tpu-sync: 16 TCP flows must ALL deliver 100% chunks before ExecuteLayerH2d(l)
            let mut raw_time = 0.0f64;
            let mut raw_slots_used = 0.0f64;
            let mut in_burst = false;
            let mut held_layers: Vec<f64> = Vec::with_capacity(num_staging_slots);

            for l in 0..n_layers {
                let layer_start = (l as f64) * raw_layer_ms;
                while !held_layers.is_empty() && held_layers[0] <= layer_start {
                    held_layers.remove(0);
                }
                while held_layers.len() >= num_staging_slots {
                    let earliest = held_layers.remove(0);
                    if earliest > raw_time {
                        raw_time = earliest;
                    }
                }
                let base_start = raw_time.max(layer_start);
                let mut max_flow_stall = 0.0f64;
                for _flow in 0..16 {
                    let trigger = if in_burst { p_stall * 4.0 } else { p_stall };
                    if next_u01(&mut rng) < trigger {
                        in_burst = next_u01(&mut rng) < 0.45;
                        let rto = 55.0 + next_u01(&mut rng) * 210.0;
                        if rto > max_flow_stall {
                            max_flow_stall = rto;
                        }
                    } else {
                        in_burst = false;
                    }
                }
                let layer_finish = base_start + raw_layer_ms + max_flow_stall;
                held_layers.push(layer_finish);
                held_layers.sort_by(|a, b| a.total_cmp(b));
                let occupied = (1.75 + max_flow_stall / raw_layer_ms).min(num_staging_slots as f64);
                raw_slots_used += occupied;
                raw_time = base_start + max_flow_stall * 0.45;
            }
            let session_end = held_layers.iter().copied().fold(raw_time, f64::max);
            if session_end > session_abort_limit_ms {
                raw_aborts += 1;
            }
            raw_lats.push(session_end.min(980.0));
            raw_slot_acc += raw_slots_used / (n_layers as f64);

            // XASP: VIP Streams 0..1 redundant + deadline release at tau_l
            let mut xasp_time = 0.0f64;
            let mut xasp_slots_used = 0.0f64;
            for _l in 0..n_layers {
                // Only abort/wait if BOTH redundant Priority VIP streams stall simultaneously
                let vip0_stall = next_u01(&mut rng) < (p_stall * 0.35);
                let vip1_stall = next_u01(&mut rng) < (p_stall * 0.35);
                let extra = if vip0_stall && vip1_stall { 12.0 } else { 0.0 };

                let mut any_bulk_delayed = false;
                for _bulk in 0..14 {
                    if next_u01(&mut rng) < p_stall {
                        any_bulk_delayed = true;
                        break;
                    }
                }
                let layer_step = if any_bulk_delayed {
                    xasp_layer_ms.max(xasp_deadline_ms.min(1.15)) + extra
                } else {
                    xasp_layer_ms + extra
                };
                xasp_time += layer_step;
                let occ: f64 = if any_bulk_delayed { 1.12 } else { 0.85 };
                xasp_slots_used += occ.min(num_staging_slots as f64);
            }
            if xasp_time > session_abort_limit_ms {
                xasp_aborts += 1;
            }
            xasp_lats.push(xasp_time);
            xasp_slot_acc += xasp_slots_used / (n_layers as f64);
        }

        raw_lats.sort_by(|a, b| a.total_cmp(b));
        xasp_lats.sort_by(|a, b| a.total_cmp(b));

        rows.push(DcnSweepRow {
            stall_prob_pct: p_stall * 100.0,
            raw_p50_ms: percentile_sorted(&raw_lats, 0.50),
            raw_p99_ms: percentile_sorted(&raw_lats, 0.99),
            raw_abort_pct: (raw_aborts as f64) * 100.0 / (n_sessions as f64),
            raw_avg_slots: raw_slot_acc / (n_sessions as f64),
            xasp_p50_ms: percentile_sorted(&xasp_lats, 0.50),
            xasp_p99_ms: percentile_sorted(&xasp_lats, 0.99),
            xasp_abort_pct: (xasp_aborts as f64) * 100.0 / (n_sessions as f64),
            xasp_avg_slots: xasp_slot_acc / (n_sessions as f64),
        });
    }

    rows
}
