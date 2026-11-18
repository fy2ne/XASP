// Copyright (c) 2026 fy2ne (https://xasp.fy2ne.me). All rights reserved.
// Dual-Licensed:
//   1. Noncommercial Evaluation & Academic Research: PolyForm-Noncommercial-1.0.0
//   2. Commercial / Production Deployment: Requires a paid Commercial License from fy2ne.
//      Source-available on https://github.com/fy2ne/xasp (not distributed via crates.io).

use crate::tile_gen::SyntheticKvTile;
use std::collections::VecDeque;
use std::io::Write;
use std::time::{Duration, Instant};
use xasp_math::polar_qjl::encode_bulk_row;
use xasp_math::{
    Bf16, EncodedLayerHalf, LocalSpectralCodec, TileTransportBackend, decode_layer_half,
    decode_layer_half_into,
};
use xasp_tcp::deadline_rx::DeadlineAssembler;
use xasp_tcp::pool::bind_loopback_pair;
use xasp_tcp::stripe_mux::{MuxedFrame, NUM_STREAMS, mux_encoded_half};

#[derive(Clone, Debug)]
pub struct LoopbackPerfStats {
    pub cpu_cores_used: usize,
    pub encode_gb_per_sec: f64,
    pub decode_gb_per_sec: f64,
    pub wire_compression_ratio: f64,
    pub avg_bits_per_coord: f64,
    pub layers_transferred: usize,
    pub loopback_wall_ms: f64,
    pub late_drained_frames: u64,
    pub late_drained_bytes: u64,
    pub sample_frames: Vec<MuxedFrame>,
}

#[derive(Clone, Debug)]
pub struct AccuracyBenchRow {
    pub scheme: String,
    pub compression_x: f64,
    pub stalled_streams: usize,
    pub inner_prod_rel_rmse: f64,
    pub attn_kl_div: f64,
    pub delta_ppl_proxy: f64,
}

pub fn measure_cpu_throughput_gbps(tile: &SyntheticKvTile) -> (usize, f64, f64) {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(4);
    let iters_per_thread = 256usize;
    let raw_bytes_per_iter = tile.n_rows * 128 * 2 * 2;
    let total_bytes = (cores * iters_per_thread * raw_bytes_per_iter) as f64;

    let t0 = Instant::now();
    std::thread::scope(|scope| {
        for tid in 0..cores {
            scope.spawn(move || {
                let codec = LocalSpectralCodec::default();
                let mut k_out = EncodedLayerHalf::default();
                let mut v_out = EncodedLayerHalf::default();
                for iter in 0..iters_per_thread {
                    codec.encode_kv_layer_into(
                        &tile.keys,
                        &tile.values,
                        tile.n_rows,
                        2,
                        1000 + (tid as u64),
                        iter as u16,
                        &mut k_out,
                        &mut v_out,
                    );
                }
            });
        }
    });
    let enc_elapsed = t0.elapsed().as_secs_f64().max(1e-6);

    let codec = LocalSpectralCodec::default();
    let (sample_k, sample_v) =
        codec.encode_kv_layer(&tile.keys, &tile.values, tile.n_rows, 2, 1000, 0);
    let mask_1_stall = u128::MAX ^ (((1u128 << 9) - 1) << 45);

    let t1 = Instant::now();
    std::thread::scope(|scope| {
        for tid in 0..cores {
            let sk = &sample_k;
            let sv = &sample_v;
            scope.spawn(move || {
                let mut k_dst = vec![Bf16::default(); tile.n_rows * 128];
                let mut v_dst = vec![Bf16::default(); tile.n_rows * 128];
                for iter in 0..iters_per_thread {
                    decode_layer_half_into(
                        sk,
                        mask_1_stall,
                        1000 + (tid as u64),
                        iter as u16,
                        false,
                        &mut k_dst,
                    );
                    decode_layer_half_into(
                        sv,
                        mask_1_stall,
                        1000 + (tid as u64),
                        iter as u16,
                        true,
                        &mut v_dst,
                    );
                }
            });
        }
    });
    let dec_elapsed = t1.elapsed().as_secs_f64().max(1e-6);
    (cores, total_bytes / enc_elapsed / 1e9, total_bytes / dec_elapsed / 1e9)
}

pub fn run_live_loopback_16stream(tile: &SyntheticKvTile) -> LoopbackPerfStats {
    let (cores, encode_gbps, decode_gbps) = measure_cpu_throughput_gbps(tile);
    let (mut tx, mut rx) = bind_loopback_pair(0).expect("bind 16 loopback tcp streams");
    let mut asm = DeadlineAssembler::new();
    let codec = LocalSpectralCodec::default();
    let n_layers = 16usize;

    let (sample_k, _) = codec.encode_kv_layer(&tile.keys, &tile.values, tile.n_rows, 2, 404, 0);
    let sample_frames = mux_encoded_half(&sample_k, false, 404, 0, 0, 0, 0);
    let mut single_wire_bytes = 0usize;
    for f in &sample_frames {
        if f.header.stream_id != 1 {
            single_wire_bytes += f.payload.len();
        }
    }
    let raw_bytes = tile.n_rows * 128 * 2;
    let compression_ratio = (raw_bytes as f64) / (single_wire_bytes as f64);
    let avg_bits_per_coord = 16.0 / compression_ratio;

    let t0 = Instant::now();
    let mut queues: Vec<VecDeque<(Instant, Vec<u8>)>> =
        (0..NUM_STREAMS).map(|_| VecDeque::new()).collect();
    let mut ready_at = [Instant::now(); NUM_STREAMS];

    for l in 0..n_layers {
        let frames = mux_encoded_half(&sample_k, false, 404, 0, l as u16, (l % 4) as u8, 1_000_000);
        let stall_sid = if l % 4 == 1 { Some(7usize) } else { None };
        let now = Instant::now();

        for sid in 0..NUM_STREAMS {
            while let Some((t, _)) = queues[sid].front() {
                if *t <= now {
                    let (_, b) = queues[sid].pop_front().unwrap();
                    tx.streams[sid].write_all(&b).unwrap();
                } else {
                    break;
                }
            }
        }

        for frame in frames {
            let sid = frame.header.stream_id as usize;
            let stall = if Some(sid) == stall_sid {
                Duration::from_millis(8)
            } else {
                Duration::ZERO
            };
            if stall.is_zero() && queues[sid].is_empty() && ready_at[sid] <= now {
                tx.streams[sid].write_all(&frame.header.to_bytes()).unwrap();
                tx.streams[sid].write_all(&frame.payload).unwrap();
            } else {
                let mut wire = Vec::with_capacity(64 + frame.payload.len());
                wire.extend_from_slice(&frame.header.to_bytes());
                wire.extend_from_slice(&frame.payload);
                let release = ready_at[sid].max(now) + stall;
                ready_at[sid] = release;
                queues[sid].push_back((release, wire));
            }
        }

        let _assembled = asm
            .collect_layer_until_deadline(
                &mut rx,
                404,
                l as u16,
                Instant::now() + Duration::from_millis(2),
            )
            .unwrap();
    }

    loop {
        let now = Instant::now();
        let mut any = false;
        for sid in 0..NUM_STREAMS {
            while let Some((t, _)) = queues[sid].front() {
                if *t <= now {
                    let (_, b) = queues[sid].pop_front().unwrap();
                    tx.streams[sid].write_all(&b).unwrap();
                } else {
                    any = true;
                    break;
                }
            }
        }
        if !any {
            break;
        }
        std::thread::yield_now();
    }
    std::thread::sleep(Duration::from_millis(1));
    asm.drain_late_frames(&mut rx).unwrap();
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    LoopbackPerfStats {
        cpu_cores_used: cores,
        encode_gb_per_sec: encode_gbps,
        decode_gb_per_sec: decode_gbps,
        wire_compression_ratio: compression_ratio,
        avg_bits_per_coord,
        layers_transferred: n_layers,
        loopback_wall_ms: wall_ms,
        late_drained_frames: asm.stats.late_drained_frames,
        late_drained_bytes: asm.stats.late_drained_bytes,
        sample_frames,
    }
}

fn eval_metrics(
    tile: &SyntheticKvTile,
    k_hat: &[Bf16],
    v_hat: &[Bf16],
) -> (f64, f64, f64) {
    let n = tile.n_rows;
    let n_queries = 16usize;
    let mut sum_rel_rmse = 0.0f64;
    let mut sum_kl = 0.0f64;

    for q_idx in 0..n_queries {
        let r_vip = (q_idx * 28) % n;
        let r_bulk = (q_idx * 31 + 5) % n;
        let mut q = [0.0f64; 128];
        let mut q_norm_sq = 0.0f64;
        for c in 0..128 {
            let v1 = tile.keys[r_vip * 128 + c].to_f32() as f64;
            let v2 = tile.keys[r_bulk * 128 + c].to_f32() as f64;
            let g = tile.query[(c + q_idx * 7) % 128] as f64;
            let val_q = v1 * 0.22 + v2 * 0.78 + g * 0.4;
            q[c] = val_q;
            q_norm_sq += val_q * val_q;
        }
        let inv_q_norm = 1.0 / q_norm_sq.sqrt().max(1e-12);
        for c in 0..128 {
            q[c] *= inv_q_norm;
        }

        let mut logits_true = vec![0.0f64; n];
        let mut logits_hat = vec![0.0f64; n];
        for r in 0..n {
            let mut dt = 0.0f64;
            let mut dh = 0.0f64;
            for c in 0..128 {
                dt += q[c] * (tile.keys[r * 128 + c].to_f32() as f64);
                dh += q[c] * (k_hat[r * 128 + c].to_f32() as f64);
            }
            logits_true[r] = dt;
            logits_hat[r] = dh;
        }

        let mut sorted = logits_true.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let max_t = sorted[n - 1];
        let med_t = sorted[n / 2];
        // temperature scales max - median logit to 3.6 so Stratum-1 holds ~12-15% softmax mass
        let temp = ((max_t - med_t) / 3.6).max(1e-4);

        let mut max_h = f64::NEG_INFINITY;
        for r in 0..n {
            logits_true[r] /= temp;
            logits_hat[r] /= temp;
            if logits_hat[r] > max_h {
                max_h = logits_hat[r];
            }
        }
        let scaled_max_t = max_t / temp;

        let mut p_t = vec![0.0f64; n];
        let mut p_h = vec![0.0f64; n];
        let mut sum_t = 0.0f64;
        let mut sum_h = 0.0f64;
        for r in 0..n {
            p_t[r] = (logits_true[r] - scaled_max_t).clamp(-40.0, 0.0).exp();
            p_h[r] = (logits_hat[r] - max_h).clamp(-40.0, 0.0).exp();
            sum_t += p_t[r];
            sum_h += p_h[r];
        }

        let mut kl = 0.0f64;
        let mut out_t = [0.0f64; 128];
        let mut out_h = [0.0f64; 128];
        for r in 0..n {
            let pt = p_t[r] / sum_t.max(1e-12);
            let ph = p_h[r] / sum_h.max(1e-12);
            if pt > 1e-12 {
                kl += pt * (pt / ph.max(1e-12)).ln();
            }
            for c in 0..128 {
                out_t[c] += pt * (tile.values[r * 128 + c].to_f32() as f64);
                out_h[c] += ph * (v_hat[r * 128 + c].to_f32() as f64);
            }
        }

        let mut out_diff_sq = 0.0f64;
        let mut out_ref_sq = 0.0f64;
        for c in 0..128 {
            let d = out_t[c] - out_h[c];
            out_diff_sq += d * d;
            out_ref_sq += out_t[c] * out_t[c];
        }
        sum_rel_rmse += (out_diff_sq / out_ref_sq.max(1e-12)).sqrt();
        sum_kl += kl;
    }

    let avg_rmse = sum_rel_rmse / (n_queries as f64);
    let avg_kl = sum_kl / (n_queries as f64);
    let delta_ppl_proxy = (avg_kl * 0.35).min(4.5);
    (avg_rmse, avg_kl, delta_ppl_proxy)
}

fn unstratified_uniform_quant(src: &[Bf16], n_rows: usize, bits: u8) -> Vec<Bf16> {
    let levels = ((1u32 << bits) - 1) as f32;
    let mut out = vec![Bf16::default(); n_rows * 128];
    for r in 0..n_rows {
        let off = r * 128;
        let mut amax = 1e-6f32;
        for c in 0..128 {
            amax = amax.max(src[off + c].to_f32().abs());
        }
        let step = (2.0 * amax) / levels;
        for c in 0..128 {
            let x = src[off + c].to_f32();
            let q = ((x + amax) / step).round().clamp(0.0, levels);
            out[off + c] = Bf16::from_f32(q * step - amax);
        }
    }
    out
}

fn naive_byte_stripe_drop(src: &[Bf16], n_rows: usize, stalled_mask: u128) -> Vec<Bf16> {
    let mut out = vec![Bf16::default(); n_rows * 128];
    for r in 0..n_rows {
        let off = r * 128;
        for c in 0..128 {
            if ((stalled_mask >> c) & 1) != 0 {
                out[off + c] = src[off + c];
            }
        }
    }
    out
}

pub fn run_accuracy_matrix(tile: &SyntheticKvTile) -> Vec<AccuracyBenchRow> {
    let mut rows = Vec::new();
    let mask_1_stall = u128::MAX ^ (((1u128 << 9) - 1) << 45);
    let mask_2_stall = mask_1_stall ^ (((1u128 << 9) - 1) << 99);

    let configs = [
        (4u8, 0.05f32, 4usize, 0usize, u128::MAX),
        (3u8, 0.05f32, 4usize, 1usize, mask_1_stall),
        (2u8, 0.05f32, 4usize, 1usize, mask_1_stall),
        (2u8, 0.05f32, 4usize, 2usize, mask_2_stall),
        (2u8, 0.025f32, 4usize, 1usize, mask_1_stall),
    ];

    let raw_bytes = (tile.n_rows * 128 * 2) as f64;
    for &(bits, vip_ratio, sinks, n_stalled, arrived_mask) in &configs {
        let codec = LocalSpectralCodec {
            vip_ratio,
            sink_rows: sinks,
        };
        let (k_enc, v_enc) =
            codec.encode_kv_layer(&tile.keys, &tile.values, tile.n_rows, bits, 404, 1);
        let frames = mux_encoded_half(&k_enc, false, 404, 0, 1, 0, 0);
        let wire: usize = frames
            .iter()
            .filter(|f| f.header.stream_id != 1)
            .map(|f| f.payload.len())
            .sum();
        let comp_x = raw_bytes / (wire as f64);

        let k_hat = decode_layer_half(&k_enc, arrived_mask, 404, 1, false);
        let v_hat = decode_layer_half(&v_enc, arrived_mask, 404, 1, true);
        let (rmse, kl, ppl) = eval_metrics(tile, &k_hat, &v_hat);
        rows.push(AccuracyBenchRow {
            scheme: format!("XASP (b={bits}, VIP={:.1}%)", vip_ratio * 100.0),
            compression_x: comp_x,
            stalled_streams: n_stalled,
            inner_prod_rel_rmse: rmse,
            attn_kl_div: kl,
            delta_ppl_proxy: ppl,
        });
    }

    // QJL-Only (No VIP anchors) across b=4,3,2 with 1 stalled stream
    for &bits in &[4u8, 3, 2] {
        let mut bulk_k = Vec::with_capacity(tile.n_rows);
        let mut bulk_v = Vec::with_capacity(tile.n_rows);
        for r in 0..tile.n_rows {
            let mut kf = [0.0f32; 128];
            let mut vf = [0.0f32; 128];
            for c in 0..128 {
                kf[c] = tile.keys[r * 128 + c].to_f32();
                vf[c] = tile.values[r * 128 + c].to_f32();
            }
            bulk_k.push(encode_bulk_row(r as u16, &kf, bits, 404, 1));
            bulk_v.push(encode_bulk_row(r as u16, &vf, bits, 404 ^ 0xA5A5_5A5A, 1));
        }
        let k_enc = xasp_math::EncodedLayerHalf {
            n_rows: tile.n_rows,
            quant_bits: bits,
            vip_rows: Vec::new(),
            bulk_rows: bulk_k,
        };
        let v_enc = xasp_math::EncodedLayerHalf {
            n_rows: tile.n_rows,
            quant_bits: bits,
            vip_rows: Vec::new(),
            bulk_rows: bulk_v,
        };
        let k_hat = decode_layer_half(&k_enc, mask_1_stall, 404, 1, false);
        let v_hat = decode_layer_half(&v_enc, mask_1_stall, 404, 1, true);
        let (rmse, kl, ppl) = eval_metrics(tile, &k_hat, &v_hat);
        let comp_x = 16.0 / (bits as f64 + 0.25);
        rows.push(AccuracyBenchRow {
            scheme: format!("QJL-Only (No VIP, b={bits})"),
            compression_x: comp_x,
            stalled_streams: 1,
            inner_prod_rel_rmse: rmse,
            attn_kl_div: kl,
            delta_ppl_proxy: ppl,
        });
    }

    // Naive Byte Stripe (drops unrotated coordinates when 1 or 2 streams stall)
    for (n_stalled, mask) in [(1usize, mask_1_stall), (2usize, mask_2_stall)] {
        let k_hat = naive_byte_stripe_drop(&tile.keys, tile.n_rows, mask);
        let v_hat = naive_byte_stripe_drop(&tile.values, tile.n_rows, mask);
        let (rmse, kl, ppl) = eval_metrics(tile, &k_hat, &v_hat);
        rows.push(AccuracyBenchRow {
            scheme: "Naive Byte Stripe".to_string(),
            compression_x: if n_stalled == 1 { 2.0 } else { 4.0 },
            stalled_streams: n_stalled,
            inner_prod_rel_rmse: rmse,
            attn_kl_div: kl,
            delta_ppl_proxy: ppl,
        });
    }

    // Unstratified scalar INT4 / INT3 / INT2
    for &bits in &[4u8, 3, 2] {
        let k_hat = unstratified_uniform_quant(&tile.keys, tile.n_rows, bits);
        let v_hat = unstratified_uniform_quant(&tile.values, tile.n_rows, bits);
        let (rmse, kl, ppl) = eval_metrics(tile, &k_hat, &v_hat);
        rows.push(AccuracyBenchRow {
            scheme: "Unstratified Scalar INT".to_string(),
            compression_x: 16.0 / (bits as f64),
            stalled_streams: 1,
            inner_prod_rel_rmse: rmse,
            attn_kl_div: kl,
            delta_ppl_proxy: ppl,
        });
    }

    rows
}
