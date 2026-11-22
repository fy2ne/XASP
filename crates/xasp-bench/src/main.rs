// Copyright (c) 2026 fy2ne
// SPDX-License-Identifier: Apache-2.0

mod chart_render;
mod loopback_netem;
mod staging_emu;
mod tile_gen;

use chart_render::{render_chart1, render_chart2, render_chart3};
use loopback_netem::{run_accuracy_matrix, run_live_loopback_16stream};
use staging_emu::run_staging_sweep;
use std::fs;
use std::path::PathBuf;
use tile_gen::generate_outlier_tpu_tile;

fn parse_out_dir() -> PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--out-dir" {
            if let Some(p) = args.next() {
                return PathBuf::from(p);
            }
        }
    }
    PathBuf::from("bench_out")
}

fn main() {
    let out_dir = parse_out_dir();
    let csv_dir = out_dir.join("results_csv");
    let charts_dir = out_dir.join("charts");
    fs::create_dir_all(&csv_dir).expect("create results_csv dir");
    fs::create_dir_all(&charts_dir).expect("create charts dir");

    let tile = generate_outlier_tpu_tile(512, 0x2026_1118);
    let net_stats = run_live_loopback_16stream(&tile);
    let sweep = run_staging_sweep(8, &net_stats.sample_frames);
    let acc_rows = run_accuracy_matrix(&tile);

    // Chart 1 CSV
    let mut c1 = String::from(
        "stall_prob_pct,raw_p50_ms,raw_p99_ms,raw_abort_pct,xasp_p50_ms,xasp_p99_ms,xasp_abort_pct\n",
    );
    for r in &sweep {
        c1.push_str(&format!(
            "{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}\n",
            r.stall_prob_pct,
            r.raw_p50_ms,
            r.raw_p99_ms,
            r.raw_abort_pct,
            r.xasp_p50_ms,
            r.xasp_p99_ms,
            r.xasp_abort_pct
        ));
    }
    fs::write(csv_dir.join("chart1_dcn_latency_aborts.csv"), &c1).unwrap();

    // Chart 2 CSV
    let mut c2 = String::from(
        "stall_prob_pct,num_staging_slots,raw_avg_occupied_slots,xasp_avg_occupied_slots\n",
    );
    for r in &sweep {
        c2.push_str(&format!(
            "{:.2},8,{:.3},{:.3}\n",
            r.stall_prob_pct, r.raw_avg_slots, r.xasp_avg_slots
        ));
    }
    fs::write(csv_dir.join("chart2_staging_occupancy.csv"), &c2).unwrap();

    // Chart 3 CSV
    let mut c3 = String::from(
        "scheme,compression_x,stalled_streams,inner_prod_rel_rmse,attn_kl_div,delta_ppl_proxy\n",
    );
    for r in &acc_rows {
        c3.push_str(&format!(
            "\"{}\",{:.3},{},{:.5},{:.5},{:.5}\n",
            r.scheme,
            r.compression_x,
            r.stalled_streams,
            r.inner_prod_rel_rmse,
            r.attn_kl_div,
            r.delta_ppl_proxy
        ));
    }
    fs::write(csv_dir.join("chart3_rmse_ppl_retention.csv"), &c3).unwrap();

    render_chart1(&sweep, &charts_dir.join("chart1_dcn_transfer_cliff.png")).unwrap();
    render_chart2(&sweep, 8, &charts_dir.join("chart2_staging_slot_occupancy.png")).unwrap();
    render_chart3(&acc_rows, &charts_dir.join("chart3_accuracy_rmse_compression.png")).unwrap();

    println!(
        "xasp-bench ok: cores={} cpu_encode={:.2} GB/s cpu_decode={:.2} GB/s ratio={:.2}x ({:.2} b/coord) layers={} wall={:.2}ms drained={} ({}B)",
        net_stats.cpu_cores_used,
        net_stats.encode_gb_per_sec,
        net_stats.decode_gb_per_sec,
        net_stats.wire_compression_ratio,
        net_stats.avg_bits_per_coord,
        net_stats.layers_transferred,
        net_stats.loopback_wall_ms,
        net_stats.late_drained_frames,
        net_stats.late_drained_bytes
    );
    for r in &sweep {
        println!(
            "  stall={:.1}% | raw_p99={:.1}ms raw_abort={:.1}% raw_slots={:.2}/8 | xasp_p99={:.1}ms xasp_abort={:.1}% xasp_slots={:.2}/8",
            r.stall_prob_pct,
            r.raw_p99_ms,
            r.raw_abort_pct,
            r.raw_avg_slots,
            r.xasp_p99_ms,
            r.xasp_abort_pct,
            r.xasp_avg_slots
        );
    }
    for r in &acc_rows {
        println!(
            "  {:<28} comp={:.2}x stalled={} rmse={:.2}% kl={:.4} dppl={:.4}",
            r.scheme,
            r.compression_x,
            r.stalled_streams,
            r.inner_prod_rel_rmse * 100.0,
            r.attn_kl_div,
            r.delta_ppl_proxy
        );
    }
}
