// Copyright (c) 2026 fy2ne (https://xasp.fy2ne.me). All rights reserved.
// Dual-Licensed:
//   1. Noncommercial Evaluation & Academic Research: PolyForm-Noncommercial-1.0.0
//   2. Commercial / Production Deployment: Requires a paid Commercial License from fy2ne.
//      Source-available on https://github.com/fy2ne/xasp (not distributed via crates.io).

use xasp_math::Bf16;

pub struct SyntheticKvTile {
    pub n_rows: usize,
    pub keys: Vec<Bf16>,
    pub values: Vec<Bf16>,
    pub query: [f32; 128],
}

fn next_f32(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = ((*state >> 32) as u32) as f32 / (u32::MAX as f32);
    bits * 2.0 - 1.0
}

fn next_normal_approx(state: &mut u64) -> f32 {
    // Irwin-Hall sum of 6 uniforms -> approx N(0, 0.5)
    let mut acc = 0.0f32;
    for _ in 0..6 {
        acc += next_f32(state);
    }
    acc * 0.707_106_8
}

/// Builds a [n_rows, 128] BF16 physical TPU tile featuring:
/// - 4 initial high-energy attention sink rows
/// - 5% heavy-tailed Christoffel leverage rows
/// - 5 persistent outlier minor lanes (30x channel spike across tokens)
pub fn generate_outlier_tpu_tile(n_rows: usize, seed: u64) -> SyntheticKvTile {
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut keys = Vec::with_capacity(n_rows * 128);
    let mut values = Vec::with_capacity(n_rows * 128);

    for r in 0..n_rows {
        let row_gain = if r < 4 {
            14.0
        } else if r % 28 == 0 {
            5.5
        } else {
            0.28
        };

        for c in 0..128 {
            let lane_spike = match c {
                7 | 31 | 63 | 95 | 119 => 30.0,
                _ => 1.0,
            };
            let kf = next_normal_approx(&mut s) * row_gain * lane_spike;
            let vf = next_normal_approx(&mut s) * row_gain;
            keys.push(Bf16::from_f32(kf));
            values.push(Bf16::from_f32(vf));
        }
    }

    let mut query = [0.0f32; 128];
    for c in 0..128 {
        query[c] = keys[c].to_f32() * 0.002 + next_normal_approx(&mut s) * 0.25;
    }

    SyntheticKvTile {
        n_rows,
        keys,
        values,
        query,
    }
}
