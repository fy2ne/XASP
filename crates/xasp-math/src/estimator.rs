// Copyright (c) 2026 fy2ne (https://xasp.fy2ne.me). All rights reserved.
// Dual-Licensed:
//   1. Noncommercial Evaluation & Academic Research: PolyForm-Noncommercial-1.0.0
//   2. Commercial / Production Deployment: Requires a paid Commercial License from fy2ne.
//      Source-available on https://github.com/fy2ne/xasp (not distributed via crates.io).

use crate::fwht::inverse_srht_128;
use crate::polar_qjl::{BulkEncodedRow, dequantize_lloyd_max};
use crate::rademacher::rademacher_mask_128;
use crate::{Bf16, f16_to_f32};

const INV_SQRT_128: f32 = 0.088_388_35;
// sqrt(pi / (2 * 128)) = sqrt(pi / 256)
const QJL_GAIN: f32 = 0.110_778_37;

pub fn reconstruct_bulk_row(
    encoded: &BulkEncodedRow,
    arrived_mask: u128,
    quant_bits: u8,
    request_id_lo: u64,
    layer_id: u16,
) -> [f32; 128] {
    let arrived_count = arrived_mask.count_ones();
    if arrived_count == 0 {
        return [0.0f32; 128];
    }

    let phi_mask = rademacher_mask_128(request_id_lo, layer_id, 0x1111_2222);
    let psi_mask = rademacher_mask_128(request_id_lo, layer_id, 0x3333_4444);

    let rescale = 128.0 / (arrived_count as f32);
    let rho_x = f16_to_f32(encoded.rho_x_f16);
    let rho_r = f16_to_f32(encoded.rho_r_f16);
    let scale_x = rho_x * INV_SQRT_128 * rescale;
    let scale_r = QJL_GAIN * rho_r * rescale;

    let mut q_hat = [0.0f32; 128];
    let mut z_psi = [0.0f32; 128];

    for lane in 0..128 {
        if ((arrived_mask >> lane) & 1) == 0 {
            continue;
        }
        q_hat[lane] = dequantize_lloyd_max(encoded.q_codes[lane], quant_bits) * scale_x;
        let sign = if ((encoded.qjl_signs >> lane) & 1) != 0 {
            1.0
        } else {
            -1.0
        };
        z_psi[lane] = sign * scale_r;
    }

    inverse_srht_128(&mut z_psi, psi_mask);
    for lane in 0..128 {
        q_hat[lane] += z_psi[lane];
    }
    inverse_srht_128(&mut q_hat, phi_mask);
    q_hat
}

pub fn write_reconstructed_row_bf16(
    dst_tile: &mut [Bf16],
    row_idx: usize,
    row_f32: &[f32; 128],
) {
    let off = row_idx * 128;
    for lane in 0..128 {
        dst_tile[off + lane] = Bf16::from_f32(row_f32[lane]);
    }
}
