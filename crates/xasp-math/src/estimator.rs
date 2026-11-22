// Copyright (c) 2026 fy2ne
// SPDX-License-Identifier: Apache-2.0

use crate::fwht::{
    INV_SQRT_128, fwht_128_regs_avx2, has_avx2, inverse_srht_128, store_bf16_row_regs_avx2,
};
use crate::polar_qjl::{BulkEncodedRow, C1_MAG, dequantize_lloyd_max};
use crate::rademacher::{build_avx2_sign_vecs, rademacher_mask_128};
use crate::{Bf16, f16_to_f32};

// sqrt(pi / (2 * 128)) = sqrt(pi / 256)
const QJL_GAIN: f32 = 0.110_778_37;

const fn build_sign_lut() -> [[u32; 8]; 256] {
    let mut tbl = [[0u32; 8]; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut k = 0usize;
        while k < 8 {
            tbl[b][k] = if ((b >> k) & 1) != 0 {
                0
            } else {
                0x8000_0000
            };
            k += 1;
        }
        b += 1;
    }
    tbl
}

const fn build_arrived_lut() -> [[u32; 8]; 256] {
    let mut tbl = [[0u32; 8]; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut k = 0usize;
        while k < 8 {
            tbl[b][k] = if ((b >> k) & 1) != 0 {
                0xFFFF_FFFF
            } else {
                0
            };
            k += 1;
        }
        b += 1;
    }
    tbl
}

static SIGN_MASK_LUT: [[u32; 8]; 256] = build_sign_lut();
static ARRIVED_MASK_LUT: [[u32; 8]; 256] = build_arrived_lut();

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn reconstruct_bulk_row_to_bf16_avx2(
    encoded: &BulkEncodedRow,
    arrived_mask: u128,
    arrived_count: u32,
    quant_bits: u8,
    phi_vecs: &[std::arch::x86_64::__m256; 16],
    psi_vecs: &[std::arch::x86_64::__m256; 16],
    dst_ptr: *mut Bf16,
) {
    use std::arch::x86_64::*;

    if arrived_count == 0 {
        let z = _mm256_setzero_si256();
        let out = dst_ptr as *mut __m256i;
        for idx in 0..8 {
            _mm256_storeu_si256(out.add(idx), z);
        }
        return;
    }

    let rescale = 128.0 / (arrived_count as f32);
    let rho_x = f16_to_f32(encoded.rho_x_f16);
    let rho_r = f16_to_f32(encoded.rho_r_f16);
    let scale_x = rho_x * INV_SQRT_128 * rescale;
    let scale_r = QJL_GAIN * rho_r * rescale;

    let r_mag = _mm256_set1_ps(scale_r);
    let arr_ptr = ARRIVED_MASK_LUT.as_ptr();
    let sgn_ptr = SIGN_MASK_LUT.as_ptr();

    let mut q_vecs = [_mm256_setzero_ps(); 16];
    let mut z_vecs = [_mm256_setzero_ps(); 16];

    if quant_bits == 2 {
        let x_mag = _mm256_set1_ps(C1_MAG * scale_x);
        for idx in 0..16 {
            let shift = idx * 8;
            let a_byte = ((arrived_mask >> shift) & 0xFF) as usize;
            let q0_byte = ((encoded.q_plane0 >> shift) & 0xFF) as usize;
            let z_byte = ((encoded.qjl_signs >> shift) & 0xFF) as usize;

            let a_mask = _mm256_loadu_ps(arr_ptr.add(a_byte) as *const f32);
            let q_sign = _mm256_loadu_ps(sgn_ptr.add(q0_byte) as *const f32);
            let z_sign = _mm256_loadu_ps(sgn_ptr.add(z_byte) as *const f32);

            q_vecs[idx] = _mm256_and_ps(_mm256_xor_ps(x_mag, q_sign), a_mask);
            z_vecs[idx] = _mm256_and_ps(_mm256_xor_ps(r_mag, z_sign), a_mask);
        }
    } else if quant_bits == 3 {
        let c_inner = _mm256_set1_ps(0.452_780_0 * scale_x);
        let c_outer = _mm256_set1_ps(1.510_417_6 * scale_x);
        for idx in 0..16 {
            let shift = idx * 8;
            let a_byte = ((arrived_mask >> shift) & 0xFF) as usize;
            let q0_byte = ((encoded.q_plane0 >> shift) & 0xFF) as usize;
            let q1_byte = ((encoded.q_plane1 >> shift) & 0xFF) as usize;
            let z_byte = ((encoded.qjl_signs >> shift) & 0xFF) as usize;
            let outer_byte = (!(q0_byte ^ q1_byte)) & 0xFF;

            let a_mask = _mm256_loadu_ps(arr_ptr.add(a_byte) as *const f32);
            let outer_mask = _mm256_loadu_ps(arr_ptr.add(outer_byte) as *const f32);
            let pos_sign = _mm256_loadu_ps(sgn_ptr.add(q1_byte) as *const f32);
            let z_sign = _mm256_loadu_ps(sgn_ptr.add(z_byte) as *const f32);

            let q_mag = _mm256_blendv_ps(c_inner, c_outer, outer_mask);
            q_vecs[idx] = _mm256_and_ps(_mm256_xor_ps(q_mag, pos_sign), a_mask);
            z_vecs[idx] = _mm256_and_ps(_mm256_xor_ps(r_mag, z_sign), a_mask);
        }
    } else {
        let mut q_arr = [0.0f32; 128];
        for lane in 0..128 {
            if ((arrived_mask >> lane) & 1) == 0 {
                continue;
            }
            q_arr[lane] = dequantize_lloyd_max(encoded, lane, quant_bits) * scale_x;
        }
        for idx in 0..16 {
            let shift = idx * 8;
            let a_byte = ((arrived_mask >> shift) & 0xFF) as usize;
            let z_byte = ((encoded.qjl_signs >> shift) & 0xFF) as usize;
            let a_mask = _mm256_loadu_ps(arr_ptr.add(a_byte) as *const f32);
            let z_sign = _mm256_loadu_ps(sgn_ptr.add(z_byte) as *const f32);
            q_vecs[idx] = _mm256_loadu_ps(q_arr.as_ptr().add(idx * 8));
            z_vecs[idx] = _mm256_and_ps(_mm256_xor_ps(r_mag, z_sign), a_mask);
        }
    }

    fwht_128_regs_avx2(&mut z_vecs);
    for idx in 0..16 {
        let psi_res = _mm256_xor_ps(z_vecs[idx], psi_vecs[idx]);
        q_vecs[idx] = _mm256_add_ps(q_vecs[idx], psi_res);
    }

    fwht_128_regs_avx2(&mut q_vecs);
    for idx in 0..16 {
        q_vecs[idx] = _mm256_xor_ps(q_vecs[idx], phi_vecs[idx]);
    }
    store_bf16_row_regs_avx2(&q_vecs, dst_ptr);
}

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
        q_hat[lane] = dequantize_lloyd_max(encoded, lane, quant_bits) * scale_x;
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

pub fn reconstruct_bulk_rows_into_tile(
    bulk_rows: &[BulkEncodedRow],
    arrived_mask: u128,
    quant_bits: u8,
    seed: u64,
    layer_id: u16,
    dst_tile: &mut [Bf16],
) {
    let arrived_count = arrived_mask.count_ones();
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            let phi_mask = rademacher_mask_128(seed, layer_id, 0x1111_2222);
            let psi_mask = rademacher_mask_128(seed, layer_id, 0x3333_4444);
            unsafe {
                let phi_vecs = build_avx2_sign_vecs(phi_mask);
                let psi_vecs = build_avx2_sign_vecs(psi_mask);
                for bulk in bulk_rows {
                    let row_ptr = dst_tile.as_mut_ptr().add((bulk.row_idx as usize) * 128);
                    reconstruct_bulk_row_to_bf16_avx2(
                        bulk,
                        arrived_mask,
                        arrived_count,
                        quant_bits,
                        &phi_vecs,
                        &psi_vecs,
                        row_ptr,
                    );
                }
            }
            return;
        }
    }

    for bulk in bulk_rows {
        let row_f = reconstruct_bulk_row(bulk, arrived_mask, quant_bits, seed, layer_id);
        write_reconstructed_row_bf16(dst_tile, bulk.row_idx as usize, &row_f);
    }
}
