// Copyright (c) 2026 fy2ne
// SPDX-License-Identifier: Apache-2.0

use crate::fwht::{
    INV_SQRT_128, forward_srht_128, fwht_128_regs_avx2, has_avx2, load_bf16_row_regs_avx2,
};
use crate::rademacher::{build_avx2_sign_vecs, rademacher_mask_128};
use crate::{Bf16, f16_to_f32, f32_to_f16};

const SQRT_128: f32 = 11.313_708;

// Gaussian Lloyd-Max optimal centroids and boundaries for N(0, 1)
pub const C1_MAG: f32 = 0.797_884_6;
pub const C1: [f32; 2] = [-0.797_884_6, 0.797_884_6];

pub const T2: [f32; 3] = [-0.981_598_8, 0.0, 0.981_598_8];
pub const C2: [f32; 4] = [-1.510_417_6, -0.452_780_0, 0.452_780_0, 1.510_417_6];

pub const T3: [f32; 7] = [
    -1.747_981_4,
    -1.050_005_3,
    -0.500_599_3,
    0.0,
    0.500_599_3,
    1.050_005_3,
    1.747_981_4,
];
pub const C3: [f32; 8] = [
    -2.151_992_6,
    -1.343_947_5,
    -0.756_005_2,
    -0.245_094_2,
    0.245_094_2,
    0.756_005_2,
    1.343_947_5,
    2.151_992_6,
];

#[derive(Clone, Copy, Debug, Default)]
pub struct BulkEncodedRow {
    pub row_idx: u16,
    pub rho_x_f16: u16,
    pub rho_r_f16: u16,
    pub q_plane0: u128,
    pub q_plane1: u128,
    pub q_plane2: u128,
    pub qjl_signs: u128,
}

#[inline]
pub fn quantize_scalar_lloyd_max(u: f32, quant_bits: u8) -> (u8, f32) {
    match quant_bits {
        2 => {
            if u < 0.0 {
                (0, C1[0])
            } else {
                (1, C1[1])
            }
        }
        3 => {
            let mut idx = 0usize;
            while idx < 3 && u > T2[idx] {
                idx += 1;
            }
            (idx as u8, C2[idx])
        }
        _ => {
            let mut idx = 0usize;
            while idx < 7 && u > T3[idx] {
                idx += 1;
            }
            (idx as u8, C3[idx])
        }
    }
}

#[inline]
pub fn dequantize_lloyd_max(row: &BulkEncodedRow, lane: usize, quant_bits: u8) -> f32 {
    let b0 = ((row.q_plane0 >> lane) & 1) as usize;
    match quant_bits {
        2 => C1[b0],
        3 => {
            let b1 = ((row.q_plane1 >> lane) & 1) as usize;
            C2[(b1 << 1) | b0]
        }
        _ => {
            let b1 = ((row.q_plane1 >> lane) & 1) as usize;
            let b2 = ((row.q_plane2 >> lane) & 1) as usize;
            C3[(b2 << 2) | (b1 << 1) | b0]
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s4 = _mm_add_ps(lo, hi);
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0x01));
    _mm_cvtss_f32(s1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn encode_regs_fused_avx2(
    row_idx: u16,
    vecs: &mut [std::arch::x86_64::__m256; 16],
    rho_x_sq: f32,
    quant_bits: u8,
    psi_vecs: &[std::arch::x86_64::__m256; 16],
) -> BulkEncodedRow {
    use std::arch::x86_64::*;

    let rho_x = rho_x_sq.sqrt().max(1e-8);
    let rho_x_f16 = f32_to_f16(rho_x);
    let rho_x_q = f16_to_f32(rho_x_f16).max(1e-8);
    let from_unit = rho_x_q * INV_SQRT_128;

    let zero = _mm256_setzero_ps();
    let sign_bit_mask = _mm256_set1_ps(-0.0);

    let mut q_plane0 = 0u128;
    let mut q_plane1 = 0u128;
    let mut q_plane2 = 0u128;
    let mut res_sq_acc = _mm256_setzero_ps();

    if quant_bits == 2 {
        let c1_mag = _mm256_set1_ps(C1_MAG * from_unit);
        for idx in 0..16 {
            let v = vecs[idx];
            let ge_mask = _mm256_cmp_ps(v, zero, _CMP_GE_OQ);
            q_plane0 |= (_mm256_movemask_ps(ge_mask) as u128) << (idx * 8);

            let v_sign = _mm256_and_ps(v, sign_bit_mask);
            let centroid = _mm256_xor_ps(c1_mag, v_sign);
            let diff = _mm256_sub_ps(v, centroid);
            res_sq_acc = _mm256_add_ps(res_sq_acc, _mm256_mul_ps(diff, diff));
            vecs[idx] = _mm256_xor_ps(diff, psi_vecs[idx]);
        }
    } else if quant_bits == 3 {
        let to_unit = _mm256_set1_ps(SQRT_128 / rho_x_q);
        let t_outer = _mm256_set1_ps(0.981_598_8);
        let c_inner = _mm256_set1_ps(0.452_780_0 * from_unit);
        let c_outer = _mm256_set1_ps(1.510_417_6 * from_unit);
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
        let all_ones = _mm256_castsi256_ps(_mm256_set1_epi32(-1));

        for idx in 0..16 {
            let v = vecs[idx];
            let u = _mm256_mul_ps(v, to_unit);
            let abs_u = _mm256_and_ps(u, abs_mask);

            let ge0 = _mm256_cmp_ps(u, zero, _CMP_GE_OQ);
            let gt_outer = _mm256_cmp_ps(abs_u, t_outer, _CMP_GT_OQ);
            let b0_mask = _mm256_blendv_ps(_mm256_xor_ps(gt_outer, all_ones), gt_outer, ge0);

            q_plane1 |= (_mm256_movemask_ps(ge0) as u128) << (idx * 8);
            q_plane0 |= (_mm256_movemask_ps(b0_mask) as u128) << (idx * 8);

            let mag = _mm256_blendv_ps(c_inner, c_outer, gt_outer);
            let v_sign = _mm256_and_ps(v, sign_bit_mask);
            let centroid = _mm256_xor_ps(mag, v_sign);
            let diff = _mm256_sub_ps(v, centroid);
            res_sq_acc = _mm256_add_ps(res_sq_acc, _mm256_mul_ps(diff, diff));
            vecs[idx] = _mm256_xor_ps(diff, psi_vecs[idx]);
        }
    } else {
        let mut tilde = [0.0f32; 128];
        for idx in 0..16 {
            _mm256_storeu_ps(tilde.as_mut_ptr().add(idx * 8), vecs[idx]);
        }
        let to_unit = SQRT_128 / rho_x_q;
        let mut res_arr = [0.0f32; 128];
        let mut res_sq = 0.0f32;
        for lane in 0..128 {
            let x = tilde[lane];
            let (code, centroid) = quantize_scalar_lloyd_max(x * to_unit, quant_bits);
            let c = code as u128;
            q_plane0 |= (c & 1) << lane;
            q_plane1 |= ((c >> 1) & 1) << lane;
            q_plane2 |= ((c >> 2) & 1) << lane;
            let d = x - centroid * from_unit;
            res_arr[lane] = d;
            res_sq += d * d;
        }
        let rho_r_f16 = f32_to_f16(res_sq.sqrt());
        for idx in 0..16 {
            let rv = _mm256_loadu_ps(res_arr.as_ptr().add(idx * 8));
            vecs[idx] = _mm256_xor_ps(rv, psi_vecs[idx]);
        }
        fwht_128_regs_avx2(vecs);
        let mut qjl_signs = 0u128;
        for idx in 0..16 {
            let s_mask = _mm256_cmp_ps(vecs[idx], zero, _CMP_GE_OQ);
            qjl_signs |= (_mm256_movemask_ps(s_mask) as u128) << (idx * 8);
        }
        return BulkEncodedRow {
            row_idx,
            rho_x_f16,
            rho_r_f16,
            q_plane0,
            q_plane1,
            q_plane2,
            qjl_signs,
        };
    }

    let rho_r_f16 = f32_to_f16(hsum256_ps(res_sq_acc).sqrt());
    fwht_128_regs_avx2(vecs);

    let mut qjl_signs = 0u128;
    for idx in 0..16 {
        let s_mask = _mm256_cmp_ps(vecs[idx], zero, _CMP_GE_OQ);
        qjl_signs |= (_mm256_movemask_ps(s_mask) as u128) << (idx * 8);
    }

    BulkEncodedRow {
        row_idx,
        rho_x_f16,
        rho_r_f16,
        q_plane0,
        q_plane1,
        q_plane2,
        qjl_signs,
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn encode_bf16_row_fused_avx2(
    row_idx: u16,
    src_bf16: *const Bf16,
    quant_bits: u8,
    phi_vecs: &[std::arch::x86_64::__m256; 16],
    psi_vecs: &[std::arch::x86_64::__m256; 16],
) -> BulkEncodedRow {
    use std::arch::x86_64::*;

    let mut vecs = load_bf16_row_regs_avx2(src_bf16);
    let mut norm_acc = _mm256_setzero_ps();
    for idx in 0..16 {
        let v = vecs[idx];
        norm_acc = _mm256_add_ps(norm_acc, _mm256_mul_ps(v, v));
        vecs[idx] = _mm256_xor_ps(v, phi_vecs[idx]);
    }
    let rho_x_sq = hsum256_ps(norm_acc);
    fwht_128_regs_avx2(&mut vecs);
    encode_regs_fused_avx2(row_idx, &mut vecs, rho_x_sq, quant_bits, psi_vecs)
}

pub fn encode_prerotated_bulk_row(
    row_idx: u16,
    tilde: &[f32; 128],
    rho_x_sq: f32,
    quant_bits: u8,
    psi_mask: u128,
) -> BulkEncodedRow {
    let rho_x = rho_x_sq.sqrt().max(1e-8);
    let rho_x_f16 = f32_to_f16(rho_x);
    let rho_x_q = f16_to_f32(rho_x_f16).max(1e-8);

    let to_unit = SQRT_128 / rho_x_q;
    let from_unit = rho_x_q * INV_SQRT_128;

    let mut q_plane0 = 0u128;
    let mut q_plane1 = 0u128;
    let mut q_plane2 = 0u128;
    let mut residual = [0.0f32; 128];
    let mut res_sq = 0.0f32;

    for lane in 0..128 {
        let (code, centroid) = quantize_scalar_lloyd_max(tilde[lane] * to_unit, quant_bits);
        let c = code as u128;
        q_plane0 |= (c & 1) << lane;
        q_plane1 |= ((c >> 1) & 1) << lane;
        q_plane2 |= ((c >> 2) & 1) << lane;
        let diff = tilde[lane] - centroid * from_unit;
        residual[lane] = diff;
        res_sq += diff * diff;
    }

    let rho_r_f16 = f32_to_f16(res_sq.sqrt());
    forward_srht_128(&mut residual, psi_mask);

    let mut qjl_signs = 0u128;
    for lane in 0..128 {
        if residual[lane] >= 0.0 {
            qjl_signs |= 1u128 << lane;
        }
    }

    BulkEncodedRow {
        row_idx,
        rho_x_f16,
        rho_r_f16,
        q_plane0,
        q_plane1,
        q_plane2,
        qjl_signs,
    }
}

#[inline]
pub fn encode_bulk_row_with_masks(
    row_idx: u16,
    row: &[f32; 128],
    quant_bits: u8,
    phi_mask: u128,
    psi_mask: u128,
) -> BulkEncodedRow {
    let mut tilde = *row;
    forward_srht_128(&mut tilde, phi_mask);
    let mut norm_sq = 0.0f32;
    for &x in &tilde {
        norm_sq += x * x;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            return unsafe {
                use std::arch::x86_64::*;
                let mut vecs = [_mm256_setzero_ps(); 16];
                let ptr = tilde.as_ptr();
                for idx in 0..16 {
                    vecs[idx] = _mm256_loadu_ps(ptr.add(idx * 8));
                }
                let psi_vecs = build_avx2_sign_vecs(psi_mask);
                encode_regs_fused_avx2(row_idx, &mut vecs, norm_sq, quant_bits, &psi_vecs)
            };
        }
    }
    encode_prerotated_bulk_row(row_idx, &tilde, norm_sq, quant_bits, psi_mask)
}

pub fn encode_bulk_row(
    row_idx: u16,
    row: &[f32; 128],
    quant_bits: u8,
    request_id_lo: u64,
    layer_id: u16,
) -> BulkEncodedRow {
    let phi_mask = rademacher_mask_128(request_id_lo, layer_id, 0x1111_2222);
    let psi_mask = rademacher_mask_128(request_id_lo, layer_id, 0x3333_4444);
    encode_bulk_row_with_masks(row_idx, row, quant_bits, phi_mask, psi_mask)
}

pub struct BitPacker {
    buf: Vec<u8>,
    acc: u64,
    bits: u32,
}

impl BitPacker {
    pub fn with_capacity_bits(total_bits: usize) -> Self {
        Self {
            buf: Vec::with_capacity((total_bits + 7) / 8),
            acc: 0,
            bits: 0,
        }
    }

    #[inline]
    pub fn push_bits(&mut self, val_bits: u32, count: u32) {
        let masked = (val_bits as u64) & ((1u64 << count) - 1);
        self.acc |= masked << self.bits;
        self.bits += count;
        while self.bits >= 8 {
            self.buf.push((self.acc & 0xFF) as u8);
            self.acc >>= 8;
            self.bits -= 8;
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        if self.bits > 0 {
            self.buf.push((self.acc & 0xFF) as u8);
        }
        self.buf
    }
}

pub struct BitUnpacker<'a> {
    src: &'a [u8],
    pos: usize,
    acc: u64,
    bits: u32,
}

impl<'a> BitUnpacker<'a> {
    pub fn new(src: &'a [u8]) -> Self {
        Self {
            src,
            pos: 0,
            acc: 0,
            bits: 0,
        }
    }

    #[inline]
    pub fn pop_bits(&mut self, count: u32) -> u32 {
        while self.bits < count && self.pos < self.src.len() {
            self.acc |= (self.src[self.pos] as u64) << self.bits;
            self.pos += 1;
            self.bits += 8;
        }
        let out = (self.acc & ((1u64 << count) - 1)) as u32;
        self.acc >>= count;
        self.bits = self.bits.saturating_sub(count);
        out
    }
}

pub fn pack_bulk_stripe(
    rows: &[BulkEncodedRow],
    coord_start: u16,
    coord_count: u16,
    quant_bits: u8,
) -> Vec<u8> {
    let q_bits = (quant_bits.saturating_sub(1)).clamp(1, 3) as u32;
    let c_cnt = coord_count as u32;
    let mask = (1u32 << c_cnt) - 1;
    let total_bits = rows.len() * (coord_count as usize) * ((q_bits + 1) as usize);
    let mut packer = BitPacker::with_capacity_bits(total_bits);
    let start = coord_start as usize;

    for row in rows {
        packer.push_bits(((row.q_plane0 >> start) as u32) & mask, c_cnt);
        if q_bits >= 2 {
            packer.push_bits(((row.q_plane1 >> start) as u32) & mask, c_cnt);
        }
        if q_bits >= 3 {
            packer.push_bits(((row.q_plane2 >> start) as u32) & mask, c_cnt);
        }
        packer.push_bits(((row.qjl_signs >> start) as u32) & mask, c_cnt);
    }
    packer.finish()
}

pub fn unpack_bulk_stripe_into(
    stripe_bytes: &[u8],
    rows: &mut [BulkEncodedRow],
    coord_start: u16,
    coord_count: u16,
    quant_bits: u8,
) {
    let q_bits = (quant_bits.saturating_sub(1)).clamp(1, 3) as u32;
    let c_cnt = coord_count as u32;
    let clear_mask = !(((1u128 << c_cnt) - 1) << coord_start);
    let mut reader = BitUnpacker::new(stripe_bytes);
    let start = coord_start as usize;

    for row in rows.iter_mut() {
        let p0 = reader.pop_bits(c_cnt) as u128;
        row.q_plane0 = (row.q_plane0 & clear_mask) | (p0 << start);
        if q_bits >= 2 {
            let p1 = reader.pop_bits(c_cnt) as u128;
            row.q_plane1 = (row.q_plane1 & clear_mask) | (p1 << start);
        }
        if q_bits >= 3 {
            let p2 = reader.pop_bits(c_cnt) as u128;
            row.q_plane2 = (row.q_plane2 & clear_mask) | (p2 << start);
        }
        let zs = reader.pop_bits(c_cnt) as u128;
        row.qjl_signs = (row.qjl_signs & clear_mask) | (zs << start);
    }
}
