// Copyright (c) 2026 fy2ne (https://xasp.fy2ne.me). All rights reserved.
// Dual-Licensed:
//   1. Noncommercial Evaluation & Academic Research: PolyForm-Noncommercial-1.0.0
//   2. Commercial / Production Deployment: Requires a paid Commercial License from fy2ne.
//      Source-available on https://github.com/fy2ne/xasp (not distributed via crates.io).

use crate::fwht::forward_srht_128;
use crate::rademacher::rademacher_mask_128;
use crate::{f16_to_f32, f32_to_f16};

const SQRT_128: f32 = 11.313_708;
const INV_SQRT_128: f32 = 0.088_388_35;

// Gaussian Lloyd-Max optimal centroids and boundaries for N(0, 1)
const C1: [f32; 2] = [-0.797_884_6, 0.797_884_6];

const T2: [f32; 3] = [-0.981_598_8, 0.0, 0.981_598_8];
const C2: [f32; 4] = [-1.510_417_6, -0.452_780_0, 0.452_780_0, 1.510_417_6];

const T3: [f32; 7] = [
    -1.747_981_4,
    -1.050_005_3,
    -0.500_599_3,
    0.0,
    0.500_599_3,
    1.050_005_3,
    1.747_981_4,
];
const C3: [f32; 8] = [
    -2.151_992_6,
    -1.343_947_5,
    -0.756_005_2,
    -0.245_094_2,
    0.245_094_2,
    0.756_005_2,
    1.343_947_5,
    2.151_992_6,
];

#[derive(Clone, Debug)]
pub struct BulkEncodedRow {
    pub row_idx: u16,
    pub rho_x_f16: u16,
    pub rho_r_f16: u16,
    pub q_codes: [u8; 128],
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
pub fn dequantize_lloyd_max(code: u8, quant_bits: u8) -> f32 {
    match quant_bits {
        2 => C1[(code & 1) as usize],
        3 => C2[(code & 3) as usize],
        _ => C3[(code & 7) as usize],
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
    let rho_x = norm_sq.sqrt().max(1e-8);
    let rho_x_f16 = f32_to_f16(rho_x);
    let rho_x_q = f16_to_f32(rho_x_f16).max(1e-8);

    let to_unit = SQRT_128 / rho_x_q;
    let from_unit = rho_x_q * INV_SQRT_128;

    let mut q_codes = [0u8; 128];
    let mut residual = [0.0f32; 128];
    let mut res_sq = 0.0f32;

    for lane in 0..128 {
        let (code, centroid) = quantize_scalar_lloyd_max(tilde[lane] * to_unit, quant_bits);
        q_codes[lane] = code;
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
        q_codes,
        qjl_signs,
    }
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
    let total_bits = rows.len() * (coord_count as usize) * ((q_bits + 1) as usize);
    let mut packer = BitPacker::with_capacity_bits(total_bits);
    let start = coord_start as usize;
    let end = start + (coord_count as usize);

    for row in rows {
        for lane in start..end {
            packer.push_bits(row.q_codes[lane] as u32, q_bits);
            let sign_bit = ((row.qjl_signs >> lane) & 1) as u32;
            packer.push_bits(sign_bit, 1);
        }
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
    let mut reader = BitUnpacker::new(stripe_bytes);
    let start = coord_start as usize;
    let end = start + (coord_count as usize);

    for row in rows.iter_mut() {
        for lane in start..end {
            row.q_codes[lane] = reader.pop_bits(q_bits) as u8;
            let sign_bit = reader.pop_bits(1) as u128;
            if sign_bit != 0 {
                row.qjl_signs |= 1u128 << lane;
            } else {
                row.qjl_signs &= !(1u128 << lane);
            }
        }
    }
}
