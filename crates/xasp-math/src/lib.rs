// Copyright (c) 2026 fy2ne (https://xasp.fy2ne.me). All rights reserved.
// Dual-Licensed:
//   1. Noncommercial Evaluation & Academic Research: PolyForm-Noncommercial-1.0.0
//   2. Commercial / Production Deployment: Requires a paid Commercial License from fy2ne.
//      Source-available on https://github.com/fy2ne/xasp (not distributed via crates.io).

pub mod christoffel;
pub mod estimator;
pub mod fwht;
pub mod polar_qjl;
pub mod rademacher;

use christoffel::{VipRowInt8, stratify_tile_fast_into};
use estimator::{reconstruct_bulk_rows_into_tile, write_reconstructed_row_bf16};
use fwht::has_avx2;
use polar_qjl::{BulkEncodedRow, encode_bulk_row_with_masks};
use rademacher::rademacher_mask_128;
use std::cell::RefCell;

pub const MINOR_TILE_DIM: usize = 128;

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Bf16(pub u16);

impl Bf16 {
    #[inline]
    pub fn from_f32(x: f32) -> Self {
        let bits = x.to_bits();
        if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
            return Self(((bits >> 16) as u16) | 0x0040);
        }
        let round_bias = 0x7FFF + ((bits >> 16) & 1);
        Self((bits.wrapping_add(round_bias) >> 16) as u16)
    }

    #[inline]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
}

#[inline]
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 255 {
        return sign | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7BFF;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let m = (mant | 0x0080_0000) >> (1 - half_exp + 13);
        return sign | (m as u16);
    }
    sign | ((half_exp as u16) << 10) | ((mant >> 13) as u16)
}

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut m = mant;
        let mut e = 127 - 14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        return f32::from_bits(sign | (e << 23) | (m << 13));
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7F80_0000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

#[derive(Clone, Debug, Default)]
pub struct EncodedLayerHalf {
    pub n_rows: usize,
    pub quant_bits: u8,
    pub vip_rows: Vec<VipRowInt8>,
    pub bulk_rows: Vec<BulkEncodedRow>,
}

#[derive(Clone, Debug)]
pub struct LocalSpectralCodec {
    pub vip_ratio: f32,
    pub sink_rows: usize,
}

impl Default for LocalSpectralCodec {
    fn default() -> Self {
        Self {
            vip_ratio: 0.05,
            sink_rows: 4,
        }
    }
}

#[derive(Default)]
struct CodecScratch {
    sketched: Vec<[f32; 8]>,
    v_sq: Vec<f32>,
    is_vip: Vec<bool>,
    vip_rows: Vec<u16>,
    bulk_rows: Vec<u16>,
    ranked: Vec<(usize, f32)>,
}

thread_local! {
    static CODEC_SCRATCH: RefCell<CodecScratch> = RefCell::new(CodecScratch::default());
}

impl LocalSpectralCodec {
    pub fn encode_kv_layer_into(
        &self,
        keys: &[Bf16],
        values: &[Bf16],
        n_rows: usize,
        quant_bits: u8,
        request_id_lo: u64,
        layer_id: u16,
        k_out: &mut EncodedLayerHalf,
        v_out: &mut EncodedLayerHalf,
    ) {
        let k_phi = rademacher_mask_128(request_id_lo, layer_id, 0x1111_2222);
        let k_psi = rademacher_mask_128(request_id_lo, layer_id, 0x3333_4444);
        let v_phi = rademacher_mask_128(request_id_lo ^ 0xA5A5_5A5A, layer_id, 0x1111_2222);
        let v_psi = rademacher_mask_128(request_id_lo ^ 0xA5A5_5A5A, layer_id, 0x3333_4444);

        k_out.n_rows = n_rows;
        k_out.quant_bits = quant_bits;
        k_out.vip_rows.clear();
        k_out.bulk_rows.clear();
        v_out.n_rows = n_rows;
        v_out.quant_bits = quant_bits;
        v_out.vip_rows.clear();
        v_out.bulk_rows.clear();

        CODEC_SCRATCH.with(|cell| {
            let mut sc = cell.borrow_mut();
            let sc = &mut *sc;

            stratify_tile_fast_into(
                keys,
                values,
                n_rows,
                self.vip_ratio,
                self.sink_rows,
                request_id_lo,
                layer_id,
                &mut sc.sketched,
                &mut sc.v_sq,
                &mut sc.is_vip,
                &mut sc.vip_rows,
                &mut sc.bulk_rows,
                &mut sc.ranked,
            );

            k_out.vip_rows.reserve(sc.vip_rows.len());
            v_out.vip_rows.reserve(sc.vip_rows.len());
            for &r in &sc.vip_rows {
                let off = (r as usize) * 128;
                k_out.vip_rows.push(VipRowInt8::pack_bf16(r, &keys[off..off + 128]));
                v_out.vip_rows.push(VipRowInt8::pack_bf16(r, &values[off..off + 128]));
            }

            k_out.bulk_rows.reserve(sc.bulk_rows.len());
            v_out.bulk_rows.reserve(sc.bulk_rows.len());

            if has_avx2() {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    let k_phi_vecs = rademacher::build_avx2_sign_vecs(k_phi);
                    let k_psi_vecs = rademacher::build_avx2_sign_vecs(k_psi);
                    let v_phi_vecs = rademacher::build_avx2_sign_vecs(v_phi);
                    let v_psi_vecs = rademacher::build_avx2_sign_vecs(v_psi);
                    let kp = keys.as_ptr();
                    let vp = values.as_ptr();
                    for &r in &sc.bulk_rows {
                        let off = (r as usize) * 128;
                        k_out.bulk_rows.push(polar_qjl::encode_bf16_row_fused_avx2(
                            r,
                            kp.add(off),
                            quant_bits,
                            &k_phi_vecs,
                            &k_psi_vecs,
                        ));
                        v_out.bulk_rows.push(polar_qjl::encode_bf16_row_fused_avx2(
                            r,
                            vp.add(off),
                            quant_bits,
                            &v_phi_vecs,
                            &v_psi_vecs,
                        ));
                    }
                }
            } else {
                let mut kf = [0.0f32; 128];
                let mut vf = [0.0f32; 128];
                for &r in &sc.bulk_rows {
                    let off = (r as usize) * 128;
                    for c in 0..128 {
                        kf[c] = keys[off + c].to_f32();
                        vf[c] = values[off + c].to_f32();
                    }
                    k_out.bulk_rows.push(encode_bulk_row_with_masks(
                        r, &kf, quant_bits, k_phi, k_psi,
                    ));
                    v_out.bulk_rows.push(encode_bulk_row_with_masks(
                        r, &vf, quant_bits, v_phi, v_psi,
                    ));
                }
            }
        });
    }

    pub fn encode_kv_layer(
        &self,
        keys: &[Bf16],
        values: &[Bf16],
        n_rows: usize,
        quant_bits: u8,
        request_id_lo: u64,
        layer_id: u16,
    ) -> (EncodedLayerHalf, EncodedLayerHalf) {
        let mut k_out = EncodedLayerHalf::default();
        let mut v_out = EncodedLayerHalf::default();
        self.encode_kv_layer_into(
            keys,
            values,
            n_rows,
            quant_bits,
            request_id_lo,
            layer_id,
            &mut k_out,
            &mut v_out,
        );
        (k_out, v_out)
    }
}

pub fn decode_layer_half_into(
    half: &EncodedLayerHalf,
    arrived_mask: u128,
    request_id_lo: u64,
    layer_id: u16,
    is_value: bool,
    dst: &mut [Bf16],
) {
    let seed = if is_value {
        request_id_lo ^ 0xA5A5_5A5A
    } else {
        request_id_lo
    };

    for vip in &half.vip_rows {
        let mut row_f = [0.0f32; 128];
        vip.unpack_into(&mut row_f);
        write_reconstructed_row_bf16(dst, vip.row_idx as usize, &row_f);
    }

    reconstruct_bulk_rows_into_tile(
        &half.bulk_rows,
        arrived_mask,
        half.quant_bits,
        seed,
        layer_id,
        dst,
    );
}

pub fn decode_layer_half(
    half: &EncodedLayerHalf,
    arrived_mask: u128,
    request_id_lo: u64,
    layer_id: u16,
    is_value: bool,
) -> Vec<Bf16> {
    let mut dst = vec![Bf16::default(); half.n_rows * 128];
    decode_layer_half_into(half, arrived_mask, request_id_lo, layer_id, is_value, &mut dst);
    dst
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fwht::{forward_srht_128, fwht_128_scalar, inverse_srht_128};
    use crate::polar_qjl::pack_bulk_stripe;
    use crate::rademacher::rademacher_mask_128;

    fn lcg_f32(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((*state >> 32) as u32) as f32 / (u32::MAX as f32);
        u * 2.0 - 1.0
    }

    #[test]
    fn fwht_scalar_and_avx2_match_and_preserve_inner_product() {
        let mut s = 0x1234_5678_9ABC_DEF0u64;
        let mut q = [0.0f32; 128];
        let mut x = [0.0f32; 128];
        for i in 0..128 {
            q[i] = lcg_f32(&mut s);
            x[i] = lcg_f32(&mut s) * if i % 29 == 0 { 25.0 } else { 1.0 };
        }

        let mut dot_orig = 0.0f32;
        for i in 0..128 {
            dot_orig += q[i] * x[i];
        }

        let mut x_scalar = x;
        fwht_128_scalar(&mut x_scalar);
        let mut x_dispatch = x;
        crate::fwht::fwht_128_inplace(&mut x_dispatch);
        for i in 0..128 {
            assert!((x_scalar[i] - x_dispatch[i]).abs() < 1e-5);
        }

        let mask = rademacher_mask_128(42, 3, 7);
        let mut q_rot = q;
        let mut x_rot = x;
        forward_srht_128(&mut q_rot, mask);
        forward_srht_128(&mut x_rot, mask);

        let mut dot_rot = 0.0f32;
        for i in 0..128 {
            dot_rot += q_rot[i] * x_rot[i];
        }
        assert!((dot_orig - dot_rot).abs() < 1e-4);

        inverse_srht_128(&mut x_rot, mask);
        for i in 0..128 {
            assert!((x[i] - x_rot[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn theorem1_and_2_compression_and_stalled_stream_accuracy() {
        let n_rows = 512usize;
        let mut s = 0xCAFE_BABE_1122_3344u64;
        let mut keys = Vec::with_capacity(n_rows * 128);
        let mut vals = Vec::with_capacity(n_rows * 128);

        for r in 0..n_rows {
            let row_amp = if r < 4 {
                14.0
            } else if r % 28 == 0 {
                5.5
            } else {
                0.28
            };
            for c in 0..128 {
                let outlier = if c == 7 || c == 63 || c == 119 { 24.0 } else { 1.0 };
                let kf = lcg_f32(&mut s) * row_amp * outlier;
                let vf = lcg_f32(&mut s) * row_amp;
                keys.push(Bf16::from_f32(kf));
                vals.push(Bf16::from_f32(vf));
            }
        }

        let codec = LocalSpectralCodec::default();
        let (k_enc, v_enc) = codec.encode_kv_layer(&keys, &vals, n_rows, 2, 9001, 5);

        let raw_bytes = n_rows * 128 * 2;
        let mut wire_bytes = k_enc.vip_rows.len() * 132 + k_enc.bulk_rows.len() * 4;
        let mut cursor = 0u16;
        for stream_idx in 0..14u16 {
            let count = if stream_idx < 12 { 9 } else { 10 };
            wire_bytes += pack_bulk_stripe(&k_enc.bulk_rows, cursor, count, 2).len();
            cursor += count;
        }
        let ratio = (raw_bytes as f64) / (wire_bytes as f64);
        assert!(ratio >= 6.0 && ratio <= 8.0, "ratio was {ratio:.2}");

        // stall 1 bulk stream (coords 45..54 missing, delta_hat = 9/128)
        let stalled_mask = u128::MAX ^ (((1u128 << 9) - 1) << 45);
        let k_dec = decode_layer_half(&k_enc, stalled_mask, 9001, 5, false);
        let v_dec = decode_layer_half(&v_enc, stalled_mask, 9001, 5, true);

        let mut q = [0.0f32; 128];
        for c in 0..128 {
            q[c] = keys[c].to_f32() * 0.05 + lcg_f32(&mut s) * 0.25;
        }

        let inv_sqrt_d = 1.0f64 / (128.0f64).sqrt();
        let mut logits_t = vec![0.0f64; n_rows];
        let mut logits_h = vec![0.0f64; n_rows];
        let mut max_t = f64::NEG_INFINITY;
        let mut max_h = f64::NEG_INFINITY;

        for r in 0..n_rows {
            let mut dt = 0.0f64;
            let mut dh = 0.0f64;
            for c in 0..128 {
                let qv = q[c] as f64;
                dt += qv * (keys[r * 128 + c].to_f32() as f64);
                dh += qv * (k_dec[r * 128 + c].to_f32() as f64);
            }
            logits_t[r] = dt * inv_sqrt_d;
            logits_h[r] = dh * inv_sqrt_d;
            max_t = max_t.max(logits_t[r]);
            max_h = max_h.max(logits_h[r]);
        }

        let mut pt = vec![0.0f64; n_rows];
        let mut ph = vec![0.0f64; n_rows];
        let (mut st, mut sh) = (0.0f64, 0.0f64);
        for r in 0..n_rows {
            pt[r] = (logits_t[r] - max_t).clamp(-40.0, 0.0).exp();
            ph[r] = (logits_h[r] - max_h).clamp(-40.0, 0.0).exp();
            st += pt[r];
            sh += ph[r];
        }

        let mut out_t = [0.0f64; 128];
        let mut out_h = [0.0f64; 128];
        for r in 0..n_rows {
            let wt = pt[r] / st;
            let wh = ph[r] / sh;
            for c in 0..128 {
                out_t[c] += wt * (vals[r * 128 + c].to_f32() as f64);
                out_h[c] += wh * (v_dec[r * 128 + c].to_f32() as f64);
            }
        }

        let mut err_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        for c in 0..128 {
            let d = out_t[c] - out_h[c];
            err_sq += d * d;
            ref_sq += out_t[c] * out_t[c];
        }
        let attn_rel_rmse = (err_sq / ref_sq).sqrt();
        assert!(attn_rel_rmse < 0.028, "attn_rel_rmse was {attn_rel_rmse:.4}");
    }
}
