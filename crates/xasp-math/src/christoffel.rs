// Copyright (c) 2026 fy2ne (https://xasp.fy2ne.me). All rights reserved.
// Dual-Licensed:
//   1. Noncommercial Evaluation & Academic Research: PolyForm-Noncommercial-1.0.0
//   2. Commercial / Production Deployment: Requires a paid Commercial License from fy2ne.
//      Source-available on https://github.com/fy2ne/xasp (not distributed via crates.io).

use crate::fwht::{has_avx2, load_bf16_row_regs_avx2};
use crate::rademacher::{build_avx2_sign_vecs, rademacher_mask_128};
use crate::{Bf16, f16_to_f32, f32_to_f16};

const SKETCH_DIM: usize = 8;

#[derive(Clone, Debug)]
pub struct VipRowInt8 {
    pub row_idx: u16,
    pub scale_f16: u16,
    pub q: [i8; 128],
}

impl VipRowInt8 {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn pack_from_bf16_avx2(row_idx: u16, src: *const Bf16) -> Self {
        use std::arch::x86_64::*;

        let vecs = load_bf16_row_regs_avx2(src);
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
        let mut vmax = _mm256_setzero_ps();
        for v in &vecs {
            vmax = _mm256_max_ps(vmax, _mm256_and_ps(*v, abs_mask));
        }
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), vmax);
        let mut amax = 0.0f32;
        for &m in &tmp {
            if m > amax {
                amax = m;
            }
        }

        let raw_step = if amax > 1e-8 { amax / 127.0 } else { 1e-8 };
        let scale_f16 = f32_to_f16(raw_step);
        let step_q = f16_to_f32(scale_f16).max(1e-8);
        let inv = _mm256_set1_ps(1.0 / step_q);
        let lo_c = _mm256_set1_ps(-127.0);
        let hi_c = _mm256_set1_ps(127.0);
        let mut q = [0i8; 128];
        let out_ptr = q.as_mut_ptr() as *mut __m128i;

        let mut i = 0usize;
        while i < 16 {
            let a = _mm256_round_ps::<0>(_mm256_min_ps(hi_c, _mm256_max_ps(lo_c, _mm256_mul_ps(vecs[i], inv))));
            let b = _mm256_round_ps::<0>(_mm256_min_ps(hi_c, _mm256_max_ps(lo_c, _mm256_mul_ps(vecs[i + 1], inv))));
            let p16 = _mm256_packs_epi32(_mm256_cvtps_epi32(a), _mm256_cvtps_epi32(b));
            let ord16 = _mm256_permute4x64_epi64(p16, 0xD8);
            let p8 = _mm256_packs_epi16(ord16, ord16);
            let lo = _mm256_castsi256_si128(p8);
            let hi = _mm256_extracti128_si256(p8, 1);
            let combined = _mm_unpacklo_epi64(lo, hi);
            _mm_storeu_si128(out_ptr.add(i / 2), combined);
            i += 2;
        }

        Self {
            row_idx,
            scale_f16,
            q,
        }
    }

    pub fn pack(row_idx: u16, row: &[f32; 128]) -> Self {
        let mut amax = 0.0f32;
        for &lane in row {
            let abs = lane.abs();
            if abs > amax {
                amax = abs;
            }
        }
        let raw_step = if amax > 1e-8 { amax / 127.0 } else { 1e-8 };
        let scale_f16 = f32_to_f16(raw_step);
        let step_q = f16_to_f32(scale_f16).max(1e-8);
        let inv = 1.0 / step_q;
        let mut q = [0i8; 128];
        for lane in 0..128 {
            q[lane] = (row[lane] * inv).round().clamp(-127.0, 127.0) as i8;
        }
        Self {
            row_idx,
            scale_f16,
            q,
        }
    }

    pub fn pack_bf16(row_idx: u16, row_src: &[Bf16]) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if has_avx2() {
                return unsafe { Self::pack_from_bf16_avx2(row_idx, row_src.as_ptr()) };
            }
        }
        let mut row = [0.0f32; 128];
        for c in 0..128 {
            row[c] = row_src[c].to_f32();
        }
        Self::pack(row_idx, &row)
    }

    pub fn unpack_into(&self, dst: &mut [f32; 128]) {
        let step = f16_to_f32(self.scale_f16);
        for lane in 0..128 {
            dst[lane] = (self.q[lane] as f32) * step;
        }
    }

    pub fn append_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.row_idx.to_le_bytes());
        out.extend_from_slice(&self.scale_f16.to_le_bytes());
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(self.q.as_ptr() as *const u8, 128) };
        out.extend_from_slice(bytes);
    }

    pub fn from_bytes(src: &[u8]) -> Option<Self> {
        if src.len() < 132 {
            return None;
        }
        let row_idx = u16::from_le_bytes([src[0], src[1]]);
        let scale_f16 = u16::from_le_bytes([src[2], src[3]]);
        let mut q = [0i8; 128];
        for i in 0..128 {
            q[i] = src[4 + i] as i8;
        }
        Some(Self {
            row_idx,
            scale_f16,
            q,
        })
    }
}

pub fn invert_spd_8(mat: &mut [[f32; SKETCH_DIM]; SKETCH_DIM]) -> [[f32; SKETCH_DIM]; SKETCH_DIM] {
    let mut l = [[0.0f32; SKETCH_DIM]; SKETCH_DIM];
    for i in 0..SKETCH_DIM {
        for j in 0..=i {
            let mut sum = mat[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                l[i][j] = if sum > 1e-12 { sum.sqrt() } else { 1e-6 };
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    let mut linv = [[0.0f32; SKETCH_DIM]; SKETCH_DIM];
    for i in 0..SKETCH_DIM {
        linv[i][i] = 1.0 / l[i][i];
        for j in (0..i).rev() {
            let mut sum = 0.0f32;
            for k in (j + 1)..=i {
                sum -= l[i][k] * linv[k][j];
            }
            linv[i][j] = sum / l[i][i];
        }
    }

    let mut inv = [[0.0f32; SKETCH_DIM]; SKETCH_DIM];
    for i in 0..SKETCH_DIM {
        for j in 0..=i {
            let mut sum = 0.0f32;
            for k in i..SKETCH_DIM {
                sum += linv[k][i] * linv[k][j];
            }
            inv[i][j] = sum;
            inv[j][i] = sum;
        }
    }
    inv
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sketch_row_avx2(
    k_ptr: *const Bf16,
    v_ptr: *const Bf16,
    sketch_vecs: &[std::arch::x86_64::__m256; 16],
    k_sketch_out: *mut f32,
) -> (f32, f32) {
    use std::arch::x86_64::*;

    let k_vecs = load_bf16_row_regs_avx2(k_ptr);
    let v_vecs = load_bf16_row_regs_avx2(v_ptr);

    let mut k_sq_acc = _mm256_setzero_ps();
    let mut v_sq_acc = _mm256_setzero_ps();
    let mut sk = _mm256_setzero_ps();

    for idx in 0..16 {
        let kv = k_vecs[idx];
        let vv = v_vecs[idx];
        k_sq_acc = _mm256_add_ps(k_sq_acc, _mm256_mul_ps(kv, kv));
        v_sq_acc = _mm256_add_ps(v_sq_acc, _mm256_mul_ps(vv, vv));
        sk = _mm256_add_ps(sk, _mm256_xor_ps(kv, sketch_vecs[idx]));
    }

    let sk_scaled = _mm256_mul_ps(sk, _mm256_set1_ps(0.25));
    _mm256_storeu_ps(k_sketch_out, sk_scaled);

    (
        crate::polar_qjl::hsum256_ps(k_sq_acc),
        crate::polar_qjl::hsum256_ps(v_sq_acc),
    )
}

pub fn stratify_tile_fast_into(
    keys: &[Bf16],
    values: &[Bf16],
    n_rows: usize,
    vip_ratio: f32,
    sink_rows: usize,
    request_id_lo: u64,
    layer_id: u16,
    sketched_buf: &mut Vec<[f32; SKETCH_DIM]>,
    v_sq_buf: &mut Vec<f32>,
    is_vip_out: &mut Vec<bool>,
    vip_rows_out: &mut Vec<u16>,
    bulk_rows_out: &mut Vec<u16>,
    ranked_scratch: &mut Vec<(usize, f32)>,
) {
    vip_rows_out.clear();
    bulk_rows_out.clear();
    if n_rows == 0 {
        return;
    }

    if sketched_buf.len() < n_rows {
        sketched_buf.resize(n_rows, [0.0f32; SKETCH_DIM]);
        v_sq_buf.resize(n_rows, 0.0f32);
        is_vip_out.resize(n_rows, false);
    }

    let sketch_mask = rademacher_mask_128(request_id_lo, layer_id, 0xC415_70FE);
    let mut frob_sq = 0.0f32;

    if has_avx2() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let sketch_vecs = build_avx2_sign_vecs(sketch_mask);
            for r in 0..n_rows {
                let (ksq, vsq) = sketch_row_avx2(
                    keys.as_ptr().add(r * 128),
                    values.as_ptr().add(r * 128),
                    &sketch_vecs,
                    sketched_buf[r].as_mut_ptr(),
                );
                frob_sq += ksq;
                v_sq_buf[r] = vsq;
            }
        }
    } else {
        for r in 0..n_rows {
            let off = r * 128;
            let mut ksq = 0.0f32;
            let mut vsq = 0.0f32;
            let mut sk = [0.0f32; SKETCH_DIM];
            for c in 0..128 {
                let kf = keys[off + c].to_f32();
                let vf = values[off + c].to_f32();
                ksq += kf * kf;
                vsq += vf * vf;
                let sgn = if ((sketch_mask >> c) & 1) != 0 { -1.0 } else { 1.0 };
                sk[c & 7] += kf * sgn;
            }
            for s in 0..SKETCH_DIM {
                sketched_buf[r][s] = sk[s] * 0.25;
            }
            frob_sq += ksq;
            v_sq_buf[r] = vsq;
        }
    }

    let mut gram = [[0.0f32; SKETCH_DIM]; SKETCH_DIM];
    let mut r = 0usize;
    while r < n_rows {
        let sk = &sketched_buf[r];
        for i in 0..SKETCH_DIM {
            let si = sk[i] * 1.414_213_5;
            for j in 0..=i {
                gram[i][j] += si * (sk[j] * 1.414_213_5);
            }
        }
        r += 2;
    }

    let mu = (frob_sq / (n_rows as f32)).max(1e-5);
    for i in 0..SKETCH_DIM {
        for j in 0..i {
            gram[j][i] = gram[i][j];
        }
        gram[i][i] += mu;
    }

    let inv_gram = invert_spd_8(&mut gram);
    ranked_scratch.clear();
    ranked_scratch.reserve(n_rows);

    for idx in 0..n_rows {
        if idx < sink_rows {
            ranked_scratch.push((idx, f32::INFINITY));
            continue;
        }
        let sk = &sketched_buf[idx];
        let mut tau = 0.0f32;
        for i in 0..SKETCH_DIM {
            let mut acc = 0.0f32;
            for j in 0..SKETCH_DIM {
                acc += inv_gram[i][j] * sk[j];
            }
            tau += sk[i] * acc;
        }
        let sc = tau.max(0.0) * v_sq_buf[idx].sqrt();
        ranked_scratch.push((idx, sc));
    }

    let target_vip = ((n_rows as f32 * vip_ratio).ceil() as usize)
        .max(sink_rows.min(n_rows))
        .min(n_rows);

    ranked_scratch.sort_by(|a, b| b.1.total_cmp(&a.1));

    for flag in is_vip_out.iter_mut().take(n_rows) {
        *flag = false;
    }
    for &(idx, _) in ranked_scratch.iter().take(target_vip) {
        is_vip_out[idx] = true;
    }

    for (idx, &flag) in is_vip_out.iter().take(n_rows).enumerate() {
        if flag {
            vip_rows_out.push(idx as u16);
        } else {
            bulk_rows_out.push(idx as u16);
        }
    }
}
