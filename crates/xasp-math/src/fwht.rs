// Copyright (c) 2026 fy2ne (https://xasp.fy2ne.me). All rights reserved.
// Dual-Licensed:
//   1. Noncommercial Evaluation & Academic Research: PolyForm-Noncommercial-1.0.0
//   2. Commercial / Production Deployment: Requires a paid Commercial License from fy2ne.
//      Source-available on https://github.com/fy2ne/xasp (not distributed via crates.io).

use crate::rademacher::apply_signs_128;

const INV_SQRT_128: f32 = 0.088_388_35;

#[inline]
pub fn fwht_128_scalar(row: &mut [f32; 128]) {
    let mut step = 1usize;
    while step < 128 {
        let span = step << 1;
        let mut base = 0usize;
        while base < 128 {
            for k in 0..step {
                let a = row[base + k];
                let b = row[base + k + step];
                row[base + k] = a + b;
                row[base + k + step] = a - b;
            }
            base += span;
        }
        step = span;
    }
    for lane in row.iter_mut() {
        *lane *= INV_SQRT_128;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn fwht_128_avx2(row: &mut [f32; 128]) {
    use std::arch::x86_64::*;

    let ptr = row.as_mut_ptr();
    let mut vecs = [_mm256_setzero_ps(); 16];
    for idx in 0..16 {
        let mut v = _mm256_loadu_ps(ptr.add(idx * 8));

        // stage 1: stride 1
        let p1 = _mm256_permute_ps(v, 0xB1);
        v = _mm256_blend_ps(_mm256_add_ps(v, p1), _mm256_sub_ps(p1, v), 0xAA);

        // stage 2: stride 2
        let p2 = _mm256_permute_ps(v, 0x4E);
        v = _mm256_blend_ps(_mm256_add_ps(v, p2), _mm256_sub_ps(p2, v), 0xCC);

        // stage 3: stride 4 (cross 128-bit lane inside ymm)
        let p4 = _mm256_permute2f128_ps(v, v, 0x01);
        v = _mm256_blend_ps(_mm256_add_ps(v, p4), _mm256_sub_ps(p4, v), 0xF0);

        vecs[idx] = v;
    }

    // stages 4..7 across full 8-lane ymm vectors
    for stage_shift in 0..4usize {
        let step = 1usize << stage_shift;
        let span = step << 1;
        let mut base = 0usize;
        while base < 16 {
            for k in 0..step {
                let a = vecs[base + k];
                let b = vecs[base + k + step];
                vecs[base + k] = _mm256_add_ps(a, b);
                vecs[base + k + step] = _mm256_sub_ps(a, b);
            }
            base += span;
        }
    }

    let scale = _mm256_set1_ps(INV_SQRT_128);
    for idx in 0..16 {
        _mm256_storeu_ps(ptr.add(idx * 8), _mm256_mul_ps(vecs[idx], scale));
    }
}

#[inline]
fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static AVX2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *AVX2.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
pub fn fwht_128_inplace(row: &mut [f32; 128]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            unsafe {
                fwht_128_avx2(row);
            }
            return;
        }
    }
    fwht_128_scalar(row);
}

#[inline]
pub fn forward_srht_128(row: &mut [f32; 128], sign_mask: u128) {
    apply_signs_128(row, sign_mask);
    fwht_128_inplace(row);
}

#[inline]
pub fn inverse_srht_128(row: &mut [f32; 128], sign_mask: u128) {
    fwht_128_inplace(row);
    apply_signs_128(row, sign_mask);
}
