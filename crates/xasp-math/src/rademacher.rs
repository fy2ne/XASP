// Copyright (c) 2026 fy2ne (https://xasp.fy2ne.me). All rights reserved.
// Dual-Licensed:
//   1. Noncommercial Evaluation & Academic Research: PolyForm-Noncommercial-1.0.0
//   2. Commercial / Production Deployment: Requires a paid Commercial License from fy2ne.
//      Source-available on https://github.com/fy2ne/xasp (not distributed via crates.io).

#[inline]
pub fn xxh3_mix(seed: u64, salt: u64) -> u64 {
    let mut h = seed ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}

#[inline]
pub fn rademacher_mask_128(request_id_lo: u64, layer_id: u16, stage: u64) -> u128 {
    let base = request_id_lo ^ ((layer_id as u64).wrapping_mul(0x517C_C1B7_2722_0A95));
    let lo = xxh3_mix(base, stage.wrapping_add(1));
    let hi = xxh3_mix(base ^ 0xD6E8_FEB8_6659_FD93, stage.wrapping_add(2));
    ((hi as u128) << 64) | (lo as u128)
}

#[inline]
pub fn apply_signs_128(row: &mut [f32; 128], mask: u128) {
    let lo = mask as u64;
    let hi = (mask >> 64) as u64;
    for lane in 0..64 {
        if ((lo >> lane) & 1) != 0 {
            row[lane] = -row[lane];
        }
        if ((hi >> lane) & 1) != 0 {
            row[lane + 64] = -row[lane + 64];
        }
    }
}
