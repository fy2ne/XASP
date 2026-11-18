// Copyright (c) 2026 fy2ne (https://xasp.fy2ne.me). All rights reserved.
// Dual-Licensed:
//   1. Noncommercial Evaluation & Academic Research: PolyForm-Noncommercial-1.0.0
//   2. Commercial / Production Deployment: Requires a paid Commercial License from fy2ne.
//      Source-available on https://github.com/fy2ne/xasp (not distributed via crates.io).

use xasp_math::{Bf16, MINOR_TILE_DIM};

pub struct PhysicalTileArenaView<'a> {
    pub slice: &'a [Bf16],
    pub n_rows: usize,
    pub is_4k_aligned: bool,
}

pub struct PhysicalTileArenaViewMut<'a> {
    pub slice: &'a mut [Bf16],
    pub n_rows: usize,
    pub is_4k_aligned: bool,
}

#[inline]
pub fn check_folded_device_layout(
    ptr: *const u16,
    n_rows: usize,
    minor_tile_dim: usize,
) -> bool {
    !ptr.is_null()
        && n_rows > 0
        && minor_tile_dim == MINOR_TILE_DIM
        && (ptr as usize) % std::mem::align_of::<u16>() == 0
}

impl<'a> PhysicalTileArenaView<'a> {
    pub unsafe fn from_raw(
        ptr: *const u16,
        n_rows: usize,
        minor_tile_dim: usize,
    ) -> Result<Self, &'static str> {
        if !check_folded_device_layout(ptr, n_rows, minor_tile_dim) {
            return Err("invalid 128-lane physical tpu tile descriptor");
        }
        let slice = std::slice::from_raw_parts(ptr as *const Bf16, n_rows * MINOR_TILE_DIM);
        Ok(Self {
            slice,
            n_rows,
            is_4k_aligned: (ptr as usize) % 4096 == 0,
        })
    }
}

impl<'a> PhysicalTileArenaViewMut<'a> {
    pub unsafe fn from_raw_mut(
        ptr: *mut u16,
        n_rows: usize,
        minor_tile_dim: usize,
    ) -> Result<Self, &'static str> {
        if !check_folded_device_layout(ptr, n_rows, minor_tile_dim) {
            return Err("invalid mutable 128-lane physical tpu tile descriptor");
        }
        let slice = std::slice::from_raw_parts_mut(ptr as *mut Bf16, n_rows * MINOR_TILE_DIM);
        Ok(Self {
            slice,
            n_rows,
            is_4k_aligned: (ptr as usize) % 4096 == 0,
        })
    }
}
