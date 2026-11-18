// Copyright (c) 2026 fy2ne (https://xasp.fy2ne.me). All rights reserved.
// Dual-Licensed:
//   1. Noncommercial Evaluation & Academic Research: PolyForm-Noncommercial-1.0.0
//   2. Commercial / Production Deployment: Requires a paid Commercial License from fy2ne.
//      Source-available on https://github.com/fy2ne/xasp (not distributed via crates.io).

pub mod tile_arena_view;

use tile_arena_view::{
    PhysicalTileArenaView, PhysicalTileArenaViewMut, check_folded_device_layout,
};
use xasp_math::{LocalSpectralCodec, TileTransportBackend, decode_layer_half};
use xasp_tcp::stripe_mux::{MuxedFrame, mux_encoded_half};

#[repr(C)]
pub struct XaspRoundTripStats {
    pub compressed_bytes: u64,
    pub raw_bytes: u64,
    pub arrived_mask_lo: u64,
    pub arrived_mask_hi: u64,
}

#[no_mangle]
pub extern "C" fn xasp_abi_version() -> u32 {
    0x0001_0000
}

#[no_mangle]
pub unsafe extern "C" fn xasp_check_folded_device_layout(
    ptr: *const u16,
    n_rows: usize,
    minor_tile_dim: usize,
) -> i32 {
    if check_folded_device_layout(ptr, n_rows, minor_tile_dim) {
        1
    } else {
        0
    }
}

/// Simulates in-place physical 128-lane tile compression + deadline erasure round-trip
/// directly over Host DRAM buffers staged by tpu_sync::transfer_d2h_async.
#[no_mangle]
pub unsafe extern "C" fn xasp_roundtrip_physical_tiles_cabi(
    k_in_ptr: *const u16,
    v_in_ptr: *const u16,
    k_out_ptr: *mut u16,
    v_out_ptr: *mut u16,
    n_rows: usize,
    quant_bits: u8,
    stalled_stream_mask: u16,
    request_id_lo: u64,
    layer_id: u16,
    stats_out: *mut XaspRoundTripStats,
) -> i32 {
    let Ok(k_view) = PhysicalTileArenaView::from_raw(k_in_ptr, n_rows, 128) else {
        return -1;
    };
    let Ok(v_view) = PhysicalTileArenaView::from_raw(v_in_ptr, n_rows, 128) else {
        return -2;
    };
    let Ok(k_dst) = PhysicalTileArenaViewMut::from_raw_mut(k_out_ptr, n_rows, 128) else {
        return -3;
    };
    let Ok(v_dst) = PhysicalTileArenaViewMut::from_raw_mut(v_out_ptr, n_rows, 128) else {
        return -4;
    };

    let codec = LocalSpectralCodec::default();
    let (k_enc, v_enc) = codec.encode_kv_layer(
        k_view.slice,
        v_view.slice,
        n_rows,
        quant_bits,
        request_id_lo,
        layer_id,
    );

    let k_frames: Vec<MuxedFrame> =
        mux_encoded_half(&k_enc, false, request_id_lo, 0, layer_id, 0, 0);
    let v_frames: Vec<MuxedFrame> =
        mux_encoded_half(&v_enc, true, request_id_lo, 0, layer_id, 0, 0);

    // Count non-redundant wire bytes (stream 0 + streams 2..15)
    let mut comp_bytes = 0u64;
    for f in &k_frames {
        if f.header.stream_id != 1 {
            comp_bytes += f.payload.len() as u64;
        }
    }
    for f in &v_frames {
        if f.header.stream_id != 1 {
            comp_bytes += f.payload.len() as u64;
        }
    }

    let mut arrived_mask = 0u128;
    for f in &k_frames {
        let sid = f.header.stream_id;
        if sid >= 2 && ((stalled_stream_mask >> sid) & 1) == 0 {
            let bits = ((1u128 << f.header.coord_count) - 1) << f.header.coord_start;
            arrived_mask |= bits;
        }
    }

    let k_dec = decode_layer_half(&k_enc, arrived_mask, request_id_lo, layer_id, false);
    let v_dec = decode_layer_half(&v_enc, arrived_mask, request_id_lo, layer_id, true);
    k_dst.slice.copy_from_slice(&k_dec);
    v_dst.slice.copy_from_slice(&v_dec);

    if !stats_out.is_null() {
        *stats_out = XaspRoundTripStats {
            compressed_bytes: comp_bytes,
            raw_bytes: (n_rows * 128 * 2 * 2) as u64,
            arrived_mask_lo: arrived_mask as u64,
            arrived_mask_hi: (arrived_mask >> 64) as u64,
        };
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cabi_zero_copy_roundtrip_works() {
        let n_rows = 64usize;
        let k_in: Vec<u16> = (0..(n_rows * 128)).map(|i| 0x3C00 + ((i % 19) as u16)).collect();
        let v_in: Vec<u16> = (0..(n_rows * 128)).map(|i| 0x3B00 + ((i % 17) as u16)).collect();
        let mut k_out = vec![0u16; n_rows * 128];
        let mut v_out = vec![0u16; n_rows * 128];
        let mut stats = XaspRoundTripStats {
            compressed_bytes: 0,
            raw_bytes: 0,
            arrived_mask_lo: 0,
            arrived_mask_hi: 0,
        };

        let rc = unsafe {
            xasp_roundtrip_physical_tiles_cabi(
                k_in.as_ptr(),
                v_in.as_ptr(),
                k_out.as_mut_ptr(),
                v_out.as_mut_ptr(),
                n_rows,
                2,
                1u16 << 7,
                55,
                2,
                &mut stats,
            )
        };
        assert_eq!(rc, 0);
        assert!(stats.raw_bytes / stats.compressed_bytes >= 5);
    }
}
