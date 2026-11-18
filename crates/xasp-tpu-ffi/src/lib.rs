// Copyright (c) 2026 fy2ne (https://xasp.fy2ne.me). All rights reserved.
// Dual-Licensed:
//   1. Noncommercial Evaluation & Academic Research: PolyForm-Noncommercial-1.0.0
//   2. Commercial / Production Deployment: Requires a paid Commercial License from fy2ne.
//      Source-available on https://github.com/fy2ne/xasp (not distributed via crates.io).

pub mod tile_arena_view;

use tile_arena_view::{
    PhysicalTileArenaView, PhysicalTileArenaViewMut, check_folded_device_layout,
};
use xasp_math::polar_qjl::unpack_bulk_stripe_into;
use xasp_math::{LocalSpectralCodec, decode_layer_half};
use xasp_tcp::frame::{FLAG_IS_VALUE_TENSOR, XaspTcpFrameHeader};
use xasp_tcp::stripe_mux::{
    MuxedFrame, decode_stratum0_payload, mux_encoded_half, stream_coord_range,
};

#[repr(C)]
pub struct XaspRoundTripStats {
    pub compressed_bytes: u64,
    pub raw_bytes: u64,
    pub arrived_mask_lo: u64,
    pub arrived_mask_hi: u64,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct XaspStripeIovec {
    pub offset: u32,
    pub len: u32,
    pub stream_id: u8,
    pub stratum_tier: u8,
    pub is_value: u8,
    pub _pad: u8,
    pub coord_start: u16,
    pub coord_count: u16,
}

#[no_mangle]
pub extern "C" fn xasp_abi_version() -> u32 {
    0x0001_0001
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

/// Encodes Host DRAM [n_rows, 128] BF16 Key and Value tiles into a caller-owned
/// staging wire buffer and emits 32 XaspStripeIovec descriptors (16 K + 16 V frames).
#[no_mangle]
pub unsafe extern "C" fn xasp_encode_kv_to_wire_blob(
    k_in_ptr: *const u16,
    v_in_ptr: *const u16,
    n_rows: usize,
    quant_bits: u8,
    request_id_lo: u64,
    layer_id: u16,
    staging_slot_idx: u8,
    deadline_ns: u64,
    out_wire_buf: *mut u8,
    out_wire_cap: usize,
    out_iovecs_32: *mut XaspStripeIovec,
    out_total_bytes: *mut usize,
) -> i32 {
    if out_wire_buf.is_null() || out_iovecs_32.is_null() || out_total_bytes.is_null() {
        return -10;
    }
    let Ok(k_view) = PhysicalTileArenaView::from_raw(k_in_ptr, n_rows, 128) else {
        return -1;
    };
    let Ok(v_view) = PhysicalTileArenaView::from_raw(v_in_ptr, n_rows, 128) else {
        return -2;
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

    let k_frames = mux_encoded_half(
        &k_enc,
        false,
        request_id_lo,
        0,
        layer_id,
        staging_slot_idx,
        deadline_ns,
    );
    let v_frames = mux_encoded_half(
        &v_enc,
        true,
        request_id_lo,
        0,
        layer_id,
        staging_slot_idx,
        deadline_ns,
    );

    let mut total_needed = 0usize;
    for f in k_frames.iter().chain(v_frames.iter()) {
        total_needed += 64 + f.payload.len();
    }
    *out_total_bytes = total_needed;
    if total_needed > out_wire_cap {
        return -11;
    }

    let wire_slice = std::slice::from_raw_parts_mut(out_wire_buf, out_wire_cap);
    let iovecs = std::slice::from_raw_parts_mut(out_iovecs_32, 32);
    let mut cursor = 0usize;

    for (idx, frame) in k_frames.iter().chain(v_frames.iter()).enumerate() {
        let frame_len = 64 + frame.payload.len();
        let hdr_bytes = frame.header.to_bytes();
        wire_slice[cursor..cursor + 64].copy_from_slice(&hdr_bytes);
        wire_slice[cursor + 64..cursor + frame_len].copy_from_slice(&frame.payload);

        iovecs[idx] = XaspStripeIovec {
            offset: cursor as u32,
            len: frame_len as u32,
            stream_id: frame.header.stream_id,
            stratum_tier: frame.header.stratum_tier,
            is_value: if (frame.header.flags & FLAG_IS_VALUE_TENSOR) != 0 {
                1
            } else {
                0
            },
            _pad: 0,
            coord_start: frame.header.coord_start,
            coord_count: frame.header.coord_count,
        };
        cursor += frame_len;
    }

    0
}

/// Reconstructs Host DRAM [n_rows, 128] BF16 Key and Value tiles from any arrived subset
/// of XASP wire frames at layer deadline tau_l.
#[no_mangle]
pub unsafe extern "C" fn xasp_decode_kv_from_wire_blob(
    frame_ptrs: *const *const u8,
    frame_lens: *const usize,
    num_frames: usize,
    k_out_ptr: *mut u16,
    v_out_ptr: *mut u16,
    n_rows: usize,
) -> i32 {
    if frame_ptrs.is_null() || frame_lens.is_null() || num_frames == 0 {
        return -10;
    }
    let Ok(k_dst) = PhysicalTileArenaViewMut::from_raw_mut(k_out_ptr, n_rows, 128) else {
        return -3;
    };
    let Ok(v_dst) = PhysicalTileArenaViewMut::from_raw_mut(v_out_ptr, n_rows, 128) else {
        return -4;
    };

    let ptrs = std::slice::from_raw_parts(frame_ptrs, num_frames);
    let lens = std::slice::from_raw_parts(frame_lens, num_frames);

    let mut k_vip_hdr: Option<XaspTcpFrameHeader> = None;
    let mut k_vip_body: Option<&[u8]> = None;
    let mut v_vip_hdr: Option<XaspTcpFrameHeader> = None;
    let mut v_vip_body: Option<&[u8]> = None;
    let mut k_bulk: Vec<(XaspTcpFrameHeader, &[u8])> = Vec::with_capacity(14);
    let mut v_bulk: Vec<(XaspTcpFrameHeader, &[u8])> = Vec::with_capacity(14);

    for i in 0..num_frames {
        let len = lens[i];
        if ptrs[i].is_null() || len < 64 {
            continue;
        }
        let raw = std::slice::from_raw_parts(ptrs[i], len);
        let hdr_arr: &[u8; 64] = match raw[..64].try_into() {
            Ok(a) => a,
            Err(_) => continue,
        };
        let Ok(hdr) = XaspTcpFrameHeader::from_bytes(hdr_arr) else {
            return -12;
        };
        if len < 64 + (hdr.payload_len as usize) {
            return -13;
        }
        let body = &raw[64..64 + (hdr.payload_len as usize)];
        let is_val = (hdr.flags & FLAG_IS_VALUE_TENSOR) != 0;

        if hdr.stream_id < 2 {
            if is_val {
                if v_vip_hdr.is_none() {
                    v_vip_hdr = Some(hdr);
                    v_vip_body = Some(body);
                }
            } else if k_vip_hdr.is_none() {
                k_vip_hdr = Some(hdr);
                k_vip_body = Some(body);
            }
        } else if hdr.stream_id < 16 {
            if is_val {
                v_bulk.push((hdr, body));
            } else {
                k_bulk.push((hdr, body));
            }
        }
    }

    let Some(kh) = k_vip_hdr else { return -14 };
    let Some(kb) = k_vip_body else { return -14 };
    let Some(vh) = v_vip_hdr else { return -15 };
    let Some(vb) = v_vip_body else { return -15 };

    let k_qbits = kh.quant_bits_br.clamp(2, 4);
    let v_qbits = vh.quant_bits_br.clamp(2, 4);

    let Ok(mut k_enc) = decode_stratum0_payload(kb, k_qbits) else {
        return -16;
    };
    let Ok(mut v_enc) = decode_stratum0_payload(vb, v_qbits) else {
        return -17;
    };
    if k_enc.n_rows != n_rows || v_enc.n_rows != n_rows {
        return -18;
    }

    let mut k_arrived = 0u128;
    for (hdr, body) in k_bulk {
        if let Some((c_start, c_count)) = stream_coord_range(hdr.stream_id as usize) {
            unpack_bulk_stripe_into(body, &mut k_enc.bulk_rows, c_start, c_count, hdr.quant_bits_br);
            k_arrived |= ((1u128 << c_count) - 1) << c_start;
        }
    }

    let mut v_arrived = 0u128;
    for (hdr, body) in v_bulk {
        if let Some((c_start, c_count)) = stream_coord_range(hdr.stream_id as usize) {
            unpack_bulk_stripe_into(body, &mut v_enc.bulk_rows, c_start, c_count, hdr.quant_bits_br);
            v_arrived |= ((1u128 << c_count) - 1) << c_start;
        }
    }

    let k_dec = decode_layer_half(&k_enc, k_arrived, kh.request_id_lo, kh.layer_id, false);
    let v_dec = decode_layer_half(&v_enc, v_arrived, vh.request_id_lo, vh.layer_id, true);
    k_dst.slice.copy_from_slice(&k_dec);
    v_dst.slice.copy_from_slice(&v_dec);
    0
}

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

    #[test]
    fn cabi_wire_blob_iovec_roundtrip_works() {
        let n_rows = 128usize;
        let k_in: Vec<u16> = (0..(n_rows * 128)).map(|i| 0x3C00 + ((i % 23) as u16)).collect();
        let v_in: Vec<u16> = (0..(n_rows * 128)).map(|i| 0x3B00 + ((i % 19) as u16)).collect();
        let mut wire_buf = vec![0u8; 64 * 1024];
        let mut iovecs = [XaspStripeIovec::default(); 32];
        let mut total_bytes = 0usize;

        let enc_rc = unsafe {
            xasp_encode_kv_to_wire_blob(
                k_in.as_ptr(),
                v_in.as_ptr(),
                n_rows,
                2,
                901,
                4,
                1,
                5_000_000,
                wire_buf.as_mut_ptr(),
                wire_buf.len(),
                iovecs.as_mut_ptr(),
                &mut total_bytes,
            )
        };
        assert_eq!(enc_rc, 0);
        assert!(total_bytes > 0 && total_bytes < wire_buf.len());

        // Drop stream 7 (both K and V) to simulate a DCN stall before tau_l
        let mut ptrs: Vec<*const u8> = Vec::new();
        let mut lens: Vec<usize> = Vec::new();
        for iov in &iovecs {
            if iov.stream_id == 7 {
                continue;
            }
            ptrs.push(unsafe { wire_buf.as_ptr().add(iov.offset as usize) });
            lens.push(iov.len as usize);
        }

        let mut k_out = vec![0u16; n_rows * 128];
        let mut v_out = vec![0u16; n_rows * 128];
        let dec_rc = unsafe {
            xasp_decode_kv_from_wire_blob(
                ptrs.as_ptr(),
                lens.as_ptr(),
                ptrs.len(),
                k_out.as_mut_ptr(),
                v_out.as_mut_ptr(),
                n_rows,
            )
        };
        assert_eq!(dec_rc, 0);
        assert!(k_out.iter().any(|&w| w != 0));
    }
}
