// Copyright (c) 2026 fy2ne
// SPDX-License-Identifier: Apache-2.0

use crate::frame::{
    FLAG_IS_LAST_STRIPE_IN_LAYER, FLAG_IS_REDUNDANT_VIP_COPY, FLAG_IS_VALUE_TENSOR,
    FLAG_PHYSICAL_TPU_TILE_128, TIER_STRATUM0_VIP, TIER_STRATUM1_BULK, XASP_MAGIC, XASP_VERSION,
    XaspTcpFrameHeader,
};
use xasp_math::EncodedLayerHalf;
use xasp_math::christoffel::VipRowInt8;
use xasp_math::polar_qjl::{BulkEncodedRow, pack_bulk_stripe};

pub const NUM_STREAMS: usize = 16;
pub const BULK_STREAMS: usize = 14;

#[inline]
pub fn stream_coord_range(stream_id: usize) -> Option<(u16, u16)> {
    if !(2..NUM_STREAMS).contains(&stream_id) {
        return None;
    }
    let idx = stream_id - 2;
    if idx < 12 {
        Some(((idx * 9) as u16, 9))
    } else {
        Some(((108 + (idx - 12) * 10) as u16, 10))
    }
}

#[derive(Clone, Debug)]
pub struct MuxedFrame {
    pub header: XaspTcpFrameHeader,
    pub payload: Vec<u8>,
}

pub fn encode_stratum0_payload(half: &EncodedLayerHalf) -> Vec<u8> {
    let cap = 6 + half.vip_rows.len() * 132 + half.bulk_rows.len() * 4;
    let mut buf = Vec::with_capacity(cap);
    buf.extend_from_slice(&(half.n_rows as u16).to_le_bytes());
    buf.extend_from_slice(&(half.vip_rows.len() as u16).to_le_bytes());
    buf.extend_from_slice(&(half.bulk_rows.len() as u16).to_le_bytes());
    for vip in &half.vip_rows {
        vip.append_bytes(&mut buf);
    }
    for bulk in &half.bulk_rows {
        buf.extend_from_slice(&bulk.rho_x_f16.to_le_bytes());
        buf.extend_from_slice(&bulk.rho_r_f16.to_le_bytes());
    }
    buf
}

pub fn decode_stratum0_payload(
    payload: &[u8],
    quant_bits: u8,
) -> Result<EncodedLayerHalf, &'static str> {
    if payload.len() < 6 {
        return Err("stratum0 payload truncated");
    }
    let n_rows = u16::from_le_bytes([payload[0], payload[1]]) as usize;
    let n_vip = u16::from_le_bytes([payload[2], payload[3]]) as usize;
    let n_bulk = u16::from_le_bytes([payload[4], payload[5]]) as usize;

    let need = 6 + n_vip * 132 + n_bulk * 4;
    if payload.len() < need {
        return Err("stratum0 body short");
    }

    let mut is_vip = vec![false; n_rows];
    let mut vip_rows = Vec::with_capacity(n_vip);
    let mut off = 6usize;
    for _ in 0..n_vip {
        let vip = VipRowInt8::from_bytes(&payload[off..off + 132]).ok_or("bad vip row")?;
        if (vip.row_idx as usize) < n_rows {
            is_vip[vip.row_idx as usize] = true;
        }
        vip_rows.push(vip);
        off += 132;
    }

    let mut bulk_rows = Vec::with_capacity(n_bulk);
    let mut row_cursor = 0usize;
    for _ in 0..n_bulk {
        while row_cursor < n_rows && is_vip[row_cursor] {
            row_cursor += 1;
        }
        let rho_x_f16 = u16::from_le_bytes([payload[off], payload[off + 1]]);
        let rho_r_f16 = u16::from_le_bytes([payload[off + 2], payload[off + 3]]);
        off += 4;
        bulk_rows.push(BulkEncodedRow {
            row_idx: row_cursor as u16,
            rho_x_f16,
            rho_r_f16,
            q_plane0: 0,
            q_plane1: 0,
            q_plane2: 0,
            qjl_signs: 0,
        });
        row_cursor += 1;
    }

    Ok(EncodedLayerHalf {
        n_rows,
        quant_bits,
        vip_rows,
        bulk_rows,
    })
}

pub fn mux_encoded_half(
    half: &EncodedLayerHalf,
    is_value: bool,
    request_id_lo: u64,
    request_id_hi: u64,
    layer_id: u16,
    staging_slot_idx: u8,
    deadline_ns: u64,
) -> Vec<MuxedFrame> {
    let base_flags =
        FLAG_PHYSICAL_TPU_TILE_128 | if is_value { FLAG_IS_VALUE_TENSOR } else { 0 };
    let mut frames = Vec::with_capacity(NUM_STREAMS);
    let vip_payload = encode_stratum0_payload(half);

    for s in 0..2u8 {
        let mut hdr = XaspTcpFrameHeader {
            magic: XASP_MAGIC,
            version: XASP_VERSION,
            flags: base_flags | if s == 1 { FLAG_IS_REDUNDANT_VIP_COPY } else { 0 },
            stream_id: s,
            stratum_tier: TIER_STRATUM0_VIP,
            staging_slot_idx,
            quant_bits_br: half.quant_bits,
            layer_id,
            coord_start: 0,
            coord_count: half.vip_rows.len() as u16,
            request_id_lo,
            request_id_hi,
            hadamard_seed: request_id_lo ^ (layer_id as u64),
            deadline_ns,
            payload_len: vip_payload.len() as u32,
            token_offset: 0,
            num_tokens: half.n_rows as u16,
            head_dim_and_kv: 0x0180,
            header_crc32c: 0,
        };
        hdr.seal();
        frames.push(MuxedFrame {
            header: hdr,
            payload: vip_payload.clone(),
        });
    }

    for s in 2..NUM_STREAMS {
        let (coord_start, coord_count) = stream_coord_range(s).unwrap();
        let payload = pack_bulk_stripe(&half.bulk_rows, coord_start, coord_count, half.quant_bits);
        let mut hdr = XaspTcpFrameHeader {
            magic: XASP_MAGIC,
            version: XASP_VERSION,
            flags: base_flags | if s == NUM_STREAMS - 1 { FLAG_IS_LAST_STRIPE_IN_LAYER } else { 0 },
            stream_id: s as u8,
            stratum_tier: TIER_STRATUM1_BULK,
            staging_slot_idx,
            quant_bits_br: half.quant_bits,
            layer_id,
            coord_start,
            coord_count,
            request_id_lo,
            request_id_hi,
            hadamard_seed: request_id_lo ^ (layer_id as u64),
            deadline_ns,
            payload_len: payload.len() as u32,
            token_offset: 0,
            num_tokens: half.bulk_rows.len() as u16,
            head_dim_and_kv: 0x0180,
            header_crc32c: 0,
        };
        hdr.seal();
        frames.push(MuxedFrame {
            header: hdr,
            payload,
        });
    }

    frames
}
