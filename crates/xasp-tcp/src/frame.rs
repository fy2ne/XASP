// Copyright (c) 2026 fy2ne (https://xasp.fy2ne.me). All rights reserved.
// Dual-Licensed:
//   1. Noncommercial Evaluation & Academic Research: PolyForm-Noncommercial-1.0.0
//   2. Commercial / Production Deployment: Requires a paid Commercial License from fy2ne.
//      Source-available on https://github.com/fy2ne/xasp (not distributed via crates.io).

pub const XASP_MAGIC: u32 = 0x5841_5350;
pub const XASP_VERSION: u8 = 0x01;

pub const FLAG_IS_VALUE_TENSOR: u8 = 1 << 0;
pub const FLAG_IS_REDUNDANT_VIP_COPY: u8 = 1 << 1;
pub const FLAG_IS_LAST_STRIPE_IN_LAYER: u8 = 1 << 2;
pub const FLAG_PHYSICAL_TPU_TILE_128: u8 = 1 << 3;

pub const TIER_STRATUM0_VIP: u8 = 0x00;
pub const TIER_STRATUM1_BULK: u8 = 0x01;

#[repr(C, align(64))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct XaspTcpFrameHeader {
    pub magic: u32,
    pub version: u8,
    pub flags: u8,
    pub stream_id: u8,
    pub stratum_tier: u8,
    pub staging_slot_idx: u8,
    pub quant_bits_br: u8,
    pub layer_id: u16,
    pub coord_start: u16,
    pub coord_count: u16,
    pub request_id_lo: u64,
    pub request_id_hi: u64,
    pub hadamard_seed: u64,
    pub deadline_ns: u64,
    pub payload_len: u32,
    pub token_offset: u32,
    pub num_tokens: u16,
    pub head_dim_and_kv: u16,
    pub header_crc32c: u32,
}

const _: () = assert!(std::mem::size_of::<XaspTcpFrameHeader>() == 64);
const _: () = assert!(std::mem::align_of::<XaspTcpFrameHeader>() == 64);

fn crc32c_sw(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0x82F6_3B78 & mask);
        }
    }
    !crc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_sse42(bytes: &[u8]) -> u32 {
    use std::arch::x86_64::{_mm_crc32_u8, _mm_crc32_u64};
    let mut crc = 0xFFFF_FFFFu32;
    let mut pos = 0usize;
    while pos + 8 <= bytes.len() {
        let word = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        crc = _mm_crc32_u64(crc as u64, word) as u32;
        pos += 8;
    }
    while pos < bytes.len() {
        crc = _mm_crc32_u8(crc, bytes[pos]);
        pos += 1;
    }
    !crc
}

#[inline]
pub fn crc32c(bytes: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse4.2") {
            return unsafe { crc32c_sse42(bytes) };
        }
    }
    crc32c_sw(bytes)
}

impl XaspTcpFrameHeader {
    pub fn seal(&mut self) {
        let raw = self.pack_body_60();
        self.header_crc32c = crc32c(&raw);
    }

    fn pack_body_60(&self) -> [u8; 60] {
        let mut out = [0u8; 60];
        out[0x00..0x04].copy_from_slice(&self.magic.to_le_bytes());
        out[0x04] = self.version;
        out[0x05] = self.flags;
        out[0x06] = self.stream_id;
        out[0x07] = self.stratum_tier;
        out[0x08] = self.staging_slot_idx;
        out[0x09] = self.quant_bits_br;
        out[0x0A..0x0C].copy_from_slice(&self.layer_id.to_le_bytes());
        out[0x0C..0x0E].copy_from_slice(&self.coord_start.to_le_bytes());
        out[0x0E..0x10].copy_from_slice(&self.coord_count.to_le_bytes());
        out[0x10..0x18].copy_from_slice(&self.request_id_lo.to_le_bytes());
        out[0x18..0x20].copy_from_slice(&self.request_id_hi.to_le_bytes());
        out[0x20..0x28].copy_from_slice(&self.hadamard_seed.to_le_bytes());
        out[0x28..0x30].copy_from_slice(&self.deadline_ns.to_le_bytes());
        out[0x30..0x34].copy_from_slice(&self.payload_len.to_le_bytes());
        out[0x34..0x38].copy_from_slice(&self.token_offset.to_le_bytes());
        out[0x38..0x3A].copy_from_slice(&self.num_tokens.to_le_bytes());
        out[0x3A..0x3C].copy_from_slice(&self.head_dim_and_kv.to_le_bytes());
        out
    }

    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        let body = self.pack_body_60();
        out[..60].copy_from_slice(&body);
        let csum = crc32c(&body);
        out[60..64].copy_from_slice(&csum.to_le_bytes());
        out
    }

    pub fn from_bytes(raw: &[u8; 64]) -> Result<Self, &'static str> {
        let magic = u32::from_le_bytes(raw[0x00..0x04].try_into().unwrap());
        if magic != XASP_MAGIC {
            return Err("bad magic");
        }
        let version = raw[0x04];
        if version != XASP_VERSION {
            return Err("bad version");
        }
        let header_crc32c = u32::from_le_bytes(raw[0x3C..0x40].try_into().unwrap());
        if crc32c(&raw[..60]) != header_crc32c {
            return Err("crc32c mismatch");
        }

        Ok(Self {
            magic,
            version,
            flags: raw[0x05],
            stream_id: raw[0x06],
            stratum_tier: raw[0x07],
            staging_slot_idx: raw[0x08],
            quant_bits_br: raw[0x09],
            layer_id: u16::from_le_bytes(raw[0x0A..0x0C].try_into().unwrap()),
            coord_start: u16::from_le_bytes(raw[0x0C..0x0E].try_into().unwrap()),
            coord_count: u16::from_le_bytes(raw[0x0E..0x10].try_into().unwrap()),
            request_id_lo: u64::from_le_bytes(raw[0x10..0x18].try_into().unwrap()),
            request_id_hi: u64::from_le_bytes(raw[0x18..0x20].try_into().unwrap()),
            hadamard_seed: u64::from_le_bytes(raw[0x20..0x28].try_into().unwrap()),
            deadline_ns: u64::from_le_bytes(raw[0x28..0x30].try_into().unwrap()),
            payload_len: u32::from_le_bytes(raw[0x30..0x34].try_into().unwrap()),
            token_offset: u32::from_le_bytes(raw[0x34..0x38].try_into().unwrap()),
            num_tokens: u16::from_le_bytes(raw[0x38..0x3A].try_into().unwrap()),
            head_dim_and_kv: u16::from_le_bytes(raw[0x3A..0x3C].try_into().unwrap()),
            header_crc32c,
        })
    }
}
