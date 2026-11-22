// Copyright (c) 2026 fy2ne
// SPDX-License-Identifier: Apache-2.0

pub mod deadline_rx;
pub mod frame;
pub mod pool;
pub mod stripe_mux;

#[cfg(test)]
mod tests {
    use crate::deadline_rx::DeadlineAssembler;
    use crate::frame::{XASP_MAGIC, XASP_VERSION, XaspTcpFrameHeader};
    use crate::pool::bind_loopback_pair;
    use crate::stripe_mux::mux_encoded_half;
    use std::io::Write;
    use std::time::{Duration, Instant};
    use xasp_math::{Bf16, LocalSpectralCodec};

    #[test]
    fn frame_header_size_align_and_crc32c_roundtrip() {
        let mut hdr = XaspTcpFrameHeader {
            magic: XASP_MAGIC,
            version: XASP_VERSION,
            flags: 0x09,
            stream_id: 7,
            stratum_tier: 1,
            staging_slot_idx: 2,
            quant_bits_br: 2,
            layer_id: 11,
            coord_start: 45,
            coord_count: 9,
            request_id_lo: 0x1122_3344_5566_7788,
            request_id_hi: 0xAABB_CCDD_EEFF_0011,
            hadamard_seed: 0xDEAD_BEEF,
            deadline_ns: 999_999,
            payload_len: 384,
            token_offset: 0,
            num_tokens: 256,
            head_dim_and_kv: 0x0180,
            header_crc32c: 0,
        };
        hdr.seal();
        let bytes = hdr.to_bytes();
        let parsed = XaspTcpFrameHeader::from_bytes(&bytes).expect("valid hdr");
        assert_eq!(parsed.layer_id, 11);
        assert_eq!(parsed.stream_id, 7);

        let mut bad = bytes;
        bad[15] ^= 0x01;
        assert!(XaspTcpFrameHeader::from_bytes(&bad).is_err());
    }

    #[test]
    fn loopback_16_stream_deadline_and_late_drain() {
        let (mut tx, mut rx) = bind_loopback_pair(0).expect("loopback bind");
        let n_rows = 128usize;
        let mut keys = Vec::with_capacity(n_rows * 128);
        let mut vals = Vec::with_capacity(n_rows * 128);
        for r in 0..n_rows {
            for c in 0..128 {
                let s = ((r * 37 + c * 13) % 97) as f32 / 97.0 - 0.5;
                keys.push(Bf16::from_f32(s * if r < 4 { 8.0 } else { 1.0 }));
                vals.push(Bf16::from_f32(s * 0.8));
            }
        }

        let codec = LocalSpectralCodec::default();
        let (k_half, _) = codec.encode_kv_layer(&keys, &vals, n_rows, 2, 777, 1);
        let frames = mux_encoded_half(&k_half, false, 777, 1, 1, 0, 1_000_000);

        let mut asm = DeadlineAssembler::new();
        // stall stream 7 past the 12ms deadline
        let mut delayed_sock = None;
        let mut delayed_bytes = Vec::new();
        for f in &frames {
            let sid = f.header.stream_id as usize;
            let hdr_bytes = f.header.to_bytes();
            if sid == 7 {
                delayed_sock = Some(tx.streams[sid].try_clone().unwrap());
                delayed_bytes.extend_from_slice(&hdr_bytes);
                delayed_bytes.extend_from_slice(&f.payload);
            } else {
                tx.streams[sid].write_all(&hdr_bytes).unwrap();
                tx.streams[sid].write_all(&f.payload).unwrap();
            }
        }
        let joiner = delayed_sock.map(|mut sock| {
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(35));
                let _ = sock.write_all(&delayed_bytes);
            })
        });

        let assembled = asm
            .collect_layer_until_deadline(&mut rx, 777, 1, Instant::now() + Duration::from_millis(12))
            .expect("deadline assemble");

        assert_eq!(assembled.arrived_streams, 15);
        assert!(assembled.delta_hat > 0.06 && assembled.delta_hat < 0.08);
        assert_eq!(assembled.tile_bf16.len(), n_rows * 128);

        if let Some(h) = joiner {
            h.join().unwrap();
        }
        std::thread::sleep(Duration::from_millis(10));
        asm.drain_late_frames(&mut rx).unwrap();
        assert_eq!(asm.stats.late_drained_frames, 1);
    }
}
