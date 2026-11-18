// Copyright (c) 2026 fy2ne (https://xasp.fy2ne.me). All rights reserved.
// Dual-Licensed:
//   1. Noncommercial Evaluation & Academic Research: PolyForm-Noncommercial-1.0.0
//   2. Commercial / Production Deployment: Requires a paid Commercial License from fy2ne.
//      Source-available on https://github.com/fy2ne/xasp (not distributed via crates.io).

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

extern "C" {

struct XaspStripeIovec {
    uint32_t offset;
    uint32_t len;
    uint8_t stream_id;
    uint8_t stratum_tier;
    uint8_t is_value;
    uint8_t _pad;
    uint16_t coord_start;
    uint16_t coord_count;
};

uint32_t xasp_abi_version();

int32_t xasp_check_folded_device_layout(
    const uint16_t* ptr, size_t n_rows, size_t minor_tile_dim);

int32_t xasp_encode_kv_to_wire_blob(
    const uint16_t* k_in_ptr,
    const uint16_t* v_in_ptr,
    size_t n_rows,
    uint8_t quant_bits,
    uint64_t request_id_lo,
    uint16_t layer_id,
    uint8_t staging_slot_idx,
    uint64_t deadline_ns,
    uint8_t* out_wire_buf,
    size_t out_wire_cap,
    XaspStripeIovec* out_iovecs_32,
    size_t* out_total_bytes);

int32_t xasp_decode_kv_from_wire_blob(
    const uint8_t* const* frame_ptrs,
    const size_t* frame_lens,
    size_t num_frames,
    uint16_t* k_out_ptr,
    uint16_t* v_out_ptr,
    size_t n_rows);

}  // extern "C"

namespace xasp {

// Drop-in wrapper over google/tpu-sync Host DRAM staging pointers (transfer_d2h_async)
// and google/peregrine Post() iovecs. Touches zero PJRT, XLA, or client HTTP/SSE code.
class RaidenStagingAdapter {
public:
    explicit RaidenStagingAdapter(uint8_t quant_bits = 2) : quant_bits_(quant_bits) {}

    bool CompressHostTile(
        const uint16_t* k_host_bf16,
        const uint16_t* v_host_bf16,
        size_t n_rows,
        uint64_t request_id_lo,
        uint16_t layer_id,
        uint8_t staging_slot_idx,
        uint64_t deadline_ns,
        std::vector<uint8_t>& wire_buf,
        XaspStripeIovec iovecs_out[32]) const {
        if (!xasp_check_folded_device_layout(k_host_bf16, n_rows, 128) ||
            !xasp_check_folded_device_layout(v_host_bf16, n_rows, 128)) {
            return false;
        }
        if (wire_buf.size() < n_rows * 192) {
            wire_buf.resize(n_rows * 192);
        }
        size_t total_bytes = 0;
        int32_t rc = xasp_encode_kv_to_wire_blob(
            k_host_bf16,
            v_host_bf16,
            n_rows,
            quant_bits_,
            request_id_lo,
            layer_id,
            staging_slot_idx,
            deadline_ns,
            wire_buf.data(),
            wire_buf.size(),
            iovecs_out,
            &total_bytes);
        if (rc == -11 && total_bytes > wire_buf.size()) {
            wire_buf.resize(total_bytes);
            rc = xasp_encode_kv_to_wire_blob(
                k_host_bf16,
                v_host_bf16,
                n_rows,
                quant_bits_,
                request_id_lo,
                layer_id,
                staging_slot_idx,
                deadline_ns,
                wire_buf.data(),
                wire_buf.size(),
                iovecs_out,
                &total_bytes);
        }
        return rc == 0;
    }

    bool ReconstructHostTileAtDeadline(
        const std::vector<const uint8_t*>& arrived_frames,
        const std::vector<size_t>& frame_lengths,
        uint16_t* k_host_out,
        uint16_t* v_host_out,
        size_t n_rows) const {
        if (arrived_frames.size() != frame_lengths.size() || arrived_frames.empty()) {
            return false;
        }
        int32_t rc = xasp_decode_kv_from_wire_blob(
            arrived_frames.data(),
            frame_lengths.data(),
            arrived_frames.size(),
            k_host_out,
            v_host_out,
            n_rows);
        return rc == 0;
    }

private:
    uint8_t quant_bits_;
};

}  // namespace xasp
