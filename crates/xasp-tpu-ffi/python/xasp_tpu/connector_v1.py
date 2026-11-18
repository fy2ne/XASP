# Copyright (c) 2026 fy2ne (https://xasp.fy2ne.me). All rights reserved.
# Dual-Licensed:
#   1. Noncommercial Evaluation & Academic Research: PolyForm-Noncommercial-1.0.0
#   2. Commercial / Production Deployment: Requires a paid Commercial License from fy2ne.
#      Source-available on https://github.com/fy2ne/xasp (not distributed via crates.io).

import ctypes
import os
from pathlib import Path


def _load_xasp_lib() -> ctypes.CDLL:
    override = os.environ.get("XASP_TPU_LIB")
    if override:
        return ctypes.CDLL(override)

    root = Path(__file__).resolve().parents[4]
    candidates = [
        root / "target" / "release" / "libxasp_tpu.so",
        root / "target" / "release" / "xasp_tpu.dll",
        root / "target" / "debug" / "libxasp_tpu.so",
        root / "target" / "debug" / "xasp_tpu.dll",
    ]
    for path in candidates:
        if path.exists():
            return ctypes.CDLL(str(path))
    raise FileNotFoundError("xasp_tpu native library missing; build with cargo build --release -p xasp-tpu-ffi")


class XaspRoundTripStats(ctypes.Structure):
    _fields_ = [
        ("compressed_bytes", ctypes.c_uint64),
        ("raw_bytes", ctypes.c_uint64),
        ("arrived_mask_lo", ctypes.c_uint64),
        ("arrived_mask_hi", ctypes.c_uint64),
    ]


class XaspConnector:
    """Drop-in KVConnectorBase_V1 adapter for vllm-jax and vllm-torchtpu."""

    def __init__(self, quant_bits: int = 2, deadline_ms: float = 35.0):
        self.quant_bits = quant_bits
        self.deadline_ms = deadline_ms
        self._lib = _load_xasp_lib()
        self._lib.xasp_check_folded_device_layout.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self._lib.xasp_check_folded_device_layout.restype = ctypes.c_int32
        self._lib.xasp_roundtrip_physical_tiles_cabi.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint8,
            ctypes.c_uint16,
            ctypes.c_uint64,
            ctypes.c_uint16,
            ctypes.POINTER(XaspRoundTripStats),
        ]
        self._lib.xasp_roundtrip_physical_tiles_cabi.restype = ctypes.c_int32

    def check_folded_device_layout(self, host_ptr: int, n_rows: int, minor_tile_dim: int = 128) -> bool:
        return bool(self._lib.xasp_check_folded_device_layout(host_ptr, n_rows, minor_tile_dim))

    def start_load_kv(self, layer_idx: int, staging_slot_idx: int) -> dict:
        return {
            "layer_id": layer_idx,
            "staging_slot_idx": staging_slot_idx,
            "deadline_ms": self.deadline_ms,
        }

    def save_kv_layer(
        self,
        k_in_ptr: int,
        v_in_ptr: int,
        k_out_ptr: int,
        v_out_ptr: int,
        n_rows: int,
        request_id_lo: int,
        layer_id: int,
        stalled_stream_mask: int = 0,
    ) -> XaspRoundTripStats:
        stats = XaspRoundTripStats()
        rc = self._lib.xasp_roundtrip_physical_tiles_cabi(
            k_in_ptr,
            v_in_ptr,
            k_out_ptr,
            v_out_ptr,
            n_rows,
            self.quant_bits,
            stalled_stream_mask,
            request_id_lo,
            layer_id,
            ctypes.byref(stats),
        )
        if rc != 0:
            raise RuntimeError(f"xasp_roundtrip_physical_tiles_cabi failed: {rc}")
        return stats

    def wait_for_layer_load(self, handle: dict) -> bool:
        return True
