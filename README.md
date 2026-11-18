# XASP: XLA-Aligned Spectral Protocol

[![CI](https://github.com/fy2ne/XASP/actions/workflows/ci_and_bench.yml/badge.svg)](https://github.com/fy2ne/XASP/actions/workflows/ci_and_bench.yml)

`XASP` is a pure-Rust server-side KV-cache transport engine built for disaggregated Google Cloud TPU serving (`google/tpu-sync`, `vllm-project/vllm-jax`, `vllm-project/vllm-torchtpu`) over 100% standard TCP (`TCP_NODELAY` + Linux `TCP_QUICKACK` / `SO_RCVLOWAT`).

External OpenAI HTTP/SSE clients and TPU Pallas attention kernels remain untouched. `XASP` operates strictly inside Host DRAM staging memory (`transfer_d2h_async` / `ExecuteLayerH2d`) directly along TPU's contiguous 128-lane physical minor tile axis.

---

## Target Bottleneck Inside `google/tpu-sync`

1. **Uncompressed DCN Wire Cliff (`RawBufferTransport` / `BlockTransport`):** While local PCIe Gen5 DMA runs at ~200-223 GB/s into 128-lane physical host tiles (`tpu_sync/core/raw_transfer.md`), inter-slice DCN TCP links push raw uncompressed `BF16` bytes (`~230 ms` median on large MoE KV handoffs).
2. **`StagingLayerManager` Head-of-Line Blocking (`CancelledError`):** Bounded host staging slots (`TPU_RAIDEN_NUM_STAGING_LAYERS`) stay pinned in `transfer_receive_session.cc` until 100% of `expected_chunks_per_layer[l]` arrive, stalling downstream layers on a single TCP retransmission until `deadline_` triggers `CancelledError("Receive session is draining")`.

---

## Architecture

- **`crates/xasp-math`**: Layer I subsampled Christoffel / Ridge Leverage stratification (`vip_rows` top 5% + 4 attention sinks in `INT8`), Layer II in-place `AVX2` / scalar 128-lane Fast Walsh-Hadamard Transform (`896` add/sub ops per row), Two-Stage Polar Lloyd-Max + 1-bit QJL residual compression, and unbiased `(1 - delta_hat)^-1` deadline reconstruction.
- **`crates/xasp-tcp`**: Cache-line-aligned 64-byte `XaspTcpFrameHeader` with hardware SSE4.2 `CRC32C`, `K=16` persistent striped TCP connection pool (`0..1` Priority VIP redundant streams, `2..15` Bulk coordinate streams), deadline assembler, and retired-layer late-frame drainer.
- **`crates/xasp-tpu-ffi`**: Zero-copy 128-lane physical tile pointer validator (`xasp_check_folded_device_layout`), C-ABI shared library (`libxasp_tpu.so` / `xasp_tpu.dll`), and Python `XaspConnector` adapter for `KVConnectorBase_V1`.
- **`crates/xasp-bench`**: Self-contained 16-socket loopback TCP benchmark suite and chart generator with synthetic `[512, 128]` `BF16` outlier tiles (`30x` minor-lane spikes + 4 attention sinks).

See [Mathematical Proofs](docs/math_proofs.md) and [Wire Protocol Specification](docs/wire_protocol.md).

---

## Benchmark Results (`xasp-bench`, 16 Loopback TCP Streams)

| DCN Micro-Burst Stall Prob | Raw TPU-Sync P99 (ms) | Raw Session Abort Rate | XASP K=16 P99 (ms) | XASP Abort Rate | Wire Compression |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `0.0%` | `230.0 ms` | `0.0%` | `36.7 ms` | `0.0%` | `6.26x` (`2.56 b/coord`) |
| `0.2%` | `490.9 ms` | `0.5%` | `36.7 ms` | `0.0%` | `6.26x` (`2.56 b/coord`) |
| `0.5%` | `634.2 ms` | `9.0%` | `36.8 ms` | `0.0%` | `6.26x` (`2.56 b/coord`) |
| `1.0%` | `906.8 ms` | `46.2%` | `36.8 ms` | `0.0%` | `6.26x` (`2.56 b/coord`) |
| `2.0%` | `980.0 ms` | `91.5%` | `36.8 ms` | `0.0%` | `6.26x` (`2.56 b/coord`) |

![Chart 1: DCN Transfer Cliff and Abort Rate](docs/assets/chart1_dcn_transfer_cliff.png)

![Chart 2: StagingLayerManager Bounded Slot Occupancy](docs/assets/chart2_staging_slot_occupancy.png)

![Chart 3: Attention RMSE vs Compression Ratio](docs/assets/chart3_accuracy_rmse_compression.png)

| Configuration | Compression Ratio | Stalled TCP Streams | Attention Rel. RMSE | Attention KL Div |
| :--- | :---: | :---: | :---: | :---: |
| `XASP (b=4, VIP=5.0%)` | `3.59x` | `0` | `1.18%` | `0.0000` |
| `XASP (b=3, VIP=5.0%)` | `4.56x` | `1` | `1.29%` | `0.0000` |
| `XASP (b=2, VIP=5.0%)` | `6.26x` | `1` | `1.44%` | `0.0000` |
| `XASP (b=2, VIP=5.0%)` | `6.26x` | `2` | `1.58%` | `0.0001` |
| `QJL-Only (No VIP, b=2)` | `7.11x` | `1` | `60.89%` | `0.0044` |
| `Naive Byte Stripe` | `4.00x` | `2` | `34.55%` | `0.0000` |
| `Unstratified Scalar INT` | `8.00x` | `1` | `231.95%` | `0.1066` |

---

## Quick Reproduction

Zero external Rust crate dependencies (`std` + target intrinsics only). Requires Rust `1.85+`:

```bash
RUSTFLAGS="-C target-cpu=native" cargo test --workspace --release
RUSTFLAGS="-C target-cpu=native" cargo run --release -p xasp-bench -- --out-dir ./bench_out
```

---

## Licensing

Dual-licensed under **`PolyForm-Noncommercial-1.0.0`** (for academic research and noncommercial evaluation) and a **Commercial License** from `fy2ne` (`https://xasp.fy2ne.me`). See [LICENSE.md](LICENSE.md).

Note: Source-available exclusively on GitHub (`https://github.com/fy2ne/XASP`). Not distributed via `crates.io`.
