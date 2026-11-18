# XASP Architecture Reference

This document maps every crate, module, symbol, memory layout, and integration boundary implemented in this repository. It makes no forward-looking claims; every struct, function, SIMD intrinsic, and socket option listed here is compiled and verified in the current workspace.

---

## 1. Execution & Integration Boundary Inside `google/tpu-sync`

```text
Prefill TPU HBM
  |  tpu_sync::transfer_d2h_async (~200 GB/s PCIe Gen5)
  v
Host DRAM Physical Tile ([N_rows, 128] BF16, 128-lane minor axis)
  |  xasp::RaidenStagingAdapter::CompressHostTile  (xasp_tpu_sync_adapter.hpp)
  |    -> xasp_check_folded_device_layout(ptr, n_rows, 128)
  |    -> xasp_encode_kv_to_wire_blob -> 32 XaspStripeIovec frames (16 K + 16 V)
  v
K=16 Standard TCP Sockets (google/peregrine Post() or RawBufferTransport)
  |  Streams 0..1  : Stratum-0 VIP INT8 + bulk row scales (2x redundant)
  |  Streams 2..15 : Stratum-1 128-lane Hadamard-QJL coordinate stripes
  v
Decode Host DRAM (StagingLayerManager slot)
  |  DeadlineAssembler::collect_layer_until_deadline (tau_l)
  |  xasp::RaidenStagingAdapter::ReconstructHostTileAtDeadline
  |    -> xasp_decode_kv_from_wire_blob -> unbiased (1 - delta_hat)^-1 tile
  |    -> ExecuteLayerH2d(l) + StagingLayerManager::Release(l)
  |    -> RetiredLayerFilter drains late packets without resetting sockets
  v
Decode TPU HBM (unchanged Pallas / XLA attention kernels; 0 client SDK changes)
```

Nothing in `XASP` alters client OpenAI HTTP/SSE requests, PJRT `CommonPjRtBuffer` descriptors, model weights, or compiled TPU HLO graphs.

---

## 2. Crate 1: `crates/xasp-math` (Host CPU Tile Compression & Reconstruction)

Zero OS, network, or external crate dependencies. Operates directly on contiguous `[n_rows, 128]` `Bf16` slices (`MINOR_TILE_DIM = 128`).

### 2.1 `src/lib.rs` (Core Types & Zero-Allocation Scratch Path)
- `Bf16(pub u16)`: `#[repr(transparent)]` IEEE-754 `bfloat16` wrapper with round-to-nearest-even `from_f32` and zero-cost `to_f32`.
- `f32_to_f16` / `f16_to_f32`: IEEE-754 `binary16` conversion used to store per-row scale norms (`scale_f16`, `rho_x_f16`, `rho_r_f16`) in 2 bytes.
- `EncodedLayerHalf`: Holds `n_rows`, `quant_bits`, `vip_rows: Vec<VipRowInt8>`, and `bulk_rows: Vec<BulkEncodedRow>`.
- `LocalSpectralCodec`: Default `vip_ratio = 0.05`, `sink_rows = 4`. Uses thread-local `CodecScratch` inside `encode_kv_layer_into` (and convenience wrapper `encode_kv_layer`) so steady-state layer encoding allocates zero intermediate scratch buffers for Christoffel scoring.
- `decode_layer_half_into` / `decode_layer_half`: Unpacks Stratum-0 `VipRowInt8` anchors and reconstructs Stratum-1 `BulkEncodedRow`s using `arrived_mask: u128`.

### 2.2 `src/rademacher.rs` (Deterministic Sign Masks)
- `xxh3_mix(seed, salt)`: SplitMix/avalanche 64-bit mixer.
- `rademacher_mask_128(request_id_lo, layer_id, stage) -> u128`: Produces deterministic 128-lane sign masks shared implicitly between Prefill TX and Decode RX via `(request_id_lo, layer_id)`.
- `build_avx2_sign_vecs(mask)`: Expands the 128-bit Rademacher mask once per layer into `[__m256; 16]` IEEE-754 sign-bit vectors (`0x8000_0000` or `0`), enabling sign flips via 16 `_mm256_xor_ps` instructions per row.

### 2.3 `src/fwht.rs` (In-Register 128-Lane Walsh-Hadamard & `BF16` Conversion)
- `has_avx2()`: Queries `is_x86_feature_detected!("avx2")` once and caches it in `std::sync::OnceLock<bool>`.
- `load_bf16_row_regs_avx2`: Loads 128 `Bf16` values into `ymm0..ymm15` using `_mm_loadu_si128`, `_mm256_cvtepu16_epi32`, and `_mm256_slli_epi32(..., 16)`.
- `store_bf16_row_regs_avx2`: Rounds `ymm0..ymm15` (`f32` to `Bf16` round-to-nearest-even) and stores 128 `Bf16` lanes via `_mm256_packus_epi32` + `_mm256_permute4x64_epi64(..., 0xD8)`.
- `fwht_128_regs_avx2`: Executes all 7 stages of the 128-lane Walsh-Hadamard butterfly (`896` add/sub ops) inside `ymm0..ymm15` with zero RAM spills:
  - Stages 1-3 (strides `1, 2, 4`): intra-register via `_mm256_permute_ps` (`0xB1`, `0x4E`), `_mm256_permute2f128_ps` (`0x01`), and `_mm256_blend_ps`.
  - Stages 4-7 (strides `8, 16, 32, 64`): cross-register `_mm256_add_ps` / `_mm256_sub_ps`.
  - Normalized by `INV_SQRT_128 = 0.088_388_35` ($1/\sqrt{128}$).
- `fwht_128_scalar`, `forward_srht_128`, `inverse_srht_128`: Portable scalar fallback and SRHT wrappers.

### 2.4 `src/christoffel.rs` (Layer I: $r=8$ Sketched Leverage Stratification & `INT8` Anchors)
- `sketch_row_avx2`: Computes an $r=8$ Rademacher projection $\tilde{k}_i \in \mathbb{R}^8$ (`_mm256_xor_ps` + `_mm256_add_ps`) alongside `||k_i||_2^2` and `||v_i||_2^2` in one AVX2 read pass over `keys` and `values` using shared `polar_qjl::hsum256_ps`.
- `stratify_tile_fast_into`: Accumulates the $8 \times 8$ covariance matrix on even rows, adds ridge term $\mu = \max(\frac{1}{N}\|K\|_F^2, 10^{-5})$, inverts via Cholesky (`invert_spd_8`), scores each row as $\tilde{\tau}_i(\mu)\|v_i\|_2$, forces rows `0..sink_rows` (`4`) to `f32::INFINITY`, and selects the top `vip_ratio` (`5%`) into `vip_rows` and remaining `95%` into `bulk_rows`.
- `VipRowInt8`: Symmetric per-row `INT8` anchor (`row_idx: u16`, `scale_f16: u16`, `q: [i8; 128]`, 132 bytes serialized). `pack_from_bf16_avx2` and `pack` quantize using `step_q = f16_to_f32(scale_f16)` so encode and decode scales match bit-for-bit, packing 128 lanes via `_mm256_packs_epi32` + `_mm256_packs_epi16`.

### 2.5 `src/polar_qjl.rs` (Layer II: Bitplane Polar Lloyd-Max + 1-Bit QJL)
- `BulkEncodedRow`: Zero-heap-allocation `Copy` bitplane representation:
  - `q_plane0`, `q_plane1`, `q_plane2` (`u128`): bitplanes storing $(b-1)$-bit Lloyd-Max codes for $b \in \{2, 3, 4\}$.
  - `qjl_signs` (`u128`): 1-bit sign sketch $\text{sign}(\Psi r_x)$ across all 128 lanes.
  - `rho_x_f16`, `rho_r_f16` (`u16`): Half-precision L2 norms of rotated row $\tilde{x}$ and quantization residual $r_x$.
- `encode_bf16_row_fused_avx2` / `encode_regs_fused_avx2`: Single-pass in-register encoder. Keeps `ymm0..ymm15` live across `BF16` load, `phi` SRHT, branchless Lloyd-Max bitplane extraction (`_mm256_cmp_ps` + `_mm256_movemask_ps`), residual accumulation, `psi` SRHT, and QJL sign extraction (`_mm256_movemask_ps`). In `encode_bulk_row_with_masks`, prerotated `f32` lanes load directly into `ymm0..ymm15` via `_mm256_loadu_ps`.
- `pack_bulk_stripe` / `unpack_bulk_stripe_into`: Uses `O(1)` bitplane shift-and-mask (`(row.q_plane0 >> coord_start) & mask`) per row to slice `9` or `10` coordinates per bulk stream via `BitPacker` / `BitUnpacker`.

### 2.6 `src/estimator.rs` (Layer II: Compile-Time LUT Deadline Reconstruction)
- `SIGN_MASK_LUT` / `ARRIVED_MASK_LUT`: Two compile-time 256-entry `[u32; 8]` lookup tables (`8 KiB` each, L1-resident) mapping any 8-bit chunk of `q_plane0..2`, `qjl_signs`, and `arrived_mask` directly into AVX2 sign-XOR and lane-zeroing masks.
- `reconstruct_bulk_row_to_bf16_avx2`: Scales arrived coordinates by $(1 - \hat{\delta})^{-1} = 128 / \text{popcnt}(\text{arrived\_mask})$, zeroes missing coordinates, applies $\Psi^\top$ (`fwht_128_regs_avx2` + `psi_vecs`) to the QJL residual, sums with dequantized Stage A centroids, applies $\Phi^\top$ (`fwht_128_regs_avx2` + `phi_vecs`), and stores `Bf16` via `store_bf16_row_regs_avx2`.

---

## 3. Crate 2: `crates/xasp-tcp` (16-Stream Wire Framing & Deadline RX)

Pure-`std` TCP networking (`std::net::{TcpListener, TcpStream}`), zero external crates.

### 3.1 `src/frame.rs` (`XaspTcpFrameHeader` & `CRC32C`)
- `XaspTcpFrameHeader`: `#[repr(C, align(64))]` struct verified at compile time (`size_of == 64`, `align_of == 64`), magic `0x5841_5350` (`"XASP"`), version `1`, bitflags (`FLAG_IS_VALUE_TENSOR`, `FLAG_IS_REDUNDANT_VIP_COPY`, `FLAG_IS_LAST_STRIPE_IN_LAYER`, `FLAG_PHYSICAL_TPU_TILE_128`), `stream_id`, `stratum_tier`, `staging_slot_idx`, `quant_bits_br`, `layer_id`, `coord_start`, `coord_count`, `request_id_lo/hi`, `hadamard_seed`, `deadline_ns`, `payload_len`, `token_offset`, `num_tokens`, `head_dim_and_kv`, and `header_crc32c`.
- `crc32c`: Dispatches to x86_64 SSE4.2 hardware instructions (`_mm_crc32_u64` / `_mm_crc32_u8`) when supported, with Castagnoli polynomial (`0x82F6_3B78`) software fallback.

### 3.2 `src/stripe_mux.rs` (`K=16` Stream Striping)
- Streams `0..1` (`TIER_STRATUM0_VIP`): Transmit `encode_stratum0_payload` redundantly (header + `VipRowInt8` anchors + `(rho_x_f16, rho_r_f16)` bulk norms), carrying `quant_bits_br = half.quant_bits` so Stratum-0 decode succeeds even when all `14` bulk streams miss `tau_l`.
- Streams `2..15` (`TIER_STRATUM1_BULK`): `stream_coord_range` assigns `9` coordinates per stream on `2..13` (`12 * 9 = 108`) and `10` coordinates per stream on `14..15` (`2 * 10 = 20`), covering all `128` Hadamard lanes.

### 3.3 `src/pool.rs` (OS Socket Tuning)
- `configure_xasp_tcp_stream`: Sets `TCP_NODELAY = 1` on all platforms; on `#[cfg(target_os = "linux")]`, issues `setsockopt` for `TCP_QUICKACK = 1`, `SO_RCVLOWAT = 64`, `SO_SNDBUF = 16 MiB`, `SO_RCVBUF = 16 MiB`, and `SO_PRIORITY = 6` on Stratum-0 VIP streams `0..1`.
- `rearm_quickack`: Re-arms `TCP_QUICKACK` after `recv()` on Linux.

### 3.4 `src/deadline_rx.rs` (Non-Blocking Frame Assembly & Late Drain)
- `RetiredLayerFilter`: 256-slot ring buffer storing `(request_id_lo, layer_id)` of completed layers.
- `StreamReaderState`: Holds non-blocking incremental header/body buffers plus a 1-slot `stashed` lookahead buffer so out-of-turn future-layer frames read during layer `l` are preserved cleanly for layer `l+1`. `poll_frame` discards frames matching `RetiredLayerFilter` and increments `DrainStats` (`late_drained_frames`, `late_drained_bytes`) without closing or resetting the socket.
- `DeadlineAssembler::collect_layer_until_deadline`: Inspects `stashed` then polls sockets up to `deadline`, marks `(request_id_lo, layer_id)` retired, unpacks arrived coordinate stripes into `arrived_mask: u128`, reads `vip_hdr.quant_bits_br`, and invokes `decode_layer_half`.

---

## 4. Crate 3: `crates/xasp-tpu-ffi` (Zero-Copy Host DRAM Bridge & `tpu-sync` C++17 Header)

Compiles to `rlib` + `cdylib` (`xasp_tpu.dll` / `libxasp_tpu.so`).

- `src/tile_arena_view.rs`: `PhysicalTileArenaView` / `PhysicalTileArenaViewMut` and `check_folded_device_layout` validating non-null, `n_rows > 0`, `minor_tile_dim == 128`, and alignment.
- `src/lib.rs`: C-ABI symbols:
  - `xasp_abi_version() -> u32`
  - `xasp_check_folded_device_layout(ptr, n_rows, minor_tile_dim) -> i32`
  - `xasp_encode_kv_to_wire_blob(...) -> i32`: Writes all 32 frames (16 K + 16 V) into caller staging memory and populates `XaspStripeIovec[32]` (`offset`, `len`, `stream_id`, `stratum_tier`, `is_value`, `coord_start`, `coord_count`).
  - `xasp_decode_kv_from_wire_blob(...) -> i32`: Decodes any arrived subset of frames (reading `quant_bits_br` from Stratum-0 headers) directly into `k_out_ptr` and `v_out_ptr`.
  - `xasp_roundtrip_physical_tiles_cabi(...) -> i32`: Single-call in-memory roundtrip helper returning `XaspRoundTripStats`.
- `include/xasp_tpu_sync_adapter.hpp`: Single-header C++17 wrapper (`xasp::RaidenStagingAdapter`) exposing `CompressHostTile` and `ReconstructHostTileAtDeadline` over raw Host DRAM `uint16_t*` pointers for `google/tpu-sync` and `google/peregrine` `Post()`.
- `python/xasp_tpu/`: `XaspConnector` (`connector_v1.py`, `ctypes` binding matching `KVConnectorBase_V1` hooks `start_load_kv`, `save_kv_layer`, `wait_for_layer_load`) and `StagingLayerHook` (`tpu_sync_hook.py`).

---

## 5. Crate 4: `crates/xasp-bench` (Benchmark Harness & PNG Renderer)

- `src/tile_gen.rs`: Generates deterministic `[512, 128]` `Bf16` physical tiles (`generate_outlier_tpu_tile`) with 4 initial attention sink rows (`gain = 14.0`), Pareto heavy-tail rows (`r % 28 == 0`, `gain = 5.5`), bulk rows (`gain = 0.28`), and 5 persistent outlier lanes (`7, 31, 63, 95, 119` at `30x` spike amplitude).
- `src/staging_emu.rs`:
  - `StagingLayerPool`: Real bounded slot semaphore (`Mutex` + `Condvar`, capacity `8`).
  - `run_staging_sweep`: Runs live `1:1` real-time 16-socket loopback sessions (`127.0.0.1`) across stall probabilities `0.0%, 0.2%, 0.5%, 1.0%, 2.0%`. Both baseline `RawBufferTransport` (`128 KB/layer`) and `XASP` (`actual_wire_bytes` summed over transmitted unique stripe frames) compute layer transmission pacing directly via `wire_pace_for_bytes` at `DCN_CONGESTED_LINK_GBPS = 0.0728`. Baseline holds slots until all 16 streams deliver 100% of layer bytes (`session_abort_ms = 520 ms` cutoff); `XASP` releases slots deterministically at `tau_l = 1.2 ms` using `VecDeque`-ordered per-stream delay queues (`25 ms` stall injection) and `drain_late_frames`.
- `src/loopback_netem.rs`:
  - `measure_cpu_throughput_gbps`: Measures multi-threaded `encode_kv_layer_into` and `decode_layer_half_into` throughput (`GB/s`) across `available_parallelism().min(16)` cores with zero per-call heap allocations on output buffers.
  - `run_live_loopback_16stream`: Transfers 16 real layers with injected 8 ms stream-7 stalls and reports wire compression ratio (`6.26x`) and drained late frames/bytes (`15` frames / `17,370 B` on `TPU v5e-1` Host VM).
  - `run_accuracy_matrix`: Evaluates attention output relative RMSE, KL divergence, and `delta_ppl_proxy` across 16 queries (`P_S1 ~ 0.14`) comparing `XASP` (`b=4,3,2`), `QJL-Only (No VIP)`, `Naive Byte Stripe`, and `Unstratified Scalar INT`.
- `src/chart_render.rs`: Pure-`std` `Canvas` + lossless zlib-stored PNG encoder writing the 3 charts to `<out_dir>/charts/`.
