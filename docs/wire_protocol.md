# XASP 64-Byte TCP Wire Protocol Specification

Every spectral stripe written onto any of the $K=16$ persistent TCP sockets is prefixed by a 64-byte cache-line-aligned little-endian header (`#[repr(C, align(64))]`). Combined with `SO_RCVLOWAT = 64` on Linux TPU Host VMs, the receiver reads a complete frame header without waking on partial header fragments.

---

## 1. Binary Layout (`XaspTcpFrameHeader`, 64 Bytes)

```text
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|         magic (0x58415350 = "XASP")           |version| flags |  0x00..0x05
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  stream_id    | stratum_tier  |staging_slot_id| quant_bits_br |  0x06..0x09
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|           layer_id            |          coord_start          |  0x0A..0x0D
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          coord_count          |     ...request_id_lo (u64)    |  0x0E..0x17
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

| Offset | Field | Type | Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| `0x00..0x04` | `magic` | `u32` | 4 B | Protocol magic: `0x5841_5350` (`"XASP"`). |
| `0x04..0x05` | `version` | `u8` | 1 B | Protocol version (`0x01`). |
| `0x05..0x06` | `flags` | `u8` | 1 B | `bit 0`: `IS_VALUE_TENSOR`, `bit 1`: `IS_REDUNDANT_VIP_COPY`, `bit 2`: `IS_LAST_STRIPE_IN_LAYER`, `bit 3`: `PHYSICAL_TPU_TILE_128`. |
| `0x06..0x07` | `stream_id` | `u8` | 1 B | TCP socket index $s \in \{0, \dots, 15\}$. |
| `0x07..0x08` | `stratum_tier` | `u8` | 1 B | `0x00` = Stratum-0 VIP (`INT8`), `0x01` = Stratum-1 Bulk Hadamard + Polar/QJL Stripe. |
| `0x08..0x09` | `staging_slot_idx` | `u8` | 1 B | Target `tpu-sync` `StagingLayerManager` host slot (`0..TPU_RAIDEN_NUM_STAGING_LAYERS - 1`). |
| `0x09..0x0A` | `quant_bits_br` | `u8` | 1 B | Bit-width $b \in \{2, 3, 4, 8\}$. |
| `0x0A..0x0C` | `layer_id` | `u16` | 2 B | Transformer layer index ($0 \dots 127$). |
| `0x0C..0x0E` | `coord_start` | `u16` | 2 B | Starting 128-lane Hadamard coordinate offset ($0 \dots 127$). |
| `0x0E..0x10` | `coord_count` | `u16` | 2 B | Number of coordinates in this stripe (or VIP row count for Stratum-0). |
| `0x10..0x18` | `request_id_lo` | `u64` | 8 B | Lower 64 bits of 128-bit request UUID. |
| `0x18..0x20` | `request_id_hi` | `u64` | 8 B | Upper 64 bits of 128-bit request UUID. |
| `0x20..0x28` | `hadamard_seed` | `u64` | 8 B | Seed generating Rademacher diagonal $D_{\text{seed}}$. |
| `0x28..0x30` | `deadline_ns` | `u64` | 8 B | Layer deadline $\tau_l$ (ns) for `ExecuteLayerH2d(l)`. |
| `0x30..0x34` | `payload_len` | `u32` | 4 B | Byte length of the stripe payload following this header. |
| `0x34..0x38` | `token_offset` | `u32` | 4 B | Starting token offset in the paged KV block. |
| `0x38..0x3A` | `num_tokens` | `u16` | 2 B | Number of tokens in this frame. |
| `0x3A..0x3C` | `head_dim_and_kv` | `u16` | 2 B | Low `u8` = `minor_tile_dim` (`128`), High `u8` = `num_kv_heads`. |
| `0x3C..0x40` | `header_crc32c` | `u32` | 4 B | Hardware `CRC32C` over bytes `0x00..0x3C`. |

---

## 2. $K=16$ Stream Mapping

| Streams | Stratum | Contents | Socket Options |
| :--- | :--- | :--- | :--- |
| `0..1` | Stratum-0 (`VIP`) | Top 5% Christoffel anchors + 4 initial Attention Sinks (`INT8`, duplicated on streams `0` and `1`) | `TCP_NODELAY=1`, Linux `TCP_QUICKACK=1`, `SO_PRIORITY=6`, `SO_RCVLOWAT=64` |
| `2..13` | Stratum-1 (`Bulk`) | 9 Hadamard coordinates per row per stream (`12 * 9 = 108` coords) | `TCP_NODELAY=1`, Linux `TCP_QUICKACK=1`, `SO_RCVLOWAT=64` |
| `14..15` | Stratum-1 (`Bulk`) | 10 Hadamard coordinates per row per stream (`2 * 10 = 20` coords) | `TCP_NODELAY=1`, Linux `TCP_QUICKACK=1`, `SO_RCVLOWAT=64` |

---

## 3. Retired-Layer Late Frame Drainer

When socket `s` stalls past $\tau_l$, the Decode side immediately runs unbiased $(1-\hat{\delta})^{-1}$ reconstruction on the arrived streams, executes `ExecuteLayerH2d(l)`, calls `StagingLayerManager::Release(l)`, and inserts `(request_id_lo, layer_id)` into a 256-slot `RetiredLayerFilter` ring buffer. When the delayed socket completes TCP retransmission later:
1. Reader consumes the 64-byte `XaspTcpFrameHeader` and validates `magic` and `header_crc32c`.
2. Reader matches `(request_id_lo, layer_id)` in `RetiredLayerFilter`.
3. Reader discards `header.payload_len` bytes into a reusable scratch buffer, re-arms `TCP_QUICKACK` on Linux, and advances immediately to the next layer without tearing down the TCP connection.

---

## 4. Recommended Linux TPU Host Kernel Tuning (`/etc/sysctl.d/99-xasp-tpu-tcp.conf`)

```ini
net.core.default_qdisc = fq
net.ipv4.tcp_congestion_control = bbr
net.core.rmem_max = 67108864
net.core.wmem_max = 67108864
net.ipv4.tcp_rmem = 4096 1048576 67108864
net.ipv4.tcp_wmem = 4096 1048576 67108864
net.ipv4.tcp_notsent_lowat = 131072
net.ipv4.tcp_autocorking = 0
net.ipv4.tcp_slow_start_after_idle = 0
net.ipv4.tcp_sack = 1
net.ipv4.tcp_recovery = 1
net.ipv4.tcp_early_retrans = 3
net.core.busy_poll = 50
net.core.busy_read = 50
```
