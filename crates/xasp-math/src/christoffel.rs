// Copyright (c) 2026 fy2ne (https://xasp.fy2ne.me). All rights reserved.
// Dual-Licensed:
//   1. Noncommercial Evaluation & Academic Research: PolyForm-Noncommercial-1.0.0
//   2. Commercial / Production Deployment: Requires a paid Commercial License from fy2ne.
//      Source-available on https://github.com/fy2ne/xasp (not distributed via crates.io).

use crate::fwht::forward_srht_128;
use crate::rademacher::rademacher_mask_128;
use crate::{Bf16, f16_to_f32, f32_to_f16};

const SKETCH_DIM: usize = 16;

#[derive(Clone, Debug)]
pub struct StratumSplit {
    pub is_vip: Vec<bool>,
    pub vip_rows: Vec<u16>,
    pub bulk_rows: Vec<u16>,
    pub scores: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct VipRowInt8 {
    pub row_idx: u16,
    pub scale_f16: u16,
    pub q: [i8; 128],
}

impl VipRowInt8 {
    pub fn pack(row_idx: u16, row: &[f32; 128]) -> Self {
        let mut amax = 0.0f32;
        for &lane in row {
            let abs = lane.abs();
            if abs > amax {
                amax = abs;
            }
        }
        let step = if amax > 1e-8 { amax / 127.0 } else { 1e-8 };
        let inv = 1.0 / step;
        let mut q = [0i8; 128];
        for lane in 0..128 {
            q[lane] = (row[lane] * inv).round().clamp(-127.0, 127.0) as i8;
        }
        Self {
            row_idx,
            scale_f16: f32_to_f16(step),
            q,
        }
    }

    pub fn unpack_into(&self, dst: &mut [f32; 128]) {
        let step = f16_to_f32(self.scale_f16);
        for lane in 0..128 {
            dst[lane] = (self.q[lane] as f32) * step;
        }
    }

    pub fn append_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.row_idx.to_le_bytes());
        out.extend_from_slice(&self.scale_f16.to_le_bytes());
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(self.q.as_ptr() as *const u8, 128)
        };
        out.extend_from_slice(bytes);
    }

    pub fn from_bytes(src: &[u8]) -> Option<Self> {
        if src.len() < 132 {
            return None;
        }
        let row_idx = u16::from_le_bytes([src[0], src[1]]);
        let scale_f16 = u16::from_le_bytes([src[2], src[3]]);
        let mut q = [0i8; 128];
        for i in 0..128 {
            q[i] = src[4 + i] as i8;
        }
        Some(Self {
            row_idx,
            scale_f16,
            q,
        })
    }
}

fn invert_spd_32(mat: &mut [[f32; SKETCH_DIM]; SKETCH_DIM]) -> [[f32; SKETCH_DIM]; SKETCH_DIM] {
    let mut l = [[0.0f32; SKETCH_DIM]; SKETCH_DIM];
    for i in 0..SKETCH_DIM {
        for j in 0..=i {
            let mut sum = mat[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                l[i][j] = if sum > 1e-12 { sum.sqrt() } else { 1e-6 };
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    let mut linv = [[0.0f32; SKETCH_DIM]; SKETCH_DIM];
    for i in 0..SKETCH_DIM {
        linv[i][i] = 1.0 / l[i][i];
        for j in (0..i).rev() {
            let mut sum = 0.0f32;
            for k in (j + 1)..=i {
                sum -= l[i][k] * linv[k][j];
            }
            linv[i][j] = sum / l[i][i];
        }
    }

    let mut inv = [[0.0f32; SKETCH_DIM]; SKETCH_DIM];
    for i in 0..SKETCH_DIM {
        for j in 0..=i {
            let mut sum = 0.0f32;
            for k in i..SKETCH_DIM {
                sum += linv[k][i] * linv[k][j];
            }
            inv[i][j] = sum;
            inv[j][i] = sum;
        }
    }
    inv
}

pub fn stratify_tile_rows(
    keys: &[Bf16],
    values: &[Bf16],
    n_rows: usize,
    vip_ratio: f32,
    sink_rows: usize,
    request_id_lo: u64,
    layer_id: u16,
) -> StratumSplit {
    if n_rows == 0 {
        return StratumSplit {
            is_vip: Vec::new(),
            vip_rows: Vec::new(),
            bulk_rows: Vec::new(),
            scores: Vec::new(),
        };
    }

    let sketch_mask = rademacher_mask_128(request_id_lo, layer_id, 0xC415_70FE);
    let mut sketched = vec![[0.0f32; SKETCH_DIM]; n_rows];
    let mut gram = [[0.0f32; SKETCH_DIM]; SKETCH_DIM];
    let mut frob_sq = 0.0f32;
    let mut v_norms = vec![0.0f32; n_rows];
    let mut row_f = [0.0f32; 128];

    for r in 0..n_rows {
        let off = r * 128;
        let mut vn = 0.0f32;
        for c in 0..128 {
            let kf = keys[off + c].to_f32();
            row_f[c] = kf;
            frob_sq += kf * kf;
            let vf = values[off + c].to_f32();
            vn += vf * vf;
        }
        v_norms[r] = vn.sqrt();

        forward_srht_128(&mut row_f, sketch_mask);
        for s in 0..SKETCH_DIM {
            sketched[r][s] = row_f[s] * 2.828_427;
        }
        if (r & 1) == 0 {
            for i in 0..SKETCH_DIM {
                let si = sketched[r][i] * 1.414_213_5;
                for j in 0..=i {
                    gram[i][j] += si * (sketched[r][j] * 1.414_213_5);
                }
            }
        }
    }

    let mu = (frob_sq / (n_rows as f32)).max(1e-5);
    for i in 0..SKETCH_DIM {
        for j in 0..i {
            gram[j][i] = gram[i][j];
        }
        gram[i][i] += mu;
    }

    let inv_gram = invert_spd_32(&mut gram);
    let mut scores = vec![0.0f32; n_rows];
    let mut ranked: Vec<(usize, f32)> = Vec::with_capacity(n_rows);

    for r in 0..n_rows {
        if r < sink_rows {
            scores[r] = f32::INFINITY;
            ranked.push((r, f32::INFINITY));
            continue;
        }
        let sk = &sketched[r];
        let mut tau = 0.0f32;
        for i in 0..SKETCH_DIM {
            let mut acc = 0.0f32;
            for j in 0..SKETCH_DIM {
                acc += inv_gram[i][j] * sk[j];
            }
            tau += sk[i] * acc;
        }
        let sc = tau.max(0.0) * v_norms[r];
        scores[r] = sc;
        ranked.push((r, sc));
    }

    let target_vip = ((n_rows as f32 * vip_ratio).ceil() as usize)
        .max(sink_rows.min(n_rows))
        .min(n_rows);

    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut is_vip = vec![false; n_rows];
    for &(idx, _) in ranked.iter().take(target_vip) {
        is_vip[idx] = true;
    }

    let mut vip_rows = Vec::with_capacity(target_vip);
    let mut bulk_rows = Vec::with_capacity(n_rows.saturating_sub(target_vip));
    for (idx, &flag) in is_vip.iter().enumerate() {
        if flag {
            vip_rows.push(idx as u16);
        } else {
            bulk_rows.push(idx as u16);
        }
    }

    StratumSplit {
        is_vip,
        vip_rows,
        bulk_rows,
        scores,
    }
}
