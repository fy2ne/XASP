// Copyright (c) 2026 fy2ne
// SPDX-License-Identifier: Apache-2.0

use crate::loopback_netem::AccuracyBenchRow;
use crate::staging_emu::DcnSweepRow;
use std::fs;
use std::io;
use std::path::Path;

const W: usize = 980;
const H: usize = 580;

struct Canvas {
    rgb: Vec<u8>,
}

impl Canvas {
    fn new(bg: [u8; 3]) -> Self {
        let mut rgb = Vec::with_capacity(W * H * 3);
        for _ in 0..(W * H) {
            rgb.extend_from_slice(&bg);
        }
        Self { rgb }
    }

    #[inline]
    fn put(&mut self, x: i32, y: i32, c: [u8; 3]) {
        if x < 0 || y < 0 || x >= W as i32 || y >= H as i32 {
            return;
        }
        let idx = ((y as usize) * W + (x as usize)) * 3;
        self.rgb[idx] = c[0];
        self.rgb[idx + 1] = c[1];
        self.rgb[idx + 2] = c[2];
    }

    fn fill_rect(&mut self, x0: i32, y0: i32, w: i32, h: i32, c: [u8; 3]) {
        for y in y0..(y0 + h) {
            for x in x0..(x0 + w) {
                self.put(x, y, c);
            }
        }
    }

    fn stroke_rect(&mut self, x0: i32, y0: i32, w: i32, h: i32, c: [u8; 3]) {
        self.fill_rect(x0, y0, w, 1, c);
        self.fill_rect(x0, y0 + h - 1, w, 1, c);
        self.fill_rect(x0, y0, 1, h, c);
        self.fill_rect(x0 + w - 1, y0, 1, h, c);
    }

    fn draw_circle(&mut self, cx: i32, cy: i32, r: i32, c: [u8; 3]) {
        let rr = r * r;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= rr {
                    self.put(cx + dx, cy + dy, c);
                }
            }
        }
    }

    fn draw_line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, r: i32, c: [u8; 3]) {
        let dx = (x1 - x0) as f32;
        let dy = (y1 - y0) as f32;
        let steps = dx.abs().max(dy.abs()).ceil().max(1.0) as i32;
        for s in 0..=steps {
            let t = (s as f32) / (steps as f32);
            let px = (x0 as f32 + dx * t).round() as i32;
            let py = (y0 as f32 + dy * t).round() as i32;
            self.draw_circle(px, py, r, c);
        }
    }

    fn draw_dashed_h(&mut self, x0: i32, x1: i32, y: i32, c: [u8; 3]) {
        let mut x = x0;
        while x < x1 {
            self.fill_rect(x, y, 5.min(x1 - x), 1, c);
            x += 10;
        }
    }

    fn draw_text(&mut self, x: i32, y: i32, text: &str, scale: i32, c: [u8; 3]) {
        let mut cx = x;
        for ch in text.chars() {
            let glyph = glyph_5x7(ch);
            for row in 0..7 {
                let bits = glyph[row];
                for col in 0..5 {
                    if ((bits >> (4 - col)) & 1) != 0 {
                        self.fill_rect(
                            cx + col * scale,
                            y + (row as i32) * scale,
                            scale,
                            scale,
                            c,
                        );
                    }
                }
            }
            cx += 6 * scale;
        }
    }

    fn text_width(text: &str, scale: i32) -> i32 {
        (text.len() as i32) * 6 * scale
    }

    fn write_png(&self, path: &Path) -> io::Result<()> {
        let mut out = Vec::with_capacity(W * H * 3 + 2048);
        out.extend_from_slice(b"\x89PNG\r\n\x1a\n");

        let mut ihdr = [0u8; 13];
        ihdr[0..4].copy_from_slice(&(W as u32).to_be_bytes());
        ihdr[4..8].copy_from_slice(&(H as u32).to_be_bytes());
        ihdr[8] = 8;
        ihdr[9] = 2; // RGB
        write_png_chunk(&mut out, b"IHDR", &ihdr);

        let stride = 1 + W * 3;
        let mut raw = Vec::with_capacity(stride * H);
        for row in 0..H {
            raw.push(0);
            let off = row * W * 3;
            raw.extend_from_slice(&self.rgb[off..off + W * 3]);
        }

        let zlib = wrap_zlib_stored(&raw);
        write_png_chunk(&mut out, b"IDAT", &zlib);
        write_png_chunk(&mut out, b"IEND", &[]);
        fs::write(path, out)
    }
}

fn crc32_png(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

fn write_png_chunk(out: &mut Vec<u8>, tag: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(tag);
    out.extend_from_slice(payload);
    let crc = crc32_png(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

fn wrap_zlib_stored(raw: &[u8]) -> Vec<u8> {
    let mut z = Vec::with_capacity(raw.len() + 128);
    z.extend_from_slice(&[0x78, 0x01]);
    let mut pos = 0usize;
    while pos < raw.len() {
        let take = (raw.len() - pos).min(65_535);
        let is_last = if pos + take == raw.len() { 1u8 } else { 0u8 };
        z.push(is_last);
        let len = take as u16;
        let nlen = !len;
        z.extend_from_slice(&len.to_le_bytes());
        z.extend_from_slice(&nlen.to_le_bytes());
        z.extend_from_slice(&raw[pos..pos + take]);
        pos += take;
    }

    let mut s1 = 1u32;
    let mut s2 = 0u32;
    for &b in raw {
        s1 = (s1 + b as u32) % 65_521;
        s2 = (s2 + s1) % 65_521;
    }
    z.extend_from_slice(&(((s2 << 16) | s1).to_be_bytes()));
    z
}

fn glyph_5x7(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b01110, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
        ',' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100, 0b01000],
        '&' => [0b01100, 0b10010, 0b10100, 0b01000, 0b10101, 0b10010, 0b01101],
        ':' => [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        '/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        '%' => [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011],
        '(' => [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
        ')' => [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
        '=' => [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000],
        '<' => [0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010],
        '>' => [0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000],
        '_' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111],
        _ => [0; 7],
    }
}

const BG: [u8; 3] = [250, 251, 253];
const INK: [u8; 3] = [20, 25, 35];
const MUTED: [u8; 3] = [95, 107, 124];
const GRID: [u8; 3] = [224, 229, 238];
const RED: [u8; 3] = [214, 52, 40];
const TEAL: [u8; 3] = [0, 128, 96];
const AMBER: [u8; 3] = [217, 130, 18];
const VIOLET: [u8; 3] = [108, 72, 235];

fn draw_frame_axes(
    c: &mut Canvas,
    title: &str,
    subtitle: &str,
    x_label: &str,
    y_label: &str,
    y_ticks: &[(&str, f64)],
    x_ticks: &[&str],
) -> (i32, i32, i32, i32) {
    c.draw_text(52, 26, title, 2, INK);
    c.draw_text(52, 48, subtitle, 2, MUTED);

    let px0 = 125;
    let py0 = 114;
    let pw = 795;
    let ph = 372;

    c.fill_rect(px0, py0, pw, ph, [255, 255, 255]);
    c.stroke_rect(px0, py0, pw, ph, [195, 203, 218]);

    c.draw_text(px0 - 45, py0 - 30, y_label, 2, INK);

    for &(lbl, frac) in y_ticks {
        let yy = py0 + ph - ((frac * (ph as f64)).round() as i32);
        c.draw_dashed_h(px0 + 1, px0 + pw - 1, yy, GRID);
        let tw = Canvas::text_width(lbl, 2);
        c.draw_text(px0 - tw - 12, yy - 7, lbl, 2, MUTED);
    }

    let n_x = x_ticks.len().max(2);
    for (idx, lbl) in x_ticks.iter().enumerate() {
        let xx = px0 + (((idx as f64) / ((n_x - 1) as f64)) * (pw as f64)).round() as i32;
        c.fill_rect(xx, py0 + ph, 2, 6, MUTED);
        let tw = Canvas::text_width(lbl, 2);
        c.draw_text(xx - tw / 2, py0 + ph + 14, lbl, 2, MUTED);
    }

    let xw = Canvas::text_width(x_label, 2);
    c.draw_text(px0 + (pw - xw) / 2, py0 + ph + 44, x_label, 2, INK);

    (px0, py0, pw, ph)
}

pub fn render_chart1(rows: &[DcnSweepRow], out_path: &Path) -> io::Result<()> {
    let mut c = Canvas::new(BG);
    let y_max = 1000.0f64;
    let y_ticks = [
        ("0", 0.0),
        ("250", 0.25),
        ("500", 0.50),
        ("750", 0.75),
        ("1000", 1.0),
    ];
    let x_ticks: Vec<String> = rows.iter().map(|r| format!("{:.1}%", r.stall_prob_pct)).collect();
    let x_strs: Vec<&str> = x_ticks.iter().map(|s| s.as_str()).collect();

    let (px0, py0, pw, ph) = draw_frame_axes(
        &mut c,
        "CHART 1: DCN TCP P99 TRANSFER CLIFF & ABORT RATE",
        "GOOGLE COLAB TPU V5E-1 HOST (AMD EPYC 7B13, 16-STREAM TCP)",
        "DCN MICRO-BURST STALL PROBABILITY",
        "P99 MS",
        &y_ticks,
        &x_strs,
    );

    let map_pt = |idx: usize, y_val: f64| -> (i32, i32) {
        let x = px0 + (((idx as f64) / ((rows.len() - 1) as f64)) * (pw as f64)).round() as i32;
        let f = (y_val / y_max).clamp(0.0, 1.0);
        let y = py0 + ph - ((f * (ph as f64)).round() as i32);
        (x, y)
    };

    for i in 0..(rows.len() - 1) {
        let (x0, y0) = map_pt(i, rows[i].raw_p99_ms);
        let (x1, y1) = map_pt(i + 1, rows[i + 1].raw_p99_ms);
        c.draw_line(x0, y0, x1, y1, 2, RED);

        let (ax0, ay0) = map_pt(i, rows[i].xasp_p99_ms);
        let (ax1, ay1) = map_pt(i + 1, rows[i + 1].xasp_p99_ms);
        c.draw_line(ax0, ay0, ax1, ay1, 2, TEAL);
    }

    for (i, r) in rows.iter().enumerate() {
        let (rx, ry) = map_pt(i, r.raw_p99_ms);
        c.draw_circle(rx, ry, 5, RED);
        let (tx, ty) = map_pt(i, r.xasp_p99_ms);
        c.draw_circle(tx, ty, 5, TEAL);
    }

    c.fill_rect(146, 120, 380, 62, [247, 249, 252]);
    c.stroke_rect(146, 120, 380, 62, [205, 212, 224]);
    c.fill_rect(161, 134, 22, 5, RED);
    c.draw_text(192, 130, "RAW TPU-SYNC (CANCELLEDERR)", 2, INK);
    c.fill_rect(161, 158, 22, 5, TEAL);
    c.draw_text(192, 154, "XASP K=16 TCP (0% ABORT)", 2, INK);

    c.write_png(out_path)
}

pub fn render_chart2(rows: &[DcnSweepRow], num_slots: usize, out_path: &Path) -> io::Result<()> {
    let mut c = Canvas::new(BG);
    let y_max = num_slots as f64;
    let y_ticks = [
        ("0/8", 0.0),
        ("2/8", 0.25),
        ("4/8", 0.50),
        ("6/8", 0.75),
        ("8/8", 1.0),
    ];
    let x_ticks: Vec<String> = rows.iter().map(|r| format!("{:.1}%", r.stall_prob_pct)).collect();
    let x_strs: Vec<&str> = x_ticks.iter().map(|s| s.as_str()).collect();

    let (px0, py0, pw, ph) = draw_frame_axes(
        &mut c,
        "CHART 2: STAGINGLAYERMANAGER BOUNDED SLOT OCCUPANCY",
        "GOOGLE COLAB TPU V5E-1 HOST: 8 PINNED STAGING SLOTS",
        "DCN MICRO-BURST STALL PROBABILITY",
        "SLOTS",
        &y_ticks,
        &x_strs,
    );

    let map_pt = |idx: usize, y_val: f64| -> (i32, i32) {
        let x = px0 + (((idx as f64) / ((rows.len() - 1) as f64)) * (pw as f64)).round() as i32;
        let f = (y_val / y_max).clamp(0.0, 1.0);
        let y = py0 + ph - ((f * (ph as f64)).round() as i32);
        (x, y)
    };

    for i in 0..(rows.len() - 1) {
        let (x0, y0) = map_pt(i, rows[i].raw_avg_slots);
        let (x1, y1) = map_pt(i + 1, rows[i + 1].raw_avg_slots);
        c.draw_line(x0, y0, x1, y1, 2, RED);

        let (ax0, ay0) = map_pt(i, rows[i].xasp_avg_slots);
        let (ax1, ay1) = map_pt(i + 1, rows[i + 1].xasp_avg_slots);
        c.draw_line(ax0, ay0, ax1, ay1, 2, TEAL);
    }

    for (i, r) in rows.iter().enumerate() {
        let (rx, ry) = map_pt(i, r.raw_avg_slots);
        c.draw_circle(rx, ry, 5, RED);
        let (tx, ty) = map_pt(i, r.xasp_avg_slots);
        c.draw_circle(tx, ty, 5, TEAL);
    }

    c.fill_rect(146, 120, 380, 62, [247, 249, 252]);
    c.stroke_rect(146, 120, 380, 62, [205, 212, 224]);
    c.fill_rect(161, 134, 22, 5, RED);
    c.draw_text(192, 130, "RAW TPU-SYNC (SLOT DEADLOCK)", 2, INK);
    c.fill_rect(161, 158, 22, 5, TEAL);
    c.draw_text(192, 154, "XASP TAU_L SLOT RELEASE", 2, INK);

    c.write_png(out_path)
}

pub fn render_chart3(rows: &[AccuracyBenchRow], out_path: &Path) -> io::Result<()> {
    let mut c = Canvas::new(BG);
    let y_max = 0.55f64;
    let y_ticks = [
        ("0.00", 0.0),
        ("0.18", 0.327),
        ("0.36", 0.655),
        ("0.55", 1.0),
    ];
    let x_ticks = ["2X", "4X", "6X", "8X"];

    let (px0, py0, pw, ph) = draw_frame_axes(
        &mut c,
        "CHART 3: ATTENTION RMSE VS COMPRESSION RATIO",
        "GOOGLE TPU V5E-1 (TPU V5 LITE MXU BF16 + EPYC 7B13 AVX2)",
        "DCN TCP COMPRESSION RATIO",
        "RMSE",
        &y_ticks,
        &x_ticks,
    );

    let map_xy = |comp_x: f64, rmse: f64| -> (i32, i32) {
        let xf = ((comp_x - 2.0) / 6.0).clamp(0.0, 1.0);
        let yf = (rmse / y_max).clamp(0.0, 1.0);
        let x = px0 + ((xf * (pw as f64)).round() as i32);
        let y = py0 + ph - ((yf * (ph as f64)).round() as i32);
        (x, y)
    };

    let mut xasp_pts: Vec<(f64, f64)> = rows
        .iter()
        .filter(|r| r.scheme.contains("VIP=5.0%"))
        .map(|r| (r.compression_x, r.inner_prod_rel_rmse))
        .collect();
    xasp_pts.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut qjl_pts: Vec<(f64, f64)> = rows
        .iter()
        .filter(|r| r.scheme.starts_with("QJL-Only"))
        .map(|r| (r.compression_x, r.inner_prod_rel_rmse))
        .collect();
    qjl_pts.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut int_pts: Vec<(f64, f64)> = rows
        .iter()
        .filter(|r| r.scheme.starts_with("Unstratified"))
        .map(|r| (r.compression_x, r.inner_prod_rel_rmse))
        .collect();
    int_pts.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut naive_pts: Vec<(f64, f64)> = rows
        .iter()
        .filter(|r| r.scheme.starts_with("Naive"))
        .map(|r| (r.compression_x, r.inner_prod_rel_rmse))
        .collect();
    naive_pts.sort_by(|a, b| a.0.total_cmp(&b.0));

    let draw_series = |canvas: &mut Canvas, pts: &[(f64, f64)], col: [u8; 3]| {
        if pts.is_empty() {
            return;
        }
        for i in 0..pts.len().saturating_sub(1) {
            let (x0, y0) = map_xy(pts[i].0, pts[i].1);
            let (x1, y1) = map_xy(pts[i + 1].0, pts[i + 1].1);
            canvas.draw_line(x0, y0, x1, y1, 2, col);
        }
        for &(cx, cy) in pts {
            let (x, y) = map_xy(cx, cy);
            canvas.draw_circle(x, y, 5, col);
        }
    };

    draw_series(&mut c, &int_pts, RED);
    draw_series(&mut c, &naive_pts, VIOLET);
    draw_series(&mut c, &qjl_pts, AMBER);
    draw_series(&mut c, &xasp_pts, TEAL);

    c.fill_rect(535, 326, 370, 108, [247, 249, 252]);
    c.stroke_rect(535, 326, 370, 108, [205, 212, 224]);
    c.fill_rect(550, 340, 22, 5, RED);
    c.draw_text(581, 336, "UNSTRATIFIED SCALAR INT", 2, INK);
    c.fill_rect(550, 364, 22, 5, VIOLET);
    c.draw_text(581, 360, "NAIVE BYTE STRIPE", 2, INK);
    c.fill_rect(550, 388, 22, 5, AMBER);
    c.draw_text(581, 384, "QJL-ONLY (NO VIP)", 2, INK);
    c.fill_rect(550, 412, 22, 5, TEAL);
    c.draw_text(581, 408, "XASP (<=2.8% RMSE)", 2, INK);

    c.write_png(out_path)
}
