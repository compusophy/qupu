//! The tabletop rig (docs/photonic/TABLETOP.md) as a program: what it
//! computes (transforms of 16×16 images), what it is shown (animated input
//! images), how its weights land on the LCD pixels, and the bench genome a
//! hill climb can rearrange: relay, f-number, the lattices weights and
//! tiles are laid out on, and cell size. Every genome maps onto a
//! `freespace::Design`, so the climb and the demo score the same physics
//! the CLI does.

use crate::core::Rng;
use crate::freespace::{enob_terms, Design, Engine, Layouts, Terms};
use crate::geometry::Lattice;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Images are SIDE × SIDE; the engine is (SIDE²) × (SIDE²).
pub const SIDE: usize = 16;
pub const N: usize = SIDE * SIDE;
/// Pixel pitch of the 5.5″ 2K mono LCDs, µm.
pub const LCD_PITCH_UM: f64 = 47.25;
/// Relay f-numbers on the lens aperture rings.
pub const F_STOPS: [f64; 7] = [1.8, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Workload {
    Blur,
    LowPass,
    BandPass,
    Spectrum,
}

impl Workload {
    pub const ALL: [Workload; 4] = [
        Workload::Blur,
        Workload::LowPass,
        Workload::BandPass,
        Workload::Spectrum,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Workload::Blur => "blur",
            Workload::LowPass => "low-pass",
            Workload::BandPass => "edges",
            Workload::Spectrum => "spectrum",
        }
    }

    /// m × n weights, row-major, scaled so the largest |w| is 1. All are
    /// dense: a fan-in engine spends its precision on Σ w·x, so sparse rows
    /// (shifts, permutations) are the wrong job for it.
    pub fn weights(self) -> Vec<f64> {
        let mut w = vec![0.0; N * N];
        match self {
            Workload::Blur => {
                let s = 2.2f64;
                for i in 0..N {
                    for j in 0..N {
                        let (dy, dx) = (
                            (i / SIDE) as f64 - (j / SIDE) as f64,
                            (i % SIDE) as f64 - (j % SIDE) as f64,
                        );
                        w[i * N + j] = (-(dx * dx + dy * dy) / (2.0 * s * s)).exp();
                    }
                }
            }
            Workload::LowPass => projection(&mut w, 4, 1.0),
            Workload::BandPass => {
                projection(&mut w, 8, 1.0);
                projection(&mut w, 3, -1.0);
            }
            Workload::Spectrum => {
                for k in 0..N {
                    for j in 0..N {
                        w[k * N + j] = dct2(k / SIDE, k % SIDE, j);
                    }
                }
            }
        }
        let max = w.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        w.iter().map(|v| v / max).collect()
    }
}

/// Orthonormal 2-D DCT-II basis function (u, v) at pixel j.
fn dct2(u: usize, v: usize, j: usize) -> f64 {
    let c = |k: usize| {
        if k == 0 {
            (1.0 / SIDE as f64).sqrt()
        } else {
            (2.0 / SIDE as f64).sqrt()
        }
    };
    let (y, x) = ((j / SIDE) as f64, (j % SIDE) as f64);
    c(u) * c(v)
        * (PI * (y + 0.5) * u as f64 / SIDE as f64).cos()
        * (PI * (x + 0.5) * v as f64 / SIDE as f64).cos()
}

/// Add `sign` × the projector onto the k × k lowest DCT frequencies.
fn projection(w: &mut [f64], k: usize, sign: f64) {
    for u in 0..k {
        for v in 0..k {
            let b: Vec<f64> = (0..N).map(|j| dct2(u, v, j)).collect();
            for i in 0..N {
                for j in 0..N {
                    w[i * N + j] += sign * b[i] * b[j];
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Input {
    Text,
    Orbits,
    Wheel,
    Face,
}

impl Input {
    pub const ALL: [Input; 4] = [Input::Text, Input::Orbits, Input::Wheel, Input::Face];

    pub fn name(self) -> &'static str {
        match self {
            Input::Text => "text",
            Input::Orbits => "orbits",
            Input::Wheel => "wheel",
            Input::Face => "face",
        }
    }

    /// The image at time `t` seconds, values in [0, 1], 4×4 supersampled.
    pub fn image(self, t: f64) -> Vec<f64> {
        let mut img = vec![0.0; N];
        for (j, px) in img.iter_mut().enumerate() {
            let (py, pxx) = ((j / SIDE) as f64, (j % SIDE) as f64);
            let mut acc = 0.0;
            for sy in 0..4 {
                for sx in 0..4 {
                    let (x, y) = (pxx + (sx as f64 + 0.5) / 4.0, py + (sy as f64 + 0.5) / 4.0);
                    acc += self.sample(x, y, t);
                }
            }
            *px = (acc / 16.0).clamp(0.0, 1.0);
        }
        img
    }

    /// Point sample at (x, y) in pixel units (0..16), y down.
    fn sample(self, x: f64, y: f64, t: f64) -> f64 {
        let (cx, cy) = (x - 8.0, y - 8.0);
        let soft = |d: f64, w: f64| (0.5 - d / w).clamp(0.0, 1.0) * 2.0f64.min(1.0);
        match self {
            Input::Text => {
                const MSG: &str = "QUPU · GLASS ENGINE · ";
                let adv = 12.0;
                let len = MSG.chars().count() as f64 * adv;
                let u = (x + 7.0 * t).rem_euclid(len);
                let (k, gx) = ((u / adv) as usize, u % adv);
                let gy = y - 1.0;
                if !(0.0..10.0).contains(&gx) || !(0.0..14.0).contains(&gy) {
                    return 0.0;
                }
                let rows = glyph(MSG.chars().nth(k).unwrap_or(' '));
                let (col, row) = ((gx / 2.0) as usize, (gy / 2.0) as usize);
                if rows[row] >> (4 - col) & 1 == 1 {
                    1.0
                } else {
                    0.0
                }
            }
            Input::Orbits => {
                let mut v = 0.0;
                for (k, (r, w, s)) in [(4.6, 1.3, 1.5), (2.4, -2.1, 1.1), (6.2, 0.7, 1.0)]
                    .iter()
                    .enumerate()
                {
                    let a = w * t + k as f64 * 2.1;
                    let (bx, by) = (r * a.cos(), r * a.sin());
                    v += (-((cx - bx).powi(2) + (cy - by).powi(2)) / (2.0 * s * s)).exp();
                }
                let ring = (cx.hypot(cy) - 6.8).abs();
                v + 0.35 * (1.0 - ring / 0.6).clamp(0.0, 1.0)
            }
            Input::Wheel => {
                let r = cx.hypot(cy);
                let rim = (r - 6.2).abs();
                let mut v: f64 = if rim < 0.8 { 1.0 } else { 0.0 };
                if r < 1.6 {
                    v = 1.0;
                }
                let a0 = 0.9 * t;
                for k in 0..5 {
                    let a = a0 + k as f64 * 2.0 * PI / 5.0;
                    let (dx, dy) = (a.cos(), a.sin());
                    let along = cx * dx + cy * dy;
                    let across = (cx * dy - cy * dx).abs();
                    if (0.0..6.2).contains(&along) && across < 0.55 {
                        v = 1.0;
                    }
                }
                v * soft(0.0, 1.0)
            }
            Input::Face => {
                let tilt = 0.25 * (0.7 * t).sin();
                let (x2, y2) = (
                    cx * tilt.cos() + cy * tilt.sin(),
                    -cx * tilt.sin() + cy * tilt.cos(),
                );
                let r = x2.hypot(y2);
                let mut v: f64 = if (r - 6.6).abs() < 0.7 { 1.0 } else { 0.0 };
                let wink = if (t % 3.0) < 0.25 { 0.25 } else { 1.0 };
                for (ex, ey) in [(-2.5, -2.0), (2.5, -2.0)] {
                    let sq = if ex > 0.0 { wink } else { 1.0 };
                    if ((x2 - ex) / 1.1).hypot((y2 - ey) / (1.3 * sq)) < 1.0 {
                        v = 1.0;
                    }
                }
                let ang = y2.atan2(x2);
                if (r - 3.9).abs() < 0.65 && ang > 0.35 && ang < PI - 0.35 {
                    v = 1.0;
                }
                v
            }
        }
    }
}

/// 5 × 7 bitmaps, rows top to bottom, MSB on the left.
fn glyph(c: char) -> [u8; 7] {
    match c {
        'G' => [14, 17, 16, 23, 17, 17, 15],
        'L' => [16, 16, 16, 16, 16, 16, 31],
        'A' => [14, 17, 17, 31, 17, 17, 17],
        'S' => [15, 16, 16, 14, 1, 1, 30],
        'E' => [31, 16, 16, 30, 16, 16, 31],
        'N' => [17, 25, 21, 19, 17, 17, 17],
        'I' => [14, 4, 4, 4, 4, 4, 14],
        'Q' => [14, 17, 17, 17, 21, 18, 13],
        'U' => [17, 17, 17, 17, 17, 17, 14],
        'P' => [30, 17, 17, 30, 16, 16, 16],
        '·' => [0, 0, 0, 4, 0, 0, 0],
        _ => [0; 7],
    }
}

/// Effective bits of an engine on its own matrix's output range, over the
/// four animated inputs: log₂(span / (√12 · rms error)).
pub fn workload_bits(e: &Engine) -> f64 {
    let mut rng = Rng::new(3);
    let (mut se, mut cnt) = (0.0, 0.0);
    let mut prev = Input::Face.image(0.0);
    for k in 0..24 {
        let x = Input::ALL[k % 4].image(k as f64 * 0.7);
        let r = e.run(&x, &prev, &mut rng);
        se +=
            r.y.iter()
                .zip(&r.ideal)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>();
        cnt += r.y.len() as f64;
        prev = x;
    }
    (e.output_span() / (12f64.sqrt() * (se / cnt).sqrt().max(1e-12))).log2()
}

/// The bench genome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rig {
    /// Lattice the weight cells sit on inside each tile.
    pub lattice: Lattice,
    /// Lattice the tiles sit on.
    pub tiles: Lattice,
    /// Weight cell size in LCD pixels (1 or 2). Only square cells can be
    /// single pixels: other lattices need room to round onto the grid.
    pub cell: u8,
    /// Index into F_STOPS.
    pub stop: usize,
    /// Two identical lenses face to face, or one lens at 2f–2f.
    pub symmetric: bool,
}

impl Rig {
    /// As costed in TABLETOP.md (= `Design::tabletop()`).
    pub fn as_built() -> Rig {
        Rig {
            lattice: Lattice::Square,
            tiles: Lattice::Square,
            cell: 1,
            stop: 2,
            symmetric: true,
        }
    }

    /// Where the demo's evolution starts: one fast lens, an awkward lattice.
    pub fn naive() -> Rig {
        Rig {
            lattice: Lattice::Quasi,
            tiles: Lattice::Quasi,
            cell: 2,
            stop: 0,
            symmetric: false,
        }
    }

    pub fn fnum(&self) -> f64 {
        F_STOPS[self.stop]
    }

    pub fn design(&self) -> Design {
        Design {
            in_lattice: self.lattice,
            tile_lattice: self.tiles,
            pitch_um: self.cell as f64 * LCD_PITCH_UM,
            na: 0.5 / self.fnum(),
            symmetric: self.symmetric,
            ..Design::tabletop()
        }
    }

    fn legal(mut self) -> Rig {
        if self.lattice != Lattice::Square {
            self.cell = 2;
        }
        self
    }

    pub fn mutate(&self, rng: &mut Rng) -> Rig {
        let mut r = *self;
        let pick = |rng: &mut Rng| Lattice::ALL[(rng.next_u64() % 4) as usize];
        match rng.next_u64() % 5 {
            0 => r.lattice = pick(rng),
            1 => r.tiles = pick(rng),
            2 => r.cell = 3 - r.cell,
            3 => {
                let up = rng.uniform() < 0.5;
                r.stop = if up {
                    (r.stop + 1).min(F_STOPS.len() - 1)
                } else {
                    r.stop.saturating_sub(1)
                };
            }
            _ => r.symmetric = !r.symmetric,
        }
        r.legal()
    }
}

/// Predicted effective bits of a rig (sampled, fixed seed: deterministic).
pub fn rig_enob(r: &Rig, layouts: &mut Layouts) -> f64 {
    enob_terms(&r.design(), layouts, 8, 32, 11, Terms::ALL).0
}

/// A hill climb over the bench, one evaluation per `step`, so a frame loop
/// can interleave it with rendering.
pub struct RigClimb {
    rng: Rng,
    pub parent: (Rig, f64),
    pub best: (Rig, f64),
    pub evals: usize,
    stuck: usize,
    memo: HashMap<Rig, f64>,
}

impl RigClimb {
    pub fn new(start: Rig, layouts: &mut Layouts, seed: u64) -> RigClimb {
        let start = start.legal();
        let e = rig_enob(&start, layouts);
        let mut memo = HashMap::new();
        memo.insert(start, e);
        RigClimb {
            rng: Rng::new(seed),
            parent: (start, e),
            best: (start, e),
            evals: 1,
            stuck: 0,
            memo,
        }
    }

    /// Try one mutant; true when it beats the best so far.
    pub fn step(&mut self, layouts: &mut Layouts) -> bool {
        let mut kid = self.parent.0.mutate(&mut self.rng);
        if self.stuck > 24 {
            // Restart from a scrambled copy of the best.
            kid = self.best.0;
            for _ in 0..3 {
                kid = kid.mutate(&mut self.rng);
            }
            self.stuck = 0;
        }
        let e = match self.memo.get(&kid) {
            Some(&e) => e,
            None => {
                self.evals += 1;
                let e = rig_enob(&kid, layouts);
                self.memo.insert(kid, e);
                e
            }
        };
        // Neutral moves are accepted: plateaus get explored, not stuck on.
        if e >= self.parent.1 {
            self.stuck = if e > self.parent.1 + 1e-9 {
                0
            } else {
                self.stuck + 1
            };
            self.parent = (kid, e);
        } else {
            self.stuck += 1;
        }
        if e > self.best.1 + 1e-9 {
            self.best = (kid, e);
            true
        } else {
            false
        }
    }
}

/// Which weight each LCD pixel shows: the active area is `size` × `size`
/// LCD pixels; `rgba` holds, per pixel, (cell j, tile t mod 256,
/// 1 + t / 256, 255) where a weight sits and zeros elsewhere. Sites are
/// rounded onto the pixel grid, as a real panel must.
pub struct LcdMap {
    pub size: usize,
    pub rgba: Vec<u8>,
    /// Active area side, mm.
    pub side_mm: f64,
}

pub fn lcd_map(engine: &Engine, cell: usize) -> LcdMap {
    let phys = &engine.phys;
    let mut sites: Vec<(i64, i64, usize, usize)> = Vec::with_capacity(engine.tiles() * N);
    for (t, tc) in phys.tiles.iter().enumerate() {
        for (j, p) in engine.layout.pts.iter().enumerate() {
            let (x, y) = (tc.0 + p.0 * phys.pitch, tc.1 + p.1 * phys.pitch);
            let off = (cell as f64 - 1.0) / 2.0;
            sites.push((
                (x / LCD_PITCH_UM - off).round() as i64,
                (y / LCD_PITCH_UM - off).round() as i64,
                j,
                t,
            ));
        }
    }
    let lo = sites.iter().fold(i64::MAX, |a, s| a.min(s.0).min(s.1));
    let hi = sites.iter().fold(i64::MIN, |a, s| a.max(s.0).max(s.1));
    // Centred on the axis: symmetric extent plus a margin.
    let half = lo.abs().max(hi.abs() + cell as i64) + 3;
    let size = ((2 * half) as usize).div_ceil(4) * 4;
    let o = (size / 2) as i64;
    let mut rgba = vec![0u8; size * size * 4];
    for &(x, y, j, t) in &sites {
        for dy in 0..cell as i64 {
            for dx in 0..cell as i64 {
                let (px, py) = (x + dx + o, y + dy + o);
                if px >= 0 && py >= 0 && (px as usize) < size && (py as usize) < size {
                    let k = (py as usize * size + px as usize) * 4;
                    rgba[k..k + 4].copy_from_slice(&[
                        j as u8,
                        (t & 255) as u8,
                        1 + (t >> 8) as u8,
                        255,
                    ]);
                }
            }
        }
    }
    LcdMap {
        size,
        rgba,
        side_mm: size as f64 * LCD_PITCH_UM / 1000.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workloads_are_dense_and_scaled() {
        for w in Workload::ALL {
            let m = w.weights();
            let max = m.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            assert!((max - 1.0).abs() < 1e-12);
            let dense = m.iter().filter(|v| v.abs() > 0.05).count() as f64 / m.len() as f64;
            assert!(dense > 0.1, "{} density {dense}", w.name());
        }
    }

    #[test]
    fn inputs_stay_in_range() {
        for i in Input::ALL {
            let mut total = 0.0;
            for t in [0.0, 1.3, 7.7] {
                let img = i.image(t);
                assert!(img.iter().all(|v| (0.0..=1.0).contains(v)));
                total += img.iter().sum::<f64>();
            }
            // Text scrolls through word gaps, so look across several times.
            assert!(total > 15.0, "{} is nearly blank", i.name());
        }
    }

    #[test]
    fn the_rig_climb_improves_on_the_naive_bench() {
        let mut l = Layouts::default();
        let mut c = RigClimb::new(Rig::naive(), &mut l, 1);
        let start = c.best.1;
        for _ in 0..60 {
            c.step(&mut l);
        }
        assert!(c.best.1 > start + 2.0, "{start:.2} → {:.2}", c.best.1);
    }

    #[test]
    fn lcd_map_places_every_weight() {
        let mut l = Layouts::default();
        let r = Rig::as_built();
        let e = Engine::new(&r.design(), &mut l, Workload::Blur.weights(), 1);
        let map = lcd_map(&e, r.cell as usize);
        let lit = map.rgba.chunks(4).filter(|p| p[2] > 0).count();
        assert_eq!(
            lit,
            e.tiles() * N,
            "square single-pixel cells never collide"
        );
    }
}
