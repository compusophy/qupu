//! Geometry for free-space engines: where emitters, mask pixels and
//! detectors sit, and how a vector's indices map onto them.
//!
//! The lattices are chosen by symmetry group, because that is what decides
//! their optics: a lattice with translational symmetry (square p4m,
//! hexagonal p6m) diffracts into discrete orders — sharp ghost images —
//! while aperiodic point sets with long-range order (the Fibonacci
//! sunflower, the golden R2 quasi-lattice) spread the same light into a
//! diffuse halo. Index ordering along a Hilbert curve (a fractal) keeps
//! vector neighbours spatial neighbours.

use std::f64::consts::PI;

pub type P2 = (f64, f64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lattice {
    Square,
    Hex,
    Sunflower,
    Quasi,
}

impl Lattice {
    pub const ALL: [Lattice; 4] = [
        Lattice::Square,
        Lattice::Hex,
        Lattice::Sunflower,
        Lattice::Quasi,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Lattice::Square => "square (p4m)",
            Lattice::Hex => "hexagonal (p6m)",
            Lattice::Sunflower => "Fibonacci sunflower",
            Lattice::Quasi => "golden quasi-lattice (R2)",
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            Lattice::Square => "square",
            Lattice::Hex => "hex",
            Lattice::Sunflower => "sunflower",
            Lattice::Quasi => "quasi",
        }
    }

    /// Translational symmetry ⇒ discrete diffraction orders.
    pub fn periodic(self) -> bool {
        matches!(self, Lattice::Square | Lattice::Hex)
    }

    /// Unit vectors of the first diffraction orders of a periodic lattice
    /// (reciprocal-lattice directions); empty for aperiodic sets.
    pub fn first_orders(self) -> Vec<P2> {
        match self {
            Lattice::Square => vec![(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)],
            Lattice::Hex => (0..6)
                .map(|k| {
                    (
                        (PI / 6.0 + k as f64 * PI / 3.0).cos(),
                        (PI / 6.0 + k as f64 * PI / 3.0).sin(),
                    )
                })
                .collect(),
            _ => vec![],
        }
    }

    /// `n` points, centred, scaled so the closest pair is exactly 1 apart
    /// (pitch is defined by the minimum spacing, which sets crosstalk).
    pub fn points(self, n: usize) -> Vec<P2> {
        let raw: Vec<P2> = match self {
            Lattice::Square => {
                let s = (n as f64).sqrt().ceil() as usize;
                (0..n).map(|k| ((k % s) as f64, (k / s) as f64)).collect()
            }
            Lattice::Hex => {
                // Hexagonal disk: lattice points sorted by radius.
                let r = ((n as f64).sqrt() as i64) + 2;
                let mut pts: Vec<P2> = Vec::new();
                for j in -r..=r {
                    for i in -r..=r {
                        pts.push((i as f64 + 0.5 * j as f64, j as f64 * 3f64.sqrt() / 2.0));
                    }
                }
                pts.sort_by(|a, b| (a.0 * a.0 + a.1 * a.1).total_cmp(&(b.0 * b.0 + b.1 * b.1)));
                pts.truncate(n);
                pts
            }
            Lattice::Sunflower => {
                // Vogel's model: the golden angle gives Fibonacci parastichies.
                let golden = PI * (3.0 - 5f64.sqrt());
                (0..n)
                    .map(|k| {
                        let r = (k as f64 + 0.5).sqrt();
                        let t = k as f64 * golden;
                        (r * t.cos(), r * t.sin())
                    })
                    .collect()
            }
            Lattice::Quasi => {
                // Roberts' R2 sequence: the 2-D generalised golden ratio.
                let g = 1.324_717_957_244_746; // plastic number
                let (a1, a2) = (1.0 / g, 1.0 / (g * g));
                let side = (n as f64).sqrt();
                (0..n)
                    .map(|k| {
                        (
                            (0.5 + a1 * k as f64).fract() * side,
                            (0.5 + a2 * k as f64).fract() * side,
                        )
                    })
                    .collect()
            }
        };
        normalise(raw)
    }
}

/// Centre a point set and scale it to unit minimum spacing.
fn normalise(mut pts: Vec<P2>) -> Vec<P2> {
    let n = pts.len() as f64;
    let (cx, cy) = pts
        .iter()
        .fold((0.0, 0.0), |a, p| (a.0 + p.0 / n, a.1 + p.1 / n));
    for p in &mut pts {
        p.0 -= cx;
        p.1 -= cy;
    }
    let d = min_spacing(&pts).max(1e-9);
    for p in &mut pts {
        p.0 /= d;
        p.1 /= d;
    }
    pts
}

/// Smallest distance between any two points (grid-bucketed, ~O(n)).
pub fn min_spacing(pts: &[P2]) -> f64 {
    if pts.len() < 2 {
        return 1.0;
    }
    let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
    for p in pts {
        lo = (lo.0.min(p.0), lo.1.min(p.1));
        hi = (hi.0.max(p.0), hi.1.max(p.1));
    }
    let cell = ((hi.0 - lo.0) * (hi.1 - lo.1) / pts.len() as f64)
        .sqrt()
        .max(1e-9);
    let key = |p: &P2| (((p.0 - lo.0) / cell) as i64, ((p.1 - lo.1) / cell) as i64);
    let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> =
        std::collections::HashMap::new();
    for (i, p) in pts.iter().enumerate() {
        grid.entry(key(p)).or_default().push(i);
    }
    let mut best = f64::MAX;
    for (i, p) in pts.iter().enumerate() {
        let (kx, ky) = key(p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(v) = grid.get(&(kx + dx, ky + dy)) {
                    for &j in v {
                        if j != i {
                            let d = ((p.0 - pts[j].0).powi(2) + (p.1 - pts[j].1).powi(2)).sqrt();
                            best = best.min(d);
                        }
                    }
                }
            }
        }
    }
    best
}

/// How vector indices map onto points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ordering {
    /// The lattice's own order (row-major for square; outward for the rest).
    Native,
    /// Along a Hilbert curve through the points' positions.
    Hilbert,
}

impl Ordering {
    pub fn name(self) -> &'static str {
        match self {
            Ordering::Native => "native",
            Ordering::Hilbert => "Hilbert",
        }
    }

    /// Reorder points so that index k sits at the returned position k.
    pub fn apply(self, pts: Vec<P2>) -> Vec<P2> {
        match self {
            Ordering::Native => pts,
            Ordering::Hilbert => {
                let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
                for p in &pts {
                    lo = (lo.0.min(p.0), lo.1.min(p.1));
                    hi = (hi.0.max(p.0), hi.1.max(p.1));
                }
                let span = (hi.0 - lo.0).max(hi.1 - lo.1).max(1e-9);
                let order = 1u32 << 10;
                let mut keyed: Vec<(u64, P2)> = pts
                    .into_iter()
                    .map(|p| {
                        let x = (((p.0 - lo.0) / span) * (order - 1) as f64) as u32;
                        let y = (((p.1 - lo.1) / span) * (order - 1) as f64) as u32;
                        (hilbert_index(order, x, y), p)
                    })
                    .collect();
                keyed.sort_by_key(|k| k.0);
                keyed.into_iter().map(|k| k.1).collect()
            }
        }
    }
}

/// Distance along a Hilbert curve filling an `n`×`n` grid (n a power of 2).
pub fn hilbert_index(n: u32, mut x: u32, mut y: u32) -> u64 {
    let mut d = 0u64;
    let mut s = n / 2;
    while s > 0 {
        let rx = u32::from((x & s) > 0);
        let ry = u32::from((y & s) > 0);
        d += (s as u64) * (s as u64) * ((3 * rx) ^ ry) as u64;
        if ry == 0 {
            if rx == 1 {
                x = s - 1 - (x & (s - 1));
                y = s - 1 - (y & (s - 1));
            }
            std::mem::swap(&mut x, &mut y);
        }
        s /= 2;
    }
    d
}

/// Worst arrival-time difference across an emitter array of width `w_mm`
/// driven from one edge: row-bus wiring vs a fractal H-tree, whose paths
/// to every leaf are equal length by construction. Electrical signals in
/// on-chip/PCB wiring travel ≈7 ps/mm.
pub fn drive_skew_s(w_mm: f64, h_tree: bool) -> f64 {
    if h_tree {
        0.0
    } else {
        w_mm * 7e-12
    }
}

/// H-trees pay for equal lengths with total wire: ≈ +25 % drive energy.
pub const H_TREE_ENERGY: f64 = 1.25;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_lattice_has_unit_minimum_spacing() {
        for l in Lattice::ALL {
            for n in [64, 256, 1000] {
                let pts = l.points(n);
                assert_eq!(pts.len(), n);
                assert!((min_spacing(&pts) - 1.0).abs() < 1e-6, "{l:?} n={n}");
            }
        }
    }

    #[test]
    fn sunflower_is_more_compact_than_square() {
        // Same count and minimum spacing: the disk-like set keeps its far
        // points closer to the axis, where lens aberrations are smallest.
        let rms = |p: &[P2]| {
            (p.iter().map(|q| q.0 * q.0 + q.1 * q.1).sum::<f64>() / p.len() as f64).sqrt()
        };
        let max_r = |p: &[P2]| {
            p.iter()
                .map(|q| (q.0 * q.0 + q.1 * q.1).sqrt())
                .fold(0.0, f64::max)
        };
        let (s, f) = (
            Lattice::Square.points(1024),
            Lattice::Sunflower.points(1024),
        );
        assert!(max_r(&f) < max_r(&s), "{} vs {}", max_r(&f), max_r(&s));
        assert!(rms(&f) < 1.4 * rms(&s));
    }

    #[test]
    fn hilbert_keeps_index_neighbours_close() {
        let pts = Lattice::Square.points(1024);
        let mean_step = |p: &[P2]| {
            p.windows(2)
                .map(|w| ((w[0].0 - w[1].0).powi(2) + (w[0].1 - w[1].1).powi(2)).sqrt())
                .sum::<f64>()
                / (p.len() - 1) as f64
        };
        let h = Ordering::Hilbert.apply(pts.clone());
        assert!(
            mean_step(&h) <= 1.0 + 1e-9,
            "Hilbert steps {}",
            mean_step(&h)
        );
        assert!(mean_step(&pts) > 1.0);
    }

    #[test]
    fn hilbert_index_is_a_bijection() {
        let n = 16;
        let mut seen = vec![false; (n * n) as usize];
        for x in 0..n {
            for y in 0..n {
                let d = hilbert_index(n, x, y) as usize;
                assert!(!seen[d]);
                seen[d] = true;
            }
        }
    }
}
