//! Free-space matrix engine — the "glass engine".
//!
//! An emitter array shows the input vector; a relay images it onto a
//! grayscale glass mask made of tiles, one per output, each holding that
//! output's weights laid out like the emitters; summing optics focus each
//! tile's transmitted light onto one detector. Each detector reads one dot
//! product with fan-in n:
//!
//!   emitters ──relay──▶ mask tiles (weights) ──summing lens──▶ detectors
//!
//! Physics modelled, all of it geometry-dependent:
//! * imaging blur: diffraction (σ₀ = 0.21 λ/NA) plus aberration growing as
//!   NA² · field² — mirror-symmetric relays cancel the odd aberrations;
//! * in-tile crosstalk: light from emitter j landing in mask pixel k ≠ j;
//! * diffraction from the mask's pixel lattice: periodic lattices (square,
//!   hex) send 1 − fill² of the light into discrete ghost orders that can
//!   land on other detectors; aperiodic ones (sunflower, quasi) spread it
//!   into a diffuse halo;
//! * summing-lens blur and lenslet spill between tiles;
//! * drive skew across the emitter array (row bus vs fractal H-tree);
//! * receiver noise, 8-bit mask and ADC quantisation, mask contrast.
//!
//! Calibration is realistic: the controller pre-compensates mask values
//! and digitally unmixes detectors using *measured* crosstalk with a few
//! percent error, then the design is scored against the true physics.

use crate::core::Rng;
use crate::econ::receiver_power;
use crate::geometry::{drive_skew_s, Lattice, Ordering, H_TREE_ENERGY, P2};
use std::collections::HashMap;
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Replication {
    /// One emitter array; lenslets / a diffractive fan-out image it onto
    /// every tile. Inputs are converted once per symbol.
    Optical,
    /// The emitter array shows a copy per tile (like a display). Every
    /// copy costs a conversion.
    Electronic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Signed {
    /// Positive and negative weights on two tiles, subtracted.
    Differential,
    /// One tile holds (w + 1)/2; the offset Σx/2 is removed digitally.
    Offset,
}

/// Which error sources are switched on: all of them for scoring; one off
/// at a time for an error budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Terms {
    pub noise: bool,
    pub calibration: bool,
    pub ghosts: bool,
    pub lenslet: bool,
    pub crosstalk: bool,
    pub skew: bool,
    pub quantisation: bool,
}

impl Terms {
    pub const ALL: Terms = Terms {
        noise: true,
        calibration: true,
        ghosts: true,
        lenslet: true,
        crosstalk: true,
        skew: true,
        quantisation: true,
    };

    /// Every single-term-off variant, labelled.
    pub fn without_each() -> Vec<(&'static str, Terms)> {
        let a = Terms::ALL;
        vec![
            ("receiver noise", Terms { noise: false, ..a }),
            (
                "calibration error",
                Terms {
                    calibration: false,
                    ..a
                },
            ),
            ("diffraction ghosts and halo", Terms { ghosts: false, ..a }),
            (
                "lenslet spill",
                Terms {
                    lenslet: false,
                    ..a
                },
            ),
            (
                "in-tile crosstalk (blur)",
                Terms {
                    crosstalk: false,
                    ..a
                },
            ),
            ("drive skew", Terms { skew: false, ..a }),
            (
                "mask and ADC quantisation",
                Terms {
                    quantisation: false,
                    ..a
                },
            ),
        ]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Design {
    pub n_in: usize,
    pub m_out: usize,
    pub in_lattice: Lattice,
    pub tile_lattice: Lattice,
    pub order: Ordering,
    pub pitch_um: f64,
    pub fill: f64,
    pub na: f64,
    /// Mirror-symmetric relay about the stop (two lens groups).
    pub symmetric: bool,
    pub replication: Replication,
    /// Engines sharing one emitter array through beam splitters: an
    /// optical hyperedge of this degree.
    pub broadcast: u32,
    pub h_tree: bool,
    pub baud: f64,
    pub signed: Signed,
    pub wavelength_um: f64,
    /// Detector power relative to the ≈6-bit receiver design point
    /// (econ::receiver_power): more light, less relative noise, more watts.
    pub light: f64,
    /// Fixed receiver noise (fraction of full scale) for camera-style
    /// readout; 0 means a TIA receiver whose noise follows from `light`.
    pub noise_override: f64,
}

/// Physical constants of the optics, kept in one place. [A] = assumption.
pub mod optics {
    /// [A] aberration blur, µm, at NA 0.1 and 10 mm field radius.
    pub const ABERRATION_UM: f64 = 4.0;
    /// [A] odd aberrations cancelled by a symmetric relay: blur ÷ 4.
    pub const SYMMETRIC_GAIN: f64 = 0.25;
    /// [A] summing-lens focal length, µm (sets where ghosts land).
    pub const F_SUM_UM: f64 = 25_000.0;
    /// [A] detector plane is the mask plane demagnified 10×.
    pub const DEMAG: f64 = 0.1;
    /// [A] detector radius as a fraction of detector pitch.
    pub const DET_RADIUS: f64 = 0.35;
    /// [A] summing-spot blur as a fraction of detector pitch.
    pub const SUM_BLUR: f64 = 0.18;
    /// [A] lenslet spill into each neighbouring tile (optical replication).
    pub const LENSLET_SPILL: f64 = 0.004;
    /// [A] relative error of calibration: end-to-end gains are measured by
    /// averaging many probes (0.2 %); small crosstalk terms far less well (5 %).
    pub const CAL_GAIN_ERROR: f64 = 0.002;
    pub const CAL_XTALK_ERROR: f64 = 0.05;
    /// [A] equivalent field of a lenslet replicating the emitter array onto a
    /// tile at radius r: viewed off-axis, it behaves like a field of 0.5 r.
    pub const LENSLET_OFF_AXIS: f64 = 0.5;
    /// [A] mask contrast (minimum transmission) and bits.
    pub const MASK_TMIN: f64 = 0.002;
    pub const MASK_BITS: u32 = 8;
    pub const ADC_BITS: u32 = 8;
}

impl Design {
    /// The fast product design scored in docs/photonic/FIRST_PRINCIPLES.md.
    pub fn fast() -> Design {
        Design {
            n_in: 1024,
            m_out: 1024,
            in_lattice: Lattice::Square,
            tile_lattice: Lattice::Square,
            order: Ordering::Native,
            pitch_um: 20.0,
            fill: 0.8,
            na: 0.12,
            symmetric: false,
            replication: Replication::Optical,
            broadcast: 1,
            h_tree: false,
            baud: 2.5e9,
            signed: Signed::Differential,
            wavelength_um: 0.85,
            light: 1.0,
            noise_override: 0.0,
        }
    }

    /// The tabletop rig (docs/photonic/TABLETOP.md): a display shows copies
    /// of the input, a resin-printer LCD is the mask, a camera reads out.
    pub fn tabletop() -> Design {
        Design {
            n_in: 256,
            m_out: 256,
            in_lattice: Lattice::Square,
            tile_lattice: Lattice::Square,
            order: Ordering::Native,
            pitch_um: 50.0,
            fill: 0.75,
            na: 0.06,
            symmetric: false,
            replication: Replication::Electronic,
            broadcast: 1,
            h_tree: false,
            baud: 120.0,
            signed: Signed::Offset,
            wavelength_um: 0.53,
            light: 1.0,
            noise_override: 0.003,
        }
    }

    /// Detector power at full scale, W.
    pub fn detector_w(&self) -> f64 {
        receiver_power(self.baud) * self.light
    }

    /// Receiver noise as a fraction of full scale: TIA thermal noise
    /// (15 pA/√Hz) plus shot noise, over baud/2.
    pub fn noise_rel(&self) -> f64 {
        if self.noise_override > 0.0 {
            return self.noise_override;
        }
        let i = self.detector_w(); // 1 A/W
        let b = self.baud / 2.0;
        ((15e-12f64).powi(2) * b + 2.0 * 1.602e-19 * i * b).sqrt() / i
    }

    pub fn tiles(&self) -> usize {
        match self.signed {
            Signed::Differential => 2 * self.m_out,
            Signed::Offset => self.m_out,
        }
    }

    /// Pixel aperture side, µm. Square lattices tile with square pixels;
    /// other lattices get round pixels of diameter fill × pitch, modelled
    /// as the equal-area square (so diagonal neighbours never overlap).
    pub fn aperture_um(&self) -> f64 {
        let s = self.fill * self.pitch_um;
        if self.in_lattice == Lattice::Square {
            s
        } else {
            s * PI.sqrt() / 2.0
        }
    }

    /// Fraction of a pixel cell that transmits (also the zero-order share).
    pub fn area_fill(&self) -> f64 {
        (self.aperture_um() / self.pitch_um).powi(2)
    }

    fn sigma0_um(&self) -> f64 {
        0.21 * self.wavelength_um / self.na
    }

    /// Blur at field radius `r_um` from the optical axis.
    pub fn blur_um(&self, r_um: f64) -> f64 {
        let sym = if self.symmetric {
            optics::SYMMETRIC_GAIN
        } else {
            1.0
        };
        self.sigma0_um()
            + optics::ABERRATION_UM * (self.na / 0.1).powi(2) * (r_um / 10_000.0).powi(2) * sym
    }
}

/// Geometry that depends only on (lattice, ordering, count): cached.
#[derive(Clone)]
pub struct Layout {
    /// Unit-spacing positions; index j sits at pts[j].
    pub pts: Vec<P2>,
    /// For each j: nearby k (unit distance ≤ NEIGHBOUR_R), including j.
    pub near: Vec<Vec<usize>>,
    pub radius: f64,
}

/// Pixels farther than this (in pitches) never receive measurable light:
/// covers nearest, diagonal and next-nearest neighbours.
const NEIGHBOUR_R: f64 = 2.3;

impl Layout {
    pub fn new(lattice: Lattice, order: Ordering, n: usize) -> Layout {
        let pts = order.apply(lattice.points(n));
        let radius = pts
            .iter()
            .map(|p| (p.0 * p.0 + p.1 * p.1).sqrt())
            .fold(0.0, f64::max);
        let cell = NEIGHBOUR_R;
        let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        let key = |p: &P2| ((p.0 / cell).floor() as i64, (p.1 / cell).floor() as i64);
        for (i, p) in pts.iter().enumerate() {
            grid.entry(key(p)).or_default().push(i);
        }
        let near = pts
            .iter()
            .map(|p| {
                let (kx, ky) = key(p);
                let mut v = Vec::new();
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        if let Some(c) = grid.get(&(kx + dx, ky + dy)) {
                            for &k in c {
                                let d2 = (p.0 - pts[k].0).powi(2) + (p.1 - pts[k].1).powi(2);
                                if d2 <= NEIGHBOUR_R * NEIGHBOUR_R {
                                    v.push(k);
                                }
                            }
                        }
                    }
                }
                v
            })
            .collect();
        Layout { pts, near, radius }
    }
}

/// Cache of layouts, so a search re-scores designs without rebuilding them.
#[derive(Default)]
pub struct Layouts(HashMap<(Lattice, Ordering, usize), Layout>);

impl Layouts {
    pub fn get(&mut self, l: Lattice, o: Ordering, n: usize) -> &Layout {
        self.0
            .entry((l, o, n))
            .or_insert_with(|| Layout::new(l, o, n))
    }
}

fn erf(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7.
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    if x >= 0.0 {
        y
    } else {
        -y
    }
}

/// Fraction of a 1-D Gaussian (σ) centred at 0 that falls in [d − a/2, d + a/2].
fn box_frac(d: f64, a: f64, s: f64) -> f64 {
    if d.abs() - a / 2.0 > 4.5 * s {
        return 0.0;
    }
    let k = 1.0 / (2f64.sqrt() * s);
    0.5 * (erf((d + a / 2.0) * k) - erf((d - a / 2.0) * k))
}

/// Built physics for one design: positions in µm and the coupling model.
pub struct Physics {
    pub design: Design,
    pub pitch: f64,
    pub tile_pitch: f64,
    pub det_pitch: f64,
    /// Tile centres in the mask plane, µm (tile t serves output t / 2 when
    /// differential: even = positive, odd = negative).
    pub tiles: Vec<P2>,
    pub field_radius: f64,
    /// Detector-plane coupling: for each detector d, (tile t, fraction).
    pub chi: Vec<Vec<(usize, f64)>>,
    /// Calibration's view of `chi` (measured, with error).
    pub chi_cal: Vec<Vec<(usize, f64)>>,
    /// Detector positions, µm.
    pub det: Vec<P2>,
    /// Aperiodic masks: diffuse halo of this radius (µm) putting this
    /// fraction of each tile's light on every other detector inside it.
    pub halo: Option<(f64, f64)>,
    pub terms: Terms,
}

impl Physics {
    pub fn new(d: &Design, layouts: &mut Layouts, seed: u64) -> Physics {
        Physics::build(d, layouts, seed, Terms::ALL)
    }

    pub fn build(d: &Design, layouts: &mut Layouts, seed: u64, terms: Terms) -> Physics {
        let pitch = d.pitch_um;
        let pix_r = layouts.get(d.in_lattice, d.order, d.n_in).radius;
        let tile_pitch = (2.0 * pix_r + 3.0) * pitch;
        let det_pitch = tile_pitch * optics::DEMAG;
        let tl = layouts.get(d.tile_lattice, Ordering::Native, d.tiles());
        let tiles: Vec<P2> = tl
            .pts
            .iter()
            .map(|p| (p.0 * tile_pitch, p.1 * tile_pitch))
            .collect();
        let field_radius = tl.radius * tile_pitch + pix_r * pitch;

        // Detector-plane coupling.
        let det: Vec<P2> = tiles
            .iter()
            .map(|p| (p.0 * optics::DEMAG, p.1 * optics::DEMAG))
            .collect();
        let r_det = optics::DET_RADIUS * det_pitch;
        let s_sum = optics::SUM_BLUR * det_pitch;
        let capture = 1.0 - (-r_det * r_det / (2.0 * s_sum * s_sum)).exp();
        let spot_into = |dist: f64| -> f64 {
            // Fraction of a Gaussian spot (σ = s_sum) landing in a detector `dist` away.
            if dist < 1e-9 {
                capture
            } else {
                (PI * r_det * r_det) * (-dist * dist / (2.0 * s_sum * s_sum)).exp()
                    / (2.0 * PI * s_sum * s_sum)
            }
            .min(capture)
        };
        let zero_order = d.area_fill();
        let diffracted = 1.0 - zero_order;
        let ghost_shift = optics::F_SUM_UM * d.wavelength_um / pitch;
        let orders = d.in_lattice.first_orders();
        let n_t = tiles.len();
        let mut chi: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_t];
        let bucket = det_pitch.max(1e-9);
        let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        let key = |p: &P2| ((p.0 / bucket).floor() as i64, (p.1 / bucket).floor() as i64);
        for (i, p) in det.iter().enumerate() {
            grid.entry(key(p)).or_default().push(i);
        }
        let near_dets = |q: P2, r: f64| -> Vec<usize> {
            let (kx, ky) = key(&q);
            let span = (r / bucket).ceil() as i64 + 1;
            let mut v = Vec::new();
            for dx in -span..=span {
                for dy in -span..=span {
                    if let Some(c) = grid.get(&(kx + dx, ky + dy)) {
                        for &k in c {
                            if (det[k].0 - q.0).hypot(det[k].1 - q.1) <= r {
                                v.push(k);
                            }
                        }
                    }
                }
            }
            v
        };
        for (t, src) in det.iter().enumerate() {
            // Zero order: own detector and neighbours through summing blur.
            for k in near_dets(*src, 3.0 * s_sum + r_det) {
                let dist = (det[k].0 - src.0).hypot(det[k].1 - src.1);
                chi[k].push((t, zero_order * spot_into(dist)));
            }
            if d.in_lattice.periodic() && terms.ghosts {
                for u in &orders {
                    let q = (src.0 + ghost_shift * u.0, src.1 + ghost_shift * u.1);
                    for k in near_dets(q, 3.0 * s_sum + r_det) {
                        let dist = (det[k].0 - q.0).hypot(det[k].1 - q.1);
                        chi[k].push((t, diffracted / orders.len() as f64 * spot_into(dist)));
                    }
                }
            }
            if d.replication == Replication::Optical && terms.lenslet {
                for k in near_dets(*src, 1.05 * det_pitch) {
                    if k != t {
                        chi[k].push((t, optics::LENSLET_SPILL));
                    }
                }
            }
        }
        for row in &mut chi {
            row.sort_by_key(|e| e.0);
            let mut merged: Vec<(usize, f64)> = Vec::with_capacity(row.len());
            for &(t, f) in row.iter() {
                match merged.last_mut() {
                    Some(last) if last.0 == t => last.1 += f,
                    _ => merged.push((t, f)),
                }
            }
            *row = merged;
        }
        let mut rng = Rng::new(seed ^ 0xC411);
        let chi_cal = chi
            .iter()
            .enumerate()
            .map(|(dd, row)| {
                row.iter()
                    .map(|&(t, f)| {
                        let e = if !terms.calibration {
                            0.0
                        } else if t == dd {
                            optics::CAL_GAIN_ERROR
                        } else {
                            optics::CAL_XTALK_ERROR
                        };
                        (t, f * (1.0 + e * rng.gauss()))
                    })
                    .collect()
            })
            .collect();
        // Aperiodic: the diffracted light becomes a diffuse halo of radius ≈
        // the first-order shift, spread evenly over the detectors inside it.
        let halo = (!d.in_lattice.periodic() && terms.ghosts).then(|| {
            (
                ghost_shift,
                diffracted * (r_det * r_det) / (ghost_shift * ghost_shift).max(r_det * r_det),
            )
        });
        Physics {
            design: d.clone(),
            pitch,
            tile_pitch,
            det_pitch,
            tiles,
            field_radius,
            chi,
            chi_cal,
            det,
            halo,
            terms,
        }
    }

    /// In-tile crosstalk for tile `t`: K[j] = list of (k, fraction of
    /// emitter j's light landing in mask pixel k).
    pub fn kernel(&self, layout: &Layout, t: usize) -> Vec<Vec<(usize, f64)>> {
        let d = &self.design;
        let a = d.aperture_um();
        let tile_r = self.tiles[t].0.hypot(self.tiles[t].1);
        layout
            .pts
            .iter()
            .enumerate()
            .map(|(j, pj)| {
                let r = match d.replication {
                    // One big relay images the whole display: full field.
                    Replication::Electronic => (self.tiles[t].0 + pj.0 * self.pitch)
                        .hypot(self.tiles[t].1 + pj.1 * self.pitch),
                    // Each lenslet images one copy, viewing the emitter array
                    // off-axis: its own field plus an off-axis equivalent.
                    Replication::Optical => {
                        (pj.0 * self.pitch).hypot(pj.1 * self.pitch)
                            + optics::LENSLET_OFF_AXIS * tile_r
                    }
                };
                let s = d.blur_um(r);
                let row: Vec<(usize, f64)> = layout.near[j]
                    .iter()
                    .map(|&k| {
                        let pk = layout.pts[k];
                        let (dx, dy) = ((pk.0 - pj.0) * self.pitch, (pk.1 - pj.1) * self.pitch);
                        let fx = box_frac(dx, a, s);
                        (
                            k,
                            if fx > 0.0 {
                                fx * box_frac(dy, a, s)
                            } else {
                                0.0
                            },
                        )
                    })
                    .filter(|e| e.1 > 1e-7)
                    .collect();
                if self.terms.crosstalk {
                    row
                } else {
                    // Perfect focus: the light that blur spread over
                    // neighbouring pixels passes the emitter's own pixel
                    // instead (discarding it would cost signal, not crosstalk).
                    vec![(j, row.iter().map(|e| e.1).sum())]
                }
            })
            .collect()
    }
}

/// Mask values for one tile so that its effective row approximates
/// `target` (in transmission units), solved against a kernel with error.
fn solve_mask(
    target: &[f64],
    k: &[Vec<(usize, f64)>],
    kcal_err: &[Vec<f64>],
    calibrate: bool,
    quantise: bool,
) -> Vec<f64> {
    let n = target.len();
    let q = ((1u32 << optics::MASK_BITS) - 1) as f64;
    let quant = |t: f64| {
        optics::MASK_TMIN
            + ((t - optics::MASK_TMIN) / (1.0 - optics::MASK_TMIN) * q).round() / q
                * (1.0 - optics::MASK_TMIN)
    };
    let diag: Vec<f64> = (0..n)
        .map(|j| k[j].iter().find(|e| e.0 == j).map_or(1.0, |e| e.1))
        .collect();
    let mut t: Vec<f64> = (0..n)
        .map(|j| (target[j] / diag[j].max(1e-6)).clamp(optics::MASK_TMIN, 1.0))
        .collect();
    if calibrate {
        // Jacobi on the measured kernel: eff_j = Σ_k T_k K'[k→j].
        // K[j] lists (k, frac of j's light into pixel k); effective weight
        // seen by input j is Σ_k T_k · frac(j→k).
        for _ in 0..12 {
            let eff: Vec<f64> = (0..n)
                .map(|j| {
                    k[j].iter()
                        .zip(&kcal_err[j])
                        .map(|(&(kk, f), e)| t[kk] * f * e)
                        .sum()
                })
                .collect();
            for j in 0..n {
                let dj = k[j]
                    .iter()
                    .zip(&kcal_err[j])
                    .find(|(e, _)| e.0 == j)
                    .map_or(diag[j], |(e, c)| e.1 * c);
                t[j] = (t[j] + (target[j] * diag[j] / diag[j].max(1e-6) - eff[j]) / dj.max(1e-6))
                    .clamp(optics::MASK_TMIN, 1.0);
            }
        }
    }
    if quantise {
        t.into_iter().map(quant).collect()
    } else {
        t
    }
}

/// Score of one design.
#[derive(Debug, Clone, Copy)]
pub struct Score {
    pub enob: f64,
    pub macs_per_s: f64,
    pub power_w: f64,
    pub cost_usd: f64,
    pub field_mm: f64,
}

impl Score {
    pub fn tmac_per_usd(&self) -> f64 {
        self.macs_per_s / self.cost_usd / 1e12
    }
    pub fn tmac_per_w(&self) -> f64 {
        self.macs_per_s / self.power_w / 1e12
    }
    pub fn fj_per_mac(&self) -> f64 {
        self.power_w / self.macs_per_s * 1e15
    }
    /// The search objective: compute per dollar × per watt, with a steep
    /// penalty (3 decades per bit) below the precision floor.
    pub fn fitness(&self, floor: f64) -> f64 {
        self.tmac_per_usd().log10() + self.tmac_per_w().log10() - 3.0 * (floor - self.enob).max(0.0)
    }
}

/// Weights of the scoring workload: rows of a DCT-II (structured like real
/// transforms and convolutions; low rows smooth, high rows oscillating).
pub fn workload(i: usize, j: usize, n: usize, m: usize) -> f64 {
    let freq = (i * n / m.max(1)) as f64;
    (PI * (j as f64 + 0.5) * freq / n as f64).cos()
}

/// Mask target (transmission) for tile `t` under the scoring workload.
fn tile_target(d: &Design, t: usize) -> Vec<f64> {
    let (o, neg) = match d.signed {
        Signed::Differential => (t / 2, t % 2 == 1),
        Signed::Offset => (t, false),
    };
    (0..d.n_in)
        .map(|j| {
            let w = workload(o, j, d.n_in, d.m_out);
            match d.signed {
                Signed::Differential => {
                    if neg {
                        (-w).max(0.0)
                    } else {
                        w.max(0.0)
                    }
                }
                Signed::Offset => 0.5 * (w + 1.0),
            }
        })
        .collect()
}

/// Couplings weaker than this are aggregated rather than modelled tile by tile.
const STRONG: f64 = 1e-3;

/// Effective-bits measurement on `samples` outputs × `vectors` inputs.
///
/// Sampled outputs get the full in-tile kernel and calibration. Tiles that
/// couple into them strongly (lenslet spill, ghost orders) are modelled as
/// rows of their own-pixel transmission; the many tiny diffuse-halo
/// couplings are folded into one vector per detector. Both enter only
/// through calibration residuals, which are what the score measures.
pub fn enob(
    design: &Design,
    layouts: &mut Layouts,
    samples: usize,
    vectors: usize,
    seed: u64,
) -> (f64, Physics) {
    enob_terms(design, layouts, samples, vectors, seed, Terms::ALL)
}

pub fn enob_terms(
    design: &Design,
    layouts: &mut Layouts,
    samples: usize,
    vectors: usize,
    seed: u64,
    terms: Terms,
) -> (f64, Physics) {
    let phys = Physics::build(design, layouts, seed, terms);
    let layout = layouts
        .get(design.in_lattice, design.order, design.n_in)
        .clone();
    let n = design.n_in;
    let m = design.m_out;
    let mut rng = Rng::new(seed);
    let tile_of = |o: usize, neg: bool| match design.signed {
        Signed::Differential => 2 * o + neg as usize,
        Signed::Offset => o,
    };
    // Outputs to evaluate: the most central, the outermost, and random ones.
    let out_r = |o: usize| {
        let t = tile_of(o, false);
        phys.tiles[t].0.hypot(phys.tiles[t].1)
    };
    let mut outs: Vec<usize> = (0..m).collect();
    outs.sort_by(|a, b| out_r(*a).total_cmp(&out_r(*b)));
    let mut pick = vec![outs[0], outs[m - 1]];
    while pick.len() < samples.min(m) {
        let o = (rng.next_u64() % m as u64) as usize;
        if !pick.contains(&o) {
            pick.push(o);
        }
    }
    let mut det_tiles: Vec<usize> = Vec::new();
    for &o in &pick {
        det_tiles.push(tile_of(o, false));
        if design.signed == Signed::Differential {
            det_tiles.push(tile_of(o, true));
        }
    }

    // Sampled tiles: full kernel, calibrated mask, true effective row.
    let mut rows: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut kdiag_sum = 0.0f64;
    let mut kdiag_n = 0.0f64;
    for &t in &det_tiles {
        let k = phys.kernel(&layout, t);
        let kerr: Vec<Vec<f64>> = k
            .iter()
            .enumerate()
            .map(|(j, row)| {
                row.iter()
                    .map(|e| {
                        let err = if !terms.calibration {
                            0.0
                        } else if e.0 == j {
                            optics::CAL_GAIN_ERROR
                        } else {
                            optics::CAL_XTALK_ERROR
                        };
                        1.0 + err * rng.gauss()
                    })
                    .collect()
            })
            .collect();
        let tm = solve_mask(&tile_target(design, t), &k, &kerr, true, terms.quantisation);
        let row: Vec<f64> = (0..n)
            .map(|j| k[j].iter().map(|&(kk, f)| tm[kk] * f).sum())
            .collect();
        for (j, kr) in k.iter().enumerate() {
            kdiag_sum += kr.iter().find(|e| e.0 == j).map_or(0.0, |e| e.1);
            kdiag_n += 1.0;
        }
        rows.insert(t, row);
    }
    let kdiag = kdiag_sum / kdiag_n.max(1.0);
    let mut targets: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut target_of = |t: usize| -> Vec<f64> {
        targets
            .entry(t)
            .or_insert_with(|| tile_target(design, t))
            .clone()
    };
    let approx = |row: Vec<f64>| -> Vec<f64> { row.iter().map(|v| v * kdiag).collect() };

    // Per sampled detector: strong couplings exactly, weak ones aggregated.
    struct Det {
        own_true: f64,
        own_cal: f64,
        strong: Vec<(usize, f64, f64)>,
        weak: Vec<f64>,
        weak_cal_ratio: f64,
    }
    let mut dets: HashMap<usize, Det> = HashMap::new();
    for &dt in &det_tiles {
        let cal: HashMap<usize, f64> = phys.chi_cal[dt].iter().cloned().collect();
        let mut det = Det {
            own_true: 0.0,
            own_cal: 1.0,
            strong: Vec::new(),
            weak: vec![0.0; n],
            weak_cal_ratio: 1.0,
        };
        let (mut wt, mut wc) = (0.0, 0.0);
        for &(t, f) in &phys.chi[dt] {
            let fc = cal.get(&t).copied().unwrap_or(f);
            if t == dt {
                det.own_true = f;
                det.own_cal = fc;
            } else if f >= STRONG {
                rows.entry(t).or_insert_with(|| approx(target_of(t)));
                det.strong.push((t, f, fc));
            } else {
                for (wv, a) in det.weak.iter_mut().zip(approx(target_of(t))) {
                    *wv += f * a;
                }
                wt += f;
                wc += fc;
            }
        }
        if let Some((radius, per)) = phys.halo {
            // Uniform halo: average a fixed stride-sample of the tiles inside
            // it (up to 32) and scale by how many there are.
            let here = phys.det[dt];
            let inside: Vec<usize> = (0..phys.det.len())
                .filter(|&t| {
                    t != dt && (phys.det[t].0 - here.0).hypot(phys.det[t].1 - here.1) <= radius
                })
                .collect();
            let count = inside.len() as f64;
            let stride = (inside.len() / 32).max(1);
            let sample: Vec<usize> = inside.iter().step_by(stride).copied().collect();
            for &t in &sample {
                let w = per * kdiag * count / sample.len() as f64;
                for (wv, a) in det.weak.iter_mut().zip(target_of(t)) {
                    *wv += w * a;
                }
            }
            // Calibration measures the halo as one aggregate: independent
            // per-term errors average down as 1/√count.
            wt += per * count;
            let e = if terms.calibration {
                optics::CAL_XTALK_ERROR
            } else {
                0.0
            };
            wc += per * count * (1.0 + e / count.max(1.0).sqrt() * rng.gauss());
        }
        if wt > 0.0 {
            det.weak_cal_ratio = wc / wt;
        }
        dets.insert(dt, det);
    }

    // Skew: arrival-time offsets across the drive direction.
    let width_mm = match design.replication {
        Replication::Optical => 2.0 * layout.radius * phys.pitch / 1000.0,
        Replication::Electronic => 2.0 * phys.field_radius / 1000.0,
    };
    let skew_frac = if terms.skew {
        (drive_skew_s(width_mm, design.h_tree) * design.baud).min(1.0)
    } else {
        0.0
    };
    let xs: Vec<f64> = layout.pts.iter().map(|p| p.0).collect();
    let (xmin, xmax) = xs
        .iter()
        .fold((f64::MAX, f64::MIN), |a, &v| (a.0.min(v), a.1.max(v)));
    let lag: Vec<f64> = xs
        .iter()
        .map(|x| skew_frac * (x - xmin) / (xmax - xmin).max(1e-9))
        .collect();

    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>();
    // Full scale: every input at 1 through a fully open tile, as the
    // detector actually sees it. The source is sized to put `detector_w`
    // there, so noise is relative to this.
    let own_mean = dets.values().map(|d| d.own_true).sum::<f64>() / dets.len().max(1) as f64;
    let fs = n as f64 * kdiag * own_mean;
    let noise_rel = design.noise_rel();
    let lsb = 1.0 / ((1u32 << optics::ADC_BITS) - 1) as f64;
    let mut err_sq = 0.0;
    let mut count = 0.0;
    for _ in 0..vectors {
        let x: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
        let xp: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
        // What is actually integrated: part of the symbol still carries the last value.
        let xi: Vec<f64> = (0..n).map(|j| x[j] + lag[j] * (xp[j] - x[j])).collect();
        let sum_x: f64 = x.iter().sum();
        let signal: HashMap<usize, f64> = rows.iter().map(|(&t, r)| (t, dot(r, &xi))).collect();
        let measure = |dt: usize, rng: &mut Rng| -> f64 {
            let d = &dets[&dt];
            let y = d.own_true * signal[&dt]
                + d.strong
                    .iter()
                    .map(|&(t, f, _)| f * signal[&t])
                    .sum::<f64>()
                + dot(&d.weak, &xi);
            let noisy = y + if terms.noise {
                noise_rel * fs * rng.gauss()
            } else {
                0.0
            };
            let read = if terms.quantisation {
                ((noisy / fs).clamp(0.0, 1.0) / lsb).round() * lsb * fs
            } else {
                noisy
            };
            // Digital unmixing with the measured couplings.
            let others: f64 = d
                .strong
                .iter()
                .map(|&(t, _, fc)| fc * signal[&t])
                .sum::<f64>()
                + d.weak_cal_ratio * dot(&d.weak, &x);
            (read - others) / d.own_cal
        };
        for &o in &pick {
            let ideal: f64 = (0..n).map(|j| workload(o, j, n, m) * x[j]).sum::<f64>() / n as f64;
            let est = match design.signed {
                Signed::Differential => {
                    (measure(tile_of(o, false), &mut rng) - measure(tile_of(o, true), &mut rng))
                        / n as f64
                }
                Signed::Offset => (2.0 * measure(tile_of(o, false), &mut rng) - sum_x) / n as f64,
            };
            err_sq += (est - ideal).powi(2);
            count += 1.0;
        }
    }
    let rms = (err_sq / count).sqrt().max(1e-12);
    ((2.0 / (rms * 12f64.sqrt())).log2(), phys)
}

/// Throughput, power and cost of a design. Component prices and energies
/// are assumptions [A]; see docs/photonic/FIRST_PRINCIPLES.md.
pub fn economics(d: &Design, phys: &Physics, enob: f64) -> Score {
    let tiles = d.tiles() as f64;
    let n = d.n_in as f64;
    let k = d.broadcast.max(1) as f64;
    let rate = (d.baud / 2.5e9).min(1.0);
    // [A] converters get dearer per sample above ~2.5 GS/s (ADC figure of merit).
    let fast = 1.0 + 0.5 * (d.baud / 2.5e9).max(1.0).log2();
    let conv_in = match d.replication {
        Replication::Optical => n,
        Replication::Electronic => n * tiles,
    } / k;
    let e_in = (0.4 + 0.6 * rate) * fast * if d.h_tree { H_TREE_ENERGY } else { 1.0 };
    let e_out = (0.6 + 0.9 * rate) * fast;
    let mask_t = match d.signed {
        Signed::Differential => 0.3,
        Signed::Offset => 0.5,
    };
    let eta = d.area_fill()
        * mask_t
        * 0.81
        * if d.symmetric { 0.9 } else { 1.0 }
        * if d.replication == Replication::Optical {
            0.7
        } else {
            1.0
        }
        * if k > 1.0 { 0.85 } else { 1.0 };
    let optical_w = tiles * d.detector_w() / eta;
    let source_w = optical_w / 0.3;
    let static_w = 0.5 + 0.0005 * tiles;
    let sram_w = (n / k + tiles) * 8.0 * 0.08e-12 * d.baud;
    let power = (conv_in * e_in + tiles * e_out) * 1e-12 * d.baud + source_w + static_w + sram_w;

    let mm2 = |um: f64| (um / 1000.0) * (um / 1000.0);
    // [A] fast emitters need room: ≥20 µm pitch above 1 GBd, ≥10 µm above
    // 100 MBd. The relay demagnifies them onto the mask pitch.
    let emit_pitch = d.pitch_um.max(if d.baud >= 1e9 {
        20.0
    } else if d.baud >= 1e8 {
        10.0
    } else {
        5.0
    });
    let emit_area = match d.replication {
        Replication::Optical => n,
        Replication::Electronic => n * tiles,
    } * mm2(emit_pitch);
    let field_mm = 2.0 * phys.field_radius / 1000.0;
    let lenses = 1.0 + if d.symmetric { 1.0 } else { 0.0 } + 1.0;
    let lens_usd = lenses * (2.0 + 0.02 * field_mm * field_mm);
    let elements = lenses
        + 1.0
        + if d.replication == Replication::Optical {
            1.0
        } else {
            0.0
        }
        + if k > 1.0 { 1.0 } else { 0.0 };
    let cost = (2.0 + 0.6 * emit_area) / k                       // emitter array, shared by k engines
        + (1.0 + conv_in * 0.002) * if d.h_tree { 1.2 } else { 1.0 } // driver CMOS
        + 2.0 + tiles * 0.004                                     // detectors + readout CMOS
        + 1.0 + tiles * mm2(phys.tile_pitch) * 0.002              // glass mask ($0.20/cm²)
        + lens_usd
        + if d.replication == Replication::Optical { 5.0 } else { 0.0 }
        + if k > 1.0 { 3.0 } else { 0.0 }
        + 15.0 + 8.0 * elements                                   // assembly and alignment
        + 8.0; // test
    Score {
        enob,
        macs_per_s: n * d.m_out as f64 * d.baud,
        power_w: power,
        cost_usd: cost,
        field_mm,
    }
}

/// Score a design end to end.
pub fn score(d: &Design, layouts: &mut Layouts, seed: u64) -> Score {
    let (e, phys) = enob(d, layouts, 9, 48, seed);
    economics(d, &phys, e)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quiet(mut d: Design) -> Design {
        d.noise_override = 1e-6;
        d
    }

    #[test]
    fn clean_optics_reach_adc_limited_precision() {
        // High NA, small field, wide pitch, aperiodic lattice: almost no crosstalk.
        let mut d = quiet(Design::fast());
        d.n_in = 64;
        d.m_out = 64;
        d.pitch_um = 60.0;
        d.na = 0.2;
        d.in_lattice = Lattice::Sunflower;
        let (e, _) = enob(&d, &mut Layouts::default(), 6, 32, 1);
        assert!(e > 6.0, "ENOB {e}");
    }

    #[test]
    fn blur_costs_bits() {
        let mut layouts = Layouts::default();
        let mut sharp = quiet(Design::fast());
        sharp.n_in = 256;
        sharp.m_out = 64;
        sharp.na = 0.15;
        let mut soft = sharp.clone();
        soft.na = 0.012; // σ₀ ≈ 15 µm against a 20 µm pitch
        let (es, _) = enob(&sharp, &mut layouts, 6, 32, 2);
        let (ef, _) = enob(&soft, &mut layouts, 6, 32, 2);
        assert!(es > ef + 0.5, "sharp {es} soft {ef}");
    }

    #[test]
    fn aperiodic_mask_spreads_ghosts() {
        // Same everything; periodic lattices put ghost orders on detectors.
        let mut layouts = Layouts::default();
        let mut d = Design::fast();
        d.n_in = 256;
        d.m_out = 256;
        d.fill = 0.6; // strong diffraction
        let worst = |d: &Design, l: &mut Layouts| {
            let p = Physics::new(d, l, 3);
            p.chi
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    r.iter()
                        .filter(|e| e.0 != i)
                        .map(|e| e.1)
                        .fold(0.0, f64::max)
                })
                .fold(0.0, f64::max)
        };
        let periodic = worst(&d, &mut layouts);
        d.in_lattice = Lattice::Sunflower;
        let aperiodic = worst(&d, &mut layouts);
        assert!(
            aperiodic < periodic,
            "periodic {periodic} aperiodic {aperiodic}"
        );
    }

    #[test]
    fn fast_design_matches_econ_ballpark() {
        let s = score(&Design::fast(), &mut Layouts::default(), 4);
        assert!((s.macs_per_s - 2.62e15).abs() / 2.62e15 < 0.01);
        assert!(
            s.fj_per_mac() > 1.0 && s.fj_per_mac() < 50.0,
            "{} fJ/MAC",
            s.fj_per_mac()
        );
    }
}
