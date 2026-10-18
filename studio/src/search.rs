//! Hill climbing over geometry and architecture.
//!
//! The genome is a `freespace::Design`: lattice symmetry group for pixels
//! and for tiles (square p4m, hexagonal p6m, Fibonacci sunflower, golden
//! quasi-lattice), index ordering (native or Hilbert), pitch, fill, NA,
//! mirror-symmetric relay, optical vs electronic replication, broadcast
//! degree (an optical hyperedge feeding k engines from one emitter array),
//! fractal H-tree drive, symbol rate, signed-weight scheme and size.
//!
//! Fitness is compute per dollar × per watt, from the physics (precision)
//! and the economics (cost, power); below 5 effective bits it collapses.
//! A (1+λ) evolution strategy climbs it, restarting when stuck, and keeps
//! a Pareto archive of $-efficient vs energy-efficient designs.

use crate::core::Rng;
use crate::freespace::{Design, Replication, Score, Signed};
use crate::geometry::{Lattice, Ordering};

pub const SIZES: [usize; 5] = [64, 144, 256, 576, 1024];
const BROADCAST: [u32; 4] = [1, 2, 4, 8];

fn pick<T: Copy>(rng: &mut Rng, v: &[T]) -> T {
    v[(rng.next_u64() % v.len() as u64) as usize]
}

fn log_uniform(rng: &mut Rng, lo: f64, hi: f64) -> f64 {
    (lo.ln() + rng.uniform() * (hi.ln() - lo.ln())).exp()
}

pub fn random_design(rng: &mut Rng) -> Design {
    Design {
        n_in: pick(rng, &SIZES),
        m_out: pick(rng, &SIZES),
        in_lattice: pick(rng, &Lattice::ALL),
        tile_lattice: pick(rng, &Lattice::ALL),
        order: pick(rng, &[Ordering::Native, Ordering::Hilbert]),
        pitch_um: log_uniform(rng, 5.0, 80.0),
        fill: 0.4 + 0.55 * rng.uniform(),
        na: log_uniform(rng, 0.01, 0.35),
        symmetric: rng.uniform() < 0.5,
        replication: pick(rng, &[Replication::Optical, Replication::Electronic]),
        broadcast: pick(rng, &BROADCAST),
        h_tree: rng.uniform() < 0.5,
        baud: log_uniform(rng, 1e7, 5e9),
        signed: pick(rng, &[Signed::Differential, Signed::Offset]),
        light: log_uniform(rng, 0.25, 16.0),
        ..Design::fast()
    }
}

/// Change one gene, sometimes two.
pub fn mutate(d: &Design, rng: &mut Rng) -> Design {
    let mut c = d.clone();
    let genes = if rng.uniform() < 0.3 { 2 } else { 1 };
    for _ in 0..genes {
        let step = |rng: &mut Rng, v: usize, set: &[usize]| {
            let i = set.iter().position(|&s| s == v).unwrap_or(0) as i64
                + if rng.uniform() < 0.5 { -1 } else { 1 };
            set[i.clamp(0, set.len() as i64 - 1) as usize]
        };
        match rng.next_u64() % 14 {
            13 => c.light = (c.light * (0.4 * rng.gauss()).exp()).clamp(0.25, 16.0),
            0 => c.n_in = step(rng, c.n_in, &SIZES),
            1 => c.m_out = step(rng, c.m_out, &SIZES),
            2 => c.in_lattice = pick(rng, &Lattice::ALL),
            3 => c.tile_lattice = pick(rng, &Lattice::ALL),
            4 => {
                c.order = if c.order == Ordering::Native {
                    Ordering::Hilbert
                } else {
                    Ordering::Native
                }
            }
            5 => c.pitch_um = (c.pitch_um * (0.3 * rng.gauss()).exp()).clamp(5.0, 80.0),
            6 => c.fill = (c.fill + 0.06 * rng.gauss()).clamp(0.4, 0.95),
            7 => c.na = (c.na * (0.3 * rng.gauss()).exp()).clamp(0.01, 0.35),
            8 => c.symmetric = !c.symmetric,
            9 => {
                c.replication = if c.replication == Replication::Optical {
                    Replication::Electronic
                } else {
                    Replication::Optical
                }
            }
            10 => {
                let i = BROADCAST
                    .iter()
                    .position(|&b| b == c.broadcast)
                    .unwrap_or(0) as i64
                    + if rng.uniform() < 0.5 { -1 } else { 1 };
                c.broadcast = BROADCAST[i.clamp(0, 3) as usize];
            }
            11 => c.h_tree = !c.h_tree,
            _ => {
                if rng.uniform() < 0.7 {
                    c.baud = (c.baud * (0.5 * rng.gauss()).exp()).clamp(1e7, 5e9);
                } else {
                    c.signed = if c.signed == Signed::Offset {
                        Signed::Differential
                    } else {
                        Signed::Offset
                    };
                }
            }
        }
    }
    c
}

pub fn describe(d: &Design) -> String {
    format!(
        "{}×{} · pixels {} ({}) · tiles {} · pitch {:.1} µm · fill {:.2} · NA {:.3}{} · {} replication · broadcast ×{} · {} drive · {} · {} · light ×{:.1}",
        d.n_in,
        d.m_out,
        d.in_lattice.short(),
        d.order.name(),
        d.tile_lattice.short(),
        d.pitch_um,
        d.fill,
        d.na,
        if d.symmetric { " symmetric relay" } else { "" },
        if d.replication == Replication::Optical { "optical" } else { "electronic" },
        d.broadcast,
        if d.h_tree { "H-tree" } else { "row-bus" },
        fmt_rate(d.baud),
        if d.signed == Signed::Differential { "differential" } else { "offset" },
        d.light,
    )
}

pub fn fmt_rate(b: f64) -> String {
    if b >= 1e9 {
        format!("{:.2} GBd", b / 1e9)
    } else {
        format!("{:.0} MBd", b / 1e6)
    }
}

#[derive(Debug, Clone)]
pub struct Candidate {
    pub design: Design,
    pub score: Score,
    pub floor: f64,
}

impl Candidate {
    pub fn fitness(&self) -> f64 {
        self.score.fitness(self.floor)
    }
}

/// Precision floor used by the search unless told otherwise.
pub const DEFAULT_FLOOR: f64 = 6.0;

pub struct Climb {
    pub best: Candidate,
    /// (evaluations so far, best fitness) at each improvement.
    pub history: Vec<(usize, f64)>,
    pub evals: usize,
    /// Non-dominated designs on (bits, TMAC/s per $, TMAC/s per W) above
    /// the floor.
    pub pareto: Vec<Candidate>,
}

fn dominates(a: &Score, b: &Score) -> bool {
    let (x, y) = (
        [a.enob, a.tmac_per_usd(), a.tmac_per_w()],
        [b.enob, b.tmac_per_usd(), b.tmac_per_w()],
    );
    x.iter().zip(&y).all(|(p, q)| p >= q) && x.iter().zip(&y).any(|(p, q)| p > q)
}

pub fn archive(pareto: &mut Vec<Candidate>, c: &Candidate, floor: f64) {
    if c.score.enob < floor
        || pareto
            .iter()
            .any(|p| dominates(&p.score, &c.score) || p.design == c.design)
    {
        return;
    }
    pareto.retain(|p| !dominates(&c.score, &p.score));
    pareto.push(c.clone());
}

/// (1+λ) evolution strategy with restarts. `fix` pins genes after every
/// mutation (for constrained runs); `eval` scores a batch (in parallel if
/// the caller likes).
pub fn climb(
    budget: usize,
    lambda: usize,
    seed: u64,
    start: Option<Design>,
    fix: &dyn Fn(&mut Design),
    eval: &mut dyn FnMut(&[Design]) -> Vec<Score>,
    floor: f64,
) -> Climb {
    let mut rng = Rng::new(seed);
    let mut first = start.unwrap_or_else(|| random_design(&mut rng));
    fix(&mut first);
    let s = eval(std::slice::from_ref(&first))[0];
    let mut parent = Candidate {
        design: first,
        score: s,
        floor,
    };
    let mut best = parent.clone();
    let mut evals = 1;
    let mut history = vec![(evals, best.fitness())];
    let mut pareto = Vec::new();
    archive(&mut pareto, &parent, floor);
    let mut stuck = 0;
    while evals < budget {
        let kids: Vec<Design> = (0..lambda)
            .map(|_| {
                let mut k = mutate(&parent.design, &mut rng);
                fix(&mut k);
                k
            })
            .collect();
        let scores = eval(&kids);
        evals += kids.len();
        let mut improved = false;
        for (d, s) in kids.into_iter().zip(scores) {
            let c = Candidate {
                design: d,
                score: s,
                floor,
            };
            archive(&mut pareto, &c, floor);
            if c.fitness().is_finite() && c.fitness() > parent.fitness() {
                parent = c;
                improved = true;
            }
        }
        if parent.fitness() > best.fitness() {
            best = parent.clone();
            history.push((evals, best.fitness()));
        }
        stuck = if improved { 0 } else { stuck + 1 };
        if stuck >= 12 {
            // Restart: half the time from a fresh random design, half from
            // a heavily mutated copy of the best.
            let mut d = if rng.uniform() < 0.5 {
                random_design(&mut rng)
            } else {
                let mut d = best.design.clone();
                for _ in 0..4 {
                    d = mutate(&d, &mut rng);
                }
                d
            };
            fix(&mut d);
            let s = eval(std::slice::from_ref(&d))[0];
            evals += 1;
            parent = Candidate {
                design: d,
                score: s,
                floor,
            };
            archive(&mut pareto, &parent, floor);
            stuck = 0;
        }
    }
    pareto.sort_by(|a, b| a.score.tmac_per_usd().total_cmp(&b.score.tmac_per_usd()));
    Climb {
        best,
        history,
        evals,
        pareto,
    }
}

/// One-gene alternatives to a design, labelled, for ablation tables.
pub fn neighbours(d: &Design) -> Vec<(String, Design)> {
    let mut v: Vec<(String, Design)> = Vec::new();
    let mut add = |label: String, f: &dyn Fn(&mut Design)| {
        let mut c = d.clone();
        f(&mut c);
        if c != *d {
            v.push((label, c));
        }
    };
    for l in Lattice::ALL {
        add(format!("pixels → {}", l.short()), &|c: &mut Design| {
            c.in_lattice = l
        });
        add(format!("tiles → {}", l.short()), &|c: &mut Design| {
            c.tile_lattice = l
        });
    }
    add(
        "ordering flipped (native ↔ Hilbert)".into(),
        &|c: &mut Design| {
            c.order = if c.order == Ordering::Native {
                Ordering::Hilbert
            } else {
                Ordering::Native
            }
        },
    );
    add("symmetric relay toggled".into(), &|c: &mut Design| {
        c.symmetric = !c.symmetric
    });
    add("H-tree drive toggled".into(), &|c: &mut Design| {
        c.h_tree = !c.h_tree
    });
    add("replication flipped".into(), &|c: &mut Design| {
        c.replication = if c.replication == Replication::Optical {
            Replication::Electronic
        } else {
            Replication::Optical
        }
    });
    add("signed scheme flipped".into(), &|c: &mut Design| {
        c.signed = if c.signed == Signed::Offset {
            Signed::Differential
        } else {
            Signed::Offset
        }
    });
    for b in BROADCAST {
        add(format!("broadcast → ×{b}"), &|c: &mut Design| {
            c.broadcast = b
        });
    }
    add("NA ×0.5".into(), &|c: &mut Design| c.na *= 0.5);
    add("NA ×2".into(), &|c: &mut Design| {
        c.na = (c.na * 2.0).min(0.35)
    });
    add("pitch ×0.5".into(), &|c: &mut Design| {
        c.pitch_um = (c.pitch_um * 0.5).max(5.0)
    });
    add("pitch ×2".into(), &|c: &mut Design| {
        c.pitch_um = (c.pitch_um * 2.0).min(80.0)
    });
    add("symbol rate ×0.25".into(), &|c: &mut Design| {
        c.baud = (c.baud * 0.25).max(1e7)
    });
    add("symbol rate ×4".into(), &|c: &mut Design| {
        c.baud = (c.baud * 4.0).min(5e9)
    });
    add("light ×0.5".into(), &|c: &mut Design| {
        c.light = (c.light * 0.5).max(0.25)
    });
    add("light ×2".into(), &|c: &mut Design| {
        c.light = (c.light * 2.0).min(16.0)
    });
    v
}
