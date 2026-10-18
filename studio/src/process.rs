//! Process model: what a fab run teaches us, learned across devices.
//!
//! Evanescent coupling decays exponentially with gap, so at fixed radius
//! ln κ²(gap) is fitted as a straight line. Propagation loss is taken per
//! radius (bend loss grows as rings shrink). Group index is shared.
//!
//! The leave-one-out test is the honesty check: fit the law without a
//! device, predict that device's loaded Q, compare with its own fit.

use crate::fit::DeviceFit;
use crate::ring::Ring;
use std::f64::consts::PI;

/// Summary of one device at the resonance nearest 1550 nm.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    pub radius_um: f64,
    pub gap_nm: f64,
    pub n_g: f64,
    pub kappa2: f64,
    pub alpha_per_m: f64,
    pub q_loaded: f64,
}

impl Sample {
    pub fn of(d: &DeviceFit) -> Self {
        let r = d.nearest(1550.0).ring;
        Sample {
            radius_um: d.radius_um,
            gap_nm: d.gap_nm,
            n_g: d.n_g,
            kappa2: r.kappa2(),
            alpha_per_m: r.alpha_per_m(),
            q_loaded: r.q_loaded(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CouplingLaw {
    pub radius_um: f64,
    /// ln κ² = intercept + slope · gap_nm
    pub intercept: f64,
    pub slope: f64,
    pub alpha_per_m: f64,
}

pub struct ProcessModel {
    pub n_g: f64,
    pub laws: Vec<CouplingLaw>,
}

impl ProcessModel {
    /// Learn from samples. Radii with a single gap get a law only if a
    /// shared decay slope is available from other radii.
    pub fn learn(samples: &[Sample]) -> Self {
        let n_g = samples.iter().map(|s| s.n_g).sum::<f64>() / samples.len() as f64;
        let mut radii: Vec<f64> = samples.iter().map(|s| s.radius_um).collect();
        radii.sort_by(f64::total_cmp);
        radii.dedup();
        let mut laws = Vec::new();
        for &r in &radii {
            let pts: Vec<&Sample> = samples.iter().filter(|s| s.radius_um == r).collect();
            let alpha = pts.iter().map(|s| s.alpha_per_m).sum::<f64>() / pts.len() as f64;
            if pts.len() >= 2 {
                let (slope, intercept) = line_fit(pts.iter().map(|s| (s.gap_nm, s.kappa2.ln())));
                laws.push(CouplingLaw {
                    radius_um: r,
                    intercept,
                    slope,
                    alpha_per_m: alpha,
                });
            }
        }
        // Borrow the mean decay slope for single-gap radii.
        if !laws.is_empty() {
            let mean_slope = laws.iter().map(|l| l.slope).sum::<f64>() / laws.len() as f64;
            for &r in &radii {
                if laws.iter().any(|l| l.radius_um == r) {
                    continue;
                }
                let pts: Vec<&Sample> = samples.iter().filter(|s| s.radius_um == r).collect();
                let s = pts[0];
                laws.push(CouplingLaw {
                    radius_um: r,
                    intercept: s.kappa2.ln() - mean_slope * s.gap_nm,
                    slope: mean_slope,
                    alpha_per_m: s.alpha_per_m,
                });
            }
        }
        ProcessModel { n_g, laws }
    }

    pub fn law(&self, radius_um: f64) -> Option<&CouplingLaw> {
        self.laws.iter().find(|l| l.radius_um == radius_um)
    }

    /// Predicted symmetric ring at 1550 nm for a geometry on a learned radius.
    pub fn predict(&self, radius_um: f64, gap_nm: f64) -> Option<Ring> {
        let law = self.law(radius_um)?;
        let kappa2 = (law.intercept + law.slope * gap_nm).exp().min(0.999);
        let r = (1.0 - kappa2).sqrt();
        let l = 2.0 * PI * radius_um * 1e-6;
        let a = (-law.alpha_per_m * l / 2.0).exp();
        Some(Ring {
            radius_um,
            n_g: self.n_g,
            r1: r,
            r2: r,
            a,
            lambda_res_nm: 1550.0,
        })
    }
}

/// Leave-one-out prediction of loaded Q for every device whose radius has
/// at least two other gaps to learn from.
pub struct Holdout {
    pub sample: Sample,
    pub predicted_q: f64,
}

impl Holdout {
    pub fn error(&self) -> f64 {
        (self.predicted_q - self.sample.q_loaded) / self.sample.q_loaded
    }
}

pub fn leave_one_out(samples: &[Sample]) -> Vec<Holdout> {
    let mut out = Vec::new();
    for (i, s) in samples.iter().enumerate() {
        let same_radius = samples
            .iter()
            .enumerate()
            .filter(|(j, o)| *j != i && o.radius_um == s.radius_um)
            .count();
        if same_radius < 2 {
            continue;
        }
        let rest: Vec<Sample> = samples
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, s)| *s)
            .collect();
        let model = ProcessModel::learn(&rest);
        if let Some(ring) = model.predict(s.radius_um, s.gap_nm) {
            out.push(Holdout {
                sample: *s,
                predicted_q: ring.q_loaded(),
            });
        }
    }
    out
}

fn line_fit(pts: impl Iterator<Item = (f64, f64)>) -> (f64, f64) {
    let v: Vec<(f64, f64)> = pts.collect();
    let n = v.len() as f64;
    let (sx, sy) = v.iter().fold((0.0, 0.0), |(a, b), (x, y)| (a + x, b + y));
    let (mx, my) = (sx / n, sy / n);
    let sxx: f64 = v.iter().map(|(x, _)| (x - mx).powi(2)).sum();
    let sxy: f64 = v.iter().map(|(x, y)| (x - mx) * (y - my)).sum();
    let slope = sxy / sxx;
    (slope, my - slope * mx)
}
