//! Optimiser module: a small Levenberg–Marquardt solver, and the
//! measured-spectrum fits built on it.
//!
//! Each resonance is fitted jointly on the through and drop ports, in dB,
//! with six parameters: phase offset, r and a (logit-transformed so they
//! stay in (0, 1)), separate through/drop insertion offsets (two different
//! grating-coupler pairs) and a shared linear tilt for the grating-coupler
//! envelope. The group index comes from the device's resonance spacing.

use crate::measure::Measurement;
use crate::ring::{add_drop, loss_from_extinction, rra_from_finesse, Ring, C};
use std::f64::consts::PI;

pub struct LmResult {
    pub params: Vec<f64>,
    pub cost: f64,
    pub iterations: usize,
}

/// Minimise ½‖r(p)‖² with a forward-difference Jacobian. Parameters should
/// be scaled to order one by the caller.
pub fn levenberg_marquardt(
    residuals: impl Fn(&[f64], &mut Vec<f64>),
    p0: &[f64],
    max_iter: usize,
) -> LmResult {
    let n = p0.len();
    let mut p = p0.to_vec();
    let mut r = Vec::new();
    residuals(&p, &mut r);
    let mut cost = 0.5 * r.iter().map(|x| x * x).sum::<f64>();
    let mut lambda = 1e-3;
    let mut rp = Vec::new();
    let mut jac = vec![0.0; n * r.len()];
    let mut it = 0;
    while it < max_iter {
        it += 1;
        let m = r.len();
        for k in 0..n {
            let h = 1e-6 * p[k].abs().max(1e-2);
            let mut q = p.clone();
            q[k] += h;
            residuals(&q, &mut rp);
            for i in 0..m {
                jac[i * n + k] = (rp[i] - r[i]) / h;
            }
        }
        // Normal equations JᵀJ δ = −Jᵀr with Marquardt diagonal damping.
        let mut jtj = vec![0.0; n * n];
        let mut jtr = vec![0.0; n];
        for i in 0..m {
            let row = &jac[i * n..(i + 1) * n];
            for a in 0..n {
                jtr[a] += row[a] * r[i];
                for b in 0..n {
                    jtj[a * n + b] += row[a] * row[b];
                }
            }
        }
        let mut improved = false;
        for _ in 0..12 {
            let mut aug = jtj.clone();
            for a in 0..n {
                aug[a * n + a] += lambda * jtj[a * n + a].max(1e-12);
            }
            let Some(delta) = solve(aug, jtr.iter().map(|x| -x).collect(), n) else {
                lambda *= 10.0;
                continue;
            };
            let q: Vec<f64> = p.iter().zip(&delta).map(|(a, b)| a + b).collect();
            residuals(&q, &mut rp);
            let c = 0.5 * rp.iter().map(|x| x * x).sum::<f64>();
            if c.is_finite() && c < cost {
                let rel = (cost - c) / cost.max(1e-300);
                p = q;
                std::mem::swap(&mut r, &mut rp);
                cost = c;
                lambda = (lambda / 3.0).max(1e-12);
                improved = true;
                if rel < 1e-12 {
                    return LmResult {
                        params: p,
                        cost,
                        iterations: it,
                    };
                }
                break;
            }
            lambda *= 4.0;
        }
        if !improved {
            break;
        }
    }
    LmResult {
        params: p,
        cost,
        iterations: it,
    }
}

/// Gaussian elimination with partial pivoting; `a` is row-major n×n.
fn solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let piv =
            (col..n).max_by(|&i, &j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs()))?;
        if a[piv * n + col].abs() < 1e-300 {
            return None;
        }
        if piv != col {
            for k in 0..n {
                a.swap(piv * n + k, col * n + k);
            }
            b.swap(piv, col);
        }
        for row in col + 1..n {
            let f = a[row * n + col] / a[col * n + col];
            for k in col..n {
                a[row * n + k] -= f * a[col * n + k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let s: f64 = (row + 1..n).map(|k| a[row * n + k] * x[k]).sum();
        x[row] = (b[row] - s) / a[row * n + row];
    }
    Some(x)
}

fn sigmoid(u: f64) -> f64 {
    1.0 / (1.0 + (-u).exp())
}

fn logit(x: f64) -> f64 {
    (x / (1.0 - x)).ln()
}

/// One fitted resonance.
#[derive(Debug, Clone)]
pub struct ResonanceFit {
    pub ring: Ring,
    pub through_offset_db: f64,
    pub drop_offset_db: f64,
    pub tilt_db_per_nm: f64,
    /// RMS residual on the through port within ±2 FWHM, dB.
    pub rms_through_db: f64,
    /// RMS residual on the drop port where it is within 20 dB of its peak
    /// and at least 3 dB above the detector noise ceiling, dB.
    pub rms_drop_db: f64,
    /// Drop peak above the detector noise ceiling, dB.
    pub drop_snr_db: f64,
}

/// Fit window on the drop port spans 20 dB below the peak, plus 3 dB margin.
pub const QUALIFIED_SNR_DB: f64 = 23.0;

impl ResonanceFit {
    /// The whole drop-port fit window sits above the noise ceiling, so the
    /// residual measures the model rather than the detector.
    pub fn qualified(&self) -> bool {
        self.drop_snr_db >= QUALIFIED_SNR_DB
    }

    pub fn worst_rms_db(&self) -> f64 {
        self.rms_through_db.max(self.rms_drop_db)
    }
}

/// Fitted device: every resonance in the grating-coupler passband.
pub struct DeviceFit {
    pub name: String,
    pub radius_um: f64,
    pub gap_nm: f64,
    pub n_g: f64,
    pub resonances: Vec<ResonanceFit>,
}

impl DeviceFit {
    /// The resonance closest to `lambda_nm` (device summaries use 1550 nm).
    pub fn nearest(&self, lambda_nm: f64) -> &ResonanceFit {
        self.resonances
            .iter()
            .min_by(|a, b| {
                (a.ring.lambda_res_nm - lambda_nm)
                    .abs()
                    .total_cmp(&(b.ring.lambda_res_nm - lambda_nm).abs())
            })
            .expect("device has resonances")
    }

    pub fn worst_rms_db(&self) -> f64 {
        self.resonances
            .iter()
            .map(|r| r.worst_rms_db())
            .fold(0.0, f64::max)
    }

    pub fn worst_qualified_rms_db(&self) -> f64 {
        self.resonances
            .iter()
            .filter(|r| r.qualified())
            .map(|r| r.worst_rms_db())
            .fold(0.0, f64::max)
    }
}

/// Group index from drop-peak positions: FSR in frequency is exact for
/// constant n_g, so n_g = c / (Δν · L) averaged over neighbouring peaks.
pub fn group_index(m: &Measurement, peaks: &[usize]) -> f64 {
    let l = 2.0 * PI * m.radius_um * 1e-6;
    let ng: Vec<f64> = peaks
        .windows(2)
        .map(|w| {
            let (a, b) = (m.wl_nm[w[0]] * 1e-9, m.wl_nm[w[1]] * 1e-9);
            C / ((C / a - C / b) * l)
        })
        .collect();
    ng.iter().sum::<f64>() / ng.len() as f64
}

pub fn fit_device(m: &Measurement) -> DeviceFit {
    // Peaks inside the grating-coupler passband; 4.2 is only a spacing guess.
    let peaks = m.resonance_peaks(4.2, 1520.0, 1578.0);
    let n_g = group_index(m, &peaks);
    let resonances = peaks
        .iter()
        .filter_map(|&p| fit_resonance(m, p, n_g))
        .collect();
    DeviceFit {
        name: m.name.clone(),
        radius_um: m.radius_um,
        gap_nm: m.gap_nm,
        n_g,
        resonances,
    }
}

/// Fit one resonance whose drop peak sits at sample `peak`.
pub fn fit_resonance(m: &Measurement, peak: usize, n_g: f64) -> Option<ResonanceFit> {
    let wl = &m.wl_nm;
    let lin = |db: f64| 10f64.powf(db / 10.0);
    let lam0 = wl[peak];
    let base = Ring {
        radius_um: m.radius_um,
        n_g,
        r1: 0.9,
        r2: 0.9,
        a: 0.99,
        lambda_res_nm: lam0,
    };
    let fsr = base.fsr_nm();

    // Initial width from the drop peak's half-power points.
    let pk = lin(m.drop_dbm[peak]);
    let (mut i, mut j) = (peak, peak);
    while i > 0 && lin(m.drop_dbm[i]) > pk / 2.0 {
        i -= 1;
    }
    while j + 1 < wl.len() && lin(m.drop_dbm[j]) > pk / 2.0 {
        j += 1;
    }
    let fwhm0 = (wl[j] - wl[i]).max(2.0 * m.step_nm());

    let half_win = (6.0 * fwhm0).min(0.45 * fsr);
    let idx: Vec<usize> = (0..wl.len())
        .filter(|&k| (wl[k] - lam0).abs() <= half_win)
        .collect();
    if idx.len() < 30 {
        return None;
    }

    // Initial r, a from finesse and through-port extinction.
    let x0 = rra_from_finesse(fsr / fwhm0).clamp(0.5, 0.9999);
    let edge: Vec<f64> = idx
        .iter()
        .filter(|&&k| (wl[k] - lam0).abs() > 4.0 * fwhm0)
        .map(|&k| m.through_dbm[k])
        .collect();
    let t_off = median(if edge.is_empty() {
        idx.iter().map(|&k| m.through_dbm[k]).collect()
    } else {
        edge
    });
    let t_min_db = idx
        .iter()
        .filter(|&&k| (wl[k] - lam0).abs() < fwhm0)
        .map(|&k| m.through_dbm[k])
        .fold(f64::MAX, f64::min);
    let t_min = lin(t_min_db - t_off).clamp(1e-4, 0.999);
    let a0 = loss_from_extinction(x0, t_min).clamp(x0 + 1e-6, 0.999_999);
    let r0 = (x0 / a0).sqrt().min(0.999_999);
    let d_off = m.drop_dbm[peak] - 10.0 * add_drop(r0, r0, a0, 0.0).1.log10();

    // Drop-port points within 20 dB of the peak, and clearly above the
    // detector's noise ceiling: below it the data is noise, not the ring.
    let ceiling = m.drop_noise_ceiling_dbm();
    let floor = (m.drop_dbm[peak] - 20.0).max(ceiling + 3.0);
    let drop_idx: Vec<usize> = idx
        .iter()
        .cloned()
        .filter(|&k| m.drop_dbm[k] > floor)
        .collect();

    let model = |p: &[f64]| {
        let r = sigmoid(p[1]);
        let a = sigmoid(p[2]);
        let ring = Ring {
            r1: r,
            r2: r,
            a,
            ..base
        };
        (ring, p[0], p[3], p[4], p[5])
    };
    let phase = |ring: &Ring, phi0: f64, lam: f64| ring.phase(lam) + phi0;
    let residuals = |p: &[f64], out: &mut Vec<f64>| {
        out.clear();
        let (ring, phi0, toff, doff, tilt) = model(p);
        for &k in &idx {
            let (t, _) = add_drop(ring.r1, ring.r2, ring.a, phase(&ring, phi0, wl[k]));
            out.push(toff + tilt * (wl[k] - lam0) + 10.0 * t.max(1e-12).log10() - m.through_dbm[k]);
        }
        for &k in &drop_idx {
            let (_, d) = add_drop(ring.r1, ring.r2, ring.a, phase(&ring, phi0, wl[k]));
            out.push(doff + tilt * (wl[k] - lam0) + 10.0 * d.max(1e-12).log10() - m.drop_dbm[k]);
        }
    };
    let p0 = [0.0, logit(r0), logit(a0), t_off, d_off, 0.0];
    let fit = levenberg_marquardt(residuals, &p0, 300);
    let (ring, phi0, toff, doff, tilt) = model(&fit.params);

    // Resonance where total phase is zero: φ(λ) + φ0 = 0.
    let l = ring.length_m();
    let lam_res = 1.0 / (1.0 / (lam0 * 1e-9) - phi0 / (2.0 * PI * n_g * l)) * 1e9;
    let ring = Ring {
        lambda_res_nm: lam_res,
        ..ring
    };

    let fwhm = ring.fwhm_nm();
    let rms = |pts: &mut dyn Iterator<Item = f64>| {
        let v: Vec<f64> = pts.collect();
        (v.iter().map(|x| x * x).sum::<f64>() / v.len().max(1) as f64).sqrt()
    };
    let rms_through_db = rms(&mut idx
        .iter()
        .filter(|&&k| (wl[k] - lam_res).abs() <= 2.0 * fwhm)
        .map(|&k| {
            let (t, _) = ring.through_drop(wl[k]);
            toff + tilt * (wl[k] - lam0) + 10.0 * t.log10() - m.through_dbm[k]
        }));
    let rms_drop_db = rms(&mut drop_idx.iter().map(|&k| {
        let (_, d) = ring.through_drop(wl[k]);
        doff + tilt * (wl[k] - lam0) + 10.0 * d.log10() - m.drop_dbm[k]
    }));
    Some(ResonanceFit {
        ring,
        through_offset_db: toff,
        drop_offset_db: doff,
        tilt_db_per_nm: tilt,
        rms_through_db,
        rms_drop_db,
        drop_snr_db: m.drop_dbm[peak] - ceiling,
    })
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
