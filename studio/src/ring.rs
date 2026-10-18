//! Add-drop microring resonator, steady-state transfer-matrix model.
//!
//! Power transmissions follow Bogaerts et al., "Silicon microring
//! resonators" (Laser Photonics Rev. 2012). The round-trip phase uses a
//! constant group index around the resonance, written in frequency so it
//! stays periodic across free spectral ranges:
//!
//!   φ(λ) = 2π · n_g · L · (1/λ − 1/λ_res)
//!
//! which is exact for linear dispersion in frequency (no GVD).

use std::f64::consts::PI;

pub const C: f64 = 299_792_458.0;

/// Through and drop power for self-coupling coefficients `r1`, `r2`,
/// single-pass amplitude transmission `a` and round-trip phase `phi`.
#[inline]
pub fn add_drop(r1: f64, r2: f64, a: f64, phi: f64) -> (f64, f64) {
    let c = phi.cos();
    let ra = r1 * r2 * a;
    let den = 1.0 - 2.0 * ra * c + ra * ra;
    let through = (r2 * r2 * a * a - 2.0 * ra * c + r1 * r1) / den;
    let drop = (1.0 - r1 * r1) * (1.0 - r2 * r2) * a / den;
    (through, drop)
}

/// One ring. Symmetric coupling (same gap to both buses) unless `r2` differs.
#[derive(Debug, Clone, Copy)]
pub struct Ring {
    pub radius_um: f64,
    pub n_g: f64,
    /// Self-coupling (field) at the input bus.
    pub r1: f64,
    /// Self-coupling (field) at the drop bus.
    pub r2: f64,
    /// Single-pass amplitude transmission (propagation + bend loss).
    pub a: f64,
    /// A resonance wavelength, nm. Others follow at one FSR spacing.
    pub lambda_res_nm: f64,
}

impl Ring {
    pub fn length_m(&self) -> f64 {
        2.0 * PI * self.radius_um * 1e-6
    }

    pub fn phase(&self, lambda_nm: f64) -> f64 {
        2.0 * PI
            * self.n_g
            * self.length_m()
            * (1.0 / (lambda_nm * 1e-9) - 1.0 / (self.lambda_res_nm * 1e-9))
    }

    pub fn through_drop(&self, lambda_nm: f64) -> (f64, f64) {
        add_drop(self.r1, self.r2, self.a, self.phase(lambda_nm))
    }

    /// Balanced weight as seen by a balanced photodetector: drop − through.
    pub fn weight(&self, lambda_nm: f64) -> f64 {
        let (t, d) = self.through_drop(lambda_nm);
        d - t
    }

    pub fn fsr_hz(&self) -> f64 {
        C / (self.n_g * self.length_m())
    }

    pub fn fsr_nm(&self) -> f64 {
        let l = self.lambda_res_nm * 1e-9;
        l * l * self.fsr_hz() / C * 1e9
    }

    pub fn finesse(&self) -> f64 {
        let x = self.r1 * self.r2 * self.a;
        PI * x.sqrt() / (1.0 - x)
    }

    pub fn fwhm_nm(&self) -> f64 {
        self.fsr_nm() / self.finesse()
    }

    pub fn fwhm_ghz(&self) -> f64 {
        self.fsr_hz() / self.finesse() / 1e9
    }

    pub fn q_loaded(&self) -> f64 {
        self.lambda_res_nm / self.fwhm_nm()
    }

    /// Intrinsic Q from propagation loss alone (no coupling).
    pub fn q_intrinsic(&self) -> f64 {
        let alpha = self.alpha_per_m();
        2.0 * PI * self.n_g / (self.lambda_res_nm * 1e-9 * alpha)
    }

    /// Power attenuation coefficient, 1/m.
    pub fn alpha_per_m(&self) -> f64 {
        -2.0 * self.a.ln() / self.length_m()
    }

    pub fn loss_db_per_cm(&self) -> f64 {
        -20.0 * self.a.log10() / (self.length_m() * 100.0)
    }

    /// Power coupling to the input bus, κ² = 1 − r1².
    pub fn kappa2(&self) -> f64 {
        1.0 - self.r1 * self.r1
    }

    /// Through-port extinction at resonance, dB (positive).
    pub fn extinction_db(&self) -> f64 {
        let (t, _) = add_drop(self.r1, self.r2, self.a, 0.0);
        -10.0 * t.log10()
    }

    pub fn drop_peak(&self) -> f64 {
        add_drop(self.r1, self.r2, self.a, 0.0).1
    }
}

/// Solve π√x/(1−x) = F for x = r1·r2·a.
pub fn rra_from_finesse(f: f64) -> f64 {
    let s = (-PI + (PI * PI + 4.0 * f * f).sqrt()) / (2.0 * f);
    s * s
}

/// Symmetric ring: given x = r²·a and the through-port minimum (linear),
/// recover a by bisection on T_min(a) = (x/a)(1−a)²/(1−x)², a ∈ (x, 1).
pub fn loss_from_extinction(x: f64, t_min: f64) -> f64 {
    let t_of = |a: f64| (x / a) * (1.0 - a).powi(2) / (1.0 - x).powi(2);
    let (mut lo, mut hi) = (x, 1.0);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if t_of(mid) > t_min {
            lo = mid
        } else {
            hi = mid
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring() -> Ring {
        Ring {
            radius_um: 10.0,
            n_g: 4.2,
            r1: 0.98,
            r2: 0.98,
            a: 0.998,
            lambda_res_nm: 1550.0,
        }
    }

    #[test]
    fn energy_is_conserved_when_lossless() {
        let r = Ring { a: 1.0, ..ring() };
        for i in 0..200 {
            let (t, d) = r.through_drop(1545.0 + i as f64 * 0.05);
            assert!((t + d - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn lossy_ring_never_creates_power() {
        let r = ring();
        for i in 0..200 {
            let (t, d) = r.through_drop(1545.0 + i as f64 * 0.05);
            assert!(t + d <= 1.0 + 1e-12 && t >= 0.0 && d >= 0.0);
        }
    }

    #[test]
    fn resonances_are_one_fsr_apart() {
        let r = ring();
        let next = 1.0 / (1.0 / (r.lambda_res_nm * 1e-9) - r.fsr_hz() / C) * 1e9;
        assert!(
            r.phase(next)
                .rem_euclid(2.0 * PI)
                .min(2.0 * PI - r.phase(next).rem_euclid(2.0 * PI))
                < 1e-9
        );
        assert!((next - r.lambda_res_nm - r.fsr_nm()).abs() / r.fsr_nm() < 0.01);
    }

    #[test]
    fn critical_coupling_gives_a_through_null() {
        // All-pass limit (r2 = 1): critical when r1 = a.
        let r = Ring {
            r1: 0.995,
            r2: 1.0,
            a: 0.995,
            ..ring()
        };
        assert!(r.through_drop(r.lambda_res_nm).0 < 1e-20);
    }

    #[test]
    fn fwhm_matches_half_power_points() {
        let r = ring();
        let half = r.drop_peak() / 2.0;
        let w = r.fwhm_nm() / 2.0;
        let (_, d) = r.through_drop(r.lambda_res_nm + w);
        assert!((d - half).abs() / half < 0.01, "{d} vs {half}");
    }

    #[test]
    fn inverse_helpers_round_trip() {
        let r = ring();
        let x = r.r1 * r.r2 * r.a;
        assert!((rra_from_finesse(r.finesse()) - x).abs() < 1e-12);
        let (t_min, _) = add_drop(r.r1, r.r2, r.a, 0.0);
        assert!((loss_from_extinction(x, t_min) - r.a).abs() < 1e-9);
    }
}
