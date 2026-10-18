//! Measured ring spectra from foundry test data.
//!
//! The bundled set is SiEPIC openEBL 2024-10 (Applied Nanotools e-beam,
//! 220 nm SOI): eight double-bus rings designed by L. Chrostowski with a
//! radius sweep and a gap sweep, measured TE at 25 C. In those files
//! `channel_2` is the through port and `channel_1` the drop port (dBm).

use crate::matfile;

pub struct Measurement {
    pub name: String,
    pub radius_um: f64,
    pub gap_nm: f64,
    pub wl_nm: Vec<f64>,
    pub through_dbm: Vec<f64>,
    pub drop_dbm: Vec<f64>,
}

/// The eight bundled devices, embedded so the browser build fits them too.
pub const OPENEBL_2024_10: [(&str, &[u8]); 8] = [
    (
        "r10g80",
        include_bytes!("../data/openEBL-2024-10/r10g80.mat"),
    ),
    (
        "r10g90",
        include_bytes!("../data/openEBL-2024-10/r10g90.mat"),
    ),
    (
        "r10g100",
        include_bytes!("../data/openEBL-2024-10/r10g100.mat"),
    ),
    (
        "r10g110",
        include_bytes!("../data/openEBL-2024-10/r10g110.mat"),
    ),
    ("r5g70", include_bytes!("../data/openEBL-2024-10/r5g70.mat")),
    ("r5g80", include_bytes!("../data/openEBL-2024-10/r5g80.mat")),
    ("r5g90", include_bytes!("../data/openEBL-2024-10/r5g90.mat")),
    ("r3g70", include_bytes!("../data/openEBL-2024-10/r3g70.mat")),
];

impl Measurement {
    /// Parse a SiEPIC `testResult` file. `name` encodes geometry: r<R>g<gap>.
    pub fn from_siepic_mat(name: &str, bytes: &[u8]) -> Result<Self, String> {
        let vars = matfile::parse(bytes)?;
        let tr = &vars
            .iter()
            .find(|(n, _)| n == "testResult")
            .ok_or("no testResult")?
            .1;
        let get = |p: &[&str]| -> Result<Vec<f64>, String> {
            Ok(tr
                .path(p)
                .and_then(|v| v.numbers())
                .ok_or(format!("missing {}", p.join(".")))?
                .to_vec())
        };
        let (radius_um, gap_nm) = parse_geometry(name).ok_or("name must look like r10g100")?;
        Ok(Self {
            name: name.to_string(),
            radius_um,
            gap_nm,
            wl_nm: get(&["header", "wavelength"])?,
            through_dbm: get(&["rows", "channel_2"])?,
            drop_dbm: get(&["rows", "channel_1"])?,
        })
    }

    pub fn bundled() -> Vec<Measurement> {
        OPENEBL_2024_10
            .iter()
            .map(|(n, b)| Measurement::from_siepic_mat(n, b).expect("bundled data parses"))
            .collect()
    }

    /// Drop-detector noise ceiling: 95th percentile of the drop port outside
    /// the grating-coupler passband (< 1510 nm), where it reads only noise.
    /// The percentile keeps the odd out-of-band resonance from skewing it.
    pub fn drop_noise_ceiling_dbm(&self) -> f64 {
        let mut v: Vec<f64> = self
            .wl_nm
            .iter()
            .zip(&self.drop_dbm)
            .filter(|(w, _)| **w < 1510.0)
            .map(|(_, d)| *d)
            .collect();
        if v.len() < 100 {
            v = self.drop_dbm.clone();
        }
        v.sort_by(f64::total_cmp);
        v[v.len() * 95 / 100]
    }

    pub fn step_nm(&self) -> f64 {
        (self.wl_nm[self.wl_nm.len() - 1] - self.wl_nm[0]) / (self.wl_nm.len() - 1) as f64
    }

    /// Indices of drop-port resonance peaks inside [lo, hi] nm: local maxima
    /// that stand `prominence_db` above everything within ±half the minimum
    /// separation (set from a group-index guess, so noise is not picked up).
    pub fn resonance_peaks(&self, n_g_guess: f64, lo: f64, hi: f64) -> Vec<usize> {
        let l = 2.0 * std::f64::consts::PI * self.radius_um * 1e-6;
        let fsr_nm = (1.55e-6f64).powi(2) / (n_g_guess * l) * 1e9;
        let half = ((0.3 * fsr_nm) / self.step_nm()) as usize;
        let y = &self.drop_dbm;
        let mut peaks = Vec::new();
        for i in half..y.len().saturating_sub(half) {
            if self.wl_nm[i] < lo || self.wl_nm[i] > hi {
                continue;
            }
            let win = &y[i - half..=i + half];
            let max = win.iter().cloned().fold(f64::MIN, f64::max);
            let min = win.iter().cloned().fold(f64::MAX, f64::min);
            if y[i] == max && max - min > 12.0 && peaks.last().is_none_or(|&p: &usize| i - p > half)
            {
                peaks.push(i);
            }
        }
        peaks
    }
}

fn parse_geometry(name: &str) -> Option<(f64, f64)> {
    let rest = name.strip_prefix('r')?;
    let (r, g) = rest.split_once('g')?;
    Some((r.parse().ok()?, g.parse().ok()?))
}
