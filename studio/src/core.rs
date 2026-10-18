//! Broadcast-and-weight ring core on a femtosecond clock.
//!
//! A comb of N wavelengths carries the input vector, one intensity per
//! channel per symbol. Every row is a bus of N add-drop rings, one per
//! channel; a heater sets each ring's detuning and therefore its balanced
//! weight. A balanced photodetector per row sums drop − through, so each
//! row produces one dot product per symbol.
//!
//! Ring optics (r, a, n_g, radius) come from the process model fitted to
//! measured silicon. Thermal, noise and energy figures are the plan's
//! assumptions (docs/photonic/PLAN.md, Appendix A) and are marked [A].
//!
//! Everything the view draws lives here: ring shifts, heater power, the
//! light on every bus, detector outputs, and the clock. The layout is part
//! of the physics too: path lengths set the propagation delays.

use crate::ring::{add_drop, Ring};

pub const FS_PER_S: f64 = 1e15;

/// Largest ring detuning used for weighting, as a fraction of channel spacing.
pub const DETUNE_MAX: f64 = 0.38;

/// Amplitude of the per-symbol random term in the synthetic workload.
pub const INPUT_NOISE: f64 = 0.06;

/// The synthetic workload, shared by the CPU core, the GPU kernel and its
/// CPU twin. Channel j at symbol s carries
///   0.5 + 0.30·sin(slow) + 0.12·sin(fast) + INPUT_NOISE·(2u − 1), clamped to [0, 1]:
/// a slow drift over tens of µs (readable at µs-per-second playback) and a
/// faster swing. Returns [slow phase, slow rate, fast phase, fast rate] at
/// symbol `s0`, rates per symbol, phases reduced mod 2π in f64 so the f32
/// GPU path only ever adds small offsets.
pub fn input_phases(baud: f64, j: usize, s0: u64) -> [f64; 4] {
    let tau = 2.0 * std::f64::consts::PI;
    let jf = j as f64;
    let slow_rate = tau / (baud * (25e-6 + 6e-6 * jf));
    let fast_rate = 0.00021 * (1.0 + 0.37 * jf);
    let slow = (s0 as f64 * slow_rate + 1.7 * jf).rem_euclid(tau);
    let fast = (s0 as f64 * fast_rate + 0.9 * jf).rem_euclid(tau);
    [slow, slow_rate, fast, fast_rate]
}

#[derive(Debug, Clone, Copy)]
pub struct CoreConfig {
    pub channels: usize,
    pub rows: usize,
    pub spacing_ghz: f64,
    pub center_nm: f64,
    /// Fitted ring template; its resonance is set per ring.
    pub ring: Ring,
    pub baud: f64,
    /// [A] heater thermal time constant.
    pub tau_th_s: f64,
    /// [A] heater efficiency, nm of red shift per mW.
    pub nm_per_mw: f64,
    /// [A] fraction of a heater's shift felt by each neighbouring ring.
    pub thermal_xtalk: f64,
    /// [A] intra-die resonance spread (1σ), nm.
    pub fab_sigma_nm: f64,
    /// [A] full-scale optical power per row at the detectors.
    pub p_det_w: f64,
    /// [A] TIA input-referred noise density, A/√Hz, over baud/2.
    pub tia_a_per_rthz: f64,
    pub adc_bits: u32,
    /// [A] energies: DAC + driver, ADC, TIA (pJ per sample).
    pub dac_pj: f64,
    pub adc_pj: f64,
    pub tia_pj: f64,
    /// [A] laser wall-plug efficiency and path loss excluding the split.
    pub laser_wpe: f64,
    pub path_loss_db: f64,
    pub seed: u64,
}

impl CoreConfig {
    pub fn new(ring: Ring, channels: usize, rows: usize) -> Self {
        CoreConfig {
            channels,
            rows,
            spacing_ghz: 100.0,
            center_nm: 1550.0,
            ring,
            baud: 10e9,
            tau_th_s: 5e-6,
            nm_per_mw: 0.25,
            thermal_xtalk: 0.08,
            fab_sigma_nm: 0.10,
            p_det_w: 200e-6,
            tia_a_per_rthz: 15e-12,
            adc_bits: 8,
            dac_pj: 1.5,
            adc_pj: 2.0,
            tia_pj: 0.5,
            laser_wpe: 0.10,
            path_loss_db: 13.0,
            seed: 0x5eed,
        }
    }

    pub fn symbol_fs(&self) -> f64 {
        FS_PER_S / self.baud
    }

    pub fn spacing_nm(&self) -> f64 {
        self.center_nm * self.center_nm * self.spacing_ghz * 1e9 / crate::ring::C * 1e-9
    }
}

/// Weight patterns the core cycles through. Each is a real operator, and
/// each looks different on the ring array once settled.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pattern {
    Identity,
    Cosine,
    Hadamard,
    Blur,
    Halo,
    Random,
}

impl Pattern {
    pub const ALL: [Pattern; 6] = [
        Pattern::Identity,
        Pattern::Cosine,
        Pattern::Hadamard,
        Pattern::Blur,
        Pattern::Halo,
        Pattern::Random,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Pattern::Identity => "identity",
            Pattern::Cosine => "cosine transform",
            Pattern::Hadamard => "Hadamard",
            Pattern::Blur => "Gaussian blur",
            Pattern::Halo => "halo",
            Pattern::Random => "random",
        }
    }

    /// Logical weight in [−1, 1] for row i, channel j.
    pub fn weight(self, i: usize, j: usize, rows: usize, n: usize, rng: &mut Rng) -> f64 {
        let (fi, fj) = (i as f64, j as f64);
        match self {
            Pattern::Identity => {
                if i % n == j {
                    1.0
                } else {
                    -1.0
                }
            }
            Pattern::Cosine => (std::f64::consts::PI * (fi + 0.5) * fj / n as f64).cos(),
            Pattern::Hadamard => {
                if (i & j).count_ones().is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                }
            }
            Pattern::Blur => {
                let d = fi * n as f64 / rows as f64 - fj;
                2.0 * (-d * d / 4.5).exp() - 1.0
            }
            Pattern::Halo => {
                let (ci, cj) = ((rows as f64 - 1.0) / 2.0, (n as f64 - 1.0) / 2.0);
                let r = ((fi - ci).powi(2) + (fj - cj).powi(2)).sqrt();
                let r0 = 0.32 * rows.min(n) as f64;
                2.0 * (-(r - r0).powi(2) / 2.0).exp() - 1.0
            }
            Pattern::Random => 2.0 * rng.uniform() - 1.0,
        }
    }
}

/// xorshift64* with Box–Muller; deterministic from the seed.
#[derive(Debug, Clone)]
pub struct Rng(u64, Option<f64>);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1), None)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn gauss(&mut self) -> f64 {
        if let Some(g) = self.1.take() {
            return g;
        }
        let u = self.uniform().max(1e-300);
        let v = 2.0 * std::f64::consts::PI * self.uniform();
        let rad = (-2.0 * u.ln()).sqrt();
        self.1 = Some(rad * v.sin());
        rad * v.cos()
    }
}

/// Chip layout in µm. Path lengths from the source set propagation delays.
#[derive(Debug, Clone)]
pub struct Layout {
    pub width: f64,
    pub height: f64,
    pub source: (f64, f64),
    pub bus_y: f64,
    pub mod_x: Vec<f64>,
    pub mod_r: f64,
    pub split_x0: f64,
    pub split_x1: f64,
    pub row_y: Vec<f64>,
    pub drop_y: Vec<f64>,
    pub ring_x: Vec<f64>,
    pub ring_r: f64,
    pub row_x0: f64,
    pub row_x1: f64,
    pub det_x: f64,
    /// Path length from the source to each row's bus start.
    pub row_start_s: Vec<f64>,
}

impl Layout {
    pub fn new(cfg: &CoreConfig) -> Self {
        let r = cfg.ring.radius_um;
        let mod_r = 3.0;
        let mod_pitch = 2.0 * mod_r + 8.0;
        let ring_pitch = 2.0 * r + 14.0;
        let row_pitch = 4.0 * r + 22.0;
        let margin = 30.0;
        let height = 2.0 * margin + row_pitch * cfg.rows as f64;
        let bus_y = height / 2.0;
        let source = (10.0, bus_y);
        let mod_x: Vec<f64> = (0..cfg.channels)
            .map(|j| 50.0 + j as f64 * mod_pitch)
            .collect();
        let split_x0 = mod_x.last().unwrap() + 30.0;
        let split_x1 = split_x0 + 40.0 + 25.0 * (cfg.rows as f64).log2().max(1.0);
        let row_y: Vec<f64> = (0..cfg.rows)
            .map(|i| margin + row_pitch * (i as f64 + 0.75))
            .collect();
        let drop_y: Vec<f64> = row_y.iter().map(|y| y - 2.0 * r - 3.0).collect();
        let row_x0 = split_x1;
        let ring_x: Vec<f64> = (0..cfg.channels)
            .map(|j| row_x0 + 24.0 + j as f64 * ring_pitch)
            .collect();
        let row_x1 = ring_x.last().unwrap() + 24.0;
        let det_x = row_x1 + 12.0;
        let width = det_x + 90.0;
        let row_start_s = row_y
            .iter()
            .map(|y| (split_x0 - source.0) + (split_x1 - split_x0) + (y - bus_y).abs() * 1.2)
            .collect();
        Layout {
            width,
            height,
            source,
            bus_y,
            mod_x,
            mod_r,
            split_x0,
            split_x1,
            row_y,
            drop_y,
            ring_x,
            ring_r: r,
            row_x0,
            row_x1,
            det_x,
            row_start_s,
        }
    }
}

/// Running totals for the ledgers.
#[derive(Debug, Clone, Default)]
pub struct Ledger {
    pub symbols: u64,
    pub macs: f64,
    pub dac_j: f64,
    pub adc_j: f64,
    pub tia_j: f64,
    pub laser_j: f64,
    pub heater_j: f64,
    pub err_sq: f64,
    pub err_n: f64,
}

impl Ledger {
    pub fn total_j(&self) -> f64 {
        self.dac_j + self.adc_j + self.tia_j + self.laser_j + self.heater_j
    }

    pub fn pj_per_mac(&self) -> f64 {
        self.total_j() / self.macs.max(1.0) * 1e12
    }

    /// Effective bits of the corrected output against the ideal dot product,
    /// over a full scale of ±1.
    pub fn enob(&self) -> f64 {
        let rms = (self.err_sq / self.err_n.max(1.0)).sqrt().max(1e-12);
        (2.0 / (rms * 12f64.sqrt())).log2()
    }
}

pub struct Core {
    pub cfg: CoreConfig,
    pub layout: Layout,
    pub channel_nm: Vec<f64>,
    /// Fabricated (unheated) resonance of each ring, rows × channels.
    pub native_nm: Vec<f64>,
    /// Heater-driven red shift now and its target, nm.
    pub shift_nm: Vec<f64>,
    pub target_shift_nm: Vec<f64>,
    /// Logical weights in [−1, 1] and the physical weights they map to.
    pub logical_w: Vec<f64>,
    /// Effective balanced response of row i to channel j, all rings included.
    pub w_eff: Vec<f64>,
    /// Power reaching each ring's input on its own channel, and dropped by it
    /// (fractions of the channel's row power, before the input value).
    pub ring_in: Vec<f64>,
    pub ring_drop: Vec<f64>,
    pub pattern: Pattern,
    pub calibrate: bool,
    /// Physical weight range reachable within the detuning limit.
    pub w_lo: f64,
    pub w_hi: f64,
    curve: Vec<(f64, f64)>,
    pub x: Vec<f64>,
    /// Mean input over the symbols computed in the last step: what a camera
    /// with that exposure would see.
    pub x_mean: Vec<f64>,
    pub y: Vec<f64>,
    pub y_ideal: Vec<f64>,
    /// Last few symbols (emission time in fs, inputs) for wavefront display.
    pub history: Vec<(u64, Vec<f64>)>,
    pub t_fs: u64,
    pub ledger: Ledger,
    pub last_program_fs: u64,
    rng: Rng,
    symbol_carry: f64,
    /// Heater state the optics were last computed for.
    optics_shift: Vec<f64>,
}

impl Core {
    pub fn new(cfg: CoreConfig) -> Self {
        let n = cfg.channels;
        let m = cfg.rows;
        let spacing = cfg.spacing_nm();
        let channel_nm: Vec<f64> = (0..n)
            .map(|j| cfg.center_nm + (j as f64 - (n as f64 - 1.0) / 2.0) * spacing)
            .collect();
        let mut rng = Rng::new(cfg.seed);

        // Balanced weight vs detuning for the fitted ring. Detuning stops at
        // 0.38 of the spacing: further out, a ring's tail starts dropping its
        // neighbour's channel faster than it gains range.
        let d_max = DETUNE_MAX * spacing;
        let curve: Vec<(f64, f64)> = (0..=512)
            .map(|k| {
                let d = d_max * k as f64 / 512.0;
                let (t, dr) = add_drop(
                    cfg.ring.r1,
                    cfg.ring.r2,
                    cfg.ring.a,
                    phase_of(&cfg.ring, cfg.center_nm, d),
                );
                (d, dr - t)
            })
            .collect();
        let (w_hi, w_lo) = (curve[0].1, curve[512].1);

        // Rings are drawn blue of their channel by the full detuning range plus
        // 3σ of fab spread, so heaters (red shift only) can always reach.
        let park = d_max + 3.0 * cfg.fab_sigma_nm;
        let native_nm = (0..m * n)
            .map(|k| channel_nm[k % n] - park + cfg.fab_sigma_nm * rng.gauss())
            .collect();

        let mut core = Core {
            layout: Layout::new(&cfg),
            channel_nm,
            native_nm,
            shift_nm: vec![0.0; m * n],
            target_shift_nm: vec![0.0; m * n],
            logical_w: vec![0.0; m * n],
            w_eff: vec![0.0; m * n],
            ring_in: vec![0.0; m * n],
            ring_drop: vec![0.0; m * n],
            pattern: Pattern::Halo,
            calibrate: true,
            w_lo,
            w_hi,
            curve,
            x: vec![0.0; n],
            x_mean: vec![0.0; n],
            y: vec![0.0; m],
            y_ideal: vec![0.0; m],
            history: Vec::new(),
            t_fs: 0,
            ledger: Ledger::default(),
            last_program_fs: 0,
            rng,
            symbol_carry: 0.0,
            optics_shift: Vec::new(),
            cfg,
        };
        core.program(Pattern::Halo);
        core.recompute_optics();
        core
    }

    pub fn n(&self) -> usize {
        self.cfg.channels
    }

    /// Logical ±1 maps to 75 % of the reachable physical range. The other
    /// quarter is headroom for the controller to cancel neighbours' tails:
    /// with less, patterns full of ±1 weights (identity, Hadamard) cannot
    /// be calibrated below a few % error.
    pub fn weight_scale(&self) -> (f64, f64) {
        (
            0.5 * (self.w_hi + self.w_lo),
            0.375 * (self.w_hi - self.w_lo),
        )
    }

    /// Physical weight for a logical one.
    fn physical(&self, w: f64) -> f64 {
        let (mid, half) = self.weight_scale();
        mid + half * w.clamp(-1.0, 1.0)
    }

    /// Detuning (nm, ring blue of channel) that gives physical weight `w`.
    fn detuning_for(&self, w: f64) -> f64 {
        let w = w.clamp(self.w_lo, self.w_hi);
        let k = self
            .curve
            .partition_point(|&(_, cw)| cw > w)
            .clamp(1, self.curve.len() - 1);
        let (d0, w0) = self.curve[k - 1];
        let (d1, w1) = self.curve[k];
        d0 + (d1 - d0) * (w0 - w) / (w0 - w1).max(1e-15)
    }

    /// Load a new weight matrix: set heater targets. The rings then slide
    /// there with the thermal time constant; nothing is instantaneous.
    pub fn program(&mut self, pattern: Pattern) {
        let (m, n) = (self.cfg.rows, self.cfg.channels);
        self.pattern = pattern;
        for i in 0..m {
            for j in 0..n {
                self.logical_w[i * n + j] = pattern.weight(i, j, m, n, &mut self.rng);
            }
        }
        let mut wanted: Vec<f64> = self.logical_w.iter().map(|&w| self.physical(w)).collect();
        self.set_targets(&wanted);
        if self.calibrate {
            // Crosstalk-aware control: solve on the steady state, including
            // neighbouring rings' tails and heater crosstalk, and pre-correct.
            let goal: Vec<f64> = wanted.clone();
            for _ in 0..6 {
                let settled = self.response(&self.effective_shift(&self.target_shift_nm));
                for k in 0..m * n {
                    wanted[k] += 0.8 * (goal[k] - settled.0[k]);
                }
                self.set_targets(&wanted);
            }
        }
        self.last_program_fs = self.t_fs;
    }

    fn set_targets(&mut self, physical: &[f64]) {
        let n = self.cfg.channels;
        let raw: Vec<f64> = (0..physical.len())
            .map(|k| (self.channel_nm[k % n] - self.detuning_for(physical[k])) - self.native_nm[k])
            .collect();
        // Calibrated: undo heater crosstalk, own + c·(neighbours) = raw,
        // by Jacobi sweeps along each row. Uncalibrated: naive per-ring.
        let c = self.cfg.thermal_xtalk;
        let mut own: Vec<f64> = raw.iter().map(|r| r.max(0.0)).collect();
        for _ in 0..if self.calibrate { 20 } else { 0 } {
            let prev = own.clone();
            for k in 0..own.len() {
                let j = k % n;
                let left = if j > 0 { prev[k - 1] } else { 0.0 };
                let right = if j + 1 < n { prev[k + 1] } else { 0.0 };
                own[k] = (raw[k] - c * (left + right)).max(0.0);
            }
        }
        self.target_shift_nm = own;
    }

    /// Shift each ring actually sees: its own heater plus neighbours' leak.
    fn effective_shift(&self, own: &[f64]) -> Vec<f64> {
        let n = self.cfg.channels;
        let c = self.cfg.thermal_xtalk;
        (0..own.len())
            .map(|k| {
                let j = k % n;
                let left = if j > 0 { own[k - 1] } else { 0.0 };
                let right = if j + 1 < n { own[k + 1] } else { 0.0 };
                own[k] + c * (left + right)
            })
            .collect()
    }

    /// Full optical response for given effective shifts: (w_eff, ring_in, ring_drop).
    fn response(&self, eff_shift: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (m, n) = (self.cfg.rows, self.cfg.channels);
        let tpl = self.cfg.ring;
        let mut w = vec![0.0; m * n];
        let mut ring_in = vec![0.0; m * n];
        let mut ring_drop = vec![0.0; m * n];
        let mut td = vec![(0.0, 0.0); n];
        for i in 0..m {
            for j in 0..n {
                // Channel j along row i: through-bus power falls ring by ring;
                // light dropped at ring k returns on the drop bus past rings < k.
                for (k, slot) in td.iter_mut().enumerate() {
                    let res = self.native_nm[i * n + k] + eff_shift[i * n + k];
                    let ring = Ring {
                        lambda_res_nm: res,
                        ..tpl
                    };
                    *slot = ring.through_drop(self.channel_nm[j]);
                }
                // The return path passes the same rings < k, and a symmetric
                // ring transmits the same both ways, so it costs p again.
                let mut p = 1.0;
                let mut dropped = 0.0;
                for (k, &(t, d)) in td.iter().enumerate() {
                    if k == j {
                        ring_in[i * n + j] = p;
                        ring_drop[i * n + j] = p * d;
                    }
                    dropped += p * d * p;
                    p *= t;
                }
                w[i * n + j] = dropped - p;
            }
        }
        (w, ring_in, ring_drop)
    }

    pub fn recompute_optics(&mut self) {
        self.optics_shift = self.shift_nm.clone();
        let eff = self.effective_shift(&self.shift_nm);
        let (w, ri, rd) = self.response(&eff);
        self.w_eff = w;
        self.ring_in = ri;
        self.ring_drop = rd;
    }

    /// Fraction of heaters whose thermal transient is done: within 1 % of the
    /// detuning range of their target shift.
    pub fn settled(&self) -> f64 {
        let tol = 0.01 * DETUNE_MAX * self.cfg.spacing_nm();
        let ok = self
            .shift_nm
            .iter()
            .zip(&self.target_shift_nm)
            .filter(|(s, t)| (*s - *t).abs() < tol)
            .count();
        ok as f64 / self.shift_nm.len() as f64
    }

    /// RMS weight error against the programmed matrix, as a fraction of the
    /// logical range (crosstalk, fab spread and thermal state all included).
    pub fn weight_error(&self) -> f64 {
        let (_, half) = self.weight_scale();
        let n = self.logical_w.len() as f64;
        let s: f64 = (0..self.logical_w.len())
            .map(|k| ((self.w_eff[k] - self.physical(self.logical_w[k])) / (2.0 * half)).powi(2))
            .sum();
        (s / n).sqrt()
    }

    /// Receiver noise (TIA thermal + shot at full-scale photocurrent, over
    /// baud/2) as a fraction of full scale.
    pub fn receiver_sigma(&self) -> f64 {
        let cfg = &self.cfg;
        let i_fs = cfg.p_det_w; // responsivity 1 A/W
        let b = cfg.baud / 2.0;
        (cfg.tia_a_per_rthz.powi(2) * b + 2.0 * 1.602e-19 * i_fs * b).sqrt() / i_fs
    }

    pub fn heater_mw(&self) -> f64 {
        self.shift_nm.iter().sum::<f64>() / self.cfg.nm_per_mw
    }

    pub fn ring_heater_mw(&self, k: usize) -> f64 {
        self.shift_nm[k] / self.cfg.nm_per_mw
    }

    /// Resonance a ring sits at right now, nm.
    pub fn ring_res_nm(&self, k: usize) -> f64 {
        let eff = self.shift_nm[k]
            + self.cfg.thermal_xtalk
                * ((if !k.is_multiple_of(self.n()) {
                    self.shift_nm[k - 1]
                } else {
                    0.0
                }) + (if k % self.n() + 1 < self.n() {
                    self.shift_nm[k + 1]
                } else {
                    0.0
                }));
        self.native_nm[k] + eff
    }

    /// Input vector for symbol `s` (see `input_phases`).
    fn input(&mut self, s: u64) -> Vec<f64> {
        (0..self.cfg.channels)
            .map(|j| {
                let p = input_phases(self.cfg.baud, j, s);
                let u = self.rng.uniform();
                (0.5 + 0.30 * p[0].sin() + 0.12 * p[2].sin() + INPUT_NOISE * (2.0 * u - 1.0))
                    .clamp(0.0, 1.0)
            })
            .collect()
    }

    /// Advance simulated time by `dt_fs`. Thermal state relaxes exactly;
    /// every symbol in the interval is counted in the ledgers; up to
    /// `max_symbols` of them are computed in full (noise, ADC, error).
    pub fn step(&mut self, dt_fs: u64, max_symbols: usize) {
        let cfg = self.cfg;
        let dt_s = dt_fs as f64 / FS_PER_S;
        let k = 1.0 - (-dt_s / cfg.tau_th_s).exp();
        let mut heater_avg_mw = 0.0;
        for (s, t) in self.shift_nm.iter_mut().zip(&self.target_shift_nm) {
            let before = *s;
            *s += (t - *s) * k;
            heater_avg_mw += 0.5 * (before + *s) / cfg.nm_per_mw;
        }
        // Rings only need re-solving once some heater has moved by more than
        // 0.05 pm (≈ 0.7 mK); a settled core streams symbols without it.
        let moved = self.optics_shift.len() != self.shift_nm.len()
            || self
                .shift_nm
                .iter()
                .zip(&self.optics_shift)
                .any(|(a, b)| (a - b).abs() > 5e-5);
        if moved {
            self.recompute_optics();
        }

        let t0 = self.t_fs;
        self.t_fs += dt_fs;
        self.symbol_carry += dt_fs as f64 / cfg.symbol_fs();
        let n_sym = self.symbol_carry.floor() as u64;
        self.symbol_carry -= n_sym as f64;

        let (m, n) = (cfg.rows, cfg.channels);
        let led = &mut self.ledger;
        led.symbols += n_sym;
        led.macs += (n_sym as f64) * (m * n) as f64;
        led.dac_j += n_sym as f64 * n as f64 * cfg.dac_pj * 1e-12;
        led.adc_j += n_sym as f64 * m as f64 * cfg.adc_pj * 1e-12;
        led.tia_j += n_sym as f64 * m as f64 * cfg.tia_pj * 1e-12;
        let laser_w = m as f64 * cfg.p_det_w / 10f64.powf(-cfg.path_loss_db / 10.0) / cfg.laser_wpe;
        led.laser_j += laser_w * dt_s;
        led.heater_j += heater_avg_mw * 1e-3 * dt_s;

        // Compute the most recent symbols in full.
        let full = (n_sym as usize).min(max_symbols);
        let first = self.ledger.symbols - full as u64;
        let sigma = self.receiver_sigma();
        let levels = (1u64 << cfg.adc_bits) as f64;
        let (mid, half) = self.weight_scale();
        let mut x_sum = vec![0.0; n];
        for s in first..self.ledger.symbols {
            let x = self.input(s);
            for (a, b) in x_sum.iter_mut().zip(&x) {
                *a += b;
            }
            let sum_x: f64 = x.iter().sum();
            for i in 0..m {
                let row = i * n..(i + 1) * n;
                let acc: f64 = self.w_eff[row.clone()]
                    .iter()
                    .zip(&x)
                    .map(|(w, x)| w * x)
                    .sum();
                let ideal: f64 = self.logical_w[row].iter().zip(&x).map(|(w, x)| w * x).sum();
                // Photocurrent as a fraction of full scale, plus receiver noise,
                // quantised by the ADC over ±1, then digitally corrected for the
                // weight mapping's offset and scale (the controller knows Σx).
                let meas = acc / n as f64 + sigma * self.rng.gauss();
                let q = ((meas.clamp(-1.0, 1.0) + 1.0) / 2.0 * (levels - 1.0)).round()
                    / (levels - 1.0)
                    * 2.0
                    - 1.0;
                let corrected = (q - mid * sum_x / n as f64) / half;
                let target = ideal / n as f64;
                self.y[i] = corrected;
                self.y_ideal[i] = target;
                self.ledger.err_sq += (corrected - target).powi(2);
                self.ledger.err_n += 1.0;
            }
            if self.ledger.symbols - s <= 8 {
                let emitted =
                    t0 + ((s - (self.ledger.symbols - n_sym)) as f64 * cfg.symbol_fs()) as u64;
                if self.history.len() >= 8 {
                    self.history.remove(0);
                }
                self.history.push((emitted, x.clone()));
            }
            self.x = x;
        }
        if full > 0 {
            self.x_mean = x_sum.iter().map(|v| v / full as f64).collect();
        }
    }

    /// Group velocity, µm per fs.
    pub fn v_um_per_fs(&self) -> f64 {
        crate::ring::C / self.cfg.ring.n_g * 1e6 / FS_PER_S
    }

    /// Input vector present at path length `s_um` from the source right now:
    /// the symbol whose wavefront has reached that point.
    pub fn x_at(&self, s_um: f64) -> &[f64] {
        let delay = (s_um / self.v_um_per_fs()) as u64;
        let t = self.t_fs.saturating_sub(delay);
        self.history
            .iter()
            .rev()
            .find(|(e, _)| *e <= t)
            .map(|(_, x)| x.as_slice())
            .unwrap_or(&self.x)
    }

    pub fn reset_ledger(&mut self) {
        self.ledger = Ledger::default();
    }
}

fn phase_of(ring: &Ring, center_nm: f64, detuning_nm: f64) -> f64 {
    Ring {
        lambda_res_nm: center_nm - detuning_nm,
        ..*ring
    }
    .phase(center_nm)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring() -> Ring {
        Ring {
            radius_um: 5.0,
            n_g: 4.185,
            r1: 0.977,
            r2: 0.977,
            a: 0.9995,
            lambda_res_nm: 1550.0,
        }
    }

    #[test]
    fn weight_curve_spans_both_signs() {
        let core = Core::new(CoreConfig::new(ring(), 8, 8));
        assert!(
            core.w_hi > 0.5 && core.w_lo < -0.5,
            "{} .. {}",
            core.w_lo,
            core.w_hi
        );
    }

    #[test]
    fn rings_settle_to_programmed_weights() {
        let mut core = Core::new(CoreConfig::new(ring(), 8, 8));
        core.step(200_000_000, 0); // 200 ns: nowhere near settled
        let early = core.settled();
        for _ in 0..100 {
            core.step(1_000_000_000, 0); // 100 µs total
        }
        assert!(
            early < 0.5 && core.settled() > 0.95,
            "early {early}, late {}",
            core.settled()
        );
    }

    #[test]
    fn calibrated_core_computes_dot_products() {
        let mut core = Core::new(CoreConfig::new(ring(), 8, 8));
        core.step(100_000_000_000, 0);
        core.reset_ledger();
        core.step(100_000_000, 1000); // 1000 symbols
        assert!(core.ledger.enob() > 4.0, "ENOB {}", core.ledger.enob());
        assert_eq!(core.ledger.symbols, 1000);
    }

    #[test]
    fn clock_is_integer_femtoseconds_and_drift_free() {
        let mut core = Core::new(CoreConfig::new(ring(), 4, 4));
        for _ in 0..1000 {
            core.step(33_333, 0); // a third of a symbol each time
        }
        assert_eq!(core.t_fs, 33_333_000);
        assert_eq!(core.ledger.symbols, 333);
    }
}
