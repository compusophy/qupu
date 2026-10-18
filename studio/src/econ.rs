//! First-principles economics: what a MAC/s costs to build, to run, and in
//! embodied energy — for today's accelerators and for photonic engines
//! built from the same physical accounting as `core`.
//!
//! Every optical matrix engine is reduced to one identity. Each symbol, an
//! engine takes `inputs` values, produces `outputs` values, and each output
//! sums `fan_in` products, so
//!
//!   MACs per symbol     = outputs · fan_in
//!   energy per symbol   = inputs·E_in + outputs·E_out + optical + static + memory
//!   ⇒ energy per MAC   ≈ E_in / fan_out + E_out / fan_in + photons + …
//!
//! Converters are paid once per value and amortised over fan-in and
//! fan-out. That single fact ranks architectures more than anything else.
//!
//! Incumbent figures are third-party estimates (Silicon Analysts, Epoch AI;
//! ±15–20 %) and datasheet dense peaks. Photonic figures are assumptions
//! with ranges; the ones that decide the answer are marked as such.

/// Shared economic assumptions.
#[derive(Debug, Clone, Copy)]
pub struct World {
    pub usd_per_kwh: f64,
    pub pue: f64,
    pub years: f64,
    /// Electronic-grade polysilicon, $/kg.
    pub si_usd_per_kg: f64,
    pub glass_usd_per_kg: f64,
}

impl Default for World {
    fn default() -> Self {
        World {
            usd_per_kwh: 0.08,
            pue: 1.3,
            years: 3.0,
            si_usd_per_kg: 50.0,
            glass_usd_per_kg: 10.0,
        }
    }
}

/// Grams of silicon consumed per mm² of die: a 775 µm wafer at 2.33 g/cm³.
const SI_G_PER_MM2: f64 = 0.0775 * 0.01 * 2.33;
/// Usable die area on a 300 mm wafer (gross, before yield), mm².
pub const WAFER_MM2: f64 = 60_000.0;

/// Processed-wafer price and manufacturing electricity per 300 mm wafer.
#[derive(Debug, Clone, Copy)]
pub struct Process {
    pub name: &'static str,
    pub usd_per_wafer: f64,
    pub kwh_per_wafer: f64,
}

impl Process {
    pub fn usd_per_mm2(&self) -> f64 {
        self.usd_per_wafer / WAFER_MM2
    }
    pub fn kwh_per_mm2(&self) -> f64 {
        self.kwh_per_wafer / WAFER_MM2
    }
}

// Wafer prices: Silicon Analysts 2026 averages (N5/N4 $18.5k, N7 $9.5k,
// 28 nm $3k); silicon photonics $3–5k is an unverified industry estimate.
// Electricity: 800–1 500 kWh per advanced-logic wafer, 200–400 legacy.
pub const N4: Process = Process {
    name: "N4/N5 logic",
    usd_per_wafer: 18_500.0,
    kwh_per_wafer: 1_200.0,
};
pub const N7: Process = Process {
    name: "N7 logic",
    usd_per_wafer: 9_500.0,
    kwh_per_wafer: 900.0,
};
pub const N28: Process = Process {
    name: "28 nm logic",
    usd_per_wafer: 3_000.0,
    kwh_per_wafer: 500.0,
};
pub const SIPH: Process = Process {
    name: "Si photonics",
    usd_per_wafer: 4_000.0,
    kwh_per_wafer: 300.0,
};
pub const DRAM: Process = Process {
    name: "DRAM (HBM)",
    usd_per_wafer: 8_000.0,
    kwh_per_wafer: 900.0,
};
pub const INTERPOSER: Process = Process {
    name: "Si interposer",
    usd_per_wafer: 2_000.0,
    kwh_per_wafer: 250.0,
};

#[derive(Debug, Clone)]
pub struct System {
    pub name: String,
    pub precision: &'static str,
    pub macs_per_s: f64,
    pub power_w: f64,
    /// Manufacturing cost, itemised.
    pub cost: Vec<(&'static str, f64)>,
    pub embodied_kwh: f64,
    pub raw_material_usd: f64,
    /// How much to trust the numbers.
    pub basis: &'static str,
}

impl System {
    pub fn cogs(&self) -> f64 {
        self.cost.iter().map(|(_, c)| c).sum()
    }
    pub fn pj_per_mac(&self) -> f64 {
        self.power_w / self.macs_per_s * 1e12
    }
    pub fn energy_usd(&self, w: &World) -> f64 {
        self.power_w / 1e3 * w.pue * 24.0 * 365.0 * w.years * w.usd_per_kwh
    }
    pub fn tco(&self, w: &World) -> f64 {
        self.cogs() + self.energy_usd(w)
    }
    /// Musk's "idiot index": finished cost over raw-material cost.
    pub fn idiot_index(&self) -> f64 {
        self.cogs() / self.raw_material_usd
    }
}

/// Silicon raw material for a set of die areas (mm²).
fn si_raw(w: &World, mm2: f64) -> f64 {
    mm2 * SI_G_PER_MM2 / 1000.0 * w.si_usd_per_kg
}

/// Today's accelerators. COGS: Silicon Analysts / Epoch AI estimates.
/// Throughput: dense (no sparsity) FP8 or FP4 peak; 1 MAC = 2 FLOP.
/// Areas: logic, HBM DRAM and interposer, mm² (estimates, for embodied
/// energy and raw material only).
pub fn incumbents(w: &World) -> Vec<System> {
    let gpu = |name: &str,
               precision,
               flops: f64,
               watts,
               cost: Vec<(&'static str, f64)>,
               logic: f64,
               dram: f64,
               interposer: f64| {
        let embodied = logic * N4.kwh_per_mm2()
            + dram * DRAM.kwh_per_mm2()
            + interposer * INTERPOSER.kwh_per_mm2();
        System {
            name: name.to_string(),
            precision,
            macs_per_s: flops / 2.0,
            power_w: watts,
            cost,
            embodied_kwh: embodied,
            raw_material_usd: si_raw(w, logic + dram + interposer) + 1.0, // + package metals, substrate
            basis: "third-party COGS estimate ±20 %; datasheet dense peak",
        }
    };
    vec![
        gpu(
            "NVIDIA H100 SXM",
            "FP8",
            1979e12,
            700.0,
            vec![
                ("logic die", 300.0),
                ("HBM3", 1350.0),
                ("CoWoS", 750.0),
                ("test, assembly", 920.0),
            ],
            814.0,
            4_000.0,
            2_500.0,
        ),
        gpu(
            "NVIDIA B200",
            "FP8",
            4.5e15,
            1000.0,
            vec![
                ("logic dies", 850.0),
                ("HBM3e", 2900.0),
                ("packaging, test", 2650.0),
            ],
            1_600.0,
            7_700.0,
            3_500.0,
        ),
        gpu(
            "NVIDIA B200",
            "FP4",
            9.0e15,
            1000.0,
            vec![
                ("logic dies", 850.0),
                ("HBM3e", 2900.0),
                ("packaging, test", 2650.0),
            ],
            1_600.0,
            7_700.0,
            3_500.0,
        ),
        gpu(
            "AMD MI300X",
            "FP8",
            2614.9e12,
            750.0,
            vec![("total BOM", 5300.0)],
            2_400.0,
            9_600.0,
            3_000.0,
        ),
        gpu(
            "Intel Gaudi 3",
            "FP8",
            1835e12,
            900.0,
            vec![
                ("logic", 1500.0),
                ("HBM2e", 1950.0),
                ("packaging", 1200.0),
                ("other", 1850.0),
            ],
            1_200.0,
            6_400.0,
            2_500.0,
        ),
    ]
}

/// One optical matrix engine, described by the quantities that set its
/// economics. Units: pJ, W, mm², $.
#[derive(Debug, Clone)]
pub struct Engine {
    pub name: &'static str,
    pub precision: &'static str,
    pub basis: &'static str,
    /// Values entering per symbol (each reused by `fan_out` outputs).
    pub inputs: f64,
    pub fan_out: f64,
    /// Values leaving per symbol (each summing `fan_in` products).
    pub outputs: f64,
    pub fan_in: f64,
    pub baud: f64,
    /// Engines per system (one package).
    pub units: f64,
    pub e_in_pj: f64,
    pub e_out_pj: f64,
    /// Full-scale optical power each output detector needs (receiver-limited).
    pub p_det_w: f64,
    pub optical_loss_db: f64,
    pub source_wpe: f64,
    pub source_usd_per_w: f64,
    /// Fixed cost of emitters per unit, when the source is an array of
    /// directly modulated emitters rather than a laser feeding modulators.
    pub emitter_usd: f64,
    pub static_w: f64,
    pub sram_pj_per_bit: f64,
    pub bits: f64,
    pub photonic: (Process, f64),
    pub cmos: (Process, f64),
    pub per_unit_usd: Vec<(&'static str, f64)>,
    pub per_system_usd: Vec<(&'static str, f64)>,
    /// Glass and optics mass per unit, g (raw material, embodied energy).
    pub glass_g: f64,
}

impl Engine {
    pub fn macs_per_symbol(&self) -> f64 {
        self.outputs * self.fan_in
    }

    pub fn optical_w(&self) -> f64 {
        self.outputs * self.p_det_w / 10f64.powf(-self.optical_loss_db / 10.0)
    }

    /// Electrical power per unit, by component.
    pub fn power_parts(&self) -> Vec<(&'static str, f64)> {
        vec![
            (
                "inputs (DAC, driver)",
                self.inputs * self.e_in_pj * 1e-12 * self.baud,
            ),
            (
                "outputs (TIA, ADC)",
                self.outputs * self.e_out_pj * 1e-12 * self.baud,
            ),
            ("light source", self.optical_w() / self.source_wpe),
            ("static (tuning, thermal)", self.static_w),
            (
                "SRAM traffic",
                (self.inputs + self.outputs) * self.bits * self.sram_pj_per_bit * 1e-12 * self.baud,
            ),
        ]
    }

    pub fn system(&self, w: &World) -> System {
        let u = self.units;
        let power: f64 = self.power_parts().iter().map(|(_, p)| p).sum::<f64>() * u;
        let mut cost: Vec<(&'static str, f64)> = vec![
            (
                "photonic silicon",
                self.photonic.1 * self.photonic.0.usd_per_mm2() * u,
            ),
            ("CMOS silicon", self.cmos.1 * self.cmos.0.usd_per_mm2() * u),
            (
                "light source",
                (self.optical_w() * self.source_usd_per_w + self.emitter_usd) * u,
            ),
        ];
        cost.extend(self.per_unit_usd.iter().map(|&(k, v)| (k, v * u)));
        cost.extend(self.per_system_usd.iter().cloned());
        let si_mm2 = (self.photonic.1 + self.cmos.1) * u;
        System {
            name: self.name.to_string(),
            precision: self.precision,
            macs_per_s: self.macs_per_symbol() * self.baud * u,
            power_w: power,
            cost,
            embodied_kwh: (self.photonic.1 * self.photonic.0.kwh_per_mm2() + self.cmos.1 * self.cmos.0.kwh_per_mm2()) * u
                + self.glass_g * u * 10e-3 // ≈10 kWh/kg to melt, form and finish glass
                + 1.0, // emitters, lasers, package
            raw_material_usd: si_raw(w, si_mm2)
                + self.glass_g * u / 1000.0 * w.glass_usd_per_kg
                + 1.0,
            basis: self.basis,
        }
    }

    /// Lower bound with ideal light and no static power: converters only.
    pub fn converter_floor_fj(&self) -> f64 {
        (self.e_in_pj / self.fan_out + self.e_out_pj / self.fan_in) * 1e3
    }
}

/// The three photonic engines compared in docs/photonic/FIRST_PRINCIPLES.md.
pub fn engines() -> Vec<Engine> {
    // Free-space matrix engine: an emitter array (display-like) shines
    // through a fixed glass weight mask onto a detector array (camera-like).
    // Fan-in is the whole input row: 1024.
    let glass = glass_engine();
    vec![
        // The plan's core (32 λ × 2 FSR orders × 64 rows), packaged as a
        // module the way silicon photonics is packaged today.
        Engine {
            name: "Ring core module",
            precision: "≈5.5 bit analog",
            basis: "plan assumptions; ring optics fitted; packaging per module",
            inputs: 64.0,
            fan_out: 64.0,
            outputs: 128.0,
            fan_in: 32.0,
            baud: 10e9,
            units: 1.0,
            e_in_pj: 1.5,
            e_out_pj: 2.5,
            p_det_w: 200e-6,
            optical_loss_db: 13.0,
            source_wpe: 0.10,
            source_usd_per_w: 100.0,
            emitter_usd: 0.0,
            static_w: 2048.0 * 1.6e-3,
            sram_pj_per_bit: 0.05,
            bits: 8.0,
            photonic: (SIPH, 8.0),
            cmos: (N7, 15.0),
            per_unit_usd: vec![
                ("3D integration", 10.0),
                ("fibre attach, package", 100.0),
                ("test, calibration", 15.0),
            ],
            per_system_usd: vec![],
            glass_g: 0.0,
        },
        // Same cores, but the wafer is the product: photonic and 28 nm CMOS
        // wafers hybrid-bonded like stacked image sensors, one light input
        // and one package per wafer, non-volatile (trimmed/PCM) weights.
        Engine {
            name: "Ring wafer",
            precision: "≈5.5 bit analog",
            basis: "speculative: bonded receivers, $20/W lasers, non-volatile weights",
            inputs: 64.0,
            fan_out: 64.0,
            outputs: 128.0,
            fan_in: 32.0,
            baud: 10e9,
            units: (WAFER_MM2 / 20.0).floor(), // 28 nm converters + SRAM set ≈20 mm² per core
            e_in_pj: 2.25,
            e_out_pj: 3.75,
            p_det_w: 100e-6,
            optical_loss_db: 13.0,
            source_wpe: 0.15,
            source_usd_per_w: 20.0,
            emitter_usd: 0.0,
            static_w: 0.05,
            sram_pj_per_bit: 0.08,
            bits: 8.0,
            photonic: (SIPH, 20.0),
            cmos: (N28, 20.0),
            per_unit_usd: vec![],
            per_system_usd: vec![
                ("wafer-to-wafer bonding", 1_000.0),
                ("wafer package, cold plate", 5_000.0),
                ("test, calibration", 2_000.0),
            ],
            glass_g: 0.0,
        },
        glass.clone(),
        // The same glass engine at a speed free-space optical matrix
        // processors have already been claimed to run (Lenslet EnLight,
        // 2003: 256 × 256, 125 MHz, 8 bit). Slow converters cost no less here.
        Engine {
            name: "Glass engine @ 125 MHz",
            precision: "≈5–8 bit analog",
            basis: "speed class claimed in 2003 (at 256²); 1024² is the step",
            baud: 125e6,
            p_det_w: receiver_power(125e6),
            ..glass.clone()
        },
    ]
}

/// Full-scale detector power for ≈6 bits: 200 µW at 10 GBd (PLAN.md §4.4),
/// scaled by √(rate) because receiver noise integrates over bandwidth.
fn receiver_power(baud: f64) -> f64 {
    200e-6 * (baud / 10e9).sqrt()
}

/// Free-space engine at 2.5 GBd: an array of directly modulated emitters,
/// a lithographic grayscale glass weight mask, a detector array with CMOS
/// readout. Signed weights use balanced detector pairs (counted in e_out).
fn glass_engine() -> Engine {
    Engine {
        name: "Glass engine (free space)",
        precision: "≈5 bit analog (unproven at speed)",
        basis: "speculative: 1024² at 2.5 GBd not yet demonstrated",
        inputs: 1024.0,
        fan_out: 1024.0,
        outputs: 1024.0,
        fan_in: 1024.0,
        baud: 2.5e9,
        units: 1.0,
        e_in_pj: 1.0,
        e_out_pj: 1.5,
        p_det_w: receiver_power(2.5e9),
        optical_loss_db: 10.0,
        source_wpe: 0.30,
        source_usd_per_w: 0.0,
        emitter_usd: 20.0,
        static_w: 1.0,
        sram_pj_per_bit: 0.08,
        bits: 8.0,
        photonic: (SIPH, 0.0),
        cmos: (N28, 60.0),
        per_unit_usd: vec![
            ("optics + weight mask", 20.0),
            ("assembly, alignment", 50.0),
            ("test, calibration", 10.0),
        ],
        per_system_usd: vec![],
        glass_g: 50.0,
    }
}

/// Swing one engine parameter to `low` and `high` and report the effect on
/// (TMAC/s per $ of COGS, fJ/MAC): the tornado that says what matters.
pub fn sensitivity(e: &Engine, w: &World) -> Vec<(&'static str, [(f64, f64); 2])> {
    type Knob = (&'static str, fn(&mut Engine, f64), f64, f64);
    let knobs: [Knob; 6] = [
        (
            "source $/W or emitter $",
            |e, k| {
                e.source_usd_per_w *= k;
                e.emitter_usd *= k
            },
            0.2,
            5.0,
        ),
        (
            "packaging + assembly $",
            |e, k| {
                for c in e.per_unit_usd.iter_mut().chain(e.per_system_usd.iter_mut()) {
                    c.1 *= k
                }
            },
            0.3,
            3.0,
        ),
        (
            "converter energy",
            |e, k| {
                e.e_in_pj *= k;
                e.e_out_pj *= k
            },
            0.33,
            2.0,
        ),
        (
            "detector power (receiver noise)",
            |e, k| e.p_det_w *= k,
            0.25,
            2.0,
        ),
        ("symbol rate", |e, k| e.baud *= k, 0.5, 2.0),
        ("static power", |e, k| e.static_w *= k, 0.1, 3.0),
    ];
    knobs
        .iter()
        .map(|&(name, f, lo, hi)| {
            let at = |k: f64| {
                let mut c = e.clone();
                f(&mut c, k);
                let s = c.system(w);
                (s.macs_per_s / s.cogs() / 1e12, s.pj_per_mac() * 1e3)
            };
            (name, [at(lo), at(hi)])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_module_matches_the_plan_energy_ledger() {
        // PLAN.md §6.1: ≈0.30 pJ/MAC nominal for a 41 TMAC/s core.
        let s = engines()[0].system(&World::default());
        assert!((s.macs_per_s - 40.96e12).abs() < 1e9);
        assert!((0.25..0.40).contains(&s.pj_per_mac()), "{}", s.pj_per_mac());
    }

    #[test]
    fn energy_per_mac_identity_holds() {
        // With no light, static or memory cost, energy per MAC is exactly
        // E_in / fan_out + E_out / fan_in.
        let mut e = engines()[2].clone();
        e.p_det_w = 0.0;
        e.static_w = 0.0;
        e.sram_pj_per_bit = 0.0;
        let s = e.system(&World::default());
        assert!((s.pj_per_mac() * 1e3 - e.converter_floor_fj()).abs() < 1e-9);
    }

    #[test]
    fn raw_material_is_a_rounding_error() {
        for s in incumbents(&World::default()) {
            assert!(s.idiot_index() > 500.0, "{} {}", s.name, s.idiot_index());
        }
    }
}
