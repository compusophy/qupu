//! `studio` CLI.
//!
//!   cargo run --release            fit the bundled measured rings, run the
//!                                  leave-one-out check, print predictions
//!   cargo run --release -- core    benchmark the core simulator (CPU)
//!   cargo run --release -- econ    first-principles cost and energy vs GPUs
//!   cargo run --release -- gpu     verify the GPU kernel against its CPU
//!                                  twin, then benchmark every-symbol speed

use studio::fit::{fit_device, DeviceFit};
use studio::measure::Measurement;
use studio::process::{leave_one_out, ProcessModel, Sample};

/// Step-1 acceptance thresholds (docs/photonic/PLAN.md, "Next steps").
const MAX_RMS_DB: f64 = 1.0;
const MAX_Q_ERROR: f64 = 0.15;

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("core") => bench_core(),
        Some("gpu") => bench_gpu(),
        Some("econ") => econ(),
        Some("resonances") => resonances(),
        _ => report(),
    }
}

fn resonances() {
    for d in Measurement::bundled().iter().map(fit_device) {
        for r in &d.resonances {
            println!(
                "{:8} λ {:8.3}  Q_L {:6.0}  ER {:5.1} dB  rms through {:.2} dB  drop {:.2} dB",
                d.name,
                r.ring.lambda_res_nm,
                r.ring.q_loaded(),
                r.ring.extinction_db(),
                r.rms_through_db,
                r.rms_drop_db
            );
        }
    }
}

fn report() {
    let t0 = std::time::Instant::now();
    let devices: Vec<DeviceFit> = Measurement::bundled().iter().map(fit_device).collect();
    let fit_ms = t0.elapsed().as_secs_f64() * 1e3;

    println!("Measured silicon: SiEPIC openEBL 2024-10, double-bus rings, TE, 25 C\n");
    println!(
        "{:8} {:>4} {:>4} {:>4} {:>7} {:>7} {:>8} {:>8} {:>7} {:>7} {:>8} {:>9}",
        "device", "R", "gap", "res", "n_g", "κ²", "loss", "Q_L", "Q_i", "FWHM", "ER", "worst rms"
    );
    println!(
        "{:8} {:>4} {:>4} {:>4} {:>7} {:>7} {:>8} {:>8} {:>7} {:>7} {:>8} {:>9}",
        "", "µm", "nm", "", "", "@1550", "dB/cm", "", "", "GHz", "dB", "dB"
    );
    for d in &devices {
        let r = d.nearest(1550.0).ring;
        println!("{:8} {:>4.0} {:>4.0} {:>4} {:>7.3} {:>7.4} {:>8.2} {:>8.0} {:>7.0} {:>7.1} {:>8.1} {:>9.2}",
            d.name, d.radius_um, d.gap_nm, d.resonances.len(), d.n_g, r.kappa2(), r.loss_db_per_cm(),
            r.q_loaded(), r.q_intrinsic(), r.fwhm_ghz(), r.extinction_db(), d.worst_rms_db());
    }
    let all: Vec<_> = devices.iter().flat_map(|d| &d.resonances).collect();
    let worst = all.iter().map(|r| r.worst_rms_db()).fold(0.0, f64::max);
    let qualified: Vec<_> = all.iter().filter(|r| r.qualified()).collect();
    let worst_q = qualified
        .iter()
        .map(|r| r.worst_rms_db())
        .fold(0.0, f64::max);
    let under = all.iter().filter(|r| r.worst_rms_db() < MAX_RMS_DB).count();
    println!(
        "\n{} resonances fitted in {:.0} ms ({} under {:.1} dB RMS).",
        all.len(),
        fit_ms,
        under,
        MAX_RMS_DB
    );
    println!(
        "  strict, all resonances:                 worst RMS {:.2} dB → {}",
        worst,
        verdict(worst < MAX_RMS_DB)
    );
    println!(
        "  SNR-qualified ({} of {}, drop peak ≥ {:.0} dB above noise): worst RMS {:.2} dB → {}",
        qualified.len(),
        all.len(),
        studio::fit::QUALIFIED_SNR_DB,
        worst_q,
        verdict(worst_q < MAX_RMS_DB)
    );
    for r in all.iter().filter(|r| !r.qualified()) {
        println!("    excluded: {:.2} nm, drop SNR {:.1} dB, RMS {:.2} dB (band edge; drop wings carry a stray-light fringe)",
            r.ring.lambda_res_nm, r.drop_snr_db, r.worst_rms_db());
    }

    let samples: Vec<Sample> = devices.iter().map(Sample::of).collect();
    println!(
        "\nLeave-one-gap-out: learn κ²(gap) and loss without the device, predict its loaded Q"
    );
    println!(
        "{:8} {:>9} {:>9} {:>8}",
        "held out", "measured", "predicted", "error"
    );
    let holdouts = leave_one_out(&samples);
    for h in &holdouts {
        println!(
            "r{:<.0}g{:<4.0} {:>9.0} {:>9.0} {:>+7.1}%",
            h.sample.radius_um,
            h.sample.gap_nm,
            h.sample.q_loaded,
            h.predicted_q,
            100.0 * h.error()
        );
    }
    let worst_q = holdouts.iter().map(|h| h.error().abs()).fold(0.0, f64::max);
    println!(
        "Worst |error| {:.1}% (limit {:.0}%) → {}",
        100.0 * worst_q,
        100.0 * MAX_Q_ERROR,
        verdict(worst_q < MAX_Q_ERROR)
    );

    let model = ProcessModel::learn(&samples);
    println!(
        "\nProcess model (all eight devices): n_g = {:.3}",
        model.n_g
    );
    for l in &model.laws {
        println!("  R = {:>2.0} µm: κ² = {:.3e} · exp({:+.4} · gap/nm)  (κ² halves every {:.1} nm), loss {:.2} dB/cm",
            l.radius_um, l.intercept.exp(), l.slope, (0.5f64).ln() / l.slope, l.alpha_per_m * 10.0 / 10f64.ln() / 100.0);
    }
    println!("\nPre-registered predictions for devices not yet fabricated (1550 nm):");
    for (r, g) in [
        (10.0, 120.0),
        (10.0, 130.0),
        (5.0, 100.0),
        (5.0, 110.0),
        (3.0, 80.0),
    ] {
        if let Some(p) = model.predict(r, g) {
            println!("  r{r:.0}g{g:.0}: Q_L {:>6.0}, FWHM {:>5.1} GHz, κ² {:.4}, through extinction {:>4.1} dB",
                p.q_loaded(), p.fwhm_ghz(), p.kappa2(), p.extinction_db());
        }
    }
}

fn verdict(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "FAIL"
    }
}

fn bench_core() {
    use studio::core::{Core, CoreConfig, Pattern};
    let samples: Vec<Sample> = Measurement::bundled()
        .iter()
        .map(fit_device)
        .map(|d| Sample::of(&d))
        .collect();
    let model = ProcessModel::learn(&samples);
    println!("Cores built from the fitted process model; random weights; heaters settled first.\n");
    println!(
        "{:>7} {:>9} {:>9} {:>4} {:>8} {:>6} {:>6} {:>7} {:>8} {:>10} {:>9}",
        "core",
        "ring",
        "ch/FSR",
        "cal",
        "w error",
        "ENOB",
        "heat W",
        "pJ/MAC",
        "symbols",
        "wall",
        "slowdown"
    );
    for (size, radius, gap) in [(16, 5.0, 80.0), (32, 3.0, 70.0), (64, 3.0, 70.0)] {
        let ring = model.predict(radius, gap).expect("learned radius");
        let per_fsr = (ring.fsr_hz() / 100e9).floor();
        for calibrate in [false, true] {
            let mut core = Core::new(CoreConfig::new(ring, size, size));
            core.calibrate = calibrate;
            core.program(Pattern::Random);
            core.step(150_000_000_000, 0); // 150 µs: heaters settle
            core.reset_ledger();
            let t0 = std::time::Instant::now();
            let hw_fs: u64 = 10_000_000_000; // 10 µs = 100 000 symbols, every one computed
            core.step(hw_fs, usize::MAX);
            let wall = t0.elapsed().as_secs_f64();
            let hw = hw_fs as f64 * 1e-15;
            println!("{:>4}×{:<2} r{:.0}g{:<4.0} {:>9.0} {:>4} {:>7.2}% {:>6.2} {:>6.2} {:>7.3} {:>8} {:>7.1} ms {:>8.0}×",
                size, size, radius, gap, per_fsr, if calibrate { "on" } else { "off" }, 100.0 * core.weight_error(),
                core.ledger.enob(), core.heater_mw() / 1e3, core.ledger.pj_per_mac(), core.ledger.symbols, wall * 1e3, wall / hw);
        }
    }
    println!("\nch/FSR: 100 GHz channels that fit in one free spectral range of that ring.");
    println!(
        "Ring optics: fitted. Heater, noise, converter and laser figures: plan assumptions [A]."
    );
    println!("Simulator: one CPU thread, every symbol computed (noise, ADC, error); no GPU yet.");
}

fn bench_gpu() {
    use studio::core::{Core, CoreConfig, Pattern};
    use studio::gpu::{cpu_twin, Batch, Gpu, MAX_SYMBOLS};
    let mut gpu = match pollster::block_on(Gpu::new()) {
        Ok(g) => g,
        Err(e) => {
            println!("GPU unavailable: {e}");
            return;
        }
    };
    println!("Adapter: {}\n", gpu.adapter);
    let samples: Vec<Sample> = Measurement::bundled()
        .iter()
        .map(fit_device)
        .map(|d| Sample::of(&d))
        .collect();
    let model = ProcessModel::learn(&samples);
    let enob =
        |err_sq: f64, outputs: f64| (2.0 / ((err_sq / outputs).sqrt() * 12f64.sqrt())).log2();

    println!("Verification: GPU kernel (f32) against its CPU twin (f64), same inputs and RNG");
    println!(
        "{:>7} {:>6} {:>11} {:>11} {:>12} {:>16}",
        "core", "noise", "ENOB gpu", "ENOB twin", "Δ ENOB", "last outputs Δ"
    );
    for (size, radius, gap) in [(16, 5.0, 80.0), (32, 3.0, 70.0), (64, 3.0, 70.0)] {
        let ring = model.predict(radius, gap).unwrap();
        let mut core = Core::new(CoreConfig::new(ring, size, size));
        core.program(Pattern::Random);
        core.step(150_000_000_000, 0);
        for noise in [false, true] {
            let b = Batch::from_core(&core, 123_456_789, 8192, noise);
            let g = gpu.run(b.clone());
            let t = cpu_twin(&b);
            let lsb = 2.0 / (b.levels - 1.0) / b.half;
            let dy = g
                .y_last
                .iter()
                .zip(&t.y_last)
                .map(|(a, c)| (a - c).abs())
                .fold(0.0, f64::max)
                / lsb;
            let (eg, et) = (enob(g.err_sq, g.outputs()), enob(t.err_sq, t.outputs()));
            println!(
                "{:>4}×{:<2} {:>6} {:>11.4} {:>11.4} {:>12.5} {:>12.2} LSB",
                size,
                size,
                if noise { "on" } else { "off" },
                eg,
                et,
                eg - et,
                dy
            );
        }
    }

    println!("\nCross-check: GPU (every symbol) against the CPU core path (its own RNG stream)");
    let ring = model.predict(5.0, 80.0).unwrap();
    let mut core = Core::new(CoreConfig::new(ring, 16, 16));
    core.program(Pattern::Random);
    core.step(150_000_000_000, 0);
    core.reset_ledger();
    core.step(10_000_000_000, usize::MAX); // 100 000 symbols on the CPU path
    let g = gpu.run(Batch::from_core(&core, 0, 100_000, true));
    println!(
        "  16×16, 100 000 symbols: CPU core ENOB {:.3}, GPU ENOB {:.3}",
        core.ledger.enob(),
        enob(g.err_sq, g.outputs())
    );

    println!(
        "\nThroughput: {} symbols per dispatch, noise on",
        MAX_SYMBOLS
    );
    println!(
        "{:>7} {:>10} {:>14} {:>16} {:>14} {:>12}",
        "core", "wall", "symbols/s", "simulated MAC/s", "hw time", "slowdown"
    );
    for (size, radius, gap) in [(16, 5.0, 80.0), (32, 3.0, 70.0), (64, 3.0, 70.0)] {
        let ring = model.predict(radius, gap).unwrap();
        let mut core = Core::new(CoreConfig::new(ring, size, size));
        core.program(Pattern::Random);
        core.step(150_000_000_000, 0);
        let b = Batch::from_core(&core, 0, MAX_SYMBOLS, true);
        let _ = gpu.run(b.clone()); // warm-up: pipeline compile, first allocation
        let t0 = std::time::Instant::now();
        let reps = 3;
        for _ in 0..reps {
            let _ = gpu.run(b.clone());
        }
        let wall = t0.elapsed().as_secs_f64() / reps as f64;
        let syms = MAX_SYMBOLS as f64;
        let hw = syms / core.cfg.baud;
        println!(
            "{:>4}×{:<2} {:>8.1} ms {:>14} {:>16} {:>11.0} µs {:>11.0}×",
            size,
            size,
            wall * 1e3,
            si(syms / wall),
            si(syms * (size * size) as f64 / wall),
            hw * 1e6,
            wall / hw
        );
    }
}

fn si(v: f64) -> String {
    let (d, u) = [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")]
        .into_iter()
        .find(|(d, _)| v >= *d)
        .unwrap_or((1.0, ""));
    format!("{:.2} {u}", v / d)
}

fn econ() {
    use studio::econ::{engines, incumbents, sensitivity, World};
    let w = World::default();
    println!(
        "First-principles economics. Electricity ${}/kWh, PUE {}, {} years at full power;",
        w.usd_per_kwh, w.pue, w.years
    );
    println!(
        "silicon ${}/kg, glass ${}/kg. Incumbent COGS: third-party estimates (±20 %).\n",
        w.si_usd_per_kg, w.glass_usd_per_kg
    );
    let mut all = incumbents(&w);
    all.extend(engines().iter().map(|e| e.system(&w)));
    println!(
        "{:<27} {:<10} {:>8} {:>7} {:>7} {:>8} {:>9} {:>9} {:>10} {:>9} {:>7} {:>7}",
        "system",
        "precision",
        "PMAC/s",
        "W",
        "fJ/MAC",
        "COGS $",
        "TMAC/s/$",
        "3y TCO $",
        "PMAC/s/k$",
        "kWh made",
        "raw $",
        "idiot"
    );
    for s in &all {
        println!("{:<27} {:<10} {:>8.3} {:>7.0} {:>7.0} {:>8.0} {:>9.3} {:>9.0} {:>10.3} {:>9.1} {:>7.2} {:>7.0}",
            s.name, s.precision.split(' ').next().unwrap_or(""), s.macs_per_s / 1e15, s.power_w, s.pj_per_mac() * 1e3, s.cogs(),
            s.macs_per_s / s.cogs() / 1e12, s.tco(&w), s.macs_per_s / s.tco(&w) / 1e12, s.embodied_kwh, s.raw_material_usd, s.idiot_index());
    }
    println!(
        "\nidiot = manufacturing cost ÷ raw-material cost. kWh made = manufacturing electricity."
    );
    println!("Analog ≈5–6 bit is not FP8: equal task accuracy is unproven and is the first thing to test.");

    for e in engines() {
        let s = e.system(&w);
        println!("\n{} — {} ({})", e.name, e.precision, e.basis);
        println!("  {:.0} engines · fan-in {:.0} · fan-out {:.0} · {:.1} GBd · {:.2} PMAC/s · {:.0} W · ${:.0}",
            e.units, e.fan_in, e.fan_out, e.baud / 1e9, s.macs_per_s / 1e15, s.power_w, s.cogs());
        let total = s.cogs();
        let mut cost = s.cost.clone();
        cost.sort_by(|a, b| b.1.total_cmp(&a.1));
        let cs: Vec<String> = cost
            .iter()
            .filter(|(_, v)| *v > 0.0)
            .map(|(k, v)| format!("{k} {:.0} %", 100.0 * v / total))
            .collect();
        println!("  cost:   {}", cs.join(", "));
        let parts = e.power_parts();
        let pt: f64 = parts.iter().map(|(_, p)| p).sum();
        let ps: Vec<String> = parts
            .iter()
            .filter(|(_, p)| *p > 0.0)
            .map(|(k, p)| format!("{k} {:.0} %", 100.0 * p / pt))
            .collect();
        println!("  power:  {}", ps.join(", "));
        println!(
            "  converter floor: {:.1} fJ/MAC (ideal light, no static power)",
            e.converter_floor_fj()
        );
        println!("  what moves it (low ↔ high):  TMAC/s per $        fJ/MAC");
        for (name, [lo, hi]) in sensitivity(&e, &w) {
            println!(
                "    {:<34} {:>7.2} ↔ {:<7.2}  {:>6.1} ↔ {:<6.1}",
                name, lo.0, hi.0, lo.1, hi.1
            );
        }
    }
}
