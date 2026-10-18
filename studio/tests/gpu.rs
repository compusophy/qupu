//! The GPU kernel must reproduce its f64 CPU twin. Skips (passes with a
//! note) when the machine has no GPU adapter at all.

use studio::core::{Core, CoreConfig, Pattern};
use studio::fit::fit_device;
use studio::gpu::{cpu_twin, Batch, Gpu};
use studio::measure::Measurement;
use studio::process::{ProcessModel, Sample};

#[test]
fn gpu_kernel_matches_cpu_twin() {
    let mut gpu = match pollster::block_on(Gpu::new()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let samples: Vec<Sample> = Measurement::bundled()
        .iter()
        .map(fit_device)
        .map(|d| Sample::of(&d))
        .collect();
    let ring = ProcessModel::learn(&samples).predict(5.0, 80.0).unwrap();
    let mut core = Core::new(CoreConfig::new(ring, 16, 16));
    core.program(Pattern::Random);
    core.step(150_000_000_000, 0);
    for noise in [false, true] {
        let b = Batch::from_core(&core, 987_654_321, 4096, noise);
        let (g, t) = (gpu.run(b.clone()), cpu_twin(&b));
        let rel = (g.err_sq - t.err_sq).abs() / t.err_sq;
        assert!(
            rel < 1e-3,
            "noise {noise}: err_sq gpu {} twin {} (rel {rel})",
            g.err_sq,
            t.err_sq
        );
        let lsb = 2.0 / (b.levels - 1.0) / b.half;
        for (a, c) in g.y_last.iter().zip(&t.y_last) {
            assert!((a - c).abs() <= lsb * 1.01, "output {a} vs {c}");
        }
    }
}
