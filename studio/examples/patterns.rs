use studio::core::{Core, CoreConfig, Pattern};
use studio::fit::fit_device;
use studio::measure::Measurement;
use studio::process::{ProcessModel, Sample};
fn main() {
    let s: Vec<Sample> = Measurement::bundled()
        .iter()
        .map(fit_device)
        .map(|d| Sample::of(&d))
        .collect();
    let ring = ProcessModel::learn(&s).predict(5.0, 80.0).unwrap();
    for p in Pattern::ALL {
        let mut core = Core::new(CoreConfig::new(ring, 16, 16));
        core.program(p);
        core.step(200_000_000_000, 0);
        core.reset_ledger();
        core.step(100_000_000, 2000);
        println!(
            "{:18} weight error {:5.2}%  ENOB {:.2}",
            p.name(),
            100.0 * core.weight_error(),
            core.ledger.enob()
        );
    }
}
