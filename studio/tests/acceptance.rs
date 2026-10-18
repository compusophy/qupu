//! Step-1 acceptance: the ring model against measured silicon.
//! Thresholds are the ones written down before fitting (PLAN.md next steps).

use studio::fit::{fit_device, DeviceFit};
use studio::measure::Measurement;
use studio::process::{leave_one_out, Sample};

fn devices() -> Vec<DeviceFit> {
    Measurement::bundled().iter().map(fit_device).collect()
}

#[test]
fn bundled_measurements_parse() {
    let ms = Measurement::bundled();
    assert_eq!(ms.len(), 8);
    for m in &ms {
        assert_eq!(m.wl_nm.len(), 20_000);
        assert!((m.wl_nm[0] - 1500.0).abs() < 0.01 && (m.wl_nm[19_999] - 1580.0).abs() < 0.01);
        assert_eq!(m.through_dbm.len(), m.drop_dbm.len());
    }
}

#[test]
fn group_index_is_consistent_across_devices() {
    let ng: Vec<f64> = devices().iter().map(|d| d.n_g).collect();
    let mean = ng.iter().sum::<f64>() / ng.len() as f64;
    assert!((4.15..4.22).contains(&mean), "n_g = {mean}");
    assert!(ng.iter().all(|x| (x - mean).abs() / mean < 0.005), "{ng:?}");
}

#[test]
fn model_fits_every_snr_qualified_resonance_within_1_db() {
    let ds = devices();
    let qualified: Vec<_> = ds
        .iter()
        .flat_map(|d| &d.resonances)
        .filter(|r| r.qualified())
        .collect();
    assert!(
        qualified.len() >= 30,
        "only {} qualified resonances",
        qualified.len()
    );
    for r in qualified {
        assert!(
            r.worst_rms_db() < 1.0,
            "{:.2} nm: {:.2} dB",
            r.ring.lambda_res_nm,
            r.worst_rms_db()
        );
    }
}

#[test]
fn leave_one_gap_out_predicts_loaded_q_within_15_percent() {
    let samples: Vec<Sample> = devices().iter().map(Sample::of).collect();
    let holdouts = leave_one_out(&samples);
    assert!(holdouts.len() >= 7);
    for h in holdouts {
        assert!(
            h.error().abs() < 0.15,
            "r{}g{}: {:+.1}%",
            h.sample.radius_um,
            h.sample.gap_nm,
            100.0 * h.error()
        );
    }
}

#[test]
fn wider_gap_means_weaker_coupling_and_higher_q() {
    let ds = devices();
    for radius in [10.0, 5.0] {
        let mut s: Vec<Sample> = ds
            .iter()
            .filter(|d| d.radius_um == radius)
            .map(Sample::of)
            .collect();
        s.sort_by(|a, b| a.gap_nm.total_cmp(&b.gap_nm));
        for w in s.windows(2) {
            assert!(
                w[1].kappa2 < w[0].kappa2 && w[1].q_loaded > w[0].q_loaded,
                "R={radius}: {w:?}"
            );
        }
    }
}
