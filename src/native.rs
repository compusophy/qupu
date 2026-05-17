//! Native binary entry. Opens a window via eframe and mounts the QuPU app.
//! `cargo run --release --bin qupu-native`

fn main() -> eframe::Result {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .init();

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([900.0, 620.0])
            .with_title("QuPU // virtual quantum processing unit"),
        ..Default::default()
    };

    eframe::run_native(
        "QuPU",
        options,
        Box::new(|cc| Ok(Box::new(qupu::app::QupuApp::new(cc)))),
    )
}
