//! QuPU — virtual quantum processing unit. The crate exposes:
//!   * `sim::Simulator` — state vector + scheduler + kernels
//!   * `circuit` — gate kinds and preset programs
//!   * `app::QupuApp` — the egui application
//!   * `start` — wasm entry point (mounts the app on a `<canvas>`)
//!
//! The simulator's own fields are what the renderer reads. There is no
//! parallel JS model; what you see on screen is the simulator's state.

pub mod circuit;
pub mod sim;
pub mod app;

#[cfg(target_arch = "wasm32")]
mod web {
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;

    #[wasm_bindgen(start)]
    pub fn _setup_panic_hook() {
        console_error_panic_hook::set_once();
    }

    /// Mount the QuPU app on the `<canvas>` with the given id.
    #[wasm_bindgen]
    pub fn start(canvas_id: String) -> Result<(), JsValue> {
        let canvas = web_sys::window()
            .ok_or_else(|| JsValue::from_str("no window"))?
            .document()
            .ok_or_else(|| JsValue::from_str("no document"))?
            .get_element_by_id(&canvas_id)
            .ok_or_else(|| JsValue::from_str("canvas not found"))?
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .map_err(|_| JsValue::from_str("element is not a canvas"))?;

        let web_options = eframe::WebOptions::default();

        wasm_bindgen_futures::spawn_local(async move {
            eframe::WebRunner::new()
                .start(
                    canvas,
                    web_options,
                    Box::new(|cc| Ok(Box::new(crate::app::QupuApp::new(cc)))),
                )
                .await
                .expect("failed to start eframe");
        });

        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
pub use web::start;
