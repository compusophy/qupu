//! Studio — a photonic design studio, starting from one honest loop:
//! measured silicon → fitted ring physics → a process model that predicts
//! devices it has not seen → a live simulator of a weight-bank core built
//! from those fitted parameters.
//!
//!   * `ring`    — add-drop microring transfer-matrix model (simulator)
//!   * `matfile` — MATLAB level-5 reader (measurement ingest)
//!   * `measure` — foundry spectra; bundled SiEPIC openEBL 2024-10 rings
//!   * `fit`     — Levenberg–Marquardt and per-resonance fits (optimiser)
//!   * `process` — coupling-vs-gap law, loss, n_g; leave-one-out check
//!   * `core`    — broadcast-and-weight core on a femtosecond clock
//!   * `gpu`     — every symbol of the core on the GPU (wgpu), with a CPU twin
//!   * `web`     — the browser view (wasm32 only)

pub mod core;
pub mod fit;
pub mod gpu;
pub mod matfile;
pub mod measure;
pub mod process;
pub mod ring;

#[cfg(target_arch = "wasm32")]
mod web;
