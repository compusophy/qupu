# QuPU

A virtual **quantum processing unit** in the browser — and in a native window —
built end-to-end in Rust.

The state-vector simulator and the UI live in the same crate. The renderer
reads the simulator's own fields directly: `sim.state` is the phasor grid you
see, `sim.cores[i].busy_for` is the meter that's filling, `sim.queue` is the
work that's about to dispatch. **The visualization is the simulation**, not a
serialized snapshot of it.

```
                          ┌───────────────────────────────────────┐
                          │   App::ui(&mut self, ui, frame)       │
                          │                                       │
   ┌──────────┐  reads    │   egui Painter draws sim.state,       │
   │ Simulator├──────────►│        sim.cores, sim.queue           │
   │   tick() │  writes   │   egui buttons mutate sim.circuit     │
   └─────▲────┘◄──────────│                                       │
         │                └───────────────────────────────────────┘
         │ dt = stable_dt
         ▼
   schedule_next() ─► partition gate into tiles aligned to 2^(target+1)
   per-core busy_for ─► counts down; on zero, apply_tile() mutates state
```

Each "core" is one slot in the Rust scheduler. Gates are partitioned into
tiles aligned by `2^(target+1)` amplitudes (so amplitude pairs never straddle
a tile boundary), then cores pick tiles off the queue and animate them in
real time at the configurable clock rate. Same dispatch model a GPU uses
for kernel workgroups — applied to quantum amplitude pairs.

## Build & run

```sh
npm run build         # cargo build --target wasm32-unknown-unknown + wasm-bindgen
npm start             # tiny static server on :8080
cargo run --release --bin qupu-native   # native window — same app
cargo test --lib --release              # simulator + tiling tests
```

Browser → <http://localhost:8080>.

## What you can do

- **Compose** circuits from `H, X, Y, Z, S, T, RX, RY, RZ, CNOT, CZ, SWAP`
  with selectable target/control qubits and a θ field for rotations.
- **Presets**: Bell · GHZ · QFT · Grover (1 iteration) · Scramble.
- Scale the register up to 8 qubits, the core count up to 16, and the
  scheduler clock from 1 to 60 Hz live from the config panel.
- Watch the **die** cells light up as tiles dispatch, the **state vector**
  grid breathe (size = magnitude, hue = phase), and **telemetry** track
  tiles, gates, core utilization, wall time, and norm drift.

## Layout

```
src/lib.rs        wasm entry point — mounts the egui app on a <canvas>
src/native.rs     `cargo run --bin qupu-native` — same app in a desktop window
src/sim.rs        Simulator, Core, scheduler, gate kernels (unit tests live here)
src/circuit.rs    GateKind, Op, gate matrices, preset programs
src/app.rs        eframe::App — die / telemetry / config / state / circuit panels
web/index.html    ~20-line shell: one canvas, six lines of JS to load the wasm
web/pkg/          wasm-bindgen output
server.mjs        ~30-line static server, no deps
```

That's the whole client. There is no JS app code, no CSS framework, no DOM
widgets. Every pixel inside the canvas is drawn by `egui::Painter` from Rust.

## Why "quantum GPU"

A real GPU exposes a programming model where you write one kernel and the
hardware fans it across thousands of threads. A state-vector quantum simulator
has the same shape: one gate, fanned across amplitude pairs. The clock rate
slider lets you slow that fan-out down to human speeds so you can *see* it
happen.

## Notes & limits

- Stable Rust 1.94. eframe 0.34 with the glow backend.
- The "cores" model concurrency at the scheduler level (idle/busy timers,
  per-core utilization). The actual kernel runs on a single thread; the
  alignment math, queue, and dispatch order are honest — only the wall-clock
  parallelism is virtual. Adding real Web Workers would mean dropping the
  unified-Rust-model property; that's the trade we made.
- Up to 8 qubits in the UI (256 amplitudes — still legible). Going higher
  works but the grid cells shrink past readability.
- The Grover preset uses an exact Toffoli decomposition only for n=3 and a
  rough CZ-ladder proxy for larger n.

## Related

- [`docs/photonic/PLAN.md`](docs/photonic/PLAN.md) — project plan for a
  browser-based microring-resonator optical computing simulator (with a
  reproducible back-of-envelope calculator in `docs/photonic/estimates.py`).
