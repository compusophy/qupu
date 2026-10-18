# Project plan: a browser-based simulator for a microring-resonator optical computing architecture

*Working draft v0.1 · 2026-10-17 · companion calculator: [`estimates.py`](estimates.py)*

> **Update 2026-10-18.** A first-principles cost model
> ([FIRST_PRINCIPLES.md](FIRST_PRINCIPLES.md)) finds that the ring core
> below is roughly at parity with GPUs per dollar and per joule. WDM fan-in
> is capped near 38 channels, which leaves a converter floor of about 100
> fJ/MAC. A free-space "glass engine" with fan-in of about 1000 is the
> stronger candidate. The ring work stays as the measured, fab-ready
> foundation.

**How to read this document.**
Every number below comes from a stated assumption, and the calculator
reproduces every derived figure (`python3 docs/photonic/estimates.py`).
Change an assumption there and you can see what moves. The following tags mark
confidence:

- **[K]** known: published or textbook physics.
- **[A]** assumption: a parameter with a stated range that has to be measured or quoted.
- **[S]** speculative: an untested idea, a novelty claim, or a forecast.

§13 collects every [S] item in one place.

---

## 0. Summary

**What.** A Rust→WebAssembly + WebGPU simulator of an optical
matrix-multiply engine built from silicon microring resonators (MRRs). It
models physics, simulated time, energy and precision honestly enough that
its conclusions can steer a real tape-out.

**Headline numbers (64 λ × 64 rows at 10 GBd, full utilisation; §6, §7):**

| Quantity | Value | Note |
|---|---|---|
| Peak throughput | **40.96 TMAC/s** (81.9 TOPS if 1 MAC = 2 ops) | The brief's figure is correct as arithmetic |
| Energy per MAC, nominal | **≈ 0.30 pJ** (12.4 W) | Thermal tuning (53 %) and laser (21 %) dominate |
| Energy per MAC, range | 0.046 – 1.56 pJ | Optimistic and pessimistic assumption sets |
| NVIDIA B200 (datasheet, dense) | 0.44 pJ/MAC FP8, 0.22 pJ/MAC FP4 | TDP ÷ peak. Crude, see §7 |
| One core vs. one B200 (FP8) | 1/55 of the throughput | Photonics wins on energy or latency, not raw rate per core |
| End-to-end latency of one MVM | ≈ 5.7 ns, of which light-in-flight ≈ 3 % | Converters and memory dominate, not *c* |
| Temperature stability for 6-bit weights | **≈ 17 mK** (Q_L ≈ 7.7 k) | 4.3 mK for 8-bit. Closed-loop control is mandatory |
| Weight reload (thermal) | ≈ 28 µs = 277 k symbol slots | Batch-1 LLM decode is not viable on thermal weights |

**Five findings that shape the plan:**

1. **64 wavelengths in one free spectral range (FSR) at 10 GBd is not
   realistic in standard silicon rings.** At 10 GBd a ring needs a linewidth
   (FWHM) of about 25 GHz or more. Fitting 64 channels into the 2.2 THz FSR
   of a 5 µm ring forces 35 GHz spacing, which leaks −9.5 dB into the
   nearest neighbour. The plan reaches the same 41 TMAC/s with **32 λ × 2 FSR
   orders × 64 rows** instead (§2.3).
2. **On energy, a baseline MRR core sits roughly level with current GPUs,
   not 10× better.** Under nominal assumptions it is no better than a B200.
   It wins only with trimmed rings (little heater power), efficient lasers
   and cheap converters. The simulator's first job is to find out which
   regime is real.
3. **Weight reconfiguration, not light speed, is the throughput killer.**
   Thermal weights only pay off when each weight tile is reused for roughly
   10⁵–10⁶ symbols. That suits CNNs, large-batch inference, LLM prefill and
   fixed signal-processing operators. It does not suit batch-1 decode or
   attention, where both operands change.
4. **Precision of 4–8 bits is a trade between channel count and thermal
   stability.** Narrow rings pack more channels but need millikelvin
   control. Wide rings are robust but fewer fit. The simulator has to
   expose this trade, not hide it.
5. **The new architecture should go after these real bottlenecks**
   (reload latency, static power, calibration), not the arithmetic, which
   is already cheap (§1.4).

---

## 1. Vision and novelty

### 1.1 Vision

The goal is an instrument, not just a calculator: a live, inspectable,
cycle-accounted model of a photonic processor that runs in a browser tab.
In the spirit of QuPU, *the visualisation is the simulation*. The same
GPU buffers that hold ring detunings, spectra and outputs are what get
drawn on screen.

The simulator has three jobs:

1. **Decide.** Find the workloads and design points where an MRR
   architecture beats electronics on a stated metric, with error bars, or
   show that none exist.
2. **Design.** Explore architecture mechanisms (§1.4) quickly at
   transaction level, then precisely at symbol and waveform level.
3. **Fabricate.** Emit parameters and layout intent that line up with a
   foundry PDK, so a small test chip can confirm or refute the models
   (§8).

### 1.2 What already exists (and so is not our novelty)

| Approach | Representative work | Status |
|---|---|---|
| Coherent MZI meshes (unitary decomposition) | Shen et al. 2017; Bandyopadhyay et al. 2024 | [K] |
| Broadcast-and-weight (B&W) MRR weight banks, balanced detection | Tait et al. 2014, 2016, 2017 | [K] |
| Feedback and dither control of weight banks, >9-bit weight precision in the lab | Tait et al. 2018; Zhang et al. 2022 | [K] |
| PCM weights + microcomb "photonic tensor core" | Feldmann et al. 2021 | [K] |
| Time-wavelength interleaved convolution (dispersion delays) | Xu et al. 2021 | [K] |
| Time-integrating photoelectric multiplication; weights streamed over fibre | Hamerly et al. 2019; Sludds et al. 2022 (Netcast) | [K] |
| **Multi-FSR MRR crossbars** (one ring weights several resonance orders) | OFC 2023 W3G.4; reviewed in Photonic NNs based on Si microresonators | [K] |
| Modulation-and-weight in one ring (PN + heater) | arXiv:2505.11369 (2025) | [K] |
| Time-domain weight sharing to cut reconfigurations | Shastri lab, CLEO 2025 | [K] |
| Pilot-tone labelling and thermal-crosstalk cancellation in PICs | Politecnico di Milano "control and calibration recipes" | [K] |
| Commercial large-scale photonic accelerators | Ahmed et al. 2025; Hua et al. 2025 (*Nature*) | [K] |

A brief prior-art search on 2026-10-17 found **multi-FSR weight banks and
pilot-tone calibration are already published.** This plan therefore uses
them as baseline features and does not claim them as novelty.

### 1.3 Thesis

> Existing MRR tensor cores are **fast optics with slow, expensive edges**:
> thermal weight loading (µs), calibration (ms), converters (pJ), and static
> heater power (W). The arithmetic in the light is already nearly free.
> The proposed architecture treats those edges as **first-class, scheduled,
> overlappable resources**. Weights, wavelengths, spectral orders, time and
> calibration all appear explicitly in the instruction set, and hardware
> mechanisms turn the slow edges into pipelined operations. [S]

### 1.4 Proposed mechanisms, prior-art status and kill criteria

| ID | Mechanism | Attacks | Prior-art status | Kill criterion (measured in simulation) |
|---|---|---|---|---|
| **N1** | **Root-switched ping-pong weight arrays with just-in-time programming.** Two physically separate weight arrays per core. One fast electro-optic 1×2 switch at the broadcast-tree root sends all wavelengths to array A or array B. Both arrays feed the same dual-input balanced photodetectors. B is programmed while A computes, and B's heaters are powered only ≈ one settle time before the swap. | Reload latency | **Candidate novelty.** No direct precedent found in a preliminary search. Needs a proper search (§12) [S] | Fails if achieved TMAC/s gain is under 1.3×, *or* system pJ/MAC rises, on the target workload mix |
| **N2** | **Spectral-order batching** (multi-FSR). 32 channels × M = 2 orders. Orders are split before modulation (separate modulator buses) and before detection. The same heaters weight both orders, so each heater serves M× more MACs. | Static power per MAC, channel count | Prior art exists [K]. **Our contribution:** quantifying the inter-order weight mismatch caused by coupler dispersion, and joint heater calibration | Fails if inter-order mismatch after joint calibration exceeds 0.5 LSB at 6 bits in >20 % of Monte Carlo fab instances |
| **N3** | **In-band calibration from live traffic.** (a) Each heater carries a sub-LSB pilot dither on its own frequency (1–30 kHz). Lock-in on the compute detector's digitised output gives every ring's local slope at once. (b) On a sampled subset (≈ 10⁻³) of symbols, a least-squares fit of the measured outputs against the *known* inputs estimates the effective weight matrix continuously. No dedicated monitor photodiodes, and compute never stops. | Calibration downtime, drift, monitor area | Ingredients exist (dither control; pilot tones; single-monitor array locking) [K]. Applying them during live compute through the compute detectors *may* be new [S] | Fails if it cannot hold 6-bit weights under a 10 mK/s ambient ramp, or costs more than 2 % of throughput |
| **N4** | **Subcarrier-tagged readout.** Each FSR order (or group of rows) is tagged with an RF subcarrier, so a single detector per row carries M outputs. They are separated digitally by Fourier analysis, which draws directly on the collaborator's channel-separation expertise. This removes the optical order demultiplexer (≈ 1–2 dB) and its guard bands. | Detector count, optical loss | Subcarrier multiplexing is standard in communications [K]. Using it for photonic MVM readout is believed new [S]. Physics may be against it: ADC energy per conversion rises with sample rate | Fails if system pJ/MAC rises versus the optical-demux baseline |
| **N5** | **Explicit photonic ISA + reuse-aware compiler + honest simulated time.** Weight programming, swaps, calibration and streaming are instructions with physical costs, and the compiler schedules them. | All of the above (makes them optimisable) | Methodological: architecture simulators exist (gem5, SST), but not this target with these costs | — (infrastructure) |

What would count as success: at least one of N1–N4 improves a primary
metric (achieved TMAC/s or system pJ/MAC on the target workloads) by
**≥ 1.5×** over plain B&W under nominal assumptions, while ≥ 80 % of
Monte Carlo fab instances keep **≥ 6 effective bits**. That gate (G2)
decides whether a novel test chip is taped out (§9).

**A worked N1 trade-off.** A naive ping-pong doubles the largest energy
term, because the idle bank's heaters must hold its weights. Under nominal
assumptions:

- Single bank: 12.4 W. Ping-pong with both banks always heated: 18.9 W
  (+53 %). It therefore only wins on energy when the utilisation gain is
  above 1.53×. That happens only in a band of batch sizes around
  "stream time ≈ settle time" (around 277 k symbols per tile).
- With **just-in-time programming** (idle bank unpowered until one settle
  time before the swap), the extra power is P_thermal · min(1, t_settle /
  t_stream). At a batch of 10⁶ that is +1.8 W (+15 %) against a
  utilisation gain of 1.28×, so it wins.

The catch: powering the idle bank up is a thermal step that the active bank
feels through the substrate, and the controller has to reject it. This is
exactly the kind of non-obvious scheduling × physics interaction the
simulator exists to find.

---

## 2. Reference architecture (what gets simulated)

### 2.1 Baseline core: broadcast-and-weight

```
 laser source (DFB array or comb, 32 λ x M orders)
        │  fibre → edge/grating coupler
        ▼
 ┌─ order demux ─┬───────────────┐            (M = 2: two buses)
 │               ▼               ▼
 │   ring modulator bus 1   ring modulator bus 2     ← 32 DAC lanes each, 10 GBd
 │               └──── order mux ┘
 │                       ▼
 │        [N1 root switch: A | B]                   ← one fast EO 1x2 (ns)
 │                       ▼
 │        1→64 broadcast splitter tree
 │      ┌────────┬───────┴───────┬────────┐
 │      ▼        ▼               ▼        ▼
 │   row 0    row 1     …     row 62   row 63        ← each row: 32 add-drop rings
 │   ○○○…○    ○○○…○            ○○○…○    ○○○…○           (heater per ring, sets weight)
 │   │drop│thru                                       drop/through → balanced PD pair
 │   order demux per row (or N4 subcarrier tags)
 │   ▼
 │   balanced PD → TIA → ADC (10 GS/s)  x 64 rows x M
 │                     ▼
 └──────────── digital: accumulate partial sums, rescale, SRAM ───────────
```

A weight is w = P_drop − P_through ∈ [−1, 1] (balanced detection gives
the sign). Inputs are non-negative intensities, so signed inputs need an
offset term or a second pass. Each row computes one dot product per symbol
per order.

### 2.2 Design point and why

| Parameter | Value | Constraint that sets it |
|---|---|---|
| Symbol rate | 10 GBd [A] | Per the brief. Ring modulators manage 25–50 GBd, but ADC energy per conversion worsens above about 10–20 GS/s |
| Ring FWHM | 25–40 GHz (Q_L ≈ 7.7 k – 4.8 k) [A] | Must be ≳ 2.5× the signal bandwidth, or the ring filters the symbols (ISI that depends on the weight). Checked at L3 |
| Channel spacing | 100 GHz | 4× FWHM → −18.1 dB nearest-neighbour leakage before calibration |
| Ring radius | ≈ 3.3 µm (Si strip, n_g ≈ 4.3) | FSR 3.36 THz → 33 channels per FSR → **32 used** plus a guard |
| FSR orders M | 2 (N2) | Gives 64 effective λ channels: span ≈ 6.7 THz ≈ 54 nm, C+L band |
| Rows | 64 | Per the brief. Every extra row needs more power per line, and modulator-ring power handling (§4.8) caps rows at roughly 100–200 at this P_det [A] |
| Peak | 32 × 2 × 64 × 10 GBd = **40.96 TMAC/s** | Same as the brief, with one difference: a 64 × 32 matrix applied to **2 vectors** per symbol, not one 64 × 64 matrix. Any tiled GEMM with batch ≥ 2 maps onto it |
| Alternative band | O-band on a 32 λ / 100 GHz CW-WDM MSA grid | Commodity multi-λ lasers from co-packaged optics. Decide in M0 (§12) |

### 2.3 Why not 64 λ in one FSR

| Ring radius | FSR | Channels per FSR @ 100 GHz | Comment |
|---|---|---|---|
| 5.0 µm | 2.22 THz (17.8 nm) | 22 | 64 channels would need 35 GHz spacing → −9.5 dB crosstalk |
| 3.3 µm | 3.36 THz (26.9 nm) | 33 | Baseline |
| 2.0 µm | 5.55 THz (44.5 nm) | 55 | Microdisk regime: bend loss, higher-order modes, more fab sensitivity [S] |

A single-FSR 64-channel bank at 10 GBd is marked **[S]**. It would need
Vernier double-ring weights or ~2 µm microdisks, and both carry precision
risks of their own.

### 2.4 Target "full architecture" (end of roadmap, speculative [S])

- 8 cores × (32 λ × 2 orders × 64 rows) → 328 TMAC/s peak, with N1 on
  each core.
- Electronic-photonic co-integration: monolithic (GlobalFoundries
  45SPCLO-class) or 3D-stacked CMOS for 32 k heater DACs, the ADCs and
  DACs, and SRAM. Pad-limited wire bonding stops working above roughly 100
  heaters (§8.7).
- External multi-λ laser module. Optional SiN microcomb if its wall-plug
  efficiency is ever good enough (§4.7).
- On-die SRAM sized for activation reuse. HBM-class interface for larger
  models.

### 2.5 Photonic ISA sketch (N5)

```
WPROG  b, tile_addr        ; load ring setpoints for bank b (heater DACs); retires when settled to tolerance
WHEAT  b                   ; power bank b's heaters to their setpoints (just-in-time variant of WPROG)
WSWAP  core                ; flip root switch A<->B (ns)
STREAM core, src, n, dst   ; n symbols in → n x rows x M outputs out
ACC    dst, src, n         ; digital partial-sum accumulation
SCALE  dst, gain, bias     ; per-output range/offset correction (folds in calibration)
CAL    core, mode          ; pilot-dither lock-in step or sampled least-squares update (N3)
FENCE  | WAIT event
XFER   node, src, n        ; chip-to-chip / mesh (Phase 6)
```

Each instruction's latency, initiation interval and energy come from the
parameter set (Appendix A). The DES (§5) executes them, and the compiler
(§3.3) emits them. The design mirrors writing an emulator for a real ISA
(as with v86), with a physical cost model attached to every instruction.

---

## 3. Simulator system architecture

### 3.1 Principles

1. **The visualisation is the simulation.** Render passes bind the same
   GPU storage buffers the compute kernels write. There is no serialised
   snapshot.
2. **Every number carries provenance.** `Param { value, unit, low, high,
   source, confidence }`. Reports print which outputs depend on
   low-confidence inputs.
3. **Determinism.** Integer simulated time, a seeded counter-based RNG,
   and a fixed reduction order in kernels. Same seed + same device gives
   bit-identical results. Across devices, results agree within a stated
   tolerance (§3.6).
4. **A ladder of fidelity, all on one clock.** The same program runs at
   L0–L3 (§3.4). Fidelity changes how values are computed, never how
   simulated time is defined.
5. **A golden model on the CPU in f64.** Every GPU kernel has a Rust f64
   reference. CI compares them.

### 3.2 Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Workload graph (GEMM, conv, transformer block, FIR/DFT, …)              │
├─────────────────────────────────────────────────────────────────────────┤
│ Compiler / mapper: tiling, reuse analysis, ping-pong & order scheduling │  → ISA program
├─────────────────────────────────────────────────────────────────────────┤
│ DES timing engine (Rust/wasm): u64 fs clock, resources, events          │  ← param set (+PDK import)
├───────────────────────────────┬─────────────────────────────────────────┤
│ Functional engine: CPU f64    │ Functional engine: WebGPU f32 kernels   │
│ (golden, small configs)       │ (spectra, effective weights, noisy MVM, │
│                               │  CMT time stepping)                     │
├───────────────────────────────┴─────────────────────────────────────────┤
│ Ledgers: time · energy · precision · utilisation · provenance           │
├─────────────────────────────────────────────────────────────────────────┤
│ Views: die/heatmap, spectrum vs comb, timeline, ledger, trace export    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Code layout (proposed)

```
photon/                      new workspace member or sibling repo (decision: §12)
  crates/
    phot-core    units, constants, complex<f64>, TMM/CMT golden models, noise; no deps
    phot-arch    declarative architecture spec (RON/TOML) → component graph; PDK import
    phot-time    DES engine, SimTime(u64 fs), rational clocks, resources, trace
    phot-gpu     WGSL kernels + command encoder; talks to JS shim via extern "C" imports
    phot-isa     ISA definition, assembler/disassembler, interpreter hooks
    phot-comp    workload graphs → tiled ISA programs (reuse- and reload-aware)
    phot-ledger  energy/area/precision accounting, report generation
    phot-web     wasm entry (#[no_mangle] exports), render passes
    phot-cli     native harness (wgpu) for CI, regression, golden comparison
  web/
    index.html   the loader line
    gpu.js       ~200–300-line WebGPU shim (handle table + command decoder)
  tools/         offline Python: MEEP FDTD, mode solving, GDSFactory layout, PDK extraction
```

The architecture spec is the **single source of truth.** The simulator, the
energy model and the GDSFactory layout generator (§8.3) all read the same
file.

### 3.4 Fidelity levels

| Level | What is computed | Time step | Typical use | Slowdown (wall ÷ simulated), 64×64 core |
|---|---|---|---|---|
| L0 | Roofline / analytic | — | Sanity bounds | ≪ 1 (instant) |
| L1 | Transaction-level: bursts occupy resources, ideal values on demand | Per instruction | Design-space sweeps, scheduling | ≪ 1 (faster than hardware) |
| L2 | Symbol-accurate: per-symbol noisy MVM with effective weights (crosstalk folded in), converter models, quantisation | Per symbol (quasi-static rings, valid because 100 ps ≫ τ_photon ≈ 6 ps) | Accuracy and precision studies, benchmarks | ≈ 52× at 2 TFLOP/s effective (10× at 10 TFLOP/s) |
| L3 | Waveform-accurate: time-domain CMT envelopes per (row, ring, λ), modulator and detector bandwidths | 16–32 samples per symbol | ISI, ring dynamics, validating L2's quasi-static assumption | ≈ 10⁵–5×10⁵× (short windows, small configs) |
| L4 | Full-wave FDTD/EME of single components | — | Offline only (MEEP etc.) | n/a |

Two corrections to the brief's framing:

- In an *incoherent* B&W core the per-symbol arithmetic is **real-valued**
  (powers), so L2 is a real GEMM.
- Complex arithmetic belongs to the field transfer functions (spectra and
  crosstalk), coherent crosstalk paths, the L3 envelopes, and any coherent
  comparison baseline (MZI mesh). WebGPU complex kernels are still central,
  just not on the per-symbol path.

### 3.5 Rust/wasm ↔ WebGPU without wasm-bindgen or web-sys

WebGPU is a JavaScript API. Wasm cannot call it without imports, so a
literal one-line loader is not enough. The minimum is a small hand-written
shim, keeping everything else wasm-bindgen-free:

```js
const g = gpu(); const { instance } = await WebAssembly.instantiateStreaming(fetch('photon.wasm'), { env: g.imports }); g.bind(instance);
```

- **Handle table.** `gpu.js` owns `GPUDevice`, buffers and pipelines in
  arrays and hands wasm integer handles.
- **Command buffers in linear memory.** Each frame or step, wasm writes a
  compact binary command list (create, write, dispatch, copy, read) into
  a ring buffer and calls `gpu_submit(ptr, len)`. The shim decodes it and
  issues WebGPU calls. Bulk data goes over with `queue.writeBuffer(buf, 0,
  memory.buffer, ptr, len)`, with no copying in JS.
- **Async readback.** `mapAsync` cannot block wasm. A read is posted with
  a ticket, and on completion the shim copies into wasm memory and calls
  the export `on_gpu_done(ticket)`. This fits the DES naturally (§5.3).
- **Wasm exports:** `alloc`, `init`, `step(budget_ms)`,
  `on_gpu_done(ticket)`, `on_input(kind, a, b)`. All are `extern "C"`
  with `(ptr, len)` strings.
- **Threads (optional):** wasm threads need `SharedArrayBuffer`, which
  needs cross-origin isolation (COOP/COEP headers). `server.mjs` can send
  them. Static hosts that can't (e.g. GitHub Pages) need a
  service-worker workaround.
- **Conflict to flag:** QuPU's current web build uses eframe, which pulls
  in wasm-bindgen and web-sys. The photonic simulator should not inherit
  that. It renders through its own WebGPU passes, which also makes
  "visualisation = simulation" literally zero-copy.

### 3.6 GPU kernels, precision and limits

| Kernel | Work | Notes |
|---|---|---|
| `ring_spectrum` | Complex TMM response for each ring × each spectral sample (e.g. 4096 × 512) | Phases φ mod 2π reduced **on the CPU in f64**, then uploaded as f32 detunings. WGSL has no f64 in core |
| `effective_weights` | Per (row, λ): cascaded through-port products along the bus + own drop → W_eff, including crosstalk | Recomputed only when heaters or temperatures change |
| `mvm_noisy` | Tiled f32 GEMM + shot/thermal/RIN noise + ADC quantisation/INL | Workgroup tiling. Default limits: 16 KiB workgroup storage, 256 invocations per workgroup |
| `cmt_step` (L3) | Exact exponential update per envelope: a ← a·e^{pΔt} + jκs·(e^{pΔt} − 1)/p, p = jΔω − 1/τ | Unconditionally stable for piecewise-constant drive |
| `thermal_step` | ΔT = M·P (steady state) + per-node first-order dynamics | Crosstalk matrix M from FEM or measurement |

- **RNG.** A counter-based generator (Philox4x32-10 or a squares-style
  hash) keyed by (seed, stream, counter). WGSL has no 64-bit multiply or
  mul-hi, so it is emulated with 16-bit limbs. The payoff is
  reproducibility across runs, the native harness and mesh nodes.
- **f32 is adequate** relative to 4–8-bit hardware. Absolute phase is the
  exception: φ ≈ 300 rad for a 31 µm ring and ≈ 10⁵ rad for cm-scale
  delays, which loses precision in f32. That is why phases are reduced in
  f64 on the CPU.
- **Limits.** Default WebGPU caps are a 128 MiB storage binding and a 256
  MiB buffer (adapters may grant more). Large sweeps get tiled. Timestamp
  queries are optional and may be coarsened by browsers, so wall-clock is
  also measured with `performance.now()` around submit/complete.
- **Cross-device agreement.** Vendors differ in FMA contraction and
  transcendentals, so bit-exactness is guaranteed only on one device. The
  CPU f64 golden model is the arbiter.

### 3.7 Native harness and CI

`phot-cli` runs the same WGSL through `wgpu` natively, which is fine
outside the browser constraint. It also runs the f64 golden model and the
regression suite: analytic ring tests, GPU-vs-CPU tolerances, determinism,
and the reproductions of published results (§8.5).

### 3.8 Views

- Die heatmap of heater power and ring temperature.
- Spectrum view: ring resonances against laser lines, per row and per order.
- Live energy ledger.
- Timeline view: per-resource occupancy, plus a ruler showing **simulated
  hardware time vs. wall-clock and the slowdown factor**.
- Precision view: measured ENOB per output.
- Chrome trace-event JSON export, so any run opens in Perfetto.

---

## 4. Physical models and their accuracy limits

### 4.1 Single ring

**Transfer matrix (steady state)** [K] (Bogaerts et al. 2012). r₁ and r₂
are the self-coupling coefficients, a the single-pass amplitude
transmission, and φ = 2π n_eff(λ) L / λ:

$$
\frac{E_t}{E_{in}}=\frac{r_2 a e^{j\phi}-r_1}{1-r_1 r_2 a e^{j\phi}},\qquad
\frac{E_d}{E_{in}}=\frac{-\kappa_1\kappa_2\sqrt{a}\,e^{j\phi/2}}{1-r_1 r_2 a e^{j\phi}}
$$

with FSR = λ²/(n_g L), finesse F = π√(r₁r₂a)/(1 − r₁r₂a), and
Q_L = π n_g L √(r₁r₂a) / (λ(1 − r₁r₂a)).

Dispersion is captured through n_eff(λ) = n_eff(λ₀) − (λ − λ₀)(n_g − n_eff)/λ₀
plus a GVD term, and through coupling that depends on wavelength, κ(λ)
(critical for N2).

**Coupled-mode theory (dynamics)** [K] (Haus; Little et al. 1997):

$$
\frac{da}{dt}=\Big(j\Delta\omega-\frac{1}{\tau}\Big)a + j\kappa_1 s_{in},\quad
\frac{1}{\tau}=\frac{1}{\tau_0}+\frac{1}{\tau_{e1}}+\frac{1}{\tau_{e2}},\quad \kappa_i^2=\frac{2}{\tau_{ei}}
$$

Here |a|² is the stored energy, and the field amplitude decays with time
constant τ, so Q_L = ω₀τ/2. An optional **two-mode CMT** (clockwise and
counter-clockwise waves coupled by sidewall backscatter) reproduces
resonance splitting.

### 4.2 Weight bank

- **Cascade.** The power reaching ring k is the product of the
  through-transmissions of rings 0…k−1 at that wavelength. The drop bus
  also passes rings, and both directions are modelled.
- **Effective matrix.** W_eff[i][j] = drop-bus power − through-bus power
  for channel j in row i, summed over *all* rings in the row. That is where
  inter-channel crosstalk lives: it is linear and deterministic, so it can
  be calibrated.
- **Channel assignment.** Rings are assigned to channels by sorting their
  native resonances (red-shift-only heaters), minimising total heater
  power. This is a known technique in WDM transceivers.
- **Signal filtering.** When a weight is biased on the slope, the ring acts
  as a frequency discriminator for the modulated signal. L3 quantifies
  the resulting ISI. L2 assumes it is negligible, which is checked rather
  than presumed.

### 4.3 Modulators

- Ring modulators (depletion PN) with a Lorentzian transfer function. A
  pre-distortion lookup table linearises them, and a residual INL is
  modelled.
- Bandwidth: a photon-lifetime pole plus an RC pole, 20–40 GHz [A].
- Each modulator sees the **full per-line power** (≈ 2.5 mW after the chip
  coupler, §4.7) before the 1→64 split. Self-heating and bistability are
  modelled, and they cap power per line.

### 4.4 Detection and noise

Per balanced row, with a 5 GHz noise bandwidth, 15 pA/√Hz TIA input noise
[A] and 1 A/W responsivity [A]:

| P_det per row | Thermal | Shot | RIN −150 dB/Hz | → ENOB (full scale) | With RIN −135 dB/Hz (comb-like) |
|---|---|---|---|---|---|
| 100 µW | 1.06 µA | 0.40 µA | 0.22 µA | 5.6 | 5.1 |
| 200 µW (nominal) | 1.06 µA | 0.57 µA | 0.45 µA | **6.5** | 5.4 |
| 400 µW | 1.06 µA | 0.80 µA | 0.89 µA | 7.2 | 5.5 |

Two things stand out:

- **Noisy lasers erase the benefit of more power.** RIN scales with the
  signal, so above about 100 µW a comb-like −135 dB/Hz source pins the
  result near 5 bits. Balanced detection only partly cancels RIN, because
  each line's RIN is independent and weighted differently.
- **Dynamic-range penalty.** A 64-term sum with random signs typically has
  about 1/8 of full-scale amplitude, which costs **½·log₂64 = 3 bits**
  unless the output range is rescaled per layer (`SCALE`) and clipping is
  accepted. Published photonic precision figures often leave this out.

### 4.5 Converters

- **DAC:** ENOB, INL/DNL, latency, and energy per sample. Nominal 1.5 pJ
  per sample at 10 GS/s with 6–8 bits, including the driver; range 0.5–3
  [A].
- **ADC:** the same model, plus a Walden-style figure of merit for scaling
  energy with rate and bits (Murmann survey). Nominal 2 pJ per
  conversion; range 0.5–5 [A]. Pipeline latency 1–5 ns.
- **Heater DACs:** slow (MHz), 12–16 bits, because heater power goes as V²
  and a weight is a steep function of detuning.

### 4.6 Thermal

- **Tuning:** dλ/dT ≈ 70 pm/K [K]. Heater efficiency 0.25 nm/mW for plain
  heaters, about 1 nm/mW with an undercut [A]. Average tuning 0.4 nm (fab
  offset + setpoint) gives **1.6 mW per ring, 6.55 W for 4096 rings**
  nominally.
- **Dynamics:** τ_th ≈ 2–10 µs plain, 20–100 µs undercut [A]. An undercut
  saves power but costs speed. Settling to 1/256 takes ln(256)·τ ≈ 5.5τ,
  so 27.7 µs at τ = 5 µs.
- **Crosstalk:** a matrix M (5–20 % between neighbours without trenches
  [A]), from FEM and then fitted to measurement.
- **Drift:** an Ornstein–Uhlenbeck ambient process plus self-heating from
  circulating power.

**Required stability** (worst case at the steepest bias, w = 2D − 1):

| FWHM | 4-bit | 6-bit | 8-bit |
|---|---|---|---|
| 25 GHz | 73 mK | **17.5 mK** | 4.3 mK |
| 40 GHz | 118 mK | 28 mK | 6.9 mK |

This is the central precision trade. A wider linewidth relaxes thermal
control but gives fewer channels per FSR, and it moves the limit from
temperature onto channel count and crosstalk.

### 4.7 Lasers

The nominal path loss is 13 dB [A]:

- chip coupling: 2 dB
- input modulator: 4 dB
- splitter excess: 1.5 dB
- weight bank: 2.5 dB
- routing: 1 dB
- margin: 2 dB

On top of that comes the 1/64 broadcast split. That works out to
**4.0 mW per line out of the laser** and 255 mW optical in total.

Wall-plug efficiency then sets the electrical power:

- DFB array, 10–20 %: 1.3–2.6 W [A].
- Kerr comb + EDFA, 1–3 %: 8–25 W [A].

Kerr combs are attractive for line count but currently poor on wall-plug
efficiency, and RIN per line is often worse. Comb vs. DFB array is a
first-order energy decision, not a detail.

### 4.8 Nonlinearity

About 11 µW per line reaches each row's bank. With ≈ 40× intracavity
build-up (finesse ≈ 134), the weight rings hold about 0.5 mW. That is well
below the thresholds for two-photon absorption, free-carrier absorption and
self-heating, *except in the modulator rings*, which see the full ≈ 2.5 mW
per line. Self-heating and TPA/FCA
are modelled as power-dependent resonance shifts. Any configuration that
pushes a ring past 5 % of its FWHM in self-shift is flagged.

### 4.9 Fabrication variation

Resonance offsets come from a spatially correlated random field over width
and thickness (after Lu et al. 2017) [A]:

- width σ of 3–5 nm die-to-die and about 1 nm locally
- thickness σ of 1–3 nm
- sensitivity of 0.5–1 nm per nm of width and 1–2 nm per nm of thickness

The result is a resonance spread of about 1–5 nm die-to-die and about
0.1–0.5 nm within a bank.

Monte Carlo over that field gives *distributions* of heater power,
precision and yield, never single numbers. Post-fab trimming (permanent
resonance correction) is a parameter, and the optimistic energy case
depends on it.

### 4.10 Accuracy limits

| Model | Captures | Ignores | Expected error, given correct parameters | Validated against |
|---|---|---|---|---|
| TMM (steady state) | Spectrum, coupling, loss, FSR, dispersion | Dynamics, backscatter, nonlinearity | < 1 % in \|t\|², \|d\|². **Absolute λ₀ unpredictable to ±1–5 nm** (fabrication, not model) | Analytic limits; FDTD; TC1 |
| CMT (dynamics) | Photon lifetime, modulation response, ISI | Valid only for Q ≫ 1 and weak coupling | Errors of order 1/Q: negligible at Q ~ 5 k | TMM steady state; L3 vs measured eye diagrams (TC2) |
| Two-mode CMT | + backscatter splitting | Roughness statistics need input | Qualitative unless roughness is measured | TC1 spectra |
| Thermal (lumped + matrix) | Tuning, crosstalk, τ | Nonlinear conductivity, package boundary | ±10–20 % before fit, ±5 % after | FEM; TC1 heater sweeps |
| Noise (Gaussian, white) | Shot, thermal, RIN | 1/f, laser phase-to-intensity conversion via reflections | ±0.3 bit ENOB | TC2 |
| Converters (behavioural) | ENOB, INL, latency, energy | Circuit-level effects | Energy ±2× (vendor-dependent) | Quotes, datasheets |
| FDTD (offline, L4) | Full-wave fields | Fabrication uncertainty dominates | Mesh-limited, ≪ fab spread | Measurement |

**The dominant uncertainty is the fabricated geometry, not the equations.**
The plan is therefore built around calibration: fit the models to TC1
measurements, then predict TC2.

### 4.11 Offline full-wave validation

MEEP (FDTD) and a mode solver (e.g. femwell or MPB) extract n_eff(λ),
n_g, bend loss and κ(λ, gap) for the chosen PDK cross-section. Those feed
the TMM/CMT parameter tables. FDTD never runs in the browser; it only
produces the tables.

---

## 5. Simulated-time methodology

### 5.1 Time base

- `SimTime(u64)` in **femtoseconds**, which spans 5.12 hours. That covers
  everything from a ring round trip (0.30 ps = 300 fs) up to Monte Carlo
  drift runs of minutes.
- **Rational clocks.** Each clock domain stores its period as num/den fs,
  with a Bresenham-style remainder. A 10 GBd symbol is exactly 100 000 fs.
  A 28 GBd symbol is 35 714.2857… fs and must not drift.
- **Domains:** optical symbol clock, ADC and DAC clocks, digital clock
  (e.g. 1 GHz), heater control loop (≈ 1 MHz), thermal dynamics (µs),
  calibration (kHz), network (ms).

### 5.2 Discrete-event engine

- Binary heap keyed by (time, sequence number), which gives deterministic
  tie-breaking.
- Resources have **latency and initiation interval separately**. An ADC
  pipeline might accept a sample every 100 ps but deliver it 2 ns later. A
  request at time t starts at max(t, free_at).
- Components are state machines: `handle(event) → [(delay, event)]`.
- QuPU's scheduler already has the seed of this idea: per-core `busy_for`
  timers draining a tile queue. The new engine replaces the wall-clock
  `dt` with integer simulated time.

### 5.3 Timing-directed functional simulation

The timing model decides **when** each operation happens. The functional
engine computes **what** values result, in batched GPU dispatches: one
`STREAM` of n symbols is one dispatch.

GPU results arrive asynchronously in wall-clock time but are applied at
their simulated completion events, so results are deterministic. The DES
runs ahead freely for operations whose timing does not depend on data,
which covers almost all of them: dataflow is static.

Operations that depend on data are the exception, chiefly the closed-loop
calibration (N3). They block at a *lookahead horizon* until their readback
lands. This is conservative synchronisation, as in parallel DES.

Calibration loops are slow (kHz), so each readback covers about 10⁶
symbols and stalls are rare.

### 5.4 Physical cost table (nominal; full provenance in Appendix A)

| Operation | Simulated cost | Derivation |
|---|---|---|
| Ring round trip (R = 3.3 µm) | 0.30 ps | n_g L / c |
| Photon energy lifetime (FWHM 25 GHz) | 6.4 ps | Q_L / ω₀ |
| Bus transit, 32 rings at 25 µm pitch | 11.5 ps | 0.8 mm × n_g / c |
| Broadcast tree + routing, ≈ 3 mm | ≈ 43 ps | Geometry |
| Symbol period | 100 ps | 10 GBd |
| Modulator rise time | 10–20 ps | 20–40 GHz bandwidth |
| DAC latency | 1–2 ns | [A] |
| ADC latency | 1–5 ns | [A] |
| SRAM access | ≈ 1 ns | 1 GHz digital clock |
| Heater settle to 8 bits | 27.7 µs | 5.5 τ_th, τ_th = 5 µs [A] |
| Calibration update (N3) | ≈ 1 ms per full refresh | Pilot tones in 1–30 kHz, lock-in window ≥ 1 ms |
| Laser / comb start-up | ms – s, one-off | [A] |
| Chip-to-chip fibre | 5 ns per metre | n ≈ 1.47 |

### 5.5 Worked example: the first demo (one 8-ring bank, one dot product)

| t (ns) | Event | Resource |
|---|---|---|
| 0.00 | Read 8 inputs from SRAM | SRAM port |
| 1.00 | DAC conversion + driver | 8 DAC lanes |
| 2.50 | Light leaves the modulators | — |
| 2.55 | Light through the ring bank (≈ 0.2 mm + photon lifetime) | Optical |
| 2.65 | One symbol period integrated | Symbol clock |
| 2.70 | Balanced PD + TIA output | Receiver |
| 4.70 | ADC result | ADC pipeline |
| 5.70 | Written back | SRAM port |

So one dot product takes 5.7 ns end to end, of which the light accounts
for about 0.15 ns (≈ 3 %). In streaming mode a new vector enters every 100
ps: 8 MAC per 100 ps = 80 GMAC/s for this bank. The demo prints both
numbers, plus the wall-clock time the browser actually took.

### 5.6 Worked example: a tiled layer with weight reloads

A 4096 × 4096 layer on one 64 × 64 core needs 4096 tiles. Settle time is
27.7 µs per tile, and the stream time is batch × 100 ps.

| Batch (reuse per tile) | Single bank | Ping-pong (N1) |
|---|---|---|
| 1 (LLM decode) | 0.15 GMAC/s (0.0004 %) | ≈ same |
| 10³ | 0.15 TMAC/s (0.4 %) | 0.15 TMAC/s |
| 10⁵ | 10.9 TMAC/s (26.5 %) | 14.8 TMAC/s (36 %) |
| 10⁶ | 32.1 TMAC/s (78 %) | 41.0 TMAC/s (≈ 100 %) |

### 5.7 What gets reported

- **Simulated hardware time** T_hw: the authoritative result.
- **Wall-clock** T_wall.
- **Slowdown** S = T_wall / T_hw. L1 can show S < 1.
- Per-resource utilisation and stall reasons: waiting on settle, ADC,
  SRAM bandwidth, or calibration.
- Energy per instruction class and per component.
- Achieved TMAC/s and system pJ/MAC.
- Precision per output (ENOB measured against the f64 ideal).

### 5.8 Validating the timing model

1. Hand-computed timelines for small programs (like §5.5) are unit tests.
2. L1, L2 and L3 must agree on simulated time for the same program.
   Fidelity changes values, never time.
3. On TC2, measured latency and throughput with real converters are
   compared against prediction (target: ±10 %).

---

## 6. Energy, bandwidth and precision accounting: the honest bottlenecks

### 6.1 Energy ledger (64 λ × 64 rows at 10 GBd, 100 % utilisation)

| Component | Nominal | Optimistic | Pessimistic | Key assumption (nominal) |
|---|---|---|---|---|
| DAC + driver | 23.4 fJ | 7.8 | 46.9 | 1.5 pJ/sample, ÷ 64 rows of reuse |
| ADC | 31.2 | 7.8 | 78.1 | 2 pJ/conversion, ÷ 64 MAC per output |
| TIA | 7.8 | 3.1 | 15.6 | 0.5 pJ/sample |
| Laser (wall-plug) | 62.4 | 7.8 | 988 | 200 µW per row at the PD, 13 dB loss, 10 % WPE |
| Thermal tuning | **160** | 10 | 400 | 1.6 mW/ring (0.4 nm at 0.25 nm/mW) |
| SRAM in/out | 12.5 | 6.2 | 25 | 0.05 pJ/bit, 8-bit I/O |
| Digital accumulate + control | 5 | 3 | 10 | [A] |
| **Total** | **302 fJ/MAC (12.4 W)** | **46 fJ (1.9 W)** | **1.56 pJ (64 W)** | |

What drives the spread:

- **Pessimistic** is dominated by a comb laser at 2 % WPE (40 W).
- **Optimistic** requires post-fab trimming *and* undercut heaters (0.1
  mW/ring), a 20 % WPE laser, and 0.5 pJ converters.
- Off-chip memory (HBM, ≈ 2.5–4 pJ/bit [A]) is excluded here and charged
  per workload by the memory model.

### 6.2 What dominates

Thermal tuning and the laser, which are **static** costs, dominate the
nominal case. Static costs per MAC fall as utilisation and throughput
rise, which is why N1 (utilisation) and N2 (more MACs per heater) target
them. Converters are second-order *because* B&W reuses each input across 64
rows and each output across 64 inputs. Shrinking the core to 16 × 16
would quadruple converter energy per MAC.

### 6.3 Memory bandwidth

At 8-bit I/O, one core consumes **640 GB/s of inputs** and produces 640
GB/s of outputs (1.28 TB/s if partial sums are 16-bit). That is
comparable to a large share of an H100's 3.35 TB/s HBM. Without on-die
activation reuse, a multi-core chip is memory-bound. The memory model
(SRAM banks plus an HBM-class interface) is therefore part of M3, not an
afterthought.

### 6.4 Precision budget (target: 6 effective bits at the output)

The output error is the sum of several independent contributions:

- weight setting (heater DAC + drift + residual crosstalk)
- input DAC + modulator INL
- receiver noise (§4.4)
- ADC quantisation
- range clipping

The simulator measures each one separately by switching it off, and
reports the result as a waterfall. Noise-aware training is a mitigation
for the workloads: the simulator exports its noise model so networks can
be fine-tuned against it.

---

## 7. Benchmarking methodology and fair GPU comparisons

### 7.1 Rules

1. **Same workload and the same *task-level* accuracy.** Compare at equal
   top-1, perplexity or SNR, never at a nominal bit-width. A 6-bit analog
   MAC is not the same thing as FP8 (E4M3) or INT8.
2. **Same system boundary.** Report "core" and "system" numbers
   separately, and never mix them. The system number includes lasers,
   thermal control (and any TEC), converters, SRAM, memory traffic and
   digital periphery.
3. **Achieved, not peak.** Report achieved MAC/s on the workload and give
   utilisation for both sides.
4. **Dense, not sparse,** GPU figures. 1 MAC = 2 FLOP, and the convention
   is stated every time.
5. **Measured GPU baselines** wherever possible:
   - cuBLAS/TensorRT runs with NVML board energy.
   - A **WebGPU baseline on the author's own machine**, using the same
     kernels as the simulator, for an apples-to-apples browser
     comparison.
6. **Same era.** A photonic chip is at least 3–5 years from deployment, so
   it is compared against GPUs projected to that date as well as current
   ones. Trend assumption [A]: ~1.5–2× better energy per MAC per 2-year
   generation, mostly from lower precision.
7. **Distributions, not points.** Monte Carlo over the parameter ranges in
   Appendix A, with tornado charts showing which assumptions move the
   answer.
8. **Pre-register.** Metrics, workloads, boundaries and assumption sets
   are fixed before results come in (G1).
9. **Latency and throughput** are reported separately, with batch size
   stated.
10. **Area.** Report TMAC/s/mm² and pJ/MAC together. A photonic core that
    wins on energy but needs 10× the silicon is a different product.

### 7.2 Workload suite

| Class | Examples | Expected fit for thermal-weight MRR |
|---|---|---|
| GEMM / GEMV microbenchmarks | 64 → 8192 square, batch 1 → 10⁶ | Maps the reuse crossover (§5.6) |
| CNN inference | ResNet-50 conv layers (≈ 4 GMAC per 224² image; weights reused over thousands of positions) | Good |
| Transformer prefill | BERT-base, 1 B-parameter LLM prefill with long contexts | Good for weight GEMMs |
| Transformer decode | Same models at batch 1–64 | **Poor** (reload-bound) unless the model is resident |
| Attention QKᵀ, AV | Both operands dynamic | **Poor** for thermal weights. Would need fast weights (modulation-and-weight rings) |
| Signal processing | FIR filter banks, 64-point DFT as a real/imag split, microwave-photonic beamforming | **Best**, especially when the input is already optical (no input DAC) |

### 7.3 Reference points (datasheet dense peak ÷ TDP; context only)

| Device | Peak TMAC/s | pJ/MAC | × one 64×64 photonic core |
|---|---|---|---|
| H100 SXM, BF16 | 495 | 1.41 | 12× |
| H100 SXM, FP8 | 990 | 0.71 | 24× |
| B200, FP8 | 2 250 | 0.44 | 55× |
| B200, FP4 | 4 500 | 0.22 | 110× |
| Photonic core (nominal / optimistic / pessimistic) | 41 | 0.30 / 0.046 / 1.56 | 1× |

TDP ÷ peak is crude. Real workloads run below peak *and* below TDP, so the
harness measures both sides on actual workloads. B200 figures are from
third-party spec sheets that agree with each other, but should be checked
against NVIDIA's own datasheet.

### 7.4 Where photonics could plausibly win [S]

- When the input is already light (RF photonics, fibre, LIDAR, optical
  interconnect), so the input DAC disappears.
- Fixed operators with huge reuse, such as filters, beamformers and
  convolution stems.
- Small resident models where ns-scale latency matters: a single MVM
  takes ≈ 5.7 ns, against µs-scale GPU kernel launches.

The benchmark report states, for each workload, which regime applies.

---

## 8. Path to fabrication

### 8.1 Platform choice (decide in M0)

- **C-band 220 nm SOI.** The best-documented ring platform: academic PDKs
  are plentiful, and most published weight-bank work uses it.
- **O-band SOI with CW-WDM MSA lasers.** Commodity 16/32-λ sources from
  co-packaged optics, and lower silicon dispersion. Measured in frequency,
  ring FSRs are only a few % smaller for the same radius (higher n_g), so
  channel counts are similar.
- **SiN** only for a comb source (low loss, Kerr), never for the weight
  rings, because the thermo-optic coefficient is low.

### 8.2 Foundry and MPW options (verify current offerings, cadence and pricing with brokers)

| Use | Options | Notes |
|---|---|---|
| TC1: passive + heaters, quick turn | Applied Nanotools (e-beam, SiEPIC EBeam PDK), Cornerstone (UK), SiEPICfab | Months, not quarters. Order-of-magnitude cost for a small die: low thousands of USD/EUR [A, low confidence] |
| TC2: actives (ring modulators, Ge PDs, heaters) | imec iSiPP (via Europractice), AIM Photonics, AMF, Tower PH18, IHP | MPW slots tens of thousands of USD/EUR, 6–12 months turn-around [A, low confidence] |
| TC4: electronic-photonic co-integration | GlobalFoundries 45SPCLO / Fotonix (monolithic), or 3D integration with a CMOS die | Needed to drive thousands of heaters (§8.7) |
| Comb source | Ligentec, LioniX, imec SiN; or a purchased module | Optional |

### 8.3 PDK considerations

- **Parameter import.** PDK compact models (ring, heater, PD, coupler,
  waveguide loss, corner and statistical models) are mapped onto
  `phot-arch` parameters. Each one keeps its provenance tag: "PDK typical",
  "PDK corner" or "measured".
- **Layout from the same spec.** A GDSFactory script reads the same
  architecture file and generates the weight bank, splitter tree, test
  structures and pads. KLayout runs DRC with the foundry deck.
- **Independent circuit check.** SAX (open-source circuit simulator)
  cross-checks the simulator's TMM on the generated netlist.
- **Design rules that bite.** Minimum bend radius for small rings, heater
  and metal spacing (thermal crosstalk), undercut availability, pad pitch
  and count, and whether trimming services exist.

### 8.4 Test-chip sequence

| Chip | Content | Purpose |
|---|---|---|
| **TC1** | 8 λ × 4-row bank plus test structures: isolated rings with a gap sweep, rings with a radius sweep, heater-crosstalk arrays, spirals (loss). Plain and (if offered) undercut heaters | Fit Q, κ(λ), FSR, resonance-spread statistics, heater efficiency, crosstalk matrix, τ_th. Validate calibration algorithms. External lasers and PDs |
| **TC2** | 16–32 λ × 4–8 rows with ring modulators and balanced Ge PDs, at speed through an RFSoC-class FPGA board (multi-GS/s DAC/ADC) | MVM accuracy at 10 GBd, ENOB, energy of the optical parts, L3 eye-diagram validation |
| **TC3** | Novel mechanisms that pass G2: N1 root switch + dual-bank, N2 two-order banks with dispersion-engineered couplers, N3 live calibration | Confirm or refute the novelty claims in hardware |
| **TC4** | Co-integrated control electronics | Scale beyond pad limits |

### 8.5 Validation ladder

1. **Reproduce before innovating.** The simulator reproduces published
   weight-bank behaviour: Tait et al. 2016 crosstalk/channel-density
   trends, Zhang et al. 2022 precision versus control scheme (as a trend),
   and the energy estimates of Nahmias et al. 2020 (within 2× under their
   assumptions).
2. FDTD → TMM parameter tables agree within the tolerances in §4.10.
3. TC1 measurement → refit → **predict** held-out TC1 structures (e.g.
   other gaps) before seeing their data.
4. TC2: predicted vs. measured ENOB within ±0.5 bit, and component energy
   within ±30 %.
5. TC3: mechanism gains in hardware within ±30 % of simulated.

### 8.6 Lab and packaging needs

- Tunable laser, OSA or swept-source measurement, power meters, fibre
  array alignment, TEC stage.
- Multichannel heater drivers (commercial or custom DAC boards).
- For TC2: an RFSoC board, high-speed RF probes or wire-bonded PCB, and
  fibre-attach packaging.
- Most practical route: the collaborator's lab, a university photonics
  test facility, or a packaging partner.

### 8.7 The I/O wall

4096 heaters cannot be wire-bonded: realistic first chips stop at about
50–150 pads. TC1 and TC2 therefore stay small, and anything beyond needs
on-chip heater DACs (TC4).

The simulator models heater-DAC area and power from the start, so the
"full architecture" numbers already include it.

---

## 9. Milestones

The timeline assumes the author at ~1 FTE and the collaborator at ~0.1–0.2
FTE. Durations are estimates [A].

| # | Milestone | Deliverables | Acceptance criteria | Duration |
|---|---|---|---|---|
| **M0** | Foundations & prior art | Repo layout; `SimTime`/DES core; f64 TMM + CMT golden models; parameter-provenance system; **written prior-art review of N1–N4 with the collaborator**; platform decision (C vs O band) | Analytic tests pass (FSR, critical-coupling null, Q from fitted FWHM, energy conservation \|t\|² + \|d\|² + loss = 1); WebGPU GEMM throughput measured on the author's machine (sets the real slowdown factors) | 4 wk |
| **M1** | **First demo: one ring weight bank, one dot product** | 8-ring bank (configurable to 64), balanced PD, DAC/ADC/noise models; GPU path via the hand-written shim; spectrum view; time and energy ledger; §5.5 timeline printed live | Noiseless f64 vs. analytic error < 1e-9; GPU f32 vs. CPU f64 < 1e-5 relative; measured ENOB matches §4.4 formulas ±0.25 bit; bit-identical replay from a seed | 4 wk |
| **M2** | Physics depth | Thermal model (crosstalk, dynamics, drift), heater DACs, calibration loops, fab Monte Carlo, self-heating check, channel assignment | Calibration converges from random fab offsets in a bounded simulated time; precision waterfall (§6.4) produced; reproduces the Tait 2016 crosstalk trend | 6 wk |
| **M3** | Full core on WebGPU | 64-row core (32 λ × 2 orders), ISA + assembler, compiler/tiler, SRAM/HBM memory model, L1 and L2 modes on one clock, Perfetto export | L1 = L2 simulated time on the same program; L2 slowdown within 2× of the M0 prediction; 4096×4096 tiling reproduces §5.6 | 8 wk |
| **M4** | Benchmarks v1 (baseline B&W) | Pre-registered methodology (§7.1); workload suite; measured GPU and WebGPU baselines; Monte Carlo energy/precision report | **Gate G1** (below) | 6 wk |
| **M5** | Novel mechanisms | N1 (with just-in-time heating), N2 (with coupler-dispersion mismatch), N3 (pilot + sampled least squares), N4 (subcarrier readout) | **Gate G2** (below) | 12 wk |
| **M6** | Waveform fidelity & PDK | L3 CMT mode; MEEP/mode-solver parameter extraction; PDK import; GDSFactory + SAX cross-check | L3 confirms or bounds L2's quasi-static error (< 0.25 bit at 10 GBd for FWHM ≥ 25 GHz, or L2 is corrected); SAX vs. TMM < 1 % | 8 wk (in parallel with M5) |
| **M7** | TC1 tape-out | Layout from the spec, DRC-clean; test plan; measurement scripts already written against the simulator | Foundry accepts the submission | 8 wk design + fab wait (≈ 2–6 months) |
| **M8** | TC1 measurement & model correlation | Measured parameter set; refit; held-out prediction | **Gate G3** (below) | 8 wk after chips arrive |
| **M9** | TC2 (and TC3 if G2 passed) | Active chip(s) at speed | §8.5 steps 4–5 | ≈ 12 months |
| **M10** | Mesh (optional, any time after M4) | §10 | §10.5 | 6–8 wk |

**Gates:**

- **G1, end of M4: is the baseline credible?** The simulator reproduces
  at least two published results within stated tolerance, and the baseline
  report gives energy/throughput distributions for the workload suite.
  Outcome: narrow the target workloads to where the baseline is closest to
  winning.
- **G2, end of M5: is there a mechanism worth fabricating?** At least one
  of N1–N4 meets the §1.4 success criterion (≥ 1.5× on a primary metric,
  ≥ 6 effective bits in ≥ 80 % of fab instances).
  - If yes: TC3 is planned.
  - If no: TC1/TC2 still go ahead as *model-validation* chips, which are
    valuable on their own, and the novelty question is reopened with what
    was learned.
- **G3, end of M8: does the model predict silicon?** The statistical
  resonance-spread distribution matches, heater efficiency is within ±15
  %, thermal crosstalk within ±20 %, and weight-programming error within
  ±0.5 bit. If not, recalibrate and fix the models before committing to
  TC2.

**Calendar:** about 10 months to G2, about 16–20 months to TC1 data, and
about 30 months to an at-speed active chip. MPW calendars dominate the
second half, so align M7 with a published shuttle date.

---

## 10. Optional later phase: a decentralised mesh

### 10.1 Two different goals (keep them separate)

1. **Speed up the simulator** (wall-clock): spread work across browsers or
   nodes.
2. **Model a distributed photonic system** (simulated time): chips joined
   by optical links. Netcast-style weight broadcast over fibre is the
   hardware analogue.

Both share the DES. Only the first involves real networks.

### 10.2 Partitioning rule

A work unit should compute for **at least 10× (RTT + transfer time).**

| RTT | Uplink | Payload | Network time | Minimum compute per unit |
|---|---|---|---|---|
| 50 ms | 50 Mb/s | 4 KB (seeds, parameters, summary statistics) | 0.05 s | ≥ 0.5 s |
| 150 ms | 10 Mb/s | 4 KB | 0.15 s | ≥ 1.5 s |
| 50 ms | 50 Mb/s | 64 MB (a tile's activations) | 10.3 s | ≥ 103 s: **ship descriptions, not data** |

The best fits are embarrassingly parallel Monte Carlo fab instances and
design-space sweep points. Each node receives (seed, parameter vector) and
returns statistics measured in KB.

Partitioning a single large virtual chip by layer or core works only with
deep pipelining and small activations.

Fine-grained dot products across a WAN are never viable. A 64-wide dot
product takes 100 ps of simulated time and about 1 µs of wall-clock, so
the network is 10⁴–10⁵× slower.

### 10.3 Transport, determinism, trust

- **Transport.** Start with a star topology: a coordinator over WebSocket
  or WebTransport. Add WebRTC data channels (signalling + STUN/TURN) for a
  true mesh only if the coordinator becomes the bottleneck.
- **Determinism.** The counter-based RNG keyed by (experiment seed, unit
  id) lets any node recompute any unit exactly. Untrusted results are
  verified by redundant execution (1.1–2× replication, scaled to the
  node's trust history) and random spot checks.
- **Isolation.** Nodes run only the signed wasm bundle. No arbitrary code
  ships.

### 10.4 Simulated-time semantics across partitions

- **Conservative parallel DES.** Lookahead equals the minimum simulated
  latency across a partition boundary: ≥ 5 ns for a 1 m inter-chip fibre,
  or a full layer's latency for layer partitions.
- **Sync rounds cost wall-clock RTT,** so each round must cover a large
  lookahead window (e.g. ≥ 1000 layer evaluations).
- **Two network delays are reported separately:** simulated link latency,
  which is part of the hardware model, and real network delay, which is
  only part of the wall-clock cost.

### 10.5 Acceptance

A Monte Carlo sweep of 10⁴ fab instances runs on 10+ browser nodes with
≥ 0.7 parallel efficiency, and its results are bit-identical to a
single-node rerun of a random 1 % sample.

---

## 11. Risks

| # | Risk | Likelihood | Impact | Mitigation | Early warning |
|---|---|---|---|---|---|
| R1 | N1–N4 turn out to have precedent | Med–High | Med | M0 prior-art review; the plan's value (an honest simulator plus validated chips) doesn't depend on any single novelty | Review finds a direct match |
| R2 | Thermal static power dominates and kills the energy case | High | High | Trimming, undercut heaters, N1 just-in-time heating, N2; evaluate PCM for *resident* weights only | Nominal ledger stays > 0.25 pJ/MAC after M5 |
| R3 | Reload latency limits viable workloads to a niche | High | Med | Pick high-reuse workloads deliberately; N1; fast-weight variants (PN tuning) as an option | §5.6 crossover above the realistic batch sizes for target workloads |
| R4 | Fewer usable channels than 32 per order (crosstalk, fab spread) | Med | Med | Crosstalk-aware calibration; 40 GHz FWHM option; 16-λ fallback | TC1 spread exceeds the heater range |
| R5 | Below 6 effective bits at speed | Med | High | Range rescaling; noise-aware training using exported sim noise; lower baud | L2 waterfall shows a dominant term without a fix |
| R6 | Laser efficiency/RIN (comb) | Med | High | DFB arrays or CW-WDM modules; RIN spec in procurement | RIN > −140 dB/Hz per line |
| R7 | Model mismatch with silicon | High for absolute λ, Low for trends | Med | Built around calibration; G3 | Held-out prediction fails |
| R8 | WebGPU constraints (no f64, buffer caps, async readback, timestamp coarsening, browser differences) | Med | Low–Med | f64 phase reduction on CPU; tiling; DES lookahead; native wgpu harness as reference | Kernel results differ across browsers beyond tolerance |
| R9 | The no-wasm-bindgen constraint vs. a JS-only GPU API | Certain | Low | The ~200–300-line shim (§3.5); everything else stays in Rust | — |
| R10 | Wasm threads need COOP/COEP | Med | Low | Server headers; service-worker workaround; or single-threaded wasm with the GPU doing the heavy lifting | — |
| R11 | Scope vs. solo bandwidth | High | High | Gates G1–G3 with explicit kill/narrow decisions; mesh deferred | M3 slips > 50 % |
| R12 | Fab cost/schedule (MPW cadence, packaging) | Med | Med | Quick-turn e-beam TC1; align with shuttle dates; partner lab | Missed shuttle |
| R13 | Inter-order mismatch (N2) | Med–High | Med | Dispersion-engineered couplers; joint calibration; per-order digital gain | Residual > 0.5 LSB in Monte Carlo |
| R14 | Thermal disturbance from shadow-bank switching (N1) | Med | Med | Trenches and spacing; feed-forward heater compensation; ramped power-up | Active-bank weight error spikes at swap |
| R15 | **IP / disclosure.** This repository is public, so committing novel architecture details is a public disclosure; many jurisdictions (e.g. the EPO) have no grace period | Certain once pushed | Depends on intent | Decide on open publication vs. patents *before* detailing N1–N4 further in public; agree with the collaborator what of their technique is publishable | — |

---

## 12. Open questions and unknowns

1. **The collaborator's channel-separation technique.** Pilot tones?
   OFDM-like subcarriers? Compressive or spectral estimation? Which parts
   are published or protected? (Shapes N3 and N4.)
2. **Target workload class.** The plan proposes high-reuse inference plus
   signal processing. Is there an application, artistic or otherwise,
   where the input is already optical?
3. **Band:** C-band SOI (most literature) or O-band with CW-WDM lasers
   (commodity sources)?
4. **Weight technology:** thermal only, or also PN fast tuning, PCM, or
   post-fab trimming? It depends on foundry availability, and it moves the
   energy case more than anything else.
5. **Accuracy target:** which task-level metric, and how close to an FP16
   baseline counts as acceptable?
6. **Where the code lives:** a `photon/` member of this repository, or a
   new repository? (QuPU's eframe + wasm-bindgen stack would not be
   reused.)
7. **IP strategy and funding** for MPW runs (R12, R15).
8. **Is the mesh for simulation speed-up or for modelling distributed
   hardware?** (§10.1)
9. **Unknowns that only silicon resolves:** the real resonance-spread
   distribution, heater crosstalk, achievable τ_th versus efficiency, and
   per-line RIN of the chosen source.

---

## 13. Register of speculative claims

| Claim | Where | How it is tested |
|---|---|---|
| N1 is novel and improves achieved TMAC/s ≥ 1.3× without raising pJ/MAC | §1.4 | M0 prior-art review; M5; TC3 |
| N2 inter-order mismatch can be held under 0.5 LSB | §1.4 | M5 Monte Carlo; TC3 |
| N3 holds 6-bit weights under drift without stopping compute | §1.4 | M5; TC3 |
| N4 reduces system energy | §1.4 | M5 (physics may say no) |
| A single-FSR 64-λ bank at 10 GBd is feasible | §2.3 | Not planned; flagged as unlikely |
| 8-core, 328 TMAC/s "full architecture" | §2.4 | Only after G2/G3 |
| Optimistic 46 fJ/MAC | §6.1 | Requires trimming + undercut + 20 % WPE + 0.5 pJ converters, all to be confirmed |
| Photonics wins for optical-input, high-reuse and latency-critical workloads | §7.4 | M4/M5 benchmark report |
| The GPU efficiency trend of 1.5–2× per generation continues | §7.1 | Revisited at each report |

---

## 14. References

Compiled from memory and a brief search on 2026-10-17. **Verify
bibliographic details before citing externally.**

- A. N. Tait et al., "Broadcast and weight: an integrated network for scalable photonic spike processing," *J. Lightwave Technol.* 32(21), 2014.
- A. N. Tait et al., "Microring weight banks," *IEEE JSTQE* 22(6), 2016.
- A. N. Tait et al., "Neuromorphic photonic networks using silicon photonic weight banks," *Sci. Rep.* 7, 7430, 2017.
- A. N. Tait et al., "Feedback control for microring weight banks," *Opt. Express* 26(20), 26422, 2018.
- W. Zhang et al., "Silicon microring synapses enable photonic deep learning beyond 9-bit precision," *Optica* 9(5), 2022.
- Y. Shen et al., "Deep learning with coherent nanophotonic circuits," *Nat. Photonics* 11, 2017.
- J. Feldmann et al., "Parallel convolutional processing using an integrated photonic tensor core," *Nature* 589, 2021.
- X. Xu et al., "11 TOPS photonic convolutional accelerator for optical neural networks," *Nature* 589, 2021.
- R. Hamerly et al., "Large-scale optical neural networks based on photoelectric multiplication," *Phys. Rev. X* 9, 021032, 2019.
- A. Sludds et al., "Delocalized photonic deep learning on the internet's edge," *Science* 378, 2022.
- F. Ashtiani, A. J. Geers, F. Aflatouni, "An on-chip photonic deep neural network for image classification," *Nature* 606, 2022.
- S. Bandyopadhyay et al., "Single-chip photonic deep neural network with forward-only training," *Nat. Photonics* 18, 2024.
- S. R. Ahmed et al., "Universal photonic artificial intelligence acceleration," *Nature* 640, 2025.
- S. Hua et al., "An integrated large-scale photonic accelerator with ultralow latency," *Nature* 640, 2025.
- "Wavelength-parallel photonic tensor core based on multi-FSR microring resonator crossbar array," OFC 2023, W3G.4. <https://opg.optica.org/abstract.cfm?URI=OFC-2023-W3G.4>
- "Photonic neural networks based on integrated silicon microresonators," *Intelligent Computing* (review). <https://spj.science.org/doi/10.34133/icomputing.0067>
- "Compact, large-scale photonic neurons by modulation-and-weight microring resonators," arXiv:2505.11369, 2025.
- Shastri lab, time-domain weight sharing for photonic NNs, CLEO 2025.
- M. Padmaraju et al., "Wavelength locking and thermally stabilizing microring resonators using dithering signals," *J. Lightwave Technol.* 32(3), 2014.
- Simultaneous wavelength locking of a microring modulator array with a single monitor, *Opt. Express*, 2017 (TU/e).
- Politecnico di Milano, "Control and calibration recipes for photonic integrated circuits" (pilot-tone labelling, thermal-crosstalk cancellation).
- "Photonic fully-connected hybrid beamforming using microring weight banks," *Commun. Eng.*, 2025.
- "A case study on the performance metrics of integrated photonic computing," arXiv:2511.00186, 2025.
- M. A. Nahmias et al., "Photonic multiply-accumulate operations for neural networks," *IEEE JSTQE* 26(1), 2020.
- B. J. Shastri et al., "Photonics for artificial intelligence and neuromorphic computing," *Nat. Photonics* 15, 2021.
- W. Bogaerts et al., "Silicon microring resonators," *Laser Photonics Rev.* 6(1), 2012.
- B. E. Little et al., "Microring resonator channel dropping filters," *J. Lightwave Technol.* 15(6), 1997.
- H. A. Haus, *Waves and Fields in Optoelectronics*, Prentice-Hall, 1984.
- Z. Lu et al., "Performance prediction for silicon photonics integrated circuits with layout-dependent correlated manufacturing variability," *Opt. Express* 25(9), 2017.
- J. K. Salmon et al., "Parallel random numbers: as easy as 1, 2, 3," SC'11.
- B. Murmann, "ADC performance survey," Stanford (updated annually).
- M. Horowitz, "Computing's energy problem (and what we can do about it)," ISSCC 2014.
- CW-WDM MSA specification (O-band multi-wavelength laser grids).
- NVIDIA H100 and B200 datasheets; B200 dense figures cross-checked against third-party spec sheets (e.g. Lenovo Press LP2226).
- Tools: MEEP, GDSFactory, SAX, KLayout, SiEPIC EBeam PDK, wgpu, Perfetto.

---

## Appendix A. Assumption register (nominal values; ranges drive the Monte Carlo)

| Parameter | Nominal | Range | Source / confidence |
|---|---|---|---|
| Group index n_g (Si strip, C-band) | 4.3 | 4.1–4.5 | [K] typical. **Measured 4.186 ± 0.004** on 8 fabricated rings (openEBL 2024-10, `measured_rings.py`). Figures here still use 4.3, which understates FSR by ≈ 3 % |
| Ring FWHM at R = 3–5 µm, gap 70–90 nm | — | — | **Measured 27–45 GHz** (Q_L 4.3 k–7.1 k): already inside the 25–40 GHz design window. R = 3 µm gives FSR 3.82 THz → 38 channels at 100 GHz |
| dλ/dT | 70 pm/K | 60–85 | [K] |
| Ring radius | 3.3 µm | 2–5 | Design |
| Ring FWHM | 25 GHz | 25–40 | Design, from signal bandwidth |
| Channel spacing | 100 GHz | 80–200 | Design |
| Symbol rate | 10 GBd | 5–25 | Brief |
| Heater efficiency | 0.25 nm/mW | 0.1–1.0 (undercut) | [A] PDK-dependent |
| Mean tuning per ring | 0.4 nm | 0.1 (trimmed) – 1.0 | [A] fab-dependent |
| τ_th | 5 µs | 2–100 | [A] trades against efficiency |
| Thermal crosstalk, nearest neighbour | 10 % | 2–20 % | [A] layout-dependent |
| TIA noise density | 15 pA/√Hz | 8–25 | [A] |
| PD responsivity | 1.0 A/W | 0.8–1.1 | [K] Ge PD typical |
| RIN per line | −150 dB/Hz | −160 (DFB) – −130 (comb) | [A] source-dependent |
| P_det per row | 200 µW | 100–400 | Design (sets ENOB) |
| Path loss (excl. split) | 13 dB | 10–15 | [A] |
| Laser WPE | 10 % | 2 % (comb + EDFA) – 20 % | [A] |
| DAC energy | 1.5 pJ/sample | 0.5–3 | [A] node-dependent |
| ADC energy | 2.0 pJ/conv | 0.5–5 | [A] Murmann-survey scale |
| TIA energy | 0.5 pJ/sample | 0.2–1 | [A] |
| SRAM energy | 0.05 pJ/bit | 0.025–0.1 | [A] advanced node |
| HBM energy | 3 pJ/bit | 2.5–4 | [A] |
| Width σ, die-to-die / local | 4 nm / 1 nm | 3–5 / 0.5–2 | [A] foundry-dependent |
| dλ/dw | 0.8 nm/nm | 0.5–1 | [K] for 500 nm strip |
| Browser WebGPU effective fp32 | 2 TFLOP/s | 1–10 | Measured in M0 |
