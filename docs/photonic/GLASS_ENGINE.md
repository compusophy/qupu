# Glass engine: free-space simulator and design search

This doc covers the free-space matrix engine from
[FIRST_PRINCIPLES.md](FIRST_PRINCIPLES.md) §4. An emitter array shows a
vector, a patterned glass or LCD mask holds the weights, and detectors read
the sums. It is now modelled from the optics up (`studio/src/freespace.rs`).

A hill climb then searches its geometry and architecture
(`studio/src/geometry.rs`, `studio/src/search.rs`). Run it with:

```sh
cd studio
cargo run --release -- search 1600    # ≈ 50 s on 4 cores
cargo run --release -- tabletop       # predictions for the tabletop rig (TABLETOP.md)
```

Everything below comes from those two commands. Where a number rests on an
assumption, the assumption is named. None of the optics constants is fitted
to hardware yet; fitting them is what the [tabletop rig](TABLETOP.md) is for.

## What the simulator models

Each design is scored by building its physics and then running real dot
products through it. The workload is DCT rows as weights, random inputs in
[0, 1], and fresh noise and fabrication draws per sample. From those runs
it measures:

- **ENOB**, log₂(2 / (√12 · rms error)) on (1/n)·Σwⱼxⱼ
- **cost and power**, from `econ.rs`

| Effect | Model | Status |
|---|---|---|
| Imaging blur | σ = 0.21 λ/NA + aberration ∝ NA²·field² (4 µm at NA 0.1, 10 mm) | Diffraction term: physics. Aberration constant: assumed. |
| Symmetric relay | Odd aberrations cancel; aberration ×0.25 | Assumed factor |
| Pixel crosstalk | Gaussian spot over the neighbours' apertures (erf box fractions) | Physics |
| Mask diffraction | Periodic lattice → discrete ghosts at Δ = f λ / pitch; aperiodic lattice → diffuse halo. Zero-order share = area fill. | Physics, scalar approximation |
| Lenslet spill (optical replication) | 0.4 % into adjacent tiles | Assumed |
| Summing optics | 25 mm focal length, 0.1× demagnification, detector radius 0.35 pitch, blur 0.18 | Assumed |
| Drive skew | 7 ps/mm across a row bus; an H-tree has none but costs +25 % drive energy | Physics, rough |
| Receiver noise | TIA 15 pA/√Hz plus shot noise over baud/2, at detector power set by the `light` gene | Physics |
| Quantisation | 8-bit mask, 8-bit ADC, 500:1 mask contrast | Assumed parts |
| Calibration | Couplings measured with 0.2 % gain error and 5 % crosstalk error, then unmixed digitally | Assumed |
| Cost and energy | `econ.rs`: volume wafer and process costs, converter energy (rising above 2.5 GBd), emitter pitch floor (20 µm above 1 GBd) | Model, see FIRST_PRINCIPLES.md |

`Terms` switches each effect off in turn, which gives an error budget.
Turning crosstalk off means perfect focus: the spread light goes to the
emitter's own pixel, rather than being discarded.

## The genome

| Gene | Options |
|---|---|
| Pixel lattice (symmetry group) | Square p4m, hexagonal p6m, Fibonacci sunflower (Vogel, golden angle), golden R2 quasi-lattice (plastic number) |
| Tile lattice | Same four options |
| Index ordering | Native, or Hilbert curve (fractal; keeps vector neighbours spatial neighbours) |
| Pitch | 5–80 µm |
| Fill | 0.4–0.95 |
| NA | 0.01–0.35 |
| Relay | Symmetric or not |
| Replication | Optical (lenslets copy one emitter image to every tile) or electronic (the emitter array shows a copy per tile) |
| Broadcast | ×1–8 (one emitter array lights k engines: an optical hyperedge) |
| Drive | Row bus or H-tree |
| Symbol rate | 10 MBd – 5 GBd |
| Signed weights | Differential or offset |
| Light | ×0.25–16 detector power relative to the 6-bit receiver design point |
| Size | 64–1024 inputs and outputs |

**Fitness** = log₁₀(TMAC/s per $) + log₁₀(TMAC/s per W), minus 3 per bit
below a 6-bit floor.

**Search method.**
- (1+8) evolution strategy with restarts after 12 stalled generations.
- A 3-objective Pareto archive: bits, TMAC/s per $, TMAC/s per W.
- One constrained climb per pixel-lattice symmetry group.
- Single-gene ablation of the winner.

## Results (1,600 + 4 × 800 designs, 50 s)

**Reference.** B200 at FP4: 0.70 TMAC/s per $ of COGS, 222 fJ/MAC, fitness
0.50.

| Design | Bits | TMAC/s per $ | fJ/MAC | $ COGS | W | Fitness |
|---|---|---|---|---|---|---|
| Starting point (hand-designed glass engine from `econ`) | 6.57 | 11.5 | 8.8 | 228 | 23.0 | 3.12 |
| Unconstrained climb, 1,605 designs | 6.05 | 56.2 | 6.9 | 93 | 36.2 | 3.91 |
| **Best overall** (hex-constrained climb) | **6.07** | **59.3** | **4.2** | **88** | **22.2** | **4.15** |

The best design:
- 1024×1024, hex pixels, hex tiles
- pitch 8.5 µm, fill 0.95, NA 0.132, no symmetric relay
- optical replication, broadcast ×8, row-bus drive
- 5 GBd, offset-signed, light ×1.8

**On paper, that is 84× a B200 per dollar and 52× per joule, at 6 bits.**

**Held out.** Re-measured on 4 fresh seeds, the winner scores 6.05, 5.92,
6.02 and 6.03 bits. The search keeps lucky noise draws at the floor, but
the effect is under 0.1 bit here.

### Error budget of the winner

| Source switched off | Bits | Change |
|---|---|---|
| None | 6.05 | |
| Receiver noise | 6.79 | **+0.74** |
| Mask and ADC quantisation | 6.11 | +0.05 |
| Blur crosstalk | 6.11 | +0.06 |
| Calibration error | 6.01 | −0.04 (noise) |
| Ghosts, lenslet spill, drive skew | 6.05 | ≤ 0.01 |

Receiver noise is the binding constraint, and the search sets the light to
just reach 6 bits. Before light became a gene, no design reached 6 bits.
The best had 5.81, and noise was its only real error. So **precision is bought with
photons, and the optics are not the limit at this design point.**

### Symmetry groups (each climb started from the winner with its pixel lattice swapped)

| Pixel lattice | Bits | TMAC/s per $ | fJ/MAC | Field | Fitness |
|---|---|---|---|---|---|
| Hexagonal (p6m) | 6.07 | 59.3 | 4.2 | 10.7 mm | **4.15** |
| Square (p4m) | 6.22 | 53.0 | 4.0 | 19.2 mm | 4.13 |
| Golden R2 quasi-lattice | 6.12 | 36.1 | 4.2 | 28.9 mm | 3.93 |
| Fibonacci sunflower | 6.00 | 54.6 | 6.7 | 14.2 mm | 3.91 |

**Periodic lattices win, hex by a hair.** In this regime the diffraction
ghosts from periodic lattices are calibrated out (≤ 0.01 bit in the error
budget). What matters is packing: field size sets aberration and lens cost.

Hex is the densest packing at a given minimum spacing. The aperiodic sets
need more field for the same spacing: the R2 set needs 2.7× the field of
hex.

**Where aperiodic order can win.** The tabletop predictions show a regime
where the aperiodic sunflower wins: small NA, ghost-limited, frame-averaged.
There, discrete ghosts landing on specific neighbours cost more than a
diffuse halo. See TABLETOP.md, prediction 3. The prediction is that the
ranking of symmetry groups flips with the regime.

### Ablation: one gene of the winner changed

| Change | Δ fitness | Bits | Why |
|---|---|---|---|
| NA × 0.5 | −8.4 | 3.19 | Diffraction blur doubles |
| Pixels → quasi | −7.5 | 3.52 | Same minimum spacing needs 2.7× the field, so aberration grows |
| Pitch × 0.5 | −7.3 | 3.56 | Crosstalk |
| Replication → electronic | −6.9 | 4.45 | 45× the energy per MAC (191 vs 4.2 fJ) and 4× less compute per $; optical copying is essential |
| Tiles → quasi | −4.6 | 4.48 | |
| Pixels → sunflower | −2.5 | 5.16 | |
| NA × 2 | −2.4 | 5.20 | Aberration ∝ NA² |
| Light × 0.5 | −1.6 | 5.44 | Receiver noise |
| Symbol rate × 0.25 | −0.6 | 6.11 | Cheaper converters, but 4× less work per $ |
| Signed → differential | −0.4 | 6.87 | Twice the tiles; more bits, worse per $ and per W |
| Broadcast × 1, 2 or 4 | −0.03 to −0.11 | 6.07 | Cheaper per $ at ×1 but 40 % worse per J |
| Symmetric relay, H-tree, Hilbert order | ≈ 0 | ≈ 6.0–6.2 | No effect at this design point |

**What the search says, plainly**

- **Fan-out by optics, not electronics.** Optical replication is the
  architecture: switching it to electronic costs 45× in energy per MAC.
- **Light is a design variable.** Halving the detector power costs 0.6 bits.
  Doubling it buys 0.3 bits, as the other errors start to show. The
  receiver, not the glass, sets the precision of a fast engine.
- **Geometry matters through packing and NA.** Symmetry group matters
  through field size: hex ≈ square > sunflower > quasi at speed. Fractal
  ordering and H-tree drive don't matter at 1024 inputs, 8.5 µm and 5 GBd:
  skew across an 11 mm field is about 0.08 ns, against 0.2 ns symbols. They
  would matter at larger arrays or faster symbols.
- **Bounds hit:**
  - The winner sits at the 5 GBd and broadcast ×8 upper limits. The
    converter energy model above 2.5 GBd and the cost of 8-way broadcast
    optics decide where it would stop.
  - Size runs into the 1024 cap too.

  These limits are choices in the genome, not physics.

## Caveats

- The optics constants (aberration, summing optics, spill, calibration
  errors) are assumptions until the tabletop rig fits them.
- The economics come from `econ.rs`:
  - volume wafer costs and published GPU figures
  - an assembly share of about half the engine's cost
  - no NRE

  See FIRST_PRINCIPLES.md.
- 5 GBd emitter and detector arrays with 6-bit analog levels and 8.5 µm
  pitch are not catalogue parts. That is the speed path's risk, not the
  simulator's.
- ENOB is measured on DCT-row weights, a structured workload. Equal task
  accuracy at about 6 analog bits vs FP4 is assumed, not shown.
