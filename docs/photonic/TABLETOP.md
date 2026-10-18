# Tabletop glass engine: parts list and predictions

A matrix-vector multiplier built from two LCD panels, two camera lenses
and a camera. It costs about $450–1,000 and needs no foundry. It is the
first physical test of the free-space model in `studio/src/freespace.rs`.

**What it tests**
- precision (effective bits) of an incoherent free-space multiply-and-sum
- crosstalk, relay aberrations, diffraction from the mask lattice
- calibration from measured couplings
- the simulator's specific predictions, listed below

**What it does not test**
- speed: the LCDs refresh at about 50 Hz
- energy: the camera and panels draw watts
- cost at volume

The speed path in [FIRST_PRINCIPLES.md](FIRST_PRINCIPLES.md) §5 is there
to test those.

```
 green LED    LCD A: inputs        two 50 mm lenses        LCD B: weights      camera + lens
 backlight    (copies of x,     face to face, f/4         (+ and − tile       (each tile lands
 + diffuser    one per tile)    1:1 symmetric relay        per output)          on a pixel block)
    ║║║  ───►  ▓▓▓▓▓▓▓▓  ───►  (|  |)═══(|  |)  ───►  ░▒▓░▒▓░▒  ───►   [■]──USB──► computer
                47.25 µm                                47.25 µm                    (drives A and B
                                                                                    over HDMI)
```

## How it computes

1. **LCD A** shows 512 copies of the input vector x, one per tile, each as a
   16×16 block of gray levels. This is "electronic replication" in the
   simulator.
2. **The relay** images LCD A onto LCD B pixel for pixel. Both panels are the
   same model, so a 1:1 relay maps every input pixel onto exactly one weight
   pixel.
3. **LCD B** holds the weight matrix W. Each output row gets two tiles: one
   holding the positive weights w⁺ and one holding the negative weights w⁻.
   Each pixel transmits xⱼ·wᵢⱼ.
4. **The camera** images LCD B. All the light from a tile lands on one block
   of camera pixels, and the block's total is that row's dot product.
5. **Signed weights are differential:** each output is the positive tile
   minus the negative tile.
6. **Exposure is set per workload.** The camera exposure is chosen so that
   the brightest tile the loaded matrix can produce, with all inputs on,
   sits near saturation.

Steps 5 and 6 together are worth 1–6 bits on real workloads (prediction 0).
The obvious alternative wastes most of that: an offset scheme storing
(w+1)/2, with exposure fixed for an all-open tile.

**Why the camera sum still counts as a measurement.** Summing a block of
camera pixels in software is equivalent to one large detector, plus extra
read noise. At about 2–3 e⁻ per pixel, that extra noise is far below the
shot noise. So it doesn't change the precision measurement. It does change
the energy, which this rig doesn't claim to measure. For a fully optical
sum, see the line-sensor option below.

**Why two identical lenses face to face.** In a symmetric relay working at
1:1, the odd aberrations (coma, distortion, lateral colour) cancel.

This is the `symmetric` gene in the search. The rig can test it directly:
take one lens out and compare (prediction 1).

## Bill of materials

Prices are approximate single-unit prices seen in listings in October 2026.
They vary widely by seller, so check them before buying. Part numbers are
examples, not requirements.

| # | Part | Role | Example | Approx. price |
|---|---|---|---|---|
| 1 | 5.5″ 2K mono LCD with HDMI-to-MIPI driver board | LCD A, inputs | Sharp LS055R1SX03 or SX04 kit: 2560×1440, 47.25 µm pitch, sold for DIY resin printers | $100–185 |
| 2 | The same kit again | LCD B, weights | Identical pitch, so the 1:1 relay is pixel to pixel | $100–185 |
| 3 | Green LED panel (520–530 nm) with diffuser | Backlight for LCD A | Any uniform LED light pad | $10–30 |
| 4 | Two identical 50 mm f/1.8 lenses with **manual aperture rings** | 1:1 symmetric relay | Used M42 or Nikon AI 50 mm primes. Avoid electronic-aperture mounts, which can't be stopped down without a camera body. | $30–130 each |
| 5 | Male-to-male filter-thread coupler | Mounts the lenses front to front | Sized to the lenses' filter thread | $5–15 |
| 6 | Mono global-shutter USB camera | Detector array | OV9281 class: 1280×800, 3 µm pixels, up to 120 fps | $40–90 |
| 7 | C or CS-mount lens, 25–35 mm | Images LCD B onto the camera | Any machine-vision lens covering the sensor | $20–60 |
| 8 | Raspberry Pi 5 with power supply and cables | Drives both panels and reads the camera | The Pi 5 has two HDMI outputs and USB 3; a laptop with two display outputs also works | $80–100 |
| 9 | 2020 aluminium extrusion (≈ 500 mm), 3D-printed holders, tip/tilt screws | Optical rail | An optical breadboard with posts ($200–400) is stiffer | $30–60 |
| 10 | Black flocking or a matte enclosure | Stray-light control | | $10–20 |
| | **Total** | | | **≈ $455–1,005** |

### Practical notes

- **Driver-board refresh rate.** Some driver boards for these panels run only
  at 50 Hz. Check the panel suffix and the board.
- **Backlights.** The panels are sold for resin printers, which light them
  with 405 nm UV and remove the stock backlight. They work as ordinary
  monochrome LCDs in visible light. LCD B needs no backlight, because the
  relayed image of LCD A is its light.
- **Field.** 512 tiles of 16×16 pixels at 47.25 µm, with the gaps the
  simulator leaves between tiles, cover about 26 mm square (37 mm
  diagonal). That is inside a full-frame lens's 43 mm image circle.
- **LCD transmission.** Each panel transmits roughly 5–10 % at full open.
  That suits the rig, because the camera needs very little light.

## Calibration and measurement procedure

1. **Dark and flat frames.**
   - Dark frame: both panels black.
   - Flat frame: both panels white.
2. **Per-panel gray-level curve.** Step one panel's gray level with the other
   panel white, and record transmission against level for each panel. The
   result is the 8-bit look-up table that `solve_mask` assumes is linear
   after correction.
3. **Couplings.** Poke single input pixels and single weight pixels, and
   record where their light lands. This measures the coupling matrix: own
   pixel, neighbours, and diffraction ghosts in neighbouring tiles. The
   simulator models this measurement with 0.2 % error on gains and 5 % on
   crosstalk terms. The rig measures the real errors.
4. **Test.**
   - Load DCT rows as W (the simulator's scoring workload). Show random x in
     [0, 1].
   - Unmix with the measured couplings.
   - Compute ENOB = log₂(2 / (√12 · rms error)) on the normalised dot
     product (1/n)·Σ wⱼxⱼ. That is the same definition the simulator uses.
   - For image workloads, also report bits on the workload's own output
     span, (1/n)·Σ|wⱼ|. That is what prediction 0 uses, and what decides
     whether a blurred or edge-filtered image comes out clean.
5. **Repeat across configurations.** Average N frames, change the f-stop,
   swap the relay for one lens, and redraw the lattice. Compare each result
   with the predictions below.

## Predictions to check

Run `cargo run --release -- tabletop` in `studio/`. Results are ENOB as the
mean ± sd of 4 simulated runs.

The model's lens aberration is a generic constant, not a model of any
particular 50 mm lens. That is one of the things the rig will fit.

**0. Signed encoding and exposure.** This is the largest effect the
simulator found for this rig. The table gives bits on each workload's own
output range. Inputs are the demo's scrolling text, orbits, wheel and face.

| Encoding | Blur | Low-pass | Edges | DCT spectrum |
|---|---|---|---|---|
| Offset tiles, fixed exposure | 2.08 | 1.03 | 0.18 | 4.27 |
| Offset tiles, exposure per workload | 2.92 | 2.00 | 1.17 | 4.68 |
| Differential tiles, fixed exposure | 2.80 | 1.70 | 0.89 | 4.79 |
| **Differential tiles, exposure per workload (as built)** | **5.55** | **5.91** | **6.09** | **5.77** |

**Mechanism.** A fan-in engine's noise is a fixed fraction of the
detector's full scale. A blur or band-pass row uses only a few percent of
an all-open tile's light.

- **Offset encoding** stores a bias of half the light in every pixel. That
  caps how far the exposure can rise, and with large blur its crosstalk
  errors add coherently.
- **Differential tiles** carry only the signal, so the exposure can rise
  until the workload's brightest tile fills the detector.

Full-scale ENOB (predictions 1–3) hides this. Measured on the scoring
workload, the as-built rig is at 6.95 bits either way.

**1. Relay symmetry.** Same field, same lenses.

| Relay | Predicted ENOB | Edge blur |
|---|---|---|
| Two lenses face to face, f/4 (as built) | 6.95 ± 0.03 | 2.2 µm |
| Two lenses face to face, f/1.8 | 6.66 ± 0.07 | 6.7 µm |
| One lens, f/8 | 7.01 ± 0.03 | 3.0 µm |
| One lens, f/4 | 6.80 ± 0.07 | 6.0 µm |
| One lens, f/2.8 | 6.31 ± 0.09 | 11.1 µm |
| One lens, f/1.8 | 5.68 ± 0.08 | 25.8 µm |

The claim being tested: opened to f/1.8, the symmetric pair loses about
0.3 bit and a single lens about 1.3 bits. Stopped down to f/8, a single
lens matches the pair.

**2. What limits the as-built rig.**

| Source switched off | ENOB | Change |
|---|---|---|
| None | 6.95 | |
| Camera noise | 8.41 | +1.47 |
| Mask and ADC quantisation | 7.05 | +0.10 |
| Diffraction ghosts | 6.97 | +0.02 |
| Calibration error | 6.96 | +0.01 |
| Blur crosstalk | 6.95 | 0.00 |

Camera noise dominates. Each 4× more frames averaged halves it, which is
worth up to 1 bit while noise still dominates, saturating near 8.4 bits.
Where it actually saturates measures the calibration and quantisation
floor.

**3. Geometry is software on this rig.** Draw each weight as a 2×2-pixel
cell (94.5 µm) and place the cells on a lattice. Sites are rounded to the
47 µm pixel grid in practice.

| Lattice (symmetry group) | Ghost-limited: one lens at f/8, 64 frames averaged | Aberration-limited: one lens at f/4 |
|---|---|---|
| Square (p4m) | **8.22 ± 0.07** | 6.32 ± 0.06 |
| Hexagonal (p6m) | 7.76 ± 0.25 | **6.41 ± 0.07** |
| Fibonacci sunflower | 7.97 ± 0.06 | 5.91 ± 0.13 |
| Golden quasi-lattice (R2) | 6.31 ± 0.11 | 5.34 ± 0.34 |

The prediction: square leads or ties in both regimes, while hexagonal and
the sunflower swap places.

**Mechanism, ghost-limited.** Periodic lattices diffract into discrete
orders, sharp ghosts that land on specific neighbouring tiles. Calibration
leaves a 5 % residual on each ghost. An aperiodic sunflower spreads the same
light into a diffuse halo, whose errors average out, so it beats
hexagonal.

Hexagonal suffers most among the periodic lattices. Its round equal-area
cells pass only half the light into the zero order, and it has six orders
to square's four. Square's 64 % zero order keeps it ahead of both.

**Mechanism, aberration-limited.** Blur grows with field², so the densest
packing at a given minimum spacing wins. Hexagonal is the densest circle
packing. The sunflower and especially the R2 set need more field for the
same spacing.

The search of the fast design ([GLASS_ENGINE.md](GLASS_ENGINE.md)) sits in
the second regime, which is why it picks hex.

## Option: a fully optical sum on a line sensor

This follows Goodman, Dias & Woody, *Opt. Lett.* 2, 1 (1978).

1. Draw each input xⱼ as a full vertical stripe on LCD A.
2. LCD B holds W.
3. A cylindrical lens collapses each row of LCD B onto one pixel of a
   linear CCD.

That gives one physical detector per output and no software sum.

**Parts:**
- A TCD1304 board: 3648 pixels of 8×200 µm. About $100–150 as a ready USB
  board, or about $23 for the bare chip plus a microcontroller driver.
- A cylindrical lens: $20–60.

The model for this geometry is not in the simulator yet.

## Precedent

Wang et al., *Nature Communications* 13, 123 (2022), "An optical neural
network using less than 1 photon per multiplication", used a similar
architecture.

| Function | What Wang et al. used | This rig |
|---|---|---|
| Inputs | A phone's OLED display (Google Pixel 2016) | LCD A |
| Imaging | A zoom lens (Navitar Resolve 4K) at ≈0.16× | Two used camera lenses |
| Weights | A reflective phase-only liquid-crystal SLM (Meadowlark P1920, 9.2 µm pixels), used for intensity in double pass | A printer LCD |
| Summing | A 4× objective onto a single-pixel photon counter (Hamamatsu MPPC C13366-3050GA) | A camera |

Their results:
- up to 711×711 = 505,521 multiplications in parallel
- dot-product error down to about 0.2 % at ≥ 2 detected photons per
  multiplication
- multiplication at ≥ 10 MHz with under one photon each
- 99 % MNIST accuracy at about 3 photons per multiplication

This rig swaps the research-grade SLM and zoom lens for parts sold to
hobbyists. Whether those parts are good enough is exactly what it measures.

## Sources

- Wang, Ma, Wright, Onodera, Richard, McMahon, [An optical neural network using less than 1 photon per multiplication](https://arxiv.org/abs/2104.13467), *Nat. Commun.* 13, 123 (2022)
- 5.5″ 2K mono LCD and HDMI driver kits: [Hackaday.io DIY resin printer log](https://hackaday.io/project/28331/log/115360); seller listings (panel LS055R1SX03, 2560×1440, 0.04725 mm pitch, HDMI-to-MIPI)
- 6.6″ 4K mono LCD spare (alternative; ≈ $72, 4096×2560): [printer-hub listing](https://printer-hub.ru/en/components/anycubic-photon-mono-2-lcd-screen)
- TCD1304 prices: [LCSC TCD1304DG](https://fat.lcsc.com/product-image/C5348961.html); eBay USB board listings
- Lens arrays (if a lenslet-per-tile sum is wanted later; ≈ $900 per 10×10 mm part): [Edmund Optics 500 µm pitch microlens array](https://www.edmundoptics.com/p/microlens-array-10-x-10mm-500mum-pitch-1deg-divergence/19192/)
- Goodman, Dias & Woody, "Fully parallel, high-speed incoherent optical method for performing discrete Fourier transforms", *Opt. Lett.* 2, 1–3 (1978)
