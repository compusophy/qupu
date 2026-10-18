# First principles: what a multiply costs

*2026-10-18 · numbers from `cd studio && cargo run --release -- econ`
(`studio/src/econ.rs`). Change an assumption there to see what moves.*

The question is whether any optical architecture beats Nvidia, AMD and
Intel on compute per dollar and per joule, counted from raw materials to
the electricity bill. The method follows the SpaceX reusable-rocket
argument:

1. Find what the product costs relative to its raw materials.
2. Find the dominant term in that gap.
3. Pick the architecture that removes it.

## 1. Raw materials are not the cost

| System | Cost to build | Raw materials | Idiot index (cost ÷ raw materials) |
|---|---|---|---|
| NVIDIA H100 | $3,320 | ≈ $1.7 | ≈ 2,000 |
| NVIDIA B200 | $6,400 | ≈ $2.2 | ≈ 3,000 |
| AMD MI300X | $5,300 | ≈ $2.4 | ≈ 2,300 |
| Intel Gaudi 3 | $6,500 | ≈ $1.9 | ≈ 3,400 |

Raw materials are counted as about 9 000 mm² of silicon at $50/kg, plus
package metals.

An AI accelerator is roughly three thousand times more expensive than its
atoms. Where that cost goes:

- **Logic silicon is a small share:** about $850 of a B200.
- **HBM memory:** about $2,900.
- **Advanced packaging and test:** about $2,650.

Sources: Silicon Analysts and Epoch AI, ±20 %.

So "compute per dollar of raw material" is the wrong denominator; every
design wins on it. The useful denominators are:

- what it costs to process and assemble,
- what it costs to run,
- how much energy it takes to make.

**The $14k/mm² problem is a prototype price, not a production price.** A
silicon photonics wafer at volume costs about $3–5k for roughly 60 000 mm²
of usable area: **≈ $0.07/mm²**. The multi-project shuttle price is about
250 000× that, because it carries the one-off cost of making a wafer at
all. Prototypes will cost thousands; products will not.

## 2. One identity ranks every optical architecture

Each symbol, an optical matrix engine takes in `inputs` values and puts out
`outputs` values, and each output sums `fan_in` products. Converters are
paid once per value, so

```
energy per MAC ≈ E_in / fan_out + E_out / fan_in + light + static + memory
```

Each term:

- **E_in:** DAC plus driver, about 1–2 pJ per sample.
- **E_out:** TIA plus ADC, about 1.5–2.5 pJ per sample.
- **Light:** set by how many photons each detector needs for about 6 bits.

The arithmetic in the light is free. **What you pay for is converting
values in and out, and fan-in is what amortises it.**

- **WDM microring banks:** fan-in is capped by how many 100 GHz channels fit
  in one free spectral range. That is 38 for the 3 µm rings we measured.
  The converter floor alone is about **100 fJ/MAC**.
- **Free-space engines:** light from an emitter array passes through a weight
  mask onto a detector array, so fan-in can be the whole row (1024). The
  floor drops to about **2.4 fJ/MAC**.

## 3. The comparison

| System | Precision | PMAC/s | W | fJ/MAC | Cost to build | TMAC/s per $ | 3-year TCO | PMAC/s per $1k TCO |
|---|---|---|---|---|---|---|---|---|
| H100 | FP8 | 0.99 | 700 | 707 | $3,320 | 0.30 | $5,233 | 0.19 |
| B200 | FP8 | 2.25 | 1000 | 444 | $6,400 | 0.35 | $9,133 | 0.25 |
| B200 | FP4 | 4.50 | 1000 | 222 | $6,400 | 0.70 | $9,133 | 0.49 |
| MI300X | FP8 | 1.31 | 750 | 574 | $5,300 | 0.25 | $7,350 | 0.18 |
| Gaudi 3 | FP8 | 0.92 | 900 | 981 | $6,500 | 0.14 | $8,960 | 0.10 |
| **Ring core module** (the plan's core) | ≈5.5 bit | 0.04 | 13 | 325 | $179 | 0.23 | $215 | 0.19 |
| **Ring wafer** (per-wafer package) [S] | ≈5.5 bit | 123 | 27 664 | 225 | $30,324 | 4.1 | $105,933 | 1.2 |
| **Glass engine, 2.5 GBd** [S] | ≈5 bit | 2.62 | 14 | 5 | $103 | 25 | $142 | 18.5 |
| **Glass engine, 125 MHz** | ≈5–8 bit | 0.13 | 2 | 17 | $103 | 1.3 | $109 | 1.2 |

Assumptions: electricity $0.08/kWh, PUE 1.3, 3 years at full power. GPU
throughput is the dense datasheet peak. The ≈5–6 bit analog results are
**not** FP8, and equal accuracy on real tasks is unproven; that is the
first thing to test.

### What the numbers say

**Ring core module (the plan's design): no win.**
- Cost per MAC/s and energy per MAC are roughly level with an H100 or B200.
- Fibre attach and packaging are 56 % of cost, and lasers 29 %.
- Silicon is about 1 %.
- Building this core as a product would put real effort into a parity
  result.

**Ring wafer: wins on cost, not on energy.**
- Never dice the wafer. Bond the photonic wafer to a 28 nm CMOS wafer the way
  stacked image sensors are made, with one light input and one package per
  wafer, and non-volatile weights.
- That gives 4–11× a B200 on MAC/s per dollar, *if* lasers reach about
  $20 per optical watt.
- Energy stays level with FP4 GPUs, because converters dominate (52 % of
  power) at fan-in 32.
- It also holds only about 6 M weights per wafer.

**Glass engine: the only architecture here that wins on both, by a wide
margin.**
- The parts are an emitter array (display-like), a grayscale glass weight
  mask, and a detector array with CMOS readout (camera-like).
- **The energy win survives at any speed:** 5–17 fJ/MAC against 222 for
  FP4, because fan-in is 1024.
- **The cost win depends on speed.** At 2.5 GBd it is about 36× a B200 per
  dollar. At 125 MHz, the class of speed a commercial optical vector-matrix
  processor already claimed in 2003 (Lenslet EnLight, at 256×256), it is
  about 1.8×.
- Its idiot index is about 70, against about 3,000 for a GPU.
- The deciding terms are assembly and alignment (49 % of cost) and symbol
  rate.

### What decides it (sensitivity, from the tool)

- **Ring wafer:** laser $/W (0.2× → 6.8 TMAC/s/$; 5× → 1.3) and converter
  energy (123–378 fJ/MAC).
- **Glass engine:**
  - symbol rate: 0.5× → 12.7 TMAC/s/$, 2× → 50.9
  - assembly cost: 0.3× → 56, 3× → 10
  - the light a detector needs: 4.7–9.3 fJ/MAC

## 4. Interface as computation

The glass engine is literally **a display shining through glass onto a
camera**: the same component families a computer already uses to talk to
people.

That points at the interface idea concretely. A perception-to-display
pipeline could compute in light from sensor to screen, converting only at
the ends that already exist. There's no reason to anchor on HDMI or coaxial
links: the machine's native I/O is arrays of emitters and detectors.

**[S]** Today's photonic chips work in the infrared, so a version that
shares parts with visible displays needs visible-light optics. That is
natural for glass and silicon nitride, not silicon.

## 5. A path to building something real

The same rule as before: measure, fit, predict.

1. **Tabletop glass engine, under about $1k, no foundry** (precedent: Wang
   et al., *Nat. Commun.* 2022, an optical neural network under one photon
   per multiplication using a display, a modulator and a camera).
   - **Inputs:** a monitor or phone screen showing the input vector as bars.
   - **Weights:** a monochrome LCD from a resin 3D printer, a 4K–12K grayscale
     panel sold as a roughly $50 spare part.
   - **Optics:** cylindrical lenses to spread and sum.
   - **Detector:** a linear CCD (the TCD1304 used in DIY spectrometers) or a
     mono camera.
   - It runs at 60–240 symbols/s, which is slow, but it measures what
     matters: precision, crosstalk and calibration at 1000×1000.
   - **Parts list, procedure and the simulator's predictions for it:**
     [TABLETOP.md](TABLETOP.md). The costed build is about $450–1,000:
     two HDMI-driven printer LCDs, two used 50 mm lenses face to face as a
     symmetric relay, and a mono global-shutter camera.
2. **Free-space simulator in the studio:** diffraction, aberration,
   crosstalk, mask quantisation and detector noise, fitted to the tabletop
   measurements, like the ring model was fitted to openEBL chips.
3. **Speed path:** VCSEL or µLED emitter arrays and photodiode arrays with
   CMOS readout (datacom parts) at 10 MHz → 125 MHz → GHz. Each step is
   gated by measured precision.
4. **Model in glass:** lithographic grayscale masks generated by the studio
   from a trained network and the fitted process model. Glass is the
   cheapest patterned material there is.
5. **Rings still matter** where they are strong: WDM links between engines,
   and the most fab-ready step (openEBL, with measured data already in hand).

## 6. Risks and unknowns

| Unknown | Why it decides things | First test |
|---|---|---|
| Precision at 1024×1024: crosstalk, aberration, mask uniformity, stray light | Below about 5 bits on real tasks, the energy advantage buys nothing | Tabletop rig plus fitted simulator |
| Emitter and detector arrays at 100 MHz–GHz with analog levels | Symbol rate is the largest cost lever | Datacom VCSEL/PD array evaluation |
| Assembly and alignment cost at volume, and thermal drift | 49 % of the engine's cost | Camera-module-style assembly study |
| Fixed weights (about 1 M per engine; LCD reloads at 60–240 Hz) | Suits resident models; LLM decode needs many engines | Map real models onto engines |
| Equal task accuracy, analog ≈5 bit vs FP8/FP4 | All comparisons above assume it | Noise-aware training against the fitted noise model |
| GPUs keep improving about 1.5–2× per generation | The target moves | Re-run `econ` with each new generation |

## Sources

- Wafer prices: [Silicon Analysts, wafer pricing by node](https://siliconanalysts.com/data/wafer-pricing)
- GPU cost breakdowns: [Epoch AI, B200 cost](https://epoch.ai/data-insights/b200-cost-breakdown); [Silicon Analysts, AI chip BOM](https://siliconanalysts.com/market-data/ai-chip-bom); [Silicon Analysts, AI accelerator costs](https://siliconanalysts.com/data/ai-chip-costs)
- Fab electricity: [Silicon Analysts, chip manufacturing energy](https://siliconanalysts.com/analysis/chip-manufacturing-energy-cost)
- Photonics packaging share: [NSF fibre-to-chip packaging project](https://nsf.elsevierpure.com/en/projects/i-corps-fiber-to-chip-photonic-packaging-with-low-loss-2/); [Semiconductor Engineering, Light in a package](https://semiengineering.com/light-in-a-package)
- Silicon photonics wafer cost estimate: [Live in the Future, SiPh fab capacity](https://liveinthefuture.org/stories/silicon-photonics-fab-capacity-race)
- Free-space optical NN below one photon per MAC: Wang et al., *Nature Communications* 13, 123 (2022)
- Lenslet EnLight optical vector-matrix processor (2003): company announcements; treat as claims
