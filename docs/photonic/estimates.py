#!/usr/bin/env python3
"""Back-of-envelope numbers for docs/photonic/PLAN.md.

Every figure quoted in the plan is reproduced here from its stated
assumptions, so a reader can change an assumption and see what moves.
Standard library only. Run:  python3 docs/photonic/estimates.py
"""

import math

C = 299_792_458.0          # m/s
Q_E = 1.602176634e-19      # C


def hdr(t):
    print()
    print(t)
    print("-" * len(t))


# ---------------------------------------------------------------- throughput
hdr("1. Peak throughput (broadcast-and-weight, 1 MAC per ring per symbol)")
for n_lam, n_rows, orders, baud in [(64, 64, 1, 10e9), (32, 64, 2, 10e9), (8, 1, 1, 10e9)]:
    macs = n_lam * n_rows * orders * baud
    print(f"  {n_lam:3d} λ x {n_rows:3d} rows x {orders} order(s) @ {baud/1e9:.0f} GBd "
          f"= {macs/1e12:8.3f} TMAC/s  ({2*macs/1e12:.2f} TOPS if MAC=2 ops)")

# ---------------------------------------------------------------- ring optics
hdr("2. Ring geometry (Si strip, n_g = 4.3 C-band, λ = 1550 nm)")
LAM = 1550e-9
NU = C / LAM
N_G = 4.3
for r_um in (2.0, 3.3, 5.0, 10.0):
    L = 2 * math.pi * r_um * 1e-6
    fsr_hz = C / (N_G * L)
    fsr_nm = LAM**2 / (N_G * L) * 1e9
    t_rt = N_G * L / C
    print(f"  R = {r_um:4.1f} µm: L = {L*1e6:5.1f} µm, FSR = {fsr_hz/1e12:5.2f} THz "
          f"({fsr_nm:5.1f} nm), round trip = {t_rt*1e12:4.2f} ps")

hdr("2b. Linewidth needed for 10 GBd, and what it implies")
for fwhm in (25e9, 40e9):
    q_l = NU / fwhm
    tau_energy = q_l / (2 * math.pi * NU)
    fwhm_nm = LAM**2 * fwhm / C * 1e9
    print(f"  FWHM {fwhm/1e9:.0f} GHz ({fwhm_nm:.3f} nm): loaded Q = {q_l:,.0f}, "
          f"photon (energy) lifetime = {tau_energy*1e12:.2f} ps")

hdr("2c. Channels per FSR at 100 GHz spacing (spacing = 4 x 25 GHz FWHM)")
for r_um in (2.0, 3.3, 5.0):
    fsr_hz = C / (N_G * 2 * math.pi * r_um * 1e-6)
    print(f"  R = {r_um:.1f} µm: {int(fsr_hz // 100e9)} channels/FSR")
lor = lambda d, fwhm: 1 / (1 + (2 * d / fwhm) ** 2)
print(f"  nearest-neighbour Lorentzian leakage at 100 GHz, FWHM 25 GHz: "
      f"{lor(100e9, 25e9)*100:.2f} % ({10*math.log10(lor(100e9, 25e9)):.1f} dB)")
print(f"  ... at 35 GHz spacing (64 ch in a 5 µm ring's FSR): "
      f"{lor(35e9, 25e9)*100:.1f} % ({10*math.log10(lor(35e9, 25e9)):.1f} dB)")

hdr("3. Thermal stability needed for a balanced weight w = 2D - 1 in [-1, 1]")
DLAM_DT = 70e-12           # m/K, Si ring resonance shift
max_dD_du = 2 * (1 / math.sqrt(3)) / (4 / 3) ** 2   # 0.6495, Lorentzian max slope in u = 2δ/Γ
for fwhm in (25e9, 40e9):
    gamma_m = LAM**2 * fwhm / C
    max_dw_ddelta = 2 * max_dD_du * 2 / gamma_m     # per metre of detuning
    for bits in (4, 6, 8):
        lsb = 2 / (2**bits - 1)
        d_tol = (lsb / 2) / max_dw_ddelta
        print(f"  FWHM {fwhm/1e9:.0f} GHz, {bits} bits: δλ ≤ {d_tol*1e12:5.2f} pm "
              f"→ ΔT ≤ {d_tol/DLAM_DT*1e3:5.1f} mK (worst case, max-slope bias)")

# ---------------------------------------------------------------- noise
hdr("4. Receiver noise → effective bits (per balanced row, 5 GHz noise BW)")
B_E = 5e9
I_N_DENS = 15e-12          # A/√Hz TIA input-referred
RESP = 1.0                 # A/W
for p_det in (100e-6, 200e-6, 400e-6):
    i = RESP * p_det
    th = I_N_DENS * math.sqrt(B_E)
    sh = math.sqrt(2 * Q_E * i * B_E)
    for rin_db in (-150, -135):
        rn = i * math.sqrt(10 ** (rin_db / 10) * B_E)
        tot = math.sqrt(th**2 + sh**2 + rn**2)
        snr_db = 20 * math.log10((i / math.sqrt(2)) / tot)
        enob = (snr_db - 1.76) / 6.02
        print(f"  P_det {p_det*1e6:3.0f} µW, RIN {rin_db} dB/Hz: thermal {th*1e6:.2f}, "
              f"shot {sh*1e6:.2f}, RIN {rn*1e6:.2f} µA rms → SNR {snr_db:4.1f} dB, "
              f"ENOB ≈ {enob:3.1f} (full-scale)")
print(f"  typical-signal penalty for a 64-term random-sign sum: "
      f"½·log2(64) = {0.5*math.log2(64):.0f} bits unless range is rescaled per layer")


# ---------------------------------------------------------------- energy
def ledger(name, n_lam, n_rows, baud, e_dac, e_adc, e_tia, p_det, loss_db, wpe,
           mw_per_ring, e_sram_bit, bits_io, e_digital_fj, orders=1):
    macs = n_lam * n_rows * orders * baud
    n_in = n_lam * orders
    n_out = n_rows * orders
    p = {}
    p["DAC+driver"] = e_dac * n_in * baud
    p["ADC"] = e_adc * n_out * baud
    p["TIA"] = e_tia * n_out * baud
    p_opt_det = p_det * n_out
    p_laser_opt = p_opt_det / 10 ** (-loss_db / 10)
    p["laser (wall-plug)"] = p_laser_opt / wpe
    p["thermal tuning"] = n_lam * n_rows * mw_per_ring * 1e-3
    p["SRAM in/out"] = e_sram_bit * bits_io * (n_in + n_out) * baud
    p["digital acc+ctrl"] = e_digital_fj * 1e-15 * macs
    tot = sum(p.values())
    print(f"  {name}: {macs/1e12:.2f} TMAC/s, total {tot:.2f} W, "
          f"{tot/macs*1e15:.0f} fJ/MAC  (laser optical out {p_laser_opt*1e3:.0f} mW)")
    for k, v in p.items():
        print(f"      {k:18s} {v:7.3f} W  {v/macs*1e15:6.1f} fJ/MAC")
    return tot / macs


hdr("5. Energy ledger, 64 λ x 64 rows @ 10 GBd, 100 % utilisation")
nominal = ledger("NOMINAL    ", 64, 64, 10e9, 1.5e-12, 2.0e-12, 0.5e-12, 200e-6, 13, 0.10,
                 1.6, 0.05e-12, 8, 5)
optimist = ledger("OPTIMISTIC ", 64, 64, 10e9, 0.5e-12, 0.5e-12, 0.2e-12, 100e-6, 10, 0.20,
                  0.1, 0.025e-12, 8, 3)
pessim = ledger("PESSIMISTIC", 64, 64, 10e9, 3.0e-12, 5.0e-12, 1.0e-12, 400e-6, 15, 0.02,
                4.0, 0.10e-12, 8, 10)

hdr("5b. Laser budget per line (nominal path)")
path = {"chip coupling": 2.0, "input modulator": 4.0, "split excess": 1.5,
        "weight bank": 2.5, "routing/crossings": 1.0, "margin": 2.0}
loss_db = sum(path.values())
p_line_at_det = 200e-6 / 64
p_line_laser = p_line_at_det * 64 / 10 ** (-loss_db / 10)
print(f"  path loss {loss_db:.1f} dB + 1/64 broadcast split; "
      f"{p_line_at_det*1e6:.2f} µW per line per row at the PD "
      f"→ {p_line_laser*1e3:.1f} mW per line out of the laser")
db = lambda x: 10 ** (-x / 10)
p_mod = p_line_laser * db(path["chip coupling"])
p_bank = p_mod * db(path["input modulator"]) * db(path["split excess"]) / 64
finesse = (C / (N_G * 2 * math.pi * 3.3e-6)) / 25e9
print(f"  modulator ring sees {p_mod*1e3:.1f} mW; each row's bank sees {p_bank*1e6:.1f} µW per line; "
      f"finesse {finesse:.0f} → build-up ≈ F/π = {finesse/math.pi:.0f}x "
      f"→ ≈ {p_bank*finesse/math.pi*1e3:.2f} mW circulating")

hdr("6. GPU reference points (datasheet dense peak / TDP; crude, not workload-level)")
gpus = [("H100 SXM BF16", 989.5e12, 700), ("H100 SXM FP8 ", 1979e12, 700),
        ("B200 FP8     ", 4.5e15, 1000), ("B200 FP4     ", 9.0e15, 1000)]
for name, flops, watts in gpus:
    macs = flops / 2
    print(f"  {name}: {macs/1e12:7.1f} TMAC/s, {watts/macs*1e12:.3f} pJ/MAC, "
          f"= {macs/40.96e12:5.1f} x one 64x64 photonic core")

# ---------------------------------------------------------------- utilisation
hdr("7. Weight-reload-limited utilisation (one 64x64 core, 4096x4096 layer)")
TAU = 5e-6
t_settle = math.log(2**8) * TAU   # settle to 8-bit
print(f"  thermal τ = {TAU*1e6:.0f} µs → settle to 1/256 in {t_settle*1e6:.1f} µs "
      f"= {t_settle*10e9:,.0f} symbol slots at 10 GBd")
tiles = (4096 // 64) ** 2
for batch in (1, 1_000, 100_000, 1_000_000):
    t_stream = batch / 10e9
    t_single = tiles * (t_settle + t_stream)
    t_pingpong = tiles * max(t_settle, t_stream) + t_settle
    macs = 4096 * 4096 * batch
    print(f"  batch {batch:>9,}: single-bank {macs/t_single/1e12:8.4f} TMAC/s "
          f"({100*macs/t_single/40.96e12:5.1f} %), ping-pong {macs/t_pingpong/1e12:8.4f} TMAC/s "
          f"({100*macs/t_pingpong/40.96e12:5.1f} %)")

hdr("8. Simulator cost (wall-clock per simulated second), 64x64 core")
flop_per_symbol = 2 * 4096 + 64 * 25 + 64 * 10   # GEMM + noise RNG/Box-Muller + quantise
for gpu_tflops in (1.0, 2.0, 10.0):
    s = flop_per_symbol * 10e9 / (gpu_tflops * 1e12)
    print(f"  L2 symbol-accurate on {gpu_tflops:4.1f} TFLOP/s effective: slowdown ≈ {s:5.0f} x")
steps = 32
odes = 64 * 64 * 64
flop_per_ode_step = 12
for gpu_tflops in (2.0, 10.0):
    s = odes * steps * 10e9 * flop_per_ode_step / (gpu_tflops * 1e12)
    print(f"  L3 waveform (32 samples/symbol, all λ dynamic) on {gpu_tflops:.0f} TFLOP/s: "
          f"slowdown ≈ {s:,.0f} x")

hdr("9. Simulated-time base")
print(f"  u64 femtoseconds spans {2**64 * 1e-15 / 3600:.2f} h of simulated time")
print(f"  10 GBd symbol = {1e15/10e9:,.0f} fs;  28 GBd symbol = {1e15/28e9:,.4f} fs "
      f"(non-integer → rational clock accumulator)")

hdr("10. Mesh work-unit sizing (compute ≥ 10 x (RTT + transfer))")
for rtt, up_mbps, payload_mb in [(0.05, 50, 0.004), (0.05, 50, 64), (0.15, 10, 0.004)]:
    t_net = rtt + payload_mb * 8 / up_mbps
    print(f"  RTT {rtt*1e3:.0f} ms, uplink {up_mbps} Mb/s, payload {payload_mb} MB: "
          f"network {t_net:.3f} s → unit should compute ≥ {10*t_net:.1f} s")
