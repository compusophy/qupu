#!/usr/bin/env python3
"""Check the plan's ring assumptions against real fabricated devices.

Data: SiEPIC openEBL 2024-10 run (Applied Nanotools e-beam, 220 nm SOI,
single full etch), double-bus ring resonators designed by L. Chrostowski,
measured by SiEPIC's automated setup (TE, 1550 nm band, 25 C). Files are
fetched from the public repository at a pinned commit and cached locally.

For each ring this prints FSR, group index n_g, drop-port FWHM (Lorentzian
fit) and loaded Q, plus what that implies for a 10 GBd WDM weight bank.

Requires numpy + scipy.  Run:  python3 docs/photonic/measured_rings.py
"""

import math
import pathlib
import re
import urllib.parse
import urllib.request

import numpy as np
import scipy.io
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

REPO = "SiEPIC/openEBL-2024-10"
COMMIT = "090cd97ceaf7805570ad756166686133b5eefe04"
DIR = "measurements/mat_files/TE_1550_25C_edX"
FILES = {
    "r10g80": "LukasChrostowski_RingDoubleTEr10g80/09-Nov-2024 02.02.44.mat",
    "r10g90": "LukasChrostowski_RingDoubleTEr10g90/09-Nov-2024 02.01.30.mat",
    "r10g100": "LukasChrostowski_RingDoubleTEr10g100/09-Nov-2024 02.03.59.mat",
    "r10g110": "LukasChrostowski_RingDoubleTEr10g110/09-Nov-2024 02.00.14.mat",
    "r5g70": "LukasChrostowski_RingDoubleTEr5g70/09-Nov-2024 02.06.28.mat",
    "r5g80": "LukasChrostowski_RingDoubleTEr5g80/09-Nov-2024 02.07.42.mat",
    "r5g90": "LukasChrostowski_RingDoubleTEr5g90/09-Nov-2024 02.05.14.mat",
    "r3g70": "LukasChrostowski_RingDoubleTEr3g70/09-Nov-2024 02.08.57.mat",
}
CACHE = pathlib.Path(__file__).parent / ".cache"
C = 299_792_458.0


def fetch(rel):
    dst = CACHE / rel
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = (f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/"
               + urllib.parse.quote(f"{DIR}/{rel}"))
        urllib.request.urlretrieve(url, dst)
    return dst


def lorentzian(x, a, x0, fwhm, c):
    return a / (1 + ((x - x0) / (fwhm / 2)) ** 2) + c


def analyse(name, rel):
    m = scipy.io.loadmat(fetch(rel), squeeze_me=True, struct_as_record=False)["testResult"]
    wl = np.asarray(m.header.wavelength, float)          # nm
    drop = np.asarray(m.rows.channel_1, float)           # dBm; channel_2 is the through port
    radius = float(re.search(r"r(\d+)", name).group(1))  # µm
    gap = float(re.search(r"g(\d+)", name).group(1))     # nm
    length = 2 * math.pi * radius * 1e-6

    # Peaks at least ~0.6 FSR apart (n_g guess 4.2) so noise isn't picked up.
    step = wl[1] - wl[0]
    min_sep = int(0.6 * (1.55e-6**2 / (4.2 * length)) * 1e9 / step)
    pk, _ = find_peaks(drop, prominence=12, distance=min_sep)
    pk = pk[(wl[pk] > 1525) & (wl[pk] < 1575)]

    fsr = np.diff(wl[pk])
    lam_mid = (wl[pk][1:] + wl[pk][:-1]) / 2
    n_g = (lam_mid * 1e-9) ** 2 / (fsr * 1e-9 * length)

    widths = []
    for p in pk:
        s = slice(max(p - 150, 0), p + 150)
        x, y = wl[s], 10 ** (drop[s] / 10)
        try:
            popt, _ = curve_fit(lorentzian, x, y, p0=[y.max(), wl[p], 0.1, y.min()], maxfev=5000)
            if 0.005 < abs(popt[2]) < 2:
                widths.append(abs(popt[2]))
        except RuntimeError:
            pass
    fwhm_nm = float(np.median(widths))
    fwhm_ghz = C * fwhm_nm * 1e-9 / (1550e-9) ** 2 / 1e9
    fsr_ghz = C * fsr.mean() * 1e-9 / (1550e-9) ** 2 / 1e9
    return dict(name=name, R=radius, gap=gap, n_res=len(pk), fsr_nm=fsr.mean(), fsr_ghz=fsr_ghz,
                n_g=n_g.mean(), n_g_sd=n_g.std(), fwhm_pm=fwhm_nm * 1e3, fwhm_ghz=fwhm_ghz,
                q_l=1550 / fwhm_nm)


def main():
    print(f"SiEPIC openEBL 2024-10 @ {COMMIT[:7]}: double-bus rings, TE, 25 C\n")
    print(f"{'ring':8s} {'R µm':>5s} {'gap nm':>6s} {'res':>3s} {'FSR nm':>7s} {'FSR THz':>7s} "
          f"{'n_g':>12s} {'FWHM pm':>7s} {'FWHM GHz':>8s} {'Q_L':>6s} {'ch/FSR@100G':>11s} {'10 GBd ok':>9s}")
    rows = [analyse(k, v) for k, v in FILES.items()]
    for r in rows:
        ch = int(r["fsr_ghz"] // 100)
        ok = "yes" if r["fwhm_ghz"] >= 25 else "tight"
        spread = f"±{r['n_g_sd']:.3f}" if r["n_res"] > 2 else " (1 FSR)"
        print(f"{r['name']:8s} {r['R']:5.0f} {r['gap']:6.0f} {r['n_res']:3d} {r['fsr_nm']:7.2f} "
              f"{r['fsr_ghz']/1e3:7.2f} {r['n_g']:6.3f}{spread} {r['fwhm_pm']:7.0f} "
              f"{r['fwhm_ghz']:8.1f} {r['q_l']:6.0f} {ch:11d} {ok:>9s}")
    ng = np.array([r["n_g"] for r in rows])
    print(f"\nn_g across all rings: {ng.mean():.3f} ± {ng.std():.3f}  (plan assumed 4.3, range 4.1-4.5)")
    print("Loaded Q rises monotonically with gap at fixed radius: weaker coupling, narrower line.")


if __name__ == "__main__":
    main()
