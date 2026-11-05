"""
Path B final fit: BGS redshift-success vs r_fiber magnitude, restricted to
genuinely extended (non-point-source-contaminant) targets.

Key finding from exploratory analysis (see docs/bgs_path_b_results.md):
success rate vs r_fiber magnitude is NON-monotonic in the raw BGS_BRIGHT
sample -- it is anomalously LOW at the bright end (r_fiber<18) because that
population is dominated by compact/point-source contaminants (SHAPE_R<0.5",
fiber-flux-fraction > 0.5) that redrock correctly fails to fit as GALAXY --
a target-selection-purity effect, not an SNR effect. Restricting to
SHAPE_R > 1.0" isolates the genuine extended-galaxy population, recovering
a clean, physically sensible success-vs-magnitude relation.
"""
import argparse
import os

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("catalog", help="Path to BGS_BRIGHT_full_HPmapcut.dat.fits "
                     "(see README.md in this directory for the download URL)")
parser.add_argument("--out-dir", default=".", help="Directory to write output CSV/fit params to")
args = parser.parse_args()

CATALOG = args.catalog
OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

with fits.open(CATALOG, memmap=True) as hdul:
    data = hdul[1].data
    cols = ["ZWARN", "DELTACHI2", "SPECTYPE", "FIBERFLUX_R", "FLUX_R",
            "COADD_FIBERSTATUS", "SHAPE_R"]
    t = Table({c: np.asarray(data[c]) for c in cols})

good_fiber = t["COADD_FIBERSTATUS"] == 0
extended = t["SHAPE_R"] > 1.0  # arcsec; excludes point-source contaminants
success = (t["ZWARN"] == 0) & (t["DELTACHI2"] > 40) & (t["SPECTYPE"] == "GALAXY")
rfibmag = 22.5 - 2.5 * np.log10(np.clip(t["FIBERFLUX_R"], 1e-3, None))

sel = good_fiber & extended & np.isfinite(rfibmag)
print(f"Extended-source sample: {sel.sum()} / {len(t)} ({sel.sum()/len(t):.3%})")
print(f"Overall success rate (extended sample): {success[sel].mean():.5f}")

rmag = rfibmag[sel]
succ = success[sel].astype(float)

# Faint edge is the survey's own limit: the Hahn et al. 2023 Eq. 2 fiber
# magnitude cut, r_fiber < 22.9, rather than an arbitrary analysis range.
mag_bins = np.linspace(17.0, 22.9, 60)
centers, rates, ns = [], [], []
for i in range(len(mag_bins) - 1):
    m = (rmag >= mag_bins[i]) & (rmag < mag_bins[i + 1])
    n = m.sum()
    if n > 200:
        centers.append(0.5 * (mag_bins[i] + mag_bins[i + 1]))
        rates.append(succ[m].mean())
        ns.append(n)
centers, rates, ns = map(np.array, (centers, rates, ns))

print("\nr_fiber_mag | N | success_rate (extended sources only)")
for c, n, r in zip(centers, ns, rates):
    print(f"  {c:6.3f}     | {n:7d} | {r:.4f}")

def logistic(x, x0, k, floor, ceil):
    return floor + (ceil - floor) / (1.0 + np.exp((x - x0) / k))

p0 = [21.0, 0.5, 0.0, rates.max()]
popt, pcov = curve_fit(logistic, centers, rates, p0=p0, sigma=1.0 / np.sqrt(ns), maxfev=50000)
print(f"\nFitted logistic P(success | r_fiber_mag), extended sources:")
print(f"  x0={popt[0]:.4f}  k={popt[1]:.4f}  floor={popt[2]:.4f}  ceil={popt[3]:.4f}")
print(f"  (x0 = magnitude at which success rate crosses the midpoint between floor and ceil)")

out = Table({"r_fiber_mag": centers, "success_rate": rates, "n": ns})
out.write(f"{OUT_DIR}/bgs_redshift_success_extended_vs_rfibermag.csv", format="csv", overwrite=True)
with open(f"{OUT_DIR}/logistic_fit_extended_rfibermag.txt", "w") as f:
    f.write(f"x0={popt[0]}\nk={popt[1]}\nfloor={popt[2]}\nceil={popt[3]}\n")
print(f"\nWrote {OUT_DIR}/bgs_redshift_success_extended_vs_rfibermag.csv")
print(f"Wrote {OUT_DIR}/logistic_fit_extended_rfibermag.txt")

# Sanity check against Hahn+23 anchors
near_195 = (centers > 19.3) & (centers < 19.7)
print(f"\nSanity check vs Hahn+23 (success > 95% for BGS Bright, r<19.5):")
print(f"  model success near r_fiber~19.5: {rates[near_195].mean():.4f}")
