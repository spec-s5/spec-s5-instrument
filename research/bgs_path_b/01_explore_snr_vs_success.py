"""
Path B analysis: derive an empirical BGS redshift-success-vs-SNR relation
from the public DESI DR1 LSS "full" catalog (BGS_BRIGHT, iron reduction).

Success criterion follows DESI's own binary redshift-recovery definition
(Hahn et al. 2023; Guy et al. 2023): ZWARN == 0 and DELTACHI2 > 40.

Input : BGS_BRIGHT_full_HPmapcut.dat.fits (DR1 LSS catalog, v1.5)
Output: a tabulated success-rate-vs-SNR curve + fitted logistic parameters,
        written as CSV, for use as the spec5 BGS calibration data file.
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
parser.add_argument("--out-dir", default=".", help="Directory to write output CSVs to")
args = parser.parse_args()

CATALOG = args.catalog
OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

COLS = ["TARGETID", "ZWARN", "DELTACHI2", "SPECTYPE", "Z_not4clus",
        "TSNR2_BGS", "TSNR2_BGS_R", "FIBERFLUX_R", "SHAPE_R",
        "COADD_EXPTIME", "COADD_NUMEXP", "mod_success_rate", "PHOTSYS"]

print(f"Reading {CATALOG} ...")
with fits.open(CATALOG, memmap=True) as hdul:
    data = hdul[1].data
    t = Table({c: np.asarray(data[c]) for c in COLS})
print(f"Loaded {len(t)} rows")

# ---------------------------------------------------------------------------
# DESI binary redshift-success criterion (Hahn+23 / Guy+23)
# ---------------------------------------------------------------------------
is_galaxy = t["SPECTYPE"] == "GALAXY"
success = (t["ZWARN"] == 0) & (t["DELTACHI2"] > 40) & is_galaxy

print(f"GALAXY spectype: {is_galaxy.sum()} / {len(t)}")
print(f"Overall success rate (all GALAXY rows): {success[is_galaxy].mean():.5f}")

# ---------------------------------------------------------------------------
# SNR proxy: sqrt(TSNR2_BGS) is DESI's own effective-SNR metric for the BGS
# template (combined b+r+z arms); TSNR2_BGS_R isolates the r-arm contribution,
# the literal continuum band relevant for BGS redshift features at z<~0.4.
# ---------------------------------------------------------------------------
snr_full = np.sqrt(np.clip(t["TSNR2_BGS"], 0, None))
snr_r = np.sqrt(np.clip(t["TSNR2_BGS_R"], 0, None))

valid = is_galaxy & np.isfinite(snr_full) & (snr_full > 0) & np.isfinite(t["FIBERFLUX_R"]) & (t["FIBERFLUX_R"] > 0)
print(f"Valid rows for fitting: {valid.sum()}")

logsnr = np.log10(snr_full[valid])
succ_v = success[valid].astype(float)
fiberflux_r = t["FIBERFLUX_R"][valid]
rfibmag = 22.5 - 2.5 * np.log10(np.clip(fiberflux_r, 1e-3, None))

# ---------------------------------------------------------------------------
# Bin success rate vs log10(SNR_BGS)
# ---------------------------------------------------------------------------
nbins = 40
bin_edges = np.linspace(np.percentile(logsnr, 0.5), np.percentile(logsnr, 99.5), nbins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
rate = np.full(nbins, np.nan)
count = np.zeros(nbins, dtype=int)
for i in range(nbins):
    m = (logsnr >= bin_edges[i]) & (logsnr < bin_edges[i + 1])
    count[i] = m.sum()
    if count[i] > 0:
        rate[i] = succ_v[m].mean()

ok = count > 50  # drop sparse bins
print("\nlog10(SNR_BGS) bin | SNR_BGS | N | success_rate")
for bc, n, r in zip(bin_centers[ok], count[ok], rate[ok]):
    print(f"  {bc:6.3f}        | {10**bc:8.2f} | {n:7d} | {r:.4f}")

# ---------------------------------------------------------------------------
# Fit a logistic curve: P(success) = 1 / (1 + exp(-(logSNR - x0)/k))
# ---------------------------------------------------------------------------
def logistic(x, x0, k, floor, ceil):
    return floor + (ceil - floor) / (1.0 + np.exp(-(x - x0) / k))

p0 = [bin_centers[ok][np.argmin(np.abs(rate[ok] - 0.5))] if (rate[ok] < 0.5).any() else bin_centers[ok][0],
      0.3, 0.0, 1.0]
try:
    popt, pcov = curve_fit(logistic, bin_centers[ok], rate[ok], p0=p0,
                            sigma=1.0 / np.sqrt(count[ok]), maxfev=20000)
    print("\nFitted logistic P(success | log10 SNR_BGS):")
    print(f"  x0={popt[0]:.4f}  k={popt[1]:.4f}  floor={popt[2]:.4f}  ceil={popt[3]:.4f}")
except RuntimeError as e:
    popt = None
    print("Logistic fit failed:", e)

# ---------------------------------------------------------------------------
# Cross-check against DESI's own mod_success_rate column
# ---------------------------------------------------------------------------
msr = t["mod_success_rate"][valid]
msr_ok = np.isfinite(msr)
print(f"\nmod_success_rate finite for {msr_ok.sum()}/{valid.sum()} valid rows")
if msr_ok.any():
    print(f"  mean mod_success_rate: {msr[msr_ok].mean():.4f}")
    print(f"  mean empirical success (same rows): {succ_v[msr_ok].mean():.4f}")
    corr = np.corrcoef(msr[msr_ok], succ_v[msr_ok])[0, 1]
    print(f"  point-biserial correlation(mod_success_rate, success): {corr:.4f}")

# ---------------------------------------------------------------------------
# Save calibration table
# ---------------------------------------------------------------------------
out = Table({"log10_snr_bgs": bin_centers[ok], "snr_bgs": 10**bin_centers[ok],
             "success_rate": rate[ok], "n": count[ok]})
out.write(f"{OUT_DIR}/bgs_redshift_success_vs_snr.csv", format="csv", overwrite=True)
print(f"\nWrote {OUT_DIR}/bgs_redshift_success_vs_snr.csv")

if popt is not None:
    with open(f"{OUT_DIR}/logistic_fit_params.txt", "w") as f:
        f.write(f"x0={popt[0]}\nk={popt[1]}\nfloor={popt[2]}\nceil={popt[3]}\n")
    print(f"Wrote {OUT_DIR}/logistic_fit_params.txt")

# Also bin vs r_fiber magnitude for the implementation's magnitude-based API
mag_bins = np.linspace(17.0, 21.0, 33)
mag_centers = 0.5 * (mag_bins[:-1] + mag_bins[1:])
mag_rate = np.full(len(mag_centers), np.nan)
mag_count = np.zeros(len(mag_centers), dtype=int)
for i in range(len(mag_centers)):
    m = (rfibmag >= mag_bins[i]) & (rfibmag < mag_bins[i + 1])
    mag_count[i] = m.sum()
    if mag_count[i] > 0:
        mag_rate[i] = succ_v[m].mean()
mag_ok = mag_count > 50
out2 = Table({"r_fiber_mag": mag_centers[mag_ok], "success_rate": mag_rate[mag_ok], "n": mag_count[mag_ok]})
out2.write(f"{OUT_DIR}/bgs_redshift_success_vs_rfibermag.csv", format="csv", overwrite=True)
print(f"Wrote {OUT_DIR}/bgs_redshift_success_vs_rfibermag.csv")
