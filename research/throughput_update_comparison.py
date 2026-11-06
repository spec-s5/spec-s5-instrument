"""
Before/after plots for the SS5-0022 v1.1 throughput update (PR #8).

Compares two checkouts of this package:
  before: e577d8a (main before PR #8, version 0.1.0)
  after:  the current working tree (version 0.2.0)

Usage:
    git worktree add --detach /tmp/spec5_before e577d8a
    python research/throughput_update_comparison.py /tmp/spec5_before [outdir]

Each checkout is evaluated in its own subprocess (so the two copies of
``spec5`` never share an interpreter), and the results are plotted here.
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Evaluated inside each checkout. Prints one JSON object.
PROBE = r'''
import json, sys, warnings
import numpy as np
sys.path.insert(0, sys.argv[1])
warnings.simplefilter("ignore")
from spec5.instrument import stellar_etc as se
from spec5.instrument.bgs_etc import (compute_bgs_snr, compute_redshift_success,
                                      _desi_equivalent_mag)
import inspect
out = {}

# 1. Spec-S5 and DESI throughput tables, read back through the core.
#    ph / (photon flux * area * texp) recovers thru at each node.
lam = None
for fd in (107, 120):
    core = se._photon_counting_core(np.array([20.0]), 1, 1.0, fiber_diameter=fd)
    lam = core["lam_t"]
    fnu = 10 ** (-0.4 * (20.0 + 48.6))
    phot = fnu * (se._CLIGHT * 1e8) / lam**2 / (se._HPLANCK * se._CLIGHT * 1e8 / lam)
    out[f"thru_s5_{fd}"] = (core["ph_s5"][0] / (phot * np.pi * 300.0**2)).tolist()
    out["thru_d"] = (core["ph_d"][0] / (phot * np.pi * (379.7 / 2) ** 2)).tolist()
out["lam"] = lam.tolist()

# 2. Stellar RV precision vs LSST z magnitude (defaults: 3 x 1200 s, giant).
mags = np.linspace(16.0, 24.0, 81)
out["mags"] = mags.tolist()
for fd in (107, 120):
    r = se.compute_measurement_errors(mags, fiber_diameter=fd)
    out[f"vrad_{fd}"] = np.asarray(r["vrad_err"]).tolist()
    r0 = se.compute_measurement_errors(mags, fiber_diameter=fd, vrad_sys=0.0)
    out[f"vrad_stat_{fd}"] = np.asarray(r0["vrad_err"]).tolist()

# 3. BGS redshift success vs r_fiber (1 x 180 s, hlr 1.5", defaults).
rf = np.linspace(19.0, 24.0, 201)
out["rfib"] = rf.tolist()
out["p_spec5"] = compute_redshift_success(rf, "spec5", half_light_radius=1.5).tolist()
out["p_spec5_60"] = compute_redshift_success(rf, "spec5", texp=60.0,
                                            half_light_radius=1.5).tolist()
out["p_desi"] = compute_redshift_success(rf, "desi", half_light_radius=1.5).tolist()

# 4. BGS depth gain over DESI vs half-light radius at r_fiber = 21.5.
hlrs = np.geomspace(0.05, 4.0, 40)
out["hlr"] = hlrs.tolist()
has_desi_seeing = "desi_seeing_fwhm" in inspect.signature(compute_bgs_snr).parameters
def gain(h, **kw):
    r = compute_bgs_snr(21.5, 1, 180.0, half_light_radius=h, **kw)
    eq = lambda k: _desi_equivalent_mag(np.atleast_1d(r[k]), np.atleast_1d(h), 1.0)[0]
    return 21.5 - eq("snr_r_s5"), eq("snr_r_d")
out["gain_default"] = [gain(h)[0] for h in hlrs]
if has_desi_seeing:
    # Both instruments in the same 0.96" seeing: Spec-S5's equivalent mag
    # minus DESI's.
    out["gain_equal_096"] = []
    for h in hlrs:
        g_s5, eq_d = gain(h, seeing_fwhm=0.96, desi_seeing_fwhm=0.96)
        out["gain_equal_096"].append(eq_d - (21.5 - g_s5))   # m'_DESI - m'_S5
print(json.dumps(out))
'''


def evaluate(root):
    res = subprocess.run([sys.executable, "-c", PROBE, str(root)], capture_output=True,
                         text=True, check=True, cwd=root)
    return {k: np.asarray(v) for k, v in json.loads(res.stdout).items()}


# Reference palette (dataviz skill, light mode): slot 1 blue, slot 2 orange.
AFTER, BEFORE = "#2a78d6", "#eb6834"
NEUTRAL, INK, MUTED, GRID = "#7a7974", "#1f1f1e", "#6b6a64", "#e6e5e0"


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def new_figure(title, height_ratios=(3, 1.4)):
    fig, axs = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True,
                            gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08})
    fig.patch.set_facecolor("#fcfcfb")
    fig.suptitle(title, x=0.08, ha="left", fontsize=12, color=INK)
    for ax in axs:
        style(ax)
    return fig, axs


def plot_throughput(b, a, outdir):
    lam = a["lam"] / 10.0
    fig, (ax, rx) = new_figure("Spec-S5 source throughput: before vs after SS5-0022 v1.1")
    for fd, ls in ((107, "-"), (120, ":")):
        ax.plot(lam, b[f"thru_s5_{fd}"], ls, color=BEFORE, lw=2, marker="o", ms=4,
                label=f"before, {fd} µm")
        ax.plot(lam, a[f"thru_s5_{fd}"], ls, color=AFTER, lw=2, marker="o", ms=4,
                label=f"after, {fd} µm")
    ax.set_ylabel("Point-source throughput")
    ax.legend(frameon=False, fontsize=9, ncol=2, labelcolor=INK)
    for fd, ls in ((107, "-"), (120, ":")):
        rx.plot(lam, a[f"thru_s5_{fd}"] / b[f"thru_s5_{fd}"], ls, color=AFTER, lw=2,
                marker="o", ms=4, label=f"Spec-S5 {fd} µm")
    rx.axhline(1, color=MUTED, lw=0.8)
    rx.set_ylim(0.97, 1.01)
    d = np.median(a["thru_d"] / b["thru_d"])
    rx.text(0.99, 0.08, f"DESI source throughput: x{d:.2f} (off scale)", ha="right",
            transform=rx.transAxes, fontsize=8, color=MUTED)
    rx.set_ylabel("after / before")
    rx.set_xlabel("Wavelength [nm]")
    rx.legend(frameon=False, fontsize=8, ncol=2, labelcolor=INK, loc="upper right")
    ax.text(0.99, 0.04, "120 µm before and after coincide", ha="right",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    path = outdir / "throughput_change.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_rv(b, a, outdir):
    m = a["mags"]
    fig, (ax, rx) = new_figure("Spec-S5 stellar RV precision: before vs after "
                               "(3 x 1200 s, giant, 107 µm)")
    ax.semilogy(m, b["vrad_107"], color=BEFORE, lw=2, label="before")
    ax.semilogy(m, a["vrad_107"], color=AFTER, lw=2, label="after")
    ax.axhline(0.6, color=MUTED, lw=0.8, ls="--")
    ax.text(m[0] + 0.1, 0.63, "0.6 km/s systematic floor", color=MUTED, fontsize=8)
    ax.set_ylabel("RV error [km/s]")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    rx.plot(m, a["vrad_stat_107"] / b["vrad_stat_107"], color=AFTER, lw=2,
            label="107 µm, statistical only")
    rx.plot(m, a["vrad_107"] / b["vrad_107"], color=AFTER, lw=1.5, ls="--",
            label="107 µm, with floor")
    rx.plot(m, a["vrad_stat_120"] / b["vrad_stat_120"], color=NEUTRAL, lw=1.5, ls=":",
            label="120 µm, statistical only")
    rx.axhline(1, color=MUTED, lw=0.8)
    rx.set_ylabel("after / before")
    rx.set_xlabel("LSST z [AB mag]")
    rx.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    path = outdir / "rv_precision_change.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_bgs(b, a, outdir):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"wspace": 0.22})
    fig.patch.set_facecolor("#fcfcfb")
    fig.suptitle("BGS Bright redshift recovery: before vs after (1 x 180 s)",
                 x=0.06, ha="left", fontsize=12, color=INK)
    ax, gx = axs
    for x in axs:
        style(x)
    rf = a["rfib"]
    ax.plot(rf, a["p_desi"], color=NEUTRAL, lw=2, label="DESI (DR1 curve, unchanged)")
    ax.plot(rf, b["p_spec5"], color=BEFORE, lw=2, label="Spec-S5 before (1.1\" Gaussian)")
    ax.plot(rf, a["p_spec5"], color=AFTER, lw=2, label="Spec-S5 after (0.96\" Moffat)")
    ax.plot(rf, a["p_spec5_60"], color=AFTER, lw=1.5, ls="--", label="Spec-S5 after, 60 s")
    ax.set(xlabel="r_fiber [AB mag]", ylabel="P(successful redshift)", ylim=(0, 1.02))
    ax.set_title("hlr = 1.5\" exponential disk", fontsize=10, color=MUTED, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    h = a["hlr"]
    gx.semilogx(h, b["gain_default"], color=BEFORE, lw=2, label="before")
    gx.semilogx(h, a["gain_default"], color=AFTER, lw=1.5, ls="--",
                label="after, Spec-S5 0.96\" vs DESI curve (1.1\")")
    if "gain_equal_096" in a:
        gx.semilogx(h, a["gain_equal_096"], color=AFTER, lw=2,
                    label="after, both at 0.96\" (like for like)")
    gx.set(xlabel="Half-light radius [arcsec]", ylabel="Depth gain over DESI [mag]",
           ylim=(0, None))
    gx.set_title("r_fiber = 21.5", fontsize=10, color=MUTED, loc="left")
    gx.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    path = outdir / "bgs_redshift_recovery_change.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def main():
    before_root = Path(sys.argv[1]).resolve()
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "research" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    b, a = evaluate(before_root), evaluate(REPO)
    for p in (plot_throughput(b, a, outdir), plot_rv(b, a, outdir), plot_bgs(b, a, outdir)):
        print(p)
    # Headline numbers for the summary.
    i = lambda arr, x, v: float(np.interp(v, x, arr))
    print("thru_s5_107 after/before:", np.round(a["thru_s5_107"] / b["thru_s5_107"], 4))
    print("thru_s5_120 after/before:", np.round(a["thru_s5_120"] / b["thru_s5_120"], 4))
    for mz in (18, 20, 21, 22, 23):
        print(f"z={mz}: vrad before {i(b['vrad_107'], b['mags'], mz):.3f}  "
              f"after {i(a['vrad_107'], a['mags'], mz):.3f}  "
              f"stat ratio {i(a['vrad_stat_107'], a['mags'], mz) / i(b['vrad_stat_107'], b['mags'], mz):.4f}")
    for p in (0.9, 0.5):
        f = lambda arr: float(np.interp(-p, -arr[(a['rfib'] < 22.9)],
                                        a["rfib"][(a['rfib'] < 22.9)]))
        print(f"r_fiber at P={p}: DESI {f(a['p_desi']):.2f}  before {f(b['p_spec5']):.2f}  "
              f"after {f(a['p_spec5']):.2f}")


if __name__ == "__main__":
    main()
