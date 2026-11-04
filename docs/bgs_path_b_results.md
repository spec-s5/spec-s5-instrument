# Path B results: BGS redshift success vs. SNR/magnitude from DESI DR1

This is the detailed writeup of the Path B analysis referenced in
`bgs_plan.md` §2.5. Scripts and raw outputs live in `research/bgs_path_b/`.

## Data

DESI DR1, iron reduction, LSS catalog v1.5:
`BGS_BRIGHT_full_HPmapcut.dat.fits` (5.2 GB, 6,419,406 rows, 118 columns).
Downloaded from
`https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/`.
The "full" catalog (as opposed to "clustering") retains every targeted
object including redshift failures — required to measure a success *rate*.

Relevant columns: `ZWARN`, `DELTACHI2`, `SPECTYPE`, `TSNR2_BGS`,
`TSNR2_BGS_R`, `FIBERFLUX_R`, `FLUX_R`, `SHAPE_R`, `COADD_EXPTIME`,
`COADD_FIBERSTATUS`, `mod_success_rate`.

## Success criterion

Following DESI's own binary redshift-recovery definition (Hahn et al. 2023;
Guy et al. 2023):

```
success = (ZWARN == 0) & (DELTACHI2 > 40) & (SPECTYPE == 'GALAXY')
```

applied over all `COADD_FIBERSTATUS == 0` (good fiber) rows — not just rows
already classified `SPECTYPE == 'GALAXY'`, since a misclassification away
from GALAXY is itself a failure mode that must count against the rate.

Overall success rate across the full good-fiber sample: **97.2%** (consistent
with, slightly below, the literature DR1 benchmark of ~99.35% quoted for
all BGS single-exposure tiles in arXiv:2405.17208 — the difference is
mostly the point-source contamination discussed below, which that benchmark
likely already excludes via its own target-purity cuts).

## Finding 1 — the survey's adaptive exposure strategy compresses the
realized SNR range

Binning success against DESI's own effective-SNR² metric
(`sqrt(TSNR2_BGS)`) gives an almost flat curve (0.96-0.99) because DESI
exposes each BGS tile adaptively until the EFFTIME target is reached — this
by design removes most of the SNR dynamic range that *would* exist if every
target got the same fixed exposure time. Normalizing `TSNR2_BGS_R` to a
fixed 180s reference exposure (`SNR_nominal = sqrt(TSNR2_BGS_R * 180 /
COADD_EXPTIME)`) recovers the expected trend but it remains shallow (floor
0.967, ceiling 0.970 over SNR~5-40) because BGS Bright's flux cut (r<19.5)
doesn't probe deep into the low-SNR regime to begin with.

**Implication for the implementation**: predicting Spec-S5's SNR_r
analytically (via the existing `compute_snr`-style photon-counting physics)
and mapping it through a success-rate curve calibrated on real survey SNR
is viable, but the *real* survey's SNR axis is too compressed to calibrate
the faint-end slope precisely — the magnitude-based curve (Finding 3) is
the more robust calibration target for v1.

## Finding 2 — naive success vs. magnitude is non-monotonic because of
point-source contamination, not SNR

Binning success against `r_fiber` magnitude over the *full* sample gives a
surprising result: success is **anomalously low at the bright end**
(r_fiber~16.75 → 78%, rising only gradually to ~98% by r_fiber~19). This
looks backwards — brighter should mean easier, not harder.

Diagnosis: restricting to r_fiber<18 and binning by `SHAPE_R` (photometric
half-light radius):

| SHAPE_R (arcsec) | N | success rate |
|---|---|---|
| 0.0–0.5 | 5021 | **8.1%** |
| 0.5–1.0 | 529 | 78.1% |
| 1.0–1.5 | 1529 | 95.5% |
| 1.5–2.0 | 2346 | 98.2% |
| 2.0–3.0 | 6226 | 98.5% |
| 3.0–5.0 | 10741 | 99.0% |
| 5.0–8.0 | 6127 | 99.1% |
| >8.0 | 3525 | 99.4% |

and by fiber-flux-fraction (`FIBERFLUX_R / FLUX_R` — how much of the
object's total light the fiber actually captures; →1 means point-source-like):

| fiber flux fraction | N | success rate |
|---|---|---|
| 0.0–0.1 | 7205 | 99.4% |
| 0.1–0.2 | 14979 | 99.0% |
| 0.2–0.3 | 6617 | 98.1% |
| 0.3–0.4 | 1860 | 93.2% |
| 0.4–0.5 | 340 | 86.8% |
| 0.5–0.7 | 182 | **18.1%** |
| 0.7–1.0 | 4861 | **7.9%** |

The bright-end failures are concentrated almost entirely in compact,
point-source-like objects — i.e. stars or QSOs that leaked through BGS's
morphological target selection. Redrock correctly refuses to fit these as
`SPECTYPE=GALAXY`, so by the success criterion above they register as
"failures," but this has nothing to do with spectroscopic SNR. **This is a
target-selection-purity effect**, out of scope for an SNR/ETC model, and
should not be absorbed into the SNR-vs-success calibration curve.

## Finding 3 — the clean relation, after excluding point-source contaminants

Restricting to `SHAPE_R > 1.0″` (genuinely resolved sources; ~49% of the
full sample) recovers a clean, monotonic, physically sensible relation:

| r_fiber mag | success rate |
|---|---|
| 17–20.5 | 98.4–98.8% (flat) |
| 20.7 | 98.8% (peak) |
| 21.0 | 98.4% |
| 21.5 | 94.3% |
| 21.75 | 88.7% |
| 22.0 | 79.6% |

(full table: `research/bgs_path_b/bgs_redshift_success_extended_vs_rfibermag.csv`)

This matches Hahn et al. 2023's qualitative claim (>95% success for BGS
Bright r<19.5, declining gradually through BGS Faint) and is the curve
adopted as the v1 calibration for `compute_redshift_success` (§3 of
`bgs_plan.md`).

Fitted logistic, `P(success | r_fiber_mag) = floor + (ceil-floor) / (1 +
exp((m - x0)/k))`:

```
x0    = 21.8926
k     =  0.1840
floor =  0.6614
ceil  =  0.9863
```

Sanity check against the Hahn+23 design anchor (>95% success for BGS Bright
near r<19.5): this fit gives **98.4%** at r_fiber~19.5 — consistent, with
headroom.

## Cross-check: DESI's own `mod_success_rate` column

The LSS catalog carries a `mod_success_rate` column — DESI's own fiducial
completeness-weighting model. Its mean (99.19%) is close to but slightly
above the directly-measured empirical rate on the same rows (98.91%,
pre-SHAPE_R-cut sample), and it correlates only weakly with the per-object
binary outcome (point-biserial r=0.26) — expected, since it's a smoothed
model, not a per-object predictor. Treated as a rough external consistency
check rather than a calibration source.

## What's not yet done (next steps)

- Run the same analysis for **BGS Faint** (`BGS_BRIGHT-21.5` / `BGS_ANY`
  catalogs are available at the same LSS path) to get the second calibration
  curve referenced in `bgs_plan.md` §3.
- Decide the production `SHAPE_R` (or fiber-flux-fraction) cut threshold
  more rigorously — 1.0″ was a reasonable round-number choice validated
  against the table above, not an optimized value.
- The fiber-aperture-loss term for Spec-S5 (different fiber diameter than
  DESI's 1.51″) still needs its own derivation (`bgs_plan.md` §2.3 step 5) —
  this analysis calibrates DESI's success-vs-magnitude curve, not yet the
  translation to Spec-S5's own aperture.
