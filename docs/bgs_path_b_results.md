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

Following the Hahn et al. 2023 (§5.3) redshift-success criteria that can be
evaluated without a deep comparison spectrum:

```
success = (ZWARN == 0) & (DELTACHI2 > 40) & (SPECTYPE == 'GALAXY')
```

applied over all `COADD_FIBERSTATUS == 0` (good fiber) rows — not just rows
already classified `SPECTYPE == 'GALAXY'`, since a misclassification away
from GALAXY is itself a failure mode that must count against the rate.

Other definitions exist, so this choice matters. Hahn et al. 2023 do require
`SPECTYPE == 'GALAXY'`, but the LSS code does not: `goodz_infull` in
`desihub/LSS` (`py/LSS/common_tools.py`) uses only `ZWARN == 0` and
`DELTACHI2 > 40` for BGS, and the `desispec_validate` copy in the same file
rejects only `SPECTYPE == 'STAR'`. Hahn et al.'s `ZERR < 0.0005(1+z)` criterion
is omitted here because it changes the rate by only ~0.0002 at most.
For the extended sample (`SHAPE_R > 1"`):

| r_fiber | Hahn subset (adopted) | SPECTYPE != STAR | no SPECTYPE cut (LSS) |
|---|---|---|---|
| all | 97.61% | 98.30% | 98.61% |
| 19.05 | 98.43% | 99.66% | 99.98% |
| 21.05 | 98.42% | 98.66% | 98.84% |
| 21.95 | 79.63% | 79.82% | 80.03% |
| 22.85 | 39.08% | 39.28% | 40.62% |

So the ~1.3% shortfall on the bright plateau comes from QSO and STAR
classifications rather than SNR. The choice barely moves the faint-end decline.

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
full sample) recovers a clean, physically sensible relation. Bins run to the
survey's own limit, the `r_fiber < 22.9` cut of Hahn et al. 2023 Eq. 2:

| r_fiber mag | success rate | N in bin |
|---|---|---|
| 17–20.5 | 98.4–99.2% (flat within noise) | |
| 20.75 | 98.8% | 197,825 |
| 21.05 | 98.4% | 137,835 |
| 21.55 | 94.3% | 41,502 |
| 21.95 | 79.6% | 12,454 |
| 22.25 | 62.7% | 5,826 |
| 22.55 | 42.2% | 2,726 |
| 22.85 | 39.1% | 1,492 |

(full table: `research/bgs_path_b/bgs_redshift_success_extended_vs_rfibermag.csv`)

An earlier version of the fit stopped at 22.0 only because of its binning
range, which looked like an edge at 0.80. The sample really continues to 22.9.

The decline flattens to ~0.4 for r_fiber > 22.5. This analysis did not
establish why. Success at these magnitudes may depend more on emission lines
than continuum; if so, the curve's faint end won't track continuum SNR, and
results that map onto it (r_fiber-equivalent > 22.5) should be read with
caution.

This matches Hahn et al. 2023's qualitative claim (>95% success for BGS
Bright r<19.5, declining gradually through BGS Faint) and is the v1 calibration
for `compute_redshift_success`. At runtime, the table is made monotone with
N-weighted isotonic regression, then interpolated with PCHIP. PCHIP alone would
keep the noise on the bright plateau, and added exposure could then lower
P(success). Past r_fiber = 22.9 no targets exist, and the code returns 0, a
deliberately conservative step down from 0.39.

The fitted logistic, `P(success | r_fiber_mag) = floor + (ceil-floor) / (1 +
exp((m - x0)/k))`, is a compact summary and is not used at runtime:

```
x0    = 22.1482
k     =  0.2316
floor =  0.3604
ceil  =  0.9870
```

Sanity check against the Hahn+23 design anchor (>95% success for BGS Bright
near r<19.5): this fit gives **98.7%** at r_fiber~19.5 — consistent, with
headroom.

Compact sources are not calibrated. For `0 < SHAPE_R < 1″` the rate climbs from
~35% at r_fiber~17 to ~99% at r_fiber~20.7. That runs opposite to an SNR trend
and looks contaminant-driven, so `compute_redshift_success` warns for
`half_light_radius < 1″` rather than using a compact curve.

## Translating the curve to Spec-S5

`compute_redshift_success` finds the magnitude m′ at which DESI reaches the
r-arm SNR Spec-S5 achieves, then evaluates the curve at m′:

- **Fixed reference.** The curve corresponds to DESI at 1×180 s effective time
  in nominal conditions (dark sky, 1.1″ seeing; Hahn et al. 2023 §4.1), because
  DESI's ETC scales exposures to that effective time. The reference doesn't
  change with the Spec-S5 `texp`/`nexp`.
- **Exact inversion.** m′ solves SNR_DESI(m′) = SNR_S5(m), including sky and
  read noise. The r-arm is sky-limited at these magnitudes, so
  `m - 5 log10(ratio)` would overstate the gain (about twofold, ~0.2 mag at
  r_fiber = 22).
- **Fiber losses.** The throughput tables already contain point-source fiber
  losses, and `FIBERFLUX_R` already excludes light outside a 1.5″ fiber in 1″
  seeing. Source photons are therefore scaled by
  `C(gal⊗PSF_obs, D) / C(PSF_table, D) / C(gal⊗PSF_LS, 1.5″)`, for each
  instrument's fiber diameter D. `PSF_obs` and `PSF_table` are Moffat profiles
  (β = 3.5), as in the throughput budgets; `PSF_LS` is the Gaussian of the
  FIBERFLUX definition. `PSF_table` is fixed for each instrument (below), so
  worse observing seeing lowers source photons for galaxies and point sources
  alike.
- **Throughput tables.** From the Spec-S5 Systems Engineering Throughput
  Budget, SS5-0022 v1.1, "Comparison to DESI" tab. Both source tables are
  point-source ("star") curves:
  - **Spec-S5:** Moffat 0.96″ FWHM (g-band atmospheric seeing) in the 107 µm
    (1.02″) fiber, with 9 µm mirror blur. Telescope blur and fiber offsets are
    separate fixed losses in the budget, so 0.96″ is atmospheric seeing only.
    The budget has no 120 µm curve; the code scales the 107 µm curve by the
    previous tables' 120/107 ratio (1.13).
  - **DESI:** DESI-347-v17 at Moffat 1.1″ FWHM, given in the budget as the DESI
    galaxy curve × 1.24.

  `seeing_fwhm` is therefore atmospheric Moffat FWHM in the same sense. At the
  nominal 1.1″, Spec-S5 is observed in worse seeing than its 0.96″ table, and a
  point source loses ~17% of its table flux from the 1.02″ fiber.

Depth gain m − m′ at r_fiber = 21.5, 1×180 s, n = 1 (DESI at its nominal
reference; changing r_fiber to 22 alters these by < 0.01 mag):

| Spec-S5 seeing | point-like (0.01″) | hlr 1.0″ | hlr 1.5″ | hlr 3.0″ |
|---|---|---|---|---|
| 1.1″ (nominal) | 0.49 mag | 0.30 mag | 0.26 mag | 0.23 mag |
| 0.96″ | 0.70 mag | 0.40 mag | 0.34 mag | 0.27 mag |

Before the SS5-0022 v1.1 update the nominal hlr = 1.5″ gain was quoted as
0.72 mag. Using the DESI star curve instead of the galaxy curve (DESI was
understated by 1.24×) took it to 0.47 mag, and the Moffat PSF with per-table
seeing to 0.26 mag. Most of the remaining gap to the 2.5× collecting-area
ratio comes from Spec-S5's smaller fiber in arcseconds (1.02″ vs 1.51″), which
captures less light at 1.1″ seeing.

Known limitations:

- **Throughput-table approximations.** The budget's seeing is achromatic
  (g-band 0.96″ at every wavelength for Spec-S5), and its instrument blur and
  fiber-offset losses are computed for a point source but applied to galaxies
  too. The DESI star curve is a fixed 1.24× scaling of the galaxy curve.
- **Faint plateau under translation.** The curve's ~0.4 plateau is unlikely to
  be SNR-driven, but the translation shifts it in magnitude anyway. At the
  default settings it extends to about r_fiber = 23.6 for Spec-S5, where it steps
  to 0.
- **Sky level.** The tabulated sky is the moonless-continuum model in SS5-0022,
  cited there to Hanuschik (2003, A&A 407, 1157), a sky atlas from the VLT
  at Paranal. It is not a measured KPNO dark sky. It is ~21.3 AB mag/arcsec²
  at 6500 Å, against Hahn et al.'s nominal r = 21.07. It's the same for both
  instruments, so `texp` means an effective exposure under that sky. For real
  exposures under brighter sky, use `sky_scale`, which scales the sky grey and
  has no moon-sky spectral shape. Both the budget and the code apply
  atmospheric transmission to this sky; whether the Hanuschik spectrum already
  includes extinction is unconfirmed.
- **Single SNR metric.** The translation uses continuum SNR only, with no
  allowance for emission-line success.

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
