# BGS Path B analysis — redshift success vs. SNR/magnitude from DESI DR1

Implements Path B of `docs/bgs_plan.md` §2.2: re-deriving the BGS redshift
success-rate relation directly from the public DESI DR1 LSS catalog, rather
than relying on literature anchor points alone.

## Data

Download the DR1 BGS Bright "full" LSS catalog (iron reduction, v1.5,
HEALPix-map-cut veto applied — 5.2 GB):

```bash
curl -O https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/BGS_BRIGHT_full_HPmapcut.dat.fits
```

This "full" catalog retains every targeted object (both successful and
failed redshifts), which is what's needed to measure a success *rate* — the
clustering-ready "clustering.dat.fits" files drop failures and so can't be
used for this.

## Scripts

Run in order, pointing both at the downloaded catalog:

1. `01_explore_snr_vs_success.py <catalog> --out-dir <dir>` — exploratory
   pass: success rate vs. `TSNR2_BGS`/`TSNR2_BGS_R` (DESI's own
   effective-SNR² pipeline metric) and vs. raw `FIBERFLUX_R`. Establishes
   that the *realized* SNR distribution in the survey is heavily compressed
   by DESI's adaptive-exposure strategy (see Findings below) — a real result
   worth keeping even though its output isn't the final calibration curve.
2. `02_fit_success_vs_rfibermag.py <catalog> --out-dir <dir>` — the
   calibration fit actually used downstream: success rate vs. `r_fiber`
   magnitude, restricted to extended sources (`SHAPE_R > 1.0"`). Produces
   `bgs_redshift_success_extended_vs_rfibermag.csv` and a fitted logistic
   `P(success | r_fiber_mag)`.

## Key findings

See `../../docs/bgs_path_b_results.md` for the full writeup. Short version:

1. **DESI's adaptive exposure-time strategy compresses the realized SNR
   range.** Tiles are exposed until a target effective time is reached, so
   `TSNR2_BGS` (and hence realized SNR) varies only ~5x across the whole
   BGS Bright sample — nowhere near enough dynamic range to cleanly recover
   a success-vs-SNR curve from survey data directly. Normalizing
   `TSNR2_BGS_R` to a fixed 180 s reference exposure recovers the expected
   fainter-is-worse trend, but it's shallow (floor 0.967, ceiling 0.970 in
   the probed range) because the BGS Bright flux-limited sample (r<19.5)
   itself doesn't probe very low SNR.
2. **Naive success-rate vs. `FIBERFLUX_R` magnitude is non-monotonic** — it's
   anomalously *low* at the bright end (r_fiber<18: success ~78-90%). This
   is not an SNR effect: those failures are concentrated in targets with
   `SHAPE_R < 0.5"` and fiber-flux-fraction (`FIBERFLUX_R/FLUX_R`) > 0.5 —
   i.e. targets that are essentially point sources (stars/QSOs that leaked
   through BGS's morphological target selection), which Redrock correctly
   refuses to fit as `SPECTYPE=GALAXY`. This is a target-selection-purity
   effect, not a spectroscopic SNR effect, and is out of scope for an SNR
   ETC.
3. **Restricting to extended sources (`SHAPE_R > 1.0"`) recovers the clean,
   physically expected relation**: success rate flat at ~98.4-98.8% from
   r_fiber~17 to ~20.7, declining smoothly through the BGS Faint regime to
   ~80% by r_fiber~22. This matches Hahn et al. 2023's qualitative
   description (>95% success for BGS Bright, declining gradually into BGS
   Faint) and is the curve carried forward into the `spec5` implementation.

Fitted logistic `P(success | r_fiber_mag) = floor + (ceil-floor) / (1 +
exp((m - x0)/k))` on the extended-source sample: `x0=21.89, k=0.184,
floor=0.661, ceil=0.986` (see `logistic_fit_extended_rfibermag.txt`).
