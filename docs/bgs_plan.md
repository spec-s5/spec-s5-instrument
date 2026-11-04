# BGS SNR Scaling & Redshift Recovery — Research & Implementation Plan

Status: research done for BGS Bright (§2.5); v1 implementation done (§3) —
branch `bgs`
Author: kadrlica (plan drafted with Claude)

**Implemented**: `spec5/instrument/galaxy_photometry.py`
(`fiber_capture_fraction`), `spec5/instrument/bgs_etc.py`
(`compute_bgs_snr`, `compute_redshift_success`),
`observe_bgs_with_spec5`/`observe_bgs_with_desi` in `mock_observations.py`,
calibration data (`spec5/instrument/data/bgs_redshift_success_bright.csv`),
tests (`tests/test_bgs_etc.py`, `tests/test_bgs_mock_observations.py`), a
README section, and a demo notebook (`notebooks/BGS_SNR_ETC.ipynb`).
**Not yet done**: BGS Faint calibration, and validation against a mock BGS
catalog with a realistic (non-toy) half-light-radius distribution (see §4
and §3 item 5 below).

## 1. Motivation

The package currently models point-source **stellar** targets only
(`spec5/instrument/stellar_etc.py`): `compute_snr` computes photon-counting
SNR per Å for Spec-S5 and DESI, and `compute_measurement_errors` converts
z-arm continuum SNR into RV, proper-motion, and distance errors via empirical
relations calibrated on DESI MWS data and Li et al. (2025).

We want the analogous capability for the DESI **Bright Galaxy Survey (BGS)**:
given a galaxy's fiber magnitude (and size), predict (a) the Spec-S5
spectroscopic SNR, and (b) the probability/quality of a successful redshift
measurement — then compare Spec-S5 to DESI the same way `observe_with_desi`
mirrors `observe_with_spec5` today.

Two things are genuinely different from the stellar case and drive the plan
below:

1. **BGS galaxies are low-z, absorption/4000Å-break-dominated targets**, not
   point sources — the redshift success relevant wavelength range is DESI's
   **r arm** (~5760–7620 Å), not the z-arm used for stellar RV.
2. **BGS targets are (mildly) extended** — DESI's own pipeline assumes a 1.5″
   half-light radius template for BGS SNR normalization (vs. 1.0″ LRG, 0.45″
   ELG), so a fiber-aperture-loss term is needed that doesn't exist in the
   current point-source `compute_snr`.

## 2. Research plan — identifying the BGS SNR / redshift-success scaling

### 2.1 Literature already gathered

- **Hahn et al. 2023** (AJ 165, 253; arXiv:2208.08512) — BGS final target
  selection & validation. Defines the redshift-success criterion: no Redrock
  ZWARN flag, SPECTYPE=GALAXY, **Δχ² > 40**, and agreement with a deep-coadd
  truth redshift to |Δz|/(1+z) < 0.0033. At Δχ²=40: **>99.5% purity / >99%
  completeness (BGS Bright)**, **>99.5%/>98.5% (BGS Faint)**. Fig. 5 sets the
  **t_nom = 180 s** nominal exposure as the time giving **>95% redshift
  efficiency** for BGS Bright under nominal dark-time conditions. Figs. 9–12
  show success vs. Δχ², r_fiber, and r-mag as **hexbin plots only** — no
  closed-form fit is published.
- **Guy et al. 2023** (AJ 165, 144; arXiv:2209.14482) — pipeline paper
  defining **TSNR2** (template-weighted SNR²) per tracer class, with BGS
  normalized to a **1.5″ half-light-radius** template (Table 2/Fig. 43), and
  **EFFTIME_BGS = 12.15 s × TSNR2_LRG**-style effective time (correlates with
  median Δχ² per exposure, r=0.97). The `desispec.efftime` module
  (desispec.readthedocs.io) gives the explicit production formula:
  `efftime_bright = exptime * (fiberfac_bgs/airfac)**2 * (sky_nom/effsky_bright) / ebvfac**2`,
  with `fiber_diameter_arcsec=1.52`, `sky_nom=3.73 nMgy/arcsec²`,
  `exptime_nom=1000 s`. Re-exposure threshold for bright/BGS tiles:
  EFFTIME < 153 s (85% of the 180 s goal).
- **DESI DR1 systematics paper** (arXiv:2405.17208) — benchmark single-exposure
  BGS success ≈ **99.35%**, degrading slightly with repeat exposures
  (99.15% @ 2 exp, 98.95% @ 3 exp).
- **Public data products** for independent re-derivation: DR1 redshift
  catalogs (`zall-pix-iron.fits`, under
  `spectro/redux/iron/zcatalog/v1/`, data.desi.lbl.gov/public/dr1) carry
  `DELTACHI2`, `ZWARN`, `EFFTIME_BGS` per target; LSS/clustering catalogs
  (`LSScats/v1.2+`) and the FastSpecFit VAC add fiber photometry and
  emission-line fits.

### 2.2 Gap: no closed-form relation exists yet

Hahn et al. give qualitative hexbin figures, not a fitted curve — unlike the
stellar case where the literature already supplies
`log10(σ_vrad) = f(log10(SNR_z))`. **There is no equivalent off-the-shelf
formula for BGS**; it must be derived. Two viable paths, not mutually
exclusive:

- **Path A (fast, approximate):** Use the known anchor points — t_nom=180s ↔
  ~95–99% efficiency (Hahn+23), EFFTIME re-exposure floor at 153s, DR1 overall
  99.35% — to fit a simple parametric curve (e.g. a logistic or saturating
  power law in SNR_r or EFFTIME_BGS) through 3–4 calibration points. Fast to
  implement, but only as good as the sparse anchors.
- **Path B (rigorous, recommended primary deliverable):** Pull a BGS subset of
  the DR1 `zall-pix-iron` redshift catalog + fiber photometry, bin
  pass/fail (Δχ²>40 and ZWARN==0) against a computed SNR_r (continuum SNR in
  the r-arm range, from fiber flux + EFFTIME_BGS), and fit a logistic/sigmoid
  success-rate(SNR_r) curve directly — the direct analog of how
  `dist_err_vs_snr_giant.csv` was built from Li et al. (2025). This is the
  main research task and should be budgeted the bulk of the time.

### 2.3 Research steps

1. Pull DR1 BGS zcatalog rows (`SURVEY='main'`, `PROGRAM='bright'`,
   `SPECTYPE='GALAXY'`) with `DELTACHI2`, `ZWARN`, `EFFTIME_BGS`, and fiber-flux
   columns (`FIBERFLUX_R` from the targeting photometry, joined by `TARGETID`).
2. Convert `FIBERFLUX_R` + `EFFTIME_BGS` into a continuum SNR_r using the same
   photon-counting physics already in `compute_snr` (DESI throughput curves,
   sky background, read noise) restricted to the r-arm (5760–7620 Å) instead
   of z-arm — this gives an apples-to-apples SNR_r for every galaxy in the
   sample.
3. Bin `success = (DELTACHI2 > 40) & (ZWARN == 0)` against SNR_r (or
   equivalently r_fiber mag) and fit a sigmoid/logistic:
   `P(success | SNR_r) = 1 / (1 + exp(-(log10(SNR_r) - x0)/k))`, analogous in
   spirit to the existing cubic-spline `dist_err_frac(snr)` lookup.
4. Cross-check the fit reproduces: (a) ~95–99% at the t_nom=180s/BGS Bright
   r<19.5 regime, (b) the DR1 99.35% overall benchmark, (c) the re-exposure
   floor behavior near EFFTIME≈153s.
5. Separately quantify the **fiber-aperture-loss** term for extended BGS
   sources: using the TSNR2 half-light-radius convention (1.5″ default for
   BGS) and a Sersic/Gaussian light profile convolved with the fiber aperture
   (Spec-S5: 1.02″/1.144″ for the 107/120 µm fiber options; DESI: 1.51″),
   derive a flux-capture-fraction correction to apply before photon counting.
   Source for the BGS size distribution: Hahn et al. 2023 or Legacy
   Survey/SGA morphology catalogs for the BGS Bright/Faint samples.
6. Translate: with the DESI-calibrated `P(success | SNR_r)` relation and the
   fiber-aperture-loss term in hand, compute Spec-S5's predicted SNR_r (6m
   mirror, Spec-S5 throughputs, chosen fiber diameter) for a given BGS fiber
   magnitude and exposure time, and map it through the same success relation
   — exactly mirroring how `observe_with_desi` reuses `compute_measurement_errors`
   with a mirror-area magnitude offset today.

### 2.4 Open questions — resolved

- **Redshift recovery is modeled as binary success/fail (resolved).**
  Following DESI's own framework (Hahn et al. 2023; Guy et al. 2023): Δχ² is
  a detection/non-detection statistic, not a precision metric, and DESI
  itself has no published `sigma_z(SNR)` for successes. `compute_redshift_success`
  returns `P(success | SNR_r or r_fiber_mag)` and the mock-observation layer
  draws a Bernoulli outcome from it — no continuous z-scatter model.
- Single fixed half-light radius (1.5″, matching DESI's TSNR2 convention) vs.
  a BGS size distribution — start with the fixed value for v1, matching how
  `star_type` is a fixed enum in the stellar code today.
- BGS Bright vs. BGS Faint as one combined relation or two separate curves —
  Hahn+23 gives separate numbers for both, so plan for two calibration curves
  (mirrors `star_type='giant'|'dwarf'` having two data files today). Path B
  (§2.5) has so far only been run for BGS Bright; BGS Faint is the natural
  next slice of the same analysis.

### 2.5 Path B results (executed)

Path B has been run against the real DR1 BGS Bright "full" LSS catalog
(iron reduction, v1.5). Scripts, data, and full writeup:
`research/bgs_path_b/` and `docs/bgs_path_b_results.md`.

Headline finding: a naive success-rate-vs-magnitude fit is **non-monotonic**
and misleading at the bright end — it's driven by point-source/compact
contaminants in the targeting catalog (SHAPE_R < 0.5″), not by SNR. Once
restricted to genuinely extended sources (SHAPE_R > 1.0″), the relation is
clean: success flat at ~98.4-99.2% from r_fiber~17-20.7, declining through
~80% at r_fiber~22 and flattening near 0.4 out to the r_fiber < 22.9 survey
limit — consistent with Hahn et al. 2023's qualitative description. This binned
curve, made monotone at load time, is the calibration used by
`compute_redshift_success` (§3); the logistic in
`research/bgs_path_b/logistic_fit_extended_rfibermag.txt` is only a summary.

A second finding worth keeping in mind for §3's SNR_r calculation: DESI's
*realized* SNR_BGS has very little dynamic range across the BGS Bright
sample, because the survey adaptively exposes each tile until a target
effective time is reached — so SNR-vs-success is nearly flat in the raw
data and the signal only emerges when normalized to a fixed reference
exposure time (or, as used here, by binning on magnitude instead, which
isn't affected by each tile's individual exposure history).

## 3. Implementation plan

Mirrors the existing module structure/conventions in `spec5/instrument/`.

1. **`spec5/instrument/bgs_etc.py`** (new module, parallel to `stellar_etc.py`):
   - Refactor the shared photon-counting core out of `compute_snr` into a
     private helper (telescope area, throughput curves, sky background, CCD
     read noise) so both stellar and BGS code reuse it instead of duplicating
     ~80 lines. Keep `compute_snr`'s public signature/behavior unchanged.
   - `compute_bgs_snr(m_fiber, nexp, texp, fiber_diameter=107,
     half_light_radius=1.5, bgs_type='bright')`: calls the shared core with
     an r-arm wavelength selection and applies the fiber-aperture-loss
     correction from §2.3 step 5 before computing captured photon counts.
   - `compute_redshift_success(m_fiber, nexp, texp, fiber_diameter=107,
     half_light_radius=1.5, bgs_type='bright')`: computes SNR_r via
     `compute_bgs_snr`, then evaluates the calibrated logistic/sigmoid
     `P(success | SNR_r)` from a tabulated data file (loaded via
     `scipy.interpolate`, same pattern as `dist_err_vs_snr_*.csv`).
2. **Data files**: `spec5/instrument/data/bgs_redshift_success_bright.csv` —
   copy of `research/bgs_path_b/bgs_redshift_success_extended_vs_rfibermag.csv`
   (success rate vs. r_fiber mag, extended-source-only, §2.5) plus a
   `..._faint.csv` once that slice is run. No `pyproject.toml` change needed
   (`data/*.csv` is already globbed).
3. **`spec5/instrument/mock_observations.py`**: add `observe_bgs_with_spec5`
   and `observe_bgs_with_desi`, mirroring `observe_with_spec5`/
   `observe_with_desi` — input a mock galaxy catalog (fiber mag, half-light
   radius, true redshift), draw a Bernoulli success outcome per galaxy via
   `compute_redshift_success`, and return observed redshift (true z for
   successes; NaN or a catastrophic-failure draw for failures) plus the
   underlying SNR_r and success probability. Reuse the existing
   `_DESI_MIRROR_MAG_OFFSET` constant for the DESI comparison.
4. **Tests**: `tests/test_compute_bgs_snr.py` and
   `tests/test_redshift_success.py`, following the fixed-value regression
   pattern in `test_compute_snr.py` (hardcoded expected SNR/success-rate at a
   few reference magnitudes/exposure times).
5. **Notebook**: `notebooks/BGS_SNR_ETC.ipynb`, mirroring the structure of
   `SpecS5_SNR_ETC.ipynb` / `mock_observations_demo_with_desi.ipynb` — show
   SNR_r vs. fiber mag, the redshift-success curve, a mock BGS catalog
   observation, and a Spec-S5 vs. DESI comparison.
6. **README**: add a "BGS redshift recovery" section parallel to the existing
   "Measurement error models" section, documenting the success-rate relation,
   its DR1 calibration source, and validity range (r_fiber range, z range,
   BGS Bright vs. Faint).

## 4. Risks

- The core research risk is §2.2 — no published closed-form relation exists;
  Path B (re-deriving from public zcatalog data) is real analysis work, not a
  literature lookup, and should be scoped/time-boxed accordingly before
  committing to an implementation date.
- Fiber-aperture-loss modeling for extended sources is new physics not
  present anywhere in the current point-source `compute_snr` — worth a design
  review once a first version exists, since it changes the shared
  photon-counting core that stellar code also depends on.
