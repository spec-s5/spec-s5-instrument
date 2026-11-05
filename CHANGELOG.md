# Changelog

## 0.2.0 (unreleased)

Default outputs change substantially in this release.

### Changed
- **Throughput tables** now come from the Spec-S5 Systems Engineering
  Throughput Budget, SS5-0022 v1.1 (PR #8):
  - **DESI source throughput:** now the budget's DESI *star* curve, 1.24× the
    galaxy curve used before, which had been treated as a point source.
  - **Spec-S5 107 µm source throughput:** 2.1% lower (the budget's star curve
    with 9 µm mirror blur).
  - **Spec-S5 120 µm source throughput:** derived from the budget's own fiber
    capture ratio. It is unchanged to within 0.0001.
  - **Sky:** the sky throughputs and sky spectrum are unchanged.
- **Stellar ETC:** Spec-S5 SNR drops 1.2–1.8%, and radial-velocity, distance
  and SNR-derived errors rise by a similar amount. DESI SNR from
  `compute_snr` rises 1.24×.
- **BGS seeing model:** the BGS ETC now uses a Moffat PSF (β = 3.5), as in the
  DESI and Spec-S5 budgets, instead of a Gaussian. Each throughput table has
  its own seeing: `SPEC5_TABLE_SEEING_FWHM = 0.96"` and
  `DESI_TABLE_SEEING_FWHM = 1.1"`. These replace the private
  `_TABLE_SEEING_FWHM`.
- **BGS seeing defaults:** Spec-S5 defaults to 0.96" seeing in
  `compute_bgs_snr`, `compute_redshift_success` and `observe_bgs_with_spec5`.
  DESI stays at 1.1".
- **BGS depth gain** at r_fiber = 21.5, 180 s, for a 1.5" exponential disk:
  0.72 → 0.34 mag (0.26 mag with Spec-S5 also at 1.1"). For a point source:
  0.96 → 0.70 mag (0.49 mag).
- **Capture functions:** `point_source_capture_fraction` and
  `seeing_capture_fraction` default to `psf="moffat"`. Pass `psf="gaussian"`
  for the previous behaviour.

### Added
- **Moffat PSF support:** a `psf` and `beta` option in
  `point_source_capture_fraction` and `seeing_capture_fraction`.
- **DESI seeing option:** `desi_seeing_fwhm` in `compute_bgs_snr`.
- **Provenance tests:** the throughput tables are checked against a CSV export
  of SS5-0022 v1.1 in `tests/data/`.

Earlier changes since 0.1.0 are not itemized here: the BGS model (PR #6) and
source Poisson noise in the stellar ETC (PR #7).
