# Spec-S5 Instrument

Spec-S5 Instrument Characteristics

## Installation

Requires Python 3.9+ and depends on `numpy`, `scipy`, and `matplotlib`.

**From source (recommended for development):**
```bash
git clone https://github.com/your-org/spec-s5-instrument.git
cd spec-s5-instrument
pip install -e .
```

**As a regular install from source:**
```bash
pip install .
```

## Usage

```python
from spec5.instrument.stellar_etc import compute_snr, plot_snr_comparison, compute_measurement_errors

# Compute SNR for a mag=22 source, 1-hour exposure split into 3 sub-exposures
result = compute_snr(m=22.0, nexp=3, texp=3600.0)

# Plot photon rate and SNR vs wavelength
plot_snr_comparison(result)

# Compute measurement errors from an LSST z-band magnitude (default)
errors = compute_measurement_errors(magnitude=22.0, pm_model='gaia_dr5', star_type='giant')
print(errors['vrad_err'])       # km/s
print(errors['pm_err'])         # mas/yr
print(errors['dist_err_frac'])  # fractional distance error

# Compute measurement errors from a Gaia G magnitude
errors = compute_measurement_errors(magnitude=22.0, mag_band='gaia_g',
                                    pm_model='gaia_dr5', star_type='giant')

# Vectorised: pass an array of magnitudes
import numpy as np
mags = np.arange(19, 24)
errors = compute_measurement_errors(magnitude=mags, pm_model='lsst1', star_type='giant')
```

## Measurement error models

### Radial velocity error

The radial velocity error is derived from the empirical DESI relation between
SNR in the z-arm (7470–9800 Å) and radial velocity precision:

```
log10(σ_vrad) = 1.389 − 0.975 × log10(SNR_z) − 0.975 × log10(0.8)
```

This relation is calibrated on DESI stellar spectra and applies to metal-poor
RGB stars observed in the Spec-S5 z-arm.  A configurable systematic floor
`vrad_sys` (default 0.6 km/s) is added in quadrature with the statistical
error.

### Proper motion error

Two sets of proper motion models are supported:

**Gaia DR4 / DR5** — analytical scaling relations from the
[ESA Gaia science-performance page](https://www.cosmos.esa.int/web/gaia/science-performance):

```
σ_ϖ [μas]    = T × sqrt(40 + 800·z + 30·z²),   z = max(10^−0.8, 10^(0.4·(G−15)))
σ_μ [μas/yr] = f × σ_ϖ   (sky-averaged)
```

| Release   | T (temporal factor) | f (PM scale) |
|-----------|---------------------|--------------|
| Gaia DR4  | 0.749               | 0.54         |
| Gaia DR5  | 0.527               | 0.27         |

Valid for G ≤ 20.7; returns NaN for fainter stars.  The lookup uses Gaia G
magnitude; photometric transformations from LSST z or i are applied
automatically using the RTN-099 colour relations.

**LSST Y1 / Y10** (`lsst1`, `lsst10`) — tabulated proper motion errors as a
function of LSST i magnitude, derived from LSST survey projections for DECam.
Bright-end extrapolation is clamped to the table boundary value; faint-end
extrapolation uses a linear fit.  The lookup uses LSST i magnitude, converted
from the input band via a fixed i−z = 0.2 colour offset appropriate for
metal-poor halo RGB stars.

### Distance error

The fractional distance error as a function of spectroscopic SNR is taken from
**Li et al. (2025)**, tabulated separately for RGB giants (log g < 3.8) and
dwarfs.  The interpolation uses cubic splines; values outside the tabulated SNR
range are clamped to the nearest boundary value.

## BGS redshift recovery

```python
from spec5.instrument.bgs_etc import compute_bgs_snr, compute_redshift_success

# r-arm continuum SNR for a BGS galaxy (r_fiber_mag in the FIBERFLUX_R
# convention: 1.5" fiber, 1" seeing), half_light_radius in arcsec
snr = compute_bgs_snr(r_fiber_mag=19.5, nexp=1, texp=180.0, half_light_radius=1.5)
print(snr['snr_r_s5'], snr['snr_r_d'])

# Probability of a successful DESI/Spec-S5 redshift measurement
p_spec5 = compute_redshift_success(19.5, instrument='spec5')
p_desi  = compute_redshift_success(19.5, instrument='desi')
```

Redshift recovery is modeled as a **binary** success/failure outcome,
using the Hahn et al. 2023 criteria that don't need a deep comparison spectrum:
`ZWARN==0 & DELTACHI2>40 & SPECTYPE=='GALAXY'`. There is no continuous
redshift-error model — DESI's own pipeline doesn't publish one either, since
Δχ² is a detection statistic, not a precision metric.

The success-rate-vs-`r_fiber_mag` relation is empirically derived from the
public DESI DR1 BGS_BRIGHT LSS catalog (iron reduction, v1.5), restricted to
extended sources (`SHAPE_R > 1.0"`) to exclude point-source/compact
contaminants that Redrock correctly fails to classify as galaxies — see
`docs/bgs_path_b_results.md` for the full derivation, including why a naive
fit (without that restriction) is misleadingly non-monotonic. Only the
**BGS Bright** class is currently calibrated (`bgs_type='faint'` raises
`ValueError`; see `docs/bgs_plan.md`).

For extended sources, only part of the galaxy's light falls within the fiber.
This is modeled with a Sersic profile convolved with Moffat seeing (beta = 3.5,
`spec5.instrument.galaxy_photometry.seeing_capture_fraction`), by default an
exponential disk (`sersic_n=1.0`), and is applied relative to a point source
because the throughput tables already include point-source fiber losses. The
tables come from the Spec-S5 Systems Engineering Throughput Budget, SS5-0022
v1.1, which assumes 0.96" seeing for Spec-S5 (the median DESI measured at the
Mayall) and 1.1" for DESI (DESI's nominal design value); each instrument is
observed at its table's seeing by default. Predicting Spec-S5's
redshift-success probability (`instrument='spec5'`) finds the magnitude at which DESI, at its reference 1x180 s effective exposure in
nominal conditions, reaches the SNR Spec-S5 achieves, then evaluates the
DESI-calibrated curve there. `seeing_fwhm` and `sky_scale` set Spec-S5's observing
conditions; by default the tabulated moonless-continuum sky applies, so `texp` is
an effective exposure under that sky. Radii below 1" raise a warning because they
fall outside the calibration sample, and magnitudes past the `r_fiber < 22.9`
sample limit return 0. At 1x180 s with both instruments in the same seeing,
Spec-S5 reaches about 0.27 mag deeper than DESI for a 1.5" exponential disk and
0.55 mag for a point source (0.26 and 0.49 mag at 1.1"). Setting the default
Spec-S5 prediction (0.96") against the DESI curve (1.1" effective time) gives
0.34 and 0.70 mag instead; see `docs/bgs_path_b_results.md`.

Mock observations (drawing a per-galaxy Bernoulli success outcome) are
available via `observe_bgs_with_spec5` / `observe_bgs_with_desi` in
`spec5.instrument.mock_observations`, mirroring the stellar
`observe_with_spec5` / `observe_with_desi` pattern.

See `docs/bgs_plan.md` for the full research/implementation plan and
`research/bgs_path_b/` for the calibration scripts and derived data.

## Mock fiber assignment

`mock_fiber_assign` selects a spatially uniform, density-limited subsample of
a catalog within a circular field of view. Positions `x, y` are in arcmin
relative to the field centre.

```python
from spec5.instrument import mock_fiber_assign

observed = mock_fiber_assign(catalog, n_fibers=13000, fov_radius_arcmin=33.0,
                             priority_column='zmag', seed=0)
```

The fibers are spread over a grid in proportion to each cell's area inside the
field. Each cell keeps up to its allocation of objects, chosen by priority
within the cell. Fibers left over in sparse cells are not moved elsewhere, so a
clustered catalog returns fewer targets than fibers. Fiber collisions and
patrol radii are not modelled. See `notebooks/mock_fiberassign_demo.ipynb`.

## Running the tests

```bash
pip install pytest
pytest tests/
```
