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

## Running the tests

```bash
pip install pytest
pytest tests/
```
