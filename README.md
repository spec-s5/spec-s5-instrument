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

## Running the tests

```bash
pip install pytest
pytest tests/
```
