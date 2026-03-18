# spec-s5-instrument
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
from spec_s5_instrument import compute_snr, plot_snr_comparison, compute_measurement_errors

# Compute SNR for a mag=22 source, 1-hour exposure split into 3 sub-exposures
result = compute_snr(m=22.0, nexp=3, texp=3600.0)

# Plot photon rate and SNR vs wavelength
plot_snr_comparison(result)

# Compute radial velocity, proper motion, and distance errors
errors = compute_measurement_errors(magnitude=22.0, pm_model='gaia_dr5', star_type='giant')
print(errors['vrad_err'])       # km/s
print(errors['pm_err'])         # mas/yr
print(errors['dist_err_frac'])  # fractional distance error
```
