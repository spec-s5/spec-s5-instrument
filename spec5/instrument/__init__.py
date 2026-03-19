"""
spec5.instrument — Spec-S5 Exposure Time Calculator
"""

from .stellar_etc import compute_snr, plot_snr_comparison, compute_measurement_errors
from .mock_observations import galactocentric_to_observed, observe_with_spec5

__all__ = ["compute_snr", "plot_snr_comparison", "compute_measurement_errors",
           "galactocentric_to_observed", "observe_with_spec5"]
