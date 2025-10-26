"""
Statistical analysis routines (pure inference, non-predictive) for lesion-symptom mapping.
"""

from .chisq import ChiSquareResult, run_chi_square
from .ttest import TTestResult, TTestAssumptionResult, run_ttest

__all__ = [
    "ChiSquareResult",
    "run_chi_square",
    "TTestResult",
    "TTestAssumptionResult",
    "run_ttest",
]
