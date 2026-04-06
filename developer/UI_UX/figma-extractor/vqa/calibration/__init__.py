"""Calibration module for VQA pipeline.

Provides automatic threshold tuning based on user feedback.
"""

from .auto_tuner import (
    AutoTuner,
    ThresholdSuggestion,
    suggest_threshold_adjustments,
    get_auto_tuner,
)

__all__ = [
    "AutoTuner",
    "ThresholdSuggestion",
    "suggest_threshold_adjustments",
    "get_auto_tuner",
]
