"""Extraction module for VQA pipeline.

Provides multi-state and responsive capture capabilities.
"""

from .state_capture import (
    StateCapture,
    CapturedState,
    capture_hover_state,
    capture_focus_state,
    capture_active_state,
    capture_breakpoints,
    capture_multi_state,
)

__all__ = [
    "StateCapture",
    "CapturedState",
    "capture_hover_state",
    "capture_focus_state",
    "capture_active_state",
    "capture_breakpoints",
    "capture_multi_state",
]
