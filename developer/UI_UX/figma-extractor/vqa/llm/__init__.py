"""VQA LLM Module - Gemini API integration for visual analysis.

This module provides the LLM-powered visual analysis layer for the VQA pipeline:

- gemini_client: Async API wrapper with batching, retry, and token management
- pass_a_blind: Blind visual diff analysis with full screenshots
- pass_b_targeted: Targeted validation of DOM-detected differences
- pass_c_behavioral: Interactive state validation
- refinement_pass: Stage 4 re-examination of uncertain findings
- prompts/: Prompt templates for each analysis pass
- few_shot/: Calibration examples for improved accuracy
"""

from .gemini_client import (
    GeminiClient,
    GeminiResponse,
    BatchResult,
    create_client,
)

from .pass_a_blind import (
    run_blind_visual_pass,
    run_blind_visual_pass_with_retry,
    parse_blind_visual_response,
)

from .pass_b_targeted import (
    run_targeted_validation_pass,
    run_targeted_validation_single,
    parse_validation_response,
)

from .pass_c_behavioral import (
    run_behavioral_pass,
    validate_hover_states,
)

from .refinement_pass import (
    run_refinement_pass,
    refine_single_finding,
    batch_refine_findings,
    filter_for_refinement,
)

from .few_shot.calibration_store import (
    CalibrationStore,
    get_calibration_store,
)

__all__ = [
    # Gemini client
    "GeminiClient",
    "GeminiResponse",
    "BatchResult",
    "create_client",
    # Pass A: Blind visual
    "run_blind_visual_pass",
    "run_blind_visual_pass_with_retry",
    "parse_blind_visual_response",
    # Pass B: Targeted validation
    "run_targeted_validation_pass",
    "run_targeted_validation_single",
    "parse_validation_response",
    # Pass C: Behavioral
    "run_behavioral_pass",
    "validate_hover_states",
    # Refinement pass
    "run_refinement_pass",
    "refine_single_finding",
    "batch_refine_findings",
    "filter_for_refinement",
    # Calibration
    "CalibrationStore",
    "get_calibration_store",
]
