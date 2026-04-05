"""VQA LLM Prompts - Prompt templates for Gemini visual analysis.

This module contains prompt templates for each analysis pass:

- blind_visual: Pass A prompts for blind visual diff
- targeted_validation: Pass B prompts for DOM-detected difference validation
- behavioral_check: Pass C prompts for interactive state checking
- refinement: Stage 4 prompts for uncertain finding re-examination
"""

from .blind_visual import (
    BLIND_VISUAL_SYSTEM_PROMPT,
    BLIND_VISUAL_RESPONSE_SCHEMA,
    build_blind_visual_prompt,
    normalize_category,
    normalize_severity,
    confidence_to_enum,
    CATEGORY_MAPPING,
    SEVERITY_MAPPING,
)

from .targeted_validation import (
    TARGETED_VALIDATION_SYSTEM_PROMPT,
    TARGETED_VALIDATION_RESPONSE_SCHEMA,
    TARGETED_BATCH_RESPONSE_SCHEMA,
    build_targeted_validation_prompt,
    build_pair_context,
    build_batch_prompt,
    interpret_verdict,
    normalize_property_type,
)

from .behavioral_check import (
    BEHAVIORAL_CHECK_SYSTEM_PROMPT,
    BEHAVIORAL_RESPONSE_SCHEMA,
    BEHAVIORAL_BATCH_RESPONSE_SCHEMA,
    build_behavioral_check_prompt,
    build_interaction_context,
    get_expected_changes_for_interaction,
    interpret_behavioral_result,
    INTERACTION_EXPECTED_CHANGES,
    COMMON_BEHAVIORAL_ISSUES,
)

from .refinement import (
    REFINEMENT_SYSTEM_PROMPT,
    REFINEMENT_RESPONSE_SCHEMA,
    build_refinement_prompt,
    build_finding_context,
    build_few_shot_block,
    format_few_shot_example,
    interpret_refinement_result,
    adjust_confidence_from_cot,
)

__all__ = [
    # Blind visual (Pass A)
    "BLIND_VISUAL_SYSTEM_PROMPT",
    "BLIND_VISUAL_RESPONSE_SCHEMA",
    "build_blind_visual_prompt",
    "normalize_category",
    "normalize_severity",
    "confidence_to_enum",
    "CATEGORY_MAPPING",
    "SEVERITY_MAPPING",
    # Targeted validation (Pass B)
    "TARGETED_VALIDATION_SYSTEM_PROMPT",
    "TARGETED_VALIDATION_RESPONSE_SCHEMA",
    "TARGETED_BATCH_RESPONSE_SCHEMA",
    "build_targeted_validation_prompt",
    "build_pair_context",
    "build_batch_prompt",
    "interpret_verdict",
    "normalize_property_type",
    # Behavioral check (Pass C)
    "BEHAVIORAL_CHECK_SYSTEM_PROMPT",
    "BEHAVIORAL_RESPONSE_SCHEMA",
    "BEHAVIORAL_BATCH_RESPONSE_SCHEMA",
    "build_behavioral_check_prompt",
    "build_interaction_context",
    "get_expected_changes_for_interaction",
    "interpret_behavioral_result",
    "INTERACTION_EXPECTED_CHANGES",
    "COMMON_BEHAVIORAL_ISSUES",
    # Refinement
    "REFINEMENT_SYSTEM_PROMPT",
    "REFINEMENT_RESPONSE_SCHEMA",
    "build_refinement_prompt",
    "build_finding_context",
    "build_few_shot_block",
    "format_few_shot_example",
    "interpret_refinement_result",
    "adjust_confidence_from_cot",
]
