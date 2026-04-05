"""
VQA Behavioral Module - Interaction state testing with Playwright.

This module provides comprehensive behavioral testing capabilities for
comparing interactive states between Figma designs and web implementations.

Key Components:
    - InteractionRunner: Playwright-based interaction harness
    - InteractionSpec: Component interaction definitions
    - StateCapturer: Before/after screenshot management

Usage:
    from vqa.behavioral import (
        InteractionRunner,
        InteractionRunnerConfig,
        InteractionResult,
        StateCapturer,
        StateCaptureConfig,
        StateCapture,
        InteractionType,
        ComponentType,
        run_behavioral_tests,
        capture_hover_states,
        capture_focus_states,
    )
    
    # Run behavioral tests on a Playwright page
    runner = InteractionRunner(page)
    results = await runner.capture_all_states(elements)
    
    # Or use the convenience function
    results = await run_behavioral_tests(page)

This module catches ~20% of QA issues that are about interactive BEHAVIOR:
- Hover states with wrong gradients or colors
- Missing focus rings on focusable elements
- Disabled states that look like active states
- Dropdowns that don't match trigger width
- Sticky elements that don't stay fixed
"""

from .interaction_specs import (
    InteractionType,
    ComponentType,
    InteractionSpec,
    ComponentSpec,
    COMPONENT_SPECS,
    INTERACTION_EXPECTED_CHANGES,
    COMMON_BEHAVIORAL_ISSUES,
    get_spec_for_component,
    get_interaction_spec,
    get_default_interactions,
    get_expected_changes,
    detect_component_type,
    get_interactions_for_element,
    get_undo_action_for_interaction,
    get_wait_time_for_interaction,
    get_validation_rules,
)

from .interaction_runner import (
    InteractionRunner,
    InteractionRunnerConfig,
    InteractionResult,
    run_behavioral_tests,
    capture_hover_states,
    capture_focus_states,
    capture_click_states,
)

from .state_capturer import (
    StateCapturer,
    StateCaptureConfig,
    StateCapture,
    convert_interaction_results_to_captures,
    capture_and_prepare_for_llm,
)


__all__ = [
    # Enums and Types
    "InteractionType",
    "ComponentType",
    
    # Spec Classes
    "InteractionSpec",
    "ComponentSpec",
    
    # Spec Data
    "COMPONENT_SPECS",
    "INTERACTION_EXPECTED_CHANGES",
    "COMMON_BEHAVIORAL_ISSUES",
    
    # Spec Functions
    "get_spec_for_component",
    "get_interaction_spec",
    "get_default_interactions",
    "get_expected_changes",
    "detect_component_type",
    "get_interactions_for_element",
    "get_undo_action_for_interaction",
    "get_wait_time_for_interaction",
    "get_validation_rules",
    
    # Runner Classes
    "InteractionRunner",
    "InteractionRunnerConfig",
    "InteractionResult",
    
    # Runner Convenience Functions
    "run_behavioral_tests",
    "capture_hover_states",
    "capture_focus_states",
    "capture_click_states",
    
    # Capturer Classes
    "StateCapturer",
    "StateCaptureConfig",
    "StateCapture",
    
    # Capturer Convenience Functions
    "convert_interaction_results_to_captures",
    "capture_and_prepare_for_llm",
]
