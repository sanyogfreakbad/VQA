"""
Pass C: Behavioral State Checks.

This pass handles interaction states that require Playwright to trigger:
- Hover states (gradients, color changes, cursor)
- Focus states (outlines, highlights)
- Active/pressed states
- Disabled states
- Dropdown open states
- Toast/notification appearance and dismissal
- Scroll behavior effects

For each interaction, Playwright captures before + after screenshots.
The LLM compares the "after" state against the Figma spec for that state.

This catches ~20% of QA issues that are about interactive BEHAVIOR,
not static appearance.
"""

import asyncio
import uuid
import logging
from typing import List, Dict, Optional, Any
from PIL import Image

from .gemini_client import GeminiClient, GeminiResponse
from .prompts.behavioral_check import (
    build_behavioral_check_prompt,
    build_interaction_context,
    get_expected_changes_for_interaction,
    interpret_behavioral_result,
)
from ..models.finding import Finding, DOMEvidence
from ..models.region import BoundingBox
from ..models.enums import Category, Severity, Confidence, DiffType

logger = logging.getLogger(__name__)


async def run_behavioral_pass(
    client: GeminiClient,
    interaction_results: List[Dict],
    figma_states: Dict[str, Image.Image] = None,
) -> List[Finding]:
    """Run Pass C: Behavioral state validation.
    
    Validates that interactive states match the Figma design spec.
    
    Args:
        client: Initialized GeminiClient
        interaction_results: List of interaction result dicts from InteractionRunner
            Each should have:
            - element: element selector/name
            - interaction: interaction type (hover, click, focus, etc.)
            - before_screenshot: Image before interaction
            - after_screenshot: Image after interaction
            - before_crop: Cropped image before (optional)
            - after_crop: Cropped image after (optional)
            - element_type: Component type
            - element_bbox: Bounding box
        figma_states: Optional dict mapping "element_interaction" to Figma state images
    
    Returns:
        List of Finding objects for behavioral differences
    """
    if not interaction_results:
        logger.info("Pass C: No interaction results to validate")
        return []
    
    logger.info(f"Pass C: Validating {len(interaction_results)} interactive states")
    
    # Build the system prompt
    system_prompt = build_behavioral_check_prompt(include_schema=True)
    
    findings = []
    
    # Process each interaction
    for idx, result in enumerate(interaction_results):
        try:
            interaction_findings = await validate_single_interaction(
                client=client,
                interaction_result=result,
                figma_state=_get_figma_state(result, figma_states),
                system_prompt=system_prompt,
                index=idx,
            )
            findings.extend(interaction_findings)
        except Exception as e:
            logger.warning(f"Failed to validate interaction {idx}: {e}")
            continue
    
    logger.info(f"Pass C found {len(findings)} behavioral issues")
    
    return findings


def _get_figma_state(
    result: Dict,
    figma_states: Dict[str, Image.Image] = None,
) -> Optional[Image.Image]:
    """Get the Figma state image for an interaction.
    
    Args:
        result: Interaction result
        figma_states: Dict of Figma state images
    
    Returns:
        Figma state image if available
    """
    if not figma_states:
        return None
    
    element = result.get("element", "")
    interaction = result.get("interaction", "")
    
    # Try different key formats
    keys_to_try = [
        f"{element}_{interaction}",
        f"{interaction}_{element}",
        element,
        interaction,
    ]
    
    for key in keys_to_try:
        if key in figma_states:
            return figma_states[key]
    
    return None


async def validate_single_interaction(
    client: GeminiClient,
    interaction_result: Dict,
    figma_state: Optional[Image.Image],
    system_prompt: str,
    index: int,
) -> List[Finding]:
    """Validate a single interaction state.
    
    Args:
        client: GeminiClient instance
        interaction_result: Single interaction result
        figma_state: Expected Figma state image
        system_prompt: System prompt
        index: Index for logging
    
    Returns:
        List of findings for this interaction
    """
    element = interaction_result.get("element", "Unknown element")
    interaction = interaction_result.get("interaction", "hover")
    element_type = interaction_result.get("element_type", "unknown")
    
    # Get images
    after_crop = interaction_result.get("after_crop")
    after_screenshot = interaction_result.get("after_screenshot")
    
    # Use crop if available, otherwise use full screenshot
    web_state = after_crop if after_crop else after_screenshot
    
    if not web_state:
        logger.warning(f"No after-state image for interaction {index}")
        return []
    
    # If we don't have a Figma state, use before state as reference
    # (assumes before state is the "default" state which should match Figma)
    reference_state = figma_state
    if not reference_state:
        reference_state = interaction_result.get("before_crop") or interaction_result.get("before_screenshot")
    
    if not reference_state:
        logger.warning(f"No reference state for interaction {index}")
        return []
    
    # Build context
    expected_changes = get_expected_changes_for_interaction(interaction)
    context = build_interaction_context(
        interaction_type=interaction,
        element_name=element,
        element_type=element_type,
        interaction_index=index,
        expected_changes=expected_changes,
    )
    
    # Build prompt
    prompt = f"""
{context}

Analyze these two images:
- Image 1: Expected state (Figma design / reference)
- Image 2: Actual state after {interaction} interaction on the web

Does the web's interactive state match the expected design?
Focus on: {', '.join(expected_changes)}
"""
    
    # Call the LLM
    response = await client.analyze_images(
        images=[reference_state, web_state],
        prompt=prompt,
        system_prompt=system_prompt,
    )
    
    if not response.success:
        logger.debug(f"Behavioral check failed for {element}: {response.error}")
        return []
    
    # Parse response
    return parse_behavioral_response(
        response.data,
        interaction_result,
    )


def parse_behavioral_response(
    response_data: Any,
    interaction_result: Dict,
) -> List[Finding]:
    """Parse behavioral check response into findings.
    
    Args:
        response_data: LLM response data
        interaction_result: Original interaction result
    
    Returns:
        List of Finding objects
    """
    if not response_data:
        return []
    
    # Handle dict response
    if not isinstance(response_data, dict):
        return []
    
    state_matches = response_data.get("state_matches", True)
    
    # If states match, no findings
    if state_matches:
        return []
    
    differences = response_data.get("differences", [])
    if not differences:
        return []
    
    findings = []
    element = interaction_result.get("element", "Unknown")
    interaction = interaction_result.get("interaction", "hover")
    element_type = interaction_result.get("element_type", "unknown")
    
    # Get bounding box
    bbox_data = interaction_result.get("element_bbox")
    bbox = None
    if bbox_data:
        if isinstance(bbox_data, dict):
            bbox = BoundingBox.from_dict(bbox_data)
        elif hasattr(bbox_data, "x"):
            bbox = BoundingBox(
                x=bbox_data.x,
                y=bbox_data.y,
                width=bbox_data.width,
                height=bbox_data.height,
            )
    
    for diff in differences:
        # Map property to diff type
        property_name = diff.get("property", "unknown")
        diff_type = map_behavioral_property_to_diff_type(property_name, interaction)
        
        # Map severity
        severity_str = diff.get("severity", "minor").lower()
        severity_map = {
            "critical": Severity.CRITICAL,
            "major": Severity.MAJOR,
            "minor": Severity.MINOR,
        }
        severity = severity_map.get(severity_str, Severity.MINOR)
        
        # Map confidence
        conf_val = diff.get("confidence", 0.7)
        if conf_val >= 0.85:
            confidence = Confidence.HIGH
        elif conf_val >= 0.6:
            confidence = Confidence.MEDIUM
        else:
            confidence = Confidence.LOW
        
        # Build reasoning
        expected = diff.get("expected", "N/A")
        actual = diff.get("actual", "N/A")
        reasoning = diff.get("reasoning", "")
        
        visual_reasoning = (
            f"{interaction.capitalize()} state: Expected '{expected}', got '{actual}'. "
            f"{reasoning}"
        )
        
        finding = Finding(
            id=f"pass_c_{uuid.uuid4().hex[:8]}",
            category=Category.BEHAVIORAL,
            diff_type=diff_type,
            severity=severity,
            confidence=confidence,
            element_name=f"{element} ({interaction} state)",
            element_type=element_type,
            web_bbox=bbox,
            dom_evidence=DOMEvidence(
                figma_value=expected,
                web_value=actual,
                delta=f"Behavioral: {interaction}",
            ),
            visual_reasoning=visual_reasoning,
            source="gemini_visual",
            pass_name="pass_c",
            is_interactive=True,
        )
        
        findings.append(finding)
    
    return findings


def map_behavioral_property_to_diff_type(
    property_name: str,
    interaction: str,
) -> DiffType:
    """Map behavioral property to DiffType.
    
    Args:
        property_name: CSS property or behavioral property name
        interaction: Interaction type
    
    Returns:
        Appropriate DiffType
    """
    prop_lower = property_name.lower()
    
    if "color" in prop_lower or "gradient" in prop_lower:
        return DiffType.COLOR_BACKGROUND
    
    if "shadow" in prop_lower:
        return DiffType.SHADOW_BOX
    
    if "outline" in prop_lower or "ring" in prop_lower or "focus" in prop_lower:
        return DiffType.OUTLINE
    
    if "transform" in prop_lower or "scale" in prop_lower:
        return DiffType.TRANSFORM
    
    if "cursor" in prop_lower:
        return DiffType.CURSOR
    
    if "opacity" in prop_lower:
        return DiffType.OPACITY
    
    if "border" in prop_lower:
        return DiffType.BORDER_RADIUS
    
    if "height" in prop_lower or "width" in prop_lower:
        return DiffType.SIZE_HEIGHT
    
    return DiffType.COMPONENT_STATE


async def validate_hover_states(
    client: GeminiClient,
    elements: List[Dict],
    page: Any,
    figma_states: Dict[str, Image.Image] = None,
) -> List[Finding]:
    """Convenience function to validate hover states for a list of elements.
    
    Captures before/after screenshots for each element's hover state.
    
    Args:
        client: GeminiClient instance
        elements: List of element specs with selectors
        page: Playwright page object
        figma_states: Optional dict mapping element keys to Figma state images
    
    Returns:
        List of findings
    """
    try:
        from ..behavioral.interaction_runner import InteractionRunner, InteractionType
        
        runner = InteractionRunner(page)
        
        for elem in elements:
            if "interactions" not in elem:
                elem["interactions"] = [InteractionType.HOVER]
        
        results = await runner.capture_all_states(elements)
        
        result_dicts = [r.to_dict() if hasattr(r, 'to_dict') else r for r in results]
        
        return await run_behavioral_pass(client, result_dicts, figma_states)
    
    except ImportError:
        logger.warning("InteractionRunner not available, skipping hover validation")
        return []
    except Exception as e:
        logger.error(f"Failed to validate hover states: {e}")
        return []


async def validate_focus_states(
    client: GeminiClient,
    elements: List[Dict],
    page: Any,
    figma_states: Dict[str, Image.Image] = None,
) -> List[Finding]:
    """Convenience function to validate focus states for a list of elements.
    
    Args:
        client: GeminiClient instance
        elements: List of element specs with selectors
        page: Playwright page object
        figma_states: Optional dict mapping element keys to Figma state images
    
    Returns:
        List of findings
    """
    try:
        from ..behavioral.interaction_runner import InteractionRunner, InteractionType
        
        runner = InteractionRunner(page)
        
        for elem in elements:
            if "interactions" not in elem:
                elem["interactions"] = [InteractionType.FOCUS]
        
        results = await runner.capture_all_states(elements)
        result_dicts = [r.to_dict() if hasattr(r, 'to_dict') else r for r in results]
        
        return await run_behavioral_pass(client, result_dicts, figma_states)
    
    except ImportError:
        logger.warning("InteractionRunner not available, skipping focus validation")
        return []
    except Exception as e:
        logger.error(f"Failed to validate focus states: {e}")
        return []


async def validate_click_states(
    client: GeminiClient,
    elements: List[Dict],
    page: Any,
    figma_states: Dict[str, Image.Image] = None,
) -> List[Finding]:
    """Convenience function to validate click states (toggles, dropdowns).
    
    Args:
        client: GeminiClient instance
        elements: List of element specs with selectors
        page: Playwright page object
        figma_states: Optional dict mapping element keys to Figma state images
    
    Returns:
        List of findings
    """
    try:
        from ..behavioral.interaction_runner import InteractionRunner, InteractionType
        
        runner = InteractionRunner(page)
        
        for elem in elements:
            if "interactions" not in elem:
                elem["interactions"] = [InteractionType.CLICK]
        
        results = await runner.capture_all_states(elements)
        result_dicts = [r.to_dict() if hasattr(r, 'to_dict') else r for r in results]
        
        return await run_behavioral_pass(client, result_dicts, figma_states)
    
    except ImportError:
        logger.warning("InteractionRunner not available, skipping click validation")
        return []
    except Exception as e:
        logger.error(f"Failed to validate click states: {e}")
        return []


async def run_full_behavioral_pass(
    client: GeminiClient,
    page: Any,
    figma_states: Dict[str, Image.Image] = None,
    element_selectors: List[str] = None,
    max_elements: int = 50,
) -> List[Finding]:
    """Run a complete behavioral pass on a page.
    
    Discovers interactive elements automatically and tests all default interactions.
    
    Args:
        client: GeminiClient instance
        page: Playwright page object
        figma_states: Optional dict mapping element keys to Figma state images
        element_selectors: Optional list of specific selectors to test
        max_elements: Maximum number of elements to test
    
    Returns:
        List of findings
    """
    try:
        from ..behavioral.interaction_runner import InteractionRunner
        
        runner = InteractionRunner(page)
        
        if element_selectors:
            elements = [{"selector": s} for s in element_selectors]
        else:
            elements = await runner.discover_interactive_elements()
            if len(elements) > max_elements:
                elements = elements[:max_elements]
                logger.info(f"Limited to {max_elements} elements for behavioral testing")
        
        results = await runner.capture_all_states(elements)
        result_dicts = [r.to_dict() if hasattr(r, 'to_dict') else r for r in results]
        
        return await run_behavioral_pass(client, result_dicts, figma_states)
    
    except ImportError:
        logger.warning("Behavioral module not available, skipping behavioral pass")
        return []
    except Exception as e:
        logger.error(f"Failed to run full behavioral pass: {e}")
        return []


async def capture_and_validate_interactions(
    client: GeminiClient,
    page: Any,
    interactions: List[Dict[str, Any]],
    figma_states: Dict[str, Image.Image] = None,
) -> List[Finding]:
    """Capture and validate specific interactions.
    
    Each interaction dict should have:
    - selector: CSS selector
    - interaction: Interaction type ("hover", "click", "focus")
    - element_name: Optional human-readable name
    
    Args:
        client: GeminiClient instance
        page: Playwright page object
        interactions: List of interaction specifications
        figma_states: Optional Figma state images
    
    Returns:
        List of findings
    """
    try:
        from ..behavioral.interaction_runner import InteractionRunner, InteractionType
        
        runner = InteractionRunner(page)
        
        elements = []
        for interaction in interactions:
            selector = interaction.get("selector")
            if not selector:
                continue
            
            interaction_type = interaction.get("interaction", "hover")
            if isinstance(interaction_type, str):
                try:
                    interaction_type = InteractionType(interaction_type)
                except ValueError:
                    interaction_type = InteractionType.HOVER
            
            elements.append({
                "selector": selector,
                "interactions": [interaction_type],
                "element_name": interaction.get("element_name", selector),
            })
        
        results = await runner.capture_all_states(elements)
        result_dicts = [r.to_dict() if hasattr(r, 'to_dict') else r for r in results]
        
        return await run_behavioral_pass(client, result_dicts, figma_states)
    
    except ImportError:
        logger.warning("Behavioral module not available")
        return []
    except Exception as e:
        logger.error(f"Failed to capture and validate interactions: {e}")
        return []
