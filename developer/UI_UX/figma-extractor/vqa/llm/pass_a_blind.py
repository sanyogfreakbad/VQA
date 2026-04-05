"""
Pass A: Blind Visual Diff.

This pass sends full screenshots to the LLM with NO DOM hints.
The model identifies all visual differences it can see independently.

This catches issues the DOM comparator CANNOT detect:
- Missing shadows (the model sees "flat vs elevated")
- Wrong icons (visual comparison, not ID comparison)
- Visual hierarchy issues
- Missing dividers
- Cropped elements
- Image differences
- General "looks wrong" problems

This is the most important LLM pass because it simulates how a human
would review the implementation by just looking at both versions.
"""

import asyncio
import uuid
import logging
from typing import List, Dict, Optional, Any
from PIL import Image

from .gemini_client import GeminiClient, GeminiResponse
from .prompts.blind_visual import (
    build_blind_visual_prompt,
    normalize_category,
    normalize_severity,
    confidence_to_enum,
)
from ..models.finding import Finding, DOMEvidence
from ..models.region import BoundingBox
from ..models.enums import Category, Severity, Confidence, DiffType

logger = logging.getLogger(__name__)


async def run_blind_visual_pass(
    client: GeminiClient,
    figma_screenshot: Image.Image,
    web_screenshot: Image.Image,
    page_width: int = None,
    page_height: int = None,
    additional_context: str = None,
) -> List[Finding]:
    """Run Pass A: Blind visual diff analysis.
    
    Sends both full screenshots to Gemini without any DOM hints.
    The model independently identifies visual differences.
    
    Args:
        client: Initialized GeminiClient
        figma_screenshot: Full Figma design screenshot
        web_screenshot: Full web implementation screenshot
        page_width: Optional page width for bbox calculation
        page_height: Optional page height for bbox calculation
        additional_context: Optional extra context to include
    
    Returns:
        List of Finding objects from blind visual analysis
    """
    # Use image dimensions if not provided
    if not page_width:
        page_width = figma_screenshot.width
    if not page_height:
        page_height = figma_screenshot.height
    
    # Build the prompt
    system_prompt = build_blind_visual_prompt(
        include_schema=True,
        additional_context=additional_context,
    )
    
    logger.info(f"Running Pass A blind analysis on {page_width}x{page_height} screenshots")
    
    # Call the LLM
    response = await client.blind_analysis(
        figma_screenshot=figma_screenshot,
        web_screenshot=web_screenshot,
        system_prompt=system_prompt,
    )
    
    if not response.success:
        logger.error(f"Pass A failed: {response.error}")
        return []
    
    # Parse the response into findings
    findings = parse_blind_visual_response(
        response.data,
        page_width,
        page_height,
    )
    
    logger.info(f"Pass A found {len(findings)} visual differences")
    
    return findings


def parse_blind_visual_response(
    response_data: Any,
    page_width: int,
    page_height: int,
) -> List[Finding]:
    """Parse LLM response into Finding objects.
    
    Args:
        response_data: Parsed JSON from LLM (list of findings)
        page_width: Page width for converting percentage bboxes
        page_height: Page height for converting percentage bboxes
    
    Returns:
        List of normalized Finding objects
    """
    if not response_data:
        return []
    
    # Handle both list and dict responses
    if isinstance(response_data, dict):
        # Single finding wrapped in dict
        if "findings" in response_data:
            items = response_data["findings"]
        elif "differences" in response_data:
            items = response_data["differences"]
        else:
            items = [response_data]
    elif isinstance(response_data, list):
        items = response_data
    else:
        logger.warning(f"Unexpected response type: {type(response_data)}")
        return []
    
    findings = []
    
    for item in items:
        try:
            finding = parse_single_finding(item, page_width, page_height)
            if finding:
                findings.append(finding)
        except Exception as e:
            logger.warning(f"Failed to parse finding: {e}")
            continue
    
    return findings


def parse_single_finding(
    item: Dict,
    page_width: int,
    page_height: int,
) -> Optional[Finding]:
    """Parse a single finding item from LLM response.
    
    Args:
        item: Single finding dict from LLM
        page_width: Page width for bbox conversion
        page_height: Page height for bbox conversion
    
    Returns:
        Finding object or None if parsing fails
    """
    # Extract and normalize category
    raw_category = item.get("category", "other")
    normalized_category = normalize_category(raw_category)
    
    try:
        category = Category(normalized_category)
    except ValueError:
        category = Category.OTHER
    
    # Extract and normalize severity
    raw_severity = item.get("severity", "minor")
    normalized_severity = normalize_severity(raw_severity)
    
    try:
        severity = Severity(normalized_severity)
    except ValueError:
        severity = Severity.MINOR
    
    # Extract confidence
    raw_confidence = item.get("confidence", 0.7)
    if isinstance(raw_confidence, str):
        try:
            raw_confidence = float(raw_confidence)
        except ValueError:
            raw_confidence = 0.7
    
    confidence_enum = confidence_to_enum(raw_confidence)
    
    try:
        confidence = Confidence(confidence_enum)
    except ValueError:
        confidence = Confidence.MEDIUM
    
    # Map category to diff type
    diff_type = map_category_to_diff_type(category, item)
    
    # Parse bounding box from percentage values
    bbox = None
    bbox_data = item.get("bounding_box_estimate") or item.get("bbox")
    if bbox_data:
        bbox = parse_percentage_bbox(bbox_data, page_width, page_height)
    
    # Build element name from location
    location = item.get("location_description", "")
    element_name = location or f"Visual element ({raw_category})"
    
    # Build visual reasoning from observations
    figma_obs = item.get("figma_observation", "")
    web_obs = item.get("web_observation", "")
    reasoning = item.get("reasoning", "")
    
    visual_reasoning = ""
    if figma_obs:
        visual_reasoning += f"Figma: {figma_obs}. "
    if web_obs:
        visual_reasoning += f"Web: {web_obs}. "
    if reasoning:
        visual_reasoning += reasoning
    
    # Create the finding
    finding = Finding(
        id=f"pass_a_{uuid.uuid4().hex[:8]}",
        category=category,
        diff_type=diff_type,
        severity=severity,
        confidence=confidence,
        element_name=element_name,
        web_bbox=bbox,
        visual_reasoning=visual_reasoning.strip() or None,
        source="gemini_visual",
        pass_name="pass_a",
        saliency_score=calculate_saliency_from_bbox(bbox, page_height) if bbox else 0.5,
    )
    
    return finding


def map_category_to_diff_type(category: Category, item: Dict) -> DiffType:
    """Map visual category to the most appropriate DiffType.
    
    Args:
        category: Normalized category
        item: Original item dict for additional context
    
    Returns:
        Most appropriate DiffType
    """
    category_to_diff = {
        Category.SHADOW: DiffType.SHADOW_BOX,
        Category.ICONS: DiffType.ICON_DIFFERENT,
        Category.BORDER: DiffType.DIVIDER_MISSING,
        Category.TEXT: DiffType.TEXT_FONT_SIZE,  # Generic text diff
        Category.SPACING: DiffType.SPACING_ITEM_GAP,
        Category.PADDING: DiffType.SPACING_PADDING_TOP,
        Category.COLOR: DiffType.COLOR_BACKGROUND,
        Category.COMPONENTS: DiffType.COMPONENT_STATE,
        Category.BEHAVIORAL: DiffType.COMPONENT_STATE,
        Category.OVERFLOW: DiffType.TEXT_OVERFLOW,
        Category.LAYOUT: DiffType.LAYOUT_MODE,
        Category.POSITION: DiffType.POSITION_TYPE,
        Category.MISSING_ELEMENTS: DiffType.ELEMENT_MISSING_IN_WEB,
        Category.IMAGES: DiffType.IMAGE_DIFFERENT,
    }
    
    diff_type = category_to_diff.get(category, DiffType.VISUAL_HIERARCHY)
    
    # Refine based on raw category text
    raw_cat = item.get("category", "").lower()
    
    if "missing" in raw_cat:
        if "icon" in raw_cat:
            return DiffType.ICON_MISSING
        elif "image" in raw_cat:
            return DiffType.IMAGE_MISSING
        elif "shadow" in raw_cat:
            return DiffType.SHADOW_BOX
        elif "divider" in raw_cat or "border" in raw_cat:
            return DiffType.DIVIDER_MISSING
        else:
            return DiffType.ELEMENT_MISSING_IN_WEB
    
    if "cropped" in raw_cat or "truncat" in raw_cat:
        return DiffType.CROPPED_ELEMENT
    
    return diff_type


def parse_percentage_bbox(
    bbox_data: Dict,
    page_width: int,
    page_height: int,
) -> Optional[BoundingBox]:
    """Parse percentage-based bounding box to absolute pixels.
    
    The LLM outputs bounding boxes as percentages (0.0-1.0) because
    it doesn't know the actual pixel dimensions.
    
    Args:
        bbox_data: Dict with x_pct, y_pct, w_pct, h_pct or x, y, width, height
        page_width: Page width in pixels
        page_height: Page height in pixels
    
    Returns:
        BoundingBox with absolute pixel values
    """
    try:
        # Handle percentage format (from LLM)
        if "x_pct" in bbox_data:
            x = bbox_data["x_pct"] * page_width
            y = bbox_data["y_pct"] * page_height
            width = bbox_data["w_pct"] * page_width
            height = bbox_data["h_pct"] * page_height
        # Handle absolute format
        elif "x" in bbox_data:
            x = bbox_data["x"]
            y = bbox_data["y"]
            width = bbox_data.get("width", bbox_data.get("w", 100))
            height = bbox_data.get("height", bbox_data.get("h", 100))
            
            # Check if values look like percentages (0-1 range)
            if all(0 <= v <= 1 for v in [x, y] if v is not None):
                x = x * page_width
                y = y * page_height
                if width <= 1:
                    width = width * page_width
                if height <= 1:
                    height = height * page_height
        else:
            return None
        
        # Ensure non-negative and within bounds
        x = max(0, min(x, page_width))
        y = max(0, min(y, page_height))
        width = max(10, min(width, page_width - x))
        height = max(10, min(height, page_height - y))
        
        return BoundingBox(x=x, y=y, width=width, height=height)
    
    except (KeyError, TypeError, ValueError) as e:
        logger.debug(f"Failed to parse bbox: {e}")
        return None


def calculate_saliency_from_bbox(
    bbox: BoundingBox,
    page_height: int,
    above_fold_threshold: int = 800,
) -> float:
    """Calculate saliency score based on bounding box position.
    
    Higher scores for:
    - Elements above the fold
    - Larger elements
    - Elements in the center of the viewport
    
    Args:
        bbox: Element bounding box
        page_height: Total page height
        above_fold_threshold: Pixel height considered "above fold"
    
    Returns:
        Saliency score 0.0-1.0
    """
    saliency = 0.5  # Base score
    
    # Above-fold bonus
    if bbox.y < above_fold_threshold:
        saliency += 0.2
    elif bbox.y < above_fold_threshold * 2:
        saliency += 0.1
    
    # Size bonus (larger = more visible)
    area = bbox.width * bbox.height
    if area > 50000:  # Large element
        saliency += 0.15
    elif area > 20000:
        saliency += 0.1
    elif area > 5000:
        saliency += 0.05
    
    return min(1.0, max(0.0, saliency))


async def run_blind_visual_pass_with_retry(
    client: GeminiClient,
    figma_screenshot: Image.Image,
    web_screenshot: Image.Image,
    max_retries: int = 2,
    **kwargs,
) -> List[Finding]:
    """Run Pass A with automatic retry on failure.
    
    If the first attempt returns no findings (unusual), retry with
    a slightly modified prompt to encourage more thorough scanning.
    
    Args:
        client: GeminiClient instance
        figma_screenshot: Figma screenshot
        web_screenshot: Web screenshot
        max_retries: Maximum retry attempts
        **kwargs: Additional arguments for run_blind_visual_pass
    
    Returns:
        List of findings
    """
    for attempt in range(max_retries + 1):
        additional_context = kwargs.get("additional_context", "")
        
        if attempt > 0:
            # Add retry-specific context
            additional_context += (
                "\n\nPREVIOUS SCAN FOUND VERY FEW DIFFERENCES. "
                "Please scan more carefully, especially for:\n"
                "- Subtle shadow differences\n"
                "- Small spacing/padding differences\n"
                "- Font weight or size variations\n"
                "- Color shade differences"
            )
            kwargs["additional_context"] = additional_context
        
        findings = await run_blind_visual_pass(
            client, figma_screenshot, web_screenshot, **kwargs
        )
        
        # If we got findings, return them
        if findings:
            return findings
        
        # Otherwise log and retry
        if attempt < max_retries:
            logger.info(f"Pass A attempt {attempt + 1} found no findings, retrying...")
            await asyncio.sleep(1)
    
    return []
