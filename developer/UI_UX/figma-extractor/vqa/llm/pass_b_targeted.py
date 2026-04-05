"""
Pass B: Targeted Validation.

For each DOM-detected difference, this pass sends the model a CROPPED PAIR
showing just that element from both screenshots, plus the specific difference data.

The model's job is to CONFIRM or REJECT the DOM finding, and assess severity.
This eliminates false positives from:
- CSS transforms that make visually identical results
- Visual overrides that hide DOM differences
- Differences that exist in the DOM but aren't visually perceptible

Key principles:
1. Batch crop pairs to minimize API calls (5-8 pairs per call)
2. Include DOM diff context so the model knows what to look for
3. Capture additional findings the model notices in the crops
"""

import asyncio
import uuid
import logging
from typing import List, Dict, Optional, Any, Callable
from PIL import Image

from .gemini_client import GeminiClient, GeminiResponse
from .prompts.targeted_validation import (
    build_targeted_validation_prompt,
    build_pair_context,
    build_batch_prompt,
    interpret_verdict,
    normalize_property_type,
)
from ..models.finding import Finding, DOMEvidence
from ..models.region import BoundingBox
from ..models.enums import Category, Severity, Confidence, DiffType
from ..vision.region_cropper import CropPair

logger = logging.getLogger(__name__)


async def run_targeted_validation_pass(
    client: GeminiClient,
    crop_pairs: List[CropPair],
    dom_diffs: List[Dict],
    calibration_store: Any = None,
    batch_size: int = 6,
) -> List[Finding]:
    """Run Pass B: Targeted validation of DOM-detected differences.
    
    For each DOM diff with a crop pair, send to LLM for visual validation.
    The model confirms or rejects each difference based on visual analysis.
    
    Args:
        client: Initialized GeminiClient
        crop_pairs: List of CropPair objects with figma/web crops
        dom_diffs: List of DOM diff dicts (same order as crop_pairs)
        calibration_store: Optional calibration store for few-shot examples
        batch_size: Number of pairs per API call
    
    Returns:
        List of Finding objects (confirmed differences only)
    """
    if not crop_pairs:
        logger.info("Pass B: No crop pairs to validate")
        return []
    
    logger.info(f"Pass B: Validating {len(crop_pairs)} DOM differences")
    
    # Match crop pairs with their DOM diffs
    paired_data = match_crops_with_diffs(crop_pairs, dom_diffs)
    
    # Build the system prompt
    system_prompt = build_targeted_validation_prompt(include_schema=True)
    
    # Process in batches
    batches = create_batches(paired_data, batch_size)
    
    all_findings = []
    
    for batch_idx, batch in enumerate(batches):
        logger.debug(f"Processing batch {batch_idx + 1}/{len(batches)}")
        
        batch_findings = await process_batch(
            client, batch, system_prompt, batch_idx
        )
        all_findings.extend(batch_findings)
    
    logger.info(f"Pass B validated {len(all_findings)} findings")
    
    return all_findings


def match_crops_with_diffs(
    crop_pairs: List[CropPair],
    dom_diffs: List[Dict],
) -> List[Dict]:
    """Match crop pairs with their corresponding DOM diffs.
    
    Tries to match by:
    1. Region ID
    2. Element name
    3. Bounding box proximity
    4. Index order
    
    Args:
        crop_pairs: List of CropPair objects
        dom_diffs: List of DOM diff dicts
    
    Returns:
        List of dicts with 'crop_pair' and 'diff' keys
    """
    paired = []
    used_diff_indices = set()
    
    for crop in crop_pairs:
        matched_diff = None
        
        # Try to match by region_id
        if crop.region_id:
            for i, diff in enumerate(dom_diffs):
                if i in used_diff_indices:
                    continue
                if diff.get("region_id") == crop.region_id or diff.get("id") == crop.region_id:
                    matched_diff = diff
                    used_diff_indices.add(i)
                    break
        
        # Try to match by element_name
        if not matched_diff and crop.element_name:
            for i, diff in enumerate(dom_diffs):
                if i in used_diff_indices:
                    continue
                if diff.get("element_name") == crop.element_name or diff.get("element") == crop.element_name:
                    matched_diff = diff
                    used_diff_indices.add(i)
                    break
        
        # Try to match by bbox proximity
        if not matched_diff and crop.bbox:
            for i, diff in enumerate(dom_diffs):
                if i in used_diff_indices:
                    continue
                diff_bbox = diff.get("web_position") or diff.get("bbox")
                if diff_bbox and is_bbox_close(crop.bbox, diff_bbox):
                    matched_diff = diff
                    used_diff_indices.add(i)
                    break
        
        # Fall back to index-based matching
        if not matched_diff:
            for i, diff in enumerate(dom_diffs):
                if i not in used_diff_indices:
                    matched_diff = diff
                    used_diff_indices.add(i)
                    break
        
        # Create empty diff if no match found
        if not matched_diff:
            matched_diff = {
                "diff_type": "unknown",
                "element_name": crop.element_name or "Unknown",
            }
        
        paired.append({
            "crop_pair": crop,
            "diff": matched_diff,
        })
    
    return paired


def is_bbox_close(bbox1: BoundingBox, bbox2: Dict, threshold: float = 50) -> bool:
    """Check if two bounding boxes are close to each other.
    
    Args:
        bbox1: BoundingBox object
        bbox2: Dict with x, y, width, height
        threshold: Maximum distance (pixels) to consider "close"
    
    Returns:
        True if centers are within threshold distance
    """
    try:
        c1_x = bbox1.center_x
        c1_y = bbox1.center_y
        
        x2 = bbox2.get("x", 0)
        y2 = bbox2.get("y", 0)
        w2 = bbox2.get("width", 0)
        h2 = bbox2.get("height", 0)
        c2_x = x2 + w2 / 2
        c2_y = y2 + h2 / 2
        
        distance = ((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2) ** 0.5
        return distance < threshold
    except (AttributeError, TypeError):
        return False


def create_batches(items: List, batch_size: int) -> List[List]:
    """Split items into batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


async def process_batch(
    client: GeminiClient,
    batch: List[Dict],
    system_prompt: str,
    batch_idx: int,
) -> List[Finding]:
    """Process a batch of crop pairs through the LLM.
    
    Args:
        client: GeminiClient instance
        batch: List of paired dicts with 'crop_pair' and 'diff'
        system_prompt: System prompt for validation
        batch_idx: Batch index for logging
    
    Returns:
        List of confirmed Finding objects
    """
    # Prepare crop pairs and contexts for the API call
    api_pairs = []
    contexts = []
    
    for idx, item in enumerate(batch):
        crop_pair = item["crop_pair"]
        diff = item["diff"]
        
        api_pairs.append({
            "figma_crop": crop_pair.figma_crop,
            "web_crop": crop_pair.web_crop,
            "element_name": crop_pair.element_name or diff.get("element_name", "Unknown"),
            "diff": diff,
            "original_index": idx,
        })
        
        context = build_pair_context(
            diff_data=diff,
            pair_index=idx,
            element_name=crop_pair.element_name or diff.get("element_name"),
        )
        contexts.append(context)
    
    # Build the combined prompt
    batch_prompt = build_batch_prompt(contexts)
    
    # Define the per-pair prompt builder
    def pair_prompt_builder(pair_data: Dict, idx: int) -> str:
        return contexts[idx] if idx < len(contexts) else ""
    
    # Call the LLM with batched pairs
    responses = await client.analyze_crop_pairs(
        crop_pairs=api_pairs,
        system_prompt=system_prompt,
        per_pair_prompt_builder=pair_prompt_builder,
    )
    
    # Process responses into findings
    findings = []
    
    for idx, response in enumerate(responses):
        if idx >= len(batch):
            break
        
        item = batch[idx]
        crop_pair = item["crop_pair"]
        diff = item["diff"]
        
        if not response.success:
            logger.debug(f"Validation failed for item {idx}: {response.error}")
            continue
        
        # Parse the validation result
        finding = parse_validation_response(
            response.data,
            crop_pair,
            diff,
        )
        
        if finding:
            findings.append(finding)
    
    return findings


def parse_validation_response(
    response_data: Any,
    crop_pair: CropPair,
    original_diff: Dict,
) -> Optional[Finding]:
    """Parse validation response and create Finding if confirmed.
    
    Args:
        response_data: LLM response data
        crop_pair: Original CropPair
        original_diff: Original DOM diff data
    
    Returns:
        Finding if confirmed, None if rejected
    """
    if not response_data:
        return None
    
    # Handle dict response
    if isinstance(response_data, dict):
        verdict = response_data.get("verdict", "uncertain")
        confidence = response_data.get("confidence", 0.7)
        severity = response_data.get("severity", "minor")
        reasoning = response_data.get("reasoning", "")
        additional = response_data.get("additional_findings", [])
    else:
        # Unexpected format
        return None
    
    # Interpret the verdict
    interpretation = interpret_verdict(verdict, confidence)
    
    # Skip rejected findings
    if interpretation["is_valid"] is False:
        logger.debug(f"Rejected: {original_diff.get('element_name', 'unknown')}")
        return None
    
    # Skip uncertain findings with low confidence
    if interpretation["is_valid"] is None and interpretation["confidence"] < 0.5:
        return None
    
    # Map to category
    diff_type_str = original_diff.get("diff_type", original_diff.get("sub_type", "other"))
    category = diff_type_to_category(diff_type_str)
    
    # Map to DiffType enum
    try:
        diff_type = DiffType(diff_type_str)
    except ValueError:
        diff_type = DiffType.VISUAL_HIERARCHY
    
    # Map severity
    severity_map = {
        "critical": Severity.CRITICAL,
        "major": Severity.MAJOR,
        "minor": Severity.MINOR,
        "negligible": Severity.NIT,
        "nit": Severity.NIT,
    }
    severity_enum = severity_map.get(severity.lower(), Severity.MINOR)
    
    # Map confidence
    conf_val = interpretation["confidence"]
    if conf_val >= 0.85:
        confidence_enum = Confidence.HIGH
    elif conf_val >= 0.6:
        confidence_enum = Confidence.MEDIUM
    elif conf_val >= 0.4:
        confidence_enum = Confidence.LOW
    else:
        confidence_enum = Confidence.UNCERTAIN
    
    # Build DOM evidence
    dom_evidence = DOMEvidence(
        figma_value=original_diff.get("figma_value"),
        web_value=original_diff.get("web_value"),
        delta=original_diff.get("delta", ""),
        figma_node_id=original_diff.get("figma_node_id"),
        web_node_id=original_diff.get("web_node_id"),
        web_locator=original_diff.get("web_locator"),
    )
    
    # Build visual reasoning
    visual_reasoning = reasoning
    if additional:
        visual_reasoning += f" Additional: {'; '.join(additional)}"
    
    # Create the finding
    finding = Finding(
        id=f"pass_b_{uuid.uuid4().hex[:8]}",
        category=category,
        diff_type=diff_type,
        severity=severity_enum,
        confidence=confidence_enum,
        element_name=crop_pair.element_name or original_diff.get("element_name", "Unknown"),
        element_text=original_diff.get("element_text"),
        element_type=original_diff.get("element_type"),
        web_bbox=crop_pair.bbox if crop_pair.bbox else None,
        dom_evidence=dom_evidence,
        visual_reasoning=visual_reasoning.strip() if visual_reasoning else None,
        source="gemini_visual",
        pass_name="pass_b",
        saliency_score=crop_pair.metadata.get("saliency_score", 0.5) if crop_pair.metadata else 0.5,
    )
    
    return finding


def diff_type_to_category(diff_type: str) -> Category:
    """Map diff type string to Category enum.
    
    Args:
        diff_type: Diff type string
    
    Returns:
        Category enum value
    """
    diff_type_lower = diff_type.lower()
    
    # Text properties
    if any(x in diff_type_lower for x in ["text", "font", "typography"]):
        return Category.TEXT
    
    # Spacing/padding
    if "padding" in diff_type_lower:
        return Category.PADDING
    if any(x in diff_type_lower for x in ["spacing", "gap", "margin"]):
        return Category.SPACING
    
    # Color
    if "color" in diff_type_lower:
        return Category.COLOR
    
    # Size
    if any(x in diff_type_lower for x in ["width", "height", "size"]):
        return Category.SIZE
    
    # Shadow
    if "shadow" in diff_type_lower:
        return Category.SHADOW
    
    # Border
    if any(x in diff_type_lower for x in ["border", "stroke", "divider"]):
        return Category.BORDER
    
    # Overflow
    if any(x in diff_type_lower for x in ["overflow", "truncat", "clip"]):
        return Category.OVERFLOW
    
    # Layout
    if any(x in diff_type_lower for x in ["layout", "flex", "align", "justify"]):
        return Category.LAYOUT
    
    # Position
    if any(x in diff_type_lower for x in ["position", "z-index", "zindex"]):
        return Category.POSITION
    
    # Opacity
    if "opacity" in diff_type_lower:
        return Category.OPACITY
    
    # Visibility
    if any(x in diff_type_lower for x in ["visibility", "display"]):
        return Category.VISIBILITY
    
    # Aspect ratio
    if "aspect" in diff_type_lower:
        return Category.ASPECT_RATIO
    
    # Icons
    if "icon" in diff_type_lower:
        return Category.ICONS
    
    # Images
    if "image" in diff_type_lower:
        return Category.IMAGES
    
    # Components
    if "component" in diff_type_lower:
        return Category.COMPONENTS
    
    # Missing elements
    if "missing" in diff_type_lower or "element" in diff_type_lower:
        return Category.MISSING_ELEMENTS
    
    return Category.OTHER


async def run_targeted_validation_single(
    client: GeminiClient,
    figma_crop: Image.Image,
    web_crop: Image.Image,
    diff_data: Dict,
) -> Optional[Finding]:
    """Run targeted validation on a single crop pair.
    
    Useful for testing or when only one difference needs validation.
    
    Args:
        client: GeminiClient instance
        figma_crop: Figma crop image
        web_crop: Web crop image
        diff_data: DOM diff data
    
    Returns:
        Finding if confirmed, None otherwise
    """
    system_prompt = build_targeted_validation_prompt(include_schema=True)
    
    response = await client.targeted_analysis(
        figma_crop=figma_crop,
        web_crop=web_crop,
        reported_diff=diff_data,
        system_prompt=system_prompt,
    )
    
    if not response.success:
        return None
    
    # Create a minimal CropPair for parsing
    crop_pair = CropPair(
        figma_crop=figma_crop,
        web_crop=web_crop,
        bbox=BoundingBox.from_dict(diff_data.get("web_position", {"x": 0, "y": 0, "width": 100, "height": 100})),
        padded_bbox=BoundingBox(x=0, y=0, width=100, height=100),
        element_name=diff_data.get("element_name", "Unknown"),
    )
    
    return parse_validation_response(response.data, crop_pair, diff_data)
