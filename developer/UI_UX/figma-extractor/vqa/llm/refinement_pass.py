"""
Stage 4: Refinement Pass.

This runs ONLY on findings where confidence < 0.7 from Stage 2.
It uses:
1. Higher-resolution crops (2x zoom on the region)
2. Few-shot examples from the calibration library
3. Explicit chain-of-thought reasoning steps

The goal is to make a definitive determination on uncertain findings,
reducing false positives while not missing real issues.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any
from PIL import Image

from .gemini_client import GeminiClient, GeminiResponse
from .prompts.refinement import (
    build_refinement_prompt,
    build_finding_context,
    build_few_shot_block,
    interpret_refinement_result,
)
from .few_shot.calibration_store import CalibrationStore, get_calibration_store
from ..models.finding import Finding
from ..models.enums import Confidence, Severity
from ..vision.region_cropper import crop_at_zoom_level, CropConfig

logger = logging.getLogger(__name__)


async def run_refinement_pass(
    client: GeminiClient,
    uncertain_findings: List[Finding],
    figma_screenshot: Image.Image,
    web_screenshot: Image.Image,
    calibration_store: Optional[CalibrationStore] = None,
    config: Dict = None,
    zoom_level: float = 2.0,
) -> List[Finding]:
    """Run Stage 4: Selective refinement of uncertain findings.
    
    Re-examines findings with confidence < threshold using:
    - Higher-resolution crops
    - Few-shot calibration examples
    - Chain-of-thought reasoning
    
    Args:
        client: Initialized GeminiClient
        uncertain_findings: Findings needing refinement (confidence < 0.7)
        figma_screenshot: Full Figma screenshot
        web_screenshot: Full web screenshot
        calibration_store: Optional calibration store for few-shot examples
        config: Optional configuration dict
        zoom_level: Zoom factor for higher-res crops (default 2.0)
    
    Returns:
        List of refined Finding objects (updated confidence/severity)
    """
    if not uncertain_findings:
        logger.info("Refinement pass: No uncertain findings to refine")
        return []
    
    config = config or {}
    max_items = config.get("refinement_max_items", 20)
    
    # Limit the number of findings to refine (cost control)
    to_refine = uncertain_findings[:max_items]
    
    logger.info(f"Refinement pass: Refining {len(to_refine)} uncertain findings")
    
    # Get calibration store
    if calibration_store is None:
        calibration_store = get_calibration_store()
    
    # Ensure screenshots are same size
    if web_screenshot.size != figma_screenshot.size:
        web_screenshot = web_screenshot.resize(
            figma_screenshot.size, Image.Resampling.LANCZOS
        )
    
    # Process findings
    refined_findings = []
    
    for finding in to_refine:
        try:
            refined = await refine_single_finding(
                client=client,
                finding=finding,
                figma_screenshot=figma_screenshot,
                web_screenshot=web_screenshot,
                calibration_store=calibration_store,
                zoom_level=zoom_level,
            )
            
            if refined:
                refined_findings.append(refined)
        
        except Exception as e:
            logger.warning(f"Failed to refine finding {finding.id}: {e}")
            # Keep original finding if refinement fails
            refined_findings.append(finding)
    
    logger.info(
        f"Refinement pass complete: {len(refined_findings)} findings "
        f"({sum(1 for f in refined_findings if f.confidence in (Confidence.HIGH, Confidence.MEDIUM))} high/medium confidence)"
    )
    
    return refined_findings


async def refine_single_finding(
    client: GeminiClient,
    finding: Finding,
    figma_screenshot: Image.Image,
    web_screenshot: Image.Image,
    calibration_store: CalibrationStore,
    zoom_level: float = 2.0,
) -> Optional[Finding]:
    """Refine a single uncertain finding.
    
    Args:
        client: GeminiClient instance
        finding: Finding to refine
        figma_screenshot: Full Figma screenshot
        web_screenshot: Full web screenshot
        calibration_store: Calibration store for few-shot examples
        zoom_level: Zoom factor for crops
    
    Returns:
        Refined Finding or None if rejected
    """
    # Get bounding box for cropping
    bbox = finding.get_primary_bbox()
    
    if not bbox:
        logger.debug(f"Finding {finding.id} has no bbox, skipping refinement")
        return finding
    
    # Get category for few-shot examples
    category = finding.category.value if hasattr(finding.category, 'value') else str(finding.category)
    
    # Build few-shot block
    few_shot_block = calibration_store.build_few_shot_block(
        category=category,
        max_examples=4,
    )
    
    # Build system prompt with few-shot examples
    system_prompt = build_refinement_prompt(
        few_shot_block=few_shot_block,
        include_schema=True,
    )
    
    # Create higher-resolution crops
    crop_config = CropConfig(
        padding=20,  # Less padding for focused analysis
        min_size=100,
        max_size=800,  # Allow larger crops for zoom
    )
    
    crop_pair = crop_at_zoom_level(
        figma_img=figma_screenshot,
        web_img=web_screenshot,
        bbox=bbox.to_dict(),
        zoom=zoom_level,
        config=crop_config,
    )
    
    # Build finding context
    finding_dict = finding.to_dict()
    context = build_finding_context(finding_dict, finding_index=0)
    
    # Build prompt
    prompt = f"""
{context}

These are HIGH-RESOLUTION ZOOMED crops (2x magnification) for detailed analysis.
Use the step-by-step methodology to make your determination.
"""
    
    # Call the LLM
    response = await client.analyze_images(
        images=[crop_pair.figma_crop, crop_pair.web_crop],
        prompt=prompt,
        system_prompt=system_prompt,
    )
    
    if not response.success:
        logger.debug(f"Refinement failed for {finding.id}: {response.error}")
        return finding  # Keep original
    
    # Parse and apply refinement result
    return apply_refinement_result(finding, response.data)


def apply_refinement_result(
    finding: Finding,
    result_data: Any,
) -> Optional[Finding]:
    """Apply refinement result to a finding.
    
    Args:
        finding: Original finding
        result_data: LLM refinement response
    
    Returns:
        Updated finding, or None if rejected
    """
    if not result_data or not isinstance(result_data, dict):
        return finding
    
    # Interpret the result
    interpretation = interpret_refinement_result(result_data)
    
    is_confirmed = interpretation.get("is_confirmed", True)
    new_confidence = interpretation.get("confidence", 0.5)
    new_severity = interpretation.get("severity", finding.severity.value)
    reasoning = interpretation.get("reasoning", "")
    
    # If rejected with high confidence, filter out
    if not is_confirmed and new_confidence >= 0.7:
        logger.debug(f"Refinement rejected finding {finding.id}")
        return None
    
    # Update finding
    updated = Finding(
        id=finding.id,
        category=finding.category,
        diff_type=finding.diff_type,
        severity=_map_severity(new_severity),
        confidence=_map_confidence(new_confidence),
        element_name=finding.element_name,
        element_text=finding.element_text,
        element_type=finding.element_type,
        figma_bbox=finding.figma_bbox,
        web_bbox=finding.web_bbox,
        dom_evidence=finding.dom_evidence,
        visual_reasoning=_update_reasoning(finding.visual_reasoning, reasoning),
        source=finding.source,
        pass_name="refinement",
        is_above_fold=finding.is_above_fold,
        is_interactive=finding.is_interactive,
        saliency_score=finding.saliency_score,
        merged_from=finding.merged_from,
    )
    
    return updated


def _map_severity(severity_str: str) -> Severity:
    """Map severity string to enum."""
    severity_map = {
        "critical": Severity.CRITICAL,
        "major": Severity.MAJOR,
        "minor": Severity.MINOR,
        "negligible": Severity.NIT,
        "nit": Severity.NIT,
    }
    return severity_map.get(severity_str.lower(), Severity.MINOR)


def _map_confidence(confidence_val: float) -> Confidence:
    """Map confidence value to enum."""
    if confidence_val >= 0.85:
        return Confidence.HIGH
    elif confidence_val >= 0.6:
        return Confidence.MEDIUM
    elif confidence_val >= 0.4:
        return Confidence.LOW
    else:
        return Confidence.UNCERTAIN


def _update_reasoning(original: str, refinement: str) -> str:
    """Combine original and refinement reasoning."""
    parts = []
    if original:
        parts.append(original)
    if refinement:
        parts.append(f"[Refined] {refinement}")
    return " | ".join(parts) if parts else None


async def batch_refine_findings(
    client: GeminiClient,
    findings: List[Finding],
    figma_screenshot: Image.Image,
    web_screenshot: Image.Image,
    calibration_store: Optional[CalibrationStore] = None,
    batch_size: int = 4,
    config: Dict = None,
) -> List[Finding]:
    """Batch refine multiple findings with concurrency.
    
    More efficient than refining one at a time for large lists.
    
    Args:
        client: GeminiClient instance
        findings: List of findings to refine
        figma_screenshot: Figma screenshot
        web_screenshot: Web screenshot
        calibration_store: Calibration store
        batch_size: Concurrent refinements
        config: Configuration dict
    
    Returns:
        List of refined findings
    """
    if not findings:
        return []
    
    calibration_store = calibration_store or get_calibration_store()
    
    # Ensure screenshots are same size
    if web_screenshot.size != figma_screenshot.size:
        web_screenshot = web_screenshot.resize(
            figma_screenshot.size, Image.Resampling.LANCZOS
        )
    
    # Create batches
    batches = [
        findings[i:i + batch_size]
        for i in range(0, len(findings), batch_size)
    ]
    
    all_refined = []
    
    for batch in batches:
        # Process batch concurrently
        tasks = [
            refine_single_finding(
                client=client,
                finding=f,
                figma_screenshot=figma_screenshot,
                web_screenshot=web_screenshot,
                calibration_store=calibration_store,
                zoom_level=config.get("zoom_level", 2.0) if config else 2.0,
            )
            for f in batch
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Refinement failed: {result}")
                all_refined.append(batch[i])  # Keep original
            elif result is not None:
                all_refined.append(result)
    
    return all_refined


def filter_for_refinement(
    findings: List[Finding],
    confidence_threshold: float = 0.7,
) -> tuple:
    """Split findings into certain and uncertain groups.
    
    Args:
        findings: List of all findings
        confidence_threshold: Below this confidence needs refinement
    
    Returns:
        Tuple of (certain_findings, uncertain_findings)
    """
    certain = []
    uncertain = []
    
    confidence_to_value = {
        Confidence.HIGH: 0.9,
        Confidence.MEDIUM: 0.7,
        Confidence.LOW: 0.5,
        Confidence.UNCERTAIN: 0.3,
    }
    
    for finding in findings:
        conf_value = confidence_to_value.get(finding.confidence, 0.5)
        
        if conf_value >= confidence_threshold:
            certain.append(finding)
        else:
            uncertain.append(finding)
    
    return certain, uncertain
