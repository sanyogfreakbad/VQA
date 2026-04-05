"""
Merge + Deduplicate - Stage 3 of the VQA pipeline.

Merges findings from all three passes and deduplicates by bounding box overlap.

Logic:
1. Collect findings from Pass A (blind), Pass B (targeted), and Pass C (behavioral)
2. Normalize all findings to the same Finding model
3. Deduplicate by bounding box overlap (IoU > 0.5 = same finding)
4. When duplicates found, keep the version with highest confidence
5. Merge evidence from both (DOM data from Pass B + visual description from Pass A)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass

from ..models.finding import Finding, DOMEvidence
from ..models.region import BoundingBox
from ..models.enums import Category, Severity, Confidence, DiffType

logger = logging.getLogger(__name__)


@dataclass
class MergeResult:
    """Result of merge and deduplication."""
    merged_findings: List[Finding]
    total_input: int
    duplicates_removed: int
    pass_a_count: int
    pass_b_count: int
    pass_c_count: int
    dom_only_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "merged_count": len(self.merged_findings),
            "total_input": self.total_input,
            "duplicates_removed": self.duplicates_removed,
            "by_source": {
                "pass_a_blind": self.pass_a_count,
                "pass_b_targeted": self.pass_b_count,
                "pass_c_behavioral": self.pass_c_count,
                "dom_comparator": self.dom_only_count,
            }
        }


def merge_findings(
    pass_a_findings: List[Finding],
    pass_b_findings: List[Finding],
    pass_c_findings: List[Finding],
    dom_diffs: Optional[List[Dict[str, Any]]] = None,
    iou_threshold: float = 0.5,
) -> MergeResult:
    """Merge and deduplicate findings from all passes.
    
    Priority when merging duplicates:
    1. Keep the finding with higher confidence
    2. Attach DOM evidence from pass_b if available
    3. Attach visual description from pass_a if available
    4. Behavioral findings (pass_c) are rarely duplicates — usually unique
    
    Args:
        pass_a_findings: Findings from blind visual analysis
        pass_b_findings: Findings from targeted validation
        pass_c_findings: Findings from behavioral testing
        dom_diffs: Optional validated DOM diffs to include
        iou_threshold: IoU threshold for considering findings as duplicates
    
    Returns:
        MergeResult with deduplicated, merged list of findings
    """
    all_findings: List[Finding] = []
    
    for f in pass_a_findings:
        f.source = "gemini_visual"
        f.pass_name = "pass_a"
        all_findings.append(f)
    
    for f in pass_b_findings:
        f.source = "gemini_visual" if f.source != "dom" else "dom"
        f.pass_name = "pass_b"
        all_findings.append(f)
    
    for f in pass_c_findings:
        f.source = "behavioral"
        f.pass_name = "pass_c"
        all_findings.append(f)
    
    if dom_diffs:
        for diff in dom_diffs:
            if _is_validated_dom_diff(diff):
                dom_finding = _convert_dom_diff_to_finding(diff)
                if dom_finding:
                    all_findings.append(dom_finding)
    
    total_input = len(all_findings)
    
    merged = _deduplicate_findings(all_findings, iou_threshold)
    
    duplicates_removed = total_input - len(merged)
    
    pass_a_in_merged = sum(1 for f in merged if f.pass_name == "pass_a")
    pass_b_in_merged = sum(1 for f in merged if f.pass_name == "pass_b")
    pass_c_in_merged = sum(1 for f in merged if f.pass_name == "pass_c")
    dom_in_merged = sum(1 for f in merged if f.source == "dom")
    
    logger.info(
        f"Merged {total_input} findings into {len(merged)} "
        f"(removed {duplicates_removed} duplicates)"
    )
    
    return MergeResult(
        merged_findings=merged,
        total_input=total_input,
        duplicates_removed=duplicates_removed,
        pass_a_count=pass_a_in_merged,
        pass_b_count=pass_b_in_merged,
        pass_c_count=pass_c_in_merged,
        dom_only_count=dom_in_merged,
    )


def _deduplicate_findings(
    findings: List[Finding],
    iou_threshold: float,
) -> List[Finding]:
    """Deduplicate findings by bounding box overlap.
    
    Args:
        findings: All findings to deduplicate
        iou_threshold: IoU above this = same finding
    
    Returns:
        Deduplicated list of findings
    """
    if not findings:
        return []
    
    findings = sorted(findings, key=lambda f: _confidence_score(f), reverse=True)
    
    merged: List[Finding] = []
    used: Set[int] = set()
    
    for i, finding in enumerate(findings):
        if i in used:
            continue
        
        overlapping: List[int] = []
        for j, other in enumerate(findings):
            if j <= i or j in used:
                continue
            
            iou = _compute_finding_iou(finding, other)
            if iou > iou_threshold:
                overlapping.append(j)
                used.add(j)
        
        if overlapping:
            for j in overlapping:
                finding = _merge_finding_evidence(finding, findings[j])
        
        merged.append(finding)
        used.add(i)
    
    return merged


def _compute_finding_iou(f1: Finding, f2: Finding) -> float:
    """Compute IoU between two findings based on their bounding boxes.
    
    Uses web_bbox preferentially, falls back to figma_bbox.
    """
    bbox1 = f1.web_bbox or f1.figma_bbox
    bbox2 = f2.web_bbox or f2.figma_bbox
    
    if not bbox1 or not bbox2:
        return 0.0
    
    return bbox1.iou(bbox2)


def _confidence_score(finding: Finding) -> float:
    """Convert confidence enum to numeric score for sorting."""
    confidence_values = {
        Confidence.HIGH: 0.9,
        Confidence.MEDIUM: 0.7,
        Confidence.LOW: 0.4,
        Confidence.UNCERTAIN: 0.2,
    }
    return confidence_values.get(finding.confidence, 0.5)


def _merge_finding_evidence(
    primary: Finding,
    secondary: Finding,
) -> Finding:
    """Merge evidence from secondary finding into primary.
    
    Primary keeps its verdict and confidence. Secondary adds:
    - DOM evidence if primary lacks it
    - Visual description if primary lacks it
    - Additional sources for traceability
    
    Args:
        primary: Primary finding (higher confidence)
        secondary: Secondary finding to merge from
    
    Returns:
        Primary finding with merged evidence
    """
    if not primary.dom_evidence and secondary.dom_evidence:
        primary.dom_evidence = secondary.dom_evidence
    
    if not primary.visual_reasoning and secondary.visual_reasoning:
        primary.visual_reasoning = secondary.visual_reasoning
    elif primary.visual_reasoning and secondary.visual_reasoning:
        if secondary.visual_reasoning not in primary.visual_reasoning:
            primary.visual_reasoning = f"{primary.visual_reasoning} | {secondary.visual_reasoning}"
    
    if not primary.web_bbox and secondary.web_bbox:
        primary.web_bbox = secondary.web_bbox
    if not primary.figma_bbox and secondary.figma_bbox:
        primary.figma_bbox = secondary.figma_bbox
    
    if secondary.id not in primary.merged_from:
        primary.merged_from.append(secondary.id)
    
    if secondary.pass_name and secondary.pass_name not in str(primary.pass_name):
        pass_names = primary.pass_name or ""
        if pass_names:
            primary.pass_name = f"{pass_names}+{secondary.pass_name}"
        else:
            primary.pass_name = secondary.pass_name
    
    return primary


def _is_validated_dom_diff(diff: Dict[str, Any]) -> bool:
    """Check if a DOM diff should be included as a finding.
    
    Only include diffs that:
    - Have severity of 'warning' or higher
    - Haven't been explicitly rejected by LLM validation
    - Have position information
    """
    if diff.get("validated") is False:
        return False
    if diff.get("rejected"):
        return False
    
    severity = diff.get("severity", "info")
    if severity in ("info", "nit"):
        return False
    
    if not _has_position_info(diff):
        return False
    
    return True


def _has_position_info(diff: Dict[str, Any]) -> bool:
    """Check if a diff has usable position information."""
    return bool(
        diff.get("web_position") or 
        diff.get("figma_position") or 
        diff.get("position") or
        diff.get("bbox")
    )


def _convert_dom_diff_to_finding(diff: Dict[str, Any]) -> Optional[Finding]:
    """Convert a DOM diff dictionary to a Finding object.
    
    Args:
        diff: DOM diff dictionary from comparator
    
    Returns:
        Finding object or None if conversion fails
    """
    try:
        category = _map_diff_type_to_category(diff.get("sub_type", "other"))
        diff_type = _map_string_to_diff_type(diff.get("sub_type", "other"))
        severity = _map_string_to_severity(diff.get("severity", "minor"))
        
        web_bbox = None
        if diff.get("web_position"):
            web_bbox = BoundingBox.from_dict(diff["web_position"])
        
        figma_bbox = None
        if diff.get("figma_position"):
            figma_bbox = BoundingBox.from_dict(diff["figma_position"])
        
        dom_evidence = DOMEvidence(
            figma_value=diff.get("figma_value"),
            web_value=diff.get("web_value"),
            delta=diff.get("delta"),
            figma_node_id=diff.get("figma_node_id"),
            web_node_id=diff.get("web_node_id"),
            web_locator=diff.get("web_locator"),
        )
        
        import uuid
        finding = Finding(
            id=f"dom_{uuid.uuid4().hex[:8]}",
            category=category,
            diff_type=diff_type,
            severity=severity,
            confidence=Confidence.HIGH,
            element_name=diff.get("element", "Unknown element"),
            element_text=diff.get("element_text"),
            element_type=diff.get("element_type"),
            web_bbox=web_bbox,
            figma_bbox=figma_bbox,
            dom_evidence=dom_evidence,
            source="dom",
            pass_name="dom_comparator",
        )
        
        return finding
        
    except Exception as e:
        logger.warning(f"Failed to convert DOM diff to finding: {e}")
        return None


def _map_diff_type_to_category(diff_type: str) -> Category:
    """Map diff type string to Category enum."""
    type_to_category = {
        "text_": Category.TEXT,
        "spacing_": Category.SPACING,
        "padding": Category.PADDING,
        "color_": Category.COLOR,
        "size_": Category.SIZE,
        "shadow": Category.SHADOW,
        "border": Category.BORDER,
        "overflow": Category.OVERFLOW,
        "layout": Category.LAYOUT,
        "position": Category.POSITION,
        "opacity": Category.OPACITY,
        "visibility": Category.VISIBILITY,
        "aspect": Category.ASPECT_RATIO,
        "component": Category.COMPONENTS,
        "icon": Category.ICONS,
        "image": Category.IMAGES,
        "element_missing": Category.MISSING_ELEMENTS,
        "gap": Category.SPACING,
    }
    
    diff_lower = diff_type.lower()
    for prefix, category in type_to_category.items():
        if diff_lower.startswith(prefix) or prefix in diff_lower:
            return category
    
    return Category.OTHER


def _map_string_to_diff_type(type_str: str) -> DiffType:
    """Map string to DiffType enum."""
    try:
        return DiffType(type_str)
    except ValueError:
        type_lower = type_str.lower()
        
        if "font_size" in type_lower:
            return DiffType.TEXT_FONT_SIZE
        if "font_weight" in type_lower:
            return DiffType.TEXT_FONT_WEIGHT
        if "color" in type_lower and "text" in type_lower:
            return DiffType.TEXT_COLOR
        if "background" in type_lower:
            return DiffType.COLOR_BACKGROUND
        if "padding" in type_lower:
            return DiffType.SPACING_PADDING_TOP
        if "gap" in type_lower or "spacing" in type_lower:
            return DiffType.SPACING_ITEM_GAP
        if "shadow" in type_lower:
            return DiffType.SHADOW_BOX
        if "border" in type_lower:
            return DiffType.BORDER_WEIGHT
        if "width" in type_lower:
            return DiffType.SIZE_WIDTH
        if "height" in type_lower:
            return DiffType.SIZE_HEIGHT
        
        return DiffType.VISUAL_HIERARCHY


def _map_string_to_severity(sev_str: str) -> Severity:
    """Map string to Severity enum."""
    mapping = {
        "critical": Severity.CRITICAL,
        "major": Severity.MAJOR,
        "minor": Severity.MINOR,
        "warning": Severity.MAJOR,
        "error": Severity.CRITICAL,
        "nit": Severity.NIT,
        "info": Severity.INFO,
    }
    return mapping.get(sev_str.lower(), Severity.MINOR)


def filter_by_confidence(
    findings: List[Finding],
    min_confidence: Confidence = Confidence.LOW,
) -> List[Finding]:
    """Filter findings by minimum confidence level.
    
    Args:
        findings: List of findings
        min_confidence: Minimum confidence to include
    
    Returns:
        Filtered list of findings
    """
    confidence_rank = {
        Confidence.UNCERTAIN: 0,
        Confidence.LOW: 1,
        Confidence.MEDIUM: 2,
        Confidence.HIGH: 3,
    }
    
    min_rank = confidence_rank.get(min_confidence, 1)
    
    return [
        f for f in findings
        if confidence_rank.get(f.confidence, 0) >= min_rank
    ]


def get_findings_for_refinement(
    findings: List[Finding],
    confidence_threshold: Confidence = Confidence.MEDIUM,
    max_refinement_count: int = 20,
) -> Tuple[List[Finding], List[Finding]]:
    """Split findings into those needing refinement and those that don't.
    
    Args:
        findings: All merged findings
        confidence_threshold: Below this confidence = needs refinement
        max_refinement_count: Maximum findings to refine (cost control)
    
    Returns:
        Tuple of (needs_refinement, already_confident)
    """
    confidence_rank = {
        Confidence.UNCERTAIN: 0,
        Confidence.LOW: 1,
        Confidence.MEDIUM: 2,
        Confidence.HIGH: 3,
    }
    
    threshold_rank = confidence_rank.get(confidence_threshold, 2)
    
    needs_refinement = []
    already_confident = []
    
    for f in findings:
        if confidence_rank.get(f.confidence, 0) < threshold_rank:
            needs_refinement.append(f)
        else:
            already_confident.append(f)
    
    needs_refinement.sort(key=lambda f: f.saliency_score, reverse=True)
    needs_refinement = needs_refinement[:max_refinement_count]
    
    return needs_refinement, already_confident
