"""
Prioritizer - Stage 5 of the VQA pipeline.

Final prioritization: rank findings by importance.

Factors:
1. Severity (from LLM judgment): critical > major > minor > negligible
2. Confidence (from LLM + algorithmic): higher = more certain
3. Saliency (from region segmenter): above-fold, interactive, large elements
4. Category (from issue type): functional > visual > cosmetic

Output is a sorted list ready for the report.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..models.finding import Finding
from ..models.enums import Category, Severity, Confidence
from ..config.thresholds import Thresholds, THRESHOLDS

logger = logging.getLogger(__name__)


SEVERITY_WEIGHTS = {
    Severity.CRITICAL: 4.0,
    Severity.MAJOR: 3.0,
    Severity.MINOR: 2.0,
    Severity.NIT: 1.0,
    Severity.INFO: 0.5,
}

CONFIDENCE_WEIGHTS = {
    Confidence.HIGH: 1.0,
    Confidence.MEDIUM: 0.75,
    Confidence.LOW: 0.5,
    Confidence.UNCERTAIN: 0.25,
}

CATEGORY_WEIGHTS = {
    Category.MISSING_ELEMENTS: 1.4,
    Category.BEHAVIORAL: 1.3,
    Category.ICONS: 1.2,
    Category.BUTTONS_CTA: 1.2,
    Category.COMPONENTS: 1.15,
    Category.SHADOW: 1.1,
    Category.OVERFLOW: 1.1,
    Category.LAYOUT: 1.05,
    Category.IMAGES: 1.05,
    Category.POSITION: 1.0,
    Category.TEXT: 1.0,
    Category.BORDER: 1.0,
    Category.SIZE: 0.95,
    Category.SPACING: 0.9,
    Category.PADDING: 0.9,
    Category.COLOR: 0.9,
    Category.OPACITY: 0.85,
    Category.VISIBILITY: 0.85,
    Category.ASPECT_RATIO: 0.8,
    Category.OTHER: 0.7,
}


@dataclass
class PrioritizationResult:
    """Result of prioritization stage."""
    prioritized_findings: List[Finding]
    by_severity: Dict[str, List[Finding]]
    by_category: Dict[str, List[Finding]]
    total_findings: int
    filtered_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_findings": self.total_findings,
            "filtered_count": self.filtered_count,
            "by_severity": {k: len(v) for k, v in self.by_severity.items()},
            "by_category": {k: len(v) for k, v in self.by_category.items()},
        }


def prioritize_findings(
    findings: List[Finding],
    min_confidence: Confidence = Confidence.LOW,
    max_findings: int = 100,
    thresholds: Optional[Thresholds] = None,
) -> PrioritizationResult:
    """Sort and filter findings by importance.
    
    1. Remove findings below confidence threshold
    2. Compute composite priority score
    3. Sort by priority descending
    4. Group by category and severity
    
    Args:
        findings: List of merged findings
        min_confidence: Minimum confidence to include
        max_findings: Maximum findings to return
        thresholds: Optional custom thresholds
    
    Returns:
        PrioritizationResult with sorted and grouped findings
    """
    thresholds = thresholds or THRESHOLDS
    
    filtered = _filter_by_confidence(findings, min_confidence)
    filtered_count = len(findings) - len(filtered)
    
    for finding in filtered:
        finding.saliency_score = _compute_priority_score(finding, thresholds)
    
    filtered.sort(key=lambda f: f.saliency_score, reverse=True)
    
    filtered = filtered[:max_findings]
    
    for i, finding in enumerate(filtered, 1):
        finding.serial_number = i
    
    by_severity = _group_by_severity(filtered)
    by_category = _group_by_category(filtered)
    
    logger.info(
        f"Prioritized {len(filtered)} findings "
        f"(filtered {filtered_count} low-confidence)"
    )
    
    return PrioritizationResult(
        prioritized_findings=filtered,
        by_severity=by_severity,
        by_category=by_category,
        total_findings=len(filtered),
        filtered_count=filtered_count,
    )


def _filter_by_confidence(
    findings: List[Finding],
    min_confidence: Confidence,
) -> List[Finding]:
    """Filter findings by minimum confidence level."""
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


def _compute_priority_score(
    finding: Finding,
    thresholds: Thresholds,
) -> float:
    """Compute composite priority score for a finding.
    
    Score formula:
    - 40% from severity
    - 30% from confidence  
    - 20% from saliency (position/visibility)
    - 10% from category importance
    
    Args:
        finding: Finding to score
        thresholds: Comparison thresholds
    
    Returns:
        Priority score (0.0 to ~1.5)
    """
    severity_score = SEVERITY_WEIGHTS.get(finding.severity, 2.0) / 4.0
    
    confidence_score = CONFIDENCE_WEIGHTS.get(finding.confidence, 0.5)
    
    saliency_score = _compute_saliency_score(finding, thresholds)
    
    category_multiplier = CATEGORY_WEIGHTS.get(finding.category, 1.0)
    
    priority = (
        severity_score * 0.4 +
        confidence_score * 0.3 +
        saliency_score * 0.2 +
        (category_multiplier - 0.7) * 0.33
    )
    
    return priority


def _compute_saliency_score(
    finding: Finding,
    thresholds: Thresholds,
) -> float:
    """Compute saliency score based on position and interactivity.
    
    Factors:
    - Above-fold position: +0.2
    - Interactive element: +0.15
    - Large element: +0.1
    """
    score = 0.5
    
    if finding.is_above_fold:
        score += 0.2
    
    if finding.is_interactive:
        score += 0.15 * thresholds.interactive_weight
    
    bbox = finding.web_bbox or finding.figma_bbox
    if bbox:
        area = bbox.area
        if area > thresholds.large_element_threshold:
            score += 0.15
        elif area > thresholds.large_element_threshold / 2:
            score += 0.1
        elif area > thresholds.large_element_threshold / 5:
            score += 0.05
    
    return min(score, 1.0)


def _group_by_severity(findings: List[Finding]) -> Dict[str, List[Finding]]:
    """Group findings by severity level."""
    groups: Dict[str, List[Finding]] = {
        "critical": [],
        "major": [],
        "minor": [],
        "nit": [],
        "info": [],
    }
    
    for finding in findings:
        sev_key = finding.severity.value
        if sev_key in groups:
            groups[sev_key].append(finding)
        else:
            groups["minor"].append(finding)
    
    return {k: v for k, v in groups.items() if v}


def _group_by_category(findings: List[Finding]) -> Dict[str, List[Finding]]:
    """Group findings by category."""
    groups: Dict[str, List[Finding]] = {}
    
    for finding in findings:
        cat_key = finding.category.value
        if cat_key not in groups:
            groups[cat_key] = []
        groups[cat_key].append(finding)
    
    return groups


def get_summary_stats(
    findings: List[Finding],
) -> Dict[str, Any]:
    """Generate summary statistics for a list of findings.
    
    Args:
        findings: List of findings
    
    Returns:
        Dict with summary statistics
    """
    if not findings:
        return {
            "total": 0,
            "by_severity": {},
            "by_category": {},
            "by_source": {},
            "above_fold_count": 0,
            "interactive_count": 0,
        }
    
    severity_counts = {}
    for f in findings:
        sev = f.severity.value
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    category_counts = {}
    for f in findings:
        cat = f.category.value
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    source_counts = {}
    for f in findings:
        src = f.source or "unknown"
        source_counts[src] = source_counts.get(src, 0) + 1
    
    above_fold = sum(1 for f in findings if f.is_above_fold)
    interactive = sum(1 for f in findings if f.is_interactive)
    
    return {
        "total": len(findings),
        "by_severity": severity_counts,
        "by_category": category_counts,
        "by_source": source_counts,
        "above_fold_count": above_fold,
        "interactive_count": interactive,
    }


def compute_quality_score(
    findings: List[Finding],
    total_compared_elements: int = 100,
) -> Dict[str, Any]:
    """Compute an overall implementation quality score.
    
    Score is based on:
    - Number of findings relative to compared elements
    - Severity distribution (critical issues hurt score more)
    - Confidence-weighted severity
    
    Args:
        findings: List of findings
        total_compared_elements: Total elements that were compared
    
    Returns:
        Dict with quality score and breakdown
    """
    if total_compared_elements == 0:
        return {
            "score": 100,
            "grade": "A+",
            "penalty_breakdown": {},
            "interpretation": "No elements compared",
        }
    
    severity_penalties = {
        "critical": 20,
        "major": 10,
        "minor": 3,
        "nit": 1,
        "info": 0,
    }
    
    total_penalty = 0
    penalty_breakdown = {}
    
    for finding in findings:
        sev = finding.severity.value
        penalty = severity_penalties.get(sev, 3)
        
        conf_multiplier = CONFIDENCE_WEIGHTS.get(finding.confidence, 0.5)
        weighted_penalty = penalty * conf_multiplier
        
        total_penalty += weighted_penalty
        penalty_breakdown[sev] = penalty_breakdown.get(sev, 0) + weighted_penalty
    
    max_penalty = total_compared_elements * 2
    normalized_penalty = min(100, (total_penalty / max_penalty) * 100)
    
    score = max(0, 100 - normalized_penalty)
    
    if score >= 95:
        grade = "A+"
    elif score >= 90:
        grade = "A"
    elif score >= 85:
        grade = "B+"
    elif score >= 80:
        grade = "B"
    elif score >= 75:
        grade = "C+"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"
    
    return {
        "score": round(score, 1),
        "grade": grade,
        "total_penalty": round(total_penalty, 1),
        "penalty_breakdown": {k: round(v, 1) for k, v in penalty_breakdown.items()},
        "interpretation": _get_score_interpretation(score, len(findings)),
    }


def _get_score_interpretation(score: float, finding_count: int) -> str:
    """Get human-readable interpretation of quality score."""
    if score >= 95:
        return "Excellent implementation, very few issues found."
    elif score >= 85:
        return "Good implementation with minor differences."
    elif score >= 75:
        return "Acceptable implementation, some issues need attention."
    elif score >= 60:
        return f"Implementation needs work. {finding_count} issues found."
    else:
        return f"Significant deviations from design. {finding_count} issues require attention."
