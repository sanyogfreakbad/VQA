"""
Pre-Filter - Stage 1 triage logic.

Classifies each region as CLEAN / SUSPECT / DIRTY based on:
- SSIM scores (perceptual similarity)
- Pixel diff percentages
- DOM comparison results

This determines what gets sent to the LLM (expensive) vs skipped (free).

Decision matrix:
┌─────────────────┬──────────────┬──────────────┬──────────────┐
│                 │ No DOM diff  │ Minor DOM    │ Major DOM    │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ SSIM > 0.97     │ CLEAN        │ SUSPECT      │ SUSPECT      │
│ SSIM 0.90-0.97  │ SUSPECT      │ DIRTY        │ DIRTY        │
│ SSIM < 0.90     │ DIRTY        │ DIRTY        │ DIRTY        │
└─────────────────┴──────────────┴──────────────┴──────────────┘
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..models.region import Region, BoundingBox
from ..models.enums import RegionStatus, Severity
from ..config.thresholds import Thresholds, THRESHOLDS

logger = logging.getLogger(__name__)


@dataclass
class TriageResult:
    """Result of pre-filter triage."""
    clean_regions: List[Region]
    suspect_regions: List[Region]
    dirty_regions: List[Region]
    total_regions: int
    skipped_count: int
    
    @property
    def regions_to_analyze(self) -> List[Region]:
        """All regions that need LLM analysis (suspect + dirty)."""
        return self.suspect_regions + self.dirty_regions
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_regions": self.total_regions,
            "clean_count": len(self.clean_regions),
            "suspect_count": len(self.suspect_regions),
            "dirty_count": len(self.dirty_regions),
            "skipped_count": self.skipped_count,
            "to_analyze_count": len(self.regions_to_analyze),
        }


def triage_regions(
    regions: List[Region],
    ssim_scores: Dict[str, float],
    pixel_scores: Dict[str, float],
    dom_diffs: List[Dict[str, Any]],
    thresholds: Optional[Thresholds] = None,
) -> TriageResult:
    """Classify each region based on all three signal sources.
    
    Args:
        regions: List of regions from segmenter
        ssim_scores: Dict mapping region_id to SSIM score (0-1)
        pixel_scores: Dict mapping region_id to pixel mismatch percentage (0-1)
        dom_diffs: List of DOM diff items with location info
        thresholds: Comparison thresholds (uses defaults if not provided)
    
    Returns:
        TriageResult with regions classified into clean/suspect/dirty
    """
    thresholds = thresholds or THRESHOLDS
    
    dom_severity_by_region = _map_dom_diffs_to_regions(dom_diffs, regions)
    
    clean = []
    suspect = []
    dirty = []
    
    for region in regions:
        ssim = ssim_scores.get(region.id, 0.95)
        pixel_pct = pixel_scores.get(region.id, 0.0)
        dom_severity = dom_severity_by_region.get(region.id, "none")
        
        region.ssim_score = ssim
        
        status = _classify_region(
            ssim, pixel_pct, dom_severity, thresholds
        )
        region.status = status
        
        if status == RegionStatus.CLEAN:
            clean.append(region)
        elif status == RegionStatus.SUSPECT:
            suspect.append(region)
        else:
            dirty.append(region)
    
    dirty.sort(key=lambda r: (r.ssim_score or 1.0, -r.visual_weight))
    suspect.sort(key=lambda r: (r.ssim_score or 1.0, -r.visual_weight))
    
    logger.info(
        f"Triage complete: {len(clean)} clean, {len(suspect)} suspect, {len(dirty)} dirty"
    )
    
    return TriageResult(
        clean_regions=clean,
        suspect_regions=suspect,
        dirty_regions=dirty,
        total_regions=len(regions),
        skipped_count=len(clean),
    )


def _classify_region(
    ssim: float,
    pixel_pct: float,
    dom_severity: str,
    thresholds: Thresholds,
) -> RegionStatus:
    """Classify a single region based on signals.
    
    Args:
        ssim: SSIM score (0-1, higher = more similar)
        pixel_pct: Pixel mismatch percentage (0-1, lower = more similar)
        dom_severity: DOM diff severity ("none", "info", "minor", "warning", "error")
        thresholds: Comparison thresholds
    
    Returns:
        RegionStatus classification
    """
    is_ssim_clean = ssim >= thresholds.ssim_clean_threshold
    is_ssim_dirty = ssim < thresholds.ssim_dirty_threshold
    
    is_pixel_clean = pixel_pct < 0.02
    is_pixel_dirty = pixel_pct > 0.10
    
    is_dom_clean = dom_severity in ("none", "info")
    is_dom_major = dom_severity in ("warning", "error", "critical")
    
    if is_ssim_dirty or is_pixel_dirty:
        return RegionStatus.DIRTY
    
    if is_dom_major:
        return RegionStatus.DIRTY
    
    if is_ssim_clean and is_pixel_clean and is_dom_clean:
        return RegionStatus.CLEAN
    
    return RegionStatus.SUSPECT


def _map_dom_diffs_to_regions(
    dom_diffs: List[Dict[str, Any]],
    regions: List[Region],
) -> Dict[str, str]:
    """Map DOM diffs to their corresponding regions.
    
    Finds which region each DOM diff belongs to based on bounding box overlap.
    Returns the highest severity found for each region.
    
    Args:
        dom_diffs: List of DOM diff dictionaries
        regions: List of regions
    
    Returns:
        Dict mapping region_id to highest severity ("none", "info", "minor", "warning", "error")
    """
    region_severities: Dict[str, List[str]] = {}
    
    for diff in dom_diffs:
        diff_bbox = _extract_bbox_from_diff(diff)
        if not diff_bbox:
            continue
        
        diff_severity = diff.get("severity", "info")
        
        best_match = None
        best_iou = 0.0
        
        for region in regions:
            iou = region.bbox.iou(diff_bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = region
            
            if region.bbox.contains(diff_bbox) or diff_bbox.contains(region.bbox):
                best_match = region
                break
        
        if best_match and best_iou > 0.1:
            if best_match.id not in region_severities:
                region_severities[best_match.id] = []
            region_severities[best_match.id].append(diff_severity)
    
    severity_rank = {"none": 0, "info": 1, "nit": 2, "minor": 3, "warning": 4, "major": 5, "error": 6, "critical": 7}
    
    result = {}
    for region_id, severities in region_severities.items():
        max_severity = max(severities, key=lambda s: severity_rank.get(s, 0))
        result[region_id] = max_severity
    
    for region in regions:
        if region.id not in result:
            result[region.id] = "none"
    
    return result


def _extract_bbox_from_diff(diff: Dict[str, Any]) -> Optional[BoundingBox]:
    """Extract bounding box from a DOM diff item.
    
    Args:
        diff: DOM diff dictionary
    
    Returns:
        BoundingBox or None if not available
    """
    if "web_position" in diff:
        pos = diff["web_position"]
        return BoundingBox(
            x=pos.get("x", 0),
            y=pos.get("y", 0),
            width=pos.get("width", 100),
            height=pos.get("height", 100),
        )
    
    if "position" in diff:
        pos = diff["position"]
        return BoundingBox(
            x=pos.get("x", 0),
            y=pos.get("y", 0),
            width=pos.get("width", 100),
            height=pos.get("height", 100),
        )
    
    if "bbox" in diff:
        return BoundingBox.from_dict(diff["bbox"])
    
    return None


def compute_ssim_scores_for_regions(
    regions: List[Region],
    ssim_result: Any,
) -> Dict[str, float]:
    """Extract SSIM scores for each region from the full SSIM result.
    
    Args:
        regions: List of regions
        ssim_result: SSIMResult from ssim_scorer
    
    Returns:
        Dict mapping region_id to SSIM score
    """
    scores = {}
    
    if hasattr(ssim_result, 'ssim_map') and ssim_result.ssim_map is not None:
        import numpy as np
        ssim_map = ssim_result.ssim_map
        
        for region in regions:
            x = int(max(0, region.bbox.x))
            y = int(max(0, region.bbox.y))
            w = int(region.bbox.width)
            h = int(region.bbox.height)
            
            if y + h > ssim_map.shape[0]:
                h = ssim_map.shape[0] - y
            if x + w > ssim_map.shape[1]:
                w = ssim_map.shape[1] - x
            
            if w > 0 and h > 0:
                region_ssim = ssim_map[y:y+h, x:x+w]
                scores[region.id] = float(np.mean(region_ssim))
            else:
                scores[region.id] = 0.95
    else:
        for region in regions:
            matched = None
            if hasattr(ssim_result, 'region_scores'):
                for rs in ssim_result.region_scores:
                    if (abs(rs.x - region.bbox.x) < 10 and 
                        abs(rs.y - region.bbox.y) < 10):
                        matched = rs
                        break
            
            if matched:
                scores[region.id] = matched.ssim_score
            else:
                scores[region.id] = 0.95
    
    return scores


def compute_pixel_scores_for_regions(
    regions: List[Region],
    pixel_result: Any,
) -> Dict[str, float]:
    """Extract pixel diff scores for each region.
    
    Args:
        regions: List of regions  
        pixel_result: PixelDiffResult from pixel_diff
    
    Returns:
        Dict mapping region_id to pixel mismatch percentage
    """
    scores = {}
    
    if hasattr(pixel_result, 'heatmap') and pixel_result.heatmap is not None:
        import numpy as np
        heatmap = pixel_result.heatmap
        threshold = 30
        
        for region in regions:
            x = int(max(0, region.bbox.x))
            y = int(max(0, region.bbox.y))
            w = int(region.bbox.width)
            h = int(region.bbox.height)
            
            if y + h > heatmap.shape[0]:
                h = heatmap.shape[0] - y
            if x + w > heatmap.shape[1]:
                w = heatmap.shape[1] - x
            
            if w > 0 and h > 0:
                region_heatmap = heatmap[y:y+h, x:x+w]
                mismatch_count = np.sum(region_heatmap > threshold)
                total_pixels = region_heatmap.size
                scores[region.id] = mismatch_count / total_pixels if total_pixels > 0 else 0.0
            else:
                scores[region.id] = 0.0
    else:
        for region in regions:
            matched = None
            if hasattr(pixel_result, 'region_scores'):
                for rs in pixel_result.region_scores:
                    if (abs(rs.x - region.bbox.x) < 10 and 
                        abs(rs.y - region.bbox.y) < 10):
                        matched = rs
                        break
            
            if matched:
                scores[region.id] = matched.mismatch_pct
            else:
                scores[region.id] = 0.0
    
    return scores


def run_pre_filter(
    regions: List[Region],
    ssim_result: Any,
    pixel_result: Any,
    dom_diffs: List[Dict[str, Any]],
    thresholds: Optional[Thresholds] = None,
) -> TriageResult:
    """Main entry point for pre-filter stage.
    
    Combines SSIM scores, pixel diff scores, and DOM diffs to triage
    each region into CLEAN/SUSPECT/DIRTY.
    
    Args:
        regions: Regions from segmenter
        ssim_result: SSIMResult from vision module
        pixel_result: PixelDiffResult from vision module
        dom_diffs: DOM comparison differences
        thresholds: Optional custom thresholds
    
    Returns:
        TriageResult with classified regions
    """
    ssim_scores = compute_ssim_scores_for_regions(regions, ssim_result)
    pixel_scores = compute_pixel_scores_for_regions(regions, pixel_result)
    
    result = triage_regions(
        regions, ssim_scores, pixel_scores, dom_diffs, thresholds
    )
    
    return result


def filter_dom_diffs_by_triage(
    dom_diffs: List[Dict[str, Any]],
    triage_result: TriageResult,
) -> Tuple[List[Dict], List[Dict]]:
    """Filter DOM diffs based on region triage status.
    
    Separates diffs into those in dirty/suspect regions (need validation)
    and those in clean regions (can skip).
    
    Args:
        dom_diffs: All DOM diffs
        triage_result: Triage classification result
    
    Returns:
        Tuple of (diffs_to_validate, diffs_to_skip)
    """
    regions_to_analyze = triage_result.regions_to_analyze
    clean_regions = triage_result.clean_regions
    
    to_validate = []
    to_skip = []
    
    for diff in dom_diffs:
        diff_bbox = _extract_bbox_from_diff(diff)
        if not diff_bbox:
            to_validate.append(diff)
            continue
        
        in_dirty_region = False
        for region in regions_to_analyze:
            if region.bbox.iou(diff_bbox) > 0.1:
                in_dirty_region = True
                break
        
        if in_dirty_region:
            to_validate.append(diff)
        else:
            in_clean = False
            for region in clean_regions:
                if region.bbox.iou(diff_bbox) > 0.1:
                    in_clean = True
                    break
            
            if in_clean:
                to_skip.append(diff)
            else:
                to_validate.append(diff)
    
    logger.info(f"DOM diffs: {len(to_validate)} to validate, {len(to_skip)} skipped")
    
    return to_validate, to_skip
