"""
Region Segmenter - Segment page into comparable regions.

Strategy:
1. Use the DOM structure to identify logical regions (nav, header, cards, tables, footer)
2. Each region gets a bounding box
3. Overlay the pixel-diff heatmap to find additional regions the DOM missed

Regions are the unit of analysis for the rest of the pipeline.
"""

import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..models.region import Region, BoundingBox
from ..models.enums import RegionStatus

logger = logging.getLogger(__name__)


INTERACTIVE_ELEMENTS = {
    "button", "input", "select", "textarea", "a", "checkbox", "radio",
    "toggle", "dropdown", "switch", "slider", "datepicker", "tab", "modal",
}

SEMANTIC_TYPE_KEYWORDS = {
    "nav": ["nav", "navigation", "menu", "sidebar", "breadcrumb"],
    "header": ["header", "topbar", "appbar", "toolbar"],
    "footer": ["footer", "bottom-nav"],
    "hero": ["hero", "banner", "jumbotron", "carousel"],
    "cta": ["cta", "call-to-action", "action-button", "primary-button"],
    "form": ["form", "input-group", "field", "login", "signup", "register"],
    "modal": ["modal", "dialog", "popup", "overlay", "drawer"],
    "card": ["card", "tile", "panel", "box", "container"],
    "table": ["table", "grid", "list", "data-table", "datagrid"],
    "content": ["content", "main", "body", "article", "section"],
    "decorative": ["decoration", "ornament", "divider", "separator", "spacer"],
}


def segment_from_dom(
    figma_nodes: List[Dict[str, Any]],
    web_nodes: List[Dict[str, Any]],
    page_width: int,
    page_height: int,
    min_region_area: float = 2500,
    max_region_count: int = 100,
) -> List[Region]:
    """Create regions from DOM structure.
    
    Extracts top-level frames and significant elements from both Figma
    and Web DOM structures, then creates comparable regions.
    
    Args:
        figma_nodes: List of Figma node dictionaries
        web_nodes: List of Web DOM node dictionaries  
        page_width: Page width in pixels
        page_height: Page height in pixels
        min_region_area: Minimum area (px²) for a region
        max_region_count: Maximum regions to create
    
    Returns:
        List of Region objects ready for triage
    """
    regions = []
    seen_bboxes = set()
    
    figma_regions = _extract_regions_from_nodes(
        figma_nodes, page_width, page_height, "figma", min_region_area
    )
    
    web_regions = _extract_regions_from_nodes(
        web_nodes, page_width, page_height, "web", min_region_area
    )
    
    for region in figma_regions + web_regions:
        bbox_key = _bbox_hash(region.bbox)
        if bbox_key not in seen_bboxes:
            seen_bboxes.add(bbox_key)
            regions.append(region)
    
    for region in regions:
        region.visual_weight = _compute_saliency(
            region, page_width, page_height
        )
    
    regions.sort(key=lambda r: r.visual_weight, reverse=True)
    regions = regions[:max_region_count]
    
    logger.info(f"Segmented page into {len(regions)} regions")
    return regions


def _extract_regions_from_nodes(
    nodes: List[Dict[str, Any]],
    page_width: int,
    page_height: int,
    source: str,
    min_area: float,
) -> List[Region]:
    """Extract regions from a list of DOM nodes.
    
    Args:
        nodes: List of node dictionaries
        page_width: Page width
        page_height: Page height
        source: "figma" or "web"
        min_area: Minimum region area
    
    Returns:
        List of Region objects
    """
    regions = []
    
    for node in nodes:
        node_type = node.get("type", "").upper()
        name = node.get("name", "").lower()
        
        x = node.get("x", node.get("absoluteX", 0))
        y = node.get("y", node.get("absoluteY", 0))
        width = node.get("width", 0)
        height = node.get("height", 0)
        
        if "position" in node:
            pos = node["position"]
            x = pos.get("x", x)
            y = pos.get("y", y)
            width = pos.get("width", width)
            height = pos.get("height", height)
        
        if "bounds" in node:
            bounds = node["bounds"]
            x = bounds.get("x", x)
            y = bounds.get("y", y)
            width = bounds.get("width", width)
            height = bounds.get("height", height)
        
        area = width * height
        if area < min_area:
            continue
        
        if x < 0 or y < 0 or x + width > page_width * 1.1 or y + height > page_height * 1.5:
            continue
        
        is_frame = node_type in ("FRAME", "GROUP", "COMPONENT", "INSTANCE", "DIV", "SECTION")
        is_significant = area > min_area * 4 or _is_interactive_element(node)
        
        if not is_frame and not is_significant:
            continue
        
        semantic_type = _classify_semantic_type(node, name)
        is_interactive = _is_interactive_element(node)
        is_above_fold = y < 800
        
        bbox = BoundingBox(x=x, y=y, width=width, height=height)
        
        region = Region(
            id=f"region_{source}_{uuid.uuid4().hex[:8]}",
            bbox=bbox,
            status=RegionStatus.SUSPECT,
            element_name=node.get("name", name),
            element_type=semantic_type,
            figma_node_id=node.get("id") if source == "figma" else None,
            web_node_id=node.get("nodeId", node.get("id")) if source == "web" else None,
            is_above_fold=is_above_fold,
            is_interactive=is_interactive,
            visual_weight=1.0,
            figma_element=node if source == "figma" else None,
            web_element=node if source == "web" else None,
        )
        
        regions.append(region)
        
        children = node.get("children", [])
        if children and len(children) <= 20:
            child_regions = _extract_regions_from_nodes(
                children, page_width, page_height, source, min_area
            )
            regions.extend(child_regions)
    
    return regions


def _classify_semantic_type(node: Dict[str, Any], name: str) -> str:
    """Classify the semantic type of a node.
    
    Args:
        node: Node dictionary
        name: Lowercase node name
    
    Returns:
        Semantic type string
    """
    node_type = node.get("type", "").lower()
    tag_name = node.get("tagName", "").lower()
    role = node.get("role", "").lower()
    
    combined = f"{name} {node_type} {tag_name} {role}"
    
    for semantic_type, keywords in SEMANTIC_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in combined:
                return semantic_type
    
    if tag_name == "nav" or role == "navigation":
        return "nav"
    if tag_name == "header" or role == "banner":
        return "header"
    if tag_name == "footer" or role == "contentinfo":
        return "footer"
    if tag_name == "form":
        return "form"
    if tag_name == "table" or role == "grid":
        return "table"
    if tag_name in ("button", "a") or role == "button":
        return "cta"
    if tag_name == "dialog" or role == "dialog":
        return "modal"
    
    return "content"


def _is_interactive_element(node: Dict[str, Any]) -> bool:
    """Check if a node represents an interactive element.
    
    Args:
        node: Node dictionary
    
    Returns:
        True if interactive
    """
    name = node.get("name", "").lower()
    node_type = node.get("type", "").lower()
    tag_name = node.get("tagName", "").lower()
    
    for interactive in INTERACTIVE_ELEMENTS:
        if interactive in name or interactive in tag_name:
            return True
    
    if node.get("isInteractive") or node.get("clickable"):
        return True
    
    if "reactions" in node and node["reactions"]:
        return True
    
    if node.get("cursor") == "pointer":
        return True
    
    return False


def _compute_saliency(
    region: Region,
    page_width: int,
    page_height: int,
) -> float:
    """Compute saliency score for prioritization.
    
    Factors:
    - Position: above-fold (top 800px) gets 2x weight
    - Size: larger elements are more important
    - Interactivity: buttons, forms, CTAs get boost
    - Semantic type: nav, hero, CTA > footer, decorative
    
    Args:
        region: Region to score
        page_width: Page width
        page_height: Page height
    
    Returns:
        Saliency score 0.0 to 1.0
    """
    score = 0.5
    
    if region.is_above_fold:
        score += 0.2
    elif region.bbox.y < 1600:
        score += 0.1
    
    total_area = page_width * page_height
    if total_area > 0:
        area_ratio = region.bbox.area / total_area
        score += min(area_ratio * 2, 0.2)
    
    type_bonus = {
        "nav": 0.15,
        "header": 0.15,
        "hero": 0.2,
        "cta": 0.2,
        "form": 0.15,
        "modal": 0.15,
        "card": 0.1,
        "table": 0.1,
        "content": 0.05,
        "footer": 0.0,
        "decorative": -0.1,
    }
    score += type_bonus.get(region.element_type, 0)
    
    if region.is_interactive:
        score += 0.1
    
    return min(max(score, 0.0), 1.0)


def _bbox_hash(bbox: BoundingBox) -> str:
    """Create a hash key for deduplicating similar bboxes."""
    grid_x = int(bbox.x / 50)
    grid_y = int(bbox.y / 50)
    grid_w = int(bbox.width / 50)
    grid_h = int(bbox.height / 50)
    return f"{grid_x}_{grid_y}_{grid_w}_{grid_h}"


def segment_from_pixel_diff(
    pixel_regions: List[Dict[str, Any]],
    existing_regions: List[Region],
    page_width: int,
    page_height: int,
    overlap_threshold: float = 0.3,
) -> List[Region]:
    """Create additional regions from pixel diff hot spots.
    
    Identifies areas with significant pixel differences that weren't
    captured by DOM-based segmentation.
    
    Args:
        pixel_regions: Regions from pixel diff analysis
        existing_regions: Already-identified DOM regions
        page_width: Page width
        page_height: Page height
        overlap_threshold: IoU threshold to consider covered
    
    Returns:
        New regions from pixel diff (not overlapping existing)
    """
    new_regions = []
    
    for pr in pixel_regions:
        bbox = BoundingBox(
            x=pr.get("x", 0),
            y=pr.get("y", 0),
            width=pr.get("width", 64),
            height=pr.get("height", 64),
        )
        
        is_covered = False
        for existing in existing_regions:
            iou = bbox.iou(existing.bbox)
            if iou > overlap_threshold:
                is_covered = True
                break
        
        if not is_covered:
            region = Region(
                id=f"region_pixel_{uuid.uuid4().hex[:8]}",
                bbox=bbox,
                status=RegionStatus.DIRTY,
                element_name="Pixel diff region",
                element_type="content",
                is_above_fold=bbox.y < 800,
                is_interactive=False,
                visual_weight=0.6,
                pixel_diff_count=pr.get("mismatch_count", 0),
            )
            new_regions.append(region)
    
    return new_regions


def merge_overlapping_regions(
    regions: List[Region],
    iou_threshold: float = 0.5,
) -> List[Region]:
    """Merge regions that significantly overlap.
    
    Args:
        regions: List of regions to merge
        iou_threshold: IoU above this = merge
    
    Returns:
        Deduplicated list of regions
    """
    if not regions:
        return []
    
    regions = sorted(regions, key=lambda r: r.visual_weight, reverse=True)
    
    merged = []
    used = set()
    
    for i, region in enumerate(regions):
        if i in used:
            continue
        
        merged_bbox = region.bbox
        merged_ids = [i]
        
        for j, other in enumerate(regions):
            if j <= i or j in used:
                continue
            
            if merged_bbox.iou(other.bbox) > iou_threshold:
                merged_bbox = BoundingBox(
                    x=min(merged_bbox.x, other.bbox.x),
                    y=min(merged_bbox.y, other.bbox.y),
                    width=max(merged_bbox.x2, other.bbox.x2) - min(merged_bbox.x, other.bbox.x),
                    height=max(merged_bbox.y2, other.bbox.y2) - min(merged_bbox.y, other.bbox.y),
                )
                merged_ids.append(j)
                used.add(j)
        
        region.bbox = merged_bbox
        merged.append(region)
        used.add(i)
    
    return merged


def get_regions_for_triage(
    figma_nodes: List[Dict[str, Any]],
    web_nodes: List[Dict[str, Any]],
    pixel_regions: Optional[List[Dict[str, Any]]],
    page_width: int,
    page_height: int,
) -> List[Region]:
    """Main entry point: get all regions ready for triage.
    
    Combines DOM-based and pixel-based region detection, then merges
    overlapping regions.
    
    Args:
        figma_nodes: Figma DOM nodes
        web_nodes: Web DOM nodes
        pixel_regions: Optional pixel diff regions
        page_width: Page width
        page_height: Page height
    
    Returns:
        Complete list of regions for pre-filter triage
    """
    dom_regions = segment_from_dom(
        figma_nodes, web_nodes, page_width, page_height
    )
    
    if pixel_regions:
        pixel_new = segment_from_pixel_diff(
            pixel_regions, dom_regions, page_width, page_height
        )
        dom_regions.extend(pixel_new)
    
    merged_regions = merge_overlapping_regions(dom_regions)
    
    logger.info(f"Final region count after merge: {len(merged_regions)}")
    return merged_regions
