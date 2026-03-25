#!/usr/bin/env python3
"""
Design Comparator Module

Core comparison logic for Figma vs Web design data.
Figma is treated as the source of truth - we only flag:
- Elements in Figma that are missing in Web
- Properties that differ between Figma (expected) and Web (actual)

We do NOT flag elements that exist in Web but not in Figma,
as those are implementation additions, not design violations.

No external dependencies except standard library.
"""

import re
import colorsys
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from collections import defaultdict


class DiffType(str, Enum):
    TEXT_CONTENT = "text_content"
    TEXT_FONT_FAMILY = "text_font_family"
    TEXT_FONT_SIZE = "text_font_size"
    TEXT_FONT_WEIGHT = "text_font_weight"
    TEXT_LINE_HEIGHT = "text_line_height"
    TEXT_LETTER_SPACING = "text_letter_spacing"
    TEXT_ALIGNMENT = "text_alignment"
    TEXT_COLOR = "text_color"
    
    SPACING_PADDING_TOP = "spacing_padding_top"
    SPACING_PADDING_RIGHT = "spacing_padding_right"
    SPACING_PADDING_BOTTOM = "spacing_padding_bottom"
    SPACING_PADDING_LEFT = "spacing_padding_left"
    SPACING_ITEM_GAP = "spacing_item_gap"
    SPACING_VERTICAL_GAP = "spacing_vertical_gap"
    
    COLOR_BACKGROUND = "color_background"
    COLOR_BORDER = "color_border"
    
    SIZE_WIDTH = "size_width"
    SIZE_HEIGHT = "size_height"
    SIZE_WIDTH_RATIO = "size_width_ratio"
    SIZE_HEIGHT_RATIO = "size_height_ratio"
    
    BORDER_RADIUS = "border_radius"
    BORDER_WEIGHT = "border_weight"
    
    LAYOUT_MODE = "layout_mode"
    LAYOUT_ALIGNMENT = "layout_alignment"
    
    COMPONENT_TYPE = "component_type"
    COMPONENT_STATE = "component_state"
    
    ELEMENT_MISSING_IN_WEB = "element_missing_in_web"
    ELEMENT_ORDER = "element_order"


# Placeholder text patterns to skip in missing-element reports
_PLACEHOLDER_PATTERNS = [
    "lorem ipsum",
    "dolor sit amet",
    "consectetur adipiscing",
]


def _is_placeholder_text(text: Optional[str]) -> bool:
    """Check if text is placeholder/lorem ipsum content."""
    if not text:
        return False
    lower = text.strip().lower()
    return any(pattern in lower for pattern in _PLACEHOLDER_PATTERNS)


@dataclass
class NormalizedElement:
    """Normalized element format for comparison."""
    source: str
    element_type: str
    element_id: str
    name: str
    text: Optional[str] = None
    index: int = 0
    column: Optional[str] = None
    
    font_family: Optional[str] = None
    font_weight: Optional[int] = None
    font_size: Optional[float] = None
    line_height: Optional[float] = None
    letter_spacing: Optional[float] = None
    text_align_h: Optional[str] = None
    text_align_v: Optional[str] = None
    text_color: Optional[str] = None
    
    width: Optional[float] = None
    height: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    
    container_width: Optional[float] = None
    container_height: Optional[float] = None
    width_ratio: Optional[float] = None
    height_ratio: Optional[float] = None
    
    padding_top: Optional[float] = None
    padding_right: Optional[float] = None
    padding_bottom: Optional[float] = None
    padding_left: Optional[float] = None
    item_spacing: Optional[float] = None
    
    bg_color: Optional[str] = None
    border_color: Optional[str] = None
    border_radius: Optional[float] = None
    stroke_weight: Optional[float] = None
    
    layout_mode: Optional[str] = None
    component_type: Optional[str] = None
    
    raw_node: Dict = field(default_factory=dict)


@dataclass
class DiffItem:
    """Single difference item for the comparison table."""
    element_name: str
    element_text: Optional[str]
    diff_type: str
    figma_value: Any
    web_value: Any
    delta: Any
    severity: str = "info"
    web_position: Optional[Dict[str, float]] = None
    web_node_id: Optional[str] = None
    web_locator: Optional[str] = None


def _web_pos(elem: Optional['NormalizedElement']) -> Optional[Dict[str, float]]:
    """Extract position dict from a NormalizedElement (for web elements)."""
    if elem is None:
        return None
    if elem.x is None and elem.y is None:
        return None
    return {
        "x": round(elem.x, 2) if elem.x is not None else 0,
        "y": round(elem.y, 2) if elem.y is not None else 0,
        "width": round(elem.width, 2) if elem.width is not None else 0,
        "height": round(elem.height, 2) if elem.height is not None else 0,
    }


def _web_node_id(elem: Optional['NormalizedElement']) -> Optional[str]:
    """Extract the DOM walker node ID (e.g. 'node_1079') from a web element."""
    if elem is None or not elem.raw_node:
        return None
    return elem.raw_node.get("id")


def _web_locator(elem: Optional['NormalizedElement']) -> Optional[str]:
    """Return the XPath locator pre-computed by the DOM walker.

    The locator is generated in-browser (inside the DOM_WALKER_SCRIPT)
    where the live DOM is available, using a priority chain:
      @id > @data-testid > @name > @role+@aria-label > @role+text()
      > @placeholder > text() > @aria-label > @class > bare tag.

    Returns None for Figma-only elements or when no raw_node data exists.
    """
    if elem is None or not elem.raw_node:
        return None
    return elem.raw_node.get("locator") or None


def rgb_to_hex(color: Dict) -> Optional[str]:
    """Convert RGB color dict to hex string."""
    if not color:
        return None
    r = int(color.get("r", 0))
    g = int(color.get("g", 0))
    b = int(color.get("b", 0))
    return f"#{r:02x}{g:02x}{b:02x}".upper()


def extract_color_from_fills(fills: List[Dict]) -> Optional[str]:
    """Extract hex color from fills array."""
    if not fills:
        return None
    for fill in fills:
        if fill.get("type") == "SOLID" and fill.get("color"):
            return rgb_to_hex(fill["color"])
    return None


def color_distance(hex1: Optional[str], hex2: Optional[str]) -> float:
    """Calculate perceptual color distance between two hex colors."""
    if not hex1 or not hex2:
        return float('inf') if (hex1 or hex2) else 0
    
    def hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    
    rgb1 = hex_to_rgb(hex1)
    rgb2 = hex_to_rgb(hex2)
    
    lab1 = colorsys.rgb_to_hls(*rgb1)
    lab2 = colorsys.rgb_to_hls(*rgb2)
    
    return sum((a - b) ** 2 for a, b in zip(lab1, lab2)) ** 0.5


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, strip whitespace)."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.lower().strip())


def text_similarity(t1: str, t2: str) -> float:
    """Calculate text similarity ratio (0-1) using multiple methods."""
    t1_norm, t2_norm = normalize_text(t1), normalize_text(t2)
    if not t1_norm and not t2_norm:
        return 1.0
    if not t1_norm or not t2_norm:
        return 0.0
    
    if t1_norm == t2_norm:
        return 1.0
    
    words1 = t1_norm.split()
    words2 = t2_norm.split()
    
    if len(words1) >= 2 and len(words2) >= 2:
        set1 = set(words1)
        set2 = set(words2)
        intersection = set1 & set2
        union = set1 | set2
        jaccard = len(intersection) / len(union) if union else 0
        
        seq_match = 0
        min_len = min(len(words1), len(words2))
        for i in range(min_len):
            if words1[i] == words2[i]:
                seq_match += 1
        seq_score = seq_match / max(len(words1), len(words2))
        
        return max(jaccard, seq_score)
    
    if t1_norm in t2_norm or t2_norm in t1_norm:
        return min(len(t1_norm), len(t2_norm)) / max(len(t1_norm), len(t2_norm))
    
    return 0.0


def is_same_report(t1: str, t2: str) -> bool:
    """Check if two texts refer to the same report (strict matching)."""
    t1_norm = normalize_text(t1)
    t2_norm = normalize_text(t2)
    
    if t1_norm == t2_norm:
        return True
    
    t1_clean = t1_norm.replace("report", "").replace("reconciliation", "recon").strip()
    t2_clean = t2_norm.replace("report", "").replace("reconciliation", "recon").strip()
    
    if t1_clean == t2_clean:
        return True
    
    words1 = set(t1_clean.split())
    words2 = set(t2_clean.split())
    
    words1.discard("")
    words2.discard("")
    
    if not words1 or not words2:
        return t1_norm == t2_norm
    
    if words1 == words2:
        return True
    
    if words1.issubset(words2) or words2.issubset(words1):
        larger = max(len(words1), len(words2))
        smaller = min(len(words1), len(words2))
        if smaller >= 2 and (smaller / larger) >= 0.7:
            return True
    
    diff = words1.symmetric_difference(words2)
    if len(diff) <= 1 and len(words1) >= 2:
        return True
    
    return False


class DesignDataExtractor:
    """Extract and normalize design data from Figma/Web JSON."""
    
    def __init__(self, data: Dict, source: str):
        self.data = data
        self.source = source
        self.nodes = data.get("nodes", [])
        self.container_cache = {}
        
    def find_container_dimensions(self) -> Tuple[float, float]:
        """Find the main content container dimensions."""
        for node in self.nodes:
            if node.get("type") == "FRAME":
                w = node.get("width", 0)
                h = node.get("height", 0)
                if w > 500 and h > 300:
                    return (w, h)
        return (1920, 1080)
    
    def extract_text_elements(self) -> List[NormalizedElement]:
        """Extract all text elements with normalized properties."""
        elements = []
        container_w, container_h = self.find_container_dimensions()
        
        for node in self.nodes:
            if node.get("type") != "TEXT":
                continue
            
            chars = node.get("characters", "")
            if not chars or chars.strip() == "":
                continue
            
            elem = NormalizedElement(
                source=self.source,
                element_type="TEXT",
                element_id=node.get("id", ""),
                name=node.get("name", ""),
                text=chars,
                
                font_family=node.get("fontFamily"),
                font_weight=node.get("fontWeight"),
                font_size=node.get("fontSize"),
                line_height=node.get("lineHeightPx"),
                letter_spacing=node.get("letterSpacing"),
                text_align_h=node.get("textAlignHorizontal"),
                text_align_v=node.get("textAlignVertical"),
                text_color=extract_color_from_fills(node.get("fills", [])),
                
                width=node.get("width"),
                height=node.get("height"),
                x=node.get("x"),
                y=node.get("y"),
                
                container_width=container_w,
                container_height=container_h,
                
                raw_node=node
            )
            
            if elem.width and container_w:
                elem.width_ratio = round(elem.width / container_w, 4)
            
            elements.append(elem)
        
        return elements
    
    def extract_frame_elements(self) -> List[NormalizedElement]:
        """Extract frame/container elements (cards, sections, etc.)."""
        elements = []
        container_w, container_h = self.find_container_dimensions()
        
        for node in self.nodes:
            if node.get("type") != "FRAME":
                continue
            
            w = node.get("width", 0)
            h = node.get("height", 0)
            
            if w < 50 or h < 20:
                continue
            
            elem = NormalizedElement(
                source=self.source,
                element_type="FRAME",
                element_id=node.get("id", ""),
                name=node.get("name", ""),
                
                width=w,
                height=h,
                x=node.get("x"),
                y=node.get("y"),
                
                container_width=container_w,
                container_height=container_h,
                
                padding_top=node.get("paddingTop"),
                padding_right=node.get("paddingRight"),
                padding_bottom=node.get("paddingBottom"),
                padding_left=node.get("paddingLeft"),
                item_spacing=node.get("itemSpacing"),
                
                bg_color=extract_color_from_fills(node.get("fills", [])),
                border_color=extract_color_from_fills(node.get("strokes", [])),
                border_radius=node.get("cornerRadius"),
                stroke_weight=node.get("strokeWeight"),
                
                layout_mode=node.get("layoutMode"),
                
                raw_node=node
            )
            
            if w and container_w:
                elem.width_ratio = round(w / container_w, 4)
            if h and container_h:
                elem.height_ratio = round(h / container_h, 4)
            
            elements.append(elem)
        
        return elements
    
    def extract_button_elements(self) -> List[NormalizedElement]:
        """Extract button/CTA components."""
        elements = []
        
        for node in self.nodes:
            node_type = node.get("type", "")
            name = node.get("name", "").lower()
            
            is_button = (
                node_type == "COMPONENT" or
                "button" in name or
                "cta" in name or
                "call to action" in name
            )
            
            if not is_button:
                continue
            
            elem = NormalizedElement(
                source=self.source,
                element_type="BUTTON",
                element_id=node.get("id", ""),
                name=node.get("name", ""),
                text=node.get("characters"),
                
                width=node.get("width"),
                height=node.get("height"),
                x=node.get("x"),
                y=node.get("y"),
                
                padding_top=node.get("paddingTop"),
                padding_right=node.get("paddingRight"),
                padding_bottom=node.get("paddingBottom"),
                padding_left=node.get("paddingLeft"),
                
                bg_color=extract_color_from_fills(node.get("fills", [])),
                border_color=extract_color_from_fills(node.get("strokes", [])),
                border_radius=node.get("cornerRadius"),
                
                component_type=node.get("componentType"),
                
                raw_node=node
            )
            
            elements.append(elem)
        
        return elements
    
    def extract_report_cards(self) -> List[NormalizedElement]:
        """Extract report card elements specifically (for this use case)."""
        elements = []
        container_w, container_h = self.find_container_dimensions()
        
        report_keywords = [
            "report", "payment", "variance", "audit", "aging", "approval",
            "register", "receipt", "goods", "match", "accrual", "invoice",
            "supplier", "price", "list", "po ", "requisition", "unmatched"
        ]
        
        text_nodes = [n for n in self.nodes if n.get("type") == "TEXT"]
        
        card_index = 0
        for node in text_nodes:
            chars = node.get("characters", "").lower()
            if not any(kw in chars for kw in report_keywords):
                continue
            
            if len(chars) < 5:
                continue
            
            parent_frame = self._find_parent_card(node)
            
            elem = NormalizedElement(
                source=self.source,
                element_type="REPORT_CARD",
                element_id=node.get("id", ""),
                name=node.get("name", ""),
                text=node.get("characters"),
                index=card_index,
                
                font_family=node.get("fontFamily"),
                font_weight=node.get("fontWeight"),
                font_size=node.get("fontSize"),
                line_height=node.get("lineHeightPx"),
                text_color=extract_color_from_fills(node.get("fills", [])),
                
                width=parent_frame.get("width") if parent_frame else node.get("width"),
                height=parent_frame.get("height") if parent_frame else node.get("height"),
                x=node.get("x"),
                y=node.get("y"),
                
                container_width=container_w,
                container_height=container_h,
                
                padding_top=parent_frame.get("paddingTop") if parent_frame else None,
                padding_right=parent_frame.get("paddingRight") if parent_frame else None,
                padding_bottom=parent_frame.get("paddingBottom") if parent_frame else None,
                padding_left=parent_frame.get("paddingLeft") if parent_frame else None,
                
                bg_color=extract_color_from_fills(parent_frame.get("fills", [])) if parent_frame else None,
                border_color=extract_color_from_fills(parent_frame.get("strokes", [])) if parent_frame else None,
                border_radius=parent_frame.get("cornerRadius") if parent_frame else None,
                stroke_weight=parent_frame.get("strokeWeight") if parent_frame else None,
                
                raw_node=node
            )
            
            if elem.width and container_w:
                elem.width_ratio = round(elem.width / container_w, 4)
            
            elements.append(elem)
            card_index += 1
        
        return elements
    
    def _find_parent_card(self, text_node: Dict) -> Optional[Dict]:
        """Find the parent card frame for a text node."""
        text_x = text_node.get("x", 0)
        text_y = text_node.get("y", 0)
        
        best_match = None
        best_area = float('inf')
        
        for node in self.nodes:
            if node.get("type") != "FRAME":
                continue
            
            fills = node.get("fills", [])
            strokes = node.get("strokes", [])
            if not fills and not strokes:
                continue
            
            nx, ny = node.get("x", 0), node.get("y", 0)
            nw, nh = node.get("width", 0), node.get("height", 0)
            
            if nx <= text_x <= nx + nw and ny <= text_y <= ny + nh:
                area = nw * nh
                if area < best_area and area > 100:
                    best_area = area
                    best_match = node
        
        return best_match


class DesignComparator:
    """Compare Figma and Web design extractions."""
    
    def __init__(self, figma_data: Dict, web_data: Dict):
        self.figma_extractor = DesignDataExtractor(figma_data, "figma")
        self.web_extractor = DesignDataExtractor(web_data, "web")
        self.diffs: List[DiffItem] = []
        
        self.tolerance = {
            "font_size": 2,
            "spacing": 4,
            "size": 10,
            "ratio": 0.05,
            "color": 0.1,
        }
    
    def compare_all(self) -> Dict:
        """Run all comparisons and return structured results."""
        self.diffs = []
        self._matched_figma_ids = set()
        self._matched_web_ids = set()
        
        self._compare_report_cards()
        self._compare_remaining_text_elements()
        self._compare_layout_structure()
        
        self._deduplicate_diffs()
        
        return self._format_results()
    
    def _deduplicate_diffs(self):
        """Remove duplicate diff entries, resolve conflicts, and filter placeholder text."""
        seen_keys = set()
        seen_texts_missing = set()
        unique_diffs = []
        
        missing_diffs = [d for d in self.diffs if d.diff_type == DiffType.ELEMENT_MISSING_IN_WEB.value]
        other_diffs = [d for d in self.diffs if d.diff_type != DiffType.ELEMENT_MISSING_IN_WEB.value]
        
        texts_with_other_diffs = set()
        for diff in other_diffs:
            text_key = normalize_text(diff.element_text or "")
            if text_key:
                texts_with_other_diffs.add(text_key)
        
        for diff in other_diffs:
            # Skip diffs where the element name is placeholder text
            if _is_placeholder_text(diff.element_name):
                continue
            
            text_key = normalize_text(diff.element_text or "")
            key = (text_key, diff.diff_type, str(diff.figma_value), str(diff.web_value))
            
            if key not in seen_keys:
                seen_keys.add(key)
                unique_diffs.append(diff)
        
        for diff in missing_diffs:
            # Skip missing-element diffs where the element name is placeholder text
            if _is_placeholder_text(diff.element_name):
                continue
            
            text_key = normalize_text(diff.element_text or "")
            
            if text_key in texts_with_other_diffs:
                continue
            
            if text_key in seen_texts_missing:
                continue
            
            seen_texts_missing.add(text_key)
            unique_diffs.append(diff)
        
        self.diffs = unique_diffs
    
    def _compare_remaining_text_elements(self):
        """Compare text elements not already matched as report cards."""
        figma_texts = self.figma_extractor.extract_text_elements()
        web_texts = self.web_extractor.extract_text_elements()
        
        figma_texts = [t for t in figma_texts if t.element_id not in self._matched_figma_ids]
        web_texts = [t for t in web_texts if t.element_id not in self._matched_web_ids]
        
        matched_web = set()
        
        for f_elem in figma_texts:
            best_match = None
            best_score = 0.7
            
            for i, w_elem in enumerate(web_texts):
                if i in matched_web:
                    continue
                
                if is_same_report(f_elem.text or "", w_elem.text or ""):
                    best_score = 1.0
                    best_match = (i, w_elem)
                    break
                
                score = text_similarity(f_elem.text or "", w_elem.text or "")
                if score > best_score:
                    best_score = score
                    best_match = (i, w_elem)
            
            if best_match and best_score >= 0.7:
                matched_web.add(best_match[0])
                self._compare_text_properties(f_elem, best_match[1], include_match_info=True)
            else:
                if len(f_elem.text or "") > 2:
                    self.diffs.append(DiffItem(
                        element_name=f_elem.name,
                        element_text=f_elem.text,
                        diff_type=DiffType.ELEMENT_MISSING_IN_WEB.value,
                        figma_value=f_elem.text,
                        web_value=None,
                        delta="Missing in Web",
                        severity="error"
                    ))
    
    def _compare_text_properties(self, figma: NormalizedElement, web: NormalizedElement, include_match_info: bool = False):
        """Compare properties of matched text elements."""
        elem_name = figma.name or figma.text or "Unknown"
        elem_text = figma.text
        pos = _web_pos(web)
        nid = _web_node_id(web)
        loc = _web_locator(web)
        
        f_text_norm = normalize_text(figma.text or "")
        w_text_norm = normalize_text(web.text or "")
        
        if f_text_norm and w_text_norm and f_text_norm != w_text_norm:
            if not is_same_report(figma.text or "", web.text or ""):
                self.diffs.append(DiffItem(
                    element_name=elem_name,
                    element_text=elem_text,
                    diff_type=DiffType.TEXT_CONTENT.value,
                    figma_value=figma.text,
                    web_value=web.text,
                    delta="Text differs",
                    severity="error",
                    web_position=pos,
                    web_node_id=nid,
                    web_locator=loc,
                ))
        
        if figma.font_family and web.font_family:
            f_font = figma.font_family.lower()
            w_font = web.font_family.lower()
            if f_font != w_font and f_font not in w_font and w_font not in f_font:
                self.diffs.append(DiffItem(
                    element_name=elem_name,
                    element_text=elem_text,
                    diff_type=DiffType.TEXT_FONT_FAMILY.value,
                    figma_value=figma.font_family,
                    web_value=web.font_family,
                    delta=f"{figma.font_family} → {web.font_family}",
                    severity="warning",
                    web_position=pos,
                    web_node_id=nid,
                    web_locator=loc,
                ))
        
        if figma.font_size and web.font_size:
            diff = abs(figma.font_size - web.font_size)
            if diff > self.tolerance["font_size"]:
                self.diffs.append(DiffItem(
                    element_name=elem_name,
                    element_text=elem_text,
                    diff_type=DiffType.TEXT_FONT_SIZE.value,
                    figma_value=f"{figma.font_size}px",
                    web_value=f"{web.font_size}px",
                    delta=f"{diff:+.1f}px",
                    severity="warning" if diff > 4 else "info",
                    web_position=pos,
                    web_node_id=nid,
                    web_locator=loc,
                ))
        
        if figma.font_weight and web.font_weight:
            if figma.font_weight != web.font_weight:
                self.diffs.append(DiffItem(
                    element_name=elem_name,
                    element_text=elem_text,
                    diff_type=DiffType.TEXT_FONT_WEIGHT.value,
                    figma_value=figma.font_weight,
                    web_value=web.font_weight,
                    delta=f"{figma.font_weight} → {web.font_weight}",
                    severity="warning",
                    web_position=pos,
                    web_node_id=nid,
                    web_locator=loc,
                ))
        
        if figma.text_color and web.text_color:
            dist = color_distance(figma.text_color, web.text_color)
            if dist > self.tolerance["color"]:
                self.diffs.append(DiffItem(
                    element_name=elem_name,
                    element_text=elem_text,
                    diff_type=DiffType.TEXT_COLOR.value,
                    figma_value=figma.text_color,
                    web_value=web.text_color,
                    delta=f"Distance: {dist:.2f}",
                    severity="warning",
                    web_position=pos,
                    web_node_id=nid,
                    web_locator=loc,
                ))
    
    def _compare_report_cards(self):
        """Compare report card elements specifically with strict matching."""
        figma_cards = self.figma_extractor.extract_report_cards()
        web_cards = self.web_extractor.extract_report_cards()
        
        matched_web_indices = set()
        matched_figma_indices = set()
        
        for i, f_card in enumerate(figma_cards):
            for j, w_card in enumerate(web_cards):
                if j in matched_web_indices:
                    continue
                if is_same_report(f_card.text or "", w_card.text or ""):
                    matched_web_indices.add(j)
                    matched_figma_indices.add(i)
                    self._matched_figma_ids.add(f_card.element_id)
                    self._matched_web_ids.add(w_card.element_id)
                    self._compare_card_properties(f_card, w_card)
                    break
        
        for i, f_card in enumerate(figma_cards):
            if i in matched_figma_indices:
                continue
            
            best_match = None
            best_score = 0.8
            
            for j, w_card in enumerate(web_cards):
                if j in matched_web_indices:
                    continue
                
                score = text_similarity(f_card.text or "", w_card.text or "")
                if score > best_score:
                    best_score = score
                    best_match = (j, w_card)
            
            if best_match:
                matched_web_indices.add(best_match[0])
                matched_figma_indices.add(i)
                self._matched_figma_ids.add(f_card.element_id)
                self._matched_web_ids.add(best_match[1].element_id)
                self._compare_card_properties(f_card, best_match[1])
            else:
                self.diffs.append(DiffItem(
                    element_name=f_card.text or "Report Card",
                    element_text=f_card.text,
                    diff_type=DiffType.ELEMENT_MISSING_IN_WEB.value,
                    figma_value=f_card.text,
                    web_value=None,
                    delta="Missing in Web",
                    severity="error"
                ))
    
    def _compare_card_properties(self, figma: NormalizedElement, web: NormalizedElement):
        """Compare properties of matched report cards."""
        elem_name = figma.text[:40] if figma.text else "Report Card"
        elem_text = figma.text
        pos = _web_pos(web)
        nid = _web_node_id(web)
        loc = _web_locator(web)
        
        f_text = (figma.text or "").strip()
        w_text = (web.text or "").strip()
        if f_text and w_text and f_text != w_text:
            self.diffs.append(DiffItem(
                element_name=elem_name,
                element_text=elem_text,
                diff_type=DiffType.TEXT_CONTENT.value,
                figma_value=f_text,
                web_value=w_text,
                delta=self._describe_text_diff(f_text, w_text),
                severity="error",
                web_position=pos,
                web_node_id=nid,
                web_locator=loc,
            ))
        
        if figma.font_weight and web.font_weight and figma.font_weight != web.font_weight:
            self.diffs.append(DiffItem(
                element_name=elem_name,
                element_text=elem_text,
                diff_type=DiffType.TEXT_FONT_WEIGHT.value,
                figma_value=figma.font_weight,
                web_value=web.font_weight,
                delta=f"{figma.font_weight} → {web.font_weight}",
                severity="warning",
                web_position=pos,
                web_node_id=nid,
                web_locator=loc,
            ))
        
        if figma.font_family and web.font_family:
            f_font = figma.font_family.lower()
            w_font = web.font_family.lower()
            if f_font != w_font and f_font not in w_font and w_font not in f_font:
                self.diffs.append(DiffItem(
                    element_name=elem_name,
                    element_text=elem_text,
                    diff_type=DiffType.TEXT_FONT_FAMILY.value,
                    figma_value=figma.font_family,
                    web_value=web.font_family,
                    delta=f"{figma.font_family} → {web.font_family}",
                    severity="warning",
                    web_position=pos,
                    web_node_id=nid,
                    web_locator=loc,
                ))
        
        if figma.text_color and web.text_color:
            dist = color_distance(figma.text_color, web.text_color)
            if dist > self.tolerance["color"]:
                self.diffs.append(DiffItem(
                    element_name=elem_name,
                    element_text=elem_text,
                    diff_type=DiffType.TEXT_COLOR.value,
                    figma_value=figma.text_color,
                    web_value=web.text_color,
                    delta=f"Distance: {dist:.2f}",
                    severity="warning",
                    web_position=pos,
                    web_node_id=nid,
                    web_locator=loc,
                ))
        
        if figma.width_ratio and web.width_ratio:
            diff = abs(figma.width_ratio - web.width_ratio)
            if diff > self.tolerance["ratio"]:
                self.diffs.append(DiffItem(
                    element_name=elem_name,
                    element_text=elem_text,
                    diff_type=DiffType.SIZE_WIDTH_RATIO.value,
                    figma_value=f"{figma.width_ratio:.2%}",
                    web_value=f"{web.width_ratio:.2%}",
                    delta=f"{diff:.2%}",
                    severity="warning",
                    web_position=pos,
                    web_node_id=nid,
                    web_locator=loc,
                ))
        
        if figma.height and web.height:
            diff = abs(figma.height - web.height)
            if diff > self.tolerance["size"]:
                self.diffs.append(DiffItem(
                    element_name=elem_name,
                    element_text=elem_text,
                    diff_type=DiffType.SIZE_HEIGHT.value,
                    figma_value=f"{figma.height}px",
                    web_value=f"{web.height}px",
                    delta=f"{diff:+.0f}px",
                    severity="info",
                    web_position=pos,
                    web_node_id=nid,
                    web_locator=loc,
                ))
        
        for prop, diff_type in [
            ("padding_top", DiffType.SPACING_PADDING_TOP),
            ("padding_right", DiffType.SPACING_PADDING_RIGHT),
            ("padding_bottom", DiffType.SPACING_PADDING_BOTTOM),
            ("padding_left", DiffType.SPACING_PADDING_LEFT),
        ]:
            f_val = getattr(figma, prop)
            w_val = getattr(web, prop)
            
            if f_val is not None and w_val is not None:
                diff = abs(f_val - w_val)
                if diff > self.tolerance["spacing"]:
                    self.diffs.append(DiffItem(
                        element_name=elem_name,
                        element_text=elem_text,
                        diff_type=diff_type.value,
                        figma_value=f"{f_val}px",
                        web_value=f"{w_val}px",
                        delta=f"{diff:+.0f}px",
                        severity="warning" if diff > 8 else "info",
                        web_position=pos,
                        web_node_id=nid,
                        web_locator=loc,
                    ))
        
        if figma.bg_color and web.bg_color:
            dist = color_distance(figma.bg_color, web.bg_color)
            if dist > self.tolerance["color"]:
                self.diffs.append(DiffItem(
                    element_name=elem_name,
                    element_text=elem_text,
                    diff_type=DiffType.COLOR_BACKGROUND.value,
                    figma_value=figma.bg_color,
                    web_value=web.bg_color,
                    delta=f"Distance: {dist:.2f}",
                    severity="warning",
                    web_position=pos,
                    web_node_id=nid,
                    web_locator=loc,
                ))
        
        if figma.border_color and web.border_color:
            dist = color_distance(figma.border_color, web.border_color)
            if dist > self.tolerance["color"]:
                self.diffs.append(DiffItem(
                    element_name=elem_name,
                    element_text=elem_text,
                    diff_type=DiffType.COLOR_BORDER.value,
                    figma_value=figma.border_color,
                    web_value=web.border_color,
                    delta=f"Distance: {dist:.2f}",
                    severity="info",
                    web_position=pos,
                    web_node_id=nid,
                    web_locator=loc,
                ))
        
        if figma.border_radius is not None and web.border_radius is not None:
            diff = abs(figma.border_radius - web.border_radius)
            if diff > 0:
                self.diffs.append(DiffItem(
                    element_name=elem_name,
                    element_text=elem_text,
                    diff_type=DiffType.BORDER_RADIUS.value,
                    figma_value=f"{figma.border_radius}px",
                    web_value=f"{web.border_radius}px",
                    delta=f"{diff:+.0f}px",
                    severity="info",
                    web_position=pos,
                    web_node_id=nid,
                    web_locator=loc,
                ))
        
    
    def _compare_layout_structure(self):
        """Compare overall layout structure."""
        figma_frames = self.figma_extractor.extract_frame_elements()
        web_frames = self.web_extractor.extract_frame_elements()
        
        figma_content = None
        web_content = None
        
        for frame in figma_frames:
            if frame.item_spacing is not None and frame.width and frame.width > 500:
                figma_content = frame
                break
        
        for frame in web_frames:
            if frame.width and frame.width > 500 and frame.height and frame.height > 200:
                web_content = frame
                break
        
        if figma_content and web_content:
            if figma_content.item_spacing and web_content.item_spacing is None:
                f_gap = figma_content.item_spacing
                w_gap = 16
                if f_gap != w_gap:
                    self.diffs.append(DiffItem(
                        element_name="Content Layout",
                        element_text=None,
                        diff_type=DiffType.SPACING_ITEM_GAP.value,
                        figma_value=f"{f_gap}px",
                        web_value=f"{w_gap}px (estimated)",
                        delta=f"{w_gap - f_gap:+.0f}px",
                        severity="warning",
                        web_position=_web_pos(web_content),
                        web_node_id=_web_node_id(web_content),
                        web_locator=_web_locator(web_content),
                    ))
    
    def _describe_text_diff(self, figma_text: str, web_text: str) -> str:
        """Describe the nature of a text difference."""
        f_lower = figma_text.lower()
        w_lower = web_text.lower()
        
        if "report" in f_lower and "report" not in w_lower:
            return "Missing 'Report' suffix"
        if "report" not in f_lower and "report" in w_lower:
            return "Extra 'Report' suffix"
        
        if len(figma_text) > len(web_text) + 3:
            return "Text truncated"
        if len(web_text) > len(figma_text) + 3:
            return "Text expanded"
        
        if "reconciliation" in f_lower and "recon" in w_lower:
            return "Abbreviated"
        if "recon" in f_lower and "reconciliation" in w_lower:
            return "Expanded abbreviation"
        
        return "Text differs"
    
    def _get_main_category(self, diff_type: str) -> str:
        """Map detailed diff type to main category."""
        if diff_type.startswith("text_"):
            return "text"
        elif diff_type in ["spacing_padding_top", "spacing_padding_right", 
                          "spacing_padding_bottom", "spacing_padding_left"]:
            return "padding"
        elif diff_type.startswith("spacing_"):
            return "spacing"
        elif diff_type.startswith("color_"):
            return "color"
        elif diff_type.startswith("size_"):
            return "size"
        elif diff_type.startswith("layout_"):
            return "components"
        elif diff_type.startswith("element_"):
            return "missing_elements"
        elif diff_type.startswith("button_") or diff_type.startswith("component_"):
            return "buttons_cta"
        else:
            return "other"
    
    def _get_sub_type(self, diff_type: str) -> str:
        """Get the specific sub-type for display."""
        sub_type_map = {
            "text_content": "content",
            "text_font_family": "font",
            "text_font_size": "size",
            "text_font_weight": "weight",
            "text_line_height": "line_height",
            "text_letter_spacing": "letter_spacing",
            "text_alignment": "alignment",
            "text_color": "color",
            "spacing_padding_top": "top",
            "spacing_padding_right": "right",
            "spacing_padding_bottom": "bottom",
            "spacing_padding_left": "left",
            "spacing_item_gap": "gap",
            "spacing_vertical_gap": "vertical_gap",
            "color_background": "background",
            "color_border": "border",
            "size_width": "width",
            "size_height": "height",
            "size_width_ratio": "width_ratio",
            "size_height_ratio": "height_ratio",
            "border_radius": "border_radius",
            "border_weight": "border_weight",
            "layout_mode": "layout_mode",
            "layout_alignment": "alignment",
            "component_type": "type",
            "component_state": "state",
            "element_missing_in_web": "missing",
        }
        return sub_type_map.get(diff_type, diff_type)
    
    def _format_results(self) -> Dict:
        """Format comparison results for output."""
        error_count = sum(1 for d in self.diffs if d.severity == "error")
        warning_count = sum(1 for d in self.diffs if d.severity == "warning")
        info_count = sum(1 for d in self.diffs if d.severity == "info")
        
        category_counts = defaultdict(int)
        by_category = defaultdict(list)
        
        for diff in self.diffs:
            main_cat = self._get_main_category(diff.diff_type)
            category_counts[main_cat] += 1
            by_category[main_cat].append(diff)
        
        formatted_by_category = {}
        for cat, diffs in by_category.items():
            formatted_by_category[cat] = []
            for diff in diffs:
                sub_type = self._get_sub_type(diff.diff_type)
                item = {
                    "element": diff.element_name,
                    "text": diff.element_text[:50] + "..." if diff.element_text and len(diff.element_text) > 50 else diff.element_text,
                    "sub_type": sub_type,
                    "figma_value": diff.figma_value,
                    "web_value": diff.web_value,
                    "delta": diff.delta,
                    "severity": diff.severity,
                }
                if diff.web_position:
                    item["web_position"] = diff.web_position
                # Include DOM node id and CSS locator for non-missing elements
                if cat != "missing_elements":
                    if diff.web_node_id:
                        item["web_node_id"] = diff.web_node_id
                    if diff.web_locator:
                        item["web_locator"] = diff.web_locator
                formatted_by_category[cat].append(item)
        
        ordered_categories = ["text", "spacing", "padding", "color", "buttons_cta", 
                            "components", "size", "missing_elements"]
        sorted_categories = {k: category_counts.get(k, 0) for k in ordered_categories if k in category_counts}
        for k, v in category_counts.items():
            if k not in sorted_categories:
                sorted_categories[k] = v
        
        return {
            "summary": {
                "total_differences": len(self.diffs),
                "errors": error_count,
                "warnings": warning_count,
                "info": info_count,
                "categories": dict(sorted_categories),
            },
            "by_category": formatted_by_category,
        }


def compare_designs(figma_data: Dict, web_data: Dict, tolerance: Dict = None) -> Dict:
    """
    Main entry point for design comparison.
    
    Args:
        figma_data: JSON dict from Figma extraction
        web_data: JSON dict from web extraction
        tolerance: Optional custom tolerance values
    
    Returns:
        Comparison results with summary and diff table
    """
    comparator = DesignComparator(figma_data, web_data)
    if tolerance:
        comparator.tolerance.update(tolerance)
    return comparator.compare_all()