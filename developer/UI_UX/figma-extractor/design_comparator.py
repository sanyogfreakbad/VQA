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
    """Calculate text similarity ratio (0-1)."""
    t1, t2 = normalize_text(t1), normalize_text(t2)
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    
    if t1 == t2:
        return 1.0
    
    if t1 in t2 or t2 in t1:
        return min(len(t1), len(t2)) / max(len(t1), len(t2))
    
    words1 = set(t1.split())
    words2 = set(t2.split())
    if words1 and words2:
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)
    
    return 0.0


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
            
            column = "left" if (elem.x or 0) < container_w / 2 else "right"
            elem.column = column
            
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
        
        self._compare_text_elements()
        self._compare_report_cards()
        self._compare_layout_structure()
        
        return self._format_results()
    
    def _compare_text_elements(self):
        """Compare text elements between Figma and Web."""
        figma_texts = self.figma_extractor.extract_text_elements()
        web_texts = self.web_extractor.extract_text_elements()
        
        matched_web = set()
        
        for f_elem in figma_texts:
            best_match = None
            best_score = 0.3
            
            for i, w_elem in enumerate(web_texts):
                if i in matched_web:
                    continue
                
                score = text_similarity(f_elem.text or "", w_elem.text or "")
                if score > best_score:
                    best_score = score
                    best_match = (i, w_elem)
            
            if best_match:
                matched_web.add(best_match[0])
                self._compare_text_properties(f_elem, best_match[1])
            else:
                self.diffs.append(DiffItem(
                    element_name=f_elem.name,
                    element_text=f_elem.text,
                    diff_type=DiffType.ELEMENT_MISSING_IN_WEB.value,
                    figma_value=f_elem.text,
                    web_value=None,
                    delta="Missing in Web",
                    severity="error"
                ))
    
    def _compare_text_properties(self, figma: NormalizedElement, web: NormalizedElement):
        """Compare properties of matched text elements."""
        elem_name = figma.name or figma.text or "Unknown"
        elem_text = figma.text
        
        if figma.text and web.text and normalize_text(figma.text) != normalize_text(web.text):
            self.diffs.append(DiffItem(
                element_name=elem_name,
                element_text=elem_text,
                diff_type=DiffType.TEXT_CONTENT.value,
                figma_value=figma.text,
                web_value=web.text,
                delta="Text differs",
                severity="error"
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
                    severity="warning"
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
                    severity="warning" if diff > 4 else "info"
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
                    severity="warning"
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
                    severity="warning"
                ))
    
    def _compare_report_cards(self):
        """Compare report card elements specifically."""
        figma_cards = self.figma_extractor.extract_report_cards()
        web_cards = self.web_extractor.extract_report_cards()
        
        matched_web = set()
        
        for f_card in figma_cards:
            best_match = None
            best_score = 0.5
            
            for i, w_card in enumerate(web_cards):
                if i in matched_web:
                    continue
                
                score = text_similarity(f_card.text or "", w_card.text or "")
                if score > best_score:
                    best_score = score
                    best_match = (i, w_card)
            
            if best_match:
                matched_web.add(best_match[0])
                self._compare_card_properties(f_card, best_match[1])
            else:
                self.diffs.append(DiffItem(
                    element_name="Report Card",
                    element_text=f_card.text,
                    diff_type=DiffType.ELEMENT_MISSING_IN_WEB.value,
                    figma_value=f_card.text,
                    web_value=None,
                    delta="Card missing in web",
                    severity="error"
                ))
    
    def _compare_card_properties(self, figma: NormalizedElement, web: NormalizedElement):
        """Compare properties of matched report cards."""
        elem_name = f"Card: {figma.text[:30]}..." if len(figma.text or "") > 30 else f"Card: {figma.text}"
        elem_text = figma.text
        
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
                    severity="warning"
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
                    severity="info"
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
                        severity="warning" if diff > 8 else "info"
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
                    severity="warning"
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
                    severity="info"
                ))
        
        if figma.border_radius is not None and web.border_radius is not None:
            if figma.border_radius != web.border_radius:
                self.diffs.append(DiffItem(
                    element_name=elem_name,
                    element_text=elem_text,
                    diff_type=DiffType.BORDER_RADIUS.value,
                    figma_value=f"{figma.border_radius}px",
                    web_value=f"{web.border_radius}px",
                    delta=f"{web.border_radius - figma.border_radius:+.0f}px",
                    severity="info"
                ))
        
        if figma.font_weight and web.font_weight and figma.font_weight != web.font_weight:
            self.diffs.append(DiffItem(
                element_name=elem_name,
                element_text=elem_text,
                diff_type=DiffType.TEXT_FONT_WEIGHT.value,
                figma_value=figma.font_weight,
                web_value=web.font_weight,
                delta=f"{figma.font_weight} → {web.font_weight}",
                severity="warning"
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
                        severity="warning"
                    ))
    
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
        
        table_rows = []
        for diff in self.diffs:
            main_cat = self._get_main_category(diff.diff_type)
            sub_type = self._get_sub_type(diff.diff_type)
            
            table_rows.append({
                "element": diff.element_name,
                "text": diff.element_text[:50] + "..." if diff.element_text and len(diff.element_text) > 50 else diff.element_text,
                "category": main_cat,
                "sub_type": sub_type,
                "figma_value": diff.figma_value,
                "web_value": diff.web_value,
                "delta": diff.delta,
                "severity": diff.severity,
            })
        
        formatted_by_category = {}
        for cat, diffs in by_category.items():
            formatted_by_category[cat] = []
            for diff in diffs:
                sub_type = self._get_sub_type(diff.diff_type)
                formatted_by_category[cat].append({
                    "element": diff.element_name,
                    "text": diff.element_text[:50] + "..." if diff.element_text and len(diff.element_text) > 50 else diff.element_text,
                    "sub_type": sub_type,
                    "figma_value": diff.figma_value,
                    "web_value": diff.web_value,
                    "delta": diff.delta,
                    "severity": diff.severity,
                })
        
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
            "differences": table_rows,
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
