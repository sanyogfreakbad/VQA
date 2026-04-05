"""Enums for Visual QA pipeline."""

from enum import Enum


class Severity(str, Enum):
    """Severity levels for findings."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    NIT = "nit"
    INFO = "info"


class Category(str, Enum):
    """Categories for visual differences."""
    TEXT = "text"
    SPACING = "spacing"
    PADDING = "padding"
    COLOR = "color"
    SIZE = "size"
    SHADOW = "shadow"
    BORDER = "border"
    OVERFLOW = "overflow"
    LAYOUT = "layout"
    POSITION = "position"
    OPACITY = "opacity"
    VISIBILITY = "visibility"
    ASPECT_RATIO = "aspect_ratio"
    BUTTONS_CTA = "buttons_cta"
    COMPONENTS = "components"
    ICONS = "icons"
    IMAGES = "images"
    MISSING_ELEMENTS = "missing_elements"
    BEHAVIORAL = "behavioral"
    OTHER = "other"


class Confidence(str, Enum):
    """Confidence levels for findings."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


class RegionStatus(str, Enum):
    """Status after pre-filter stage."""
    CLEAN = "clean"      # SSIM > 0.97 AND no DOM diff → skip
    SUSPECT = "suspect"  # 0.90 < SSIM < 0.97 OR minor DOM diff → queue
    DIRTY = "dirty"      # SSIM < 0.90 OR major DOM diff → priority queue


class DiffType(str, Enum):
    """Detailed diff types for comparison."""
    # Text properties
    TEXT_CONTENT = "text_content"
    TEXT_FONT_FAMILY = "text_font_family"
    TEXT_FONT_SIZE = "text_font_size"
    TEXT_FONT_WEIGHT = "text_font_weight"
    TEXT_LINE_HEIGHT = "text_line_height"
    TEXT_LETTER_SPACING = "text_letter_spacing"
    TEXT_ALIGNMENT = "text_alignment"
    TEXT_COLOR = "text_color"
    TEXT_DECORATION = "text_decoration"
    TEXT_TRANSFORM = "text_transform"
    
    # Spacing / Padding
    SPACING_PADDING_TOP = "spacing_padding_top"
    SPACING_PADDING_RIGHT = "spacing_padding_right"
    SPACING_PADDING_BOTTOM = "spacing_padding_bottom"
    SPACING_PADDING_LEFT = "spacing_padding_left"
    SPACING_ITEM_GAP = "spacing_item_gap"
    SPACING_VERTICAL_GAP = "spacing_vertical_gap"
    
    # Gap (modern CSS)
    GAP = "gap"
    ROW_GAP = "row_gap"
    COLUMN_GAP = "column_gap"
    
    # Colors
    COLOR_BACKGROUND = "color_background"
    COLOR_BORDER = "color_border"
    
    # Size
    SIZE_WIDTH = "size_width"
    SIZE_HEIGHT = "size_height"
    SIZE_WIDTH_RATIO = "size_width_ratio"
    SIZE_HEIGHT_RATIO = "size_height_ratio"
    SIZE_MIN_WIDTH = "size_min_width"
    SIZE_MAX_WIDTH = "size_max_width"
    SIZE_MIN_HEIGHT = "size_min_height"
    SIZE_MAX_HEIGHT = "size_max_height"
    
    # Border (general)
    BORDER_RADIUS = "border_radius"
    BORDER_WEIGHT = "border_weight"
    
    # Individual borders (catches missing dividers, stroke lines)
    BORDER_TOP = "border_top"
    BORDER_BOTTOM = "border_bottom"
    BORDER_LEFT = "border_left"
    BORDER_RIGHT = "border_right"
    
    # Shadow
    SHADOW_BOX = "shadow_box"
    SHADOW_DROP = "shadow_drop"
    SHADOW_INNER = "shadow_inner"
    
    # Overflow / Truncation
    OVERFLOW = "overflow"
    OVERFLOW_X = "overflow_x"
    OVERFLOW_Y = "overflow_y"
    TEXT_OVERFLOW = "text_overflow"
    WHITE_SPACE = "white_space"
    
    # Aspect ratio
    ASPECT_RATIO = "aspect_ratio"
    
    # Opacity
    OPACITY = "opacity"
    
    # Position
    POSITION_TYPE = "position_type"
    Z_INDEX = "z_index"
    
    # Display / Visibility
    DISPLAY = "display"
    VISIBILITY = "visibility"
    
    # Outline (focus states)
    OUTLINE = "outline"
    
    # Transform
    TRANSFORM = "transform"
    
    # Cursor
    CURSOR = "cursor"
    
    # Background image / gradient
    BACKGROUND_IMAGE = "background_image"
    
    # Scrollbar
    SCROLLBAR_PRESENCE = "scrollbar_presence"
    SCROLLBAR_WIDTH = "scrollbar_width"
    
    # Layout
    LAYOUT_MODE = "layout_mode"
    LAYOUT_ALIGNMENT = "layout_alignment"
    LAYOUT_JUSTIFY = "layout_justify"
    FLEX_DIRECTION = "flex_direction"
    
    # Components
    COMPONENT_TYPE = "component_type"
    COMPONENT_STATE = "component_state"
    
    # Elements
    ELEMENT_MISSING_IN_WEB = "element_missing_in_web"
    ELEMENT_ORDER = "element_order"
    
    # Visual (LLM-detected)
    ICON_MISSING = "icon_missing"
    ICON_DIFFERENT = "icon_different"
    IMAGE_MISSING = "image_missing"
    IMAGE_DIFFERENT = "image_different"
    VISUAL_HIERARCHY = "visual_hierarchy"
    DIVIDER_MISSING = "divider_missing"
    CROPPED_ELEMENT = "cropped_element"
