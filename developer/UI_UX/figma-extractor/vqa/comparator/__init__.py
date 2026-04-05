"""VQA Comparator Module - Extended DOM comparison and normalization."""

# Re-export from existing design_comparator for easy access
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from design_comparator import (
    DiffType,
    DiffItem,
    NormalizedElement,
    DesignDataExtractor,
    DesignComparator,
    compare_designs,
    # New comparison functions
    parse_box_shadow,
    compare_shadows,
    compare_overflow,
    compare_borders,
    compare_aspect_ratio,
    compare_opacity,
    compare_gaps,
    compare_position,
    # Utility functions
    rgb_to_hex,
    extract_color_from_fills,
    color_distance,
    normalize_text,
    text_similarity,
    is_same_report,
)

__all__ = [
    "DiffType",
    "DiffItem",
    "NormalizedElement",
    "DesignDataExtractor",
    "DesignComparator",
    "compare_designs",
    # New comparison functions
    "parse_box_shadow",
    "compare_shadows",
    "compare_overflow",
    "compare_borders",
    "compare_aspect_ratio",
    "compare_opacity",
    "compare_gaps",
    "compare_position",
    # Utility functions
    "rgb_to_hex",
    "extract_color_from_fills",
    "color_distance",
    "normalize_text",
    "text_similarity",
    "is_same_report",
]
