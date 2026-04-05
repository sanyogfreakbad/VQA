"""Tunable thresholds for the Visual QA pipeline."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Thresholds:
    """All tunable thresholds for comparison pipeline."""
    
    # Stage 1: Pre-filter thresholds
    ssim_clean_threshold: float = 0.97      # Above this = CLEAN (skip)
    ssim_dirty_threshold: float = 0.90      # Below this = DIRTY (priority)
    pixel_diff_threshold: int = 100         # Pixel count threshold for dirty
    
    # DOM comparison tolerances
    font_size_tolerance: float = 2.0        # px difference allowed
    spacing_tolerance: float = 4.0          # px difference for padding/margins
    size_tolerance: float = 10.0            # px difference for width/height
    ratio_tolerance: float = 0.05           # 5% difference in ratios
    color_tolerance: float = 0.1            # Perceptual color distance
    
    # Shadow comparison
    shadow_blur_tolerance: float = 2.0      # px blur difference
    shadow_spread_tolerance: float = 2.0    # px spread difference
    shadow_offset_tolerance: float = 2.0    # px x/y offset difference
    
    # Aspect ratio
    aspect_ratio_tolerance: float = 0.05    # 5% aspect ratio difference
    
    # Opacity
    opacity_tolerance: float = 0.1          # 10% opacity difference
    
    # Gap/spacing
    gap_tolerance: float = 4.0              # px gap difference
    
    # Border
    border_width_tolerance: float = 1.0     # px border width difference
    
    # Stage 3: Deduplication
    iou_merge_threshold: float = 0.5        # IoU above this = same finding
    
    # Stage 4: Refinement
    refinement_confidence_threshold: float = 0.7  # Below this = needs refinement
    
    # Stage 5: Saliency
    above_fold_height: int = 800            # px from top considered "above fold"
    interactive_weight: float = 1.5         # Multiplier for interactive elements
    large_element_threshold: float = 50000  # px² area for "large" elements
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ssim_clean_threshold": self.ssim_clean_threshold,
            "ssim_dirty_threshold": self.ssim_dirty_threshold,
            "pixel_diff_threshold": self.pixel_diff_threshold,
            "font_size_tolerance": self.font_size_tolerance,
            "spacing_tolerance": self.spacing_tolerance,
            "size_tolerance": self.size_tolerance,
            "ratio_tolerance": self.ratio_tolerance,
            "color_tolerance": self.color_tolerance,
            "shadow_blur_tolerance": self.shadow_blur_tolerance,
            "shadow_spread_tolerance": self.shadow_spread_tolerance,
            "shadow_offset_tolerance": self.shadow_offset_tolerance,
            "aspect_ratio_tolerance": self.aspect_ratio_tolerance,
            "opacity_tolerance": self.opacity_tolerance,
            "gap_tolerance": self.gap_tolerance,
            "border_width_tolerance": self.border_width_tolerance,
            "iou_merge_threshold": self.iou_merge_threshold,
            "refinement_confidence_threshold": self.refinement_confidence_threshold,
            "above_fold_height": self.above_fold_height,
            "interactive_weight": self.interactive_weight,
            "large_element_threshold": self.large_element_threshold,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Thresholds":
        """Create from dictionary, using defaults for missing values."""
        defaults = cls()
        return cls(
            ssim_clean_threshold=d.get("ssim_clean_threshold", defaults.ssim_clean_threshold),
            ssim_dirty_threshold=d.get("ssim_dirty_threshold", defaults.ssim_dirty_threshold),
            pixel_diff_threshold=d.get("pixel_diff_threshold", defaults.pixel_diff_threshold),
            font_size_tolerance=d.get("font_size_tolerance", d.get("font_size", defaults.font_size_tolerance)),
            spacing_tolerance=d.get("spacing_tolerance", d.get("spacing", defaults.spacing_tolerance)),
            size_tolerance=d.get("size_tolerance", d.get("size", defaults.size_tolerance)),
            ratio_tolerance=d.get("ratio_tolerance", d.get("ratio", defaults.ratio_tolerance)),
            color_tolerance=d.get("color_tolerance", d.get("color", defaults.color_tolerance)),
            shadow_blur_tolerance=d.get("shadow_blur_tolerance", defaults.shadow_blur_tolerance),
            shadow_spread_tolerance=d.get("shadow_spread_tolerance", defaults.shadow_spread_tolerance),
            shadow_offset_tolerance=d.get("shadow_offset_tolerance", defaults.shadow_offset_tolerance),
            aspect_ratio_tolerance=d.get("aspect_ratio_tolerance", defaults.aspect_ratio_tolerance),
            opacity_tolerance=d.get("opacity_tolerance", defaults.opacity_tolerance),
            gap_tolerance=d.get("gap_tolerance", defaults.gap_tolerance),
            border_width_tolerance=d.get("border_width_tolerance", defaults.border_width_tolerance),
            iou_merge_threshold=d.get("iou_merge_threshold", defaults.iou_merge_threshold),
            refinement_confidence_threshold=d.get("refinement_confidence_threshold", defaults.refinement_confidence_threshold),
            above_fold_height=d.get("above_fold_height", defaults.above_fold_height),
            interactive_weight=d.get("interactive_weight", defaults.interactive_weight),
            large_element_threshold=d.get("large_element_threshold", defaults.large_element_threshold),
        )


# Default thresholds instance
THRESHOLDS = Thresholds()
