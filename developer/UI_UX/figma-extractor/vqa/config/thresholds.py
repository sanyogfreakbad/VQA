"""Tunable thresholds for the Visual QA pipeline.

All thresholds can be overridden via environment variables with the VQA_ prefix.
Example: VQA_SSIM_CLEAN=0.99 overrides ssim_clean_threshold.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any


def _env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    val = os.getenv(key)
    if val is not None:
        try:
            return float(val)
        except ValueError:
            pass
    return default


def _env_int(key: str, default: int) -> int:
    """Get int from environment variable."""
    val = os.getenv(key)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    return default


@dataclass
class Thresholds:
    """All tunable thresholds for comparison pipeline.
    
    Environment variable overrides (prefix with VQA_):
    - VQA_SSIM_CLEAN: ssim_clean_threshold
    - VQA_SSIM_DIRTY: ssim_dirty_threshold
    - VQA_PASS_A_FALLBACK: pass_a_fallback_threshold
    - VQA_LOCAL_DIFF_CONFIDENCE: local_diff_confidence_threshold
    """
    
    # Stage 1: Pre-filter thresholds (tuned for higher precision)
    ssim_clean_threshold: float = field(
        default_factory=lambda: _env_float("VQA_SSIM_CLEAN", 0.98)
    )
    ssim_dirty_threshold: float = field(
        default_factory=lambda: _env_float("VQA_SSIM_DIRTY", 0.88)
    )
    pixel_diff_threshold: int = field(
        default_factory=lambda: _env_int("VQA_PIXEL_DIFF", 100)
    )
    
    # Pass A fallback threshold - only run Pass A if overall SSIM is below this
    pass_a_fallback_threshold: float = field(
        default_factory=lambda: _env_float("VQA_PASS_A_FALLBACK", 0.85)
    )
    
    # Local diff engine confidence threshold - skip LLM if local confidence > this
    local_diff_confidence_threshold: float = field(
        default_factory=lambda: _env_float("VQA_LOCAL_DIFF_CONFIDENCE", 0.8)
    )
    
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
    
    # Pass B batching
    pass_b_batch_size: int = field(
        default_factory=lambda: _env_int("VQA_PASS_B_BATCH_SIZE", 10)
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ssim_clean_threshold": self.ssim_clean_threshold,
            "ssim_dirty_threshold": self.ssim_dirty_threshold,
            "pixel_diff_threshold": self.pixel_diff_threshold,
            "pass_a_fallback_threshold": self.pass_a_fallback_threshold,
            "local_diff_confidence_threshold": self.local_diff_confidence_threshold,
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
            "pass_b_batch_size": self.pass_b_batch_size,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Thresholds":
        """Create from dictionary, using defaults for missing values."""
        defaults = cls()
        return cls(
            ssim_clean_threshold=d.get("ssim_clean_threshold", defaults.ssim_clean_threshold),
            ssim_dirty_threshold=d.get("ssim_dirty_threshold", defaults.ssim_dirty_threshold),
            pixel_diff_threshold=d.get("pixel_diff_threshold", defaults.pixel_diff_threshold),
            pass_a_fallback_threshold=d.get("pass_a_fallback_threshold", defaults.pass_a_fallback_threshold),
            local_diff_confidence_threshold=d.get("local_diff_confidence_threshold", defaults.local_diff_confidence_threshold),
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
            pass_b_batch_size=d.get("pass_b_batch_size", defaults.pass_b_batch_size),
        )


# Default thresholds instance
THRESHOLDS = Thresholds()
