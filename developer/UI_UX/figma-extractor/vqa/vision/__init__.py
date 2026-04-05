"""VQA Vision Module - Image processing and analysis utilities.

This module provides the visual comparison layer for the VQA pipeline:

- pixel_diff: Pixel-level comparison with heatmap generation
- ssim_scorer: Structural similarity scoring for perceptual comparison
- region_cropper: Crop screenshot pairs for targeted LLM analysis
- image_utils: Image manipulation, encoding, and annotation utilities
"""

from .pixel_diff import (
    compute_pixel_diff,
    compute_diff_heatmap_image,
    get_high_diff_regions,
    merge_adjacent_regions,
    PixelDiffResult,
    RegionPixelScore,
)

from .ssim_scorer import (
    compute_ssim,
    compute_ssim_for_regions,
    compute_ssim_grid,
    classify_region_by_ssim,
    create_ssim_heatmap_image,
    get_low_ssim_regions,
    SSIMResult,
    RegionSSIMScore,
)

from .region_cropper import (
    crop_region,
    crop_regions,
    crop_at_zoom_level,
    create_side_by_side,
    create_grid_comparison,
    prepare_crops_for_llm,
    CropPair,
    CropConfig,
)

from .image_utils import (
    resize_for_llm,
    image_to_base64,
    base64_to_image,
    image_to_data_uri,
    prepare_for_gemini,
    annotate_with_boxes,
    highlight_differences,
    create_comparison_strip,
    get_image_info,
    estimate_base64_size,
    normalize_image_pair,
    crop_to_content,
    add_border,
    ImageSizeConfig,
)

__all__ = [
    # Pixel diff
    "compute_pixel_diff",
    "compute_diff_heatmap_image",
    "get_high_diff_regions",
    "merge_adjacent_regions",
    "PixelDiffResult",
    "RegionPixelScore",
    # SSIM
    "compute_ssim",
    "compute_ssim_for_regions",
    "compute_ssim_grid",
    "classify_region_by_ssim",
    "create_ssim_heatmap_image",
    "get_low_ssim_regions",
    "SSIMResult",
    "RegionSSIMScore",
    # Region cropper
    "crop_region",
    "crop_regions",
    "crop_at_zoom_level",
    "create_side_by_side",
    "create_grid_comparison",
    "prepare_crops_for_llm",
    "CropPair",
    "CropConfig",
    # Image utils
    "resize_for_llm",
    "image_to_base64",
    "base64_to_image",
    "image_to_data_uri",
    "prepare_for_gemini",
    "annotate_with_boxes",
    "highlight_differences",
    "create_comparison_strip",
    "get_image_info",
    "estimate_base64_size",
    "normalize_image_pair",
    "crop_to_content",
    "add_border",
    "ImageSizeConfig",
]
