"""
Pixel-level difference detection between Figma and Web screenshots.

Uses numpy for efficient pixel comparison to produce:
1. A difference heatmap image highlighting changed regions
2. A mismatch percentage per grid cell (for region-level scoring)
3. Total pixel difference statistics

This is the CHEAPEST check — runs in ~50ms, no LLM needed.
Only regions with pixel differences need further analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from PIL import Image
import numpy as np


@dataclass
class RegionPixelScore:
    """Pixel difference score for a single grid region."""
    x: int
    y: int
    width: int
    height: int
    mismatch_count: int
    mismatch_pct: float
    mean_diff_intensity: float


@dataclass
class PixelDiffResult:
    """Result of pixel-level comparison between two images."""
    total_pixels: int
    mismatch_count: int
    mismatch_pct: float
    mean_diff_intensity: float
    max_diff_intensity: float
    heatmap: np.ndarray
    diff_image: Image.Image
    region_scores: List[RegionPixelScore] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "total_pixels": self.total_pixels,
            "mismatch_count": self.mismatch_count,
            "mismatch_pct": round(self.mismatch_pct, 4),
            "mean_diff_intensity": round(self.mean_diff_intensity, 2),
            "max_diff_intensity": round(self.max_diff_intensity, 2),
            "region_scores": [
                {
                    "x": r.x, "y": r.y,
                    "width": r.width, "height": r.height,
                    "mismatch_count": r.mismatch_count,
                    "mismatch_pct": round(r.mismatch_pct, 4),
                    "mean_diff_intensity": round(r.mean_diff_intensity, 2),
                }
                for r in self.region_scores
            ]
        }


def compute_pixel_diff(
    figma_img: Image.Image,
    web_img: Image.Image,
    grid_size: int = 64,
    threshold: int = 30,
    min_region_mismatch_pct: float = 0.01,
) -> PixelDiffResult:
    """Compare two screenshots pixel-by-pixel.
    
    Uses Euclidean distance in RGB color space to measure difference.
    Produces a visual diff overlay and per-region scoring.
    
    Args:
        figma_img: Figma screenshot (PIL Image, source of truth)
        web_img: Web screenshot (PIL Image), will be resized to match Figma
        grid_size: Size of grid cells for region-level scoring (pixels)
        threshold: RGB distance threshold below which differences are
                   considered noise (antialiasing, compression artifacts)
        min_region_mismatch_pct: Minimum mismatch % to include a region
    
    Returns:
        PixelDiffResult containing heatmap, diff image, and statistics
    """
    # Ensure both images are same size (resize web to match figma)
    if figma_img.size != web_img.size:
        web_img = web_img.resize(figma_img.size, Image.Resampling.LANCZOS)
    
    # Convert to RGB numpy arrays
    f_arr = np.array(figma_img.convert("RGB")).astype(np.float32)
    w_arr = np.array(web_img.convert("RGB")).astype(np.float32)
    
    # Per-pixel Euclidean distance in RGB space
    # sqrt((r1-r2)^2 + (g1-g2)^2 + (b1-b2)^2)
    diff = np.sqrt(np.sum((f_arr - w_arr) ** 2, axis=2))
    
    # Apply threshold: differences below threshold are noise
    significant_diff = diff > threshold
    
    # Create heatmap (0-255 intensity based on diff magnitude)
    max_possible_diff = np.sqrt(3 * 255 ** 2)  # Max RGB distance
    heatmap = (np.clip(diff / max_possible_diff, 0, 1) * 255).astype(np.uint8)
    
    # Calculate statistics
    total_pixels = significant_diff.size
    mismatch_count = int(np.sum(significant_diff))
    mismatch_pct = mismatch_count / total_pixels if total_pixels > 0 else 0
    mean_diff = float(np.mean(diff[significant_diff])) if mismatch_count > 0 else 0
    max_diff = float(np.max(diff)) if diff.size > 0 else 0
    
    # Region-level scoring (grid-based)
    h, w = diff.shape
    region_scores = []
    
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            region_h = min(grid_size, h - y)
            region_w = min(grid_size, w - x)
            
            region_mask = significant_diff[y:y+region_h, x:x+region_w]
            region_diff = diff[y:y+region_h, x:x+region_w]
            
            region_mismatch = int(np.sum(region_mask))
            region_total = region_mask.size
            region_pct = region_mismatch / region_total if region_total > 0 else 0
            
            # Only include regions with significant differences
            if region_pct >= min_region_mismatch_pct:
                mean_intensity = float(np.mean(region_diff[region_mask])) if region_mismatch > 0 else 0
                region_scores.append(RegionPixelScore(
                    x=x, y=y,
                    width=region_w, height=region_h,
                    mismatch_count=region_mismatch,
                    mismatch_pct=region_pct,
                    mean_diff_intensity=mean_intensity,
                ))
    
    # Create visual diff image (red overlay on differing pixels)
    diff_image = _create_diff_overlay(figma_img, significant_diff)
    
    return PixelDiffResult(
        total_pixels=total_pixels,
        mismatch_count=mismatch_count,
        mismatch_pct=mismatch_pct,
        mean_diff_intensity=mean_diff,
        max_diff_intensity=max_diff,
        heatmap=heatmap,
        diff_image=diff_image,
        region_scores=region_scores,
    )


def _create_diff_overlay(
    base_img: Image.Image,
    diff_mask: np.ndarray,
    overlay_color: Tuple[int, int, int, int] = (255, 0, 0, 128),
) -> Image.Image:
    """Create a visual diff overlay highlighting differing pixels.
    
    Args:
        base_img: The base image to overlay on
        diff_mask: Boolean mask where True = pixel differs
        overlay_color: RGBA color for the overlay (default: semi-transparent red)
    
    Returns:
        PIL Image with red overlay on differing regions
    """
    # Convert base to RGBA
    result = base_img.copy().convert("RGBA")
    
    # Create overlay layer
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    overlay_arr = np.array(overlay)
    
    # Apply overlay color where pixels differ
    overlay_arr[diff_mask] = overlay_color
    
    overlay = Image.fromarray(overlay_arr)
    result = Image.alpha_composite(result, overlay)
    
    return result


def compute_diff_heatmap_image(
    heatmap: np.ndarray,
    colormap: str = "hot",
) -> Image.Image:
    """Convert grayscale heatmap array to colored heatmap image.
    
    Args:
        heatmap: 2D numpy array of diff intensities (0-255)
        colormap: Colormap style ('hot', 'jet', 'viridis')
    
    Returns:
        PIL Image with colored heatmap
    """
    # Simple colormap implementation without matplotlib dependency
    h, w = heatmap.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    if colormap == "hot":
        # Black -> Red -> Yellow -> White
        colored[:, :, 0] = np.clip(heatmap * 3, 0, 255)  # Red
        colored[:, :, 1] = np.clip((heatmap - 85) * 3, 0, 255)  # Green
        colored[:, :, 2] = np.clip((heatmap - 170) * 3, 0, 255)  # Blue
    elif colormap == "jet":
        # Blue -> Cyan -> Green -> Yellow -> Red
        norm = heatmap / 255.0
        colored[:, :, 0] = np.clip(1.5 - np.abs(norm - 0.75) * 4, 0, 1) * 255
        colored[:, :, 1] = np.clip(1.5 - np.abs(norm - 0.5) * 4, 0, 1) * 255
        colored[:, :, 2] = np.clip(1.5 - np.abs(norm - 0.25) * 4, 0, 1) * 255
    else:  # viridis-like
        norm = heatmap / 255.0
        colored[:, :, 0] = (68 + 187 * norm).astype(np.uint8)
        colored[:, :, 1] = (1 + 254 * norm ** 0.5).astype(np.uint8)
        colored[:, :, 2] = (84 - 84 * norm + 171 * norm ** 2).astype(np.uint8)
    
    return Image.fromarray(colored)


def get_high_diff_regions(
    result: PixelDiffResult,
    threshold_pct: float = 0.05,
    max_regions: int = 50,
) -> List[Dict]:
    """Extract regions with highest pixel differences.
    
    Useful for identifying which areas of the page need LLM attention.
    
    Args:
        result: PixelDiffResult from compute_pixel_diff
        threshold_pct: Minimum mismatch percentage to include (default 5%)
        max_regions: Maximum number of regions to return
    
    Returns:
        List of region dicts sorted by mismatch percentage descending
    """
    high_diff = [
        r for r in result.region_scores
        if r.mismatch_pct >= threshold_pct
    ]
    
    # Sort by mismatch percentage descending
    high_diff.sort(key=lambda r: r.mismatch_pct, reverse=True)
    
    return [
        {
            "x": r.x, "y": r.y,
            "width": r.width, "height": r.height,
            "mismatch_pct": r.mismatch_pct,
            "mean_diff_intensity": r.mean_diff_intensity,
        }
        for r in high_diff[:max_regions]
    ]


def merge_adjacent_regions(
    regions: List[Dict],
    gap_tolerance: int = 10,
) -> List[Dict]:
    """Merge adjacent high-diff regions into larger bounding boxes.
    
    Useful for creating crop regions that capture complete UI elements
    rather than fragmented grid cells.
    
    Args:
        regions: List of region dicts from get_high_diff_regions
        gap_tolerance: Maximum gap between regions to merge (pixels)
    
    Returns:
        List of merged region bounding boxes
    """
    if not regions:
        return []
    
    # Sort by position (top-left to bottom-right)
    sorted_regions = sorted(regions, key=lambda r: (r["y"], r["x"]))
    
    merged = []
    current = dict(sorted_regions[0])
    
    for region in sorted_regions[1:]:
        # Check if regions are close enough to merge
        x_close = (
            region["x"] <= current["x"] + current["width"] + gap_tolerance and
            region["x"] + region["width"] >= current["x"] - gap_tolerance
        )
        y_close = (
            region["y"] <= current["y"] + current["height"] + gap_tolerance and
            region["y"] + region["height"] >= current["y"] - gap_tolerance
        )
        
        if x_close and y_close:
            # Merge: expand current to encompass both
            new_x = min(current["x"], region["x"])
            new_y = min(current["y"], region["y"])
            new_x2 = max(current["x"] + current["width"], region["x"] + region["width"])
            new_y2 = max(current["y"] + current["height"], region["y"] + region["height"])
            
            current = {
                "x": new_x,
                "y": new_y,
                "width": new_x2 - new_x,
                "height": new_y2 - new_y,
                "mismatch_pct": max(current.get("mismatch_pct", 0), region.get("mismatch_pct", 0)),
                "mean_diff_intensity": max(current.get("mean_diff_intensity", 0), region.get("mean_diff_intensity", 0)),
            }
        else:
            merged.append(current)
            current = dict(region)
    
    merged.append(current)
    return merged
