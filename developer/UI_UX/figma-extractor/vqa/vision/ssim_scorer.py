"""
Structural Similarity Index (SSIM) scoring per region.

SSIM is better than pixel diff for catching PERCEPTUAL differences:
- A 1px shift scores badly on pixel diff but looks fine to humans (SSIM stays high)
- A color shift from #1a1a1a to #000000 scores as different pixels but SSIM catches
  that it's imperceptible

Use SSIM to FILTER regions: SSIM > 0.97 means "looks the same to a human."

This module provides both full-image SSIM and per-region scoring for
efficient triage of page areas before LLM analysis.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np


@dataclass
class RegionSSIMScore:
    """SSIM score for a single region of the image."""
    x: int
    y: int
    width: int
    height: int
    ssim_score: float
    is_clean: bool  # SSIM > clean threshold
    
    def to_dict(self) -> Dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "ssim_score": round(self.ssim_score, 4),
            "is_clean": self.is_clean,
        }


@dataclass 
class SSIMResult:
    """Result of SSIM comparison between two images."""
    overall_ssim: float
    ssim_map: np.ndarray  # Per-pixel SSIM values
    region_scores: List[RegionSSIMScore]
    clean_region_count: int
    dirty_region_count: int
    
    def to_dict(self) -> Dict:
        return {
            "overall_ssim": round(self.overall_ssim, 4),
            "clean_region_count": self.clean_region_count,
            "dirty_region_count": self.dirty_region_count,
            "region_scores": [r.to_dict() for r in self.region_scores],
        }


def compute_ssim(
    figma_img: Image.Image,
    web_img: Image.Image,
    window_size: int = 11,
) -> Tuple[float, np.ndarray]:
    """Compute SSIM between two images.
    
    Uses the standard SSIM formula with Gaussian-weighted windows.
    Attempts to use scikit-image if available, falls back to manual implementation.
    
    Args:
        figma_img: Reference image (Figma screenshot)
        web_img: Test image (Web screenshot), resized to match reference
        window_size: Size of the sliding window (must be odd)
    
    Returns:
        Tuple of (overall SSIM score, per-pixel SSIM map)
    """
    # Resize web to match figma if needed
    if figma_img.size != web_img.size:
        web_img = web_img.resize(figma_img.size, Image.Resampling.LANCZOS)
    
    # Convert to grayscale for SSIM (standard practice)
    f_gray = np.array(figma_img.convert("L")).astype(np.float64)
    w_gray = np.array(web_img.convert("L")).astype(np.float64)
    
    # Try scikit-image first (more accurate)
    try:
        from skimage.metrics import structural_similarity
        score, ssim_map = structural_similarity(
            f_gray, w_gray,
            full=True,
            win_size=min(window_size, min(f_gray.shape) - 1) | 1,  # Ensure odd
            data_range=255,
        )
        return float(score), ssim_map
    except ImportError:
        # Fall back to manual implementation
        return _compute_ssim_manual(f_gray, w_gray, window_size)


def _compute_ssim_manual(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int = 11,
) -> Tuple[float, np.ndarray]:
    """Manual SSIM implementation using numpy.
    
    SSIM = (2*μx*μy + C1)(2*σxy + C2) / (μx² + μy² + C1)(σx² + σy² + C2)
    
    Where:
    - μx, μy are local means
    - σx², σy² are local variances
    - σxy is local covariance
    - C1, C2 are stabilization constants
    """
    # SSIM constants (from original paper)
    K1 = 0.01
    K2 = 0.03
    L = 255  # Dynamic range
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    
    # Create Gaussian window
    window = _gaussian_window(window_size)
    
    # Compute local statistics using convolution
    mu1 = _convolve2d(img1, window)
    mu2 = _convolve2d(img2, window)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = _convolve2d(img1 ** 2, window) - mu1_sq
    sigma2_sq = _convolve2d(img2 ** 2, window) - mu2_sq
    sigma12 = _convolve2d(img1 * img2, window) - mu1_mu2
    
    # SSIM formula
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / (denominator + 1e-10)  # Avoid division by zero
    
    # Clip to valid range
    ssim_map = np.clip(ssim_map, -1, 1)
    
    overall_ssim = float(np.mean(ssim_map))
    return overall_ssim, ssim_map


def _gaussian_window(size: int, sigma: float = 1.5) -> np.ndarray:
    """Create a Gaussian window for SSIM computation."""
    coords = np.arange(size) - (size - 1) / 2
    gauss_1d = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss_2d = np.outer(gauss_1d, gauss_1d)
    return gauss_2d / gauss_2d.sum()


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution using numpy (simple sliding window approach)."""
    from scipy.ndimage import convolve
    return convolve(image, kernel, mode='reflect')


def compute_ssim_for_regions(
    figma_img: Image.Image,
    web_img: Image.Image,
    regions: List[Dict],
    clean_threshold: float = 0.97,
) -> SSIMResult:
    """Compute SSIM scores for specific regions of the images.
    
    More efficient than computing full-image SSIM when you only need
    scores for specific DOM-identified regions.
    
    Args:
        figma_img: Reference image (Figma screenshot)
        web_img: Test image (Web screenshot)
        regions: List of region dicts with x, y, width, height
        clean_threshold: SSIM above this is considered "clean" (default 0.97)
    
    Returns:
        SSIMResult with per-region scores
    """
    # Resize web to match figma if needed
    if figma_img.size != web_img.size:
        web_img = web_img.resize(figma_img.size, Image.Resampling.LANCZOS)
    
    # Compute full SSIM map (we'll extract regions from it)
    overall_ssim, ssim_map = compute_ssim(figma_img, web_img)
    
    # Score each region
    region_scores = []
    clean_count = 0
    dirty_count = 0
    
    for region in regions:
        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        w = int(region.get("width", 0))
        h = int(region.get("height", 0))
        
        # Bounds checking
        x = max(0, min(x, ssim_map.shape[1] - 1))
        y = max(0, min(y, ssim_map.shape[0] - 1))
        w = min(w, ssim_map.shape[1] - x)
        h = min(h, ssim_map.shape[0] - y)
        
        if w <= 0 or h <= 0:
            continue
        
        # Extract region SSIM
        region_ssim = ssim_map[y:y+h, x:x+w]
        mean_ssim = float(np.mean(region_ssim))
        is_clean = mean_ssim >= clean_threshold
        
        if is_clean:
            clean_count += 1
        else:
            dirty_count += 1
        
        region_scores.append(RegionSSIMScore(
            x=x, y=y, width=w, height=h,
            ssim_score=mean_ssim,
            is_clean=is_clean,
        ))
    
    return SSIMResult(
        overall_ssim=overall_ssim,
        ssim_map=ssim_map,
        region_scores=region_scores,
        clean_region_count=clean_count,
        dirty_region_count=dirty_count,
    )


def compute_ssim_grid(
    figma_img: Image.Image,
    web_img: Image.Image,
    grid_size: int = 64,
    clean_threshold: float = 0.97,
    dirty_threshold: float = 0.90,
) -> SSIMResult:
    """Compute SSIM scores for a grid of regions across the image.
    
    Useful for identifying which areas of a page need attention
    without prior knowledge of DOM regions.
    
    Args:
        figma_img: Reference image
        web_img: Test image  
        grid_size: Size of each grid cell (pixels)
        clean_threshold: SSIM above this = clean (skip LLM)
        dirty_threshold: SSIM below this = dirty (priority for LLM)
    
    Returns:
        SSIMResult with grid-based region scores
    """
    # Compute full SSIM map
    overall_ssim, ssim_map = compute_ssim(figma_img, web_img)
    
    h, w = ssim_map.shape
    region_scores = []
    clean_count = 0
    dirty_count = 0
    
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            region_h = min(grid_size, h - y)
            region_w = min(grid_size, w - x)
            
            region_ssim = ssim_map[y:y+region_h, x:x+region_w]
            mean_ssim = float(np.mean(region_ssim))
            is_clean = mean_ssim >= clean_threshold
            
            if is_clean:
                clean_count += 1
            elif mean_ssim < dirty_threshold:
                dirty_count += 1
            
            region_scores.append(RegionSSIMScore(
                x=x, y=y, width=region_w, height=region_h,
                ssim_score=mean_ssim,
                is_clean=is_clean,
            ))
    
    return SSIMResult(
        overall_ssim=overall_ssim,
        ssim_map=ssim_map,
        region_scores=region_scores,
        clean_region_count=clean_count,
        dirty_region_count=dirty_count,
    )


def classify_region_by_ssim(
    ssim_score: float,
    clean_threshold: float = 0.97,
    dirty_threshold: float = 0.90,
) -> str:
    """Classify a region based on its SSIM score.
    
    Args:
        ssim_score: SSIM score (0-1)
        clean_threshold: Above this = CLEAN (skip)
        dirty_threshold: Below this = DIRTY (priority)
    
    Returns:
        "clean", "suspect", or "dirty"
    """
    if ssim_score >= clean_threshold:
        return "clean"
    elif ssim_score < dirty_threshold:
        return "dirty"
    else:
        return "suspect"


def create_ssim_heatmap_image(
    ssim_map: np.ndarray,
    low_color: Tuple[int, int, int] = (255, 0, 0),    # Red for low SSIM
    high_color: Tuple[int, int, int] = (0, 255, 0),   # Green for high SSIM
) -> Image.Image:
    """Convert SSIM map to a colored heatmap image.
    
    Low SSIM (differences) shown in red, high SSIM (similar) in green.
    
    Args:
        ssim_map: 2D array of SSIM values (-1 to 1, typically 0 to 1)
        low_color: RGB tuple for lowest SSIM
        high_color: RGB tuple for highest SSIM
    
    Returns:
        PIL Image with colored SSIM heatmap
    """
    # Normalize SSIM to 0-1 range (SSIM can be negative in extreme cases)
    normalized = np.clip((ssim_map + 1) / 2, 0, 1)  # Map [-1,1] to [0,1]
    
    h, w = ssim_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for c in range(3):
        colored[:, :, c] = (
            low_color[c] * (1 - normalized) + 
            high_color[c] * normalized
        ).astype(np.uint8)
    
    return Image.fromarray(colored)


def get_low_ssim_regions(
    result: SSIMResult,
    threshold: float = 0.95,
    max_regions: int = 50,
) -> List[Dict]:
    """Extract regions with lowest SSIM scores (most different).
    
    Useful for prioritizing which regions need LLM attention.
    
    Args:
        result: SSIMResult from compute_ssim_grid or compute_ssim_for_regions
        threshold: SSIM below this is considered "low"
        max_regions: Maximum regions to return
    
    Returns:
        List of region dicts sorted by SSIM ascending (worst first)
    """
    low_ssim = [
        r for r in result.region_scores
        if r.ssim_score < threshold
    ]
    
    # Sort by SSIM ascending (lowest/worst first)
    low_ssim.sort(key=lambda r: r.ssim_score)
    
    return [r.to_dict() for r in low_ssim[:max_regions]]
