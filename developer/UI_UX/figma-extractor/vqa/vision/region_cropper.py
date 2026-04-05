"""
Region cropping utilities for preparing screenshot pairs for LLM analysis.

Crops corresponding regions from Figma and Web screenshots for targeted
comparison. Includes smart padding, size normalization, and batch preparation.

Key design decisions:
- Crop pairs should be the same size for easy visual comparison
- Include context padding around the target element
- Constrain crop sizes to be LLM-friendly (not too small, not too large)
- Support batch preparation for efficient API calls
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
import math

from ..models.region import Region, BoundingBox


@dataclass
class CropPair:
    """A pair of corresponding crops from Figma and Web screenshots."""
    figma_crop: Image.Image
    web_crop: Image.Image
    bbox: BoundingBox  # Original bounding box before padding
    padded_bbox: BoundingBox  # Actual crop area with padding
    region_id: Optional[str] = None
    element_name: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    @property
    def crop_size(self) -> Tuple[int, int]:
        """Width and height of the crops."""
        return self.figma_crop.size
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (without images)."""
        return {
            "region_id": self.region_id,
            "element_name": self.element_name,
            "bbox": self.bbox.to_dict(),
            "padded_bbox": self.padded_bbox.to_dict(),
            "crop_width": self.figma_crop.width,
            "crop_height": self.figma_crop.height,
            "metadata": self.metadata,
        }


@dataclass
class CropConfig:
    """Configuration for region cropping."""
    padding: int = 30           # Pixels of context around element
    min_size: int = 50          # Minimum crop dimension
    max_size: int = 600         # Maximum crop dimension (for LLM efficiency)
    maintain_aspect: bool = True  # Maintain aspect ratio when resizing
    background_color: Tuple[int, int, int] = (255, 255, 255)  # Fill for out-of-bounds


def crop_region(
    figma_img: Image.Image,
    web_img: Image.Image,
    bbox: Union[BoundingBox, Dict],
    config: CropConfig = None,
    region_id: str = None,
    element_name: str = None,
) -> CropPair:
    """Crop corresponding regions from Figma and Web screenshots.
    
    Args:
        figma_img: Full Figma screenshot
        web_img: Full Web screenshot (will be resized if different from Figma)
        bbox: Bounding box to crop (BoundingBox or dict with x, y, width, height)
        config: Crop configuration (padding, size limits)
        region_id: Optional identifier for the region
        element_name: Optional human-readable element name
    
    Returns:
        CropPair with corresponding crops from both images
    """
    config = config or CropConfig()
    
    # Normalize bbox to BoundingBox
    if isinstance(bbox, dict):
        bbox = BoundingBox.from_dict(bbox)
    
    # Ensure web image is same size as figma
    if web_img.size != figma_img.size:
        web_img = web_img.resize(figma_img.size, Image.Resampling.LANCZOS)
    
    img_width, img_height = figma_img.size
    
    # Calculate padded crop area
    padded_x = max(0, int(bbox.x - config.padding))
    padded_y = max(0, int(bbox.y - config.padding))
    padded_x2 = min(img_width, int(bbox.x + bbox.width + config.padding))
    padded_y2 = min(img_height, int(bbox.y + bbox.height + config.padding))
    
    padded_width = padded_x2 - padded_x
    padded_height = padded_y2 - padded_y
    
    # Create padded bbox
    padded_bbox = BoundingBox(
        x=padded_x, y=padded_y,
        width=padded_width, height=padded_height
    )
    
    # Crop both images
    crop_box = (padded_x, padded_y, padded_x2, padded_y2)
    figma_crop = figma_img.crop(crop_box)
    web_crop = web_img.crop(crop_box)
    
    # Handle size constraints
    figma_crop, web_crop = _enforce_size_limits(
        figma_crop, web_crop, config
    )
    
    return CropPair(
        figma_crop=figma_crop,
        web_crop=web_crop,
        bbox=bbox,
        padded_bbox=padded_bbox,
        region_id=region_id,
        element_name=element_name,
    )


def crop_regions(
    figma_img: Image.Image,
    web_img: Image.Image,
    regions: List[Union[Region, Dict]],
    config: CropConfig = None,
) -> List[CropPair]:
    """Crop multiple regions from both screenshots.
    
    Args:
        figma_img: Full Figma screenshot
        web_img: Full Web screenshot
        regions: List of Region objects or dicts with bbox info
        config: Crop configuration
    
    Returns:
        List of CropPair objects
    """
    config = config or CropConfig()
    
    # Ensure web image matches figma
    if web_img.size != figma_img.size:
        web_img = web_img.resize(figma_img.size, Image.Resampling.LANCZOS)
    
    crop_pairs = []
    
    for region in regions:
        if isinstance(region, Region):
            bbox = region.bbox
            region_id = region.id
            element_name = region.element_name
        elif isinstance(region, dict):
            if "bbox" in region:
                bbox = BoundingBox.from_dict(region["bbox"])
            else:
                bbox = BoundingBox(
                    x=region.get("x", 0),
                    y=region.get("y", 0),
                    width=region.get("width", 100),
                    height=region.get("height", 100),
                )
            region_id = region.get("id") or region.get("region_id")
            element_name = region.get("element_name")
        else:
            continue
        
        pair = crop_region(
            figma_img, web_img, bbox, config,
            region_id=region_id,
            element_name=element_name,
        )
        crop_pairs.append(pair)
    
    return crop_pairs


def _enforce_size_limits(
    figma_crop: Image.Image,
    web_crop: Image.Image,
    config: CropConfig,
) -> Tuple[Image.Image, Image.Image]:
    """Enforce minimum and maximum size limits on crops.
    
    - Crops smaller than min_size get padded with background color
    - Crops larger than max_size get downscaled
    
    Returns both crops at the same final size.
    """
    width, height = figma_crop.size
    
    # Handle crops that are too small
    if width < config.min_size or height < config.min_size:
        new_width = max(width, config.min_size)
        new_height = max(height, config.min_size)
        
        figma_crop = _pad_to_size(figma_crop, new_width, new_height, config.background_color)
        web_crop = _pad_to_size(web_crop, new_width, new_height, config.background_color)
        
        width, height = new_width, new_height
    
    # Handle crops that are too large
    if width > config.max_size or height > config.max_size:
        scale = min(config.max_size / width, config.max_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        figma_crop = figma_crop.resize((new_width, new_height), Image.Resampling.LANCZOS)
        web_crop = web_crop.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return figma_crop, web_crop


def _pad_to_size(
    img: Image.Image,
    target_width: int,
    target_height: int,
    background_color: Tuple[int, int, int],
) -> Image.Image:
    """Pad image to target size, centering the original."""
    if img.width >= target_width and img.height >= target_height:
        return img
    
    new_img = Image.new("RGB", (target_width, target_height), background_color)
    
    # Center the original image
    x_offset = (target_width - img.width) // 2
    y_offset = (target_height - img.height) // 2
    
    new_img.paste(img.convert("RGB"), (x_offset, y_offset))
    return new_img


def create_side_by_side(
    crop_pair: CropPair,
    separator_width: int = 4,
    separator_color: Tuple[int, int, int] = (128, 128, 128),
    labels: bool = True,
    label_height: int = 24,
) -> Image.Image:
    """Create a side-by-side comparison image from a crop pair.
    
    Useful for visual debugging and for sending to LLMs that
    expect comparison images in a single frame.
    
    Args:
        crop_pair: CropPair with figma and web crops
        separator_width: Width of the vertical separator line
        separator_color: RGB color for separator
        labels: Whether to add "Figma" / "Web" labels
        label_height: Height of the label area
    
    Returns:
        Single image with Figma on left, Web on right
    """
    figma = crop_pair.figma_crop.convert("RGB")
    web = crop_pair.web_crop.convert("RGB")
    
    w, h = figma.size
    total_height = h + (label_height if labels else 0)
    total_width = w * 2 + separator_width
    
    result = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    
    # Paste images
    y_offset = label_height if labels else 0
    result.paste(figma, (0, y_offset))
    result.paste(web, (w + separator_width, y_offset))
    
    # Draw separator
    for x in range(w, w + separator_width):
        for y in range(y_offset, total_height):
            result.putpixel((x, y), separator_color)
    
    # Add labels if requested
    if labels:
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(result)
            
            # Try to use a default font
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            except (OSError, IOError):
                try:
                    font = ImageFont.truetype("arial.ttf", 14)
                except (OSError, IOError):
                    font = ImageFont.load_default()
            
            # Draw labels
            draw.text((10, 4), "FIGMA (Design)", fill=(0, 100, 200), font=font)
            draw.text((w + separator_width + 10, 4), "WEB (Implementation)", fill=(200, 100, 0), font=font)
            
            # Draw line under labels
            draw.line([(0, label_height - 2), (total_width, label_height - 2)], fill=(200, 200, 200), width=1)
        except Exception:
            pass  # Skip labels if PIL drawing fails
    
    return result


def create_grid_comparison(
    crop_pairs: List[CropPair],
    max_cols: int = 2,
    cell_padding: int = 10,
    with_labels: bool = True,
) -> Image.Image:
    """Create a grid of side-by-side comparisons.
    
    Useful for creating a single image showing multiple comparisons
    for batch LLM analysis.
    
    Args:
        crop_pairs: List of CropPairs to include
        max_cols: Maximum columns in the grid
        cell_padding: Padding between grid cells
        with_labels: Add element name labels to each cell
    
    Returns:
        Single image with grid of comparisons
    """
    if not crop_pairs:
        return Image.new("RGB", (100, 100), (255, 255, 255))
    
    # Create side-by-side images for each pair
    comparisons = [
        create_side_by_side(pair, labels=with_labels)
        for pair in crop_pairs
    ]
    
    # Calculate grid dimensions
    n = len(comparisons)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    
    # Find max cell dimensions
    max_width = max(img.width for img in comparisons)
    max_height = max(img.height for img in comparisons)
    
    # Create result image
    grid_width = cols * max_width + (cols + 1) * cell_padding
    grid_height = rows * max_height + (rows + 1) * cell_padding
    
    result = Image.new("RGB", (grid_width, grid_height), (240, 240, 240))
    
    # Paste comparisons into grid
    for i, comparison in enumerate(comparisons):
        col = i % cols
        row = i // cols
        
        x = cell_padding + col * (max_width + cell_padding)
        y = cell_padding + row * (max_height + cell_padding)
        
        # Center smaller images within cell
        x_offset = (max_width - comparison.width) // 2
        y_offset = (max_height - comparison.height) // 2
        
        result.paste(comparison, (x + x_offset, y + y_offset))
    
    return result


def crop_at_zoom_level(
    figma_img: Image.Image,
    web_img: Image.Image,
    bbox: Union[BoundingBox, Dict],
    zoom: float = 2.0,
    config: CropConfig = None,
) -> CropPair:
    """Crop a region at higher zoom for refinement analysis.
    
    Used in Stage 4 (refinement) to get higher-resolution crops
    of uncertain findings.
    
    Args:
        figma_img: Full Figma screenshot
        web_img: Full Web screenshot
        bbox: Bounding box to crop
        zoom: Zoom factor (2.0 = 2x zoom)
        config: Crop configuration
    
    Returns:
        CropPair at specified zoom level
    """
    config = config or CropConfig()
    
    # Normalize bbox
    if isinstance(bbox, dict):
        bbox = BoundingBox.from_dict(bbox)
    
    # Ensure same size
    if web_img.size != figma_img.size:
        web_img = web_img.resize(figma_img.size, Image.Resampling.LANCZOS)
    
    img_w, img_h = figma_img.size
    
    # Reduce padding for zoomed crops (we want to focus on the element)
    zoom_padding = max(10, config.padding // 2)
    
    # Calculate crop area
    x1 = max(0, int(bbox.x - zoom_padding))
    y1 = max(0, int(bbox.y - zoom_padding))
    x2 = min(img_w, int(bbox.x + bbox.width + zoom_padding))
    y2 = min(img_h, int(bbox.y + bbox.height + zoom_padding))
    
    # Crop at original resolution
    figma_crop = figma_img.crop((x1, y1, x2, y2))
    web_crop = web_img.crop((x1, y1, x2, y2))
    
    # Upscale for zoom
    zoomed_width = int(figma_crop.width * zoom)
    zoomed_height = int(figma_crop.height * zoom)
    
    # Cap at max_size
    if zoomed_width > config.max_size or zoomed_height > config.max_size:
        scale = min(config.max_size / zoomed_width, config.max_size / zoomed_height)
        zoomed_width = int(zoomed_width * scale)
        zoomed_height = int(zoomed_height * scale)
    
    figma_crop = figma_crop.resize((zoomed_width, zoomed_height), Image.Resampling.LANCZOS)
    web_crop = web_crop.resize((zoomed_width, zoomed_height), Image.Resampling.LANCZOS)
    
    return CropPair(
        figma_crop=figma_crop,
        web_crop=web_crop,
        bbox=bbox,
        padded_bbox=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
        metadata={"zoom": zoom},
    )


def prepare_crops_for_llm(
    crop_pairs: List[CropPair],
    batch_size: int = 6,
    include_metadata: bool = True,
) -> List[Dict]:
    """Prepare crop pairs for LLM API calls.
    
    Groups crops into batches and prepares them with metadata
    for efficient API calls.
    
    Args:
        crop_pairs: List of CropPairs
        batch_size: Maximum crops per batch
        include_metadata: Whether to include region metadata
    
    Returns:
        List of batch dicts, each containing crop pairs and metadata
    """
    batches = []
    
    for i in range(0, len(crop_pairs), batch_size):
        batch_pairs = crop_pairs[i:i + batch_size]
        
        batch = {
            "batch_index": i // batch_size,
            "count": len(batch_pairs),
            "pairs": [],
        }
        
        for j, pair in enumerate(batch_pairs):
            pair_data = {
                "index": j,
                "figma_crop": pair.figma_crop,
                "web_crop": pair.web_crop,
                "crop_width": pair.figma_crop.width,
                "crop_height": pair.figma_crop.height,
            }
            
            if include_metadata:
                pair_data["region_id"] = pair.region_id
                pair_data["element_name"] = pair.element_name
                pair_data["bbox"] = pair.bbox.to_dict()
                pair_data["metadata"] = pair.metadata
            
            batch["pairs"].append(pair_data)
        
        batches.append(batch)
    
    return batches
