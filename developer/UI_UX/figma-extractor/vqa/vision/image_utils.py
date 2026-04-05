"""
Image utility functions for the Visual QA pipeline.

Provides:
- Image resizing for LLM submission (constrained to max dimensions)
- Base64 encoding for API payloads
- Annotation drawing (bounding boxes, labels, highlights)
- Image format conversion and optimization
"""

import base64
import io
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image, ImageDraw, ImageFont

from ..models.region import BoundingBox


@dataclass
class ImageSizeConfig:
    """Configuration for image resizing."""
    max_width: int = 1024           # Max width for LLM submission
    max_height: int = 1024          # Max height for LLM submission
    min_dimension: int = 32         # Minimum dimension to preserve
    quality: int = 90               # JPEG quality (1-100)
    format: str = "PNG"             # Output format (PNG or JPEG)
    maintain_aspect: bool = True    # Maintain aspect ratio


def resize_for_llm(
    img: Image.Image,
    config: ImageSizeConfig = None,
) -> Image.Image:
    """Resize image to fit within LLM input constraints.
    
    Most vision LLMs have optimal input sizes around 1024x1024.
    Larger images waste tokens without improving accuracy.
    
    Args:
        img: Input PIL Image
        config: Size constraints configuration
    
    Returns:
        Resized PIL Image
    """
    config = config or ImageSizeConfig()
    
    width, height = img.size
    
    # Check if resizing is needed
    if width <= config.max_width and height <= config.max_height:
        return img
    
    if config.maintain_aspect:
        # Calculate scale to fit within max dimensions
        scale = min(config.max_width / width, config.max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
    else:
        new_width = min(width, config.max_width)
        new_height = min(height, config.max_height)
    
    # Ensure minimum dimension
    new_width = max(new_width, config.min_dimension)
    new_height = max(new_height, config.min_dimension)
    
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def image_to_base64(
    img: Image.Image,
    format: str = "PNG",
    quality: int = 90,
) -> str:
    """Convert PIL Image to base64-encoded string.
    
    Args:
        img: PIL Image to encode
        format: Output format (PNG or JPEG)
        quality: JPEG quality (ignored for PNG)
    
    Returns:
        Base64-encoded string (without data URI prefix)
    """
    buffer = io.BytesIO()
    
    if format.upper() == "JPEG":
        # Convert to RGB if needed (JPEG doesn't support alpha)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        img.save(buffer, format="JPEG", quality=quality)
    else:
        img.save(buffer, format="PNG")
    
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def base64_to_image(b64_string: str) -> Image.Image:
    """Convert base64 string back to PIL Image.
    
    Handles both raw base64 and data URI format.
    
    Args:
        b64_string: Base64-encoded image string
    
    Returns:
        PIL Image
    """
    # Handle data URI prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data))


def image_to_data_uri(
    img: Image.Image,
    format: str = "PNG",
    quality: int = 90,
) -> str:
    """Convert PIL Image to data URI format.
    
    Data URIs can be embedded directly in HTML/JSON for LLM prompts.
    
    Args:
        img: PIL Image to encode
        format: Output format (PNG or JPEG)
        quality: JPEG quality
    
    Returns:
        Data URI string (e.g., "data:image/png;base64,...")
    """
    b64 = image_to_base64(img, format, quality)
    mime_type = "image/jpeg" if format.upper() == "JPEG" else "image/png"
    return f"data:{mime_type};base64,{b64}"


def prepare_for_gemini(
    img: Image.Image,
    max_width: int = 1024,
) -> Dict:
    """Prepare an image for Gemini API submission.
    
    Returns the format expected by google-generativeai library.
    
    Args:
        img: PIL Image to prepare
        max_width: Maximum width (Gemini optimal is around 1024)
    
    Returns:
        Dict with mime_type and data keys
    """
    # Resize if needed
    if img.width > max_width:
        scale = max_width / img.width
        new_height = int(img.height * scale)
        img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert to bytes
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    
    return {
        "mime_type": "image/png",
        "data": buffer.read(),
    }


def annotate_with_boxes(
    img: Image.Image,
    boxes: List[Union[BoundingBox, Dict]],
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    labels: List[str] = None,
    label_color: Tuple[int, int, int] = (255, 255, 255),
    label_bg_color: Tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """Draw bounding boxes on an image.
    
    Useful for visualizing detected differences and regions.
    
    Args:
        img: Input image (will not be modified)
        boxes: List of BoundingBox or dicts with x, y, width, height
        color: RGB color for box outline
        thickness: Line thickness in pixels
        labels: Optional list of labels (one per box)
        label_color: Text color for labels
        label_bg_color: Background color for label boxes
    
    Returns:
        New image with annotations
    """
    result = img.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except (OSError, IOError):
            font = ImageFont.load_default()
    
    for i, box in enumerate(boxes):
        if isinstance(box, BoundingBox):
            x, y, w, h = box.x, box.y, box.width, box.height
        else:
            x = box.get("x", 0)
            y = box.get("y", 0)
            w = box.get("width", 0)
            h = box.get("height", 0)
        
        # Draw rectangle
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        for t in range(thickness):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)
        
        # Draw label if provided
        if labels and i < len(labels):
            label = labels[i]
            
            # Calculate label size
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw label background
            label_x = x1
            label_y = max(0, y1 - text_height - 4)
            draw.rectangle(
                [label_x, label_y, label_x + text_width + 6, label_y + text_height + 4],
                fill=label_bg_color
            )
            
            # Draw label text
            draw.text((label_x + 3, label_y + 2), label, fill=label_color, font=font)
    
    return result


def highlight_differences(
    figma_img: Image.Image,
    web_img: Image.Image,
    diff_regions: List[Union[BoundingBox, Dict]],
    highlight_color: Tuple[int, int, int, int] = (255, 255, 0, 100),
) -> Tuple[Image.Image, Image.Image]:
    """Highlight difference regions on both images.
    
    Adds semi-transparent overlays on regions where differences were detected.
    
    Args:
        figma_img: Figma screenshot
        web_img: Web screenshot
        diff_regions: List of regions to highlight
        highlight_color: RGBA color for highlights
    
    Returns:
        Tuple of (highlighted_figma, highlighted_web)
    """
    def apply_highlights(img):
        result = img.copy().convert("RGBA")
        overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        for region in diff_regions:
            if isinstance(region, BoundingBox):
                x, y, w, h = region.x, region.y, region.width, region.height
            else:
                x = region.get("x", 0)
                y = region.get("y", 0)
                w = region.get("width", 0)
                h = region.get("height", 0)
            
            overlay_draw.rectangle(
                [int(x), int(y), int(x + w), int(y + h)],
                fill=highlight_color
            )
        
        return Image.alpha_composite(result, overlay)
    
    return apply_highlights(figma_img), apply_highlights(web_img)


def create_comparison_strip(
    images: List[Image.Image],
    labels: List[str] = None,
    direction: str = "horizontal",
    padding: int = 10,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Create a strip of images arranged horizontally or vertically.
    
    Args:
        images: List of PIL Images
        labels: Optional labels for each image
        direction: "horizontal" or "vertical"
        padding: Space between images
        background: Background color
    
    Returns:
        Combined image strip
    """
    if not images:
        return Image.new("RGB", (100, 100), background)
    
    # Ensure all images are RGB
    images = [img.convert("RGB") for img in images]
    
    label_height = 20 if labels else 0
    
    if direction == "horizontal":
        max_height = max(img.height for img in images) + label_height
        total_width = sum(img.width for img in images) + padding * (len(images) + 1)
        
        result = Image.new("RGB", (total_width, max_height + padding * 2), background)
        
        x = padding
        for i, img in enumerate(images):
            y = padding + label_height
            result.paste(img, (x, y))
            
            if labels and i < len(labels):
                draw = ImageDraw.Draw(result)
                draw.text((x + 5, padding), labels[i], fill=(0, 0, 0))
            
            x += img.width + padding
    else:
        max_width = max(img.width for img in images)
        total_height = sum(img.height + label_height for img in images) + padding * (len(images) + 1)
        
        result = Image.new("RGB", (max_width + padding * 2, total_height), background)
        
        y = padding
        for i, img in enumerate(images):
            x = padding
            
            if labels and i < len(labels):
                draw = ImageDraw.Draw(result)
                draw.text((x + 5, y), labels[i], fill=(0, 0, 0))
                y += label_height
            
            result.paste(img, (x, y))
            y += img.height + padding
    
    return result


def get_image_info(img: Image.Image) -> Dict:
    """Get metadata about an image.
    
    Args:
        img: PIL Image
    
    Returns:
        Dict with width, height, mode, format info
    """
    return {
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "format": img.format,
        "has_alpha": img.mode in ("RGBA", "LA", "PA"),
        "aspect_ratio": round(img.width / img.height, 3) if img.height > 0 else 0,
        "megapixels": round((img.width * img.height) / 1_000_000, 2),
    }


def estimate_base64_size(img: Image.Image, format: str = "PNG") -> int:
    """Estimate the base64-encoded size of an image in bytes.
    
    Useful for token budget estimation before LLM submission.
    
    Args:
        img: PIL Image
        format: Target format (PNG or JPEG)
    
    Returns:
        Estimated size in bytes
    """
    buffer = io.BytesIO()
    if format.upper() == "JPEG":
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        img.save(buffer, format="JPEG", quality=85)
    else:
        img.save(buffer, format="PNG")
    
    # Base64 encoding adds ~33% overhead
    raw_size = buffer.tell()
    b64_size = int(raw_size * 1.37)
    
    return b64_size


def normalize_image_pair(
    img1: Image.Image,
    img2: Image.Image,
) -> Tuple[Image.Image, Image.Image]:
    """Normalize a pair of images to the same size and mode.
    
    Used before comparison operations to ensure consistency.
    Resizes img2 to match img1 dimensions.
    
    Args:
        img1: Reference image
        img2: Image to resize
    
    Returns:
        Tuple of (img1, resized_img2)
    """
    # Match dimensions
    if img2.size != img1.size:
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
    
    # Match mode (both RGB for comparison)
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")
    
    return img1, img2


def crop_to_content(
    img: Image.Image,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    padding: int = 0,
) -> Image.Image:
    """Crop image to remove background-colored borders.
    
    Useful for removing excess whitespace from screenshots.
    
    Args:
        img: Input image
        background_color: Color to consider as background
        padding: Extra padding to keep around content
    
    Returns:
        Cropped image
    """
    import numpy as np
    
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb)
    
    # Find non-background pixels
    bg = np.array(background_color)
    mask = ~np.all(np.abs(arr - bg) < 10, axis=2)  # Tolerance of 10
    
    if not mask.any():
        return img
    
    # Find bounding box of content
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    # Apply padding
    y1 = max(0, y1 - padding)
    x1 = max(0, x1 - padding)
    y2 = min(img.height, y2 + padding + 1)
    x2 = min(img.width, x2 + padding + 1)
    
    return img.crop((x1, y1, x2, y2))


def add_border(
    img: Image.Image,
    width: int = 2,
    color: Tuple[int, int, int] = (200, 200, 200),
) -> Image.Image:
    """Add a border around an image.
    
    Args:
        img: Input image
        width: Border width in pixels
        color: Border color
    
    Returns:
        Image with border
    """
    bordered = Image.new(
        "RGB",
        (img.width + 2 * width, img.height + 2 * width),
        color
    )
    bordered.paste(img.convert("RGB"), (width, width))
    return bordered
