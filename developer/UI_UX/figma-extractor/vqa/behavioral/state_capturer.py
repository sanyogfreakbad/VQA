"""
Before/after screenshot management for behavioral testing.

This module handles:
1. Organizing and storing state captures from interactions
2. Creating side-by-side comparisons for visual review
3. Managing Figma state references for comparison
4. Preparing image pairs for LLM analysis
5. Archiving captures for calibration library

The StateCapturer works with InteractionRunner to provide complete
behavioral testing capabilities.
"""

import asyncio
import base64
import io
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

from ..vision.image_utils import (
    resize_for_llm,
    image_to_base64,
    create_comparison_strip,
    annotate_with_boxes,
    add_border,
)
from ..models.region import BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class StateCapture:
    """A captured state for a single interaction."""
    id: str
    interaction_id: str
    element_selector: str
    element_name: str
    interaction_type: str
    
    before_image: Optional[Image.Image] = None
    after_image: Optional[Image.Image] = None
    figma_reference: Optional[Image.Image] = None
    
    before_crop: Optional[Image.Image] = None
    after_crop: Optional[Image.Image] = None
    figma_crop: Optional[Image.Image] = None
    
    element_bbox: Optional[BoundingBox] = None
    
    captured_at: str = ""
    page_url: str = ""
    viewport_size: Optional[Tuple[int, int]] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.captured_at:
            self.captured_at = datetime.utcnow().isoformat()
        if not self.id:
            self.id = f"state_{uuid.uuid4().hex[:12]}"
    
    @property
    def has_figma_reference(self) -> bool:
        """Check if a Figma reference is available."""
        return self.figma_reference is not None or self.figma_crop is not None
    
    def get_comparison_pair(self) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """Get the image pair for LLM comparison.
        
        Returns Figma reference vs after state if Figma is available,
        otherwise returns before vs after.
        """
        reference = self.figma_crop or self.figma_reference
        actual = self.after_crop or self.after_image
        
        if reference:
            return reference, actual
        else:
            return self.before_crop or self.before_image, actual
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (images as base64)."""
        result = {
            "id": self.id,
            "interaction_id": self.interaction_id,
            "element_selector": self.element_selector,
            "element_name": self.element_name,
            "interaction_type": self.interaction_type,
            "captured_at": self.captured_at,
            "page_url": self.page_url,
            "has_figma_reference": self.has_figma_reference,
            "metadata": self.metadata,
        }
        
        if self.element_bbox:
            result["element_bbox"] = self.element_bbox.to_dict()
        
        if self.viewport_size:
            result["viewport_size"] = list(self.viewport_size)
        
        return result
    
    def to_llm_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format suitable for LLM pass_c.
        
        This matches the expected format of InteractionResult.to_dict().
        """
        return {
            "id": self.id,
            "element": self.element_selector,
            "element_name": self.element_name,
            "element_type": self.metadata.get("element_type", "unknown"),
            "interaction": self.interaction_type,
            "before_screenshot": self.before_image,
            "after_screenshot": self.after_image,
            "before_crop": self.before_crop,
            "after_crop": self.after_crop,
            "figma_reference": self.figma_reference,
            "figma_crop": self.figma_crop,
            "element_bbox": self.element_bbox.to_dict() if self.element_bbox else None,
            "expected_changes": self.metadata.get("expected_changes", []),
            "validation_rules": self.metadata.get("validation_rules", []),
        }


@dataclass
class StateCaptureConfig:
    """Configuration for state capturing."""
    output_dir: Optional[str] = None
    save_full_screenshots: bool = True
    save_crops: bool = True
    save_comparisons: bool = True
    comparison_format: str = "horizontal"
    image_format: str = "PNG"
    image_quality: int = 90
    max_image_width: int = 1024
    add_labels: bool = True
    add_borders: bool = True
    border_width: int = 2


class StateCapturer:
    """Manages before/after screenshot captures for behavioral testing."""
    
    def __init__(
        self,
        config: Optional[StateCaptureConfig] = None,
        figma_states: Optional[Dict[str, Image.Image]] = None,
    ):
        """Initialize the state capturer.
        
        Args:
            config: Capture configuration
            figma_states: Dict mapping "element_interaction" keys to Figma state images
        """
        self.config = config or StateCaptureConfig()
        self.figma_states = figma_states or {}
        self.captures: List[StateCapture] = []
        
        if self.config.output_dir:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def create_capture_from_result(
        self,
        result: Dict[str, Any],
        page_url: str = "",
    ) -> StateCapture:
        """Create a StateCapture from an InteractionResult dict.
        
        Args:
            result: InteractionResult.to_dict() output
            page_url: URL of the page being tested
        
        Returns:
            StateCapture object
        """
        capture = StateCapture(
            id=f"state_{uuid.uuid4().hex[:12]}",
            interaction_id=result.get("id", ""),
            element_selector=result.get("element", ""),
            element_name=result.get("element_name", ""),
            interaction_type=result.get("interaction", ""),
            before_image=result.get("before_screenshot"),
            after_image=result.get("after_screenshot"),
            before_crop=result.get("before_crop"),
            after_crop=result.get("after_crop"),
            page_url=page_url,
            metadata={
                "element_type": result.get("element_type", "unknown"),
                "expected_changes": result.get("expected_changes", []),
                "validation_rules": result.get("validation_rules", []),
            },
        )
        
        bbox_data = result.get("element_bbox")
        if bbox_data:
            capture.element_bbox = BoundingBox.from_dict(bbox_data)
        
        figma_ref = self._find_figma_reference(
            result.get("element", ""),
            result.get("interaction", ""),
        )
        if figma_ref:
            capture.figma_reference = figma_ref
            if capture.element_bbox:
                capture.figma_crop = self._crop_figma_reference(
                    figma_ref,
                    capture.element_bbox,
                )
        
        self.captures.append(capture)
        return capture
    
    def _find_figma_reference(
        self,
        element_selector: str,
        interaction_type: str,
    ) -> Optional[Image.Image]:
        """Find a Figma reference image for an interaction.
        
        Args:
            element_selector: Element selector
            interaction_type: Interaction type
        
        Returns:
            Figma state image if found
        """
        keys_to_try = [
            f"{element_selector}_{interaction_type}",
            f"{interaction_type}_{element_selector}",
            element_selector,
            interaction_type,
        ]
        
        element_name = element_selector.split(">>")[0].strip()
        keys_to_try.extend([
            f"{element_name}_{interaction_type}",
            element_name,
        ])
        
        for key in keys_to_try:
            if key in self.figma_states:
                return self.figma_states[key]
        
        return None
    
    def _crop_figma_reference(
        self,
        figma_image: Image.Image,
        bbox: BoundingBox,
        padding: int = 50,
    ) -> Image.Image:
        """Crop Figma reference to match element region.
        
        Args:
            figma_image: Full Figma screenshot
            bbox: Element bounding box
            padding: Padding around element
        
        Returns:
            Cropped Figma image
        """
        x = max(0, int(bbox.x) - padding)
        y = max(0, int(bbox.y) - padding)
        x2 = min(figma_image.width, int(bbox.x2) + padding)
        y2 = min(figma_image.height, int(bbox.y2) + padding)
        
        return figma_image.crop((x, y, x2, y2))
    
    def load_figma_states(
        self,
        states_dir: str,
    ) -> int:
        """Load Figma state images from a directory.
        
        Expects files named like: button_hover.png, dropdown_click.png
        
        Args:
            states_dir: Directory containing Figma state images
        
        Returns:
            Number of states loaded
        """
        states_path = Path(states_dir)
        if not states_path.exists():
            logger.warning(f"Figma states directory not found: {states_dir}")
            return 0
        
        count = 0
        for img_path in states_path.glob("*.png"):
            key = img_path.stem
            try:
                self.figma_states[key] = Image.open(img_path)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to load Figma state {img_path}: {e}")
        
        for img_path in states_path.glob("*.jpg"):
            key = img_path.stem
            try:
                self.figma_states[key] = Image.open(img_path)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to load Figma state {img_path}: {e}")
        
        logger.info(f"Loaded {count} Figma state images")
        return count
    
    def create_comparison_image(
        self,
        capture: StateCapture,
        include_figma: bool = True,
    ) -> Image.Image:
        """Create a side-by-side comparison image.
        
        Args:
            capture: State capture
            include_figma: Include Figma reference if available
        
        Returns:
            Combined comparison image
        """
        images = []
        labels = []
        
        if capture.before_crop:
            img = capture.before_crop
            if self.config.add_borders:
                img = add_border(img, self.config.border_width, (200, 200, 200))
            images.append(img)
            labels.append("Before")
        
        if capture.after_crop:
            img = capture.after_crop
            if self.config.add_borders:
                img = add_border(img, self.config.border_width, (200, 200, 200))
            images.append(img)
            labels.append("After")
        
        if include_figma and capture.figma_crop:
            img = capture.figma_crop
            if self.config.add_borders:
                img = add_border(img, self.config.border_width, (100, 150, 255))
            images.append(img)
            labels.append("Figma (expected)")
        
        if not images:
            return Image.new("RGB", (200, 100), (255, 255, 255))
        
        direction = self.config.comparison_format
        use_labels = labels if self.config.add_labels else None
        
        return create_comparison_strip(
            images=images,
            labels=use_labels,
            direction=direction,
            padding=10,
        )
    
    def save_capture(
        self,
        capture: StateCapture,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """Save a capture's images to disk.
        
        Args:
            capture: State capture to save
            output_dir: Directory to save to (uses config if not specified)
        
        Returns:
            Dict mapping image type to file path
        """
        out_dir = output_dir or self.config.output_dir
        if not out_dir:
            logger.warning("No output directory specified, skipping save")
            return {}
        
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        base_name = f"{capture.element_name}_{capture.interaction_type}_{capture.id}"
        base_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in base_name)
        
        if self.config.save_full_screenshots:
            if capture.before_image:
                path = out_path / f"{base_name}_before.png"
                capture.before_image.save(path, format="PNG")
                saved_files["before"] = str(path)
            
            if capture.after_image:
                path = out_path / f"{base_name}_after.png"
                capture.after_image.save(path, format="PNG")
                saved_files["after"] = str(path)
        
        if self.config.save_crops:
            if capture.before_crop:
                path = out_path / f"{base_name}_before_crop.png"
                capture.before_crop.save(path, format="PNG")
                saved_files["before_crop"] = str(path)
            
            if capture.after_crop:
                path = out_path / f"{base_name}_after_crop.png"
                capture.after_crop.save(path, format="PNG")
                saved_files["after_crop"] = str(path)
            
            if capture.figma_crop:
                path = out_path / f"{base_name}_figma_crop.png"
                capture.figma_crop.save(path, format="PNG")
                saved_files["figma_crop"] = str(path)
        
        if self.config.save_comparisons:
            comparison = self.create_comparison_image(capture)
            path = out_path / f"{base_name}_comparison.png"
            comparison.save(path, format="PNG")
            saved_files["comparison"] = str(path)
        
        return saved_files
    
    def save_all_captures(
        self,
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Save all captures to disk.
        
        Args:
            output_dir: Directory to save to
        
        Returns:
            List of saved file dicts
        """
        results = []
        for capture in self.captures:
            files = self.save_capture(capture, output_dir)
            if files:
                results.append({
                    "capture_id": capture.id,
                    "files": files,
                })
        return results
    
    def prepare_for_llm(
        self,
        capture: StateCapture,
        max_width: int = 1024,
    ) -> Dict[str, Any]:
        """Prepare capture images for LLM analysis.
        
        Resizes images and converts to the format expected by GeminiClient.
        
        Args:
            capture: State capture
            max_width: Maximum image width
        
        Returns:
            Dict with prepared images and metadata
        """
        reference, actual = capture.get_comparison_pair()
        
        prepared = {
            "capture_id": capture.id,
            "element_name": capture.element_name,
            "interaction_type": capture.interaction_type,
            "has_figma_reference": capture.has_figma_reference,
        }
        
        if reference:
            prepared["reference_image"] = resize_for_llm(reference)
        
        if actual:
            prepared["actual_image"] = resize_for_llm(actual)
        
        if capture.element_bbox:
            prepared["element_bbox"] = capture.element_bbox.to_dict()
        
        prepared["expected_changes"] = capture.metadata.get("expected_changes", [])
        prepared["validation_rules"] = capture.metadata.get("validation_rules", [])
        
        return prepared
    
    def prepare_batch_for_llm(
        self,
        captures: Optional[List[StateCapture]] = None,
        max_width: int = 1024,
    ) -> List[Dict[str, Any]]:
        """Prepare multiple captures for batched LLM analysis.
        
        Args:
            captures: Captures to prepare (uses all if not specified)
            max_width: Maximum image width
        
        Returns:
            List of prepared capture dicts
        """
        if captures is None:
            captures = self.captures
        
        return [
            self.prepare_for_llm(capture, max_width)
            for capture in captures
        ]
    
    def get_captures_by_interaction(
        self,
        interaction_type: str,
    ) -> List[StateCapture]:
        """Get all captures for a specific interaction type.
        
        Args:
            interaction_type: Interaction type to filter by
        
        Returns:
            List of matching captures
        """
        return [
            c for c in self.captures
            if c.interaction_type == interaction_type
        ]
    
    def get_captures_by_element(
        self,
        element_selector: str,
    ) -> List[StateCapture]:
        """Get all captures for a specific element.
        
        Args:
            element_selector: Element selector to filter by
        
        Returns:
            List of matching captures
        """
        return [
            c for c in self.captures
            if c.element_selector == element_selector
        ]
    
    def export_manifest(
        self,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export a manifest of all captures.
        
        Args:
            output_path: Path to save manifest JSON
        
        Returns:
            Manifest dict
        """
        manifest = {
            "created_at": datetime.utcnow().isoformat(),
            "capture_count": len(self.captures),
            "figma_states_loaded": len(self.figma_states),
            "captures": [c.to_dict() for c in self.captures],
            "interactions_summary": self._summarize_interactions(),
        }
        
        if output_path:
            with open(output_path, "w") as f:
                json.dump(manifest, f, indent=2)
        
        return manifest
    
    def _summarize_interactions(self) -> Dict[str, int]:
        """Summarize captures by interaction type."""
        summary = {}
        for capture in self.captures:
            interaction = capture.interaction_type
            summary[interaction] = summary.get(interaction, 0) + 1
        return summary
    
    def clear_captures(self) -> None:
        """Clear all stored captures."""
        self.captures = []
    
    def archive_for_calibration(
        self,
        capture: StateCapture,
        ground_truth: str,
        reasoning: str,
        archive_dir: str,
    ) -> str:
        """Archive a capture for the calibration library.
        
        This creates a permanent record that can be used for
        few-shot examples in future LLM prompts.
        
        Args:
            capture: Capture to archive
            ground_truth: "confirmed" or "rejected"
            reasoning: Human explanation of the verdict
            archive_dir: Directory to store archives
        
        Returns:
            Archive file path
        """
        archive_path = Path(archive_dir)
        archive_path.mkdir(parents=True, exist_ok=True)
        
        archive_id = f"cal_{uuid.uuid4().hex[:12]}"
        archive_subdir = archive_path / archive_id
        archive_subdir.mkdir()
        
        if capture.before_crop:
            capture.before_crop.save(archive_subdir / "before.png")
        if capture.after_crop:
            capture.after_crop.save(archive_subdir / "after.png")
        if capture.figma_crop:
            capture.figma_crop.save(archive_subdir / "figma.png")
        
        metadata = {
            "id": archive_id,
            "original_capture_id": capture.id,
            "element_name": capture.element_name,
            "interaction_type": capture.interaction_type,
            "ground_truth": ground_truth,
            "reasoning": reasoning,
            "archived_at": datetime.utcnow().isoformat(),
            "page_url": capture.page_url,
            "element_selector": capture.element_selector,
        }
        
        with open(archive_subdir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Archived capture {capture.id} as {archive_id}")
        return str(archive_subdir)


def convert_interaction_results_to_captures(
    results: List[Dict[str, Any]],
    figma_states: Optional[Dict[str, Image.Image]] = None,
    page_url: str = "",
) -> List[StateCapture]:
    """Convert InteractionRunner results to StateCaptures.
    
    Convenience function for integrating InteractionRunner with StateCapturer.
    
    Args:
        results: List of InteractionResult.to_dict() outputs
        figma_states: Optional Figma state images
        page_url: URL of the page
    
    Returns:
        List of StateCapture objects
    """
    capturer = StateCapturer(figma_states=figma_states)
    
    captures = []
    for result in results:
        if result.get("success", True):
            capture = capturer.create_capture_from_result(result, page_url)
            captures.append(capture)
    
    return captures


async def capture_and_prepare_for_llm(
    page: Any,
    elements: List[Dict],
    figma_states: Optional[Dict[str, Image.Image]] = None,
    max_width: int = 1024,
) -> List[Dict[str, Any]]:
    """End-to-end function: capture interactions and prepare for LLM.
    
    Args:
        page: Playwright Page object
        elements: List of elements to test
        figma_states: Optional Figma state images
        max_width: Maximum image width for LLM
    
    Returns:
        List of prepared dicts ready for LLM analysis
    """
    from .interaction_runner import InteractionRunner
    
    runner = InteractionRunner(page)
    results = await runner.capture_all_states(elements)
    
    capturer = StateCapturer(figma_states=figma_states)
    
    for result in results:
        if isinstance(result, dict):
            capturer.create_capture_from_result(result, str(page.url))
        else:
            capturer.create_capture_from_result(result.to_dict(), str(page.url))
    
    return capturer.prepare_batch_for_llm(max_width=max_width)
