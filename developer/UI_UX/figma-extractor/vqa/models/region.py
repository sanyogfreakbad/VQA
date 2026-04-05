"""Region and BoundingBox models for Visual QA pipeline."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from .enums import RegionStatus


@dataclass
class BoundingBox:
    """Bounding box coordinates."""
    x: float
    y: float
    width: float
    height: float
    
    @property
    def x2(self) -> float:
        """Right edge x coordinate."""
        return self.x + self.width
    
    @property
    def y2(self) -> float:
        """Bottom edge y coordinate."""
        return self.y + self.height
    
    @property
    def center_x(self) -> float:
        """Center x coordinate."""
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        """Center y coordinate."""
        return self.y + self.height / 2
    
    @property
    def area(self) -> float:
        """Area of the bounding box."""
        return self.width * self.height
    
    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union with another box."""
        # Calculate intersection
        x_left = max(self.x, other.x)
        y_top = max(self.y, other.y)
        x_right = min(self.x2, other.x2)
        y_bottom = min(self.y2, other.y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = self.area + other.area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def contains(self, other: "BoundingBox") -> bool:
        """Check if this box fully contains another box."""
        return (
            self.x <= other.x and
            self.y <= other.y and
            self.x2 >= other.x2 and
            self.y2 >= other.y2
        )
    
    def overlaps(self, other: "BoundingBox") -> bool:
        """Check if this box overlaps with another box."""
        return self.iou(other) > 0
    
    def expand(self, padding: float) -> "BoundingBox":
        """Return a new box expanded by padding on all sides."""
        return BoundingBox(
            x=self.x - padding,
            y=self.y - padding,
            width=self.width + 2 * padding,
            height=self.height + 2 * padding,
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "BoundingBox":
        """Create from dictionary."""
        return cls(
            x=d.get("x", 0),
            y=d.get("y", 0),
            width=d.get("width", 0),
            height=d.get("height", 0),
        )


@dataclass
class Region:
    """
    A comparable region of the page.
    
    Regions are created by the RegionSegmenter and tagged by the PreFilter
    with a status (CLEAN/SUSPECT/DIRTY) based on SSIM and DOM diff.
    """
    id: str
    bbox: BoundingBox
    status: RegionStatus = RegionStatus.SUSPECT
    
    # Pre-filter scores
    ssim_score: Optional[float] = None
    pixel_diff_count: Optional[int] = None
    dom_diff_count: Optional[int] = None
    
    # Metadata
    element_name: Optional[str] = None
    element_type: Optional[str] = None
    figma_node_id: Optional[str] = None
    web_node_id: Optional[str] = None
    
    # Saliency factors
    is_above_fold: bool = True
    is_interactive: bool = False
    visual_weight: float = 1.0  # Based on size and position
    
    # Associated data
    figma_element: Optional[Dict[str, Any]] = None
    web_element: Optional[Dict[str, Any]] = None
    dom_diffs: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "bbox": self.bbox.to_dict(),
            "status": self.status.value,
            "ssim_score": self.ssim_score,
            "pixel_diff_count": self.pixel_diff_count,
            "dom_diff_count": self.dom_diff_count,
            "element_name": self.element_name,
            "element_type": self.element_type,
            "is_above_fold": self.is_above_fold,
            "is_interactive": self.is_interactive,
            "visual_weight": self.visual_weight,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Region":
        """Create from dictionary."""
        return cls(
            id=d["id"],
            bbox=BoundingBox.from_dict(d["bbox"]),
            status=RegionStatus(d.get("status", "suspect")),
            ssim_score=d.get("ssim_score"),
            pixel_diff_count=d.get("pixel_diff_count"),
            dom_diff_count=d.get("dom_diff_count"),
            element_name=d.get("element_name"),
            element_type=d.get("element_type"),
            is_above_fold=d.get("is_above_fold", True),
            is_interactive=d.get("is_interactive", False),
            visual_weight=d.get("visual_weight", 1.0),
        )
