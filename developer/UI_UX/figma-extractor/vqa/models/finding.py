"""Finding model for Visual QA pipeline."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

from .enums import Severity, Category, Confidence, DiffType
from .region import BoundingBox


@dataclass
class DOMEvidence:
    """DOM-based evidence for a finding."""
    figma_value: Any
    web_value: Any
    delta: Any
    figma_node_id: Optional[str] = None
    web_node_id: Optional[str] = None
    web_locator: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "figma_value": self.figma_value,
            "web_value": self.web_value,
            "delta": self.delta,
            "figma_node_id": self.figma_node_id,
            "web_node_id": self.web_node_id,
            "web_locator": self.web_locator,
        }


@dataclass
class Finding:
    """
    Unified finding data model.
    
    Represents a single visual difference found between Figma and Web,
    regardless of whether it was detected by DOM comparison, pixel diff,
    or LLM visual analysis.
    """
    id: str
    category: Category
    diff_type: DiffType
    severity: Severity
    confidence: Confidence
    
    # Element info
    element_name: str
    element_text: Optional[str] = None
    element_type: Optional[str] = None
    
    # Position info
    figma_bbox: Optional[BoundingBox] = None
    web_bbox: Optional[BoundingBox] = None
    
    # Evidence
    dom_evidence: Optional[DOMEvidence] = None
    visual_reasoning: Optional[str] = None
    
    # Source tracking
    source: str = "dom"  # "dom", "pixel_diff", "gemini_visual", "behavioral"
    pass_name: Optional[str] = None  # "pass_a", "pass_b", "pass_c", "refinement"
    
    # Saliency
    is_above_fold: bool = True
    is_interactive: bool = False
    saliency_score: float = 1.0
    
    # Deduplication
    merged_from: List[str] = field(default_factory=list)
    
    # Serial number (for UI display)
    serial_number: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "category": self.category.value,
            "diff_type": self.diff_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "element_name": self.element_name,
            "element_text": self.element_text,
            "element_type": self.element_type,
            "source": self.source,
            "is_above_fold": self.is_above_fold,
            "is_interactive": self.is_interactive,
            "saliency_score": self.saliency_score,
        }
        
        if self.figma_bbox:
            result["figma_position"] = self.figma_bbox.to_dict()
        if self.web_bbox:
            result["web_position"] = self.web_bbox.to_dict()
        if self.dom_evidence:
            result["dom_evidence"] = self.dom_evidence.to_dict()
        if self.visual_reasoning:
            result["visual_reasoning"] = self.visual_reasoning
        if self.pass_name:
            result["pass_name"] = self.pass_name
        if self.serial_number is not None:
            result["serial_number"] = self.serial_number
        if self.merged_from:
            result["merged_from"] = self.merged_from
            
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Finding":
        """Create from dictionary."""
        figma_bbox = None
        if d.get("figma_position"):
            figma_bbox = BoundingBox.from_dict(d["figma_position"])
        
        web_bbox = None
        if d.get("web_position"):
            web_bbox = BoundingBox.from_dict(d["web_position"])
        
        dom_evidence = None
        if d.get("dom_evidence"):
            dom_evidence = DOMEvidence(**d["dom_evidence"])
        
        return cls(
            id=d["id"],
            category=Category(d["category"]),
            diff_type=DiffType(d["diff_type"]),
            severity=Severity(d["severity"]),
            confidence=Confidence(d.get("confidence", "medium")),
            element_name=d["element_name"],
            element_text=d.get("element_text"),
            element_type=d.get("element_type"),
            figma_bbox=figma_bbox,
            web_bbox=web_bbox,
            dom_evidence=dom_evidence,
            visual_reasoning=d.get("visual_reasoning"),
            source=d.get("source", "dom"),
            pass_name=d.get("pass_name"),
            is_above_fold=d.get("is_above_fold", True),
            is_interactive=d.get("is_interactive", False),
            saliency_score=d.get("saliency_score", 1.0),
            merged_from=d.get("merged_from", []),
            serial_number=d.get("serial_number"),
        )
    
    def get_primary_bbox(self) -> Optional[BoundingBox]:
        """Get the most relevant bounding box for this finding."""
        # For missing elements, use Figma position
        if self.diff_type == DiffType.ELEMENT_MISSING_IN_WEB:
            return self.figma_bbox
        # For everything else, prefer web position
        return self.web_bbox or self.figma_bbox
