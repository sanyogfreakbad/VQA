"""VQA Models - Data structures for Visual QA pipeline."""

from .enums import (
    Severity,
    Category,
    Confidence,
    RegionStatus,
    DiffType,
)
from .region import Region, BoundingBox
from .finding import Finding, DOMEvidence

__all__ = [
    "Severity",
    "Category",
    "Confidence",
    "RegionStatus",
    "DiffType",
    "Region",
    "BoundingBox",
    "Finding",
    "DOMEvidence",
]
