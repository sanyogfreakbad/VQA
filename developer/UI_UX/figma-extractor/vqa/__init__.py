"""
Visual QA (VQA) Pipeline

A multi-stage pipeline for comparing Figma designs to web implementations.
Uses algorithmic pre-filters first, then LLM visual analysis only where needed.

Architecture:
- Stage 1: Pre-filters (pixel diff, SSIM, DOM comparison)
- Stage 2: Parallel LLM analysis (blind visual, targeted, behavioral)
- Stage 3: Merge and deduplicate findings
- Stage 4: Selective refinement of uncertain findings
- Stage 5: Prioritization and report generation
"""

__version__ = "1.0.0"

from .models import (
    Severity,
    Category,
    Confidence,
    RegionStatus,
    DiffType,
    Region,
    BoundingBox,
    Finding,
    DOMEvidence,
)

from .config import (
    Thresholds,
    THRESHOLDS,
    LLMConfig,
    LLM_CONFIG,
)

from .llm import (
    GeminiClient,
    GeminiResponse,
    run_blind_visual_pass,
    run_targeted_validation_pass,
    run_behavioral_pass,
    run_refinement_pass,
    CalibrationStore,
    get_calibration_store,
)

from .pipeline import (
    run_vqa_pipeline,
    run_vqa_simple,
    run_vqa_sync,
    PipelineConfig,
    VQAReport,
    generate_report,
    generate_markdown_report,
    generate_html_report,
    save_report,
    TriageResult,
    MergeResult,
    PrioritizationResult,
)

__all__ = [
    # Version
    "__version__",
    
    # Models
    "Severity",
    "Category",
    "Confidence",
    "RegionStatus",
    "DiffType",
    "Region",
    "BoundingBox",
    "Finding",
    "DOMEvidence",
    
    # Config
    "Thresholds",
    "THRESHOLDS",
    "LLMConfig",
    "LLM_CONFIG",
    
    # LLM
    "GeminiClient",
    "GeminiResponse",
    "run_blind_visual_pass",
    "run_targeted_validation_pass",
    "run_behavioral_pass",
    "run_refinement_pass",
    "CalibrationStore",
    "get_calibration_store",
    
    # Pipeline
    "run_vqa_pipeline",
    "run_vqa_simple",
    "run_vqa_sync",
    "PipelineConfig",
    "VQAReport",
    "generate_report",
    "generate_markdown_report",
    "generate_html_report",
    "save_report",
    "TriageResult",
    "MergeResult",
    "PrioritizationResult",
]
