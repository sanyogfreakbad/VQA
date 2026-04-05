"""VQA Pipeline Module - Multi-stage visual comparison pipeline.

This module provides the complete VQA pipeline for comparing Figma designs
to web implementations:

- orchestrator: Main pipeline coordinator (entry point)
- region_segmenter: Page segmentation from DOM structure
- pre_filter: SSIM/pixel diff based triage (CLEAN/SUSPECT/DIRTY)
- merge_dedup: Finding merge with IoU-based deduplication
- prioritizer: Saliency-weighted ranking and severity classification
- report_generator: Final structured output (JSON, Markdown, HTML)

Usage:
    from vqa.pipeline import run_vqa_pipeline, PipelineConfig
    
    report = await run_vqa_pipeline(
        figma_data=figma_json,
        web_data=web_json,
        figma_screenshot=figma_img,
        web_screenshot=web_img,
        page_url="https://example.com",
        gemini_api_key="your-key",
    )
"""

from .orchestrator import (
    run_vqa_pipeline,
    run_vqa_simple,
    run_vqa_sync,
    PipelineConfig,
    PipelineMetadata,
)

from .region_segmenter import (
    segment_from_dom,
    segment_from_pixel_diff,
    merge_overlapping_regions,
    get_regions_for_triage,
)

from .pre_filter import (
    triage_regions,
    run_pre_filter,
    filter_dom_diffs_by_triage,
    compute_ssim_scores_for_regions,
    compute_pixel_scores_for_regions,
    TriageResult,
)

from .merge_dedup import (
    merge_findings,
    filter_by_confidence,
    get_findings_for_refinement,
    MergeResult,
)

from .prioritizer import (
    prioritize_findings,
    get_summary_stats,
    compute_quality_score,
    PrioritizationResult,
)

from .report_generator import (
    generate_report,
    generate_markdown_report,
    generate_html_report,
    save_report,
    VQAReport,
)


__all__ = [
    # Orchestrator (main entry point)
    "run_vqa_pipeline",
    "run_vqa_simple",
    "run_vqa_sync",
    "PipelineConfig",
    "PipelineMetadata",
    
    # Region segmenter
    "segment_from_dom",
    "segment_from_pixel_diff",
    "merge_overlapping_regions",
    "get_regions_for_triage",
    
    # Pre-filter
    "triage_regions",
    "run_pre_filter",
    "filter_dom_diffs_by_triage",
    "compute_ssim_scores_for_regions",
    "compute_pixel_scores_for_regions",
    "TriageResult",
    
    # Merge/dedup
    "merge_findings",
    "filter_by_confidence",
    "get_findings_for_refinement",
    "MergeResult",
    
    # Prioritizer
    "prioritize_findings",
    "get_summary_stats",
    "compute_quality_score",
    "PrioritizationResult",
    
    # Report generator
    "generate_report",
    "generate_markdown_report",
    "generate_html_report",
    "save_report",
    "VQAReport",
]
