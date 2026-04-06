"""
Pipeline Orchestrator - Main coordinator for the VQA pipeline.

This is the entry point for running the complete Visual QA pipeline.
It runs all stages in the correct order, manages parallelism, and
produces the final report.

Pipeline Stages:
1. Algorithmic pre-filters (zero LLM cost)
2. Parallel LLM analysis (Pass A, B, C)
3. Merge + deduplicate findings
4. Selective refinement (uncertain items only)
5. Prioritization + report generation

Usage:
    result = await run_vqa_pipeline(
        figma_data=figma_json,
        web_data=web_json,
        figma_screenshot=figma_img,
        web_screenshot=web_img,
        page_url="https://app.example.com/dashboard",
    )
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from PIL import Image

from .region_segmenter import get_regions_for_triage, segment_from_dom
from .pre_filter import run_pre_filter, TriageResult
from .merge_dedup import merge_findings, MergeResult, get_findings_for_refinement
from .prioritizer import prioritize_findings, PrioritizationResult
from .report_generator import generate_report, VQAReport

from ..vision import (
    compute_pixel_diff,
    compute_ssim,
    compute_ssim_for_regions,
    crop_regions,
    PixelDiffResult,
    SSIMResult,
    LocalDiffEngine,
    get_local_diff_engine,
    DiffCategory,
)

from ..llm import (
    GeminiClient,
    run_blind_visual_pass,
    run_targeted_validation_pass,
    run_behavioral_pass,
    run_refinement_pass,
    CalibrationStore,
)

from ..models.finding import Finding
from ..models.region import Region
from ..models.enums import Confidence
from ..config.thresholds import Thresholds, THRESHOLDS
from ..config.llm_config import LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the VQA pipeline.
    
    Pass A (blind visual analysis) is disabled by default and only runs as a
    fallback when the overall SSIM score is very low (below pass_a_fallback_threshold).
    This significantly reduces latency and LLM costs for most comparisons.
    
    Set enable_pass_a=True to always run Pass A regardless of SSIM score.
    """
    thresholds: Thresholds = field(default_factory=Thresholds)
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    
    # Pass A is now disabled by default - runs only as fallback for very low SSIM
    enable_pass_a: bool = False
    enable_pass_a_fallback: bool = True  # Run Pass A if SSIM < threshold
    enable_pass_b: bool = True
    enable_pass_c: bool = False
    enable_refinement: bool = True
    
    # Local diff engine (Phase 3) - reduces LLM dependency
    enable_local_diff: bool = True
    
    max_refinement_items: int = 20
    min_confidence_for_report: Confidence = Confidence.LOW
    max_findings_in_report: int = 100
    
    enable_pixel_diff: bool = True
    enable_ssim: bool = True
    
    save_intermediate_results: bool = False
    output_dir: Optional[str] = None


@dataclass
class PipelineMetadata:
    """Metadata about pipeline execution."""
    total_time_ms: float = 0
    stage_times_ms: Dict[str, float] = field(default_factory=dict)
    
    total_regions: int = 0
    regions_skipped: int = 0
    regions_analyzed: int = 0
    
    # Pass execution tracking
    pass_a_enabled: bool = False
    pass_a_fallback_triggered: bool = False
    pass_a_findings: int = 0
    pass_b_findings: int = 0
    pass_c_findings: int = 0
    
    # Local diff tracking (Phase 3)
    local_diff_enabled: bool = False
    local_diff_findings: int = 0
    llm_calls_saved: int = 0
    
    refinement_count: int = 0
    final_findings: int = 0
    
    overall_ssim: Optional[float] = None
    
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_time_ms": round(self.total_time_ms, 2),
            "stage_times_ms": {k: round(v, 2) for k, v in self.stage_times_ms.items()},
            "total_regions": self.total_regions,
            "regions_skipped": self.regions_skipped,
            "regions_analyzed": self.regions_analyzed,
            "pass_a_enabled": self.pass_a_enabled,
            "pass_a_fallback_triggered": self.pass_a_fallback_triggered,
            "pass_a_findings": self.pass_a_findings,
            "pass_b_findings": self.pass_b_findings,
            "pass_c_findings": self.pass_c_findings,
            "local_diff_enabled": self.local_diff_enabled,
            "local_diff_findings": self.local_diff_findings,
            "llm_calls_saved": self.llm_calls_saved,
            "refinement_count": self.refinement_count,
            "final_findings": self.final_findings,
            "overall_ssim": round(self.overall_ssim, 4) if self.overall_ssim else None,
            "errors": self.errors,
        }


async def run_vqa_pipeline(
    figma_data: Dict[str, Any],
    web_data: Dict[str, Any],
    figma_screenshot: Image.Image,
    web_screenshot: Image.Image,
    page_url: str = "",
    gemini_api_key: Optional[str] = None,
    playwright_page: Any = None,
    config: Optional[PipelineConfig] = None,
    dom_diffs: Optional[List[Dict[str, Any]]] = None,
) -> VQAReport:
    """Run the complete VQA pipeline.
    
    Args:
        figma_data: Figma design data (nodes, styles, etc.)
        web_data: Web DOM data (nodes, computed styles, etc.)
        figma_screenshot: Full Figma design screenshot
        web_screenshot: Full web implementation screenshot
        page_url: URL of the web page being compared
        gemini_api_key: Gemini API key (or use env var)
        playwright_page: Optional Playwright page for behavioral testing
        config: Pipeline configuration
        dom_diffs: Pre-computed DOM comparison differences
    
    Returns:
        VQAReport with all findings and metadata
    """
    config = config or PipelineConfig()
    metadata = PipelineMetadata()
    
    pipeline_start = time.time()
    
    page_width = figma_screenshot.width
    page_height = figma_screenshot.height
    
    if web_screenshot.size != figma_screenshot.size:
        web_screenshot = web_screenshot.resize(figma_screenshot.size, Image.Resampling.LANCZOS)
    
    gemini_client = None
    if config.enable_pass_a or config.enable_pass_b or config.enable_refinement:
        gemini_client = GeminiClient(api_key=gemini_api_key, config=config.llm_config)
    
    calibration_store = CalibrationStore()
    
    logger.info("=" * 60)
    logger.info("STAGE 1: ALGORITHMIC PRE-FILTERS")
    logger.info("=" * 60)
    
    stage1_start = time.time()
    
    pixel_result = None
    if config.enable_pixel_diff:
        pixel_result = compute_pixel_diff(figma_screenshot, web_screenshot)
        logger.info(f"Pixel diff: {pixel_result.mismatch_pct * 100:.2f}% mismatch")
    
    ssim_result = None
    if config.enable_ssim:
        overall_ssim, ssim_map = compute_ssim(figma_screenshot, web_screenshot)
        ssim_result = SSIMResult(
            overall_ssim=overall_ssim,
            ssim_map=ssim_map,
            region_scores=[],
            clean_region_count=0,
            dirty_region_count=0,
        )
        logger.info(f"SSIM: {overall_ssim:.4f}")
    
    figma_nodes = figma_data.get("nodes", figma_data.get("children", []))
    web_nodes = web_data.get("nodes", web_data.get("elements", []))
    
    pixel_regions = pixel_result.region_scores if pixel_result else None
    pixel_regions_dicts = [
        {"x": r.x, "y": r.y, "width": r.width, "height": r.height, "mismatch_count": r.mismatch_count}
        for r in (pixel_regions or [])
    ]
    
    regions = get_regions_for_triage(
        figma_nodes, web_nodes, pixel_regions_dicts, page_width, page_height
    )
    
    metadata.total_regions = len(regions)
    
    triage_result = run_pre_filter(
        regions=regions,
        ssim_result=ssim_result,
        pixel_result=pixel_result,
        dom_diffs=dom_diffs or [],
        thresholds=config.thresholds,
    )
    
    metadata.regions_skipped = triage_result.skipped_count
    metadata.regions_analyzed = len(triage_result.regions_to_analyze)
    
    metadata.stage_times_ms["stage1_prefilter"] = (time.time() - stage1_start) * 1000
    
    logger.info(f"Triage: {len(triage_result.clean_regions)} clean, "
                f"{len(triage_result.suspect_regions)} suspect, "
                f"{len(triage_result.dirty_regions)} dirty")
    
    # =========================================================================
    # STAGE 1.5: LOCAL DIFF ENGINE (optional - reduces LLM calls)
    # =========================================================================
    
    local_diff_findings: List[Finding] = []
    regions_for_llm = triage_result.regions_to_analyze
    
    if config.enable_local_diff and regions_for_llm:
        logger.info("=" * 60)
        logger.info("STAGE 1.5: LOCAL DIFF CLASSIFICATION")
        logger.info("=" * 60)
        
        stage15_start = time.time()
        metadata.local_diff_enabled = True
        
        try:
            local_diff_findings, regions_for_llm = await _run_local_diff(
                figma_screenshot=figma_screenshot,
                web_screenshot=web_screenshot,
                regions=regions_for_llm,
                confidence_threshold=config.thresholds.local_diff_confidence_threshold,
            )
            
            metadata.local_diff_findings = len(local_diff_findings)
            metadata.llm_calls_saved = len(triage_result.regions_to_analyze) - len(regions_for_llm)
            
            logger.info(f"Local diff: {len(local_diff_findings)} high-confidence classifications, "
                       f"{len(regions_for_llm)} regions need LLM validation")
            
        except Exception as e:
            logger.warning(f"Local diff failed, falling back to full LLM analysis: {e}")
            metadata.errors.append(f"local_diff: {str(e)}")
        
        metadata.stage_times_ms["stage15_local_diff"] = (time.time() - stage15_start) * 1000
    
    logger.info("=" * 60)
    logger.info("STAGE 2: PARALLEL LLM ANALYSIS")
    logger.info("=" * 60)
    
    stage2_start = time.time()
    
    pass_a_findings: List[Finding] = []
    pass_b_findings: List[Finding] = []
    pass_c_findings: List[Finding] = []
    
    tasks = []
    
    # Track SSIM in metadata
    if ssim_result:
        metadata.overall_ssim = ssim_result.overall_ssim
    
    # Determine if Pass A should run:
    # - Always run if enable_pass_a is True
    # - Run as fallback if enable_pass_a_fallback is True AND SSIM is very low
    should_run_pass_a = config.enable_pass_a
    metadata.pass_a_enabled = config.enable_pass_a
    
    if not should_run_pass_a and config.enable_pass_a_fallback and ssim_result:
        if ssim_result.overall_ssim < config.thresholds.pass_a_fallback_threshold:
            should_run_pass_a = True
            metadata.pass_a_fallback_triggered = True
            logger.info(f"Pass A fallback triggered: SSIM {ssim_result.overall_ssim:.4f} < "
                       f"threshold {config.thresholds.pass_a_fallback_threshold}")
    
    if should_run_pass_a and gemini_client:
        logger.info("Starting Pass A: Blind visual analysis...")
        tasks.append(("pass_a", _run_pass_a(
            gemini_client, figma_screenshot, web_screenshot, page_width, page_height
        )))
    
    if config.enable_pass_b and gemini_client and regions_for_llm:
        logger.info(f"Starting Pass B: Targeted validation ({len(regions_for_llm)} regions, "
                   f"batch_size={config.thresholds.pass_b_batch_size})...")
        tasks.append(("pass_b", _run_pass_b(
            gemini_client, figma_screenshot, web_screenshot,
            regions_for_llm, dom_diffs or [], calibration_store,
            batch_size=config.thresholds.pass_b_batch_size
        )))
    
    if config.enable_pass_c and gemini_client and playwright_page:
        logger.info("Starting Pass C: Behavioral testing...")
        tasks.append(("pass_c", _run_pass_c(
            gemini_client, playwright_page, figma_data
        )))
    
    if tasks:
        task_objs = [asyncio.create_task(t[1]) for t in tasks]
        results = await asyncio.gather(*task_objs, return_exceptions=True)
        
        for (name, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f"{name} failed: {result}")
                metadata.errors.append(f"{name}: {str(result)}")
            elif name == "pass_a":
                pass_a_findings = result
            elif name == "pass_b":
                pass_b_findings = result
            elif name == "pass_c":
                pass_c_findings = result
    
    metadata.pass_a_findings = len(pass_a_findings)
    metadata.pass_b_findings = len(pass_b_findings)
    metadata.pass_c_findings = len(pass_c_findings)
    
    metadata.stage_times_ms["stage2_llm_analysis"] = (time.time() - stage2_start) * 1000
    
    logger.info(f"Pass A: {len(pass_a_findings)}, Pass B: {len(pass_b_findings)}, Pass C: {len(pass_c_findings)}")
    if local_diff_findings:
        logger.info(f"Local Diff: {len(local_diff_findings)} (high-confidence, no LLM)")
    
    logger.info("=" * 60)
    logger.info("STAGE 3: MERGE + DEDUPLICATE")
    logger.info("=" * 60)
    
    stage3_start = time.time()
    
    # Combine local diff findings with LLM findings
    all_pass_b_findings = pass_b_findings + local_diff_findings
    
    merge_result = merge_findings(
        pass_a_findings=pass_a_findings,
        pass_b_findings=all_pass_b_findings,
        pass_c_findings=pass_c_findings,
        dom_diffs=dom_diffs,
        iou_threshold=config.thresholds.iou_merge_threshold,
    )
    
    metadata.stage_times_ms["stage3_merge_dedup"] = (time.time() - stage3_start) * 1000
    
    logger.info(f"Merged: {merge_result.total_input} -> {len(merge_result.merged_findings)} "
                f"(removed {merge_result.duplicates_removed} duplicates)")
    
    merged_findings = merge_result.merged_findings
    
    if config.enable_refinement and gemini_client:
        logger.info("=" * 60)
        logger.info("STAGE 4: SELECTIVE REFINEMENT")
        logger.info("=" * 60)
        
        stage4_start = time.time()
        
        needs_refinement, already_confident = get_findings_for_refinement(
            merged_findings,
            confidence_threshold=Confidence.MEDIUM,
            max_refinement_count=config.max_refinement_items,
        )
        
        if needs_refinement:
            logger.info(f"Refining {len(needs_refinement)} uncertain findings...")
            
            try:
                refined = await run_refinement_pass(
                    client=gemini_client,
                    findings=needs_refinement,
                    figma_screenshot=figma_screenshot,
                    web_screenshot=web_screenshot,
                    calibration_store=calibration_store,
                )
                
                merged_findings = already_confident + refined
                metadata.refinement_count = len(needs_refinement)
                
            except Exception as e:
                logger.error(f"Refinement pass failed: {e}")
                metadata.errors.append(f"refinement: {str(e)}")
                merged_findings = already_confident + needs_refinement
        
        metadata.stage_times_ms["stage4_refinement"] = (time.time() - stage4_start) * 1000
    
    logger.info("=" * 60)
    logger.info("STAGE 5: PRIORITIZATION + REPORT")
    logger.info("=" * 60)
    
    stage5_start = time.time()
    
    prioritization_result = prioritize_findings(
        findings=merged_findings,
        min_confidence=config.min_confidence_for_report,
        max_findings=config.max_findings_in_report,
        thresholds=config.thresholds,
    )
    
    metadata.final_findings = len(prioritization_result.prioritized_findings)
    
    total_compared = max(len(regions), metadata.total_regions, 50)
    
    report = generate_report(
        prioritization_result=prioritization_result,
        page_url=page_url,
        figma_file_id=figma_data.get("fileId"),
        figma_node_id=figma_data.get("nodeId"),
        pipeline_metadata=metadata.to_dict(),
        total_compared_elements=total_compared,
    )
    
    metadata.stage_times_ms["stage5_report"] = (time.time() - stage5_start) * 1000
    
    metadata.total_time_ms = (time.time() - pipeline_start) * 1000
    
    report.pipeline_metadata = metadata.to_dict()
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time: {metadata.total_time_ms:.0f}ms")
    logger.info(f"Final findings: {metadata.final_findings}")
    logger.info(f"Quality score: {report.quality_score.get('score', 'N/A')}")
    logger.info("=" * 60)
    
    return report


async def _run_pass_a(
    client: GeminiClient,
    figma_screenshot: Image.Image,
    web_screenshot: Image.Image,
    page_width: int,
    page_height: int,
) -> List[Finding]:
    """Run Pass A: Blind visual analysis."""
    try:
        findings = await run_blind_visual_pass(
            client=client,
            figma_screenshot=figma_screenshot,
            web_screenshot=web_screenshot,
            page_width=page_width,
            page_height=page_height,
        )
        return findings
    except Exception as e:
        logger.error(f"Pass A error: {e}")
        return []


async def _run_pass_b(
    client: GeminiClient,
    figma_screenshot: Image.Image,
    web_screenshot: Image.Image,
    regions: List[Region],
    dom_diffs: List[Dict[str, Any]],
    calibration_store: CalibrationStore,
    batch_size: int = 10,
) -> List[Finding]:
    """Run Pass B: Targeted validation."""
    try:
        crop_pairs = crop_regions(
            figma_img=figma_screenshot,
            web_img=web_screenshot,
            regions=regions,
        )
        
        if not crop_pairs:
            return []
        
        findings = await run_targeted_validation_pass(
            client=client,
            crop_pairs=crop_pairs,
            dom_diffs=dom_diffs,
            calibration_store=calibration_store,
            batch_size=batch_size,
        )
        return findings
    except Exception as e:
        logger.error(f"Pass B error: {e}")
        return []


async def _run_pass_c(
    client: GeminiClient,
    playwright_page: Any,
    figma_data: Dict[str, Any],
) -> List[Finding]:
    """Run Pass C: Behavioral testing."""
    try:
        findings = await run_behavioral_pass(
            client=client,
            playwright_page=playwright_page,
            figma_data=figma_data,
        )
        return findings
    except Exception as e:
        logger.error(f"Pass C error: {e}")
        return []


async def _run_local_diff(
    figma_screenshot: Image.Image,
    web_screenshot: Image.Image,
    regions: List[Region],
    confidence_threshold: float = 0.8,
) -> tuple[List[Finding], List[Region]]:
    """Run local diff classification on regions.
    
    Uses CLIP + perceptual hashing to classify visual differences locally.
    High-confidence classifications are returned as findings.
    Low-confidence regions are returned for LLM validation.
    
    Args:
        figma_screenshot: Full Figma screenshot
        web_screenshot: Full web screenshot
        regions: Regions to analyze
        confidence_threshold: Minimum confidence to skip LLM
    
    Returns:
        Tuple of (high_confidence_findings, regions_needing_llm)
    """
    import uuid
    from ..models.enums import Category, Severity, Confidence as ConfidenceEnum
    from ..models.region import BoundingBox
    
    engine = get_local_diff_engine(load_clip=True)
    
    high_confidence_findings = []
    regions_for_llm = []
    
    # Crop and classify each region
    for region in regions:
        try:
            # Crop the region from both screenshots
            box = (region.x, region.y, region.x + region.width, region.y + region.height)
            
            # Ensure box is within bounds
            box = (
                max(0, box[0]),
                max(0, box[1]),
                min(figma_screenshot.width, box[2]),
                min(web_screenshot.width, box[2]),
            )
            box = (
                box[0],
                box[1],
                box[2],
                min(figma_screenshot.height, min(web_screenshot.height, region.y + region.height)),
            )
            
            if box[2] <= box[0] or box[3] <= box[1]:
                regions_for_llm.append(region)
                continue
            
            figma_crop = figma_screenshot.crop(box)
            web_crop = web_screenshot.crop(box)
            
            # Classify the difference
            classification = engine.classify_diff(figma_crop, web_crop)
            
            if classification.category == DiffCategory.IDENTICAL:
                # Skip identical regions
                continue
            
            if classification.is_high_confidence and classification.confidence >= confidence_threshold:
                # Create a Finding from the local classification
                finding = Finding(
                    id=f"local_{uuid.uuid4().hex[:8]}",
                    category=_map_diff_category_to_category(classification.category),
                    severity=Severity.MEDIUM,
                    confidence=ConfidenceEnum.HIGH if classification.confidence > 0.9 else ConfidenceEnum.MEDIUM,
                    description=classification.description,
                    bounding_box=BoundingBox(
                        x=region.x,
                        y=region.y,
                        width=region.width,
                        height=region.height,
                    ),
                    source="local_diff",
                    pass_name="local_diff",
                )
                high_confidence_findings.append(finding)
            else:
                # Low confidence - needs LLM validation
                regions_for_llm.append(region)
                
        except Exception as e:
            logger.warning(f"Local diff failed for region: {e}")
            regions_for_llm.append(region)
    
    return high_confidence_findings, regions_for_llm


def _map_diff_category_to_category(diff_cat: DiffCategory) -> 'Category':
    """Map LocalDiff category to Finding category."""
    from ..models.enums import Category
    
    mapping = {
        DiffCategory.TEXT: Category.TEXT,
        DiffCategory.COLOR: Category.COLOR,
        DiffCategory.SPACING: Category.SPACING,
        DiffCategory.SHADOW: Category.SHADOW,
        DiffCategory.BORDER: Category.BORDER,
        DiffCategory.SIZE: Category.SIZE,
        DiffCategory.MISSING: Category.MISSING,
        DiffCategory.UNKNOWN: Category.OTHER,
    }
    return mapping.get(diff_cat, Category.OTHER)


async def run_vqa_simple(
    figma_screenshot: Image.Image,
    web_screenshot: Image.Image,
    gemini_api_key: Optional[str] = None,
    page_url: str = "",
) -> VQAReport:
    """Simplified VQA pipeline for quick analysis.
    
    Runs only Pass A (blind visual analysis) without DOM comparison.
    Useful for quick checks or when DOM data is not available.
    
    Args:
        figma_screenshot: Figma design screenshot
        web_screenshot: Web implementation screenshot
        gemini_api_key: Gemini API key
        page_url: Optional page URL
    
    Returns:
        VQAReport with findings
    """
    config = PipelineConfig(
        enable_pass_a=True,
        enable_pass_b=False,
        enable_pass_c=False,
        enable_refinement=False,
    )
    
    return await run_vqa_pipeline(
        figma_data={"nodes": []},
        web_data={"nodes": []},
        figma_screenshot=figma_screenshot,
        web_screenshot=web_screenshot,
        page_url=page_url,
        gemini_api_key=gemini_api_key,
        config=config,
    )


def run_vqa_sync(
    figma_data: Dict[str, Any],
    web_data: Dict[str, Any],
    figma_screenshot: Image.Image,
    web_screenshot: Image.Image,
    **kwargs,
) -> VQAReport:
    """Synchronous wrapper for run_vqa_pipeline.
    
    Useful for non-async contexts.
    """
    return asyncio.run(run_vqa_pipeline(
        figma_data=figma_data,
        web_data=web_data,
        figma_screenshot=figma_screenshot,
        web_screenshot=web_screenshot,
        **kwargs,
    ))
