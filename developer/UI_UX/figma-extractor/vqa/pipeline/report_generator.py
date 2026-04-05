"""
Report Generator - Final structured output for the VQA pipeline.

Generates categorized, severity-ranked reports of all visual, structural,
and behavioral differences between Figma design and web implementation.

Output formats:
- JSON (for API consumption)
- Markdown (for human review)
- HTML (for rich display with screenshots)
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from ..models.finding import Finding
from ..models.enums import Category, Severity, Confidence
from .prioritizer import PrioritizationResult, get_summary_stats, compute_quality_score

logger = logging.getLogger(__name__)


@dataclass
class VQAReport:
    """Complete VQA pipeline report."""
    generated_at: str
    page_url: str
    figma_file_id: Optional[str]
    figma_node_id: Optional[str]
    
    summary: Dict[str, Any]
    quality_score: Dict[str, Any]
    
    findings: List[Dict[str, Any]]
    by_severity: Dict[str, List[Dict[str, Any]]]
    by_category: Dict[str, List[Dict[str, Any]]]
    
    pipeline_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "generated_at": self.generated_at,
            "page_url": self.page_url,
            "figma_file_id": self.figma_file_id,
            "figma_node_id": self.figma_node_id,
            "summary": self.summary,
            "quality_score": self.quality_score,
            "findings": self.findings,
            "by_severity": self.by_severity,
            "by_category": self.by_category,
            "pipeline_metadata": self.pipeline_metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


def generate_report(
    prioritization_result: PrioritizationResult,
    page_url: str = "",
    figma_file_id: Optional[str] = None,
    figma_node_id: Optional[str] = None,
    pipeline_metadata: Optional[Dict[str, Any]] = None,
    total_compared_elements: int = 100,
) -> VQAReport:
    """Generate the final VQA report.
    
    Args:
        prioritization_result: Result from prioritization stage
        page_url: URL of the compared web page
        figma_file_id: Figma file ID
        figma_node_id: Figma node ID
        pipeline_metadata: Pipeline execution metadata
        total_compared_elements: Total elements that were compared
    
    Returns:
        VQAReport object
    """
    findings = prioritization_result.prioritized_findings
    
    summary = get_summary_stats(findings)
    
    quality_score = compute_quality_score(findings, total_compared_elements)
    
    findings_dicts = [f.to_dict() for f in findings]
    
    by_severity = {}
    for sev_key, sev_findings in prioritization_result.by_severity.items():
        by_severity[sev_key] = [f.to_dict() for f in sev_findings]
    
    by_category = {}
    for cat_key, cat_findings in prioritization_result.by_category.items():
        by_category[cat_key] = [f.to_dict() for f in cat_findings]
    
    report = VQAReport(
        generated_at=datetime.utcnow().isoformat() + "Z",
        page_url=page_url,
        figma_file_id=figma_file_id,
        figma_node_id=figma_node_id,
        summary=summary,
        quality_score=quality_score,
        findings=findings_dicts,
        by_severity=by_severity,
        by_category=by_category,
        pipeline_metadata=pipeline_metadata or {},
    )
    
    logger.info(
        f"Generated report with {len(findings)} findings, "
        f"quality score: {quality_score['score']}"
    )
    
    return report


def generate_markdown_report(report: VQAReport) -> str:
    """Generate a Markdown-formatted report for human review.
    
    Args:
        report: VQAReport object
    
    Returns:
        Markdown string
    """
    lines = []
    
    lines.append("# Visual QA Report")
    lines.append("")
    lines.append(f"**Generated:** {report.generated_at}")
    if report.page_url:
        lines.append(f"**Page URL:** {report.page_url}")
    if report.figma_file_id:
        lines.append(f"**Figma File:** {report.figma_file_id}")
    lines.append("")
    
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total Findings:** {report.summary.get('total', 0)}")
    lines.append(f"- **Quality Score:** {report.quality_score.get('score', 'N/A')} ({report.quality_score.get('grade', 'N/A')})")
    lines.append("")
    
    if report.summary.get('by_severity'):
        lines.append("### By Severity")
        lines.append("")
        for sev, count in report.summary['by_severity'].items():
            emoji = _severity_emoji(sev)
            lines.append(f"- {emoji} **{sev.title()}:** {count}")
        lines.append("")
    
    if report.summary.get('by_category'):
        lines.append("### By Category")
        lines.append("")
        for cat, count in sorted(report.summary['by_category'].items(), key=lambda x: -x[1]):
            lines.append(f"- **{_format_category(cat)}:** {count}")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    critical = report.by_severity.get('critical', [])
    if critical:
        lines.append("## Critical Issues")
        lines.append("")
        for finding in critical:
            lines.extend(_format_finding_markdown(finding))
        lines.append("")
    
    major = report.by_severity.get('major', [])
    if major:
        lines.append("## Major Issues")
        lines.append("")
        for finding in major:
            lines.extend(_format_finding_markdown(finding))
        lines.append("")
    
    minor = report.by_severity.get('minor', [])
    if minor:
        lines.append("## Minor Issues")
        lines.append("")
        for finding in minor:
            lines.extend(_format_finding_markdown(finding))
        lines.append("")
    
    nit = report.by_severity.get('nit', [])
    info = report.by_severity.get('info', [])
    low_priority = nit + info
    if low_priority:
        lines.append("## Low Priority")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Click to expand ({} items)</summary>".format(len(low_priority)))
        lines.append("")
        for finding in low_priority:
            lines.extend(_format_finding_markdown(finding, brief=True))
        lines.append("</details>")
        lines.append("")
    
    if report.pipeline_metadata:
        lines.append("---")
        lines.append("")
        lines.append("## Pipeline Metadata")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(report.pipeline_metadata, indent=2))
        lines.append("```")
    
    return "\n".join(lines)


def _severity_emoji(severity: str) -> str:
    """Get emoji for severity level."""
    return {
        "critical": "🔴",
        "major": "🟠",
        "minor": "🟡",
        "nit": "🔵",
        "info": "⚪",
    }.get(severity.lower(), "⚪")


def _format_category(category: str) -> str:
    """Format category for display."""
    return category.replace("_", " ").title()


def _format_finding_markdown(
    finding: Dict[str, Any],
    brief: bool = False,
) -> List[str]:
    """Format a single finding as Markdown.
    
    Args:
        finding: Finding dictionary
        brief: If True, use condensed format
    
    Returns:
        List of Markdown lines
    """
    lines = []
    
    serial = finding.get('serial_number', '')
    element = finding.get('element_name', 'Unknown element')
    severity = finding.get('severity', 'minor')
    category = finding.get('category', 'other')
    diff_type = finding.get('diff_type', '')
    
    emoji = _severity_emoji(severity)
    
    if brief:
        lines.append(f"- {emoji} **#{serial}** {element} — {_format_category(category)}")
    else:
        lines.append(f"### {emoji} #{serial}: {element}")
        lines.append("")
        lines.append(f"**Category:** {_format_category(category)}  ")
        lines.append(f"**Type:** {diff_type}  ")
        lines.append(f"**Confidence:** {finding.get('confidence', 'medium')}")
        lines.append("")
        
        dom_evidence = finding.get('dom_evidence')
        if dom_evidence:
            lines.append("**DOM Evidence:**")
            lines.append(f"- Figma: `{dom_evidence.get('figma_value', 'N/A')}`")
            lines.append(f"- Web: `{dom_evidence.get('web_value', 'N/A')}`")
            if dom_evidence.get('delta'):
                lines.append(f"- Delta: {dom_evidence['delta']}")
            lines.append("")
        
        reasoning = finding.get('visual_reasoning')
        if reasoning:
            lines.append("**Visual Analysis:**")
            lines.append(f"> {reasoning}")
            lines.append("")
        
        pos = finding.get('web_position') or finding.get('figma_position')
        if pos:
            lines.append(f"**Position:** x={pos.get('x', 0):.0f}, y={pos.get('y', 0):.0f}, {pos.get('width', 0):.0f}x{pos.get('height', 0):.0f}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    return lines


def generate_html_report(
    report: VQAReport,
    include_screenshots: bool = False,
    figma_screenshot_b64: Optional[str] = None,
    web_screenshot_b64: Optional[str] = None,
) -> str:
    """Generate an HTML-formatted report with optional screenshots.
    
    Args:
        report: VQAReport object
        include_screenshots: Whether to include base64 screenshots
        figma_screenshot_b64: Base64-encoded Figma screenshot
        web_screenshot_b64: Base64-encoded web screenshot
    
    Returns:
        HTML string
    """
    quality = report.quality_score
    score = quality.get('score', 0)
    grade = quality.get('grade', 'N/A')
    
    score_color = "#22c55e" if score >= 85 else "#f59e0b" if score >= 70 else "#ef4444"
    
    findings_html = []
    for finding in report.findings:
        findings_html.append(_format_finding_html(finding))
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visual QA Report</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f8fafc;
            color: #1e293b;
        }}
        .header {{ 
            background: white;
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .header h1 {{ margin: 0 0 16px 0; color: #0f172a; }}
        .meta {{ color: #64748b; font-size: 14px; }}
        .score-card {{
            display: flex;
            align-items: center;
            gap: 24px;
            background: white;
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .score-circle {{
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background: conic-gradient({score_color} {score * 3.6}deg, #e2e8f0 0deg);
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .score-inner {{
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: white;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }}
        .score-value {{ font-size: 24px; font-weight: bold; color: {score_color}; }}
        .score-grade {{ font-size: 14px; color: #64748b; }}
        .summary-stats {{ display: flex; gap: 32px; }}
        .stat {{ text-align: center; }}
        .stat-value {{ font-size: 28px; font-weight: bold; color: #0f172a; }}
        .stat-label {{ font-size: 12px; color: #64748b; text-transform: uppercase; }}
        .severity-counts {{ display: flex; gap: 16px; margin-top: 16px; }}
        .severity-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 14px;
            font-weight: 500;
        }}
        .severity-critical {{ background: #fee2e2; color: #dc2626; }}
        .severity-major {{ background: #ffedd5; color: #ea580c; }}
        .severity-minor {{ background: #fef9c3; color: #ca8a04; }}
        .severity-nit {{ background: #dbeafe; color: #2563eb; }}
        .findings {{ 
            background: white;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .finding {{
            padding: 20px;
            border-bottom: 1px solid #e2e8f0;
        }}
        .finding:last-child {{ border-bottom: none; }}
        .finding-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        .finding-title {{ 
            font-size: 16px;
            font-weight: 600;
            color: #0f172a;
        }}
        .finding-meta {{ font-size: 13px; color: #64748b; }}
        .finding-evidence {{
            background: #f8fafc;
            padding: 12px;
            border-radius: 8px;
            font-size: 14px;
            margin-top: 12px;
        }}
        .finding-evidence code {{
            background: #e2e8f0;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 13px;
        }}
        .screenshots {{
            display: flex;
            gap: 20px;
            margin: 24px 0;
            background: white;
            padding: 20px;
            border-radius: 12px;
        }}
        .screenshot {{ flex: 1; }}
        .screenshot img {{ width: 100%; border-radius: 8px; border: 1px solid #e2e8f0; }}
        .screenshot-label {{ font-size: 14px; font-weight: 500; margin-bottom: 8px; color: #64748b; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Visual QA Report</h1>
        <div class="meta">
            <p>Generated: {report.generated_at}</p>
            {f'<p>Page: <a href="{report.page_url}">{report.page_url}</a></p>' if report.page_url else ''}
        </div>
    </div>
    
    <div class="score-card">
        <div class="score-circle">
            <div class="score-inner">
                <span class="score-value">{score:.0f}</span>
                <span class="score-grade">{grade}</span>
            </div>
        </div>
        <div>
            <h2 style="margin:0 0 8px 0;">Quality Score</h2>
            <p style="margin:0;color:#64748b;">{quality.get('interpretation', '')}</p>
            <div class="severity-counts">
                {''.join(_severity_badge_html(sev, count) for sev, count in report.summary.get('by_severity', {}).items() if count > 0)}
            </div>
        </div>
        <div class="summary-stats" style="margin-left:auto;">
            <div class="stat">
                <div class="stat-value">{report.summary.get('total', 0)}</div>
                <div class="stat-label">Total Issues</div>
            </div>
            <div class="stat">
                <div class="stat-value">{report.summary.get('above_fold_count', 0)}</div>
                <div class="stat-label">Above Fold</div>
            </div>
        </div>
    </div>
    
    {_screenshots_html(figma_screenshot_b64, web_screenshot_b64) if include_screenshots else ''}
    
    <div class="findings">
        <h2 style="padding: 20px 20px 0 20px; margin: 0;">Findings</h2>
        {''.join(findings_html)}
    </div>
</body>
</html>"""
    
    return html


def _severity_badge_html(severity: str, count: int) -> str:
    """Generate HTML for severity badge."""
    return f'<span class="severity-badge severity-{severity}">{_severity_emoji(severity)} {severity.title()}: {count}</span>'


def _format_finding_html(finding: Dict[str, Any]) -> str:
    """Format a single finding as HTML."""
    serial = finding.get('serial_number', '')
    element = finding.get('element_name', 'Unknown')
    severity = finding.get('severity', 'minor')
    category = finding.get('category', 'other')
    confidence = finding.get('confidence', 'medium')
    
    evidence_html = ""
    dom_evidence = finding.get('dom_evidence')
    if dom_evidence:
        evidence_html = f"""
        <div class="finding-evidence">
            <strong>DOM Evidence:</strong><br>
            Figma: <code>{dom_evidence.get('figma_value', 'N/A')}</code><br>
            Web: <code>{dom_evidence.get('web_value', 'N/A')}</code>
            {f"<br>Delta: {dom_evidence.get('delta')}" if dom_evidence.get('delta') else ''}
        </div>"""
    
    reasoning = finding.get('visual_reasoning', '')
    reasoning_html = f'<p style="color:#64748b;margin:12px 0 0 0;font-style:italic;">"{reasoning}"</p>' if reasoning else ''
    
    return f"""
    <div class="finding">
        <div class="finding-header">
            <span class="finding-title">{_severity_emoji(severity)} #{serial}: {element}</span>
            <span class="severity-badge severity-{severity}">{severity.title()}</span>
        </div>
        <div class="finding-meta">
            Category: {_format_category(category)} • Confidence: {confidence}
        </div>
        {evidence_html}
        {reasoning_html}
    </div>"""


def _screenshots_html(figma_b64: Optional[str], web_b64: Optional[str]) -> str:
    """Generate HTML for screenshot comparison."""
    if not figma_b64 and not web_b64:
        return ""
    
    figma_img = f'<img src="data:image/png;base64,{figma_b64}" alt="Figma Design">' if figma_b64 else '<p>Not available</p>'
    web_img = f'<img src="data:image/png;base64,{web_b64}" alt="Web Implementation">' if web_b64 else '<p>Not available</p>'
    
    return f"""
    <div class="screenshots">
        <div class="screenshot">
            <div class="screenshot-label">Figma Design</div>
            {figma_img}
        </div>
        <div class="screenshot">
            <div class="screenshot-label">Web Implementation</div>
            {web_img}
        </div>
    </div>"""


def save_report(
    report: VQAReport,
    output_path: str,
    format: str = "json",
    **kwargs,
) -> str:
    """Save report to file.
    
    Args:
        report: VQAReport object
        output_path: Path to save the report
        format: Output format ("json", "markdown", "html")
        **kwargs: Additional arguments for HTML generation
    
    Returns:
        Path where report was saved
    """
    if format == "json":
        content = report.to_json()
        if not output_path.endswith('.json'):
            output_path += '.json'
    elif format == "markdown":
        content = generate_markdown_report(report)
        if not output_path.endswith('.md'):
            output_path += '.md'
    elif format == "html":
        content = generate_html_report(report, **kwargs)
        if not output_path.endswith('.html'):
            output_path += '.html'
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Saved report to {output_path}")
    return output_path
