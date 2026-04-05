"""
Prompt templates for Pass A: Blind Visual Diff.

This is the most important prompt in the system. It catches everything
the DOM comparator CANNOT: missing shadows, wrong icons, visual hierarchy
issues, missing dividers, cropped elements, and general "looks wrong" problems.

The model receives two full screenshots with NO hints about what to look for.
"""

# System prompt for blind visual analysis
BLIND_VISUAL_SYSTEM_PROMPT = """You are a senior UI QA engineer comparing a Figma design to its web implementation.

You will receive two screenshots:
- Image 1: The FIGMA DESIGN (source of truth)
- Image 2: The WEB IMPLEMENTATION (what was built)

Your job is to identify EVERY visual difference between them. Be thorough and systematic.

SCAN METHODOLOGY (follow this order):
1. LAYOUT SCAN: Are all major sections present? Same positions? Same sizes? Anything missing entirely?
2. SHADOW & ELEVATION: Compare drop shadows, card elevation, hover shadows. Missing shadows are VERY common bugs.
3. ICONS & IMAGES: Are icons the same? Right size? Right color? Any icons swapped or missing?
4. DIVIDERS & BORDERS: Check for missing separator lines, bottom borders on tables, divider lines between sections.
5. TYPOGRAPHY: Font sizes, weights, colors, alignment. Especially headers vs body text.
6. SPACING: Gaps between elements, padding inside containers, margins around sections.
7. COLORS: Background colors, text colors, border colors. Even subtle shade differences.
8. COMPONENTS: Buttons, dropdowns, toggles, checkboxes — do they match the design system?
9. STATES: Any visible interactive states (selected, active, disabled) that look different?
10. OVERFLOW: Any text that appears cropped, truncated differently, or overflowing its container?

CRITICAL RULES:
- Report ONLY differences you can actually SEE in the screenshots
- Do NOT guess or infer — if you can't see it clearly, say "low confidence"
- Be specific about LOCATION (describe where on the page)
- Describe BOTH what the design shows AND what the web shows
- Estimate SEVERITY based on visual impact

Respond with ONLY a JSON array. No other text."""


# Response schema that the model should follow
BLIND_VISUAL_RESPONSE_SCHEMA = """
[
  {
    "category": "shadow|icon|divider|typography|spacing|color|component|state|overflow|layout|missing_element",
    "location_description": "Top-left card in the dashboard grid",
    "bounding_box_estimate": {"x_pct": 0.1, "y_pct": 0.2, "w_pct": 0.3, "h_pct": 0.15},
    "figma_observation": "Card has a subtle drop shadow with ~4px blur",
    "web_observation": "Card has no shadow, appears completely flat",
    "severity": "critical|major|minor|negligible",
    "confidence": 0.95,
    "reasoning": "The shadow is clearly visible in the Figma design but completely absent in the web version. This affects the visual hierarchy of the entire card grid."
  }
]
"""


def build_blind_visual_prompt(
    include_schema: bool = True,
    additional_context: str = None,
) -> str:
    """Build the complete prompt for blind visual analysis.
    
    Args:
        include_schema: Whether to include the response schema
        additional_context: Optional extra context to add
    
    Returns:
        Complete prompt string
    """
    prompt = BLIND_VISUAL_SYSTEM_PROMPT
    
    if include_schema:
        prompt += "\n\nExpected response format:\n" + BLIND_VISUAL_RESPONSE_SCHEMA
    
    if additional_context:
        prompt += f"\n\nAdditional context:\n{additional_context}"
    
    return prompt


# Category mapping for normalizing LLM output to our enum values
CATEGORY_MAPPING = {
    "shadow": "shadow",
    "shadows": "shadow",
    "drop_shadow": "shadow",
    "box_shadow": "shadow",
    "elevation": "shadow",
    
    "icon": "icons",
    "icons": "icons",
    "icon_missing": "icons",
    "icon_different": "icons",
    
    "divider": "border",
    "dividers": "border",
    "separator": "border",
    "border": "border",
    "borders": "border",
    "stroke": "border",
    
    "typography": "text",
    "text": "text",
    "font": "text",
    "font_size": "text",
    "font_weight": "text",
    
    "spacing": "spacing",
    "gap": "spacing",
    "padding": "padding",
    "margin": "spacing",
    
    "color": "color",
    "colors": "color",
    "background": "color",
    "background_color": "color",
    
    "component": "components",
    "components": "components",
    "button": "buttons_cta",
    "buttons": "buttons_cta",
    "cta": "buttons_cta",
    
    "state": "behavioral",
    "states": "behavioral",
    "interactive": "behavioral",
    "hover": "behavioral",
    "active": "behavioral",
    
    "overflow": "overflow",
    "truncation": "overflow",
    "cropped": "overflow",
    
    "layout": "layout",
    "position": "position",
    "alignment": "layout",
    
    "missing_element": "missing_elements",
    "missing": "missing_elements",
}


def normalize_category(raw_category: str) -> str:
    """Normalize LLM category output to our standard enum values.
    
    Args:
        raw_category: Raw category string from LLM
    
    Returns:
        Normalized category matching our Category enum
    """
    if not raw_category:
        return "other"
    
    normalized = raw_category.lower().strip().replace(" ", "_").replace("-", "_")
    return CATEGORY_MAPPING.get(normalized, "other")


# Severity mapping
SEVERITY_MAPPING = {
    "critical": "critical",
    "high": "critical",
    "severe": "critical",
    
    "major": "major",
    "medium": "major",
    "significant": "major",
    
    "minor": "minor",
    "low": "minor",
    "small": "minor",
    
    "negligible": "nit",
    "nit": "nit",
    "trivial": "nit",
    "cosmetic": "nit",
}


def normalize_severity(raw_severity: str) -> str:
    """Normalize LLM severity output to our standard enum values.
    
    Args:
        raw_severity: Raw severity string from LLM
    
    Returns:
        Normalized severity matching our Severity enum
    """
    if not raw_severity:
        return "minor"
    
    normalized = raw_severity.lower().strip()
    return SEVERITY_MAPPING.get(normalized, "minor")


def confidence_to_enum(confidence: float) -> str:
    """Convert numeric confidence to enum value.
    
    Args:
        confidence: Float 0.0-1.0
    
    Returns:
        Confidence enum string
    """
    if confidence >= 0.85:
        return "high"
    elif confidence >= 0.6:
        return "medium"
    elif confidence >= 0.4:
        return "low"
    else:
        return "uncertain"
