"""
Prompt templates for Pass B: Targeted Validation.

For each DOM-detected difference, we send the model a CROPPED PAIR showing
just that element from both screenshots, plus the specific difference data.

The model's job is to CONFIRM or REJECT the DOM finding, and assess severity.
This eliminates false positives from CSS transforms, visual overrides, or
differences that exist in the DOM but aren't visually perceptible.
"""

# System prompt for targeted validation
TARGETED_VALIDATION_SYSTEM_PROMPT = """You are a UI QA expert validating reported design differences.

For each pair of cropped images I show you:
- Image 1 (LEFT): Figma design crop
- Image 2 (RIGHT): Web implementation crop
- REPORTED DIFF: What the automated comparison tool found

Your job: Determine if this reported difference is VISUALLY REAL or a FALSE POSITIVE.

A difference is REAL if you can see it in the screenshots.
A difference is a FALSE POSITIVE if:
- The DOM values differ but the visual result looks the same (e.g., #1a1a1a vs #000000)
- The difference is caused by rendering engine differences (antialiasing, sub-pixel)
- The property differs in code but a CSS override makes them look identical

Respond with ONLY JSON. No other text."""


# Response schema for individual validation
TARGETED_VALIDATION_RESPONSE_SCHEMA = """
{
  "reported_diff": "ECHOED FROM INPUT",
  "verdict": "confirmed|rejected|uncertain",
  "visual_match": true/false,
  "actual_differences": [
    {
      "property": "font-size|color|spacing|shadow|border|icon|layout|other",
      "figma_observation": "what you see in the Figma crop",
      "web_observation": "what you see in the Web crop",
      "severity": "critical|major|minor|negligible",
      "confidence": 0.0 to 1.0,
      "reasoning": "Why you believe this"
    }
  ],
  "additional_findings": [
    "Any OTHER differences you notice in these crops that weren't reported"
  ]
}
"""


# Batch response schema
TARGETED_BATCH_RESPONSE_SCHEMA = """
[
  {
    "pair_index": 0,
    "reported_diff": "echoed from input",
    "verdict": "confirmed|rejected|uncertain",
    "visual_match": true/false,
    "severity": "critical|major|minor|nit",
    "confidence": 0.0 to 1.0,
    "reasoning": "explanation",
    "additional_findings": ["other differences noticed"]
  }
]
"""


def build_targeted_validation_prompt(
    include_schema: bool = True,
) -> str:
    """Build the system prompt for targeted validation.
    
    Args:
        include_schema: Whether to include response schema
    
    Returns:
        Complete system prompt string
    """
    prompt = TARGETED_VALIDATION_SYSTEM_PROMPT
    
    if include_schema:
        prompt += "\n\nFor batched analysis, respond with:\n" + TARGETED_BATCH_RESPONSE_SCHEMA
    
    return prompt


def build_pair_context(
    diff_data: dict,
    pair_index: int,
    element_name: str = None,
) -> str:
    """Build context string for a single crop pair.
    
    Args:
        diff_data: DOM diff data with figma_value, web_value, diff_type
        pair_index: Index of this pair in the batch
        element_name: Human-readable element identifier
    
    Returns:
        Context string to include with the image pair
    """
    diff_type = diff_data.get("diff_type", diff_data.get("sub_type", "unknown"))
    figma_value = diff_data.get("figma_value", "N/A")
    web_value = diff_data.get("web_value", "N/A")
    delta = diff_data.get("delta", "")
    
    context = f"""
--- ELEMENT {pair_index + 1} ---
Element: {element_name or 'Unknown element'}
Reported difference type: {diff_type}
Figma value: {figma_value}
Web value: {web_value}
Delta: {delta}

Is this difference VISUALLY PERCEPTIBLE in the screenshots?
"""
    return context


def build_batch_prompt(
    pairs_context: list,
) -> str:
    """Build the complete prompt for a batch of crop pairs.
    
    Args:
        pairs_context: List of context strings from build_pair_context
    
    Returns:
        Complete prompt for the batch
    """
    contexts_text = "\n".join(pairs_context)
    
    return f"""Analyze each element pair below and validate the reported differences.

{contexts_text}

For EACH pair, provide your verdict in the JSON array response.
Remember: Only mark as CONFIRMED if you can actually SEE the difference in the images.
Mark as REJECTED if the difference exists in code but looks the same visually."""


# Verdict to confidence mapping
VERDICT_CONFIDENCE_MAP = {
    "confirmed": 0.9,
    "rejected": 0.85,
    "uncertain": 0.5,
}


def interpret_verdict(verdict: str, confidence: float = None) -> dict:
    """Interpret the validation verdict.
    
    Args:
        verdict: LLM verdict string
        confidence: Optional confidence from LLM
    
    Returns:
        Dict with is_valid and confidence
    """
    verdict_lower = verdict.lower().strip() if verdict else "uncertain"
    
    if verdict_lower in ("confirmed", "confirm", "yes", "real", "true"):
        return {
            "is_valid": True,
            "confidence": confidence or VERDICT_CONFIDENCE_MAP["confirmed"],
        }
    elif verdict_lower in ("rejected", "reject", "no", "false_positive", "false"):
        return {
            "is_valid": False,
            "confidence": confidence or VERDICT_CONFIDENCE_MAP["rejected"],
        }
    else:
        return {
            "is_valid": None,
            "confidence": confidence or VERDICT_CONFIDENCE_MAP["uncertain"],
        }


# Property type mapping for validation
PROPERTY_TYPE_MAPPING = {
    "font-size": "text",
    "font_size": "text",
    "fontSize": "text",
    "font-weight": "text",
    "font_weight": "text",
    "fontWeight": "text",
    "font-family": "text",
    "font_family": "text",
    "fontFamily": "text",
    "color": "color",
    "text-color": "color",
    "text_color": "color",
    "textColor": "color",
    "background-color": "color",
    "background_color": "color",
    "backgroundColor": "color",
    "spacing": "spacing",
    "padding": "padding",
    "margin": "spacing",
    "gap": "spacing",
    "shadow": "shadow",
    "box-shadow": "shadow",
    "box_shadow": "shadow",
    "boxShadow": "shadow",
    "border": "border",
    "border-radius": "border",
    "border_radius": "border",
    "borderRadius": "border",
    "icon": "icons",
    "layout": "layout",
    "other": "other",
}


def normalize_property_type(property_name: str) -> str:
    """Normalize property name to category.
    
    Args:
        property_name: Raw property name
    
    Returns:
        Normalized category string
    """
    if not property_name:
        return "other"
    
    normalized = property_name.lower().strip().replace("-", "_")
    return PROPERTY_TYPE_MAPPING.get(normalized, "other")
