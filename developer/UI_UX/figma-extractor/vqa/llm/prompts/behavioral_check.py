"""
Prompt templates for Pass C: Behavioral State Checks.

This handles interaction states that require Playwright to trigger:
- Hover states (gradients, color changes, cursor)
- Focus states (outlines, highlights)
- Active/pressed states
- Disabled states
- Dropdown open states
- Toast/notification appearance and dismissal
- Scroll behavior effects

For each interaction, Playwright captures before + after screenshots.
The LLM compares the "after" state against the Figma spec for that state.
"""

# System prompt for behavioral state checking
BEHAVIORAL_CHECK_SYSTEM_PROMPT = """You are a UI QA expert checking interactive behavior.

You will see:
- Image 1: The Figma design showing the EXPECTED state after interaction
- Image 2: The web screenshot showing the ACTUAL state after interaction
- INTERACTION: What was done (hover, click, focus, etc.)
- ELEMENT: Which element was interacted with

Check if the web's interactive state matches the Figma spec. Common issues:
- Hover state has wrong gradient or color
- Disabled state looks the same as selected state
- Focus ring is missing or wrong color
- Toast notification doesn't appear or has wrong styling
- Dropdown doesn't match the width of its trigger
- Scroll effect doesn't apply (sticky header, parallax, shadow on scroll)

Respond with ONLY JSON."""


# Response schema for behavioral checks
BEHAVIORAL_RESPONSE_SCHEMA = """
{
  "interaction": "hover|click|focus|scroll|toggle",
  "element": "element description",
  "state_matches": true/false,
  "differences": [
    {
      "property": "what differs",
      "expected": "what Figma shows",
      "actual": "what web shows",
      "severity": "critical|major|minor",
      "confidence": 0.0 to 1.0,
      "reasoning": "explanation"
    }
  ]
}
"""


# Batch response schema for multiple interactions
BEHAVIORAL_BATCH_RESPONSE_SCHEMA = """
[
  {
    "interaction_index": 0,
    "interaction": "hover",
    "element": "Submit Button",
    "state_matches": false,
    "differences": [
      {
        "property": "background-color",
        "expected": "Gradient from blue to purple",
        "actual": "Solid blue, no gradient",
        "severity": "major",
        "confidence": 0.9,
        "reasoning": "The hover state gradient is clearly visible in Figma but missing in web"
      }
    ],
    "additional_observations": ["Focus ring also differs"]
  }
]
"""


def build_behavioral_check_prompt(
    include_schema: bool = True,
) -> str:
    """Build the system prompt for behavioral checking.
    
    Args:
        include_schema: Whether to include response schema
    
    Returns:
        Complete system prompt string
    """
    prompt = BEHAVIORAL_CHECK_SYSTEM_PROMPT
    
    if include_schema:
        prompt += "\n\nExpected response format:\n" + BEHAVIORAL_BATCH_RESPONSE_SCHEMA
    
    return prompt


def build_interaction_context(
    interaction_type: str,
    element_name: str,
    element_type: str,
    interaction_index: int = 0,
    expected_changes: list = None,
) -> str:
    """Build context string for an interaction state check.
    
    Args:
        interaction_type: Type of interaction (hover, click, focus, etc.)
        element_name: Human-readable element name
        element_type: Component type (button, dropdown, etc.)
        interaction_index: Index in batch
        expected_changes: List of CSS properties expected to change
    
    Returns:
        Context string for the interaction
    """
    expected_str = ""
    if expected_changes:
        expected_str = f"Expected changes: {', '.join(expected_changes)}"
    
    return f"""
--- INTERACTION {interaction_index + 1} ---
Element: {element_name}
Element type: {element_type}
Interaction: {interaction_type}
{expected_str}

Image 1: Figma design showing expected {interaction_type} state
Image 2: Web after {interaction_type} interaction

Does the web {interaction_type} state match the Figma specification?
"""


# Interaction type to expected CSS property changes
INTERACTION_EXPECTED_CHANGES = {
    "hover": ["background-color", "box-shadow", "transform", "cursor", "color"],
    "focus": ["outline", "box-shadow", "border-color"],
    "click": ["background-color", "transform", "box-shadow"],
    "active": ["background-color", "transform"],
    "disabled": ["opacity", "cursor", "background-color", "color"],
    "selected": ["background-color", "font-weight", "border-color", "color"],
    "toggle_on": ["background-color", "transform"],
    "toggle_off": ["background-color", "transform"],
    "dropdown_open": ["height", "overflow", "box-shadow", "opacity"],
    "scroll": ["position", "box-shadow", "transform"],
}


def get_expected_changes_for_interaction(interaction_type: str) -> list:
    """Get list of CSS properties expected to change for an interaction.
    
    Args:
        interaction_type: Type of interaction
    
    Returns:
        List of CSS property names
    """
    return INTERACTION_EXPECTED_CHANGES.get(
        interaction_type.lower(),
        ["background-color", "box-shadow"]
    )


# Common behavioral issues from QA reviews
COMMON_BEHAVIORAL_ISSUES = {
    "hover_gradient": "Hover state should have gradient but shows solid color",
    "hover_no_change": "Element should change on hover but remains static",
    "focus_ring_missing": "Focus ring/outline is missing on focusable element",
    "focus_ring_color": "Focus ring color doesn't match design system",
    "disabled_same_as_active": "Disabled state looks the same as active state",
    "disabled_cursor": "Disabled element should have not-allowed cursor",
    "selected_no_indicator": "Selected state has no visual indicator",
    "toast_position": "Toast appears in wrong position",
    "toast_timing": "Toast dismisses too quickly or doesn't auto-dismiss",
    "dropdown_width_mismatch": "Dropdown width doesn't match trigger element",
    "dropdown_no_shadow": "Open dropdown should have elevation shadow",
    "sticky_not_sticky": "Element should be sticky/fixed but scrolls with page",
    "scroll_shadow_missing": "Shadow should appear on scroll but doesn't",
}


def interpret_behavioral_result(result: dict) -> dict:
    """Interpret behavioral check result.
    
    Args:
        result: Raw result from LLM
    
    Returns:
        Normalized result with findings
    """
    state_matches = result.get("state_matches", True)
    differences = result.get("differences", [])
    
    # Map to standard finding format
    findings = []
    for diff in differences:
        finding = {
            "category": "behavioral",
            "diff_type": f"behavioral_{result.get('interaction', 'unknown')}",
            "severity": diff.get("severity", "minor"),
            "confidence": diff.get("confidence", 0.7),
            "figma_value": diff.get("expected"),
            "web_value": diff.get("actual"),
            "reasoning": diff.get("reasoning", ""),
            "element_name": result.get("element", ""),
        }
        findings.append(finding)
    
    return {
        "state_matches": state_matches,
        "findings": findings,
        "interaction": result.get("interaction"),
        "element": result.get("element"),
    }
