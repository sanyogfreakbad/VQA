"""
Stage 4: Refinement prompt for uncertain findings.

This runs ONLY on findings where confidence < 0.7 from Stage 2.
It uses:
1. Higher-resolution crops (2x zoom on the region)
2. Few-shot examples from the calibration library
3. Explicit chain-of-thought reasoning steps
"""

# System prompt for refinement analysis
REFINEMENT_SYSTEM_PROMPT = """You are a senior UI QA expert performing a detailed re-examination.

A previous analysis flagged this as a POSSIBLE difference but with low confidence.
Your job is to make a definitive determination.

BEFORE analyzing, I'll show you calibrated examples of REAL and FALSE POSITIVE differences
from this same codebase. Use them to calibrate your judgment.

{few_shot_examples_block}

Now analyze the following finding carefully.

THINK STEP BY STEP:
1. What UI element is shown in these crops?
2. What specific visual property was flagged?
3. Looking at ONLY what is visible in the images — can I see a difference?
4. If I see a difference, how visually significant is it at normal viewing distance?
5. Could this be a rendering artifact (antialiasing, sub-pixel, compression)?
6. Final verdict: is this a real design deviation that a human reviewer would flag?

Respond with ONLY JSON."""


# Response schema for refinement
REFINEMENT_RESPONSE_SCHEMA = """
{
  "element_description": "what this element is",
  "step_1_element": "I can see a button/card/text element...",
  "step_2_property": "The flagged property is...",
  "step_3_visible_diff": "Looking at the images, I can/cannot see...",
  "step_4_significance": "At normal viewing distance, this would/wouldn't be noticeable because...",
  "step_5_artifact_check": "This is/isn't a rendering artifact because...",
  "final_verdict": "confirmed|rejected",
  "final_confidence": 0.0 to 1.0,
  "final_severity": "critical|major|minor|negligible",
  "reasoning_summary": "One sentence summary"
}
"""


def build_refinement_prompt(
    few_shot_block: str = None,
    include_schema: bool = True,
) -> str:
    """Build the complete refinement prompt.
    
    Args:
        few_shot_block: Few-shot examples text block
        include_schema: Whether to include response schema
    
    Returns:
        Complete prompt string
    """
    # Replace placeholder with actual few-shot examples
    if few_shot_block:
        prompt = REFINEMENT_SYSTEM_PROMPT.replace(
            "{few_shot_examples_block}",
            few_shot_block
        )
    else:
        prompt = REFINEMENT_SYSTEM_PROMPT.replace(
            "{few_shot_examples_block}",
            "No calibration examples available for this category yet."
        )
    
    if include_schema:
        prompt += "\n\nExpected response format:\n" + REFINEMENT_RESPONSE_SCHEMA
    
    return prompt


def build_finding_context(
    finding: dict,
    finding_index: int = 0,
) -> str:
    """Build context for a finding being refined.
    
    Args:
        finding: Original finding data
        finding_index: Index in batch
    
    Returns:
        Context string
    """
    category = finding.get("category", "unknown")
    diff_type = finding.get("diff_type", finding.get("sub_type", "unknown"))
    element_name = finding.get("element_name", "Unknown element")
    figma_value = finding.get("figma_value", finding.get("dom_evidence", {}).get("figma_value", "N/A"))
    web_value = finding.get("web_value", finding.get("dom_evidence", {}).get("web_value", "N/A"))
    original_confidence = finding.get("confidence", 0.5)
    original_reasoning = finding.get("reasoning", finding.get("visual_reasoning", ""))
    
    return f"""
--- FINDING {finding_index + 1} (NEEDS REFINEMENT) ---
Element: {element_name}
Category: {category}
Difference type: {diff_type}
Figma value: {figma_value}
Web value: {web_value}
Original confidence: {original_confidence}
Original reasoning: {original_reasoning}

Image 1: HIGH-RESOLUTION Figma crop (2x zoom)
Image 2: HIGH-RESOLUTION Web crop (2x zoom)

Please re-examine carefully with the step-by-step methodology.
"""


# Few-shot example format
FEW_SHOT_EXAMPLE_TEMPLATE = """
EXAMPLE {index} ({label}):
Category: {category}
Reported difference: {diff_description}
[Figma crop] [Web crop]
Verdict: {verdict}
Reasoning: {reasoning}
"""


def format_few_shot_example(
    example: dict,
    index: int,
) -> str:
    """Format a single few-shot example for the prompt.
    
    Args:
        example: Example data from calibration store
        index: Example number
    
    Returns:
        Formatted example string
    """
    ground_truth = example.get("ground_truth", "unknown")
    label = "REAL DIFFERENCE" if ground_truth == "confirmed" else "FALSE POSITIVE"
    
    return FEW_SHOT_EXAMPLE_TEMPLATE.format(
        index=index,
        label=label,
        category=example.get("category", "unknown"),
        diff_description=example.get("diff_description", ""),
        verdict=ground_truth,
        reasoning=example.get("reasoning", ""),
    )


def build_few_shot_block(examples: list) -> str:
    """Build the few-shot examples block for refinement prompt.
    
    Args:
        examples: List of calibration examples
    
    Returns:
        Formatted examples block
    """
    if not examples:
        return "No calibration examples available for this category yet."
    
    formatted = []
    for i, example in enumerate(examples):
        formatted.append(format_few_shot_example(example, i + 1))
    
    return "\n".join(formatted)


# Confidence adjustment rules based on chain-of-thought
def adjust_confidence_from_cot(result: dict) -> float:
    """Adjust confidence based on chain-of-thought reasoning.
    
    Applies rules based on the model's step-by-step analysis:
    - If artifact check is positive, reduce confidence
    - If significance is low, reduce severity
    - If multiple steps agree, increase confidence
    
    Args:
        result: Refinement result with step fields
    
    Returns:
        Adjusted confidence value
    """
    base_confidence = result.get("final_confidence", 0.5)
    
    adjustments = 0.0
    
    # Check for consistent steps
    step_3 = result.get("step_3_visible_diff", "").lower()
    step_4 = result.get("step_4_significance", "").lower()
    step_5 = result.get("step_5_artifact_check", "").lower()
    
    # Strong visibility statement
    if "clearly" in step_3 or "definitely" in step_3 or "obvious" in step_3:
        adjustments += 0.1
    elif "cannot" in step_3 or "barely" in step_3 or "subtle" in step_3:
        adjustments -= 0.1
    
    # Significance assessment
    if "noticeable" in step_4 and "wouldn't" not in step_4:
        adjustments += 0.05
    elif "wouldn't" in step_4 or "not noticeable" in step_4:
        adjustments -= 0.1
    
    # Artifact check
    if "isn't a rendering artifact" in step_5 or "is not" in step_5:
        adjustments += 0.05
    elif "is a rendering artifact" in step_5 or "could be" in step_5:
        adjustments -= 0.15
    
    adjusted = base_confidence + adjustments
    return max(0.1, min(1.0, adjusted))


def interpret_refinement_result(result: dict) -> dict:
    """Interpret refinement result and normalize.
    
    Args:
        result: Raw refinement result from LLM
    
    Returns:
        Normalized result
    """
    verdict = result.get("final_verdict", "uncertain")
    is_confirmed = verdict.lower() in ("confirmed", "confirm", "real", "yes", "true")
    
    adjusted_confidence = adjust_confidence_from_cot(result)
    
    return {
        "is_confirmed": is_confirmed,
        "confidence": adjusted_confidence,
        "severity": result.get("final_severity", "minor"),
        "reasoning": result.get("reasoning_summary", ""),
        "chain_of_thought": {
            "element": result.get("step_1_element"),
            "property": result.get("step_2_property"),
            "visible": result.get("step_3_visible_diff"),
            "significance": result.get("step_4_significance"),
            "artifact_check": result.get("step_5_artifact_check"),
        },
    }
