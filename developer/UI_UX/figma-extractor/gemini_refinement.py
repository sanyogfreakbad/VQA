#!/usr/bin/env python3
"""
Gemini Visual Refinement Layer

Takes the JSON comparison results + Figma/Web screenshots,
sends them to Gemini's multimodal API for visual validation,
and returns refined results in the same schema.

Gemini can catch things DOM-based comparison misses:
- Missing icons/SVGs
- Visual layout discrepancies
- Button styling differences
- Image/logo mismatches
- Overall visual fidelity issues
"""

import os
import json
import base64
import logging
from typing import Optional

import google.generativeai as genai
from PIL import Image
import io

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

REFINEMENT_SYSTEM_PROMPT = """You are a senior UI/UX QA engineer performing a visual comparison between a Figma design (source of truth) and a live web implementation.

You will receive:
1. A screenshot of the **Figma design** (the intended design) — this is the SOURCE OF TRUTH
2. A screenshot of the **live web page** (the actual implementation) — this is what was ACTUALLY BUILT
3. A JSON report of **automated differences** already detected by DOM comparison

CRITICAL: The Figma screenshot (Image 1) shows what SHOULD exist. The Web screenshot (Image 2) shows what ACTUALLY exists.
If something appears in Figma but NOT in Web, it is MISSING from the implementation — this is a REAL bug, NOT a false positive.

Your job is to **visually inspect both screenshots** and:

A) **VALIDATE** existing differences — BE VERY CONSERVATIVE about marking false positives:
   - ONLY mark as false positive if the element CLEARLY EXISTS and LOOKS CORRECT in the Web screenshot
   - If an element is in Figma but you cannot see it in the Web screenshot, it is GENUINELY MISSING — do NOT remove it
   - If you're unsure whether something is present in Web, do NOT mark it for removal
   - Buttons, icons, and action elements (like "+ Create", "Add", "Download") are especially important — verify they ACTUALLY exist in Web before marking as false positive

B) **FIND NEW differences** that the automated tool missed — especially:
   - Missing icons, SVG graphics, or image assets
   - Missing or different buttons, CTAs, or interactive elements (e.g., "+ Create ASN" button)
   - Color differences in backgrounds, borders, shadows
   - Layout/alignment issues (elements shifted, wrong ordering)
   - Missing sections or components (header, footer, sidebar, table rows)
   - Typography issues visible in screenshots but not caught by DOM extraction
   - Spacing/padding that looks visually wrong
   - Missing hover states, badges, tags, or status indicators
   - Missing table columns or different column headers

C) **CLASSIFY** each new finding using this exact severity system:
   - "error": Element is missing entirely, or has a functionally wrong value (wrong text content, missing button)
   - "warning": Element exists but has a noticeable visual difference (wrong color, weight, spacing)
   - "info": Minor difference that is cosmetic and unlikely to affect user experience

IMPORTANT RULES:
- COMPARE CAREFULLY: Look at the exact same area in both screenshots before deciding
- Do NOT report differences in dynamic/data content (timestamps, user-specific data, random IDs, table row content) — these are expected to differ
- Do NOT flag generic placeholder text like "Link", "Title", "Badge", "Sub Brand", "Call to action", "Body text" as missing — these are Figma placeholder labels, not real content
- DO flag real UI elements: actual buttons with real labels (e.g., "Create ASN", "Apply", "Clear All"), actual input fields, actual navigation items
- Be precise about element identification — use visible text labels or descriptions
- When validating removals, you MUST verify the element ACTUALLY EXISTS in the Web screenshot — don't assume

OUTPUT FORMAT — respond with ONLY a valid JSON object, no markdown fences, no explanation:
{
  "validated_removals": [
    {
      "element": "element name from original report",
      "text": "text value",
      "reason": "why this is a false positive — must explain WHERE you see it in the Web screenshot"
    }
  ],
  "severity_adjustments": [
    {
      "element": "element name",
      "text": "text value",
      "sub_type": "original sub_type",
      "original_severity": "warning",
      "new_severity": "error",
      "reason": "why severity changed"
    }
  ],
  "new_differences": [
    {
      "element": "descriptive element name",
      "text": "visible text or description",
      "sub_type": "missing | color | size | spacing | icon | layout | content",
      "figma_value": "what it should be (from Figma screenshot)",
      "web_value": "what it actually is (from web screenshot), or null if missing",
      "delta": "human-readable description of the difference",
      "severity": "error | warning | info",
      "category": "missing_elements | text | size | spacing | color | icon | layout"
    }
  ],
  "visual_summary": "2-3 sentence overall assessment of visual fidelity"
}
"""


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class GeminiRefinementLayer:
    """
    Sends comparison results + screenshots to Gemini for visual validation.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Args:
            api_key: Google AI API key. Falls back to GEMINI_API_KEY env var.
            model_name: Gemini model to use. Falls back to GEMINI_MODEL env var,
                        then defaults to gemini-2.5-flash.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not configured. "
                "Provide api_key parameter or set GEMINI_API_KEY in .env"
            )

        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def _prepare_image(self, image_data: bytes | str) -> Image.Image:
        """Convert raw bytes or base64 string to PIL Image."""
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data)
        return Image.open(io.BytesIO(image_data))

    def _build_user_prompt(self, comparison_json: dict) -> str:
        """Build the user message containing the comparison data."""
        # Trim to keep token usage reasonable — send summary + by_category
        trimmed = {
            "summary": comparison_json.get("summary", {}),
            "by_category": comparison_json.get("by_category", {}),
        }
        return (
            "Here is the automated comparison report between Figma and Web:\n\n"
            f"{json.dumps(trimmed, indent=2)}\n\n"
            "Please visually inspect the two screenshots (Image 1 = Figma design, "
            "Image 2 = Web implementation) and refine this report. "
            "Respond with ONLY the JSON object as specified."
        )

    async def refine_async(
        self,
        comparison_results: dict,
        figma_screenshot: bytes | str,
        web_screenshot: bytes | str,
    ) -> dict:
        """
        Send comparison + screenshots to Gemini and return refined results.

        Args:
            comparison_results: The JSON output from DesignComparator.compare_all()
            figma_screenshot: Figma screenshot as bytes or base64 string
            web_screenshot: Web screenshot as bytes or base64 string

        Returns:
            Merged comparison results with Gemini refinements applied.
        """
        figma_img = self._prepare_image(figma_screenshot)
        web_img = self._prepare_image(web_screenshot)
        user_prompt = self._build_user_prompt(comparison_results)

        # Build multimodal content
        content = [
            REFINEMENT_SYSTEM_PROMPT,
            "Image 1 — Figma Design (source of truth):",
            figma_img,
            "Image 2 — Live Web Implementation:",
            web_img,
            user_prompt,
        ]

        logger.info("Sending comparison + 2 screenshots to Gemini (%s)", self.model_name)

        # For thinking models, response_mime_type may not work well
        # Try with it first, fall back without if parsing fails
        try:
            response = await self.model.generate_content_async(
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temp for structured output
                    max_output_tokens=16384,
                    response_mime_type="application/json",  # Force JSON output
                ),
            )
        except Exception as e:
            logger.warning("generate_content_async with response_mime_type failed: %s, retrying without", e)
            response = await self.model.generate_content_async(
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=16384,
                ),
            )

        # Parse Gemini's JSON response
        raw_text = self._extract_text_from_response(response)
        print(f"\n=== GEMINI RAW RESPONSE (first 2000 chars) ===\n{raw_text[:2000]}\n=== END ===\n")
        gemini_output = self._parse_gemini_response(raw_text)

        # Merge into original results
        refined = self._merge_results(comparison_results, gemini_output)
        return refined

    def refine_sync(
        self,
        comparison_results: dict,
        figma_screenshot: bytes | str,
        web_screenshot: bytes | str,
    ) -> dict:
        """Synchronous version of refine_async."""
        figma_img = self._prepare_image(figma_screenshot)
        web_img = self._prepare_image(web_screenshot)
        user_prompt = self._build_user_prompt(comparison_results)

        content = [
            REFINEMENT_SYSTEM_PROMPT,
            "Image 1 — Figma Design (source of truth):",
            figma_img,
            "Image 2 — Live Web Implementation:",
            web_img,
            user_prompt,
        ]

        logger.info("Sending comparison + 2 screenshots to Gemini (%s) [sync]", self.model_name)

        # For thinking models, response_mime_type may not work well
        try:
            response = self.model.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=16384,
                    response_mime_type="application/json",  # Force JSON output
                ),
            )
        except Exception as e:
            logger.warning("generate_content with response_mime_type failed: %s, retrying without", e)
            response = self.model.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=16384,
                ),
            )

        raw_text = self._extract_text_from_response(response)
        print(f"\n=== GEMINI RAW RESPONSE (first 2000 chars) ===\n{raw_text[:2000]}\n=== END ===\n")
        gemini_output = self._parse_gemini_response(raw_text)
        refined = self._merge_results(comparison_results, gemini_output)
        return refined

    def _extract_text_from_response(self, response) -> str:
        """
        Extract text from Gemini response, handling thinking models (2.5-flash etc).
        
        Thinking models return candidates with parts that have a 'thought' flag.
        We skip thought parts and only use the actual text output.
        """
        # Log response structure for debugging
        logger.debug("Response type: %s", type(response))
        
        # Method 1: Try direct .text first (works for non-thinking models)
        try:
            text = response.text
            if text:
                logger.debug("Extracted via .text (first 500 chars): %s", text[:500])
                return text
        except (AttributeError, ValueError) as e:
            logger.debug("Direct .text failed: %s", e)

        # Method 2: Manual extraction from candidates/parts
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, 'finish_reason', 'unknown')
                logger.debug("Candidate finish_reason: %s", finish_reason)
                print(f"Gemini finish_reason: {finish_reason}")
                
                # Check for safety ratings / blocked content
                if hasattr(candidate, 'safety_ratings'):
                    for rating in candidate.safety_ratings:
                        logger.debug("Safety rating: %s = %s", rating.category, rating.probability)
                
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        parts = candidate.content.parts
                        text_parts = []
                        for i, part in enumerate(parts):
                            is_thought = getattr(part, 'thought', False)
                            has_text = hasattr(part, 'text') and part.text
                            logger.debug("Part %d: thought=%s, has_text=%s", i, is_thought, has_text)
                            
                            # Skip thinking/reasoning parts
                            if is_thought:
                                continue
                            if has_text:
                                text_parts.append(part.text)
                        
                        if text_parts:
                            result = "\n".join(text_parts)
                            logger.debug("Extracted from parts (first 500 chars): %s", result[:500])
                            return result
                    else:
                        logger.warning("Candidate content has no parts")
                else:
                    logger.warning("Candidate has no content")
            else:
                logger.warning("Response has no candidates")
                # Check if response was blocked
                if hasattr(response, 'prompt_feedback'):
                    logger.error("Prompt feedback: %s", response.prompt_feedback)
                    print(f"Gemini prompt_feedback: {response.prompt_feedback}")
        except (AttributeError, IndexError) as e:
            logger.error("Failed to extract text from Gemini response: %s", e)

        # Method 3: Try to get raw response data
        try:
            # Some SDK versions expose _result or similar
            if hasattr(response, '_result'):
                logger.debug("Response has _result: %s", type(response._result))
            if hasattr(response, 'to_dict'):
                resp_dict = response.to_dict()
                logger.debug("Response to_dict keys: %s", resp_dict.keys())
        except Exception as e:
            logger.debug("Could not inspect response internals: %s", e)

        # Method 4: Try to serialize the whole response for debugging
        try:
            logger.error("Could not extract text. Response repr: %s", repr(response)[:1000])
            print(f"\n=== RESPONSE OBJECT DEBUG ===\n{repr(response)[:2000]}\n=== END ===\n")
        except Exception:
            pass

        return ""

    def _parse_gemini_response(self, raw_text: str) -> dict:
        """
        Parse Gemini's response into structured JSON.
        
        Handles multiple formats:
        - Clean JSON (when response_mime_type works)
        - JSON wrapped in ```json ... ``` fences
        - JSON buried after reasoning text
        - Malformed responses
        """
        if not raw_text or not raw_text.strip():
            logger.error("Empty response from Gemini")
            return self._fallback_response("Empty response from Gemini")

        text = raw_text.strip()
        logger.info("Parsing Gemini response (length: %d chars)", len(text))
        logger.debug("Raw text preview: %s", text[:500])

        # Attempt 1: Direct parse (ideal case with response_mime_type)
        try:
            result = json.loads(text)
            logger.info("Successfully parsed JSON directly")
            return result
        except json.JSONDecodeError as e:
            logger.debug("Direct JSON parse failed: %s", e)

        # Attempt 2: Strip markdown code fences
        import re
        fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        if fence_match:
            try:
                result = json.loads(fence_match.group(1))
                logger.info("Successfully parsed JSON from code fence")
                return result
            except json.JSONDecodeError as e:
                logger.debug("Code fence JSON parse failed: %s", e)

        # Attempt 3: Find the first { ... } JSON object in the text
        brace_start = text.find('{')
        if brace_start != -1:
            logger.debug("Found opening brace at position %d", brace_start)
            # Find the matching closing brace by counting
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[brace_start:i + 1]
                        try:
                            result = json.loads(candidate)
                            logger.info("Successfully parsed JSON by brace matching")
                            return result
                        except json.JSONDecodeError as e:
                            logger.debug("Brace-matched JSON parse failed at pos %d: %s", i, e)
                            break

        # Attempt 4: Try fixing common issues (trailing commas)
        if brace_start != -1:
            candidate = text[brace_start:]
            # Find last }
            last_brace = candidate.rfind('}')
            if last_brace != -1:
                candidate = candidate[:last_brace + 1]
                # Remove trailing commas before } or ]
                cleaned = re.sub(r',\s*([}\]])', r'\1', candidate)
                try:
                    result = json.loads(cleaned)
                    logger.info("Successfully parsed JSON after cleanup")
                    return result
                except json.JSONDecodeError as e:
                    logger.debug("Cleaned JSON parse failed: %s", e)

        # Log full response for debugging when all attempts fail
        logger.error("All JSON parsing attempts failed")
        logger.error("Full raw response:\n%s", raw_text)
        return self._fallback_response(
            f"Could not parse Gemini response as JSON",
            raw_text=raw_text,
        )

    def _fallback_response(self, reason: str, raw_text: str = "") -> dict:
        """Return a safe fallback when parsing fails."""
        return {
            "validated_removals": [],
            "severity_adjustments": [],
            "new_differences": [],
            "visual_summary": f"Gemini response parsing failed: {reason}",
            "_raw_response": raw_text[:3000] if raw_text else "",
        }

    def _merge_results(self, original: dict, gemini: dict) -> dict:
        """
        Merge Gemini refinements into the original comparison results.

        Returns a new dict with:
        - Original differences (minus false positives)
        - Severity adjustments applied
        - New Gemini-found differences appended
        - gemini_refinement metadata block
        """
        merged = json.loads(json.dumps(original))  # deep copy

        removals = {
            (r.get("element"), r.get("text"))
            for r in gemini.get("validated_removals", [])
        }

        adjustments = {
            (a.get("element"), a.get("text"), a.get("sub_type")): a.get("new_severity")
            for a in gemini.get("severity_adjustments", [])
        }

        removed_count = 0
        adjusted_count = 0

        # Process existing categories
        by_cat = merged.get("by_category", {})
        for category, items in by_cat.items():
            if not isinstance(items, list):
                continue

            filtered = []
            for item in items:
                key = (item.get("element"), item.get("text"))
                if key in removals:
                    removed_count += 1
                    continue

                adj_key = (item.get("element"), item.get("text"), item.get("sub_type"))
                if adj_key in adjustments:
                    item["original_severity"] = item.get("severity")
                    item["severity"] = adjustments[adj_key]
                    item["gemini_adjusted"] = True
                    adjusted_count += 1

                filtered.append(item)

            by_cat[category] = filtered

        # Append new differences from Gemini
        new_diffs = gemini.get("new_differences", [])
        for diff in new_diffs:
            cat = diff.pop("category", "missing_elements")
            diff["source"] = "gemini_visual"  # tag so consumers know the origin
            by_cat.setdefault(cat, []).append(diff)

        # Recompute summary counts
        total = 0
        errors = 0
        warnings = 0
        info = 0
        cat_counts = {}

        for category, items in by_cat.items():
            if not isinstance(items, list):
                continue
            cat_counts[category] = len(items)
            for item in items:
                total += 1
                sev = item.get("severity", "info")
                if sev == "error":
                    errors += 1
                elif sev == "warning":
                    warnings += 1
                else:
                    info += 1

        merged["summary"] = {
            "total_differences": total,
            "errors": errors,
            "warnings": warnings,
            "info": info,
            "categories": cat_counts,
        }

        # Add Gemini metadata
        merged["gemini_refinement"] = {
            "model": self.model_name,
            "false_positives_removed": removed_count,
            "severity_adjustments": adjusted_count,
            "new_issues_found": len(new_diffs),
            "visual_summary": gemini.get("visual_summary", ""),
            "validated_removals": gemini.get("validated_removals", []),
            "severity_adjustment_details": gemini.get("severity_adjustments", []),
        }

        return merged