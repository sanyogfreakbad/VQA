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

REFINEMENT_SYSTEM_PROMPT = """You are a senior UI/UX QA engineer performing a VISUAL comparison between a Figma design and a live web implementation.

You will receive:
1. A screenshot of the Figma design (Image 1) — SOURCE OF TRUTH
2. A screenshot of the live web page (Image 2) — ACTUAL IMPLEMENTATION
3. A JSON report of automated differences already detected by DOM comparison

The automated tool has already handled:
- Text property comparisons (font weight, size, color)
- Table column set matching
- Filter field matching
- Action button presence checks

YOUR JOB is to catch what the automated tool CANNOT:
- Visual/spatial issues only visible in screenshots
- Interaction type mismatches (dropdown vs search input)
- Icon presence, alignment, and type
- Layout proportions and spacing that require visual judgment
- Component styling details (border-radius, shadows, hover states)
- Viewport/scroll behavior expectations

═══════════════════════════════════════════════════════════════════
STEP 1: VALIDATE THE AUTOMATED REPORT (remove false positives)
═══════════════════════════════════════════════════════════════════

Review each item in the JSON report and REMOVE false positives:

 Remove if ANY of these apply:
1. Table body data: ANY text from a table data row (below the header).
   Table data is dynamic — Figma uses sample values, web uses production data.
   This includes: ASN numbers, PO numbers, dates, names, status values,
   links within rows, per-row badges, action items per row.
   TEST: "Is this text inside a data row, not a column header?" → remove.

2. Figma component labels: Generic names from the Figma component library:
   "Link", "Title", "Badge", "Sub Brand", "Body text", "Call to action"
   (when used as a generic label, NOT when it contains real UI text like
   "Create ASN" or "Clear All").
   TEST: "Would a developer see this text in the final UI?" If no → remove.

3. Pagination metadata: Item counts, page numbers, items-per-page values.
   These differ because datasets differ. "01-10 of 90 items" ≠ "1-25 of 31171 items"
   but both are working pagination.

4. Sample contact/identity data: "Jd@abc.com", "+91", "USD", sample timestamps
   like "12 Mar 2024; 3:30PM" that appear in Figma mockups.

5. Structural equivalents: A Figma element "missing" that actually HAS a
   functional equivalent in web (just different label). Example: Figma has
   "Waggon / Trailer No." filter, Web has "IRN No." in the same position.
   NOTE: Reclassify these as "field_substitution" at INFO severity, not missing errors.

 Keep if:
- A real UI button is genuinely absent (e.g., "Create ASN" button)
- A table column header differs or is missing
- A filter field has the wrong interaction type
- Typography or spacing genuinely differs (confirmed by visual inspection)


═══════════════════════════════════════════════════════════════════
STEP 2: VISUAL-ONLY ANALYSIS (what code cannot catch)
═══════════════════════════════════════════════════════════════════

Scan both screenshots region by region. For each region, compare STRUCTURE
not data. Report ONLY issues not already in the automated JSON.

 2A — Page header region (top bar)
- [ ] Logo present and aligned?
- [ ] Primary action button (e.g., "+ Create ASN") present? Position correct?
- [ ] Icon buttons (settings, download/export) present and aligned?
- [ ] Profile avatar/initials: vertically centered with header?
- [ ] Notification badge present if shown in Figma?

 2B — Filter bar region
- [ ] Count filter fields in each screenshot. Same number?
- [ ] For EACH field, check the trailing icon type:
      🔍 = text search input | ▾ = dropdown/select | 📅 = date picker
      If Figma shows 🔍 but web shows ▾ → wrong interaction type (ERROR)
- [ ] Filter label text overlapping with trailing icons?
- [ ] Trailing icons horizontally aligned with each other?
- [ ] Filter field heights consistent?
- [ ] "Clear All" and "Apply" buttons present and positioned correctly?

 2C — Table structure
- [ ] Column headers: exact text match? (case-sensitive: "PO No." vs "Po No.")
- [ ] Column WIDTH proportions: visually compare relative column widths
- [ ] Row HEIGHT: compare the vertical space of a single data row
- [ ] Row separators (borders/lines) match?
- [ ] ASN number links styled correctly (color, underline)?
- [ ] Actions column: width, alignment, icon type (⋮ three-dot menu?)
- [ ] Status indicators: colored dots, badges matching design?
- [ ] Does the table fill the available viewport height, or is there
      dead whitespace below the last row?

 2D — Pagination region
- [ ] Pagination controls present (page numbers, arrows)?
- [ ] Font family for page numbers matches design spec?
- [ ] Items-per-page selector present?
- [ ] Layout/alignment of pagination components?

 2E — Sidebar
- [ ] Same icon set? Same active state indicator?
- [ ] Icon alignment and spacing?

 2F — Cross-cutting concerns
- [ ] Scroll behavior: if content overflows, does the table expand to fill
      viewport height as designed?
- [ ] Toast/notification placement: below page header (not browser top)?
- [ ] Action menu (three-dot) popover: border-radius, shadow, positioning?
- [ ] Context-dependent states: e.g., "Cancel ASN" disabled for Closed status?


═══════════════════════════════════════════════════════════════════
STEP 3: SEVERITY RULES
═══════════════════════════════════════════════════════════════════

error (must fix before release):
- Primary action button missing entirely
- Table column missing from design
- Filter field has wrong interaction type (dropdown vs search)
- Text content functionally incorrect ("Po" vs "PO" — changes meaning)

warning (should fix):
- Column width proportions off by >15%
- Row height differs by >8px
- Font weight visibly different
- Spacing gaps off by >10px
- Icon alignment issues
- Filter text overlapping icons
- Missing secondary icons (download, export)
- Profile element alignment
- Table not filling viewport height
- Font family wrong in any area

info (nice to have):
- Border-radius <4px difference
- Subtle color shade variation
- Minor spacing <10px
- Field name substitution (deliberate rename)


═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT — respond with ONLY valid JSON, no markdown fences
═══════════════════════════════════════════════════════════════════

{
  "false_positives_removed": [
    {
      "element": "element name from report",
      "text": "text value",
      "reason": "sample_data | placeholder_label | dynamic_content | structural_equivalent",
      "explanation": "Why this is false positive (1 sentence)"
    }
  ],

  "severity_adjustments": [
    {
      "element": "element name",
      "text": "text value",
      "original_severity": "error",
      "new_severity": "info",
      "reason": "Why severity changed"
    }
  ],

  "new_visual_differences": [
    {
      "element": "Descriptive name",
      "text": "Visible text or null",
      "sub_type": "missing | interaction_type | alignment | spacing | typography | layout | icon | viewport | styling",
      "figma_value": "What it should be",
      "web_value": "What it actually is (or null)",
      "delta": "Human-readable difference",
      "severity": "error | warning | info",
      "category": "page_header | filter_fields | table_structure | pagination | sidebar | layout | components",
      "zone": "page_header | filter_bar | table_header | table_body | pagination | sidebar",
      "dev_action": "SPECIFIC instruction: CSS property + value, or component to add/change"
    }
  ],

  "field_mapping": {
    "figma_filters": ["Ordered list of Figma filter field labels"],
    "web_filters": ["Ordered list of web filter field labels"],
    "figma_columns": ["Ordered list of Figma table column headers"],
    "web_columns": ["Ordered list of web table column headers"]
  },

  "visual_summary": "3-5 sentence assessment. Focus on the MOST IMPACTFUL structural and layout differences. Do NOT mention sample data mismatches."
}


═══════════════════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════════════════

1. NEVER flag table row cell values as differences. Dynamic data ≠ design bug.

2. Every new finding MUST include "dev_action" with a SPECIFIC fix instruction.
   Bad: "Fix the alignment"
   Good: "Set vertical-align: middle on .profile-icon within .page-header"

3. Be GENEROUS removing false positives, STRICT adding new findings.
   False positives waste developer time. Missing a minor spacing issue doesn't.

4. Compare REGION BY REGION, don't jump around. Header → Filters → Table → Pagination.

5. If the automated report already covers an issue, DO NOT duplicate it.
   Your job is to ADD what code missed and REMOVE what code got wrong.

6. NEVER flag elements that exist in web but NOT in Figma. Those are
   implementation additions, not design bugs. Figma = source of truth.

7. For filter field interaction types, you MUST visually check the trailing icon:
   - Magnifying glass (🔍) = text search input
   - Down chevron (▾) = dropdown/select
   - Calendar (📅) = date picker
   If the icon type doesn't match between Figma and web, that's an ERROR.
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
                    max_output_tokens=32768,  # Increased for longer responses
                    response_mime_type="application/json",  # Force JSON output
                ),
            )
        except Exception as e:
            logger.warning("generate_content_async with response_mime_type failed: %s, retrying without", e)
            response = await self.model.generate_content_async(
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=32768,
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
                    max_output_tokens=32768,  # Increased for longer responses
                    response_mime_type="application/json",  # Force JSON output
                ),
            )
        except Exception as e:
            logger.warning("generate_content with response_mime_type failed: %s, retrying without", e)
            response = self.model.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=32768,
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

        # Handle both schema variations: validated_removals OR false_positives_removed
        removal_list = gemini.get("validated_removals", []) or gemini.get("false_positives_removed", [])
        removals = {
            (r.get("element"), r.get("text"))
            for r in removal_list
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

        # Append new differences from Gemini (handle both schema variations)
        new_diffs = gemini.get("new_differences", []) or gemini.get("new_visual_differences", [])
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

        # Add Gemini metadata (without the full list of removed items)
        merged["gemini_refinement"] = {
            "model": self.model_name,
            "false_positives_removed": removed_count,
            "severity_adjustments": adjusted_count,
            "new_issues_found": len(new_diffs),
            "visual_summary": gemini.get("visual_summary", ""),
        }

        return merged