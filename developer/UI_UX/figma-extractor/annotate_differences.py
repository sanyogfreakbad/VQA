"""
Visual QA Annotator – Playwright Script
========================================
Logs into the QA-WMS app, navigates to the target page, and overlays
numbered bounding boxes on every element that has a Figma-vs-Web difference.

Colors are based on difference categories to match the frontend UI.

Usage:
    # As standalone script
    python annotate_differences.py --results results.json
    
    # As module (called from API)
    from annotate_differences import create_annotated_screenshot
    image_bytes = await create_annotated_screenshot(config, comparison_results)
"""

import asyncio
import json
import argparse
import base64
from pathlib import Path
from typing import Optional
from playwright.async_api import async_playwright

from web_extractor import execute_login, execute_post_login_steps


# Category-based colors matching frontend App.css
CATEGORY_COLORS = {
    "text": {"border": "#8b5cf6", "bg": "#8b5cf6", "text": "#FFFFFF"},
    "spacing": {"border": "#ec4899", "bg": "#ec4899", "text": "#FFFFFF"},
    "size": {"border": "#06b6d4", "bg": "#06b6d4", "text": "#FFFFFF"},
    "missing_elements": {"border": "#ef4444", "bg": "#ef4444", "text": "#FFFFFF"},
    "color": {"border": "#10b981", "bg": "#10b981", "text": "#FFFFFF"},
    "components": {"border": "#f97316", "bg": "#f97316", "text": "#FFFFFF"},
    "buttons_cta": {"border": "#0ea5e9", "bg": "#0ea5e9", "text": "#FFFFFF"},
    "padding": {"border": "#f59e0b", "bg": "#f59e0b", "text": "#FFFFFF"},
    "other": {"border": "#f59e0b", "bg": "#f59e0b", "text": "#FFFFFF"},
}

# Default config (can be overridden)
DEFAULT_CONFIG = {
    "url": "https://qa-wms.dpworld.com/asn",
    "login_url": "https://qa-wms.dpworld.com/in-warehouse",
    "credentials": {
        "username": "sunit_1",
        "password": "Test@123",
        "selectors": {
            "submit": "[role='button']:has-text('Sign In')"
        }
    },
    "post_login_steps": [
        {"action": "wait", "duration": 2000},
        {"action": "click", "selector": ".css-ai6why-control", "nth": 0},
        {"action": "click", "text": "DPW CIC CB Enterprises"},
        {"action": "click", "selector": ".css-ai6why-control", "nth": 0},
        {"action": "click", "text": "CIC - CB Warehouse 1"},
        {"action": "click", "test_id": "next"}
    ],
    "viewport": {"width": 1440, "height": 800},
    "output_screenshot": "annotated_screenshot.png",
}


def build_annotations(comparison_results: dict) -> list[dict]:
    """
    Flatten the comparison results into a deduplicated list of annotations.
    Each annotation = one bounding box on the page.

    Supports two input formats:
    1. { "text": [...], "images": [...], ... } - flat category arrays
    2. { "by_category": { "text": [...], ... } } - nested structure

    We deduplicate by (web_node_id + position) so that one element with
    multiple issues still gets ONE box but multiple issue labels.
    """
    annotations = []
    seen_boxes: dict[str, int] = {}

    # Handle both input formats
    if "by_category" in comparison_results:
        categories = comparison_results["by_category"]
    else:
        categories = {k: v for k, v in comparison_results.items() 
                      if isinstance(v, list)}

    for category, items in categories.items():
        if category == "missing_elements":
            continue

        if not isinstance(items, list):
            continue

        for item in items:
            pos = item.get("web_position")
            if not pos:
                continue

            node_id = item.get("web_node_id", "")
            locator = item.get("web_locator", "")
            sub_type = item.get("sub_type", "")
            element_text = item.get("text", item.get("element", ""))
            delta = item.get("delta", "")
            figma_value = item.get("figma_value", "")
            web_value = item.get("web_value", "")

            # Dedup key: same element bounding box
            box_key = f"{node_id}|{pos['x']}|{pos['y']}|{pos['width']}|{pos['height']}"

            # Build descriptive issue label
            if delta:
                issue_label = f"{sub_type}: {delta}"
            else:
                issue_label = f"{sub_type}: Figma={figma_value}, Web={web_value}"

            # Normalize category for color lookup
            normalized_category = category.lower().replace(" ", "_")

            if box_key in seen_boxes:
                idx = seen_boxes[box_key]
                annotations[idx]["issues"].append(issue_label)
            else:
                seen_boxes[box_key] = len(annotations)
                annotations.append({
                    "x": pos["x"],
                    "y": pos["y"],
                    "width": pos["width"],
                    "height": pos["height"],
                    "category": normalized_category,
                    "element": element_text,
                    "locator": locator,
                    "node_id": node_id,
                    "issues": [issue_label],
                })

    return annotations


def get_overlay_js() -> str:
    """Generate the JavaScript for overlay injection with category colors."""
    colors_js = json.dumps(CATEGORY_COLORS)
    return f"""
(annotations) => {{
    // Remove any previous overlay
    document.querySelectorAll('.vqa-overlay').forEach(el => el.remove());

    const COLORS = {colors_js};

    annotations.forEach((ann, idx) => {{
        const num = idx + 1;
        const color = COLORS[ann.category] || COLORS.other;

        // --- Bounding box ---
        const box = document.createElement('div');
        box.className = 'vqa-overlay';
        Object.assign(box.style, {{
            position: 'absolute',
            left:   ann.x + 'px',
            top:    ann.y + 'px',
            width:  ann.width + 'px',
            height: ann.height + 'px',
            border: '2px solid ' + color.border,
            borderRadius: '3px',
            pointerEvents: 'none',
            zIndex: '99999',
            boxSizing: 'border-box',
        }});

        // --- Number badge ---
        const badge = document.createElement('div');
        badge.className = 'vqa-overlay';
        badge.textContent = num;
        Object.assign(badge.style, {{
            position: 'absolute',
            left:   (ann.x - 2) + 'px',
            top:    (ann.y - 20) + 'px',
            minWidth: '20px',
            height: '20px',
            lineHeight: '20px',
            textAlign: 'center',
            fontSize: '11px',
            fontWeight: '700',
            fontFamily: 'system-ui, sans-serif',
            color: color.text,
            background: color.bg,
            borderRadius: '10px',
            padding: '0 5px',
            pointerEvents: 'none',
            zIndex: '100000',
            boxSizing: 'border-box',
        }});

        document.body.appendChild(box);
        document.body.appendChild(badge);
    }});

    return annotations.length;
}}
"""


async def create_annotated_screenshot(
    config: dict,
    comparison_results: dict,
    headless: bool = True
) -> Optional[bytes]:
    """
    Create an annotated screenshot from comparison results.
    
    Args:
        config: Configuration dict with url, login_url, credentials, post_login_steps, viewport
        comparison_results: The comparison results from /api/compare/urls
        headless: Whether to run browser in headless mode
    
    Returns:
        Screenshot bytes (PNG) or None if failed
    """
    annotations = build_annotations(comparison_results)
    print(f"[VQA] {len(annotations)} unique bounding boxes to draw.")

    if len(annotations) == 0:
        print("[VQA] No annotations found in the comparison results.")
        return None

    screenshot_bytes = None

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        context = await browser.new_context(
            viewport=config.get("viewport", {"width": 1440, "height": 800}),
            ignore_https_errors=True,
        )
        page = await context.new_page()

        try:
            # Step 1: Login if credentials provided
            credentials = config.get("credentials")
            login_url = config.get("login_url")
            
            if credentials and login_url:
                print(f"[VQA] Navigating to login page: {login_url}")
                await page.goto(login_url, wait_until="networkidle")
                
                login_success = await execute_login(
                    page,
                    credentials.get("username", ""),
                    credentials.get("password", ""),
                    credentials.get("selectors"),
                )
                
                if login_success:
                    print("[VQA] Logged in.")
                else:
                    print("[VQA] Warning: Login may have failed.")

                # Step 2: Execute post-login steps (using same logic as web_extractor)
                post_login_steps = config.get("post_login_steps", [])
                if post_login_steps:
                    print("[VQA] Executing post-login steps...")
                    steps_success = await execute_post_login_steps(page, post_login_steps)
                    if steps_success:
                        print("[VQA] Post-login steps completed successfully.")
                    else:
                        print("[VQA] Warning: Some post-login steps may have failed.")

            # Step 3: Navigate to target page
            target_url = config.get("url") or config.get("web_url")
            if target_url:
                print(f"[VQA] Navigating to {target_url}...")
                await page.goto(target_url, wait_until="networkidle")
                await page.wait_for_timeout(2000)

            # Step 4: Resolve positions (use existing or fall back to locators)
            annotations = await _resolve_positions_fallback_to_locators(page, annotations)

            # Step 5: Inject overlay
            overlay_js = get_overlay_js()
            count = await page.evaluate(overlay_js, annotations)
            print(f"[VQA] Injected {count} annotation overlays.")

            # Step 6: Full-page screenshot
            screenshot_bytes = await page.screenshot(full_page=True)
            print("[VQA] Screenshot captured.")

            # Print legend
            _print_legend(annotations, comparison_results)

        except Exception as e:
            print(f"[VQA] Error during annotation: {e}")
            raise
        finally:
            await browser.close()

    return screenshot_bytes


async def _resolve_positions_fallback_to_locators(page, annotations: list[dict]) -> list[dict]:
    """
    Use web_position coordinates from JSON as primary source.
    Only fall back to XPath locators if position is missing or invalid (0,0,0,0).
    """
    for ann in annotations:
        has_valid_position = (
            ann.get("x", 0) > 0 or ann.get("y", 0) > 0
        ) and ann.get("width", 0) > 0 and ann.get("height", 0) > 0
        
        if has_valid_position:
            continue
        
        locator_str = ann.get("locator", "")
        if not locator_str:
            print(f"[VQA] Warning: No valid position or locator for '{ann.get('element', 'unknown')}'")
            continue
            
        try:
            loc = page.locator(f"xpath={locator_str}").first
            if await loc.count() > 0:
                bbox = await loc.bounding_box()
                if bbox:
                    ann["x"] = bbox["x"]
                    ann["y"] = bbox["y"]
                    ann["width"] = bbox["width"]
                    ann["height"] = bbox["height"]
                    print(f"[VQA] Fallback to locator for '{ann.get('element', 'unknown')}'")
        except Exception as e:
            print(f"[VQA] Warning: Could not resolve locator '{locator_str}': {e}")

    return annotations


def _print_legend(annotations: list[dict], comparison_results: dict):
    """Print a numbered legend mapping each box to its issues."""
    if "by_category" in comparison_results:
        missing = comparison_results.get("by_category", {}).get("missing_elements", [])
    else:
        missing = comparison_results.get("missing_elements", [])

    print("\n" + "=" * 60)
    print("  ANNOTATION LEGEND")
    print("=" * 60)

    for idx, ann in enumerate(annotations):
        category_tag = ann["category"].upper()
        print(f"\n  [{idx + 1}] {ann['element']}  ({category_tag})")
        print(f"      Locator : {ann.get('locator', 'N/A')}")
        print(f"      Position: x={ann['x']:.0f}, y={ann['y']:.0f}, "
              f"{ann['width']:.0f}×{ann['height']:.0f}")
        for issue in ann["issues"]:
            print(f"        • {issue}")

    if missing:
        print(f"\n  ── Missing Elements (no overlay, {len(missing)} total) ──")
        for m in missing:
            print(f"    ✗  {m.get('element', '?')} → \"{m.get('text', '')}\"")

    print("\n" + "=" * 60)


# Standalone script entry point
async def run_standalone(comparison_results: dict):
    """Run annotation with default config (for standalone usage)."""
    screenshot_bytes = await create_annotated_screenshot(
        config=DEFAULT_CONFIG,
        comparison_results=comparison_results,
        headless=False
    )
    
    if screenshot_bytes:
        output_path = DEFAULT_CONFIG.get("output_screenshot", "annotated_screenshot.png")
        Path(output_path).write_bytes(screenshot_bytes)
        print(f"[VQA] Screenshot saved → {output_path}")


async def run_preprocessing_test():
    """
    Run just the preprocessing steps (login + post-login) without annotations.
    Useful for testing the login flow with visible browser.
    """
    print("[VQA] Running preprocessing test (login + post-login steps)...")
    
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        context = await browser.new_context(
            viewport=DEFAULT_CONFIG.get("viewport", {"width": 1440, "height": 800}),
            ignore_https_errors=True,
        )
        page = await context.new_page()

        try:
            # Step 1: Login
            credentials = DEFAULT_CONFIG.get("credentials")
            login_url = DEFAULT_CONFIG.get("login_url")
            
            if credentials and login_url:
                print(f"[VQA] Navigating to login page: {login_url}")
                await page.goto(login_url, wait_until="networkidle")
                
                login_success = await execute_login(
                    page,
                    credentials.get("username", ""),
                    credentials.get("password", ""),
                    credentials.get("selectors"),
                )
                
                if login_success:
                    print("[VQA] Logged in.")
                else:
                    print("[VQA] Warning: Login may have failed.")

                # Step 2: Execute post-login steps
                post_login_steps = DEFAULT_CONFIG.get("post_login_steps", [])
                if post_login_steps:
                    print("[VQA] Executing post-login steps...")
                    await execute_post_login_steps(page, post_login_steps)

            # Step 3: Navigate to target page
            target_url = DEFAULT_CONFIG.get("url")
            if target_url:
                print(f"[VQA] Navigating to {target_url}...")
                await page.goto(target_url, wait_until="networkidle")
                await page.wait_for_timeout(2000)

            # Take screenshot
            screenshot_bytes = await page.screenshot(full_page=True)
            output_path = DEFAULT_CONFIG.get("output_screenshot", "annotated_screenshot.png")
            Path(output_path).write_bytes(screenshot_bytes)
            print(f"[VQA] Screenshot saved → {output_path}")
            
            # Keep browser open for inspection
            print("[VQA] Preprocessing complete. Press Enter to close browser...")
            input()

        except Exception as e:
            print(f"[VQA] Error: {e}")
            raise
        finally:
            await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual QA Annotator")
    parser.add_argument(
        "--results", 
        type=str, 
        required=False,
        help="Path to the comparison results JSON file.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run preprocessing test only (login + post-login steps, no annotations).",
    )
    args = parser.parse_args()

    if args.test:
        asyncio.run(run_preprocessing_test())
    elif args.results:
        results_path = Path(args.results)
        if not results_path.exists():
            print(f"[VQA] Error: Results file not found: {results_path}")
            exit(1)
        comparison_results = json.loads(results_path.read_text(encoding="utf-8"))
        asyncio.run(run_standalone(comparison_results))
    else:
        print("[VQA] Error: Either --results or --test is required.")
        parser.print_help()
        exit(1)
