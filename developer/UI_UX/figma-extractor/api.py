#!/usr/bin/env python3
"""
Design Extractor API

REST API for extracting design data from Figma files and live web pages.
Enables automated visual comparison between design and implementation.
"""

import asyncio
import os
import re
import hashlib
import base64
from contextlib import asynccontextmanager
from typing import Optional
from collections import OrderedDict
from threading import Lock
from urllib.parse import urlparse, parse_qs

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field, HttpUrl
from dotenv import load_dotenv

import logging

from figma_extractor import fetch_figma_file, extract_design_data, fetch_figma_image
from web_extractor import extract_from_url, BrowserPool, set_browser_pool, get_browser_pool
from design_comparator import DesignComparator
from gemini_refinement import GeminiRefinementLayer
from annotate_differences import create_annotated_screenshot, build_annotations, build_figma_annotations, CATEGORY_COLORS

load_dotenv()

# Configure logging to show Gemini refinement details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Enable debug for gemini_refinement module
logging.getLogger('gemini_refinement').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup/shutdown events.
    
    Initializes the warm browser pool on startup and gracefully shuts it down
    when the application stops.
    """
    # Startup: Initialize browser pool
    pool_size = int(os.getenv("BROWSER_POOL_SIZE", "2"))
    headless = os.getenv("BROWSER_HEADLESS", "false").lower() == "true"
    
    pool = BrowserPool(pool_size=pool_size, headless=headless)
    set_browser_pool(pool)
    
    try:
        await pool.start()
        logger.info(f"Browser pool started: {pool_size} warm instances (headless={headless})")
    except Exception as e:
        logger.error(f"Failed to start browser pool: {e}")
        set_browser_pool(None)
    
    yield
    
    # Shutdown: Clean up browser pool
    if pool and pool.is_started:
        await pool.shutdown()
        logger.info("Browser pool shutdown complete")
    set_browser_pool(None)


# Screenshot storage with LRU-like behavior (max 100 screenshots in memory)
MAX_SCREENSHOT_CACHE = 100
_screenshot_cache: OrderedDict[str, bytes] = OrderedDict()
_cache_lock = Lock()


def store_screenshot(screenshot_data: str | bytes) -> str:
    """Store a screenshot and return its unique hash ID.
    
    Args:
        screenshot_data: Either base64 string or raw bytes
    
    Returns:
        SHA256 hash ID for the screenshot
    """
    if isinstance(screenshot_data, str):
        screenshot_bytes = base64.b64decode(screenshot_data)
    else:
        screenshot_bytes = screenshot_data
    
    screenshot_hash = hashlib.sha256(screenshot_bytes).hexdigest()
    
    with _cache_lock:
        if screenshot_hash in _screenshot_cache:
            _screenshot_cache.move_to_end(screenshot_hash)
        else:
            _screenshot_cache[screenshot_hash] = screenshot_bytes
            while len(_screenshot_cache) > MAX_SCREENSHOT_CACHE:
                _screenshot_cache.popitem(last=False)
    
    return screenshot_hash


def get_screenshot(screenshot_id: str) -> Optional[bytes]:
    """Retrieve a screenshot by its hash ID."""
    with _cache_lock:
        if screenshot_id in _screenshot_cache:
            _screenshot_cache.move_to_end(screenshot_id)
            return _screenshot_cache[screenshot_id]
    return None


def add_serial_numbers(comparison_results: dict) -> dict:
    """
    Add serial numbers to all difference items in the comparison results.
    
    Serial numbers are assigned sequentially (1 to N) across all categories,
    making it easy to reference specific differences in the UI and annotations.
    
    Args:
        comparison_results: The comparison results dict with by_category
        
    Returns:
        Updated comparison results with serial_number field added to each item
    """
    if "by_category" not in comparison_results:
        return comparison_results
    
    serial_number = 0
    by_category = comparison_results["by_category"]
    
    # Process all categories in a consistent order
    ordered_categories = ["text", "spacing", "padding", "color", "buttons_cta", 
                         "components", "size", "other", "missing_elements"]
    
    # First process ordered categories, then any remaining
    all_categories = list(ordered_categories)
    for cat in by_category.keys():
        if cat not in all_categories:
            all_categories.append(cat)
    
    for category in all_categories:
        if category not in by_category:
            continue
            
        items = by_category[category]
        if not isinstance(items, list):
            continue
            
        for item in items:
            serial_number += 1
            item["serial_number"] = serial_number
    
    # Update total in summary
    if "summary" in comparison_results:
        comparison_results["summary"]["total_with_serial"] = serial_number
    
    return comparison_results


def parse_figma_url(figma_url: str) -> dict:
    """
    Parse a Figma URL and extract file_key and node_id.
    
    Supported URL formats:
    - https://www.figma.com/design/FILE_KEY/File-Name?node-id=123-456
    - https://www.figma.com/file/FILE_KEY/File-Name?node-id=123-456
    - https://figma.com/design/FILE_KEY/...
    
    Returns:
        dict with 'file_key' and 'node_id' (node_id may be None)
    
    Raises:
        ValueError if URL is not a valid Figma URL
    """
    parsed = urlparse(figma_url)
    
    # Validate it's a Figma URL
    if not parsed.netloc.endswith("figma.com"):
        raise ValueError("Not a valid Figma URL. Must be from figma.com")
    
    # Extract file_key from path: /design/FILE_KEY/... or /file/FILE_KEY/...
    path_match = re.match(r"^/(design|file)/([a-zA-Z0-9]+)", parsed.path)
    if not path_match:
        raise ValueError("Could not extract file key from Figma URL. Expected format: figma.com/design/FILE_KEY/...")
    
    file_key = path_match.group(2)
    
    # Extract node-id from query parameters
    query_params = parse_qs(parsed.query)
    node_id = query_params.get("node-id", [None])[0]
    
    return {
        "file_key": file_key,
        "node_id": node_id,
    }


app = FastAPI(
    title="Design Extractor API",
    description="""
Extract design values from Figma files and live web pages.

## Features
- **Figma Extraction**: Extract design tokens from Figma files via REST API
- **Web Extraction**: Extract visual properties from rendered web pages using Playwright
- **Unified Schema**: Both sources return compatible JSON for visual comparison
- **Concurrent Extraction**: Figma and Web extraction run in parallel for faster comparison
- **Warm Browser Pool**: Reusable Playwright browser instances for ~25x faster context creation

## Use Cases
- Automated design-to-implementation comparison
- Visual regression testing
- Design system auditing
""",
    version="3.2.0",
    lifespan=lifespan,
)


class FigmaExtractRequest(BaseModel):
    """Request body for Figma URL extraction."""
    url: str = Field(
        ...,
        description="Full Figma URL (e.g., https://www.figma.com/design/FILE_KEY/Name?node-id=123-456)",
    )
    screenshot: bool = Field(
        True,
        description="Include screenshot of the node/file",
    )
    scale: float = Field(
        2,
        ge=0.5,
        le=4,
        description="Screenshot scale factor (0.5-4, default 2 for retina quality)",
    )
    token: Optional[str] = Field(
        None,
        description="Figma API token (optional if set in .env)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "https://www.figma.com/design/YQpur6AxxedY8AVPjhVv6C/Procurement-shared-file?node-id=3345-188011",
                    "screenshot": True,
                    "scale": 2
                }
            ]
        }
    }


class Credentials(BaseModel):
    """Authentication credentials for web pages requiring login."""
    username: str = Field(..., description="Username or email for login")
    password: str = Field(..., description="Password for login")
    selectors: Optional[dict] = Field(
        None,
        description="Custom CSS selectors for login form: {username, password, submit}",
    )


class PostLoginStep(BaseModel):
    """A single post-login action step."""
    action: str = Field(
        ...,
        description="Action type: 'wait_for', 'click', 'select', 'fill', 'wait'",
    )
    selector: Optional[str] = Field(
        None,
        description="CSS selector for the target element",
    )
    text: Optional[str] = Field(
        None,
        description="Text content to match (for click actions)",
    )
    role: Optional[str] = Field(
        None,
        description="ARIA role for element (e.g., 'button', 'link', 'textbox')",
    )
    name: Optional[str] = Field(
        None,
        description="Accessible name for role-based selection (used with 'role')",
    )
    test_id: Optional[str] = Field(
        None,
        description="data-testid attribute value",
    )
    nth: Optional[int] = Field(
        None,
        description="Index when multiple elements match (0-based, used with selector)",
    )
    value: Optional[str] = Field(
        None,
        description="Value for select/fill actions",
    )
    label: Optional[str] = Field(
        None,
        description="Label text for select dropdown",
    )
    index: Optional[int] = Field(
        None,
        description="Index for select dropdown (0-based)",
    )
    duration: Optional[int] = Field(
        None,
        description="Duration in milliseconds (for wait action)",
    )


class WebExtractRequest(BaseModel):
    """Request body for web page extraction."""
    url: HttpUrl = Field(..., description="Target URL to extract visual properties from")
    credentials: Optional[Credentials] = Field(
        None,
        description="Login credentials if the page requires authentication",
    )
    login_url: Optional[HttpUrl] = Field(
        None,
        description="URL of login page if different from target URL",
    )
    post_login_steps: Optional[list[PostLoginStep]] = Field(
        None,
        description="Steps to execute after login (e.g., workspace/organization selection)",
    )
    root_selector: Optional[str] = Field(
        None,
        description="CSS selector to limit extraction scope (default: body)",
    )
    max_depth: int = Field(
        50,
        ge=1,
        le=100,
        description="Maximum DOM traversal depth",
    )
    viewport: Optional[dict] = Field(
        None,
        description="Custom viewport dimensions: {width: int, height: int}",
        examples=[{"width": 1920, "height": 1080}],
    )
    wait_for_selector: Optional[str] = Field(
        None,
        description="Wait for specific element before extraction (useful for SPAs)",
    )
    screenshot: bool = Field(
        False,
        description="Include base64 screenshot in response",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "https://example.com/dashboard",
                    "credentials": {
                        "username": "user@example.com",
                        "password": "password123"
                    },
                    "login_url": "https://example.com/login",
                    "post_login_steps": [
                        {"action": "wait_for", "text": "Please select your work space"},
                        {"action": "select", "selector": "select", "index": 1},
                        {"action": "click", "text": "Next"}
                    ],
                    "wait_for_selector": ".dashboard-content",
                    "viewport": {"width": 1920, "height": 1080}
                }
            ]
        }
    }


class CompareUrlsRequest(BaseModel):
    """Request body for Figma vs Web URL comparison."""
    figma_url: str = Field(
        ...,
        description="Full Figma URL (e.g., https://www.figma.com/design/FILE_KEY/Name?node-id=123-456)",
    )
    web_url: HttpUrl = Field(
        ...,
        description="Target web URL to compare against Figma design",
    )
    login_url: Optional[HttpUrl] = Field(
        None,
        description="URL of login page if different from target URL",
    )
    credentials: Optional[Credentials] = Field(
        None,
        description="Login credentials if the page requires authentication",
    )
    post_login_steps: Optional[list[PostLoginStep]] = Field(
        None,
        description="Steps to execute after login (e.g., workspace/organization selection)",
    )
    wait_for_selector: Optional[str] = Field(
        None,
        description="Wait for specific element before extraction (useful for SPAs)",
    )
    viewport: Optional[dict] = Field(
        None,
        description="Custom viewport dimensions: {width: int, height: int}",
        examples=[{"width": 1440, "height": 800}],
    )
    max_depth: int = Field(
        50,
        ge=1,
        le=100,
        description="Maximum DOM traversal depth",
    )
    root_selector: Optional[str] = Field(
        None,
        description="CSS selector to limit extraction scope (default: body)",
    )
    tolerance: Optional[dict] = Field(
        None,
        description="Custom tolerance values: {font_size, spacing, size, ratio, color}",
    )
    figma_token: Optional[str] = Field(
        None,
        description="Figma API token (optional if set in .env)",
    )

    # ---- Gemini refinement fields ----
    use_gemini: bool = Field(
        True,
        description="Enable Gemini vision refinement layer. "
                    "Sends screenshots + comparison JSON to Gemini for visual validation. "
                    "Uses GEMINI_API_KEY and GEMINI_MODEL from .env",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "figma_url": "https://www.figma.com/design/Cq7YpYswOrCTlR8QvKUqIb/VQA-engine?node-id=1-137455",
                    "web_url": "https://qa-wms.dpworld.com/asn",
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
                    "wait_for_selector": "body",
                    "viewport": {"width": 1440, "height": 800},
                    "max_depth": 50,
                    "use_gemini": True
                }
            ]
        }
    }


class RefineRequest(BaseModel):
    """Standalone request to refine existing comparison results with Gemini."""
    comparison_results: dict = Field(
        ...,
        description="The JSON output from /api/compare/urls",
    )
    figma_screenshot_id: str = Field(
        ...,
        description="Screenshot ID of the Figma design (from extraction endpoint)",
    )
    web_screenshot_id: str = Field(
        ...,
        description="Screenshot ID of the web page (from extraction endpoint)",
    )


class AnnotateRequest(BaseModel):
    """Request body for creating annotated screenshot."""
    comparison_results: dict = Field(
        ...,
        description="The comparison results JSON from /api/compare/urls",
    )
    web_url: HttpUrl = Field(
        ...,
        description="Target web URL to annotate",
    )
    login_url: Optional[HttpUrl] = Field(
        None,
        description="URL of login page if different from target URL",
    )
    credentials: Optional[Credentials] = Field(
        None,
        description="Login credentials if the page requires authentication",
    )
    post_login_steps: Optional[list[PostLoginStep]] = Field(
        None,
        description="Steps to execute after login (e.g., workspace/organization selection)",
    )
    viewport: Optional[dict] = Field(
        {"width": 1440, "height": 800},
        description="Viewport dimensions for the screenshot",
    )
    # Filter parameters for dynamic annotation
    categories: Optional[list[str]] = Field(
        None,
        description="Filter annotations by category (e.g., ['text', 'size', 'spacing']). If None, shows all categories.",
    )
    serial_numbers: Optional[list[int]] = Field(
        None,
        description="Filter annotations by specific serial numbers (e.g., [1, 3, 5, 7]). If None, shows all.",
    )
    include_metadata: bool = Field(
        True,
        description="Include annotation metadata in response for client-side filtering",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "comparison_results": {"by_category": {}},
                    "web_url": "https://qa-wms.dpworld.com/asn",
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
                    "categories": ["text", "size"],
                    "serial_numbers": [1, 3, 5, 7],
                    "include_metadata": True
                }
            ]
        }
    }


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_token(provided_token: Optional[str] = None) -> str:
    """Get Figma token from parameter or environment."""
    token = provided_token or os.getenv("FIGMA_TOKEN")
    if not token:
        raise HTTPException(
            status_code=400,
            detail="FIGMA_TOKEN not configured. Provide token as query parameter or set in .env file.",
        )
    return token


@app.get("/")
async def root():
    """Health check endpoint with browser pool status."""
    pool = get_browser_pool()
    pool_status = await pool.health_check() if pool else {"started": False}
    
    return {
        "status": "ok",
        "message": "Design Extractor API is running",
        "version": "3.2.0",
        "browser_pool": pool_status,
        "endpoints": {
            "figma_url": "POST /api/figma/extract (accepts full Figma URL)",
            "figma": "GET /extract/{file_key}",
            "web": "POST /api/extract",
            "compare": "POST /api/compare/urls (compare Figma vs Web, use_gemini=true for AI refinement)",
            "refine": "POST /api/refine (standalone Gemini refinement with cached screenshots)",
            "annotate": "POST /api/annotate (create annotated screenshot with filters: categories, serial_numbers)",
            "annotate_metadata": "POST /api/annotate/metadata (get annotation metadata for client-side rendering)",
            "screenshot": "GET /api/extract/{screenshot_id}/image",
            "docs": "/docs",
        },
    }


@app.post("/api/figma/extract")
def extract_figma_from_url(request: FigmaExtractRequest):
    """
    Extract design data from a Figma file using the full Figma URL.
    
    Simply paste the Figma URL and the API will automatically extract
    the file key and node ID.
    
    **Example Request:**
    ```json
    {
        "url": "https://www.figma.com/design/YQpur6AxxedY8AVPjhVv6C/Procurement-shared-file?node-id=3345-188011",
        "screenshot": true,
        "scale": 2
    }
    ```
    
    **Supported URL formats:**
    - `https://www.figma.com/design/FILE_KEY/Name?node-id=123-456`
    - `https://www.figma.com/file/FILE_KEY/Name?node-id=123-456`
    
    Returns normalized JSON with frames, text, rectangles, and components.
    If screenshot=true (default), includes screenshot_id and screenshot_url in response.
    """
    # Parse the Figma URL
    try:
        parsed = parse_figma_url(request.url)
        file_key = parsed["file_key"]
        node_id = parsed["node_id"]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    figma_token = get_token(request.token)
    
    try:
        figma_data = fetch_figma_file(file_key, figma_token, node_id)
        extracted = extract_design_data(figma_data, file_key, node_id)
        extracted["source"] = "figma"
        extracted["figma_url"] = request.url
        
        # Fetch and store screenshot if requested
        if request.screenshot:
            target_node_id = node_id
            if not target_node_id:
                document = figma_data.get("document", {})
                children = document.get("children", [])
                if children:
                    target_node_id = children[0].get("id")
            
            if target_node_id:
                image_bytes = fetch_figma_image(file_key, figma_token, target_node_id, scale=request.scale)
                if image_bytes:
                    screenshot_id = store_screenshot(image_bytes)
                    extracted["screenshot_id"] = screenshot_id
                    extracted["screenshot_url"] = f"/api/extract/{screenshot_id}/image"
                else:
                    extracted["screenshot_error"] = "Failed to fetch image from Figma API"
            else:
                extracted["screenshot_error"] = "No node available for screenshot"
        
        return extracted
    except SystemExit as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.post("/api/extract")
async def extract_web_page(request: WebExtractRequest):
    """
    Extract visual properties from a rendered web page.
    
    Launches a Playwright browser session, optionally logs in using provided credentials,
    navigates to the target page, waits for React/SPA rendering to complete, then extracts
    all visual/layout properties from the DOM.
    
    Returns a JSON structure that mirrors the Figma extraction format, enabling automated
    visual comparison between design and implementation.
    
    **Request Body:**
    - **url**: Target URL to extract from
    - **credentials**: Optional login credentials {username, password, selectors}
    - **login_url**: Login page URL if different from target
    - **post_login_steps**: Steps to execute after login (e.g., workspace selection)
    - **root_selector**: CSS selector to limit extraction scope
    - **max_depth**: Maximum DOM depth to traverse (1-100)
    - **viewport**: Browser viewport dimensions
    - **wait_for_selector**: Wait for element before extracting (for SPAs)
    - **screenshot**: Include base64 screenshot in response
    
    **Post-Login Steps Actions:**
    - `wait_for`: Wait for element - {action: "wait_for", selector: ".element"} or {action: "wait_for", text: "Some text"}
    - `click`: Click element - {action: "click", selector: "button"} or {action: "click", text: "Next"}
    - `select`: Select dropdown - {action: "select", selector: "select", index: 1} or {action: "select", selector: "select", label: "Option"}
    - `fill`: Fill input - {action: "fill", selector: "input", value: "text"}
    - `wait`: Pause - {action: "wait", duration: 2000}
    
    **Response Schema:**
    Returns nodes with Figma-compatible properties:
    - Bounding box: x, y, width, height
    - Fills: background colors as {r, g, b, a}
    - Strokes: border colors and weights
    - Text: fontSize, fontFamily, fontWeight, textAlign, etc.
    - Layout: paddingTop/Right/Bottom/Left, layoutMode, itemSpacing
    """
    try:
        credentials_dict = None
        if request.credentials:
            credentials_dict = {
                "username": request.credentials.username,
                "password": request.credentials.password,
                "selectors": request.credentials.selectors,
            }
        
        # Convert post_login_steps to list of dicts
        post_login_steps = None
        if request.post_login_steps:
            post_login_steps = [step.model_dump(exclude_none=True) for step in request.post_login_steps]
        
        result = await extract_from_url(
            url=str(request.url),
            credentials=credentials_dict,
            login_url=str(request.login_url) if request.login_url else None,
            root_selector=request.root_selector,
            max_depth=request.max_depth,
            viewport=request.viewport,
            wait_for_selector=request.wait_for_selector,
            screenshot=request.screenshot,
            post_login_steps=post_login_steps,
        )
        
        # If screenshot was captured, store it and replace base64 with URL
        if result.get("screenshot"):
            screenshot_id = store_screenshot(result["screenshot"])
            result["screenshot_id"] = screenshot_id
            result["screenshot_url"] = f"/api/extract/{screenshot_id}/image"
            del result["screenshot"]
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Web extraction failed: {str(e)}",
        )


@app.get("/extract/{file_key}")
def extract_figma_design(
    file_key: str,
    node_id: Optional[str] = Query(None, description="Specific node/frame ID to extract (e.g., 3345-188011)"),
    token: Optional[str] = Query(None, description="Figma API token (optional if set in .env)"),
    screenshot: bool = Query(False, description="Include screenshot of the node/file"),
    scale: float = Query(2, ge=0.5, le=4, description="Screenshot scale factor (0.5-4)"),
):
    """
    Extract design data from a Figma file or specific frame.
    
    - **file_key**: The Figma file key from the URL (e.g., ABC123xyz from figma.com/file/ABC123xyz/...)
    - **node_id**: Optional node ID to extract only a specific frame (from node-id= in URL)
    - **token**: Optional Figma API token (uses .env FIGMA_TOKEN if not provided)
    - **screenshot**: If true, fetches a rendered image of the node
    - **scale**: Image scale factor for screenshot (default 2 for retina quality)
    
    Returns normalized JSON with frames, text, rectangles, and components.
    If screenshot=true, includes screenshot_id and screenshot_url in response.
    """
    figma_token = get_token(token)
    
    try:
        figma_data = fetch_figma_file(file_key, figma_token, node_id)
        extracted = extract_design_data(figma_data, file_key, node_id)
        extracted["source"] = "figma"
        
        # Fetch and store screenshot if requested
        if screenshot:
            # Use provided node_id or get the first frame from the document
            target_node_id = node_id
            if not target_node_id:
                # Try to get the first canvas/page as default
                document = figma_data.get("document", {})
                children = document.get("children", [])
                if children:
                    target_node_id = children[0].get("id")
            
            if target_node_id:
                image_bytes = fetch_figma_image(file_key, figma_token, target_node_id, scale=scale)
                if image_bytes:
                    screenshot_id = store_screenshot(image_bytes)
                    extracted["screenshot_id"] = screenshot_id
                    extracted["screenshot_url"] = f"/api/extract/{screenshot_id}/image"
                else:
                    extracted["screenshot_error"] = "Failed to fetch image from Figma API"
            else:
                extracted["screenshot_error"] = "No node available for screenshot"
        
        return extracted
    except SystemExit as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.get("/api/extract")
async def extract_web_page_get(
    url: str = Query(..., description="Target URL to extract visual properties from"),
    wait_for: Optional[str] = Query(None, description="CSS selector to wait for before extraction"),
    root: Optional[str] = Query(None, description="CSS selector to limit extraction scope"),
    width: int = Query(1920, description="Viewport width"),
    height: int = Query(1080, description="Viewport height"),
    screenshot: bool = Query(False, description="Include base64 screenshot"),
):
    """
    Extract visual properties from a public web page (no auth).
    
    Simple GET endpoint for quick extraction of public pages.
    For authenticated pages, use POST /api/extract with credentials.
    """
    try:
        result = await extract_from_url(
            url=url,
            credentials=None,
            root_selector=root,
            viewport={"width": width, "height": height},
            wait_for_selector=wait_for,
            screenshot=screenshot,
        )
        
        # If screenshot was captured, store it and replace base64 with URL
        if result.get("screenshot"):
            screenshot_id = store_screenshot(result["screenshot"])
            result["screenshot_id"] = screenshot_id
            result["screenshot_url"] = f"/api/extract/{screenshot_id}/image"
            del result["screenshot"]
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Web extraction failed: {str(e)}",
        )


@app.get("/api/extract/{screenshot_id}/image")
async def get_screenshot_image(screenshot_id: str):
    """
    Retrieve a stored screenshot by its ID.
    
    - **screenshot_id**: The unique hash ID returned from extraction endpoints
    
    Returns the screenshot as a PNG image.
    Screenshots are stored in memory with LRU eviction (max 100 cached).
    """
    screenshot_bytes = get_screenshot(screenshot_id)
    
    if screenshot_bytes is None:
        raise HTTPException(
            status_code=404,
            detail=f"Screenshot not found. It may have been evicted from cache or the ID is invalid.",
        )
    
    return Response(
        content=screenshot_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": f"inline; filename={screenshot_id}.png",
            "Cache-Control": "public, max-age=3600",
        },
    )


async def _extract_figma_async(
    file_key: str,
    figma_token: str,
    node_id: Optional[str],
    figma_url: str,
) -> tuple[dict, Optional[bytes]]:
    """
    Async wrapper for Figma extraction (runs sync code in thread pool).
    
    Returns:
        Tuple of (figma_extracted_data, screenshot_bytes_or_none)
    """
    def _extract_figma_sync():
        figma_data = fetch_figma_file(file_key, figma_token, node_id)
        figma_extracted = extract_design_data(figma_data, file_key, node_id)
        figma_extracted["source"] = "figma"
        figma_extracted["figma_url"] = figma_url
        
        # Get screenshot
        target_node_id = node_id
        if not target_node_id:
            document = figma_data.get("document", {})
            children = document.get("children", [])
            if children:
                target_node_id = children[0].get("id")
        
        screenshot_bytes = None
        if target_node_id:
            screenshot_bytes = fetch_figma_image(file_key, figma_token, target_node_id, scale=2)
        
        return figma_extracted, screenshot_bytes
    
    return await asyncio.to_thread(_extract_figma_sync)


async def _extract_web_async(
    web_url: str,
    credentials: Optional[dict],
    login_url: Optional[str],
    root_selector: Optional[str],
    max_depth: int,
    viewport: Optional[dict],
    wait_for_selector: Optional[str],
    post_login_steps: Optional[list[dict]],
) -> tuple[dict, Optional[bytes]]:
    """
    Async web extraction wrapper.
    
    Returns:
        Tuple of (web_extracted_data, screenshot_bytes_or_none)
    """
    web_extracted = await extract_from_url(
        url=web_url,
        credentials=credentials,
        login_url=login_url,
        root_selector=root_selector,
        max_depth=max_depth,
        viewport=viewport,
        wait_for_selector=wait_for_selector,
        screenshot=True,
        post_login_steps=post_login_steps,
    )
    web_extracted["source"] = "web"
    web_extracted["web_url"] = web_url
    
    # Extract screenshot bytes
    screenshot_bytes = None
    if web_extracted.get("screenshot"):
        screenshot_bytes = base64.b64decode(web_extracted["screenshot"])
        del web_extracted["screenshot"]
    
    return web_extracted, screenshot_bytes


@app.post("/api/compare/urls")
async def compare_figma_web_urls(request: CompareUrlsRequest):
    """
    Compare Figma design with web implementation using URLs.
    
    **Performance**: Figma and Web extraction run concurrently using asyncio.gather(),
    reducing total extraction time to the duration of the slower extraction rather
    than the sum of both.

    Set **use_gemini=true** to enable the Gemini vision refinement layer.
    When enabled, the API will:
    1. Run the normal DOM-based comparison
    2. Capture screenshots of both Figma and web
    3. Send comparison JSON + both screenshots to Gemini
    4. Gemini validates findings, removes false positives, and detects
       visual issues the DOM comparator missed (icons, images, layout)
    5. Return merged results with a `gemini_refinement` metadata block

    New Gemini-found issues are tagged with `"source": "gemini_visual"`.
    """
    # --- Step 1: Parse Figma URL and prepare parameters ---
    try:
        parsed = parse_figma_url(request.figma_url)
        file_key = parsed["file_key"]
        node_id = parsed["node_id"]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    figma_token = get_token(request.figma_token)
    
    # Prepare web extraction parameters
    credentials_dict = None
    if request.credentials:
        credentials_dict = {
            "username": request.credentials.username,
            "password": request.credentials.password,
            "selectors": request.credentials.selectors,
        }

    post_login_steps = None
    if request.post_login_steps:
        post_login_steps = [step.model_dump(exclude_none=True) for step in request.post_login_steps]

    # --- Step 2: Run Figma and Web extraction CONCURRENTLY ---
    logger.info("Starting concurrent Figma + Web extraction...")
    
    figma_task = _extract_figma_async(
        file_key=file_key,
        figma_token=figma_token,
        node_id=node_id,
        figma_url=request.figma_url,
    )
    
    web_task = _extract_web_async(
        web_url=str(request.web_url),
        credentials=credentials_dict,
        login_url=str(request.login_url) if request.login_url else None,
        root_selector=request.root_selector,
        max_depth=request.max_depth,
        viewport=request.viewport,
        wait_for_selector=request.wait_for_selector,
        post_login_steps=post_login_steps,
    )
    
    # Gather results - both run in parallel
    try:
        results = await asyncio.gather(
            figma_task,
            web_task,
            return_exceptions=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    
    # Check for exceptions in results
    figma_result, web_result = results
    
    if isinstance(figma_result, Exception):
        error_msg = str(figma_result)
        if isinstance(figma_result, SystemExit):
            raise HTTPException(status_code=400, detail=f"Figma extraction failed: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Figma extraction failed: {error_msg}")
    
    if isinstance(web_result, Exception):
        raise HTTPException(status_code=500, detail=f"Web extraction failed: {str(web_result)}")
    
    # Unpack results
    figma_extracted, figma_screenshot_bytes = figma_result
    web_extracted, web_screenshot_bytes = web_result
    
    logger.info("Concurrent extraction complete")
    
    # Store screenshots
    if figma_screenshot_bytes:
        figma_ss_id = store_screenshot(figma_screenshot_bytes)
        figma_extracted["screenshot_id"] = figma_ss_id
        figma_extracted["screenshot_url"] = f"/api/extract/{figma_ss_id}/image"
    
    if web_screenshot_bytes:
        web_ss_id = store_screenshot(web_screenshot_bytes)
        web_extracted["screenshot_id"] = web_ss_id
        web_extracted["screenshot_url"] = f"/api/extract/{web_ss_id}/image"

    # --- Step 4: Run DOM-based comparison ---
    try:
        comparator = DesignComparator(figma_extracted, web_extracted)
        if request.tolerance:
            comparator.tolerance.update(request.tolerance)
        results = comparator.compare_all()
        
        # Add serial numbers to all differences for UI display and annotation reference
        results = add_serial_numbers(results)
        
        # Add screenshot IDs to the response
        if figma_extracted.get("screenshot_id"):
            results["figma_screenshot_id"] = figma_extracted["screenshot_id"]
            results["figma_screenshot_url"] = figma_extracted["screenshot_url"]
        if web_extracted.get("screenshot_id"):
            results["web_screenshot_id"] = web_extracted["screenshot_id"]
            results["web_screenshot_url"] = web_extracted["screenshot_url"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

    # --- Step 5: Optional Gemini refinement ---
    if request.use_gemini:
        if not figma_screenshot_bytes:
            results["gemini_refinement"] = {
                "error": "Figma screenshot not available — cannot run visual refinement"
            }
            return results

        if not web_screenshot_bytes:
            results["gemini_refinement"] = {
                "error": "Web screenshot not available — cannot run visual refinement"
            }
            return results

        try:
            gemini = GeminiRefinementLayer()
            results = await gemini.refine_async(
                comparison_results=results,
                figma_screenshot=figma_screenshot_bytes,
                web_screenshot=web_screenshot_bytes,
            )
        except ValueError as e:
            results["gemini_refinement"] = {"error": str(e)}
        except Exception as e:
            logger.exception("Gemini refinement failed")
            results["gemini_refinement"] = {
                "error": f"Gemini refinement failed: {str(e)}",
                "note": "DOM-based comparison results are still valid above.",
            }

    return results


@app.post("/api/refine")
async def refine_with_gemini(request: RefineRequest):
    """
    Standalone endpoint: refine existing comparison results using Gemini.

    Use this when you already have comparison JSON + stored screenshots
    and want to run (or re-run) the Gemini visual refinement separately.

    Requires GEMINI_API_KEY in .env.
    """
    figma_bytes = get_screenshot(request.figma_screenshot_id)
    if not figma_bytes:
        raise HTTPException(status_code=404, detail="Figma screenshot not found in cache")

    web_bytes = get_screenshot(request.web_screenshot_id)
    if not web_bytes:
        raise HTTPException(status_code=404, detail="Web screenshot not found in cache")

    try:
        gemini = GeminiRefinementLayer()
        refined = await gemini.refine_async(
            comparison_results=request.comparison_results,
            figma_screenshot=figma_bytes,
            web_screenshot=web_bytes,
        )
        return refined
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini refinement failed: {str(e)}")


@app.post("/api/annotate")
async def create_annotated_image(request: AnnotateRequest):
    """
    Create an annotated screenshot highlighting differences on the web page.

    Takes comparison results from /api/compare/urls and generates a screenshot
    of the web page with visual annotations (colored bounding boxes) around
    elements that have differences.

    **Dynamic Filtering:**
    - **categories**: Filter by category (e.g., ["text", "size", "spacing"])
    - **serial_numbers**: Filter by specific difference numbers (e.g., [1, 3, 5, 7])
    - **include_metadata**: Return annotation metadata for client-side filtering

    Colors are based on difference categories:
    - Text: Purple (#8b5cf6)
    - Spacing: Pink (#ec4899)
    - Size: Cyan (#06b6d4)
    - Missing: Red (#ef4444)
    - Color: Green (#10b981)
    - Components: Orange (#f97316)
    - Buttons: Sky Blue (#0ea5e9)
    - Other: Amber (#f59e0b)

    Returns the annotated screenshot URL and optionally annotation metadata for
    client-side rendering/filtering.
    """
    try:
        # Build config from request
        config = {
            "url": str(request.web_url),
            "web_url": str(request.web_url),
            "viewport": request.viewport or {"width": 1440, "height": 800},
        }

        if request.login_url:
            config["login_url"] = str(request.login_url)

        if request.credentials:
            config["credentials"] = {
                "username": request.credentials.username,
                "password": request.credentials.password,
                "selectors": request.credentials.selectors,
            }

        if request.post_login_steps:
            config["post_login_steps"] = [
                step.model_dump(exclude_none=True) for step in request.post_login_steps
            ]

        # Create annotated screenshot with optional filters
        screenshot_bytes, annotations = await create_annotated_screenshot(
            config=config,
            comparison_results=request.comparison_results,
            headless=False,
            filter_categories=request.categories,
            filter_serial_numbers=request.serial_numbers
        )

        if screenshot_bytes is None:
            raise HTTPException(
                status_code=400,
                detail="No annotations found in comparison results or all filtered out"
            )

        # Store the screenshot and return URL
        screenshot_id = store_screenshot(screenshot_bytes)

        response = {
            "success": True,
            "annotated_screenshot_id": screenshot_id,
            "annotated_screenshot_url": f"/api/extract/{screenshot_id}/image",
            "annotation_count": len(annotations),
            "message": "Annotated screenshot created successfully"
        }
        
        # Include filters applied in response
        if request.categories:
            response["filters_applied"] = response.get("filters_applied", {})
            response["filters_applied"]["categories"] = request.categories
        if request.serial_numbers:
            response["filters_applied"] = response.get("filters_applied", {})
            response["filters_applied"]["serial_numbers"] = request.serial_numbers
        
        # Include annotation metadata for client-side filtering
        if request.include_metadata:
            response["annotations"] = annotations

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Annotation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create annotated screenshot: {str(e)}"
        )


class AnnotationMetadataRequest(BaseModel):
    """Request body for getting annotation metadata only (no screenshot)."""
    comparison_results: dict = Field(
        ...,
        description="The comparison results JSON from /api/compare/urls",
    )
    categories: Optional[list[str]] = Field(
        None,
        description="Filter annotations by category (e.g., ['text', 'size', 'spacing']). If None, returns all.",
    )
    serial_numbers: Optional[list[int]] = Field(
        None,
        description="Filter annotations by specific serial numbers (e.g., [1, 3, 5, 7]). If None, returns all.",
    )


@app.post("/api/annotate/metadata")
async def get_annotation_metadata(request: AnnotationMetadataRequest):
    """
    Get annotation metadata for client-side rendering without generating a screenshot.
    
    This endpoint returns the bounding box coordinates, categories, and serial numbers
    for all differences, which can be used by the frontend to draw annotation boxes
    dynamically on the web page screenshot.
    
    **Use this for:**
    - Client-side filtering (filter by category or serial number in the UI)
    - Real-time box visibility toggling without server round-trips
    - Custom annotation rendering in the frontend
    
    **Response includes:**
    - annotations: Array of annotation objects with x, y, width, height, category, serial_numbers
    - category_colors: Color mapping for each category (for consistent styling)
    - total_count: Total number of annotations (before filtering)
    - filtered_count: Number of annotations after applying filters
    """
    try:
        annotations, total_count = build_annotations(
            request.comparison_results,
            filter_categories=request.categories,
            filter_serial_numbers=request.serial_numbers
        )
        
        response = {
            "success": True,
            "annotations": annotations,
            "total_count": total_count,
            "filtered_count": len(annotations),
            "category_colors": CATEGORY_COLORS,
        }
        
        if request.categories:
            response["filters_applied"] = response.get("filters_applied", {})
            response["filters_applied"]["categories"] = request.categories
        if request.serial_numbers:
            response["filters_applied"] = response.get("filters_applied", {})
            response["filters_applied"]["serial_numbers"] = request.serial_numbers
            
        return response
        
    except Exception as e:
        logger.exception("Failed to build annotation metadata")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build annotation metadata: {str(e)}"
        )


class FigmaAnnotationMetadataRequest(BaseModel):
    """Request body for getting Figma annotation metadata (missing elements only)."""
    comparison_results: dict = Field(
        ...,
        description="The comparison results JSON from /api/compare/urls",
    )
    serial_numbers: Optional[list[int]] = Field(
        None,
        description="Filter missing element annotations by specific serial numbers (e.g., [32, 35, 37]). If None, returns all.",
    )


@app.post("/api/annotate/figma-metadata")
async def get_figma_annotation_metadata(request: FigmaAnnotationMetadataRequest):
    """
    Get Figma annotation metadata for missing elements (client-side rendering).
    
    Missing elements exist in Figma but not in Web. This endpoint returns the
    bounding box coordinates on the Figma screenshot for these elements.
    
    **Use this for:**
    - Split view comparison showing Figma with missing element boxes
    - Client-side filtering by serial number
    - Dynamic toggling of missing element visibility
    
    **Response includes:**
    - annotations: Array of missing element annotations with figma positions
    - category_colors: Color mapping (missing_elements uses red #ef4444)
    - total_count: Total number of missing elements
    - filtered_count: Number after applying serial_number filter
    """
    try:
        annotations, total_count = build_figma_annotations(
            request.comparison_results,
            filter_serial_numbers=request.serial_numbers
        )
        
        response = {
            "success": True,
            "annotations": annotations,
            "total_count": total_count,
            "filtered_count": len(annotations),
            "category_colors": CATEGORY_COLORS,
        }
        
        if request.serial_numbers:
            response["filters_applied"] = {"serial_numbers": request.serial_numbers}
            
        return response
        
    except Exception as e:
        logger.exception("Failed to build Figma annotation metadata")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build Figma annotation metadata: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)