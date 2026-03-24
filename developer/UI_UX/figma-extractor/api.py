#!/usr/bin/env python3
"""
Design Extractor API

REST API for extracting design data from Figma files and live web pages.
Enables automated visual comparison between design and implementation.
"""

import os
import hashlib
import base64
from typing import Optional
from collections import OrderedDict
from threading import Lock

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field, HttpUrl
from dotenv import load_dotenv

from figma_extractor import fetch_figma_file, extract_design_data, fetch_figma_image
from web_extractor import extract_from_url

load_dotenv()

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

app = FastAPI(
    title="Design Extractor API",
    description="""
Extract design values from Figma files and live web pages.

## Features
- **Figma Extraction**: Extract design tokens from Figma files via REST API
- **Web Extraction**: Extract visual properties from rendered web pages using Playwright
- **Unified Schema**: Both sources return compatible JSON for visual comparison

## Use Cases
- Automated design-to-implementation comparison
- Visual regression testing
- Design system auditing
""",
    version="2.0.0",
)


class Credentials(BaseModel):
    """Authentication credentials for web pages requiring login."""
    username: str = Field(..., description="Username or email for login")
    password: str = Field(..., description="Password for login")
    selectors: Optional[dict] = Field(
        None,
        description="Custom CSS selectors for login form: {username, password, submit}",
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
                    "wait_for_selector": ".dashboard-content",
                    "viewport": {"width": 1920, "height": 1080}
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
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Design Extractor API is running",
        "version": "2.0.0",
        "endpoints": {
            "figma": "GET /extract/{file_key}",
            "web": "POST /api/extract",
            "screenshot": "GET /api/extract/{screenshot_id}/image",
            "docs": "/docs",
        },
    }


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
    - **root_selector**: CSS selector to limit extraction scope
    - **max_depth**: Maximum DOM depth to traverse (1-100)
    - **viewport**: Browser viewport dimensions
    - **wait_for_selector**: Wait for element before extracting (for SPAs)
    - **screenshot**: Include base64 screenshot in response
    
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
        
        result = await extract_from_url(
            url=str(request.url),
            credentials=credentials_dict,
            login_url=str(request.login_url) if request.login_url else None,
            root_selector=request.root_selector,
            max_depth=request.max_depth,
            viewport=request.viewport,
            wait_for_selector=request.wait_for_selector,
            screenshot=request.screenshot,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
