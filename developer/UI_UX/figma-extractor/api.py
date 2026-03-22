#!/usr/bin/env python3
"""
Design Extractor API

REST API for extracting design data from Figma files and live web pages.
Enables automated visual comparison between design and implementation.
"""

import os
import json
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from dotenv import load_dotenv

from figma_extractor import fetch_figma_file, extract_design_data
from web_extractor import extract_from_url
from design_comparator import DesignComparator, DesignDataExtractor

load_dotenv()

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
):
    """
    Extract design data from a Figma file or specific frame.
    
    - **file_key**: The Figma file key from the URL (e.g., ABC123xyz from figma.com/file/ABC123xyz/...)
    - **node_id**: Optional node ID to extract only a specific frame (from node-id= in URL)
    - **token**: Optional Figma API token (uses .env FIGMA_TOKEN if not provided)
    
    Returns normalized JSON with frames, text, rectangles, and components.
    """
    figma_token = get_token(token)
    
    try:
        figma_data = fetch_figma_file(file_key, figma_token, node_id)
        extracted = extract_design_data(figma_data, file_key, node_id)
        extracted["source"] = "figma"
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
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Web extraction failed: {str(e)}",
        )


class CompareRequest(BaseModel):
    """Request body for design comparison."""
    figma_data: Dict = Field(..., description="Figma extraction JSON data")
    web_data: Dict = Field(..., description="Web extraction JSON data")
    tolerance: Optional[Dict] = Field(
        None,
        description="Custom tolerance values: {font_size, spacing, size, ratio, color}",
        examples=[{"font_size": 2, "spacing": 4, "size": 10, "ratio": 0.05, "color": 0.1}],
    )


@app.post("/api/compare")
def compare_designs(request: CompareRequest):
    """
    Compare Figma design with web implementation.
    
    Produces a detailed diff table with columns:
    - **element**: Element name/identifier
    - **text**: Text content (truncated if long)
    - **diff_type**: Category of difference (text_*, spacing_*, color_*, size_*, etc.)
    - **figma_value**: Value in Figma design
    - **web_value**: Value in web implementation
    - **delta**: Computed difference or description
    - **severity**: error/warning/info
    
    **Comparison Categories:**
    - **Text**: font_family, font_size, font_weight, text_color, line_height, letter_spacing
    - **Spacing**: padding (top/right/bottom/left), item_gap, vertical_gap
    - **Color**: background, border (with perceptual distance calculation)
    - **Size**: width, height, width_ratio, height_ratio (relative to container)
    - **Layout**: layout_mode, alignment
    - **Elements**: missing_in_figma, missing_in_web, order differences
    
    **Response Structure:**
    ```json
    {
      "summary": {
        "total_differences": 25,
        "errors": 2,
        "warnings": 15,
        "info": 8,
        "categories": {"text_font_weight": 5, "spacing_padding_top": 3, ...}
      },
      "differences": [
        {"element": "...", "diff_type": "...", "figma_value": "...", "web_value": "...", ...}
      ],
      "by_category": {
        "text": [...],
        "spacing": [...],
        "color": [...],
        "size": [...],
        "layout": [...],
        "elements": [...]
      }
    }
    ```
    """
    try:
        comparator = DesignComparator(request.figma_data, request.web_data)
        
        if request.tolerance:
            comparator.tolerance.update(request.tolerance)
        
        results = comparator.compare_all()
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )


@app.post("/api/compare/files")
async def compare_design_files(
    figma_file: UploadFile = File(..., description="Figma extraction JSON file"),
    web_file: UploadFile = File(..., description="Web extraction JSON file"),
):
    """
    Compare uploaded Figma and Web JSON files.
    
    Accepts two JSON files (from /extract endpoints) and returns comparison results.
    """
    try:
        figma_content = await figma_file.read()
        web_content = await web_file.read()
        
        figma_data = json.loads(figma_content)
        web_data = json.loads(web_content)
        
        comparator = DesignComparator(figma_data, web_data)
        results = comparator.compare_all()
        
        return results
        
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON file: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )


@app.post("/api/compare/report-cards")
def compare_report_cards(request: CompareRequest):
    """
    Compare only report card elements between Figma and Web.
    
    Specialized endpoint for dashboard/report card layouts.
    Matches cards by text content similarity and compares:
    - Card dimensions and ratios
    - Padding values
    - Background and border colors
    - Text styling
    """
    try:
        comparator = DesignComparator(request.figma_data, request.web_data)
        comparator.diffs = []
        comparator._compare_report_cards()
        return comparator._format_results()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )


@app.post("/api/normalize")
def normalize_design_data(data: Dict, source: str = Query("figma", description="Source: 'figma' or 'web'")):
    """
    Normalize design data into intermediate format.
    
    Returns extracted and categorized elements:
    - text_elements: All text nodes with typography properties
    - frame_elements: Container/frame nodes with layout properties
    - button_elements: Button/CTA components
    - report_cards: Report card elements (for dashboard UIs)
    
    Useful for debugging extraction or building custom comparison logic.
    """
    try:
        from dataclasses import asdict
        extractor = DesignDataExtractor(data, source)
        
        return {
            "source": source,
            "text_elements": [asdict(e) for e in extractor.extract_text_elements()],
            "frame_elements": [asdict(e) for e in extractor.extract_frame_elements()],
            "button_elements": [asdict(e) for e in extractor.extract_button_elements()],
            "report_cards": [asdict(e) for e in extractor.extract_report_cards()],
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Normalization failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
