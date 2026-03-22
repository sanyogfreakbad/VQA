#!/usr/bin/env python3
"""
Figma vs Web Design Comparison API

Standalone FastAPI application for design comparison.
Can be run separately from the main extraction API.
"""

import json
from dataclasses import asdict
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from design_comparator import (
    DesignComparator,
    DesignDataExtractor,
)


app = FastAPI(
    title="Figma vs Web Comparison API",
    description="""
Compare Figma design data with web implementation data.

## Features
- **Text Comparison**: Font family, size, weight, color, line height
- **Spacing Analysis**: Padding, margins, gaps (absolute and relative)
- **Color Matching**: Background, border, text colors with perceptual distance
- **Layout Comparison**: Width/height ratios, alignment, layout mode
- **Component Matching**: Automatic element matching by text similarity

## Usage
Upload both Figma and Web JSON files to get a detailed diff table.
""",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CompareRequest(BaseModel):
    """Request body for comparison endpoint."""
    figma_data: Dict = Field(..., description="Figma extraction JSON")
    web_data: Dict = Field(..., description="Web extraction JSON")
    tolerance: Optional[Dict] = Field(
        None,
        description="Custom tolerance values: {font_size, spacing, size, ratio, color}",
    )


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Figma vs Web Comparison API",
        "version": "1.0.0",
        "endpoints": {
            "compare": "POST /compare",
            "compare_files": "POST /compare/files",
        },
    }


@app.post("/compare")
def compare_designs(request: CompareRequest):
    """
    Compare Figma design data with web implementation.
    
    **Request Body:**
    - `figma_data`: JSON from Figma extraction API
    - `web_data`: JSON from web extraction API
    - `tolerance`: Optional custom tolerance values
    
    **Response:**
    - Summary of differences by category
    - Full difference table with columns:
        - element: Element name/identifier
        - text: Text content (if applicable)
        - diff_type: Type of difference
        - figma_value: Value in Figma
        - web_value: Value in Web
        - delta: Computed difference
        - severity: error/warning/info
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


@app.post("/compare/files")
async def compare_design_files(
    figma_file: UploadFile = File(..., description="Figma extraction JSON file"),
    web_file: UploadFile = File(..., description="Web extraction JSON file"),
):
    """
    Compare uploaded Figma and Web JSON files.
    
    Upload both JSON files to receive a detailed comparison.
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


@app.post("/compare/report-cards")
def compare_report_cards_only(request: CompareRequest):
    """
    Compare only report card elements between Figma and Web.
    
    Specialized endpoint for comparing report/dashboard card layouts.
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


@app.post("/normalize")
def normalize_design_data(data: Dict, source: str = "figma"):
    """
    Normalize design data into intermediate format.
    
    Useful for debugging or custom comparison logic.
    """
    try:
        extractor = DesignDataExtractor(data, source)
        
        return {
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
    uvicorn.run(app, host="0.0.0.0", port=8001)
