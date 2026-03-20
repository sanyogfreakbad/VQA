#!/usr/bin/env python3
"""
Figma Design Extractor

Fetches design data from Figma's REST API, filters specific node types,
normalizes bounding box data, and outputs clean JSON.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import requests
from dotenv import load_dotenv

ALLOWED_NODE_TYPES = {"FRAME", "TEXT", "RECTANGLE", "COMPONENT"}


def fetch_figma_file(file_key: str, token: str, node_id: str | None = None) -> dict:
    """Fetch a Figma file or specific node via the REST API."""
    if node_id:
        url = f"https://api.figma.com/v1/files/{file_key}/nodes?ids={node_id}"
    else:
        url = f"https://api.figma.com/v1/files/{file_key}"
    headers = {"X-Figma-Token": token}
    
    try:
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise SystemExit("Error: Invalid Figma API token. Check your FIGMA_TOKEN.")
        elif e.response.status_code == 404:
            raise SystemExit(f"Error: File not found. Check your file key: {file_key}")
        elif e.response.status_code == 429:
            raise SystemExit("Error: Rate limited by Figma API. Please wait and try again.")
        else:
            raise SystemExit(f"Error: HTTP {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"Error: Network request failed - {e}")


def normalize_bounding_box(node: dict) -> dict[str, float]:
    """Extract and normalize absoluteBoundingBox to simple x, y, width, height."""
    bbox = node.get("absoluteBoundingBox", {})
    return {
        "x": bbox.get("x", 0),
        "y": bbox.get("y", 0),
        "width": bbox.get("width", 0),
        "height": bbox.get("height", 0),
    }


def extract_fills(node: dict) -> list[dict]:
    """Extract fill information from a node."""
    fills = []
    for fill in node.get("fills", []):
        fill_data = {"type": fill.get("type")}
        if fill.get("type") == "SOLID" and "color" in fill:
            color = fill["color"]
            fill_data["color"] = {
                "r": round(color.get("r", 0) * 255),
                "g": round(color.get("g", 0) * 255),
                "b": round(color.get("b", 0) * 255),
                "a": color.get("a", 1),
            }
        elif fill.get("type") in ("GRADIENT_LINEAR", "GRADIENT_RADIAL"):
            fill_data["gradientStops"] = fill.get("gradientStops", [])
        fills.append(fill_data)
    return fills


def extract_strokes(node: dict) -> list[dict]:
    """Extract stroke information from a node."""
    strokes = []
    for stroke in node.get("strokes", []):
        stroke_data = {"type": stroke.get("type")}
        if stroke.get("type") == "SOLID" and "color" in stroke:
            color = stroke["color"]
            stroke_data["color"] = {
                "r": round(color.get("r", 0) * 255),
                "g": round(color.get("g", 0) * 255),
                "b": round(color.get("b", 0) * 255),
                "a": color.get("a", 1),
            }
        strokes.append(stroke_data)
    return strokes


def normalize_text_node(node: dict) -> dict[str, Any]:
    """Extract TEXT-specific properties."""
    style = node.get("style", {})
    return {
        "characters": node.get("characters", ""),
        "fontSize": style.get("fontSize"),
        "fontFamily": style.get("fontFamily"),
        "fontWeight": style.get("fontWeight"),
        "textAlignHorizontal": style.get("textAlignHorizontal"),
        "textAlignVertical": style.get("textAlignVertical"),
        "letterSpacing": style.get("letterSpacing"),
        "lineHeightPx": style.get("lineHeightPx"),
        "fills": extract_fills(node),
    }


def normalize_frame_node(node: dict) -> dict[str, Any]:
    """Extract FRAME-specific properties."""
    return {
        "fills": extract_fills(node),
        "strokes": extract_strokes(node),
        "strokeWeight": node.get("strokeWeight"),
        "cornerRadius": node.get("cornerRadius"),
        "paddingLeft": node.get("paddingLeft"),
        "paddingRight": node.get("paddingRight"),
        "paddingTop": node.get("paddingTop"),
        "paddingBottom": node.get("paddingBottom"),
        "itemSpacing": node.get("itemSpacing"),
        "layoutMode": node.get("layoutMode"),
    }


def normalize_rectangle_node(node: dict) -> dict[str, Any]:
    """Extract RECTANGLE-specific properties."""
    return {
        "fills": extract_fills(node),
        "strokes": extract_strokes(node),
        "strokeWeight": node.get("strokeWeight"),
        "cornerRadius": node.get("cornerRadius"),
        "rectangleCornerRadii": node.get("rectangleCornerRadii"),
    }


def normalize_component_node(node: dict) -> dict[str, Any]:
    """Extract COMPONENT-specific properties."""
    return {
        "description": node.get("description", ""),
        "fills": extract_fills(node),
        "strokes": extract_strokes(node),
        "cornerRadius": node.get("cornerRadius"),
    }


def normalize_node(node: dict) -> dict[str, Any]:
    """Normalize a single node based on its type."""
    node_type = node.get("type")
    
    normalized = {
        "id": node.get("id"),
        "name": node.get("name"),
        "type": node_type,
        **normalize_bounding_box(node),
    }
    
    if node_type == "TEXT":
        normalized.update(normalize_text_node(node))
    elif node_type == "FRAME":
        normalized.update(normalize_frame_node(node))
    elif node_type == "RECTANGLE":
        normalized.update(normalize_rectangle_node(node))
    elif node_type == "COMPONENT":
        normalized.update(normalize_component_node(node))
    
    return normalized


def filter_and_flatten_nodes(node: dict, results: list[dict]) -> None:
    """Recursively traverse the node tree, filtering and flattening allowed types."""
    node_type = node.get("type")
    
    if node_type in ALLOWED_NODE_TYPES:
        results.append(normalize_node(node))
    
    for child in node.get("children", []):
        filter_and_flatten_nodes(child, results)


def extract_design_data(figma_data: dict, file_key: str, node_id: str | None = None) -> dict:
    """Extract and structure design data from raw Figma API response."""
    nodes = []
    
    if node_id and "nodes" in figma_data:
        # Response from /files/{key}/nodes endpoint
        for nid, node_data in figma_data.get("nodes", {}).items():
            if node_data and "document" in node_data:
                filter_and_flatten_nodes(node_data["document"], nodes)
    else:
        # Response from /files/{key} endpoint
        document = figma_data.get("document", {})
        filter_and_flatten_nodes(document, nodes)
    
    return {
        "file_name": figma_data.get("name", "Unknown"),
        "file_key": file_key,
        "node_id": node_id,
        "last_modified": figma_data.get("lastModified"),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "version": figma_data.get("version"),
        "total_nodes_extracted": len(nodes),
        "nodes": nodes,
    }


def get_figma_token() -> str:
    """Get Figma API token from environment or .env file."""
    load_dotenv()
    token = os.getenv("FIGMA_TOKEN")
    
    if not token:
        raise SystemExit(
            "Error: FIGMA_TOKEN not found.\n"
            "Set it via environment variable or create a .env file.\n"
            "Get your token at: https://www.figma.com/developers/api#access-tokens"
        )
    
    return token


def main():
    parser = argparse.ArgumentParser(
        description="Extract design values from a Figma file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ABC123xyz
  %(prog)s ABC123xyz --output design.json
  %(prog)s ABC123xyz --token YOUR_TOKEN

Get your file key from the Figma URL:
  https://www.figma.com/file/ABC123xyz/... -> file_key is ABC123xyz
        """,
    )
    
    parser.add_argument(
        "file_key",
        help="Figma file key (from the file URL)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file path (default: prints to stdout)",
    )
    parser.add_argument(
        "-t", "--token",
        help="Figma API token (overrides FIGMA_TOKEN env var)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: True)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Output compact JSON (no indentation)",
    )
    
    args = parser.parse_args()
    
    token = args.token or get_figma_token()
    
    print(f"Fetching Figma file: {args.file_key}...", file=sys.stderr)
    figma_data = fetch_figma_file(args.file_key, token)
    
    print("Extracting and normalizing design data...", file=sys.stderr)
    extracted = extract_design_data(figma_data, args.file_key)
    
    indent = None if args.compact else 2
    json_output = json.dumps(extracted, indent=indent, ensure_ascii=False)
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_output)
        print(f"Saved to: {args.output}", file=sys.stderr)
        print(f"Extracted {extracted['total_nodes_extracted']} nodes.", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
