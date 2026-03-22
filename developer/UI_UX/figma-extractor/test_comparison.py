#!/usr/bin/env python3
"""
Test script for Figma vs Web comparison.

Run this to see a sample comparison between figma.txt and url.txt.
"""

import json
from design_comparator import DesignComparator, DesignDataExtractor


def load_json(filepath: str) -> dict:
    """Load JSON from file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def print_table(differences: list, title: str = "Differences"):
    """Print differences as a formatted table."""
    if not differences:
        print(f"\n{title}: No differences found!")
        return
    
    print(f"\n{'=' * 130}")
    print(f" {title} ({len(differences)} items)")
    print('=' * 130)
    print(f"{'Element':<25} | {'Text':<25} | {'Sub-Type':<15} | {'Figma':<18} | {'Web':<18} | {'Delta':<15} | {'Sev':<8}")
    print("-" * 130)
    
    for diff in differences:
        text = diff.get("text", "") or ""
        if len(text) > 23:
            text = text[:20] + "..."
        
        element = diff.get("element", "")[:23]
        sub_type = diff.get("sub_type", diff.get("diff_type", ""))[:13]
        figma = str(diff.get("figma_value", "-"))[:16]
        web = str(diff.get("web_value", "-"))[:16]
        delta = str(diff.get("delta", ""))[:13]
        sev = diff.get("severity", "info")[:6]
        
        print(f"{element:<25} | {text:<25} | {sub_type:<15} | {figma:<18} | {web:<18} | {delta:<15} | {sev:<8}")


def main():
    print("Loading design data...")
    figma_data = load_json("figma.txt")
    web_data = load_json("url.txt")
    
    print(f"Figma: {figma_data.get('total_nodes_extracted', 0)} nodes")
    print(f"Web: {web_data.get('total_nodes_extracted', 0)} nodes")
    
    print("\nRunning comparison...")
    comparator = DesignComparator(figma_data, web_data)
    results = comparator.compare_all()
    
    print("\n" + "=" * 80)
    print(" COMPARISON SUMMARY")
    print("=" * 80)
    
    summary = results["summary"]
    print(f"Total Differences: {summary['total_differences']}")
    print(f"  - Errors:   {summary['errors']}")
    print(f"  - Warnings: {summary['warnings']}")
    print(f"  - Info:     {summary['info']}")
    
    print("\nBy Category:")
    for cat, count in summary["categories"].items():
        print(f"  - {cat}: {count}")
    
    print_table(results["by_category"].get("text", []), "TEXT (font, size, weight, color)")
    print_table(results["by_category"].get("padding", []), "PADDING")
    print_table(results["by_category"].get("spacing", []), "SPACING (gaps)")
    print_table(results["by_category"].get("color", []), "COLOR (background, border)")
    print_table(results["by_category"].get("size", []), "SIZE (width, height)")
    print_table(results["by_category"].get("buttons_cta", []), "BUTTONS / CTA")
    print_table(results["by_category"].get("components", []), "COMPONENTS (layout, alignment)")
    print_table(results["by_category"].get("missing_elements", []), "MISSING ELEMENTS")
    
    print("\n" + "=" * 80)
    print(" NORMALIZED ELEMENTS PREVIEW")
    print("=" * 80)
    
    figma_extractor = DesignDataExtractor(figma_data, "figma")
    web_extractor = DesignDataExtractor(web_data, "web")
    
    figma_cards = figma_extractor.extract_report_cards()
    web_cards = web_extractor.extract_report_cards()
    
    print(f"\nFigma Report Cards ({len(figma_cards)}):")
    for card in figma_cards[:5]:
        print(f"  - {card.text} | {card.width}x{card.height}px | padding: {card.padding_left}px")
    
    print(f"\nWeb Report Cards ({len(web_cards)}):")
    for card in web_cards[:5]:
        print(f"  - {card.text} | {card.width}x{card.height}px | padding: {card.padding_left}px")
    
    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to comparison_results.json")


if __name__ == "__main__":
    main()
