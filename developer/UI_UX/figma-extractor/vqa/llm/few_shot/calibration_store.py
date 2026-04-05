"""
Few-shot calibration library.

This is the long-term accuracy multiplier. Over time, as human reviewers
confirm or reject findings, those become training examples that calibrate
the LLM's judgment for YOUR specific codebase.

Structure:
- Each example has: crop_pair (base64), ground_truth (confirmed/rejected),
  category, reasoning
- Store in JSON file, load relevant examples per category
- Include 2-3 examples per category in refinement prompts

Usage:
    store = CalibrationStore()
    
    # Add a confirmed example (human said it's a real issue)
    store.add_example(
        category="shadow",
        figma_crop=figma_img,
        web_crop=web_img,
        ground_truth="confirmed",
        reasoning="Shadow is clearly visible in Figma but missing in web"
    )
    
    # Get examples for a category
    examples = store.get_examples_for_category("shadow", max_examples=4)
    
    # Build few-shot block for prompt
    few_shot_text = store.build_few_shot_block("shadow")
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from PIL import Image

from ...vision.image_utils import image_to_base64, base64_to_image

logger = logging.getLogger(__name__)


class CalibrationStore:
    """Manage few-shot calibration examples for LLM accuracy improvement."""
    
    DEFAULT_STORE_PATH = "vqa/llm/few_shot/examples.json"
    
    def __init__(
        self,
        store_path: Optional[str] = None,
        max_examples_per_category: int = 20,
    ):
        """Initialize the calibration store.
        
        Args:
            store_path: Path to JSON file storing examples
            max_examples_per_category: Maximum examples to keep per category
        """
        self.store_path = Path(store_path or self.DEFAULT_STORE_PATH)
        self.max_examples_per_category = max_examples_per_category
        self.examples = self._load()
    
    def _load(self) -> List[Dict]:
        """Load examples from disk."""
        if not self.store_path.exists():
            # Initialize empty store
            self._save([])
            return []
        
        try:
            with open(self.store_path, 'r') as f:
                data = json.load(f)
                # Handle both formats: list of examples or dict with 'examples' key
                if isinstance(data, dict):
                    return data.get("examples", [])
                return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load calibration store: {e}")
            return []
    
    def _save(self, examples: List[Dict] = None) -> None:
        """Save examples to disk."""
        if examples is None:
            examples = self.examples
        
        # Ensure directory exists
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.store_path, 'w') as f:
                json.dump({
                    "version": "1.0",
                    "updated_at": datetime.now().isoformat(),
                    "total_examples": len(examples),
                    "examples": examples,
                }, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save calibration store: {e}")
    
    def add_example(
        self,
        category: str,
        figma_crop: Image.Image,
        web_crop: Image.Image,
        ground_truth: str,
        reasoning: str,
        diff_description: str = "",
        metadata: Dict = None,
    ) -> str:
        """Add a confirmed example to the calibration library.
        
        Call this when a human reviewer confirms or rejects a finding.
        Over time, this builds a codebase-specific knowledge base.
        
        Args:
            category: Category of the finding (shadow, typography, etc.)
            figma_crop: Figma crop image
            web_crop: Web crop image
            ground_truth: "confirmed" (real issue) or "rejected" (false positive)
            reasoning: Human's reasoning for the decision
            diff_description: Description of what was different
            metadata: Optional additional metadata
        
        Returns:
            ID of the added example
        """
        # Validate ground truth
        if ground_truth not in ("confirmed", "rejected"):
            raise ValueError("ground_truth must be 'confirmed' or 'rejected'")
        
        # Encode images
        figma_b64 = image_to_base64(figma_crop)
        web_b64 = image_to_base64(web_crop)
        
        # Generate ID
        import hashlib
        content_hash = hashlib.sha256(
            f"{category}{ground_truth}{reasoning}".encode()
        ).hexdigest()[:12]
        example_id = f"{category}_{ground_truth[:1]}_{content_hash}"
        
        example = {
            "id": example_id,
            "category": category.lower(),
            "ground_truth": ground_truth,
            "reasoning": reasoning,
            "diff_description": diff_description,
            "figma_crop_b64": figma_b64,
            "web_crop_b64": web_b64,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
        self.examples.append(example)
        
        # Prune old examples if we have too many for this category
        self._prune_category(category)
        
        # Save to disk
        self._save()
        
        logger.info(f"Added calibration example: {example_id}")
        return example_id
    
    def add_example_from_finding(
        self,
        finding: Any,
        figma_crop: Image.Image,
        web_crop: Image.Image,
        is_confirmed: bool,
        reasoning: str,
    ) -> str:
        """Add example directly from a Finding object.
        
        Convenience method for integration with the pipeline.
        
        Args:
            finding: Finding object from the pipeline
            figma_crop: Figma crop for this finding
            web_crop: Web crop for this finding
            is_confirmed: True if this is a real issue, False if false positive
            reasoning: Human's reasoning
        
        Returns:
            ID of the added example
        """
        from ...models.finding import Finding
        
        if isinstance(finding, Finding):
            category = finding.category.value if hasattr(finding.category, 'value') else str(finding.category)
            diff_description = (
                f"{finding.diff_type.value if hasattr(finding.diff_type, 'value') else finding.diff_type}: "
                f"Figma={finding.dom_evidence.figma_value if finding.dom_evidence else 'N/A'}, "
                f"Web={finding.dom_evidence.web_value if finding.dom_evidence else 'N/A'}"
            )
            metadata = {
                "finding_id": finding.id,
                "source": finding.source,
                "pass_name": finding.pass_name,
            }
        else:
            # Handle dict-like finding
            category = finding.get("category", "other")
            diff_description = finding.get("diff_description", "")
            metadata = {"finding_id": finding.get("id")}
        
        return self.add_example(
            category=category,
            figma_crop=figma_crop,
            web_crop=web_crop,
            ground_truth="confirmed" if is_confirmed else "rejected",
            reasoning=reasoning,
            diff_description=diff_description,
            metadata=metadata,
        )
    
    def _prune_category(self, category: str) -> None:
        """Remove oldest examples if category exceeds limit."""
        category_lower = category.lower()
        category_examples = [
            (i, ex) for i, ex in enumerate(self.examples)
            if ex.get("category", "").lower() == category_lower
        ]
        
        if len(category_examples) > self.max_examples_per_category:
            # Sort by created_at (oldest first)
            category_examples.sort(
                key=lambda x: x[1].get("created_at", "")
            )
            
            # Remove oldest ones
            to_remove = len(category_examples) - self.max_examples_per_category
            indices_to_remove = [idx for idx, _ in category_examples[:to_remove]]
            
            self.examples = [
                ex for i, ex in enumerate(self.examples)
                if i not in indices_to_remove
            ]
    
    def get_examples_for_category(
        self,
        category: str,
        max_examples: int = 4,
    ) -> List[Dict]:
        """Get calibrated examples for a specific category.
        
        Returns a balanced mix: ~half confirmed, ~half rejected.
        Prioritizes recent examples (they reflect current codebase state).
        
        Args:
            category: Category to get examples for
            max_examples: Maximum examples to return
        
        Returns:
            List of example dicts
        """
        category_lower = category.lower()
        matching = [
            ex for ex in self.examples
            if ex.get("category", "").lower() == category_lower
        ]
        
        if not matching:
            return []
        
        # Split by ground truth
        confirmed = [ex for ex in matching if ex.get("ground_truth") == "confirmed"]
        rejected = [ex for ex in matching if ex.get("ground_truth") == "rejected"]
        
        # Sort each by created_at (most recent first)
        confirmed.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        rejected.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        # Balance: aim for half confirmed, half rejected
        half = max_examples // 2
        result = []
        
        # Add confirmed examples
        result.extend(confirmed[:half])
        
        # Add rejected examples
        result.extend(rejected[:half])
        
        # If we have room, add more from whichever has more
        remaining = max_examples - len(result)
        if remaining > 0:
            all_remaining = (confirmed[half:] + rejected[half:])
            all_remaining.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            result.extend(all_remaining[:remaining])
        
        return result[:max_examples]
    
    def get_all_categories(self) -> List[str]:
        """Get list of all categories with examples."""
        categories = set()
        for ex in self.examples:
            cat = ex.get("category", "")
            if cat:
                categories.add(cat.lower())
        return sorted(categories)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the calibration store."""
        if not self.examples:
            return {
                "total_examples": 0,
                "categories": {},
                "confirmed_count": 0,
                "rejected_count": 0,
            }
        
        categories = {}
        confirmed_total = 0
        rejected_total = 0
        
        for ex in self.examples:
            cat = ex.get("category", "other").lower()
            gt = ex.get("ground_truth", "")
            
            if cat not in categories:
                categories[cat] = {"confirmed": 0, "rejected": 0}
            
            if gt == "confirmed":
                categories[cat]["confirmed"] += 1
                confirmed_total += 1
            elif gt == "rejected":
                categories[cat]["rejected"] += 1
                rejected_total += 1
        
        return {
            "total_examples": len(self.examples),
            "categories": categories,
            "confirmed_count": confirmed_total,
            "rejected_count": rejected_total,
            "category_count": len(categories),
        }
    
    def build_few_shot_block(
        self,
        category: str,
        max_examples: int = 4,
        include_images: bool = False,
    ) -> str:
        """Build the few-shot examples text block for insertion into prompts.
        
        Args:
            category: Category to get examples for
            max_examples: Maximum examples to include
            include_images: Whether to note that images should be included
        
        Returns:
            Formatted text block for prompt
        """
        examples = self.get_examples_for_category(category, max_examples)
        
        if not examples:
            return "No calibration examples available for this category yet."
        
        blocks = []
        for i, ex in enumerate(examples):
            gt = ex.get("ground_truth", "")
            label = "REAL DIFFERENCE" if gt == "confirmed" else "FALSE POSITIVE"
            
            block = f"""EXAMPLE {i + 1} ({label}):
Category: {ex.get('category', 'unknown')}
Difference: {ex.get('diff_description', 'N/A')}
[Figma crop] [Web crop]
Verdict: {gt}
Reasoning: {ex.get('reasoning', 'No reasoning provided')}
"""
            blocks.append(block)
        
        header = f"=== CALIBRATION EXAMPLES FOR '{category.upper()}' CATEGORY ===\n\n"
        footer = "\n=== END OF CALIBRATION EXAMPLES ===\n"
        
        return header + "\n".join(blocks) + footer
    
    def get_example_images(
        self,
        example_id: str,
    ) -> Optional[tuple]:
        """Get the images for a specific example.
        
        Args:
            example_id: ID of the example
        
        Returns:
            Tuple of (figma_crop, web_crop) PIL Images, or None if not found
        """
        for ex in self.examples:
            if ex.get("id") == example_id:
                try:
                    figma_crop = base64_to_image(ex["figma_crop_b64"])
                    web_crop = base64_to_image(ex["web_crop_b64"])
                    return (figma_crop, web_crop)
                except Exception as e:
                    logger.error(f"Failed to decode images for {example_id}: {e}")
                    return None
        return None
    
    def delete_example(self, example_id: str) -> bool:
        """Delete an example by ID.
        
        Args:
            example_id: ID of the example to delete
        
        Returns:
            True if deleted, False if not found
        """
        for i, ex in enumerate(self.examples):
            if ex.get("id") == example_id:
                del self.examples[i]
                self._save()
                return True
        return False
    
    def clear_category(self, category: str) -> int:
        """Delete all examples for a category.
        
        Args:
            category: Category to clear
        
        Returns:
            Number of examples deleted
        """
        category_lower = category.lower()
        original_count = len(self.examples)
        
        self.examples = [
            ex for ex in self.examples
            if ex.get("category", "").lower() != category_lower
        ]
        
        deleted = original_count - len(self.examples)
        if deleted > 0:
            self._save()
        
        return deleted
    
    def export_to_file(self, path: str) -> None:
        """Export all examples to a file.
        
        Args:
            path: Output file path
        """
        with open(path, 'w') as f:
            json.dump({
                "version": "1.0",
                "exported_at": datetime.now().isoformat(),
                "total_examples": len(self.examples),
                "examples": self.examples,
            }, f, indent=2)
    
    def import_from_file(
        self,
        path: str,
        merge: bool = True,
    ) -> int:
        """Import examples from a file.
        
        Args:
            path: Input file path
            merge: If True, merge with existing. If False, replace.
        
        Returns:
            Number of examples imported
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        imported = data.get("examples", [])
        
        if merge:
            # Add imported examples, avoiding duplicates by ID
            existing_ids = {ex.get("id") for ex in self.examples}
            new_examples = [
                ex for ex in imported
                if ex.get("id") not in existing_ids
            ]
            self.examples.extend(new_examples)
            count = len(new_examples)
        else:
            self.examples = imported
            count = len(imported)
        
        self._save()
        return count


# Global store instance
_global_store: Optional[CalibrationStore] = None


def get_calibration_store(store_path: str = None) -> CalibrationStore:
    """Get the global calibration store instance.
    
    Args:
        store_path: Optional path to override default
    
    Returns:
        CalibrationStore instance
    """
    global _global_store
    
    if _global_store is None or store_path:
        _global_store = CalibrationStore(store_path=store_path)
    
    return _global_store
