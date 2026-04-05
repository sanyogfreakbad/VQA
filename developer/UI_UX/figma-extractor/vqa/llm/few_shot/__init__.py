"""VQA LLM Few-Shot - Calibration examples for improved accuracy.

This module manages few-shot calibration examples that improve LLM accuracy:

- CalibrationStore: Persistent storage for calibrated examples
- get_calibration_store: Get the global calibration store instance
- examples.json: Stored examples (true positives and false positives)

Usage:
    from vqa.llm.few_shot import CalibrationStore, get_calibration_store
    
    # Get the global store
    store = get_calibration_store()
    
    # Add an example
    store.add_example(
        category="shadow",
        figma_crop=figma_img,
        web_crop=web_img,
        ground_truth="confirmed",
        reasoning="Shadow clearly visible in Figma but missing in web"
    )
    
    # Get examples for a category
    examples = store.get_examples_for_category("shadow", max_examples=4)
    
    # Build few-shot block for prompts
    few_shot_text = store.build_few_shot_block("shadow")
"""

from .calibration_store import (
    CalibrationStore,
    get_calibration_store,
)

__all__ = [
    "CalibrationStore",
    "get_calibration_store",
]
