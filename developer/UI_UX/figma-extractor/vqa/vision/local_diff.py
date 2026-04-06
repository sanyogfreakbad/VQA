"""Local Visual Diff Engine using CLIP and perceptual hashing.

This module provides local (non-LLM) visual difference detection and classification.
It reduces dependency on expensive LLM calls by handling high-confidence cases locally.

Key features:
- Perceptual hashing for quick similarity detection
- CLIP embeddings for semantic visual similarity
- Zero-shot classification of visual differences
- No training data required - uses pre-trained models

Usage:
    engine = LocalDiffEngine()
    
    # Quick similarity check
    similarity = engine.compute_similarity(img1, img2)
    
    # Classify a diff region
    classification = engine.classify_diff(figma_crop, web_crop)
    if classification.confidence > 0.8:
        # High confidence - no LLM needed
        handle_locally(classification)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy imports for heavy dependencies
_clip_model = None
_imagehash = None


def _get_imagehash():
    """Lazy import imagehash."""
    global _imagehash
    if _imagehash is None:
        import imagehash
        _imagehash = imagehash
    return _imagehash


def _get_clip_model():
    """Lazy load CLIP model."""
    global _clip_model
    if _clip_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading CLIP model (clip-ViT-B-32)...")
            _clip_model = SentenceTransformer('clip-ViT-B-32')
            logger.info("CLIP model loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not installed, CLIP features disabled")
            _clip_model = False
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            _clip_model = False
    return _clip_model if _clip_model else None


class DiffCategory(str, Enum):
    """Categories for visual differences."""
    TEXT = "text"
    COLOR = "color"
    SPACING = "spacing"
    SHADOW = "shadow"
    BORDER = "border"
    SIZE = "size"
    MISSING = "missing"
    IDENTICAL = "identical"
    UNKNOWN = "unknown"


@dataclass
class DiffClassification:
    """Result of local diff classification."""
    category: DiffCategory
    confidence: float  # 0.0 to 1.0
    description: str
    details: Dict[str, Any] = None
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if confidence is high enough to skip LLM."""
        return self.confidence >= 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "confidence": round(self.confidence, 3),
            "description": self.description,
            "details": self.details or {},
            "is_high_confidence": self.is_high_confidence,
        }


@dataclass
class RegionDiff:
    """A detected difference region."""
    x: int
    y: int
    width: int
    height: int
    classification: Optional[DiffClassification] = None
    perceptual_hash_diff: float = 0.0
    ssim_score: float = 1.0


# CLIP zero-shot classification prompts
DIFF_CATEGORY_PROMPTS = {
    DiffCategory.TEXT: [
        "the images have different text content",
        "the text is different between the images",
        "there is a text change",
    ],
    DiffCategory.COLOR: [
        "the images have different colors",
        "the color scheme is different",
        "there is a color change between the images",
    ],
    DiffCategory.SPACING: [
        "the images have different spacing or padding",
        "the layout spacing is different",
        "there is a margin or padding difference",
    ],
    DiffCategory.SHADOW: [
        "the images have different shadows",
        "one image has a shadow that the other doesn't",
        "the shadow effect is different",
    ],
    DiffCategory.BORDER: [
        "the images have different borders",
        "the border style or width is different",
        "there is a border difference",
    ],
    DiffCategory.SIZE: [
        "the elements have different sizes",
        "one element is larger or smaller",
        "there is a size difference",
    ],
    DiffCategory.MISSING: [
        "an element is missing in one image",
        "one image has an element the other doesn't",
        "there is a missing component",
    ],
    DiffCategory.IDENTICAL: [
        "the images are identical",
        "there is no difference between the images",
        "the images look the same",
    ],
}


class LocalDiffEngine:
    """Local visual diff engine using CLIP and perceptual hashing.
    
    This engine provides fast, local classification of visual differences
    without requiring LLM API calls. It uses:
    
    1. Perceptual hashing for quick similarity detection
    2. CLIP embeddings for semantic similarity
    3. Zero-shot classification for diff categorization
    
    The engine is designed to handle high-confidence cases locally,
    passing only uncertain cases to the LLM for validation.
    """
    
    def __init__(self, load_clip: bool = True):
        """Initialize the diff engine.
        
        Args:
            load_clip: Whether to load CLIP model (lazy loading if False)
        """
        self._clip_model = None
        self._category_embeddings = None
        
        if load_clip:
            self._ensure_clip_loaded()
    
    def _ensure_clip_loaded(self) -> bool:
        """Ensure CLIP model is loaded."""
        if self._clip_model is None:
            self._clip_model = _get_clip_model()
            if self._clip_model:
                self._precompute_category_embeddings()
        return self._clip_model is not None
    
    def _precompute_category_embeddings(self):
        """Pre-compute embeddings for category prompts."""
        if not self._clip_model:
            return
        
        self._category_embeddings = {}
        for category, prompts in DIFF_CATEGORY_PROMPTS.items():
            embeddings = self._clip_model.encode(prompts)
            self._category_embeddings[category] = np.mean(embeddings, axis=0)
    
    def compute_perceptual_hash(self, img: Image.Image) -> str:
        """Compute perceptual hash for an image.
        
        Perceptual hashes are robust to minor changes like compression,
        scaling, and small color adjustments.
        
        Args:
            img: PIL Image
        
        Returns:
            Hex string hash
        """
        imagehash = _get_imagehash()
        if not imagehash:
            return ""
        
        # Use average hash for speed, phash for accuracy
        ahash = imagehash.average_hash(img)
        return str(ahash)
    
    def compute_hash_similarity(
        self,
        img1: Image.Image,
        img2: Image.Image,
    ) -> float:
        """Compute similarity based on perceptual hashing.
        
        Args:
            img1: First image
            img2: Second image
        
        Returns:
            Similarity score from 0.0 to 1.0 (1.0 = identical)
        """
        imagehash = _get_imagehash()
        if not imagehash:
            return 0.5
        
        try:
            hash1 = imagehash.average_hash(img1)
            hash2 = imagehash.average_hash(img2)
            
            # Hamming distance
            distance = hash1 - hash2
            max_distance = 64  # 8x8 hash = 64 bits
            
            similarity = 1.0 - (distance / max_distance)
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.warning(f"Hash similarity computation failed: {e}")
            return 0.5
    
    def compute_clip_similarity(
        self,
        img1: Image.Image,
        img2: Image.Image,
    ) -> float:
        """Compute semantic similarity using CLIP embeddings.
        
        CLIP captures higher-level visual semantics, making it better
        at detecting meaningful differences vs. irrelevant pixel changes.
        
        Args:
            img1: First image
            img2: Second image
        
        Returns:
            Cosine similarity from -1.0 to 1.0 (1.0 = identical)
        """
        if not self._ensure_clip_loaded():
            return 0.5
        
        try:
            # Encode images
            emb1 = self._clip_model.encode(img1)
            emb2 = self._clip_model.encode(img2)
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except Exception as e:
            logger.warning(f"CLIP similarity computation failed: {e}")
            return 0.5
    
    def compute_similarity(
        self,
        img1: Image.Image,
        img2: Image.Image,
        use_clip: bool = True,
    ) -> float:
        """Compute overall similarity combining multiple methods.
        
        Args:
            img1: First image
            img2: Second image
            use_clip: Whether to include CLIP similarity
        
        Returns:
            Combined similarity score from 0.0 to 1.0
        """
        hash_sim = self.compute_hash_similarity(img1, img2)
        
        if use_clip and self._ensure_clip_loaded():
            clip_sim = self.compute_clip_similarity(img1, img2)
            # Combine: weight CLIP slightly higher for semantic accuracy
            return 0.4 * hash_sim + 0.6 * ((clip_sim + 1) / 2)  # Normalize CLIP to 0-1
        
        return hash_sim
    
    def classify_diff(
        self,
        figma_crop: Image.Image,
        web_crop: Image.Image,
    ) -> DiffClassification:
        """Classify the type of visual difference between two crops.
        
        Uses CLIP zero-shot classification to determine what kind of
        visual difference exists between the Figma and web crops.
        
        Args:
            figma_crop: Cropped region from Figma screenshot
            web_crop: Cropped region from web screenshot
        
        Returns:
            DiffClassification with category and confidence
        """
        # Quick check with perceptual hash
        hash_sim = self.compute_hash_similarity(figma_crop, web_crop)
        
        if hash_sim > 0.95:
            return DiffClassification(
                category=DiffCategory.IDENTICAL,
                confidence=hash_sim,
                description="Images are nearly identical",
                details={"hash_similarity": hash_sim},
            )
        
        if not self._ensure_clip_loaded():
            # Fallback without CLIP
            return DiffClassification(
                category=DiffCategory.UNKNOWN,
                confidence=0.5,
                description="Unable to classify (CLIP not available)",
                details={"hash_similarity": hash_sim},
            )
        
        try:
            # Create a combined image showing the difference
            # This helps CLIP understand the comparison context
            combined = self._create_comparison_image(figma_crop, web_crop)
            
            # Encode the comparison image
            img_embedding = self._clip_model.encode(combined)
            
            # Compare with category embeddings
            scores = {}
            for category, cat_embedding in self._category_embeddings.items():
                similarity = np.dot(img_embedding, cat_embedding) / (
                    np.linalg.norm(img_embedding) * np.linalg.norm(cat_embedding)
                )
                scores[category] = float(similarity)
            
            # Find best match
            best_category = max(scores, key=scores.get)
            best_score = scores[best_category]
            
            # Normalize confidence (CLIP similarities can be negative)
            confidence = (best_score + 1) / 2  # Map [-1, 1] to [0, 1]
            
            # Boost confidence if there's a clear winner
            sorted_scores = sorted(scores.values(), reverse=True)
            if len(sorted_scores) > 1:
                gap = sorted_scores[0] - sorted_scores[1]
                if gap > 0.1:
                    confidence = min(1.0, confidence + 0.1)
            
            return DiffClassification(
                category=best_category,
                confidence=confidence,
                description=f"Detected {best_category.value} difference",
                details={
                    "hash_similarity": hash_sim,
                    "category_scores": {k.value: round(v, 3) for k, v in scores.items()},
                },
            )
            
        except Exception as e:
            logger.warning(f"CLIP classification failed: {e}")
            return DiffClassification(
                category=DiffCategory.UNKNOWN,
                confidence=0.3,
                description=f"Classification failed: {str(e)}",
                details={"hash_similarity": hash_sim, "error": str(e)},
            )
    
    def _create_comparison_image(
        self,
        img1: Image.Image,
        img2: Image.Image,
    ) -> Image.Image:
        """Create a side-by-side comparison image for CLIP.
        
        Args:
            img1: First image (Figma)
            img2: Second image (web)
        
        Returns:
            Combined image showing both side by side
        """
        # Resize to same height
        target_height = 224  # CLIP input size
        
        w1, h1 = img1.size
        w2, h2 = img2.size
        
        ratio1 = target_height / h1
        ratio2 = target_height / h2
        
        new_w1 = int(w1 * ratio1)
        new_w2 = int(w2 * ratio2)
        
        img1_resized = img1.resize((new_w1, target_height), Image.Resampling.LANCZOS)
        img2_resized = img2.resize((new_w2, target_height), Image.Resampling.LANCZOS)
        
        # Create combined image
        combined_width = new_w1 + new_w2 + 10  # 10px gap
        combined = Image.new('RGB', (combined_width, target_height), (255, 255, 255))
        combined.paste(img1_resized, (0, 0))
        combined.paste(img2_resized, (new_w1 + 10, 0))
        
        return combined
    
    def batch_classify(
        self,
        pairs: List[Tuple[Image.Image, Image.Image]],
    ) -> List[DiffClassification]:
        """Classify multiple image pairs in batch.
        
        More efficient than calling classify_diff multiple times.
        
        Args:
            pairs: List of (figma_crop, web_crop) tuples
        
        Returns:
            List of DiffClassification results
        """
        results = []
        
        for figma_crop, web_crop in pairs:
            result = self.classify_diff(figma_crop, web_crop)
            results.append(result)
        
        return results
    
    def find_diff_regions(
        self,
        figma_img: Image.Image,
        web_img: Image.Image,
        grid_size: int = 64,
        threshold: float = 0.9,
    ) -> List[RegionDiff]:
        """Find regions with visual differences using a grid-based approach.
        
        Divides images into a grid and identifies cells with differences.
        
        Args:
            figma_img: Full Figma screenshot
            web_img: Full web screenshot
            grid_size: Size of each grid cell in pixels
            threshold: Similarity threshold below which to flag as different
        
        Returns:
            List of RegionDiff objects for regions with differences
        """
        # Ensure same size
        if figma_img.size != web_img.size:
            web_img = web_img.resize(figma_img.size, Image.Resampling.LANCZOS)
        
        width, height = figma_img.size
        regions = []
        
        for y in range(0, height, grid_size):
            for x in range(0, width, grid_size):
                # Calculate cell bounds
                cell_w = min(grid_size, width - x)
                cell_h = min(grid_size, height - y)
                
                if cell_w < 16 or cell_h < 16:
                    continue
                
                # Crop cells
                box = (x, y, x + cell_w, y + cell_h)
                figma_cell = figma_img.crop(box)
                web_cell = web_img.crop(box)
                
                # Compute similarity
                similarity = self.compute_hash_similarity(figma_cell, web_cell)
                
                if similarity < threshold:
                    region = RegionDiff(
                        x=x,
                        y=y,
                        width=cell_w,
                        height=cell_h,
                        perceptual_hash_diff=1.0 - similarity,
                    )
                    regions.append(region)
        
        # Merge adjacent regions
        merged = self._merge_adjacent_regions(regions, grid_size)
        
        return merged
    
    def _merge_adjacent_regions(
        self,
        regions: List[RegionDiff],
        tolerance: int = 64,
    ) -> List[RegionDiff]:
        """Merge adjacent or overlapping regions.
        
        Args:
            regions: List of detected regions
            tolerance: Distance threshold for merging
        
        Returns:
            List of merged regions
        """
        if not regions:
            return []
        
        # Sort by position
        regions = sorted(regions, key=lambda r: (r.y, r.x))
        
        merged = []
        current = regions[0]
        
        for region in regions[1:]:
            # Check if regions should be merged
            if (abs(region.x - (current.x + current.width)) <= tolerance and
                abs(region.y - current.y) <= tolerance):
                # Merge
                new_x = min(current.x, region.x)
                new_y = min(current.y, region.y)
                new_right = max(current.x + current.width, region.x + region.width)
                new_bottom = max(current.y + current.height, region.y + region.height)
                
                current = RegionDiff(
                    x=new_x,
                    y=new_y,
                    width=new_right - new_x,
                    height=new_bottom - new_y,
                    perceptual_hash_diff=max(current.perceptual_hash_diff, region.perceptual_hash_diff),
                )
            else:
                merged.append(current)
                current = region
        
        merged.append(current)
        return merged


# Global engine instance (lazy loaded)
_global_engine: Optional[LocalDiffEngine] = None


def get_local_diff_engine(load_clip: bool = False) -> LocalDiffEngine:
    """Get the global LocalDiffEngine instance.
    
    Args:
        load_clip: Whether to eagerly load CLIP model
    
    Returns:
        LocalDiffEngine instance
    """
    global _global_engine
    
    if _global_engine is None:
        _global_engine = LocalDiffEngine(load_clip=load_clip)
    
    return _global_engine
