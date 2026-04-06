"""Automatic threshold tuning based on user feedback.

This module analyzes collected feedback to suggest threshold adjustments
that reduce false positives while maintaining detection of real issues.

Key concepts:
- False positive rate (FPR): rejected / total for a category
- If FPR > 0.5, the threshold is too sensitive (too many false positives)
- If FPR < 0.1, the threshold may be too loose (missing real issues)

Usage:
    tuner = AutoTuner()
    
    # Get suggestions based on current feedback
    suggestions = tuner.analyze_feedback()
    
    # Apply suggestions
    tuner.apply_suggestions(suggestions)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from ..storage import get_database
from ..config.thresholds import Thresholds, THRESHOLDS

logger = logging.getLogger(__name__)


# Mapping from feedback categories to threshold fields
CATEGORY_TO_THRESHOLD = {
    "text": ["font_size_tolerance"],
    "spacing": ["spacing_tolerance", "gap_tolerance"],
    "padding": ["spacing_tolerance"],
    "color": ["color_tolerance"],
    "size": ["size_tolerance", "ratio_tolerance"],
    "shadow": ["shadow_blur_tolerance", "shadow_spread_tolerance"],
    "border": ["border_width_tolerance"],
}

# Default adjustment factors
INCREASE_FACTOR = 1.2  # Increase threshold by 20% to reduce false positives
DECREASE_FACTOR = 0.9  # Decrease threshold by 10% to catch more issues


@dataclass
class ThresholdSuggestion:
    """A suggested threshold adjustment."""
    category: str
    threshold_name: str
    current_value: float
    suggested_value: float
    reason: str
    confidence: float  # 0-1, based on amount of feedback data
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "threshold_name": self.threshold_name,
            "current_value": round(self.current_value, 3),
            "suggested_value": round(self.suggested_value, 3),
            "change_percent": round((self.suggested_value - self.current_value) / self.current_value * 100, 1),
            "reason": self.reason,
            "confidence": round(self.confidence, 2),
        }


class AutoTuner:
    """Automatic threshold tuner based on feedback analysis.
    
    Analyzes user feedback to identify categories with:
    - High false positive rates (need looser thresholds)
    - High confirmation rates (thresholds are well-calibrated)
    
    Generates suggestions that can be applied to the threshold config.
    """
    
    # Minimum feedback count to make suggestions
    MIN_FEEDBACK_COUNT = 10
    
    # FPR thresholds for triggering adjustments
    HIGH_FPR_THRESHOLD = 0.5   # 50%+ false positives -> increase tolerance
    LOW_FPR_THRESHOLD = 0.1    # <10% false positives -> could decrease tolerance
    
    def __init__(self, thresholds: Thresholds = None):
        """Initialize the auto-tuner.
        
        Args:
            thresholds: Current threshold config (uses defaults if not provided)
        """
        self.thresholds = thresholds or THRESHOLDS
        self._db = None
    
    @property
    def db(self):
        """Get database instance (lazy initialization)."""
        if self._db is None:
            self._db = get_database()
        return self._db
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get current feedback statistics from database."""
        return self.db.get_feedback_stats()
    
    def analyze_feedback(self) -> List[ThresholdSuggestion]:
        """Analyze feedback and generate threshold suggestions.
        
        Returns:
            List of ThresholdSuggestion objects
        """
        stats = self.get_feedback_stats()
        suggestions = []
        
        by_category = stats.get("by_category", {})
        
        for category, counts in by_category.items():
            confirmed = counts.get("confirmed", 0)
            rejected = counts.get("rejected", 0)
            total = confirmed + rejected
            
            if total < self.MIN_FEEDBACK_COUNT:
                logger.debug(f"Skipping {category}: insufficient feedback ({total} < {self.MIN_FEEDBACK_COUNT})")
                continue
            
            fpr = rejected / total
            confidence = min(1.0, total / 50)  # Confidence increases with more data
            
            # Get threshold fields for this category
            threshold_fields = CATEGORY_TO_THRESHOLD.get(category.lower(), [])
            
            if not threshold_fields:
                logger.debug(f"No threshold mapping for category: {category}")
                continue
            
            if fpr > self.HIGH_FPR_THRESHOLD:
                # Too many false positives - suggest increasing tolerance
                for field in threshold_fields:
                    current = getattr(self.thresholds, field, None)
                    if current is not None:
                        suggested = current * INCREASE_FACTOR
                        suggestions.append(ThresholdSuggestion(
                            category=category,
                            threshold_name=field,
                            current_value=current,
                            suggested_value=suggested,
                            reason=f"High false positive rate ({fpr:.0%}) - increase tolerance",
                            confidence=confidence,
                        ))
            
            elif fpr < self.LOW_FPR_THRESHOLD and confirmed > rejected * 5:
                # Very low FPR and many confirmations - could tighten threshold
                for field in threshold_fields:
                    current = getattr(self.thresholds, field, None)
                    if current is not None:
                        suggested = current * DECREASE_FACTOR
                        suggestions.append(ThresholdSuggestion(
                            category=category,
                            threshold_name=field,
                            current_value=current,
                            suggested_value=suggested,
                            reason=f"Low false positive rate ({fpr:.0%}) - could decrease tolerance",
                            confidence=confidence * 0.7,  # Lower confidence for tightening
                        ))
        
        # Sort by confidence (highest first)
        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        
        return suggestions
    
    def apply_suggestions(
        self,
        suggestions: List[ThresholdSuggestion],
        min_confidence: float = 0.5,
    ) -> Dict[str, float]:
        """Apply threshold suggestions to the config.
        
        Args:
            suggestions: List of suggestions to apply
            min_confidence: Minimum confidence to apply a suggestion
        
        Returns:
            Dict of applied changes: {threshold_name: new_value}
        """
        applied = {}
        
        for suggestion in suggestions:
            if suggestion.confidence < min_confidence:
                logger.debug(f"Skipping {suggestion.threshold_name}: confidence {suggestion.confidence} < {min_confidence}")
                continue
            
            # Apply the change
            if hasattr(self.thresholds, suggestion.threshold_name):
                setattr(self.thresholds, suggestion.threshold_name, suggestion.suggested_value)
                applied[suggestion.threshold_name] = suggestion.suggested_value
                logger.info(f"Applied threshold change: {suggestion.threshold_name} = {suggestion.suggested_value:.3f}")
        
        return applied
    
    def get_calibration_report(self) -> Dict[str, Any]:
        """Generate a comprehensive calibration report.
        
        Returns:
            Dict with current thresholds, feedback stats, and suggestions
        """
        stats = self.get_feedback_stats()
        suggestions = self.analyze_feedback()
        
        return {
            "feedback_stats": stats,
            "current_thresholds": self.thresholds.to_dict(),
            "suggestions": [s.to_dict() for s in suggestions],
            "suggestion_count": len(suggestions),
            "high_confidence_suggestions": len([s for s in suggestions if s.confidence >= 0.5]),
            "categories_analyzed": len(stats.get("by_category", {})),
        }


# Global tuner instance
_global_tuner: Optional[AutoTuner] = None


def get_auto_tuner(thresholds: Thresholds = None) -> AutoTuner:
    """Get the global auto-tuner instance.
    
    Args:
        thresholds: Optional thresholds to use
    
    Returns:
        AutoTuner instance
    """
    global _global_tuner
    
    if _global_tuner is None or thresholds is not None:
        _global_tuner = AutoTuner(thresholds=thresholds)
    
    return _global_tuner


def suggest_threshold_adjustments(feedback_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze feedback statistics and suggest threshold adjustments.
    
    This is a standalone function for quick analysis without the full tuner.
    
    Args:
        feedback_stats: Feedback statistics dict from database
    
    Returns:
        Dict with suggested adjustments
    """
    by_category = feedback_stats.get("by_category", {})
    suggestions = {}
    
    for category, counts in by_category.items():
        confirmed = counts.get("confirmed", 0)
        rejected = counts.get("rejected", 0)
        total = confirmed + rejected
        
        if total < 10:
            continue
        
        fpr = rejected / total
        
        if fpr > 0.5:
            suggestions[category] = {
                "action": "increase_tolerance",
                "reason": f"High false positive rate: {fpr:.0%}",
                "fpr": fpr,
                "total_feedback": total,
            }
        elif fpr < 0.1 and total >= 20:
            suggestions[category] = {
                "action": "could_decrease_tolerance",
                "reason": f"Low false positive rate: {fpr:.0%}",
                "fpr": fpr,
                "total_feedback": total,
            }
        else:
            suggestions[category] = {
                "action": "no_change_needed",
                "reason": f"Acceptable false positive rate: {fpr:.0%}",
                "fpr": fpr,
                "total_feedback": total,
            }
    
    return suggestions
