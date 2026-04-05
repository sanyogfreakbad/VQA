"""
Playwright-based interaction testing harness.

For each page, this module:
1. Identifies interactive elements (buttons, dropdowns, toggles, links, inputs)
2. For each element, triggers relevant interactions (hover, click, focus)
3. Captures before + after screenshots
4. Returns structured results for LLM behavioral checking

This catches ~20% of QA issues that are about interactive BEHAVIOR,
not static appearance:
- "If cursor hovers over toast, toast remains visible"
- "On clicking filter icon, table width changes"
- "Hover sub-header should not have a gradient"
- "Disabled and selected state should not have same color"
"""

import asyncio
import io
import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Union

from PIL import Image

from .interaction_specs import (
    InteractionType,
    ComponentType,
    detect_component_type,
    get_interactions_for_element,
    get_undo_action_for_interaction,
    get_wait_time_for_interaction,
    get_expected_changes,
    get_validation_rules,
)

logger = logging.getLogger(__name__)


@dataclass
class InteractionResult:
    """Result from a single interaction capture."""
    id: str
    element_selector: str
    element_name: str
    element_type: ComponentType
    interaction_type: InteractionType
    
    before_screenshot: Optional[Image.Image] = None
    after_screenshot: Optional[Image.Image] = None
    before_crop: Optional[Image.Image] = None
    after_crop: Optional[Image.Image] = None
    
    element_bbox: Optional[Dict[str, float]] = None
    
    expected_changes: List[str] = field(default_factory=list)
    validation_rules: List[str] = field(default_factory=list)
    
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for passing to LLM."""
        return {
            "id": self.id,
            "element": self.element_selector,
            "element_name": self.element_name,
            "element_type": self.element_type.value,
            "interaction": self.interaction_type.value,
            "before_screenshot": self.before_screenshot,
            "after_screenshot": self.after_screenshot,
            "before_crop": self.before_crop,
            "after_crop": self.after_crop,
            "element_bbox": self.element_bbox,
            "expected_changes": self.expected_changes,
            "validation_rules": self.validation_rules,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class InteractionRunnerConfig:
    """Configuration for the interaction runner."""
    crop_padding: int = 50
    default_wait_ms: int = 300
    scroll_into_view: bool = True
    capture_full_page: bool = False
    max_elements_per_type: int = 10
    skip_hidden_elements: bool = True
    skip_disabled_elements: bool = True
    viewport_width: int = 1280
    viewport_height: int = 800


class InteractionRunner:
    """Playwright-based interaction runner for behavioral testing."""
    
    def __init__(
        self,
        page: Any,
        config: Optional[InteractionRunnerConfig] = None,
    ):
        """Initialize the interaction runner.
        
        Args:
            page: Playwright Page object
            config: Runner configuration
        """
        self.page = page
        self.config = config or InteractionRunnerConfig()
        self._original_scroll_position: Optional[Tuple[int, int]] = None
    
    async def capture_all_states(
        self,
        interactive_elements: List[Dict],
        max_interactions: int = 50,
    ) -> List[InteractionResult]:
        """Capture before/after screenshots for each interaction.
        
        Args:
            interactive_elements: List of elements to test, each with:
                - selector: CSS selector or XPath
                - interactions: List of interaction types to test (optional)
                - element_type: Component type (optional, will be detected)
                - element_name: Human-readable name (optional)
            max_interactions: Maximum total interactions to perform
        
        Returns:
            List of InteractionResult objects
        """
        results = []
        interaction_count = 0
        
        for elem_spec in interactive_elements:
            if interaction_count >= max_interactions:
                logger.info(f"Reached max interactions ({max_interactions}), stopping")
                break
            
            try:
                elem_results = await self._test_element(elem_spec)
                results.extend(elem_results)
                interaction_count += len(elem_results)
            except Exception as e:
                logger.warning(f"Failed to test element {elem_spec.get('selector')}: {e}")
                continue
        
        logger.info(f"Captured {len(results)} interaction states")
        return results
    
    async def _test_element(
        self,
        elem_spec: Dict,
    ) -> List[InteractionResult]:
        """Test all interactions for a single element.
        
        Args:
            elem_spec: Element specification
        
        Returns:
            List of InteractionResult objects for this element
        """
        selector = elem_spec.get("selector")
        if not selector:
            return []
        
        element = self.page.locator(selector).first
        
        if not await self._is_element_testable(element):
            return []
        
        bbox = await element.bounding_box()
        if not bbox:
            return []
        
        element_data = await self._extract_element_data(element)
        
        element_type = elem_spec.get("element_type")
        if element_type:
            if isinstance(element_type, str):
                try:
                    element_type = ComponentType(element_type)
                except ValueError:
                    element_type = detect_component_type(element_data)
        else:
            element_type = detect_component_type(element_data)
        
        interactions = elem_spec.get("interactions")
        if interactions:
            interactions = [
                InteractionType(i) if isinstance(i, str) else i
                for i in interactions
            ]
        else:
            interactions = get_interactions_for_element(element_data)
        
        element_name = elem_spec.get("element_name", element_data.get("text", selector))
        
        results = []
        for interaction in interactions:
            try:
                result = await self._capture_interaction(
                    element=element,
                    selector=selector,
                    element_name=element_name,
                    element_type=element_type,
                    interaction=interaction,
                    bbox=bbox,
                )
                if result:
                    results.append(result)
            except Exception as e:
                logger.debug(f"Interaction {interaction} failed for {selector}: {e}")
                results.append(InteractionResult(
                    id=f"int_{uuid.uuid4().hex[:8]}",
                    element_selector=selector,
                    element_name=element_name,
                    element_type=element_type,
                    interaction_type=interaction,
                    success=False,
                    error=str(e),
                ))
        
        return results
    
    async def _capture_interaction(
        self,
        element: Any,
        selector: str,
        element_name: str,
        element_type: ComponentType,
        interaction: InteractionType,
        bbox: Dict,
    ) -> Optional[InteractionResult]:
        """Capture before and after state for a single interaction.
        
        IMPORTANT: After capturing, UNDO the interaction to restore page state.
        
        Args:
            element: Playwright Locator
            selector: CSS selector
            element_name: Human-readable name
            element_type: Component type
            interaction: Interaction to perform
            bbox: Element bounding box
        
        Returns:
            InteractionResult or None if failed
        """
        if self.config.scroll_into_view:
            await element.scroll_into_view_if_needed()
            await asyncio.sleep(0.1)
            bbox = await element.bounding_box()
        
        before = await self._take_screenshot()
        
        try:
            await self._perform_interaction(element, interaction)
        except Exception as e:
            logger.debug(f"Failed to perform {interaction} on {selector}: {e}")
            return None
        
        wait_time = get_wait_time_for_interaction(interaction, element_type)
        await asyncio.sleep(wait_time / 1000)
        
        after = await self._take_screenshot()
        
        padding = self.config.crop_padding
        before_crop = self._crop_to_element(before, bbox, padding)
        after_crop = self._crop_to_element(after, bbox, padding)
        
        await self._undo_interaction(element, interaction, element_type)
        
        return InteractionResult(
            id=f"int_{uuid.uuid4().hex[:8]}",
            element_selector=selector,
            element_name=element_name,
            element_type=element_type,
            interaction_type=interaction,
            before_screenshot=before,
            after_screenshot=after,
            before_crop=before_crop,
            after_crop=after_crop,
            element_bbox=bbox,
            expected_changes=get_expected_changes(interaction),
            validation_rules=get_validation_rules(element_type, interaction),
        )
    
    async def _perform_interaction(
        self,
        element: Any,
        interaction: InteractionType,
    ) -> None:
        """Perform an interaction on an element.
        
        Args:
            element: Playwright Locator
            interaction: Interaction type to perform
        """
        if interaction == InteractionType.HOVER:
            await element.hover()
        
        elif interaction == InteractionType.CLICK:
            await element.click()
        
        elif interaction == InteractionType.FOCUS:
            await element.focus()
        
        elif interaction == InteractionType.BLUR:
            await self.page.evaluate("document.activeElement?.blur()")
        
        elif interaction == InteractionType.DOUBLE_CLICK:
            await element.dblclick()
        
        elif interaction == InteractionType.RIGHT_CLICK:
            await element.click(button="right")
        
        elif interaction == InteractionType.PRESS:
            await element.click()
            await asyncio.sleep(0.05)
        
        elif interaction in (InteractionType.TOGGLE_ON, InteractionType.TOGGLE_OFF):
            await element.click()
        
        elif interaction in (InteractionType.DROPDOWN_OPEN, InteractionType.DROPDOWN_CLOSE):
            await element.click()
        
        elif interaction == InteractionType.TAB_SELECT:
            await element.click()
        
        elif interaction == InteractionType.SCROLL:
            self._original_scroll_position = await self._get_scroll_position()
            await self._scroll_element(element, 100)
        
        elif interaction == InteractionType.DRAG:
            box = await element.bounding_box()
            if box:
                await self.page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                await self.page.mouse.down()
                await self.page.mouse.move(box["x"] + box["width"] / 2 + 50, box["y"] + box["height"] / 2)
                await self.page.mouse.up()
    
    async def _undo_interaction(
        self,
        element: Any,
        interaction: InteractionType,
        element_type: ComponentType,
    ) -> None:
        """Undo an interaction to restore page state.
        
        Args:
            element: Playwright Locator
            interaction: Interaction that was performed
            element_type: Component type
        """
        undo_action = get_undo_action_for_interaction(interaction, element_type)
        
        if undo_action == "move_away":
            await self.page.mouse.move(0, 0)
        
        elif undo_action == "blur":
            await self.page.evaluate("document.activeElement?.blur()")
        
        elif undo_action == "escape":
            await self.page.keyboard.press("Escape")
        
        elif undo_action == "click_again":
            try:
                await element.click()
            except Exception:
                pass
        
        elif undo_action == "scroll_back":
            if self._original_scroll_position:
                await self._set_scroll_position(*self._original_scroll_position)
                self._original_scroll_position = None
        
        await asyncio.sleep(0.1)
    
    async def _is_element_testable(self, element: Any) -> bool:
        """Check if an element can be tested.
        
        Args:
            element: Playwright Locator
        
        Returns:
            True if element is testable
        """
        try:
            count = await element.count()
            if count == 0:
                return False
            
            if self.config.skip_hidden_elements:
                is_visible = await element.is_visible()
                if not is_visible:
                    return False
            
            if self.config.skip_disabled_elements:
                is_disabled = await element.is_disabled()
                if is_disabled:
                    return False
            
            bbox = await element.bounding_box()
            if not bbox:
                return False
            if bbox["width"] < 5 or bbox["height"] < 5:
                return False
            
            return True
            
        except Exception:
            return False
    
    async def _extract_element_data(self, element: Any) -> Dict[str, Any]:
        """Extract element attributes for component type detection.
        
        Args:
            element: Playwright Locator
        
        Returns:
            Dict with element attributes
        """
        try:
            data = await element.evaluate("""
                (el) => {
                    const rect = el.getBoundingClientRect();
                    const computed = window.getComputedStyle(el);
                    
                    return {
                        tag_name: el.tagName.toLowerCase(),
                        role: el.getAttribute('role') || '',
                        type: el.getAttribute('type') || '',
                        class_name: el.className || '',
                        id: el.id || '',
                        'aria-label': el.getAttribute('aria-label') || '',
                        'aria-expanded': el.getAttribute('aria-expanded'),
                        'aria-pressed': el.getAttribute('aria-pressed'),
                        'aria-selected': el.getAttribute('aria-selected'),
                        'aria-checked': el.getAttribute('aria-checked'),
                        'aria-disabled': el.getAttribute('aria-disabled'),
                        disabled: el.disabled || false,
                        text: el.textContent?.slice(0, 100)?.trim() || '',
                        has_overflow: el.scrollWidth > el.clientWidth || el.scrollHeight > el.clientHeight,
                        cursor: computed.cursor,
                        display: computed.display,
                        visibility: computed.visibility,
                    };
                }
            """)
            return data
        except Exception as e:
            logger.debug(f"Failed to extract element data: {e}")
            return {}
    
    async def _take_screenshot(self) -> Image.Image:
        """Take a screenshot of the page.
        
        Returns:
            PIL Image
        """
        screenshot_bytes = await self.page.screenshot(
            type="png",
            full_page=self.config.capture_full_page,
        )
        return Image.open(io.BytesIO(screenshot_bytes))
    
    def _crop_to_element(
        self,
        img: Image.Image,
        bbox: Dict,
        padding: int = 50,
    ) -> Image.Image:
        """Crop image to element bounding box with padding.
        
        Args:
            img: Full page screenshot
            bbox: Element bounding box {x, y, width, height}
            padding: Padding around element in pixels
        
        Returns:
            Cropped PIL Image
        """
        x = max(0, int(bbox["x"]) - padding)
        y = max(0, int(bbox["y"]) - padding)
        x2 = min(img.width, int(bbox["x"] + bbox["width"]) + padding)
        y2 = min(img.height, int(bbox["y"] + bbox["height"]) + padding)
        
        return img.crop((x, y, x2, y2))
    
    async def _scroll_element(
        self,
        element: Any,
        delta_y: int = 100,
    ) -> None:
        """Scroll within an element or the page.
        
        Args:
            element: Playwright Locator
            delta_y: Pixels to scroll
        """
        try:
            await element.evaluate(
                "(el, dy) => { el.scrollTop += dy; }",
                delta_y
            )
        except Exception:
            await self.page.mouse.wheel(0, delta_y)
    
    async def _get_scroll_position(self) -> Tuple[int, int]:
        """Get current scroll position.
        
        Returns:
            Tuple of (scrollX, scrollY)
        """
        return await self.page.evaluate(
            "() => [window.scrollX, window.scrollY]"
        )
    
    async def _set_scroll_position(self, x: int, y: int) -> None:
        """Set scroll position.
        
        Args:
            x: Scroll X position
            y: Scroll Y position
        """
        await self.page.evaluate(
            "(x, y) => window.scrollTo(x, y)",
            x, y
        )
    
    async def discover_interactive_elements(
        self,
        selectors: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Discover interactive elements on the page.
        
        Args:
            selectors: Optional list of selectors to search. If None, uses defaults.
        
        Returns:
            List of element specifications for testing
        """
        if selectors is None:
            selectors = [
                "button",
                "[role='button']",
                "a",
                "[role='link']",
                "input",
                "textarea",
                "select",
                "[role='tab']",
                "[role='switch']",
                "[role='checkbox']",
                "[role='radio']",
                "[role='listbox']",
                "[role='combobox']",
                "[role='slider']",
                ".card",
                ".toggle",
                ".dropdown",
                ".accordion",
            ]
        
        discovered = []
        seen_elements = set()
        
        for selector in selectors:
            try:
                locator = self.page.locator(selector)
                count = await locator.count()
                
                max_per_type = self.config.max_elements_per_type
                actual_count = min(count, max_per_type)
                
                for i in range(actual_count):
                    element = locator.nth(i)
                    
                    if not await self._is_element_testable(element):
                        continue
                    
                    element_id = await element.evaluate(
                        "(el) => el.id || el.className + el.textContent?.slice(0, 20)"
                    )
                    if element_id in seen_elements:
                        continue
                    seen_elements.add(element_id)
                    
                    element_data = await self._extract_element_data(element)
                    component_type = detect_component_type(element_data)
                    
                    unique_selector = await self._get_unique_selector(element, selector, i)
                    
                    discovered.append({
                        "selector": unique_selector,
                        "element_type": component_type,
                        "element_name": element_data.get("text", "")[:50] or unique_selector,
                        "element_data": element_data,
                    })
                    
            except Exception as e:
                logger.debug(f"Failed to discover elements for {selector}: {e}")
                continue
        
        logger.info(f"Discovered {len(discovered)} interactive elements")
        return discovered
    
    async def _get_unique_selector(
        self,
        element: Any,
        base_selector: str,
        index: int,
    ) -> str:
        """Generate a unique selector for an element.
        
        Args:
            element: Playwright Locator
            base_selector: Original selector used to find element
            index: Index in the result set
        
        Returns:
            Unique CSS selector string
        """
        try:
            element_id = await element.get_attribute("id")
            if element_id:
                return f"#{element_id}"
            
            data_testid = await element.get_attribute("data-testid")
            if data_testid:
                return f"[data-testid='{data_testid}']"
            
            aria_label = await element.get_attribute("aria-label")
            if aria_label:
                return f"[aria-label='{aria_label}']"
            
            return f"{base_selector} >> nth={index}"
            
        except Exception:
            return f"{base_selector} >> nth={index}"


async def run_behavioral_tests(
    page: Any,
    elements: Optional[List[Dict]] = None,
    config: Optional[InteractionRunnerConfig] = None,
) -> List[InteractionResult]:
    """Convenience function to run behavioral tests on a page.
    
    Args:
        page: Playwright Page object
        elements: Optional list of elements to test. If None, discovers automatically.
        config: Runner configuration
    
    Returns:
        List of InteractionResult objects
    """
    runner = InteractionRunner(page, config)
    
    if elements is None:
        elements = await runner.discover_interactive_elements()
    
    return await runner.capture_all_states(elements)


async def capture_hover_states(
    page: Any,
    selectors: List[str],
    config: Optional[InteractionRunnerConfig] = None,
) -> List[InteractionResult]:
    """Convenience function to capture hover states for specific selectors.
    
    Args:
        page: Playwright Page object
        selectors: List of CSS selectors
        config: Runner configuration
    
    Returns:
        List of InteractionResult objects
    """
    runner = InteractionRunner(page, config)
    
    elements = [
        {"selector": s, "interactions": [InteractionType.HOVER]}
        for s in selectors
    ]
    
    return await runner.capture_all_states(elements)


async def capture_focus_states(
    page: Any,
    selectors: List[str],
    config: Optional[InteractionRunnerConfig] = None,
) -> List[InteractionResult]:
    """Convenience function to capture focus states for specific selectors.
    
    Args:
        page: Playwright Page object
        selectors: List of CSS selectors
        config: Runner configuration
    
    Returns:
        List of InteractionResult objects
    """
    runner = InteractionRunner(page, config)
    
    elements = [
        {"selector": s, "interactions": [InteractionType.FOCUS]}
        for s in selectors
    ]
    
    return await runner.capture_all_states(elements)


async def capture_click_states(
    page: Any,
    selectors: List[str],
    config: Optional[InteractionRunnerConfig] = None,
) -> List[InteractionResult]:
    """Convenience function to capture click states (for toggles, dropdowns, etc.).
    
    Args:
        page: Playwright Page object
        selectors: List of CSS selectors
        config: Runner configuration
    
    Returns:
        List of InteractionResult objects
    """
    runner = InteractionRunner(page, config)
    
    elements = [
        {"selector": s, "interactions": [InteractionType.CLICK]}
        for s in selectors
    ]
    
    return await runner.capture_all_states(elements)
