"""Multi-state capture for VQA pipeline.

Captures screenshots of web pages in different states:
- Default (no interaction)
- Hover (mouse over elements)
- Focus (keyboard focus on elements)
- Active (click/press state)
- Responsive (different viewport sizes)

Usage:
    async with StateCapture(page) as capture:
        # Capture hover states for all interactive elements
        states = await capture.capture_hover_states()
        
        # Capture at different breakpoints
        breakpoints = await capture.capture_breakpoints([1920, 1280, 768, 375])
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from PIL import Image
import io

logger = logging.getLogger(__name__)


# Standard responsive breakpoints
BREAKPOINTS = {
    "desktop_large": 1920,
    "desktop": 1440,
    "laptop": 1280,
    "tablet": 768,
    "mobile": 375,
}

# Interactive element selectors
INTERACTIVE_SELECTORS = [
    "button",
    "a",
    "[role='button']",
    "input",
    "select",
    "textarea",
    "[tabindex]",
    "[onclick]",
    ".btn",
    ".button",
]


@dataclass
class CapturedState:
    """A captured state with metadata."""
    name: str
    screenshot: Image.Image
    viewport_width: int
    viewport_height: int
    selector: Optional[str] = None
    state_type: str = "default"  # default, hover, focus, active, responsive
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_bytes(self) -> bytes:
        """Convert screenshot to PNG bytes."""
        buffer = io.BytesIO()
        self.screenshot.save(buffer, format="PNG")
        return buffer.getvalue()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "viewport_width": self.viewport_width,
            "viewport_height": self.viewport_height,
            "selector": self.selector,
            "state_type": self.state_type,
            "metadata": self.metadata,
        }


class StateCapture:
    """Multi-state screenshot capture using Playwright.
    
    Captures screenshots of web pages in various interaction states.
    Uses Playwright page object for browser automation.
    """
    
    DEFAULT_WAIT_MS = 300  # Wait for CSS transitions
    
    def __init__(
        self,
        page: Any,  # Playwright Page object
        default_wait_ms: int = None,
    ):
        """Initialize state capture.
        
        Args:
            page: Playwright Page object
            default_wait_ms: Default wait time after state changes
        """
        self.page = page
        self.wait_ms = default_wait_ms or self.DEFAULT_WAIT_MS
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def get_viewport_size(self) -> tuple[int, int]:
        """Get current viewport dimensions."""
        viewport = self.page.viewport_size
        return (viewport["width"], viewport["height"])
    
    async def capture_screenshot(self) -> Image.Image:
        """Capture current page screenshot as PIL Image."""
        screenshot_bytes = await self.page.screenshot(full_page=True)
        return Image.open(io.BytesIO(screenshot_bytes))
    
    async def capture_default_state(self) -> CapturedState:
        """Capture the default (no interaction) state."""
        width, height = await self.get_viewport_size()
        screenshot = await self.capture_screenshot()
        
        return CapturedState(
            name="default",
            screenshot=screenshot,
            viewport_width=width,
            viewport_height=height,
            state_type="default",
        )
    
    async def capture_hover_state(
        self,
        selector: str,
        wait_ms: int = None,
    ) -> Optional[CapturedState]:
        """Capture screenshot with element in hover state.
        
        Args:
            selector: CSS selector for element to hover
            wait_ms: Time to wait after hovering (for CSS transitions)
        
        Returns:
            CapturedState or None if element not found
        """
        wait_ms = wait_ms or self.wait_ms
        
        try:
            element = await self.page.query_selector(selector)
            if not element:
                logger.debug(f"Element not found: {selector}")
                return None
            
            # Check if element is visible
            is_visible = await element.is_visible()
            if not is_visible:
                return None
            
            # Hover over element
            await element.hover()
            await self.page.wait_for_timeout(wait_ms)
            
            # Capture screenshot
            width, height = await self.get_viewport_size()
            screenshot = await self.capture_screenshot()
            
            # Get element info
            box = await element.bounding_box()
            
            return CapturedState(
                name=f"hover_{selector}",
                screenshot=screenshot,
                viewport_width=width,
                viewport_height=height,
                selector=selector,
                state_type="hover",
                metadata={
                    "bounding_box": box,
                },
            )
            
        except Exception as e:
            logger.warning(f"Failed to capture hover state for {selector}: {e}")
            return None
    
    async def capture_focus_state(
        self,
        selector: str,
        wait_ms: int = None,
    ) -> Optional[CapturedState]:
        """Capture screenshot with element in focus state.
        
        Args:
            selector: CSS selector for element to focus
            wait_ms: Time to wait after focusing
        
        Returns:
            CapturedState or None if element not found
        """
        wait_ms = wait_ms or self.wait_ms
        
        try:
            element = await self.page.query_selector(selector)
            if not element:
                return None
            
            is_visible = await element.is_visible()
            if not is_visible:
                return None
            
            # Focus element
            await element.focus()
            await self.page.wait_for_timeout(wait_ms)
            
            # Capture screenshot
            width, height = await self.get_viewport_size()
            screenshot = await self.capture_screenshot()
            
            box = await element.bounding_box()
            
            return CapturedState(
                name=f"focus_{selector}",
                screenshot=screenshot,
                viewport_width=width,
                viewport_height=height,
                selector=selector,
                state_type="focus",
                metadata={
                    "bounding_box": box,
                },
            )
            
        except Exception as e:
            logger.warning(f"Failed to capture focus state for {selector}: {e}")
            return None
    
    async def capture_active_state(
        self,
        selector: str,
        wait_ms: int = None,
    ) -> Optional[CapturedState]:
        """Capture screenshot with element in active (pressed) state.
        
        Uses mouse down without releasing to simulate active state.
        
        Args:
            selector: CSS selector for element
            wait_ms: Time to wait in active state
        
        Returns:
            CapturedState or None if element not found
        """
        wait_ms = wait_ms or self.wait_ms
        
        try:
            element = await self.page.query_selector(selector)
            if not element:
                return None
            
            is_visible = await element.is_visible()
            if not is_visible:
                return None
            
            box = await element.bounding_box()
            if not box:
                return None
            
            # Mouse down on element center
            center_x = box["x"] + box["width"] / 2
            center_y = box["y"] + box["height"] / 2
            
            await self.page.mouse.move(center_x, center_y)
            await self.page.mouse.down()
            await self.page.wait_for_timeout(wait_ms)
            
            # Capture screenshot
            width, height = await self.get_viewport_size()
            screenshot = await self.capture_screenshot()
            
            # Release mouse
            await self.page.mouse.up()
            
            return CapturedState(
                name=f"active_{selector}",
                screenshot=screenshot,
                viewport_width=width,
                viewport_height=height,
                selector=selector,
                state_type="active",
                metadata={
                    "bounding_box": box,
                },
            )
            
        except Exception as e:
            logger.warning(f"Failed to capture active state for {selector}: {e}")
            return None
    
    async def capture_at_breakpoint(
        self,
        width: int,
        height: int = 1080,
        url: str = None,
    ) -> CapturedState:
        """Capture screenshot at specific viewport size.
        
        Args:
            width: Viewport width
            height: Viewport height
            url: Optional URL to navigate to after resize
        
        Returns:
            CapturedState at the specified viewport
        """
        # Set viewport
        await self.page.set_viewport_size({"width": width, "height": height})
        
        # Optionally reload page (for responsive layouts that need re-render)
        if url:
            await self.page.goto(url)
            await self.page.wait_for_load_state("networkidle")
        else:
            # Just wait for potential layout changes
            await self.page.wait_for_timeout(500)
        
        # Capture screenshot
        screenshot = await self.capture_screenshot()
        
        return CapturedState(
            name=f"breakpoint_{width}",
            screenshot=screenshot,
            viewport_width=width,
            viewport_height=height,
            state_type="responsive",
        )
    
    async def capture_breakpoints(
        self,
        breakpoints: List[int] = None,
        height: int = 1080,
        url: str = None,
    ) -> Dict[int, CapturedState]:
        """Capture screenshots at multiple breakpoints.
        
        Args:
            breakpoints: List of viewport widths (defaults to standard breakpoints)
            height: Viewport height
            url: Optional URL for each breakpoint
        
        Returns:
            Dict mapping width to CapturedState
        """
        if breakpoints is None:
            breakpoints = list(BREAKPOINTS.values())
        
        results = {}
        original_viewport = self.page.viewport_size
        
        try:
            for width in breakpoints:
                state = await self.capture_at_breakpoint(width, height, url)
                results[width] = state
                logger.debug(f"Captured breakpoint {width}x{height}")
        finally:
            # Restore original viewport
            await self.page.set_viewport_size(original_viewport)
        
        return results
    
    async def capture_all_hover_states(
        self,
        selectors: List[str] = None,
        max_elements: int = 20,
    ) -> List[CapturedState]:
        """Capture hover states for all interactive elements.
        
        Args:
            selectors: CSS selectors to check (defaults to INTERACTIVE_SELECTORS)
            max_elements: Maximum number of elements to capture
        
        Returns:
            List of CapturedState for each hover state
        """
        if selectors is None:
            selectors = INTERACTIVE_SELECTORS
        
        states = []
        captured_count = 0
        
        for selector in selectors:
            if captured_count >= max_elements:
                break
            
            try:
                elements = await self.page.query_selector_all(selector)
                
                for i, element in enumerate(elements):
                    if captured_count >= max_elements:
                        break
                    
                    is_visible = await element.is_visible()
                    if not is_visible:
                        continue
                    
                    # Use a unique selector for this element
                    unique_selector = f"{selector}:nth-of-type({i + 1})"
                    state = await self.capture_hover_state(unique_selector)
                    
                    if state:
                        states.append(state)
                        captured_count += 1
                        
            except Exception as e:
                logger.warning(f"Error capturing hover states for {selector}: {e}")
        
        return states
    
    async def capture_multi_state(
        self,
        include_default: bool = True,
        include_hover: bool = True,
        include_focus: bool = False,
        hover_selectors: List[str] = None,
        focus_selectors: List[str] = None,
        max_states: int = 10,
    ) -> List[CapturedState]:
        """Capture multiple interaction states.
        
        Args:
            include_default: Include default state
            include_hover: Include hover states
            include_focus: Include focus states
            hover_selectors: Selectors for hover states
            focus_selectors: Selectors for focus states
            max_states: Maximum total states to capture
        
        Returns:
            List of all captured states
        """
        states = []
        
        if include_default:
            default = await self.capture_default_state()
            states.append(default)
        
        remaining = max_states - len(states)
        
        if include_hover and remaining > 0:
            hover_states = await self.capture_all_hover_states(
                selectors=hover_selectors,
                max_elements=min(remaining, 10),
            )
            states.extend(hover_states)
            remaining = max_states - len(states)
        
        if include_focus and remaining > 0 and focus_selectors:
            for selector in focus_selectors[:remaining]:
                focus_state = await self.capture_focus_state(selector)
                if focus_state:
                    states.append(focus_state)
        
        return states


# Convenience functions for standalone use

async def capture_hover_state(page: Any, selector: str) -> Optional[CapturedState]:
    """Capture hover state for a single element."""
    capture = StateCapture(page)
    return await capture.capture_hover_state(selector)


async def capture_focus_state(page: Any, selector: str) -> Optional[CapturedState]:
    """Capture focus state for a single element."""
    capture = StateCapture(page)
    return await capture.capture_focus_state(selector)


async def capture_active_state(page: Any, selector: str) -> Optional[CapturedState]:
    """Capture active state for a single element."""
    capture = StateCapture(page)
    return await capture.capture_active_state(selector)


async def capture_breakpoints(
    page: Any,
    breakpoints: List[int] = None,
    url: str = None,
) -> Dict[int, CapturedState]:
    """Capture screenshots at multiple breakpoints."""
    capture = StateCapture(page)
    return await capture.capture_breakpoints(breakpoints, url=url)


async def capture_multi_state(
    page: Any,
    include_hover: bool = True,
    include_focus: bool = False,
    max_states: int = 10,
) -> List[CapturedState]:
    """Capture multiple interaction states."""
    capture = StateCapture(page)
    return await capture.capture_multi_state(
        include_hover=include_hover,
        include_focus=include_focus,
        max_states=max_states,
    )
