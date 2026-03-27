#!/usr/bin/env python3
"""
Web DOM Extractor

Extracts visual/layout properties from rendered web pages using Playwright.
Returns JSON structure matching the Figma extraction format.
"""

import re
from datetime import datetime, timezone
from typing import Any, Optional

from playwright.async_api import async_playwright, Page, Browser


def parse_color(color_str: str) -> dict[str, Any]:
    """Parse CSS color string to {r, g, b, a} format."""
    if not color_str or color_str == "transparent":
        return {"r": 0, "g": 0, "b": 0, "a": 0}
    
    # Handle rgba(r, g, b, a)
    rgba_match = re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)", color_str)
    if rgba_match:
        return {
            "r": int(rgba_match.group(1)),
            "g": int(rgba_match.group(2)),
            "b": int(rgba_match.group(3)),
            "a": float(rgba_match.group(4)) if rgba_match.group(4) else 1.0,
        }
    
    # Handle hex colors
    hex_match = re.match(r"#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})?", color_str)
    if hex_match:
        return {
            "r": int(hex_match.group(1), 16),
            "g": int(hex_match.group(2), 16),
            "b": int(hex_match.group(3), 16),
            "a": int(hex_match.group(4), 16) / 255 if hex_match.group(4) else 1.0,
        }
    
    return {"r": 0, "g": 0, "b": 0, "a": 1}


def parse_pixel_value(value: str) -> float:
    """Parse CSS pixel value to float."""
    if not value:
        return 0
    match = re.match(r"([\d.]+)", str(value))
    return float(match.group(1)) if match else 0


def determine_node_type(tag_name: str, styles: dict) -> str:
    """Map HTML element to Figma-like node type."""
    text_tags = {"P", "SPAN", "H1", "H2", "H3", "H4", "H5", "H6", "LABEL", "A", "STRONG", "EM", "B", "I"}
    frame_tags = {"DIV", "SECTION", "ARTICLE", "HEADER", "FOOTER", "NAV", "MAIN", "ASIDE", "FORM"}
    
    if tag_name in text_tags:
        return "TEXT"
    if tag_name in {"IMG", "SVG", "CANVAS", "VIDEO"}:
        return "IMAGE"
    if tag_name in {"INPUT", "BUTTON", "SELECT", "TEXTAREA"}:
        return "COMPONENT"
    if tag_name in frame_tags:
        return "FRAME"
    return "RECTANGLE"


def map_text_align(css_align: str) -> str:
    """Map CSS text-align to Figma textAlignHorizontal."""
    mapping = {
        "left": "LEFT",
        "center": "CENTER",
        "right": "RIGHT",
        "justify": "JUSTIFIED",
        "start": "LEFT",
        "end": "RIGHT",
    }
    return mapping.get(css_align, "LEFT")


def map_layout_mode(display: str, flex_direction: str) -> Optional[str]:
    """Map CSS display/flex to Figma layoutMode."""
    if display == "flex" or display == "inline-flex":
        if flex_direction in ("column", "column-reverse"):
            return "VERTICAL"
        return "HORIZONTAL"
    if display == "grid":
        return "VERTICAL"
    return None


DOM_WALKER_SCRIPT = """
(config) => {
    const results = [];
    let nodeId = 0;
    
    function isVisible(el) {
        const style = window.getComputedStyle(el);
        const rect = el.getBoundingClientRect();
        
        if (style.display === 'none' || style.visibility === 'hidden') return false;
        if (rect.width === 0 && rect.height === 0) return false;
        if (parseFloat(style.opacity) === 0) return false;
        
        return true;
    }
    
    function getStyles(el) {
        const cs = window.getComputedStyle(el);
        return {
            display: cs.display,
            position: cs.position,
            flexDirection: cs.flexDirection,
            justifyContent: cs.justifyContent,
            alignItems: cs.alignItems,
            gap: cs.gap,
            backgroundColor: cs.backgroundColor,
            color: cs.color,
            borderColor: cs.borderColor,
            borderWidth: cs.borderWidth,
            borderStyle: cs.borderStyle,
            borderRadius: cs.borderRadius,
            borderTopLeftRadius: cs.borderTopLeftRadius,
            borderTopRightRadius: cs.borderTopRightRadius,
            borderBottomRightRadius: cs.borderBottomRightRadius,
            borderBottomLeftRadius: cs.borderBottomLeftRadius,
            paddingTop: cs.paddingTop,
            paddingRight: cs.paddingRight,
            paddingBottom: cs.paddingBottom,
            paddingLeft: cs.paddingLeft,
            marginTop: cs.marginTop,
            marginRight: cs.marginRight,
            marginBottom: cs.marginBottom,
            marginLeft: cs.marginLeft,
            fontFamily: cs.fontFamily,
            fontSize: cs.fontSize,
            fontWeight: cs.fontWeight,
            fontStyle: cs.fontStyle,
            lineHeight: cs.lineHeight,
            letterSpacing: cs.letterSpacing,
            textAlign: cs.textAlign,
            textDecoration: cs.textDecoration,
            textTransform: cs.textTransform,
            opacity: cs.opacity,
            overflow: cs.overflow,
            boxShadow: cs.boxShadow,
            transform: cs.transform,
            zIndex: cs.zIndex,
        };
    }
    
    function getTextContent(el) {
        let text = '';
        for (const child of el.childNodes) {
            if (child.nodeType === Node.TEXT_NODE) {
                text += child.textContent;
            }
        }
        return text.trim();
    }
    
    /**
     * Find the associated interactive element (input/select/textarea) for a
     * label-like element. Searches via:
     *   1. <label for="id">  →  document.getElementById(id)
     *   2. <label> wrapping an input  →  label.querySelector('input,select,textarea')
     *   3. Parent/sibling scan: walk up to a form-field wrapper, then find
     *      the first input/select/textarea inside it
     *   4. aria-controls / aria-describedby pointing to an input
     *
     * Returns the input element or null.
     */
    function findAssociatedInput(el) {
        // 1. <label for="...">
        if (el.tagName === 'LABEL' && el.htmlFor) {
            const target = document.getElementById(el.htmlFor);
            if (target) return target;
        }
        
        // 2. Label wrapping an input
        if (el.tagName === 'LABEL') {
            const inner = el.querySelector('input, select, textarea');
            if (inner) return inner;
        }
        
        // 3. Walk up (max 4 levels) looking for a wrapper that contains an input
        let parent = el.parentElement;
        for (let i = 0; i < 4 && parent; i++) {
            const input = parent.querySelector('input, select, textarea, [role="combobox"], [role="listbox"], [contenteditable="true"]');
            if (input && input !== el) return input;
            parent = parent.parentElement;
        }
        
        // 4. aria-controls
        const controls = el.getAttribute('aria-controls');
        if (controls) {
            const target = document.getElementById(controls);
            if (target) return target;
        }
        
        return null;
    }
    
    /**
     * For any element, find the nearest interactive ancestor or self.
     * Useful for text nodes inside buttons, links, etc.
     * Walks up max 3 levels looking for button, a, input, select, [role=button], etc.
     * Returns the interactive element or null.
     */
    function findNearestInteractive(el) {
        const interactiveTags = new Set(['BUTTON', 'A', 'INPUT', 'SELECT', 'TEXTAREA']);
        const interactiveRoles = new Set(['button', 'link', 'tab', 'menuitem', 'option', 'switch', 'checkbox', 'radio', 'combobox']);
        
        let current = el;
        for (let i = 0; i < 4 && current; i++) {
            if (interactiveTags.has(current.tagName)) return current;
            const r = current.getAttribute('role');
            if (r && interactiveRoles.has(r)) return current;
            if (current.getAttribute('data-testid')) return current;
            current = current.parentElement;
        }
        return null;
    }
    
    /**
     * Generate a best-effort XPath locator for an element.
     * Priority order (first match wins):
     *   1. //*[@id="..."]                                   — HTML id
     *   2. //*[@data-testid="..."]                          — test id
     *   3. //tag[@name="..."]                               — name attr (forms)
     *   4. //tag[@role="..." and @aria-label="..."]         — ARIA
     *   5. //tag[@role="..." and contains(text(),"...")]    — role + text
     *   6. //tag[@placeholder="..."]                        — placeholder
     *   7. //tag[text()="..."]  or contains(text(),"...")   — text content
     *   8. //tag[@aria-label="..."]                         — aria-label alone
     *   9. //tag[@class="..."]                              — full class (fallback)
     * Text is trimmed to 60 chars max for readability.
     */
    function generateLocator(el) {
        const tag = el.tagName.toLowerCase();
        
        // 1. id — most reliable
        const elId = el.id;
        if (elId) {
            return '//*[@id="' + elId + '"]';
        }
        
        // 2. data-testid
        const testId = el.getAttribute('data-testid') || el.getAttribute('data-test-id') || el.getAttribute('data-cy');
        if (testId) {
            return '//*[@data-testid="' + testId + '"]';
        }
        
        // 3. name attribute (inputs, selects, textareas)
        const nameAttr = el.getAttribute('name');
        if (nameAttr && ['INPUT','SELECT','TEXTAREA','BUTTON'].includes(el.tagName)) {
            return '//' + tag + '[@name="' + nameAttr + '"]';
        }
        
        // Gather text (direct text only, trimmed, max 60 chars)
        const directText = getTextContent(el);
        const trimText = directText.length > 60 ? directText.substring(0, 60) : directText;
        
        // Gather ARIA info
        const role = el.getAttribute('role');
        const ariaLabel = el.getAttribute('aria-label');
        
        // 4. role + aria-label
        if (role && ariaLabel) {
            return '//' + tag + '[@role="' + role + '" and @aria-label="' + ariaLabel + '"]';
        }
        
        // 5. role + text
        if (role && trimText) {
            if (directText === trimText) {
                return '//' + tag + '[@role="' + role + '" and text()=\\'' + trimText + '\\']';
            }
            return '//' + tag + '[@role="' + role + '" and contains(text(),\\'' + trimText + '\\')]';
        }
        
        // 6. placeholder
        const placeholder = el.getAttribute('placeholder');
        if (placeholder) {
            return '//' + tag + '[@placeholder="' + placeholder + '"]';
        }
        
        // 7. text content
        if (trimText) {
            if (directText === trimText && directText.length <= 50) {
                return '//' + tag + '[text()=\\'' + trimText + '\\']';
            }
            if (trimText.length >= 3) {
                return '//' + tag + '[contains(text(),\\'' + trimText + '\\')]';
            }
        }
        
        // 8. aria-label alone
        if (ariaLabel) {
            return '//' + tag + '[@aria-label="' + ariaLabel + '"]';
        }
        
        // 9. class fallback — use first meaningful class (skip hash-like)
        const className = el.className && typeof el.className === 'string' ? el.className : '';
        if (className) {
            const classes = className.split(/\\s+/).filter(c => c.length > 0 && c.length < 50);
            if (classes.length > 0) {
                return '//' + tag + '[contains(@class,"' + classes[0] + '")]';
            }
        }
        
        // 10. bare tag (last resort)
        return '//' + tag;
    }
    
    /**
     * For a given element, produce the most automation-useful locator.
     *
     * If the element is a label/span/p/h-tag that labels a form field,
     * resolve to the associated input and return that input's locator.
     *
     * If the element is text inside a button/link, resolve up to
     * the interactive parent.
     *
     * Otherwise return the element's own locator.
     */
    function resolveLocator(el) {
        const tag = el.tagName;
        const labelTags = new Set(['LABEL', 'SPAN', 'P', 'DIV', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'LEGEND']);
        
        // For label-like elements, try to find the associated input
        if (labelTags.has(tag)) {
            const input = findAssociatedInput(el);
            if (input) {
                return generateLocator(input);
            }
        }
        
        // For text inside interactive elements (button text, link text), 
        // resolve up to the interactive parent
        const textTags = new Set(['SPAN', 'STRONG', 'EM', 'B', 'I', 'SMALL']);
        if (textTags.has(tag)) {
            const interactive = findNearestInteractive(el);
            if (interactive && interactive !== el) {
                return generateLocator(interactive);
            }
        }
        
        return generateLocator(el);
    }
    
    function walkDOM(el, depth = 0, parentPath = '') {
        if (depth > config.maxDepth) return;
        if (!el || el.nodeType !== Node.ELEMENT_NODE) return;
        
        const tagName = el.tagName;
        
        // Skip script, style, noscript, and hidden elements
        const skipTags = ['SCRIPT', 'STYLE', 'NOSCRIPT', 'META', 'LINK', 'HEAD', 'BR', 'HR'];
        if (skipTags.includes(tagName)) return;
        
        if (!isVisible(el)) return;
        
        const rect = el.getBoundingClientRect();
        const styles = getStyles(el);
        const currentId = `node_${nodeId++}`;
        const path = parentPath ? `${parentPath}/${tagName}[${currentId}]` : tagName;
        
        // Get direct text content
        const textContent = getTextContent(el);
        
        // Get class and id for better identification
        const className = el.className && typeof el.className === 'string' ? el.className : '';
        const elId = el.id || '';
        
        // Generate XPath locator (resolves labels to their inputs, text to interactive parents)
        const locator = resolveLocator(el);
        
        // Collect automation-relevant attributes
        const dataTestId = el.getAttribute('data-testid') || el.getAttribute('data-test-id') || el.getAttribute('data-cy') || '';
        const role = el.getAttribute('role') || '';
        const ariaLabel = el.getAttribute('aria-label') || '';
        const nameAttr = el.getAttribute('name') || '';
        
        const nodeData = {
            id: currentId,
            tagName: tagName,
            className: className,
            elementId: elId,
            path: path,
            textContent: textContent,
            depth: depth,
            locator: locator,
            dataTestId: dataTestId,
            role: role,
            ariaLabel: ariaLabel,
            nameAttr: nameAttr,
            boundingBox: {
                x: rect.x + window.scrollX,
                y: rect.y + window.scrollY,
                width: rect.width,
                height: rect.height,
            },
            styles: styles,
        };
        
        // Get additional attributes for specific elements
        if (tagName === 'IMG') {
            nodeData.src = el.src;
            nodeData.alt = el.alt;
        } else if (tagName === 'A') {
            nodeData.href = el.href;
        } else if (tagName === 'INPUT') {
            nodeData.inputType = el.type;
            nodeData.placeholder = el.placeholder;
            nodeData.value = el.value;
        } else if (tagName === 'BUTTON') {
            nodeData.buttonType = el.type;
        }
        
        results.push(nodeData);
        
        // Recursively process children
        for (const child of el.children) {
            walkDOM(child, depth + 1, path);
        }
    }
    
    // Start from body or specified root
    const root = config.rootSelector 
        ? document.querySelector(config.rootSelector) 
        : document.body;
    
    if (root) {
        walkDOM(root, 0, '');
    }
    
    return {
        url: window.location.href,
        title: document.title,
        viewport: {
            width: window.innerWidth,
            height: window.innerHeight,
            scrollWidth: document.documentElement.scrollWidth,
            scrollHeight: document.documentElement.scrollHeight,
        },
        rawNodes: results,
    };
}
"""


def normalize_dom_node(raw_node: dict) -> dict[str, Any]:
    """Convert raw DOM node data to Figma-compatible format."""
    tag_name = raw_node.get("tagName", "DIV")
    styles = raw_node.get("styles", {})
    bbox = raw_node.get("boundingBox", {})
    text_content = raw_node.get("textContent", "")
    
    node_type = determine_node_type(tag_name, styles)
    
    # Build name from element identifiers
    name_parts = [tag_name.lower()]
    if raw_node.get("elementId"):
        name_parts.append(f"#{raw_node['elementId']}")
    if raw_node.get("className"):
        classes = raw_node["className"].split()[:2]
        name_parts.extend(f".{c}" for c in classes)
    
    normalized = {
        "id": raw_node.get("id"),
        "name": " ".join(name_parts),
        "type": node_type,
        "tagName": tag_name,
        "className": raw_node.get("className", ""),
        "elementId": raw_node.get("elementId", ""),
        "locator": raw_node.get("locator", ""),
        "dataTestId": raw_node.get("dataTestId", ""),
        "role": raw_node.get("role", ""),
        "ariaLabel": raw_node.get("ariaLabel", ""),
        "nameAttr": raw_node.get("nameAttr", ""),
        "x": round(bbox.get("x", 0), 2),
        "y": round(bbox.get("y", 0), 2),
        "width": round(bbox.get("width", 0), 2),
        "height": round(bbox.get("height", 0), 2),
    }
    
    # Extract fills (background color)
    bg_color = styles.get("backgroundColor", "")
    if bg_color and bg_color != "rgba(0, 0, 0, 0)":
        normalized["fills"] = [{
            "type": "SOLID",
            "color": parse_color(bg_color),
        }]
    
    # Extract strokes (borders)
    border_width = parse_pixel_value(styles.get("borderWidth", "0"))
    border_color = styles.get("borderColor", "")
    if border_width > 0 and styles.get("borderStyle") not in ("none", ""):
        normalized["strokes"] = [{
            "type": "SOLID",
            "color": parse_color(border_color),
        }]
        normalized["strokeWeight"] = border_width
    
    # Extract corner radius
    corner_radius = parse_pixel_value(styles.get("borderRadius", "0"))
    if corner_radius > 0:
        normalized["cornerRadius"] = corner_radius
        
        # Individual corner radii
        radii = [
            parse_pixel_value(styles.get("borderTopLeftRadius", "0")),
            parse_pixel_value(styles.get("borderTopRightRadius", "0")),
            parse_pixel_value(styles.get("borderBottomRightRadius", "0")),
            parse_pixel_value(styles.get("borderBottomLeftRadius", "0")),
        ]
        if not all(r == corner_radius for r in radii):
            normalized["rectangleCornerRadii"] = radii
    
    # Type-specific properties
    if node_type == "TEXT":
        normalized["characters"] = text_content
        normalized["fontSize"] = parse_pixel_value(styles.get("fontSize", "16"))
        normalized["fontFamily"] = styles.get("fontFamily", "").split(",")[0].strip().strip('"\'')
        
        font_weight = styles.get("fontWeight", "400")
        normalized["fontWeight"] = int(font_weight) if font_weight.isdigit() else 400
        
        normalized["textAlignHorizontal"] = map_text_align(styles.get("textAlign", "left"))
        normalized["letterSpacing"] = parse_pixel_value(styles.get("letterSpacing", "0"))
        normalized["lineHeightPx"] = parse_pixel_value(styles.get("lineHeight", "0"))
        
        text_color = styles.get("color", "")
        if text_color:
            normalized["fills"] = [{
                "type": "SOLID",
                "color": parse_color(text_color),
            }]
    
    elif node_type == "FRAME":
        # Padding
        normalized["paddingTop"] = parse_pixel_value(styles.get("paddingTop", "0"))
        normalized["paddingRight"] = parse_pixel_value(styles.get("paddingRight", "0"))
        normalized["paddingBottom"] = parse_pixel_value(styles.get("paddingBottom", "0"))
        normalized["paddingLeft"] = parse_pixel_value(styles.get("paddingLeft", "0"))
        
        # Layout mode
        layout_mode = map_layout_mode(
            styles.get("display", ""),
            styles.get("flexDirection", "")
        )
        if layout_mode:
            normalized["layoutMode"] = layout_mode
        
        # Gap / item spacing
        gap = parse_pixel_value(styles.get("gap", "0"))
        if gap > 0:
            normalized["itemSpacing"] = gap
    
    elif node_type == "COMPONENT":
        normalized["componentType"] = tag_name
        if raw_node.get("inputType"):
            normalized["inputType"] = raw_node["inputType"]
        if raw_node.get("placeholder"):
            normalized["placeholder"] = raw_node["placeholder"]
    
    elif node_type == "IMAGE":
        if raw_node.get("src"):
            normalized["imageUrl"] = raw_node["src"]
        if raw_node.get("alt"):
            normalized["altText"] = raw_node["alt"]
    
    # Add opacity if not fully opaque
    opacity = float(styles.get("opacity", "1"))
    if opacity < 1:
        normalized["opacity"] = opacity
    
    # Add box shadow if present
    box_shadow = styles.get("boxShadow", "")
    if box_shadow and box_shadow != "none":
        normalized["effects"] = [{"type": "DROP_SHADOW", "raw": box_shadow}]
    
    return normalized


async def execute_login(
    page: Page,
    username: str,
    password: str,
    selectors: Optional[dict] = None,
) -> bool:
    """Execute login flow on the page."""
    default_selectors = {
        "username": 'input[type="email"], input[type="text"], input[name="username"], input[name="email"], #username, #email',
        "password": 'input[type="password"], input[name="password"], #password',
        "submit": 'button[type="submit"], input[type="submit"], button:has-text("Log in"), button:has-text("Sign in"), button:has-text("Login")',
    }
    
    sel = {**default_selectors, **(selectors or {})}
    
    try:
        # Wait for page to load
        await page.wait_for_load_state("networkidle", timeout=10000)
        
        # Find and fill username field
        username_field = await page.query_selector(sel["username"])
        if username_field:
            await username_field.fill(username)
        else:
            return False
        
        # Find and fill password field
        password_field = await page.query_selector(sel["password"])
        if password_field:
            await password_field.fill(password)
        else:
            return False
        
        # Click submit button
        submit_btn = await page.query_selector(sel["submit"])
        if submit_btn:
            await submit_btn.click()
        else:
            # Try pressing Enter as fallback
            await password_field.press("Enter")
        
        # Wait for navigation after login
        await page.wait_for_load_state("networkidle", timeout=30000)
        
        return True
        
    except Exception as e:
        print(f"Login error: {e}")
        return False


async def execute_post_login_steps(
    page: Page,
    steps: list[dict],
    timeout: int = 30000,
) -> bool:
    """
    Execute post-login steps like workspace selection.
    
    Each step can be:
    - {"action": "wait_for", "selector": ".some-element"} - Wait for element
    - {"action": "click", "selector": "button"} - Click an element
    - {"action": "click", "text": "Next"} - Click element by text
    - {"action": "click", "role": "button", "name": "Sign In"} - Click by ARIA role
    - {"action": "click", "test_id": "next"} - Click by data-testid
    - {"action": "click", "nth": 0, "selector": ".dropdown"} - Click nth matching element
    - {"action": "select", "selector": "select", "value": "option_value"} - Select dropdown by value
    - {"action": "select", "selector": "select", "index": 1} - Select dropdown by index
    - {"action": "select", "selector": "select", "label": "Option Text"} - Select dropdown by label
    - {"action": "fill", "selector": "input", "value": "text"} - Fill input field
    - {"action": "wait", "duration": 2000} - Wait for specified milliseconds
    """
    all_success = True
    
    for step in steps:
        action = step.get("action", "")
        selector = step.get("selector")
        text = step.get("text")
        role = step.get("role")
        name = step.get("name")
        test_id = step.get("test_id")
        nth = step.get("nth")
        
        try:
            if action == "wait_for":
                if selector:
                    await page.wait_for_selector(selector, timeout=timeout)
                    print(f"[VQA] Waited for selector: {selector}")
                elif text:
                    await page.get_by_text(text).wait_for(timeout=timeout)
                    print(f"[VQA] Waited for text: {text}")
                elif test_id:
                    await page.get_by_test_id(test_id).wait_for(timeout=timeout)
                    print(f"[VQA] Waited for test_id: {test_id}")
            
            elif action == "click":
                # Determine what to click based on provided parameters
                locator = None
                click_desc = ""
                
                if role and name:
                    locator = page.get_by_role(role, name=name)
                    click_desc = f"role={role}, name={name}"
                elif test_id:
                    locator = page.get_by_test_id(test_id)
                    click_desc = f"test_id={test_id}"
                elif selector and nth is not None:
                    locator = page.locator(selector).nth(nth)
                    click_desc = f"selector={selector}, nth={nth}"
                elif selector:
                    # For react-select and similar dynamic components, try multiple selector variations
                    original_selector = selector
                    locator = page.locator(selector).first
                    click_desc = f"selector={selector}"
                    
                    # Check if element exists, if not try alternative selectors for react-select
                    try:
                        count = await locator.count()
                        if count == 0 and "css-" in selector:
                            # Try more stable react-select selectors
                            alt_selectors = [
                                '[class*="-control"]',
                                '[class*="control"]',
                                'div[class*="react-select"]',
                                '[class*="indicatorContainer"]',
                            ]
                            for alt in alt_selectors:
                                alt_locator = page.locator(alt).first
                                if await alt_locator.count() > 0:
                                    locator = alt_locator
                                    click_desc = f"selector={alt} (fallback from {original_selector})"
                                    print(f"[VQA] Using fallback selector: {alt}")
                                    break
                    except Exception:
                        pass
                elif text:
                    locator = page.get_by_text(text, exact=True)
                    click_desc = f"text={text}"
                
                if locator:
                    # Wait for element to be visible before clicking
                    try:
                        await locator.wait_for(state="visible", timeout=timeout)
                    except Exception:
                        # If exact text match fails, try partial match
                        if text and not selector:
                            locator = page.get_by_text(text).first
                            await locator.wait_for(state="visible", timeout=timeout)
                    
                    await locator.click(timeout=timeout)
                    print(f"[VQA] Clicked: {click_desc}")
                    
                    # Wait for any navigation/loading
                    try:
                        await page.wait_for_load_state("networkidle", timeout=5000)
                    except Exception:
                        pass
            
            elif action == "select":
                if not selector:
                    continue
                    
                if "value" in step:
                    await page.select_option(selector, value=step["value"], timeout=timeout)
                    print(f"[VQA] Selected value: {step['value']} in {selector}")
                elif "index" in step:
                    await page.select_option(selector, index=step["index"], timeout=timeout)
                    print(f"[VQA] Selected index: {step['index']} in {selector}")
                elif "label" in step:
                    await page.select_option(selector, label=step["label"], timeout=timeout)
                    print(f"[VQA] Selected label: {step['label']} in {selector}")
            
            elif action == "fill":
                if selector and "value" in step:
                    await page.fill(selector, step["value"], timeout=timeout)
                    print(f"[VQA] Filled {selector} with value")
                elif test_id and "value" in step:
                    await page.get_by_test_id(test_id).fill(step["value"], timeout=timeout)
                    print(f"[VQA] Filled test_id={test_id} with value")
            
            elif action == "wait":
                duration = step.get("duration", 1000)
                await page.wait_for_timeout(duration)
                print(f"[VQA] Waited {duration}ms")
            
            # Small delay between steps for stability
            await page.wait_for_timeout(500)
            
        except Exception as e:
            print(f"[VQA] Warning: {action} action failed: {e}")
            all_success = False
            # Continue with next step instead of aborting
            continue
    
    # Wait for page to stabilize after all steps
    try:
        await page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass
    
    if all_success:
        print("[VQA] Post-login steps completed successfully.")
    else:
        print("[VQA] Post-login steps completed with some failures.")
    
    return all_success


async def wait_for_react_render(page: Page, timeout: int = 10000) -> None:
    """Wait for React/SPA to finish rendering."""
    try:
        # Wait for network to be idle
        await page.wait_for_load_state("networkidle", timeout=timeout)
        
        # Additional wait for React hydration
        await page.evaluate("""
            () => new Promise((resolve) => {
                // Check if React DevTools hook exists (React app indicator)
                if (window.__REACT_DEVTOOLS_GLOBAL_HOOK__) {
                    // Give React a moment to finish rendering
                    requestIdleCallback ? requestIdleCallback(resolve) : setTimeout(resolve, 500);
                } else {
                    // Not a React app, just wait a bit
                    setTimeout(resolve, 300);
                }
            })
        """)
        
    except Exception:
        # Timeout is acceptable, continue with extraction
        pass


async def extract_from_url(
    url: str,
    credentials: Optional[dict] = None,
    login_url: Optional[str] = None,
    root_selector: Optional[str] = None,
    max_depth: int = 50,
    viewport: Optional[dict] = None,
    wait_for_selector: Optional[str] = None,
    screenshot: bool = False,
    post_login_steps: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """
    Extract visual properties from a web page.
    
    Args:
        url: Target URL to extract from
        credentials: Optional dict with 'username' and 'password'
        login_url: URL of login page (if different from target)
        root_selector: CSS selector to start extraction from (default: body)
        max_depth: Maximum DOM traversal depth
        viewport: Custom viewport {width, height}
        wait_for_selector: Wait for specific element before extraction
        screenshot: Capture screenshot as base64
        post_login_steps: List of actions to perform after login (e.g., workspace selection)
    
    Returns:
        Figma-compatible JSON structure
    """
    async with async_playwright() as p:
        browser: Browser = await p.chromium.launch(headless=True)
        
        context = await browser.new_context(
            viewport=viewport or {"width": 1920, "height": 1080},
            device_scale_factor=1,
        )
        
        page = await context.new_page()
        
        try:
            # Handle authentication if credentials provided
            if credentials and credentials.get("username") and credentials.get("password"):
                auth_url = login_url or url
                await page.goto(auth_url, wait_until="domcontentloaded")
                
                login_success = await execute_login(
                    page,
                    credentials["username"],
                    credentials["password"],
                    credentials.get("selectors"),
                )
                
                if not login_success:
                    raise Exception("Login failed - could not find login form elements")
                
                # Execute post-login steps (e.g., workspace selection)
                if post_login_steps:
                    steps_success = await execute_post_login_steps(page, post_login_steps)
                    if not steps_success:
                        print("Warning: Some post-login steps may have failed")
                
                # Navigate to target if different from login
                if login_url and login_url != url:
                    await page.goto(url, wait_until="domcontentloaded")
            else:
                await page.goto(url, wait_until="domcontentloaded")
            
            # Wait for React/SPA rendering
            await wait_for_react_render(page)
            
            # Wait for specific selector if provided
            if wait_for_selector:
                await page.wait_for_selector(wait_for_selector, timeout=10000)
            
            # Execute DOM walker script
            raw_data = await page.evaluate(
                DOM_WALKER_SCRIPT,
                {"maxDepth": max_depth, "rootSelector": root_selector},
            )
            
            # Capture screenshot if requested
            screenshot_base64 = None
            if screenshot:
                screenshot_bytes = await page.screenshot(full_page=True)
                import base64
                screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            
            # Normalize nodes to Figma format
            nodes = [normalize_dom_node(node) for node in raw_data.get("rawNodes", [])]
            
            # Filter out zero-size nodes
            nodes = [n for n in nodes if n["width"] > 0 or n["height"] > 0]
            
            result = {
                "source": "web",
                "url": raw_data.get("url", url),
                "title": raw_data.get("title", ""),
                "viewport": raw_data.get("viewport", {}),
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "total_nodes_extracted": len(nodes),
                "nodes": nodes,
            }
            
            if screenshot_base64:
                result["screenshot"] = screenshot_base64
            
            return result
            
        finally:
            await browser.close()


# Synchronous wrapper for non-async contexts
def extract_from_url_sync(
    url: str,
    credentials: Optional[dict] = None,
    **kwargs,
) -> dict[str, Any]:
    """Synchronous wrapper for extract_from_url."""
    import asyncio
    return asyncio.run(extract_from_url(url, credentials, **kwargs))