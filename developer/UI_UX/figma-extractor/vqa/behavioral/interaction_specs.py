"""
Component interaction specifications for behavioral testing.

This module defines which interactions to test for each component type,
what CSS properties are expected to change, and validation rules from
the design system.

Interaction specs are used by InteractionRunner to determine:
1. What interactions to perform (hover, click, focus, scroll)
2. What properties should change after the interaction
3. What design system rules should be validated
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum


class InteractionType(str, Enum):
    """Types of interactions that can be tested."""
    HOVER = "hover"
    CLICK = "click"
    FOCUS = "focus"
    BLUR = "blur"
    SCROLL = "scroll"
    TOGGLE_ON = "toggle_on"
    TOGGLE_OFF = "toggle_off"
    DROPDOWN_OPEN = "dropdown_open"
    DROPDOWN_CLOSE = "dropdown_close"
    TAB_SELECT = "tab_select"
    PRESS = "press"
    DRAG = "drag"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"


class ComponentType(str, Enum):
    """Component types that can be tested."""
    BUTTON = "button"
    BUTTON_PRIMARY = "button_primary"
    BUTTON_SECONDARY = "button_secondary"
    BUTTON_ICON = "button_icon"
    DROPDOWN = "dropdown"
    DROPDOWN_TRIGGER = "dropdown_trigger"
    DROPDOWN_OPTION = "dropdown_option"
    TOGGLE = "toggle"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    TAB = "tab"
    TAB_BAR = "tab_bar"
    INPUT = "input"
    INPUT_TEXT = "input_text"
    INPUT_PASSWORD = "input_password"
    INPUT_SEARCH = "input_search"
    TEXTAREA = "textarea"
    SELECT = "select"
    TOAST = "toast"
    TOOLTIP = "tooltip"
    MODAL = "modal"
    POPOVER = "popover"
    TABLE_HEADER = "table_header"
    TABLE_ROW = "table_row"
    TABLE_CELL = "table_cell"
    FILTER_ICON = "filter_icon"
    SCROLLABLE_CONTAINER = "scrollable_container"
    CARD = "card"
    LINK = "link"
    NAV_ITEM = "nav_item"
    ACCORDION = "accordion"
    SLIDER = "slider"
    DATE_PICKER = "date_picker"
    FILE_UPLOAD = "file_upload"
    MENU_ITEM = "menu_item"
    CHIP = "chip"
    BADGE = "badge"
    AVATAR = "avatar"
    UNKNOWN = "unknown"


@dataclass
class InteractionSpec:
    """Specification for testing a specific interaction on a component type."""
    component_type: ComponentType
    interaction_type: InteractionType
    expected_property_changes: List[str]
    validation_rules: List[str] = field(default_factory=list)
    undo_action: Optional[str] = None
    wait_ms_after: int = 300
    requires_visible: bool = True
    skip_if_disabled: bool = True
    capture_full_page: bool = False
    capture_viewport_only: bool = False
    
    @property
    def id(self) -> str:
        """Unique identifier for this spec."""
        return f"{self.component_type.value}_{self.interaction_type.value}"


@dataclass
class ComponentSpec:
    """Full specification for testing a component type."""
    component_type: ComponentType
    default_interactions: List[InteractionType]
    interaction_specs: Dict[InteractionType, InteractionSpec]
    detection_selectors: List[str] = field(default_factory=list)
    detection_attributes: Dict[str, List[str]] = field(default_factory=dict)
    detection_text_patterns: List[str] = field(default_factory=list)
    priority: int = 1


# Define expected CSS property changes for each interaction type
INTERACTION_EXPECTED_CHANGES: Dict[InteractionType, List[str]] = {
    InteractionType.HOVER: [
        "background-color",
        "box-shadow",
        "transform",
        "cursor",
        "color",
        "border-color",
        "opacity",
    ],
    InteractionType.FOCUS: [
        "outline",
        "box-shadow",
        "border-color",
        "outline-offset",
    ],
    InteractionType.CLICK: [
        "background-color",
        "transform",
        "box-shadow",
        "opacity",
    ],
    InteractionType.BLUR: [
        "outline",
        "box-shadow",
        "border-color",
    ],
    InteractionType.SCROLL: [
        "position",
        "box-shadow",
        "transform",
        "top",
        "opacity",
    ],
    InteractionType.TOGGLE_ON: [
        "background-color",
        "transform",
        "border-color",
    ],
    InteractionType.TOGGLE_OFF: [
        "background-color",
        "transform",
        "border-color",
    ],
    InteractionType.DROPDOWN_OPEN: [
        "height",
        "max-height",
        "overflow",
        "box-shadow",
        "opacity",
        "visibility",
        "display",
    ],
    InteractionType.DROPDOWN_CLOSE: [
        "height",
        "max-height",
        "overflow",
        "opacity",
        "visibility",
        "display",
    ],
    InteractionType.TAB_SELECT: [
        "font-weight",
        "border-bottom",
        "color",
        "background-color",
    ],
    InteractionType.PRESS: [
        "background-color",
        "transform",
        "box-shadow",
    ],
    InteractionType.DOUBLE_CLICK: [
        "user-select",
        "background-color",
    ],
}


# Component-specific interaction specifications
COMPONENT_SPECS: Dict[ComponentType, ComponentSpec] = {
    ComponentType.BUTTON: ComponentSpec(
        component_type=ComponentType.BUTTON,
        default_interactions=[
            InteractionType.HOVER,
            InteractionType.FOCUS,
            InteractionType.CLICK,
        ],
        interaction_specs={
            InteractionType.HOVER: InteractionSpec(
                component_type=ComponentType.BUTTON,
                interaction_type=InteractionType.HOVER,
                expected_property_changes=[
                    "background-color",
                    "box-shadow",
                    "transform",
                    "cursor",
                ],
                validation_rules=[
                    "Hover state should have distinct visual change",
                    "Cursor should be pointer on hover",
                ],
                undo_action="move_away",
                wait_ms_after=300,
            ),
            InteractionType.FOCUS: InteractionSpec(
                component_type=ComponentType.BUTTON,
                interaction_type=InteractionType.FOCUS,
                expected_property_changes=[
                    "outline",
                    "box-shadow",
                ],
                validation_rules=[
                    "Focus ring should be visible",
                    "Focus ring color should match design system",
                ],
                undo_action="blur",
                wait_ms_after=200,
            ),
            InteractionType.CLICK: InteractionSpec(
                component_type=ComponentType.BUTTON,
                interaction_type=InteractionType.CLICK,
                expected_property_changes=[
                    "background-color",
                    "transform",
                ],
                validation_rules=[
                    "Click should have visual feedback",
                ],
                undo_action="escape",
                wait_ms_after=500,
            ),
        },
        detection_selectors=[
            "button",
            "[role='button']",
            ".btn",
            ".button",
            "input[type='button']",
            "input[type='submit']",
        ],
        detection_attributes={
            "role": ["button"],
            "type": ["button", "submit"],
        },
        priority=2,
    ),
    ComponentType.DROPDOWN: ComponentSpec(
        component_type=ComponentType.DROPDOWN,
        default_interactions=[
            InteractionType.CLICK,
        ],
        interaction_specs={
            InteractionType.CLICK: InteractionSpec(
                component_type=ComponentType.DROPDOWN,
                interaction_type=InteractionType.CLICK,
                expected_property_changes=[
                    "height",
                    "max-height",
                    "overflow",
                    "box-shadow",
                    "opacity",
                ],
                validation_rules=[
                    "Dropdown panel width should match trigger width",
                    "Dropdown should have shadow when open",
                    "Dropdown options should be visible",
                ],
                undo_action="escape",
                wait_ms_after=500,
                capture_full_page=True,
            ),
        },
        detection_selectors=[
            "select",
            "[role='listbox']",
            "[role='combobox']",
            ".dropdown",
            ".select",
        ],
        detection_attributes={
            "role": ["listbox", "combobox"],
        },
        priority=2,
    ),
    ComponentType.TOGGLE: ComponentSpec(
        component_type=ComponentType.TOGGLE,
        default_interactions=[
            InteractionType.CLICK,
        ],
        interaction_specs={
            InteractionType.CLICK: InteractionSpec(
                component_type=ComponentType.TOGGLE,
                interaction_type=InteractionType.CLICK,
                expected_property_changes=[
                    "background-color",
                    "transform",
                ],
                validation_rules=[
                    "All toggles should be same width",
                    "Toggle state should be visually distinct",
                ],
                undo_action="click_again",
                wait_ms_after=300,
            ),
        },
        detection_selectors=[
            "[role='switch']",
            ".toggle",
            ".switch",
            "input[type='checkbox'][role='switch']",
        ],
        detection_attributes={
            "role": ["switch"],
        },
        priority=2,
    ),
    ComponentType.CHECKBOX: ComponentSpec(
        component_type=ComponentType.CHECKBOX,
        default_interactions=[
            InteractionType.CLICK,
            InteractionType.FOCUS,
        ],
        interaction_specs={
            InteractionType.CLICK: InteractionSpec(
                component_type=ComponentType.CHECKBOX,
                interaction_type=InteractionType.CLICK,
                expected_property_changes=[
                    "background-color",
                    "border-color",
                ],
                validation_rules=[
                    "Checkbox should have correct aspect ratio",
                    "Checked state should be distinct",
                ],
                undo_action="click_again",
                wait_ms_after=200,
            ),
            InteractionType.FOCUS: InteractionSpec(
                component_type=ComponentType.CHECKBOX,
                interaction_type=InteractionType.FOCUS,
                expected_property_changes=[
                    "outline",
                    "box-shadow",
                ],
                validation_rules=[
                    "Focus ring should be visible around checkbox",
                ],
                undo_action="blur",
                wait_ms_after=200,
            ),
        },
        detection_selectors=[
            "input[type='checkbox']",
            "[role='checkbox']",
            ".checkbox",
        ],
        detection_attributes={
            "role": ["checkbox"],
            "type": ["checkbox"],
        },
        priority=1,
    ),
    ComponentType.TAB: ComponentSpec(
        component_type=ComponentType.TAB,
        default_interactions=[
            InteractionType.CLICK,
            InteractionType.HOVER,
        ],
        interaction_specs={
            InteractionType.CLICK: InteractionSpec(
                component_type=ComponentType.TAB,
                interaction_type=InteractionType.CLICK,
                expected_property_changes=[
                    "font-weight",
                    "border-bottom",
                    "color",
                    "background-color",
                ],
                validation_rules=[
                    "Selected tab weight should match design system",
                    "Tab bar should remain fixed during scroll",
                    "Selected indicator should be visible",
                ],
                wait_ms_after=400,
            ),
            InteractionType.HOVER: InteractionSpec(
                component_type=ComponentType.TAB,
                interaction_type=InteractionType.HOVER,
                expected_property_changes=[
                    "background-color",
                    "color",
                ],
                validation_rules=[
                    "Hover sub-header should not have a gradient",
                ],
                undo_action="move_away",
                wait_ms_after=200,
            ),
        },
        detection_selectors=[
            "[role='tab']",
            ".tab",
            ".nav-tab",
        ],
        detection_attributes={
            "role": ["tab"],
        },
        priority=2,
    ),
    ComponentType.INPUT: ComponentSpec(
        component_type=ComponentType.INPUT,
        default_interactions=[
            InteractionType.FOCUS,
        ],
        interaction_specs={
            InteractionType.FOCUS: InteractionSpec(
                component_type=ComponentType.INPUT,
                interaction_type=InteractionType.FOCUS,
                expected_property_changes=[
                    "border-color",
                    "box-shadow",
                    "outline",
                ],
                validation_rules=[
                    "Input should have visible focus state",
                    "Label should float or change on focus",
                ],
                undo_action="blur",
                wait_ms_after=200,
            ),
        },
        detection_selectors=[
            "input[type='text']",
            "input[type='email']",
            "input[type='password']",
            "input[type='number']",
            "input[type='tel']",
            "input[type='url']",
            "input:not([type])",
            "textarea",
            "[role='textbox']",
            ".input",
            ".form-control",
        ],
        detection_attributes={
            "role": ["textbox"],
        },
        priority=1,
    ),
    ComponentType.TOAST: ComponentSpec(
        component_type=ComponentType.TOAST,
        default_interactions=[
            InteractionType.HOVER,
        ],
        interaction_specs={
            InteractionType.HOVER: InteractionSpec(
                component_type=ComponentType.TOAST,
                interaction_type=InteractionType.HOVER,
                expected_property_changes=[],
                validation_rules=[
                    "Toast remains visible while hovered",
                    "Dismiss animation plays in background",
                    "Close button should be accessible",
                ],
                undo_action="move_away",
                wait_ms_after=500,
            ),
        },
        detection_selectors=[
            "[role='alert']",
            "[role='status']",
            ".toast",
            ".notification",
            ".snackbar",
        ],
        detection_attributes={
            "role": ["alert", "status"],
        },
        priority=3,
    ),
    ComponentType.TABLE_HEADER: ComponentSpec(
        component_type=ComponentType.TABLE_HEADER,
        default_interactions=[
            InteractionType.CLICK,
            InteractionType.HOVER,
        ],
        interaction_specs={
            InteractionType.CLICK: InteractionSpec(
                component_type=ComponentType.TABLE_HEADER,
                interaction_type=InteractionType.CLICK,
                expected_property_changes=[],
                validation_rules=[
                    "Sort indicator should appear or change",
                    "Header height should not change based on CTA presence",
                    "Consistent header style (single vs multi-line)",
                ],
                wait_ms_after=500,
            ),
            InteractionType.HOVER: InteractionSpec(
                component_type=ComponentType.TABLE_HEADER,
                interaction_type=InteractionType.HOVER,
                expected_property_changes=[
                    "background-color",
                    "cursor",
                ],
                validation_rules=[
                    "Sortable headers should show hover state",
                ],
                undo_action="move_away",
                wait_ms_after=200,
            ),
        },
        detection_selectors=[
            "th",
            "[role='columnheader']",
            ".table-header",
        ],
        detection_attributes={
            "role": ["columnheader"],
        },
        priority=1,
    ),
    ComponentType.FILTER_ICON: ComponentSpec(
        component_type=ComponentType.FILTER_ICON,
        default_interactions=[
            InteractionType.CLICK,
        ],
        interaction_specs={
            InteractionType.CLICK: InteractionSpec(
                component_type=ComponentType.FILTER_ICON,
                interaction_type=InteractionType.CLICK,
                expected_property_changes=[],
                validation_rules=[
                    "Table width should not change when filter opens",
                    "No radius between table header and table when filter is active",
                    "Filter panel should align properly",
                ],
                undo_action="escape",
                wait_ms_after=500,
                capture_full_page=True,
            ),
        },
        detection_selectors=[
            "[aria-label*='filter']",
            "[aria-label*='Filter']",
            ".filter-icon",
            ".filter-button",
        ],
        priority=2,
    ),
    ComponentType.SCROLLABLE_CONTAINER: ComponentSpec(
        component_type=ComponentType.SCROLLABLE_CONTAINER,
        default_interactions=[
            InteractionType.SCROLL,
        ],
        interaction_specs={
            InteractionType.SCROLL: InteractionSpec(
                component_type=ComponentType.SCROLLABLE_CONTAINER,
                interaction_type=InteractionType.SCROLL,
                expected_property_changes=[
                    "box-shadow",
                ],
                validation_rules=[
                    "Scrollbar should have 2px padding from content",
                    "Frozen columns should have shadow",
                    "Shadow should appear on scroll",
                    "Sticky elements should remain fixed",
                ],
                wait_ms_after=300,
            ),
        },
        detection_selectors=[
            "[style*='overflow: auto']",
            "[style*='overflow: scroll']",
            "[style*='overflow-y: auto']",
            "[style*='overflow-y: scroll']",
            ".scrollable",
            ".scroll-container",
        ],
        priority=1,
    ),
    ComponentType.CARD: ComponentSpec(
        component_type=ComponentType.CARD,
        default_interactions=[
            InteractionType.HOVER,
        ],
        interaction_specs={
            InteractionType.HOVER: InteractionSpec(
                component_type=ComponentType.CARD,
                interaction_type=InteractionType.HOVER,
                expected_property_changes=[
                    "box-shadow",
                    "transform",
                    "border-color",
                ],
                validation_rules=[
                    "Cards should have consistent shadow on hover",
                    "Card elevation should increase on hover",
                ],
                undo_action="move_away",
                wait_ms_after=300,
            ),
        },
        detection_selectors=[
            ".card",
            "[role='article']",
            ".panel",
        ],
        detection_attributes={
            "role": ["article"],
        },
        priority=1,
    ),
    ComponentType.LINK: ComponentSpec(
        component_type=ComponentType.LINK,
        default_interactions=[
            InteractionType.HOVER,
            InteractionType.FOCUS,
        ],
        interaction_specs={
            InteractionType.HOVER: InteractionSpec(
                component_type=ComponentType.LINK,
                interaction_type=InteractionType.HOVER,
                expected_property_changes=[
                    "color",
                    "text-decoration",
                ],
                validation_rules=[
                    "Links should have hover state",
                    "Underline should appear or change on hover",
                ],
                undo_action="move_away",
                wait_ms_after=200,
            ),
            InteractionType.FOCUS: InteractionSpec(
                component_type=ComponentType.LINK,
                interaction_type=InteractionType.FOCUS,
                expected_property_changes=[
                    "outline",
                    "box-shadow",
                ],
                validation_rules=[
                    "Links should have visible focus indicator",
                ],
                undo_action="blur",
                wait_ms_after=200,
            ),
        },
        detection_selectors=[
            "a",
            "[role='link']",
        ],
        detection_attributes={
            "role": ["link"],
        },
        priority=1,
    ),
    ComponentType.MODAL: ComponentSpec(
        component_type=ComponentType.MODAL,
        default_interactions=[],
        interaction_specs={},
        detection_selectors=[
            "[role='dialog']",
            ".modal",
            ".dialog",
        ],
        detection_attributes={
            "role": ["dialog"],
        },
        priority=3,
    ),
    ComponentType.TOOLTIP: ComponentSpec(
        component_type=ComponentType.TOOLTIP,
        default_interactions=[],
        interaction_specs={},
        detection_selectors=[
            "[role='tooltip']",
            ".tooltip",
        ],
        detection_attributes={
            "role": ["tooltip"],
        },
        priority=1,
    ),
    ComponentType.ACCORDION: ComponentSpec(
        component_type=ComponentType.ACCORDION,
        default_interactions=[
            InteractionType.CLICK,
        ],
        interaction_specs={
            InteractionType.CLICK: InteractionSpec(
                component_type=ComponentType.ACCORDION,
                interaction_type=InteractionType.CLICK,
                expected_property_changes=[
                    "height",
                    "max-height",
                    "transform",
                ],
                validation_rules=[
                    "Accordion content should expand smoothly",
                    "Arrow/icon should rotate on expand",
                ],
                undo_action="click_again",
                wait_ms_after=500,
            ),
        },
        detection_selectors=[
            "[role='button'][aria-expanded]",
            ".accordion",
            ".collapsible",
        ],
        priority=1,
    ),
    ComponentType.SLIDER: ComponentSpec(
        component_type=ComponentType.SLIDER,
        default_interactions=[
            InteractionType.FOCUS,
            InteractionType.DRAG,
        ],
        interaction_specs={
            InteractionType.FOCUS: InteractionSpec(
                component_type=ComponentType.SLIDER,
                interaction_type=InteractionType.FOCUS,
                expected_property_changes=[
                    "outline",
                    "box-shadow",
                ],
                validation_rules=[
                    "Slider thumb should have focus indicator",
                ],
                undo_action="blur",
                wait_ms_after=200,
            ),
        },
        detection_selectors=[
            "input[type='range']",
            "[role='slider']",
            ".slider",
        ],
        detection_attributes={
            "role": ["slider"],
        },
        priority=1,
    ),
}


def get_spec_for_component(component_type: ComponentType) -> Optional[ComponentSpec]:
    """Get the specification for a component type.
    
    Args:
        component_type: Type of component
    
    Returns:
        ComponentSpec if found, None otherwise
    """
    return COMPONENT_SPECS.get(component_type)


def get_interaction_spec(
    component_type: ComponentType,
    interaction_type: InteractionType,
) -> Optional[InteractionSpec]:
    """Get the specification for a specific interaction on a component.
    
    Args:
        component_type: Type of component
        interaction_type: Type of interaction
    
    Returns:
        InteractionSpec if found, None otherwise
    """
    spec = COMPONENT_SPECS.get(component_type)
    if not spec:
        return None
    return spec.interaction_specs.get(interaction_type)


def get_default_interactions(component_type: ComponentType) -> List[InteractionType]:
    """Get default interactions to test for a component type.
    
    Args:
        component_type: Type of component
    
    Returns:
        List of interaction types
    """
    spec = COMPONENT_SPECS.get(component_type)
    if not spec:
        return [InteractionType.HOVER]
    return spec.default_interactions


def get_expected_changes(interaction_type: InteractionType) -> List[str]:
    """Get expected CSS property changes for an interaction type.
    
    Args:
        interaction_type: Type of interaction
    
    Returns:
        List of CSS property names expected to change
    """
    return INTERACTION_EXPECTED_CHANGES.get(
        interaction_type,
        ["background-color", "box-shadow"]
    )


def detect_component_type(
    element_data: Dict,
) -> ComponentType:
    """Detect the component type from element attributes.
    
    Args:
        element_data: Dict with tag_name, role, class_name, type, aria-* attributes
    
    Returns:
        Detected ComponentType
    """
    tag_name = element_data.get("tag_name", "").lower()
    role = element_data.get("role", "").lower()
    class_name = element_data.get("class_name", "").lower()
    input_type = element_data.get("type", "").lower()
    aria_expanded = element_data.get("aria-expanded")
    aria_label = element_data.get("aria-label", "").lower()
    
    if role == "switch" or "toggle" in class_name or "switch" in class_name:
        return ComponentType.TOGGLE
    
    if role == "checkbox" or (tag_name == "input" and input_type == "checkbox"):
        return ComponentType.CHECKBOX
    
    if role == "radio" or (tag_name == "input" and input_type == "radio"):
        return ComponentType.RADIO
    
    if role == "tab":
        return ComponentType.TAB
    
    if role in ("listbox", "combobox") or tag_name == "select":
        return ComponentType.DROPDOWN
    
    if role == "button" or tag_name == "button" or input_type in ("button", "submit"):
        if "icon" in class_name:
            return ComponentType.BUTTON_ICON
        if "primary" in class_name:
            return ComponentType.BUTTON_PRIMARY
        if "secondary" in class_name:
            return ComponentType.BUTTON_SECONDARY
        return ComponentType.BUTTON
    
    if role == "textbox" or tag_name == "textarea":
        return ComponentType.TEXTAREA if tag_name == "textarea" else ComponentType.INPUT
    
    if tag_name == "input" and input_type in ("text", "email", "password", "number", "tel", "url", ""):
        return ComponentType.INPUT
    
    if role in ("alert", "status") or "toast" in class_name or "notification" in class_name:
        return ComponentType.TOAST
    
    if role == "tooltip" or "tooltip" in class_name:
        return ComponentType.TOOLTIP
    
    if role == "dialog" or "modal" in class_name:
        return ComponentType.MODAL
    
    if tag_name == "th" or role == "columnheader":
        return ComponentType.TABLE_HEADER
    
    if tag_name == "tr" or role == "row":
        return ComponentType.TABLE_ROW
    
    if tag_name == "td" or role == "cell":
        return ComponentType.TABLE_CELL
    
    if "filter" in aria_label or "filter" in class_name:
        return ComponentType.FILTER_ICON
    
    if tag_name == "a" or role == "link":
        return ComponentType.LINK
    
    if "card" in class_name or role == "article":
        return ComponentType.CARD
    
    if aria_expanded is not None or "accordion" in class_name:
        return ComponentType.ACCORDION
    
    if role == "slider" or (tag_name == "input" and input_type == "range"):
        return ComponentType.SLIDER
    
    return ComponentType.UNKNOWN


def get_interactions_for_element(
    element_data: Dict,
) -> List[InteractionType]:
    """Determine which interactions to test for an element.
    
    Combines component type detection with heuristics from element data.
    
    Args:
        element_data: Dict with element attributes
    
    Returns:
        List of interactions to test
    """
    component_type = detect_component_type(element_data)
    default_interactions = get_default_interactions(component_type)
    
    interactions = list(default_interactions)
    
    aria_label = element_data.get("aria-label", "").lower()
    class_name = element_data.get("class_name", "").lower()
    
    if "filter" in aria_label or "filter" in class_name:
        if InteractionType.CLICK not in interactions:
            interactions.append(InteractionType.CLICK)
    
    if "scroll" in class_name or element_data.get("has_overflow"):
        if InteractionType.SCROLL not in interactions:
            interactions.append(InteractionType.SCROLL)
    
    return interactions


def get_undo_action_for_interaction(
    interaction_type: InteractionType,
    component_type: ComponentType = None,
) -> str:
    """Get the undo action needed to restore state after an interaction.
    
    Args:
        interaction_type: The interaction that was performed
        component_type: Optional component type for specific undo logic
    
    Returns:
        Undo action string: "move_away", "blur", "escape", "click_again", "scroll_back", etc.
    """
    spec = None
    if component_type:
        spec = get_interaction_spec(component_type, interaction_type)
    
    if spec and spec.undo_action:
        return spec.undo_action
    
    default_undo_actions = {
        InteractionType.HOVER: "move_away",
        InteractionType.FOCUS: "blur",
        InteractionType.CLICK: "escape",
        InteractionType.TOGGLE_ON: "click_again",
        InteractionType.TOGGLE_OFF: "click_again",
        InteractionType.DROPDOWN_OPEN: "escape",
        InteractionType.DROPDOWN_CLOSE: None,
        InteractionType.SCROLL: "scroll_back",
        InteractionType.TAB_SELECT: None,
        InteractionType.PRESS: None,
    }
    
    return default_undo_actions.get(interaction_type, "escape")


def get_wait_time_for_interaction(
    interaction_type: InteractionType,
    component_type: ComponentType = None,
) -> int:
    """Get the recommended wait time after an interaction.
    
    Args:
        interaction_type: The interaction type
        component_type: Optional component type for specific timing
    
    Returns:
        Wait time in milliseconds
    """
    spec = None
    if component_type:
        spec = get_interaction_spec(component_type, interaction_type)
    
    if spec:
        return spec.wait_ms_after
    
    default_wait_times = {
        InteractionType.HOVER: 300,
        InteractionType.FOCUS: 200,
        InteractionType.CLICK: 500,
        InteractionType.DROPDOWN_OPEN: 500,
        InteractionType.SCROLL: 300,
        InteractionType.TOGGLE_ON: 300,
        InteractionType.TOGGLE_OFF: 300,
        InteractionType.TAB_SELECT: 400,
    }
    
    return default_wait_times.get(interaction_type, 300)


def get_validation_rules(
    component_type: ComponentType,
    interaction_type: InteractionType,
) -> List[str]:
    """Get validation rules for a component/interaction combination.
    
    These rules are human-readable descriptions of what should be validated.
    
    Args:
        component_type: The component type
        interaction_type: The interaction type
    
    Returns:
        List of validation rule descriptions
    """
    spec = get_interaction_spec(component_type, interaction_type)
    if spec:
        return spec.validation_rules
    return []


COMMON_BEHAVIORAL_ISSUES = {
    "hover_gradient": "Hover state should have gradient but shows solid color",
    "hover_no_change": "Element should change on hover but remains static",
    "focus_ring_missing": "Focus ring/outline is missing on focusable element",
    "focus_ring_color": "Focus ring color doesn't match design system",
    "disabled_same_as_active": "Disabled state looks the same as active state",
    "disabled_cursor": "Disabled element should have not-allowed cursor",
    "selected_no_indicator": "Selected state has no visual indicator",
    "toast_position": "Toast appears in wrong position",
    "toast_timing": "Toast dismisses too quickly or doesn't auto-dismiss",
    "dropdown_width_mismatch": "Dropdown width doesn't match trigger element",
    "dropdown_no_shadow": "Open dropdown should have elevation shadow",
    "sticky_not_sticky": "Element should be sticky/fixed but scrolls with page",
    "scroll_shadow_missing": "Shadow should appear on scroll but doesn't",
    "toggle_inconsistent_width": "Toggles have inconsistent widths",
    "tab_indicator_missing": "Selected tab indicator is missing",
    "card_shadow_missing": "Card shadow is missing on hover",
    "input_focus_missing": "Input focus state is not visible",
}
