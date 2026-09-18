"""Domain models for BrowserSkill automation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class BrowserSkillCommand(str, Enum):
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    SCROLL = "scroll"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    EXTRACT = "extract"
    EXECUTE_JS = "execute_js"
    GET_COOKIES = "get_cookies"
    SET_COOKIES = "set_cookies"
    TAB_NEW = "tab_new"
    TAB_CLOSE = "tab_close"
    TAB_SWITCH = "tab_switch"
    TAB_LIST = "tab_list"
    DOWNLOAD = "download"
    UPLOAD = "upload"
    HOVER = "hover"
    DRAG_DROP = "drag_drop"
    PRESS_KEY = "press_key"
    GET_TEXT = "get_text"
    GET_HTML = "get_html"
    GET_ATTRIBUTE = "get_attribute"
    SET_ATTRIBUTE = "set_attribute"
    EVALUATE = "evaluate"


class WindowType(str, Enum):
    AGENT_WINDOW = "agent_window"
    USER_WINDOW = "user_window"


class HumanInLoopReason(str, Enum):
    CAPTCHA = "captcha"
    LOGIN = "login"
    CONFIRMATION_DIALOG = "confirmation_dialog"
    TWO_FACTOR = "two_factor"
    CUSTOM = "custom"


@dataclass
class BrowserTab:
    id: str = field(default_factory=lambda: f"tab-{uuid4().hex[:8]}")
    url: str = ""
    title: str = ""
    window_type: WindowType = WindowType.AGENT_WINDOW
    is_active: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BrowserAction:
    id: str = field(default_factory=lambda: f"act-{uuid4().hex[:8]}")
    command: BrowserSkillCommand = BrowserSkillCommand.NAVIGATE
    selector: str | None = None
    value: str | None = None
    url: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    tab_id: str | None = None
    timeout_ms: int = 30000
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ActionResult:
    action_id: str
    success: bool
    output: Any = None
    error: str | None = None
    duration_ms: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


BrowserActionResult = ActionResult


@dataclass
class HumanInLoopRequest:
    id: str = field(default_factory=lambda: f"hil-{uuid4().hex[:8]}")
    reason: HumanInLoopReason = HumanInLoopReason.CUSTOM
    message: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    tab_id: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: datetime | None = None
    resolution: str | None = None


@dataclass
class BSKConfig:
    daemon_port: int = 9222
    extension_id: str = ""
    agent_window_profile: str = "browser_skill_agent"
    user_data_dir: str = ""
    headless: bool = False
    auto_connect: bool = True
    request_timeout_ms: int = 60000
    max_retries: int = 3


@dataclass
class BrowserSession:
    id: str = field(default_factory=lambda: f"sess-{uuid4().hex[:8]}")
    config: BSKConfig = field(default_factory=BSKConfig)
    tabs: list[BrowserTab] = field(default_factory=list)
    active_tab_id: str | None = None
    daemon_pid: int | None = None
    extension_connected: bool = False
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrowserSkillResult:
    session_id: str
    actions: list[BrowserAction] = field(default_factory=list)
    results: list[ActionResult] = field(default_factory=list)
    human_in_loop_requests: list[HumanInLoopRequest] = field(default_factory=list)
    final_url: str = ""
    screenshots: list[str] = field(default_factory=list)
    extracted_data: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None
