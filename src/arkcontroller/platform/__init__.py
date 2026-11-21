"""Windows 平台权限、DPI 感知与窗口视口追踪子包。"""

from arkcontroller.platform.uac import (
    enable_per_monitor_dpi_awareness,
    is_admin,
    relaunch_as_admin,
)
from arkcontroller.platform.window_tracker import (
    ViewportRect,
    WindowTracker,
    compute_active_16_9_rect,
)

__all__ = [
    "ViewportRect",
    "WindowTracker",
    "compute_active_16_9_rect",
    "enable_per_monitor_dpi_awareness",
    "is_admin",
    "relaunch_as_admin",
]
