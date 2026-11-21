"""《明日方舟》游戏窗口与模拟器渲染区追踪及 16:9 视口坐标双向解算模块。"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable

from arkcontroller.config.models import VALID_ANCHORS

TARGET_ASPECT_16_9 = 16.0 / 9.0
MIN_VALID_WIDTH = 320
MIN_VALID_HEIGHT = 180

# Win32 常量
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
GA_ROOT = 2

# 目标进程与模拟器渲染子窗口标识
OFFICIAL_PROCESS_NAMES = frozenset({"arknights.exe"})
OFFICIAL_CLASS_NAMES = frozenset({"unitywndclass"})

MUMU_PROCESS_NAMES = frozenset(
    {
        "mumuplayer.exe",
        "mumunxmain.exe",
        "nemuplayer.exe",
        "mumuvmmsvc.exe",
    }
)
MUMU_RENDER_CLASSES = ("nemuwin", "nemurender", "mumu_render")

LDPLAYER_PROCESS_NAMES = frozenset({"dnplayer.exe", "ldplayer.exe"})
LDPLAYER_RENDER_CLASSES = ("therender",)

EXCLUDED_TOP_CLASSES = frozenset(
    {
        "consolewindowclass",
        "cascadia_hosting_window_class",
        "chrome_widgetwin_1",
        "mozillawindowclass",
        "cabinetwclass",
        "progman",
        "workerw",
    }
)


def compute_active_16_9_rect(
    client_left: int,
    client_top: int,
    client_width: int,
    client_height: int,
    target_aspect: float = TARGET_ASPECT_16_9,
) -> tuple[int, int, int, int]:
    """计算客户区内接居中 16:9 有效视口区域 (active_left, active_top, active_width, active_height)。

    - 超宽屏（如 21:9 3440x1440）：保留满高并裁剪左右黑边/扩展区。
    - 偏高屏（如 16:10 2560x1600）：保留满宽并裁剪上下黑边。
    - 标准 16:9（如 1920x1080）：直接等于客户区。
    """
    if client_width <= 0 or client_height <= 0:
        return (client_left, client_top, max(0, client_width), max(0, client_height))

    aspect = client_width / client_height
    if abs(aspect - target_aspect) < 1e-4:
        return (client_left, client_top, client_width, client_height)

    if aspect > target_aspect:
        active_height = client_height
        active_width = int(round(client_height * target_aspect))
        active_left = client_left + (client_width - active_width) // 2
        active_top = client_top
    else:
        active_width = client_width
        active_height = int(round(client_width / target_aspect))
        active_left = client_left
        active_top = client_top + (client_height - active_height) // 2

    return (active_left, active_top, active_width, active_height)


@dataclass(frozen=True)
class ViewportRect:
    """游戏有效客户区与 16:9 核心视口几何描述及坐标转换器。"""

    hwnd: int
    client_left: int
    client_top: int
    client_width: int
    client_height: int
    active_left: int
    active_top: int
    active_width: int
    active_height: int
    is_foreground: bool
    window_title: str = ""

    @classmethod
    def from_client_rect(
        cls,
        hwnd: int,
        client_left: int,
        client_top: int,
        client_width: int,
        client_height: int,
        is_foreground: bool = True,
        window_title: str = "",
        target_aspect: float = TARGET_ASPECT_16_9,
    ) -> ViewportRect:
        """由客户区屏幕绝对坐标自动计算 16:9 核心视口并构建 ViewportRect。"""
        active_left, active_top, active_width, active_height = compute_active_16_9_rect(
            client_left,
            client_top,
            client_width,
            client_height,
            target_aspect=target_aspect,
        )
        return cls(
            hwnd=hwnd,
            client_left=client_left,
            client_top=client_top,
            client_width=client_width,
            client_height=client_height,
            active_left=active_left,
            active_top=active_top,
            active_width=active_width,
            active_height=active_height,
            is_foreground=is_foreground,
            window_title=window_title,
        )

    def norm_to_screen(
        self,
        u: float,
        v: float,
        anchor: str = "center_16_9",
    ) -> tuple[int, int]:
        """将 [0.0, 1.0] 归一化坐标转换为屏幕绝对物理像素坐标。"""
        if anchor not in VALID_ANCHORS:
            raise ValueError(
                f"不支持的锚点类型 '{anchor}'，合法值: {', '.join(VALID_ANCHORS)}"
            )

        if anchor == "center_16_9":
            x = self.active_left + u * self.active_width
            y = self.active_top + v * self.active_height
        elif anchor == "left_edge":
            x = self.client_left + u * self.active_width
            y = self.active_top + v * self.active_height
        elif anchor == "right_edge":
            x = (self.client_left + self.client_width) - (1.0 - u) * self.active_width
            y = self.active_top + v * self.active_height
        elif anchor == "top_edge":
            x = self.active_left + u * self.active_width
            y = self.client_top + v * self.active_height
        elif anchor == "bottom_edge":
            x = self.active_left + u * self.active_width
            y = (self.client_top + self.client_height) - (1.0 - v) * self.active_height
        else:  # "client"
            x = self.client_left + u * self.client_width
            y = self.client_top + v * self.client_height

        return (int(round(x)), int(round(y)))

    def screen_to_norm(self, screen_x: int, screen_y: int) -> tuple[float, float]:
        """将屏幕绝对物理像素坐标逆向映射为 16:9 核心视口归一化坐标 (u, v)。"""
        u = (
            (screen_x - self.active_left) / self.active_width
            if self.active_width > 0
            else 0.0
        )
        v = (
            (screen_y - self.active_top) / self.active_height
            if self.active_height > 0
            else 0.0
        )
        return (float(u), float(v))

    def norm_rect_to_local(
        self,
        u: float,
        v: float,
        w: float,
        h: float,
    ) -> tuple[int, int, int, int]:
        """将相对 16:9 核心视口的归一化矩形 (u, v, w, h) 换算为视口局部像素矩形 (x, y, px_w, px_h)。"""
        return (
            int(round(u * self.active_width)),
            int(round(v * self.active_height)),
            int(round(w * self.active_width)),
            int(round(h * self.active_height)),
        )


@dataclass(frozen=True)
class _WindowCandidate:
    """内部窗口候选匹配项。"""

    hwnd: int
    root_hwnd: int
    title: str
    class_name: str
    process_name: str
    priority: int
    area: int


class _Win32WindowBackend:
    """基于 ctypes 的轻量级 Win32 窗口查询后端，便于单元测试替换。"""

    def __init__(self) -> None:
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        u32 = self._user32
        k32 = self._kernel32

        u32.IsWindow.argtypes = [wintypes.HWND]
        u32.IsWindow.restype = wintypes.BOOL
        u32.IsWindowVisible.argtypes = [wintypes.HWND]
        u32.IsWindowVisible.restype = wintypes.BOOL
        u32.IsIconic.argtypes = [wintypes.HWND]
        u32.IsIconic.restype = wintypes.BOOL
        u32.GetForegroundWindow.argtypes = []
        u32.GetForegroundWindow.restype = wintypes.HWND
        u32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        u32.GetAncestor.restype = wintypes.HWND
        u32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        u32.GetWindowTextLengthW.restype = ctypes.c_int
        u32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u32.GetWindowTextW.restype = ctypes.c_int
        u32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u32.GetClassNameW.restype = ctypes.c_int
        u32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        u32.GetWindowThreadProcessId.restype = wintypes.DWORD
        u32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        u32.GetClientRect.restype = wintypes.BOOL
        u32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        u32.ClientToScreen.restype = wintypes.BOOL

        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL
        k32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        k32.QueryFullProcessImageNameW.restype = wintypes.BOOL

    def is_window_valid(self, hwnd: int) -> bool:
        return bool(self._user32.IsWindow(hwnd))

    def is_window_visible(self, hwnd: int) -> bool:
        return bool(self._user32.IsWindowVisible(hwnd))

    def is_iconic(self, hwnd: int) -> bool:
        return bool(self._user32.IsIconic(hwnd))

    def get_foreground_window(self) -> int:
        return int(self._user32.GetForegroundWindow() or 0)

    def get_root_window(self, hwnd: int) -> int:
        root = int(self._user32.GetAncestor(hwnd, GA_ROOT) or 0)
        return root if root else hwnd

    def get_window_title(self, hwnd: int) -> str:
        length = int(self._user32.GetWindowTextLengthW(hwnd))
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        self._user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value

    def get_class_name(self, hwnd: int) -> str:
        buf = ctypes.create_unicode_buffer(256)
        copied = int(self._user32.GetClassNameW(hwnd, buf, 256))
        return buf.value[:copied] if copied > 0 else ""

    def get_window_pid(self, hwnd: int) -> int:
        pid = wintypes.DWORD(0)
        self._user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value)

    def get_process_name(self, pid: int) -> str:
        if pid <= 0:
            return ""
        handle = self._kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return ""
        try:
            size = wintypes.DWORD(1024)
            buf = ctypes.create_unicode_buffer(size.value)
            if self._kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return os.path.basename(buf.value).lower()
            return ""
        finally:
            self._kernel32.CloseHandle(handle)

    def get_client_rect_screen(self, hwnd: int) -> tuple[int, int, int, int] | None:
        rect = wintypes.RECT()
        if not self._user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return None
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width <= 0 or height <= 0:
            return None
        origin = wintypes.POINT(rect.left, rect.top)
        if not self._user32.ClientToScreen(hwnd, ctypes.byref(origin)):
            return None
        return (int(origin.x), int(origin.y), width, height)

    def enum_top_windows(self) -> list[int]:
        hwnds: list[int] = []
        enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def _callback(hwnd: int, _lparam: int) -> bool:
            hwnds.append(int(hwnd))
            return True

        cb = enum_proc(_callback)
        self._user32.EnumWindows(cb, 0)
        return hwnds

    def enum_child_windows(self, parent_hwnd: int) -> list[int]:
        children: list[int] = []
        enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def _callback(hwnd: int, _lparam: int) -> bool:
            children.append(int(hwnd))
            return True

        cb = enum_proc(_callback)
        self._user32.EnumChildWindows(parent_hwnd, cb, 0)
        return children


class WindowTracker:
    """持续定位《明日方舟》官方 PC 客户端或主流模拟器渲染子窗口并输出 ViewportRect。"""

    def __init__(
        self,
        preferred_titles: list[str] | None = None,
        *,
        backend: _Win32WindowBackend | None = None,
    ) -> None:
        self.preferred_titles: list[str] = (
            list(preferred_titles) if preferred_titles is not None else ["明日方舟", "Arknights"]
        )
        self._backend: _Win32WindowBackend | None = backend
        if self._backend is None and sys.platform == "win32":
            self._backend = _Win32WindowBackend()

        self._cached_hwnd: int | None = None
        self._cached_root_hwnd: int | None = None
        self._cached_title: str = ""
        self._self_pid: int = os.getpid()

    def reset_cache(self) -> None:
        """清除已缓存的窗口句柄，强制下次 update() 重新搜索。"""
        self._cached_hwnd = None
        self._cached_root_hwnd = None
        self._cached_title = ""

    def _title_matches(self, title: str) -> bool:
        lower_title = title.lower()
        return any(kw.lower() in lower_title for kw in self.preferred_titles if kw)

    def _find_render_child(
        self,
        parent_hwnd: int,
        class_keywords: tuple[str, ...],
    ) -> tuple[int, tuple[int, int, int, int]] | None:
        """在模拟器顶层窗口下查找面积最大的有效渲染子窗口。"""
        assert self._backend is not None
        best_hwnd: int | None = None
        best_rect: tuple[int, int, int, int] | None = None
        best_area = 0

        for child in self._backend.enum_child_windows(parent_hwnd):
            if not self._backend.is_window_visible(child):
                continue
            cls_name = self._backend.get_class_name(child).lower()
            if not any(kw in cls_name for kw in class_keywords):
                continue
            rect = self._backend.get_client_rect_screen(child)
            if rect is None:
                continue
            _, _, width, height = rect
            if width < MIN_VALID_WIDTH or height < MIN_VALID_HEIGHT:
                continue
            area = width * height
            if area > best_area:
                best_area = area
                best_hwnd = child
                best_rect = rect

        if best_hwnd is None or best_rect is None:
            return None
        return (best_hwnd, best_rect)

    def _collect_candidates(self) -> list[_WindowCandidate]:
        """枚举当前桌面所有窗口并按匹配优先级生成候选列表。"""
        assert self._backend is not None
        candidates: list[_WindowCandidate] = []
        pid_name_cache: dict[int, str] = {}

        for hwnd in self._backend.enum_top_windows():
            if not self._backend.is_window_visible(hwnd) or self._backend.is_iconic(hwnd):
                continue

            pid = self._backend.get_window_pid(hwnd)
            if pid == self._self_pid:
                continue

            if pid not in pid_name_cache:
                pid_name_cache[pid] = self._backend.get_process_name(pid)
            proc_name = pid_name_cache[pid]
            cls_name = self._backend.get_class_name(hwnd)
            cls_lower = cls_name.lower()
            title = self._backend.get_window_title(hwnd)
            title_matched = self._title_matches(title)

            # 1. 官方 PC 客户端：Arknights.exe + UnityWndClass 最高优先级
            if proc_name in OFFICIAL_PROCESS_NAMES:
                rect = self._backend.get_client_rect_screen(hwnd)
                if rect and rect[2] >= MIN_VALID_WIDTH and rect[3] >= MIN_VALID_HEIGHT:
                    priority = 100 if cls_lower in OFFICIAL_CLASS_NAMES else 90
                    candidates.append(
                        _WindowCandidate(
                            hwnd=hwnd,
                            root_hwnd=hwnd,
                            title=title,
                            class_name=cls_name,
                            process_name=proc_name,
                            priority=priority,
                            area=rect[2] * rect[3],
                        )
                    )
                continue

            # 2. MuMu 模拟器渲染子窗口（nemuwin）
            if proc_name in MUMU_PROCESS_NAMES:
                child_hit = self._find_render_child(hwnd, MUMU_RENDER_CLASSES)
                if child_hit is not None:
                    child_hwnd, child_rect = child_hit
                    candidates.append(
                        _WindowCandidate(
                            hwnd=child_hwnd,
                            root_hwnd=hwnd,
                            title=title,
                            class_name=self._backend.get_class_name(child_hwnd),
                            process_name=proc_name,
                            priority=85 if title_matched else 78,
                            area=child_rect[2] * child_rect[3],
                        )
                    )
                    continue

            # 3. 雷电模拟器渲染子窗口（TheRender）
            if proc_name in LDPLAYER_PROCESS_NAMES:
                child_hit = self._find_render_child(hwnd, LDPLAYER_RENDER_CLASSES)
                if child_hit is not None:
                    child_hwnd, child_rect = child_hit
                    candidates.append(
                        _WindowCandidate(
                            hwnd=child_hwnd,
                            root_hwnd=hwnd,
                            title=title,
                            class_name=self._backend.get_class_name(child_hwnd),
                            process_name=proc_name,
                            priority=85 if title_matched else 78,
                            area=child_rect[2] * child_rect[3],
                        )
                    )
                    continue

            # 4. 标题匹配兜底（含未知模拟器或自定义窗口标题，自动尝试下钻常见渲染子窗口）
            if title_matched and cls_lower not in EXCLUDED_TOP_CLASSES:
                emulator_child = self._find_render_child(
                    hwnd, MUMU_RENDER_CLASSES + LDPLAYER_RENDER_CLASSES
                )
                if emulator_child is not None:
                    child_hwnd, child_rect = emulator_child
                    candidates.append(
                        _WindowCandidate(
                            hwnd=child_hwnd,
                            root_hwnd=hwnd,
                            title=title,
                            class_name=self._backend.get_class_name(child_hwnd),
                            process_name=proc_name,
                            priority=75,
                            area=child_rect[2] * child_rect[3],
                        )
                    )
                    continue

                rect = self._backend.get_client_rect_screen(hwnd)
                if rect and rect[2] >= MIN_VALID_WIDTH and rect[3] >= MIN_VALID_HEIGHT:
                    candidates.append(
                        _WindowCandidate(
                            hwnd=hwnd,
                            root_hwnd=hwnd,
                            title=title,
                            class_name=cls_name,
                            process_name=proc_name,
                            priority=60,
                            area=rect[2] * rect[3],
                        )
                    )

        return candidates

    def _build_viewport(self, hwnd: int, root_hwnd: int, fallback_title: str) -> ViewportRect | None:
        """读取目标窗口客户区和前台状态并构造 ViewportRect。"""
        assert self._backend is not None
        if not self._backend.is_window_valid(hwnd) or not self._backend.is_window_valid(root_hwnd):
            return None
        if not self._backend.is_window_visible(hwnd) or self._backend.is_iconic(root_hwnd):
            return None

        rect = self._backend.get_client_rect_screen(hwnd)
        if rect is None:
            return None
        left, top, width, height = rect
        if width < MIN_VALID_WIDTH or height < MIN_VALID_HEIGHT:
            return None

        fg_hwnd = self._backend.get_foreground_window()
        fg_root = self._backend.get_root_window(fg_hwnd) if fg_hwnd else 0
        is_foreground = fg_hwnd in (hwnd, root_hwnd) or fg_root == root_hwnd
        title = self._backend.get_window_title(root_hwnd) or fallback_title

        return ViewportRect.from_client_rect(
            hwnd=hwnd,
            client_left=left,
            client_top=top,
            client_width=width,
            client_height=height,
            is_foreground=is_foreground,
            window_title=title,
        )

    def update(self) -> ViewportRect | None:
        """刷新并返回当前追踪到的游戏窗口 ViewportRect；未检测到或已最小化时返回 None。"""
        if self._backend is None:
            return None

        if self._cached_hwnd is not None and self._cached_root_hwnd is not None:
            viewport = self._build_viewport(
                self._cached_hwnd,
                self._cached_root_hwnd,
                self._cached_title,
            )
            if viewport is not None:
                return viewport
            self.reset_cache()

        candidates = self._collect_candidates()
        if not candidates:
            return None

        best = max(candidates, key=lambda c: (c.priority, c.area))
        viewport = self._build_viewport(best.hwnd, best.root_hwnd, best.title)
        if viewport is not None:
            self._cached_hwnd = best.hwnd
            self._cached_root_hwnd = best.root_hwnd
            self._cached_title = best.title
        return viewport
