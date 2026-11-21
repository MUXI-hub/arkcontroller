"""Win32 SendInput 虚拟键鼠合成层。

完全基于 Python 标准库 ctypes 封装 Win32 SendInput 接口，不注入任何 DLL，
支持多显示器虚拟桌面绝对坐标转换（MOUSEEVENTF_VIRTUALDESK）、
硬件扫描码（KEYEVENTF_SCANCODE）键盘输入、平滑插值拖拽及四向滑动定朝向。
"""

from __future__ import annotations

import ctypes
import threading
import time
from collections.abc import Callable
from typing import Any, Literal

# SendInput 输入类型常量
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

# 鼠标事件标志位
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x1000
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_ABSOLUTE = 0x8000

# 键盘事件标志位
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

# MapVirtualKeyW 映射模式：虚拟键码 -> 硬件扫描码
MAPVK_VK_TO_VSC = 0

# GetSystemMetrics 虚拟桌面边界查询常量
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

# 对应 Win32 ULONG_PTR（64 位 Windows 为 8 字节，32 位为 4 字节）
ULONG_PTR = ctypes.c_size_t


class MOUSEINPUT(ctypes.Structure):
    """对应 Win32 MOUSEINPUT 结构体（64 位下 32 字节）。"""

    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_uint32),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    """对应 Win32 KEYBDINPUT 结构体（64 位下 24 字节）。"""

    _fields_ = [
        ("wVk", ctypes.c_uint16),
        ("wScan", ctypes.c_uint16),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    """对应 Win32 HARDWAREINPUT 结构体（8 字节）。"""

    _fields_ = [
        ("uMsg", ctypes.c_uint32),
        ("wParamL", ctypes.c_uint16),
        ("wParamH", ctypes.c_uint16),
    ]


class _INPUT_UNION(ctypes.Union):
    """必须同时包含 MOUSEINPUT、KEYBDINPUT、HARDWAREINPUT 以保证 64 位对齐后 sizeof(INPUT) == 40。"""

    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    """对应 Win32 INPUT 结构体（64 位 Windows 严格为 40 字节）。"""

    _anonymous_ = ("_union",)
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("_union", _INPUT_UNION),
    ]


# 鼠标按钮名称 -> (按下标志位, 抬起标志位)
_MOUSE_BUTTON_FLAGS: dict[str, tuple[int, int]] = {
    "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
    "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
}

# 常用按键名称到 Windows Virtual-Key 码映射（支持官方 PC 端卫戍协议全套快捷键）
_VK_KEY_MAP: dict[str, int] = {
    "space": 0x20,
    "tab": 0x09,
    "enter": 0x0D,
    "return": 0x0D,
    "esc": 0x1B,
    "escape": 0x1B,
    "grave": 0xC0,
    "`": 0xC0,
    "~": 0xC0,
    "tilde": 0xC0,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
}
# 补充数字键 0-9 与字母键 a-z
for _digit in range(10):
    _VK_KEY_MAP[str(_digit)] = 0x30 + _digit
for _offset in range(26):
    _VK_KEY_MAP[chr(ord("a") + _offset)] = 0x41 + _offset

# 需要附加 KEYEVENTF_EXTENDEDKEY 的扩展键集合
_EXTENDED_VK_CODES: frozenset[int] = frozenset({0x25, 0x26, 0x27, 0x28})

# 四向滑动方向单位向量（屏幕坐标系：向右 X 正，向下 Y 正）
_DIRECTION_VECTORS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


class VirtualHID:
    """面向 Windows SendInput 的虚拟键鼠控制器。"""

    def __init__(
        self,
        user32: Any | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._user32_override: Any | None = user32
        self._sleep: Callable[[float], None] = sleep_fn if sleep_fn is not None else time.sleep
        self._lock = threading.RLock()

    def _get_user32(self) -> Any:
        """动态获取 user32 句柄，便于单测直接 mock ctypes.windll.user32。"""
        if self._user32_override is not None:
            return self._user32_override
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            raise RuntimeError("当前平台不支持 ctypes.windll.user32")
        return windll.user32

    def get_virtual_desktop_bounds(self) -> tuple[int, int, int, int]:
        """查询多显示器虚拟桌面范围 (vx, vy, vw, vh)。"""
        user32 = self._get_user32()
        vx = int(user32.GetSystemMetrics(SM_XVIRTUALSCREEN))
        vy = int(user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
        vw = max(1, int(user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)))
        vh = max(1, int(user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)))
        return (vx, vy, vw, vh)

    def screen_to_absolute(self, screen_x: int, screen_y: int) -> tuple[int, int]:
        """将屏幕物理像素坐标换算为 SendInput 虚拟桌面绝对坐标 [0, 65535]。"""
        vx, vy, vw, vh = self.get_virtual_desktop_bounds()
        denom_x = max(1, vw - 1)
        denom_y = max(1, vh - 1)
        abs_x = int(round(((screen_x - vx) * 65535.0) / denom_x))
        abs_y = int(round(((screen_y - vy) * 65535.0) / denom_y))
        abs_x = max(0, min(65535, abs_x))
        abs_y = max(0, min(65535, abs_y))
        return (abs_x, abs_y)

    def _send_inputs(self, *items: INPUT) -> int:
        """批量通过 Win32 SendInput 发送输入事件结构体数组。"""
        if not items:
            return 0
        user32 = self._get_user32()
        count = len(items)
        array_type = INPUT * count
        payload = array_type(*items)
        return int(user32.SendInput(count, payload, ctypes.sizeof(INPUT)))

    def _make_mouse_input(
        self,
        *,
        dx: int = 0,
        dy: int = 0,
        mouse_data: int = 0,
        flags: int = 0,
    ) -> INPUT:
        """构造单个 INPUT_MOUSE 结构体。"""
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.mi = MOUSEINPUT(
            dx=int(dx),
            dy=int(dy),
            mouseData=int(mouse_data) & 0xFFFFFFFF,
            dwFlags=int(flags),
            time=0,
            dwExtraInfo=0,
        )
        return inp

    def _resolve_vk(self, key_name: str) -> int:
        """将逻辑按键名解析为 Windows Virtual-Key 码。"""
        normalized = key_name.strip().lower()
        if normalized not in _VK_KEY_MAP:
            raise ValueError(f"不支持的键盘按键名称: {key_name!r}")
        return _VK_KEY_MAP[normalized]

    def _make_keyboard_input(self, key_name: str, *, key_up: bool) -> INPUT:
        """通过 MapVirtualKeyW 解析硬件扫描码并构造 INPUT_KEYBOARD 结构体。"""
        vk = self._resolve_vk(key_name)
        user32 = self._get_user32()
        scan_code = int(user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC)) & 0xFFFF

        flags = KEYEVENTF_SCANCODE
        w_vk = 0
        if scan_code == 0:
            # 当系统或测试替身未返回有效扫描码时，安全回退为虚拟键码模式
            flags = 0
            w_vk = vk

        if vk in _EXTENDED_VK_CODES:
            flags |= KEYEVENTF_EXTENDEDKEY
        if key_up:
            flags |= KEYEVENTF_KEYUP

        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.ki = KEYBDINPUT(
            wVk=w_vk,
            wScan=scan_code,
            dwFlags=flags,
            time=0,
            dwExtraInfo=0,
        )
        return inp

    def move_cursor(self, screen_x: int, screen_y: int) -> None:
        """将鼠标光标绝对移动至指定屏幕物理像素坐标。"""
        with self._lock:
            abs_x, abs_y = self.screen_to_absolute(screen_x, screen_y)
            inp = self._make_mouse_input(
                dx=abs_x,
                dy=abs_y,
                flags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
            )
            self._send_inputs(inp)

    def mouse_down(self, button: str = "left") -> None:
        """按下指定鼠标按键（left / right / middle）。"""
        key = button.strip().lower()
        if key not in _MOUSE_BUTTON_FLAGS:
            raise ValueError(f"不支持的鼠标按键: {button!r}")
        down_flag, _ = _MOUSE_BUTTON_FLAGS[key]
        with self._lock:
            self._send_inputs(self._make_mouse_input(flags=down_flag))

    def mouse_up(self, button: str = "left") -> None:
        """松开指定鼠标按键（left / right / middle）。"""
        key = button.strip().lower()
        if key not in _MOUSE_BUTTON_FLAGS:
            raise ValueError(f"不支持的鼠标按键: {button!r}")
        _, up_flag = _MOUSE_BUTTON_FLAGS[key]
        with self._lock:
            self._send_inputs(self._make_mouse_input(flags=up_flag))

    def click(
        self,
        screen_x: int,
        screen_y: int,
        button: str = "left",
        hold_ms: int = 30,
    ) -> None:
        """移动至目标屏幕坐标并执行一次带保持时长的鼠标点击。"""
        with self._lock:
            self.move_cursor(screen_x, screen_y)
            self.mouse_down(button)
            if hold_ms > 0:
                self._sleep(hold_ms / 1000.0)
            self.mouse_up(button)

    def double_click(
        self,
        screen_x: int,
        screen_y: int,
        interval_ms: int = 55,
    ) -> None:
        """在目标坐标执行双击（适用于商店一键极速连买与三选一快速确认）。"""
        with self._lock:
            self.click(screen_x, screen_y, button="left", hold_ms=25)
            if interval_ms > 0:
                self._sleep(interval_ms / 1000.0)
            self.click(screen_x, screen_y, button="left", hold_ms=25)

    def smooth_drag(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        duration_ms: int = 180,
        steps: int = 14,
        release_at_end: bool = True,
        pre_hover_ms: int = 35,
        post_hold_ms: int = 35,
    ) -> None:
        """从起点悬停后按住左键并分步平滑插值拖拽至终点。

        Args:
            start: 拖拽起点屏幕像素坐标 (x, y)。
            end: 拖拽终点屏幕像素坐标 (x, y)。
            duration_ms: 总拖拽耗时（毫秒）。
            steps: 线性插值步数（至少为 1）。
            release_at_end: 到达终点后是否自动松开鼠标左键。
            pre_hover_ms: 移至起点后的悬停等待时长（毫秒），确保 Unity 射线先捕获目标再起拖。
            post_hold_ms: 到达终点后松开前的短暂停留时长（毫秒），确保目标格稳定吸附。
        """
        total_steps = max(1, int(steps))
        step_sleep_sec = (max(0, int(duration_ms)) / 1000.0) / total_steps
        sx, sy = int(start[0]), int(start[1])
        ex, ey = int(end[0]), int(end[1])

        with self._lock:
            self.move_cursor(sx, sy)
            if pre_hover_ms > 0:
                self._sleep(pre_hover_ms / 1000.0)
            self.mouse_down("left")
            for idx in range(1, total_steps + 1):
                if step_sleep_sec > 0.0:
                    self._sleep(step_sleep_sec)
                ratio = idx / total_steps
                cx = int(round(sx + (ex - sx) * ratio))
                cy = int(round(sy + (ey - sy) * ratio))
                self.move_cursor(cx, cy)
            if release_at_end:
                if post_hold_ms > 0:
                    self._sleep(post_hold_ms / 1000.0)
                self.mouse_up("left")

    def smooth_drag_loopback(
        self,
        origin: tuple[int, int],
        offset: tuple[int, int],
        duration_ms: int = 130,
        steps_each_way: int = 6,
        release_at_end: bool = True,
        pre_hover_ms: int = 30,
        post_hold_ms: int = 30,
    ) -> None:
        """从起点按住左键平滑拖离原格再平滑拖回原格松开（用于棋盘干员原地唤出四向定朝向盘）。

        Args:
            origin: 起点（及回落终点）屏幕像素坐标 (ox, oy)。
            offset: 往返折返偏移量 (dx, dy)，需跨越 Unity pixelDragThreshold 触发 BeginDrag。
            duration_ms: 往返两段平滑插值总耗时（毫秒）。
            steps_each_way: 去程与回程各自的线性插值步数（至少为 1）。
            release_at_end: 回到起点停顿后是否自动松开鼠标左键。
            pre_hover_ms: 移至起点后按下左键前的悬停时长（毫秒）。
            post_hold_ms: 回到起点后松开左键前的停顿时长（毫秒）。
        """
        leg_steps = max(1, int(steps_each_way))
        total_steps = leg_steps * 2
        step_sleep_sec = (max(0, int(duration_ms)) / 1000.0) / total_steps
        ox, oy = int(origin[0]), int(origin[1])
        wx, wy = ox + int(offset[0]), oy + int(offset[1])

        with self._lock:
            self.move_cursor(ox, oy)
            if pre_hover_ms > 0:
                self._sleep(pre_hover_ms / 1000.0)
            self.mouse_down("left")
            # 去程：从 origin 平滑拖至 waypoint，触发客户端拖拽态
            for idx in range(1, leg_steps + 1):
                if step_sleep_sec > 0.0:
                    self._sleep(step_sleep_sec)
                ratio = idx / leg_steps
                cx = int(round(ox + (wx - ox) * ratio))
                cy = int(round(oy + (wy - oy) * ratio))
                self.move_cursor(cx, cy)
            # 回程：从 waypoint 平滑拖回 origin 原格
            for idx in range(1, leg_steps + 1):
                if step_sleep_sec > 0.0:
                    self._sleep(step_sleep_sec)
                ratio = idx / leg_steps
                cx = int(round(wx + (ox - wx) * ratio))
                cy = int(round(wy + (oy - wy) * ratio))
                self.move_cursor(cx, cy)
            if release_at_end:
                if post_hold_ms > 0:
                    self._sleep(post_hold_ms / 1000.0)
                self.mouse_up("left")

    def directional_flick(
        self,
        origin: tuple[int, int],
        direction: Literal["up", "down", "left", "right"],
        distance_px: int,
    ) -> None:
        """在干员部署落点处向指定四向滑动以确定干员朝向。"""
        dir_key = direction.strip().lower()
        if dir_key not in _DIRECTION_VECTORS:
            raise ValueError(f"不支持的滑动朝向: {direction!r}")
        vx, vy = _DIRECTION_VECTORS[dir_key]
        dist = abs(int(distance_px))
        target = (int(origin[0]) + vx * dist, int(origin[1]) + vy * dist)
        self.smooth_drag(
            start=(int(origin[0]), int(origin[1])),
            end=target,
            duration_ms=110,
            steps=10,
            release_at_end=True,
            pre_hover_ms=30,
            post_hold_ms=20,
        )

    def key_down(self, key_name: str) -> None:
        """发送硬件扫描码键盘按下事件。"""
        with self._lock:
            inp = self._make_keyboard_input(key_name, key_up=False)
            self._send_inputs(inp)

    def key_up(self, key_name: str) -> None:
        """发送硬件扫描码键盘抬起事件。"""
        with self._lock:
            inp = self._make_keyboard_input(key_name, key_up=True)
            self._send_inputs(inp)

    def tap_key(self, key_name: str, hold_ms: int = 30) -> None:
        """执行一次带按下保持时长的键盘敲击。"""
        with self._lock:
            self.key_down(key_name)
            if hold_ms > 0:
                self._sleep(hold_ms / 1000.0)
            self.key_up(key_name)

    def scroll(self, wheel_delta: int, horizontal: bool = False) -> None:
        """发送垂直或水平鼠标滚轮滚动事件（正值向上/向右，负值向下/向左）。"""
        delta = int(wheel_delta)
        if delta == 0:
            return
        flag = MOUSEEVENTF_HWHEEL if horizontal else MOUSEEVENTF_WHEEL
        with self._lock:
            inp = self._make_mouse_input(mouse_data=delta, flags=flag)
            self._send_inputs(inp)

    def wait_ms(self, duration_ms: int) -> None:
        """按指定毫秒数等待客户端 UI 过渡动画完成（复用可注入 sleep_fn，单测零阻塞）。"""
        if duration_ms > 0:
            self._sleep(duration_ms / 1000.0)
