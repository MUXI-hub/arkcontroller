"""XInput 手柄驱动层。

通过标准库 ctypes 直接封装 Windows 原生 XInput DLL（优先 XInput1_4.dll，
依次回退 XInput9_1_0.dll 与 xinput1_3.dll），实现全局后台非阻塞轮询、
径向缩放死区（Radial Scaled Deadzone）过滤及异步定时关停震动反馈。
"""

from __future__ import annotations

import ctypes
import logging
import math
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# XInput API 返回码
ERROR_SUCCESS = 0
ERROR_DEVICE_NOT_CONNECTED = 1167

# XInput 按键位掩码（与 Windows SDK Xinput.h 保持一致）
XINPUT_GAMEPAD_DPAD_UP = 0x0001
XINPUT_GAMEPAD_DPAD_DOWN = 0x0002
XINPUT_GAMEPAD_DPAD_LEFT = 0x0004
XINPUT_GAMEPAD_DPAD_RIGHT = 0x0008
XINPUT_GAMEPAD_START = 0x0010
XINPUT_GAMEPAD_BACK = 0x0020
XINPUT_GAMEPAD_LEFT_THUMB = 0x0040
XINPUT_GAMEPAD_RIGHT_THUMB = 0x0080
XINPUT_GAMEPAD_LEFT_SHOULDER = 0x0100
XINPUT_GAMEPAD_RIGHT_SHOULDER = 0x0200
XINPUT_GAMEPAD_A = 0x1000
XINPUT_GAMEPAD_B = 0x2000
XINPUT_GAMEPAD_X = 0x4000
XINPUT_GAMEPAD_Y = 0x8000

# 支持加载的 XInput 动态库优先级列表
_XINPUT_DLL_CANDIDATES: tuple[str, ...] = (
    "XInput1_4.dll",
    "XInput9_1_0.dll",
    "xinput1_3.dll",
)


class GamepadButton(str, Enum):
    """Xbox 手柄标准逻辑按键枚举。"""

    A = "A"
    B = "B"
    X = "X"
    Y = "Y"
    LB = "LB"
    RB = "RB"
    LT = "LT"
    RT = "RT"
    BACK = "BACK"
    START = "START"
    LS = "LS"
    RS = "RS"
    DPAD_UP = "DPAD_UP"
    DPAD_DOWN = "DPAD_DOWN"
    DPAD_LEFT = "DPAD_LEFT"
    DPAD_RIGHT = "DPAD_RIGHT"


# 位掩码到逻辑按键的映射表
_BUTTON_BITMASK_MAP: tuple[tuple[int, GamepadButton], ...] = (
    (XINPUT_GAMEPAD_A, GamepadButton.A),
    (XINPUT_GAMEPAD_B, GamepadButton.B),
    (XINPUT_GAMEPAD_X, GamepadButton.X),
    (XINPUT_GAMEPAD_Y, GamepadButton.Y),
    (XINPUT_GAMEPAD_LEFT_SHOULDER, GamepadButton.LB),
    (XINPUT_GAMEPAD_RIGHT_SHOULDER, GamepadButton.RB),
    (XINPUT_GAMEPAD_BACK, GamepadButton.BACK),
    (XINPUT_GAMEPAD_START, GamepadButton.START),
    (XINPUT_GAMEPAD_LEFT_THUMB, GamepadButton.LS),
    (XINPUT_GAMEPAD_RIGHT_THUMB, GamepadButton.RS),
    (XINPUT_GAMEPAD_DPAD_UP, GamepadButton.DPAD_UP),
    (XINPUT_GAMEPAD_DPAD_DOWN, GamepadButton.DPAD_DOWN),
    (XINPUT_GAMEPAD_DPAD_LEFT, GamepadButton.DPAD_LEFT),
    (XINPUT_GAMEPAD_DPAD_RIGHT, GamepadButton.DPAD_RIGHT),
)


class XINPUT_GAMEPAD(ctypes.Structure):
    """对应 Win32 XINPUT_GAMEPAD 结构体（12 字节）。"""

    _fields_ = [
        ("wButtons", ctypes.c_ushort),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]


class XINPUT_STATE(ctypes.Structure):
    """对应 Win32 XINPUT_STATE 结构体（16 字节）。"""

    _fields_ = [
        ("dwPacketNumber", ctypes.c_uint32),
        ("Gamepad", XINPUT_GAMEPAD),
    ]


class XINPUT_VIBRATION(ctypes.Structure):
    """对应 Win32 XINPUT_VIBRATION 结构体（4 字节）。"""

    _fields_ = [
        ("wLeftMotorSpeed", ctypes.c_ushort),
        ("wRightMotorSpeed", ctypes.c_ushort),
    ]


@dataclass(frozen=True)
class GamepadState:
    """手柄单帧归一化状态快照。

    坐标约定：
    - lx, rx：向右为正，范围 [-1.0, 1.0]
    - ly, ry：向上为正，范围 [-1.0, 1.0]
    - lt, rt：扳机深度，范围 [0.0, 1.0]；达到阈值时同步置入 buttons
    """

    connected: bool = False
    packet_number: int = 0
    buttons: frozenset[GamepadButton] = frozenset()
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    lt: float = 0.0
    rt: float = 0.0


def apply_radial_deadzone(x: float, y: float, deadzone: float) -> tuple[float, float]:
    """对二维归一化摇杆坐标执行径向缩放死区（Radial Scaled Deadzone）过滤。

    在半径 <= deadzone 的圆形盲区内输出 (0.0, 0.0)；
    超出盲区后沿原方向将幅值平滑重映射到 (0.0, 1.0]，避免十字轴吸附伪影与跳变。
    """
    magnitude = math.hypot(x, y)
    if deadzone >= 1.0 or magnitude <= max(0.0, deadzone):
        return (0.0, 0.0)

    dz = max(0.0, deadzone)
    clamped_mag = min(magnitude, 1.0)
    scaled_mag = (clamped_mag - dz) / (1.0 - dz) if dz > 0.0 else clamped_mag
    scale = scaled_mag / magnitude
    return (x * scale, y * scale)


def _normalize_thumb_axis(raw_value: int) -> float:
    """将 XInput 有符号 16 位摇杆值 [-32768, 32767] 归一化至 [-1.0, 1.0]。"""
    normalized = raw_value / 32767.0
    if normalized < -1.0:
        return -1.0
    if normalized > 1.0:
        return 1.0
    return normalized


def _clamp_unit(value: float) -> float:
    """将浮点数钳制到 [0.0, 1.0] 区间。"""
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _load_xinput_dll() -> Any | None:
    """按优先级依次尝试加载 Windows XInput DLL。"""
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        return None

    for dll_name in _XINPUT_DLL_CANDIDATES:
        try:
            dll = win_dll(dll_name)
            dll.XInputGetState.argtypes = [ctypes.c_uint32, ctypes.POINTER(XINPUT_STATE)]
            dll.XInputGetState.restype = ctypes.c_uint32
            dll.XInputSetState.argtypes = [
                ctypes.c_uint32,
                ctypes.POINTER(XINPUT_VIBRATION),
            ]
            dll.XInputSetState.restype = ctypes.c_uint32
            logger.debug("已加载 XInput 动态库: %s", dll_name)
            return dll
        except OSError:
            continue

    logger.warning("未能加载任何 XInput DLL，手柄驱动将处于无设备降级模式。")
    return None


class XInputDriver:
    """基于原生 ctypes 的 XInput 手柄轮询与震动驱动器。"""

    def __init__(
        self,
        user_index: int = 0,
        left_deadzone: float = 0.14,
        right_deadzone: float = 0.16,
        trigger_threshold: float = 0.45,
        dll: Any | None = None,
    ) -> None:
        if not 0 <= user_index <= 3:
            raise ValueError(f"XInput user_index 必须在 0..3 范围内，收到: {user_index}")

        self.user_index: int = user_index
        self.left_deadzone: float = float(left_deadzone)
        self.right_deadzone: float = float(right_deadzone)
        self.trigger_threshold: float = float(trigger_threshold)

        self._dll: Any | None = dll if dll is not None else _load_xinput_dll()
        self._connected: bool = False
        self._rumble_lock = threading.Lock()
        self._rumble_timer: threading.Timer | None = None
        self._rumble_token: int = 0

    @property
    def is_connected(self) -> bool:
        """最近一次轮询时手柄是否处于已连接状态。"""
        return self._connected

    @property
    def is_available(self) -> bool:
        """当前运行环境是否成功加载了 XInput DLL。"""
        return self._dll is not None

    def set_deadzones(self, left_deadzone: float, right_deadzone: float) -> None:
        """运行时动态更新左右摇杆径向死区半径。"""
        self.left_deadzone = float(left_deadzone)
        self.right_deadzone = float(right_deadzone)

    def poll(self) -> GamepadState:
        """非阻塞读取一次手柄硬件状态，返回归一化后的 GamepadState。"""
        if self._dll is None:
            self._connected = False
            return GamepadState(connected=False)

        raw_state = XINPUT_STATE()
        try:
            result = int(self._dll.XInputGetState(self.user_index, ctypes.byref(raw_state)))
        except OSError:
            self._connected = False
            return GamepadState(connected=False)

        if result != ERROR_SUCCESS:
            if self._connected:
                self._connected = False
                self._cancel_rumble_timer_only()
            return GamepadState(connected=False)

        self._connected = True
        gp = raw_state.Gamepad

        pressed: set[GamepadButton] = set()
        raw_buttons = int(gp.wButtons)
        for bitmask, button in _BUTTON_BITMASK_MAP:
            if raw_buttons & bitmask:
                pressed.add(button)

        lt = _clamp_unit(int(gp.bLeftTrigger) / 255.0)
        rt = _clamp_unit(int(gp.bRightTrigger) / 255.0)
        if lt >= self.trigger_threshold:
            pressed.add(GamepadButton.LT)
        if rt >= self.trigger_threshold:
            pressed.add(GamepadButton.RT)

        raw_lx = _normalize_thumb_axis(int(gp.sThumbLX))
        raw_ly = _normalize_thumb_axis(int(gp.sThumbLY))
        raw_rx = _normalize_thumb_axis(int(gp.sThumbRX))
        raw_ry = _normalize_thumb_axis(int(gp.sThumbRY))

        lx, ly = apply_radial_deadzone(raw_lx, raw_ly, self.left_deadzone)
        rx, ry = apply_radial_deadzone(raw_rx, raw_ry, self.right_deadzone)

        return GamepadState(
            connected=True,
            packet_number=int(raw_state.dwPacketNumber),
            buttons=frozenset(pressed),
            lx=lx,
            ly=ly,
            rx=rx,
            ry=ry,
            lt=lt,
            rt=rt,
        )

    def rumble(
        self,
        left_motor: float,
        right_motor: float,
        duration_ms: int = 60,
    ) -> None:
        """触发手柄非阻塞微震动，并在 duration_ms 毫秒后自动停止。

        Args:
            left_motor: 低频重马达强度，范围 [0.0, 1.0]。
            right_motor: 高频轻马达强度，范围 [0.0, 1.0]。
            duration_ms: 震动持续时间（毫秒）；<= 0 时不自动定时关停。
        """
        if self._dll is None:
            return

        left_speed = int(round(_clamp_unit(left_motor) * 65535))
        right_speed = int(round(_clamp_unit(right_motor) * 65535))

        if left_speed == 0 and right_speed == 0:
            self.stop_rumble()
            return

        with self._rumble_lock:
            self._rumble_token += 1
            current_token = self._rumble_token
            if self._rumble_timer is not None:
                self._rumble_timer.cancel()
                self._rumble_timer = None

            self._set_vibration_unlocked(left_speed, right_speed)

            if duration_ms > 0:
                timer = threading.Timer(
                    duration_ms / 1000.0,
                    self._stop_rumble_if_token_matches,
                    args=(current_token,),
                )
                timer.daemon = True
                self._rumble_timer = timer
                timer.start()

    def stop_rumble(self) -> None:
        """立即停止手柄震动并取消未到期的自动停震定时器。"""
        with self._rumble_lock:
            self._rumble_token += 1
            if self._rumble_timer is not None:
                self._rumble_timer.cancel()
                self._rumble_timer = None
            if self._dll is not None:
                self._set_vibration_unlocked(0, 0)

    def close(self) -> None:
        """释放驱动资源并确保关闭手柄马达。"""
        self.stop_rumble()

    def _cancel_rumble_timer_only(self) -> None:
        """设备断开时取消挂起定时器，避免无效回调。"""
        with self._rumble_lock:
            self._rumble_token += 1
            if self._rumble_timer is not None:
                self._rumble_timer.cancel()
                self._rumble_timer = None

    def _stop_rumble_if_token_matches(self, token: int) -> None:
        """仅当定时器令牌仍为最新一次震动请求时执行关停，防止竞态截断新震动。"""
        with self._rumble_lock:
            if token != self._rumble_token:
                return
            self._rumble_timer = None
            if self._dll is not None:
                self._set_vibration_unlocked(0, 0)

    def _set_vibration_unlocked(self, left_speed: int, right_speed: int) -> None:
        """向 XInputSetState 发送震动强度（调用方需持有 _rumble_lock）。"""
        vib = XINPUT_VIBRATION(
            wLeftMotorSpeed=left_speed,
            wRightMotorSpeed=right_speed,
        )
        try:
            self._dll.XInputSetState(self.user_index, ctypes.byref(vib))
        except OSError:
            pass
