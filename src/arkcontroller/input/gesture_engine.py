"""手柄高层手势状态机。

将 XInputDriver 采样的原始 GamepadState 帧流解析为高层语义事件：
- ButtonTap：单键点击
- ButtonHoldProgress / ButtonHoldTriggered：长按进度（如 Start 防误触开战）与触发
- ButtonRepeat：方向键及肩键按住连发
- ChordTriggered：修饰键（LT / RT / BACK）组合键，并严格抑制本次周期单键误触发
- ModifierStateChanged：修饰键按下/松开状态变化（如三选一按住 LT 查看棋盘）
- StickMotion：带 gamma 响应曲线与亚像素余量累加器的摇杆屏幕位移增量
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

from arkcontroller.config.models import KeymapConfig
from arkcontroller.input.xinput_driver import GamepadButton, GamepadState


# 修饰键优先级顺序（当多个修饰键同帧按下或嵌套时，优先级高者充当修饰键）
MODIFIER_BUTTONS: tuple[GamepadButton, ...] = (
    GamepadButton.LT,
    GamepadButton.RT,
    GamepadButton.BACK,
)
_MODIFIER_SET: frozenset[GamepadButton] = frozenset(MODIFIER_BUTTONS)

# 默认支持按住连发的导航按键（十字键 + LB/RB）
DEFAULT_REPEAT_BUTTONS: frozenset[GamepadButton] = frozenset(
    {
        GamepadButton.DPAD_UP,
        GamepadButton.DPAD_DOWN,
        GamepadButton.DPAD_LEFT,
        GamepadButton.DPAD_RIGHT,
        GamepadButton.LB,
        GamepadButton.RB,
    }
)

# 默认启用长按进度条与防误触触发的按键（Start 准备就绪）
DEFAULT_HOLD_BUTTONS: frozenset[GamepadButton] = frozenset({GamepadButton.START})

# 固定遍历顺序，保证同帧事件流输出完全确定、便于测试与重放
_BUTTON_ORDER: tuple[GamepadButton, ...] = tuple(GamepadButton)

# 允许的微小人手错帧容差（毫秒）：如用户想按 BACK + START，若 START 比 BACK 早按极短时间则合并为组合键
_REVERSE_CHORD_GRACE_MS = 120.0


@dataclass(frozen=True)
class ButtonTap:
    """单键点击事件。"""

    button: GamepadButton


@dataclass(frozen=True)
class ButtonHoldProgress:
    """长按按键的实时进度事件（0.0 ~ 1.0）；提前松开或结束时会发一次 0.0 归零。"""

    button: GamepadButton
    progress: float


@dataclass(frozen=True)
class ButtonHoldTriggered:
    """长按按键达到阈值时触发的一次性事件。"""

    button: GamepadButton


@dataclass(frozen=True)
class ButtonRepeat:
    """十字键或肩键持续按住时按节拍触发的连发事件。"""

    button: GamepadButton


@dataclass(frozen=True)
class ChordTriggered:
    """修饰键（LT / RT / BACK）与其他按键同时组合触发的事件。"""

    modifier: GamepadButton
    button: GamepadButton


@dataclass(frozen=True)
class ModifierStateChanged:
    """修饰键激活/释放状态变更事件（如按住 LT 临时隐藏三选一弹窗看战场）。"""

    modifier: GamepadButton
    active: bool


@dataclass(frozen=True)
class StickMotion:
    """摇杆推杆事件。

    属性说明：
    - stick: "left" 或 "right"
    - x, y: 原始归一化方向（向右 x > 0，向上 y > 0）
    - dx_px, dy_px: 经 gamma 响应曲线与亚像素累加器换算后的屏幕像素增量
      （屏幕坐标向右 dx_px > 0，向下 dy_px > 0，即 dy_px 对 y 取反）
    """

    stick: Literal["left", "right"]
    x: float
    y: float
    dx_px: float
    dy_px: float


@dataclass(frozen=True)
class StickDirection:
    """左摇杆离散四向导航事件（代替十字键跳格，支持滞回防抖与按住连发）。"""

    stick: Literal["left", "right"]
    direction: Literal["up", "down", "left", "right"]
    is_repeat: bool = False


GestureEvent: TypeAlias = (
    ButtonTap
    | ButtonHoldProgress
    | ButtonHoldTriggered
    | ButtonRepeat
    | ChordTriggered
    | ModifierStateChanged
    | StickMotion
    | StickDirection
)


@dataclass
class _ButtonTrack:
    """单个按键在当前按下周期内的状态追踪器。"""

    pressed: bool = False
    held_ms: float = 0.0
    next_repeat_ms: float = 0.0
    hold_triggered: bool = False
    hold_progress_emitted: bool = False
    consumed_by_chord: bool = False
    modifier_notified: bool = False

    def reset(self) -> None:
        self.pressed = False
        self.held_ms = 0.0
        self.next_repeat_ms = 0.0
        self.hold_triggered = False
        self.hold_progress_emitted = False
        self.consumed_by_chord = False
        self.modifier_notified = False


class GestureEngine:
    """将手柄状态帧转换为高层语义手势事件的状态机。"""

    def __init__(
        self,
        keymap: KeymapConfig,
        *,
        trigger_threshold: float | None = None,
        hold_buttons: frozenset[GamepadButton] = DEFAULT_HOLD_BUTTONS,
        repeat_buttons: frozenset[GamepadButton] = DEFAULT_REPEAT_BUTTONS,
    ) -> None:
        self._keymap: KeymapConfig = keymap
        self._trigger_threshold_override: float | None = (
            float(trigger_threshold) if trigger_threshold is not None else None
        )
        self._hold_buttons: frozenset[GamepadButton] = frozenset(hold_buttons)
        self._repeat_buttons: frozenset[GamepadButton] = frozenset(repeat_buttons)

        self._prev_buttons: frozenset[GamepadButton] = frozenset()
        self._tracks: dict[GamepadButton, _ButtonTrack] = {
            btn: _ButtonTrack() for btn in _BUTTON_ORDER
        }
        # 左右摇杆独立的亚像素余量累加器，保证极低速微推可稳定 1 像素步进
        self._stick_remainder: dict[Literal["left", "right"], list[float]] = {
            "left": [0.0, 0.0],
            "right": [0.0, 0.0],
        }
        # 左摇杆代替十字键离散导航追踪状态（支持滞回与按住连发）
        self._left_dpad_dir: Literal["up", "down", "left", "right"] | None = None
        self._left_dpad_held_ms: float = 0.0
        self._left_dpad_next_repeat_ms: float = 0.0

    def update_keymap(self, keymap: KeymapConfig) -> None:
        """热更新键位与手感配置（死区、速度、gamma、长按与连发阈值）。"""
        self._keymap = keymap
        if getattr(self._keymap, "left_stick_mode", "dpad") != "dpad":
            self._left_dpad_dir = None
            self._left_dpad_held_ms = 0.0
            self._left_dpad_next_repeat_ms = 0.0

    def reset(self) -> list[GestureEvent]:
        """重置所有内部按键与摇杆状态，并返回用于清理 HUD 的归零事件。"""
        events: list[GestureEvent] = []
        for btn in _BUTTON_ORDER:
            track = self._tracks[btn]
            if not track.pressed:
                continue
            if btn in self._hold_buttons and (
                track.hold_progress_emitted or track.hold_triggered
            ):
                events.append(ButtonHoldProgress(button=btn, progress=0.0))
            if btn in _MODIFIER_SET and track.modifier_notified:
                events.append(ModifierStateChanged(modifier=btn, active=False))
            track.reset()

        self._prev_buttons = frozenset()
        self._stick_remainder["left"] = [0.0, 0.0]
        self._stick_remainder["right"] = [0.0, 0.0]
        self._left_dpad_dir = None
        self._left_dpad_held_ms = 0.0
        self._left_dpad_next_repeat_ms = 0.0
        return events

    def update(self, state: GamepadState, dt: float) -> list[GestureEvent]:
        """推进一帧手势状态机，返回本帧产生的全部高层事件列表。

        Args:
            state: 当前帧的手柄归一化状态。
            dt: 距上一帧的时间间隔（秒）。
        """
        dt_sec = max(0.0, float(dt))
        dt_ms = dt_sec * 1000.0

        if not state.connected:
            return self.reset()

        events: list[GestureEvent] = []
        current_buttons = self._extract_buttons(state)

        newly_pressed = current_buttons - self._prev_buttons
        still_held = current_buttons & self._prev_buttons
        newly_released = self._prev_buttons - current_buttons

        ready_hold_ms = max(0, int(getattr(self._keymap, "ready_hold_ms", 350)))
        repeat_initial_ms = max(1, int(getattr(self._keymap, "repeat_initial_ms", 260)))
        repeat_interval_ms = max(1, int(getattr(self._keymap, "repeat_interval_ms", 95)))

        # 确定本帧有效修饰键：优先使用已在按住中的修饰键；
        # 若无已按住修饰键，且本帧同时按下修饰键与其他键，则按优先级选取修饰键。
        active_modifier = self._resolve_active_modifier(still_held, newly_pressed)

        # 1. 先统一重置并初始化本帧新按下的所有按键追踪器，避免遍历顺序覆盖已标记的组合键状态
        for btn in _BUTTON_ORDER:
            if btn not in newly_pressed:
                continue
            track = self._tracks[btn]
            track.reset()
            track.pressed = True
            track.next_repeat_ms = float(repeat_initial_ms)

        # 若激活修饰键也是本帧新按下的，先派发其激活事件，保证事件顺序恒为先 ModifierStateChanged 后 ChordTriggered
        if active_modifier is not None and active_modifier in newly_pressed:
            mod_track = self._tracks[active_modifier]
            mod_track.modifier_notified = True
            events.append(ModifierStateChanged(modifier=active_modifier, active=True))
            self._try_absorb_pending_hold_chord(active_modifier, still_held, events)

        # 再处理其余本帧新按下的按键
        for btn in _BUTTON_ORDER:
            if btn not in newly_pressed or btn == active_modifier:
                continue

            track = self._tracks[btn]

            # 若存在激活修饰键，则触发组合键并双向抑制修饰键与子键后续单键动作
            if active_modifier is not None:
                mod_track = self._tracks[active_modifier]
                mod_track.consumed_by_chord = True
                track.consumed_by_chord = True
                events.append(ChordTriggered(modifier=active_modifier, button=btn))
                continue

            if btn in _MODIFIER_SET:
                track.modifier_notified = True
                events.append(ModifierStateChanged(modifier=btn, active=True))
                # 处理极短时间内先按住长按类键（如 START）再按修饰键（如 BACK）的错帧组合
                self._try_absorb_pending_hold_chord(btn, still_held, events)
                continue

            if btn in self._hold_buttons:
                if ready_hold_ms <= 0:
                    track.hold_triggered = True
                    track.hold_progress_emitted = True
                    events.append(ButtonHoldProgress(button=btn, progress=1.0))
                    events.append(ButtonHoldTriggered(button=btn))
                # 有长按阈值时，按下当帧暂不发 Tap，等待蓄力满或提前松开发 Tap
                continue

            # 普通动作键及连发键在按下瞬间零延迟触发 Tap
            events.append(ButtonTap(button=btn))

        # 2. 处理持续按住的按键（长按进度与连发节拍）
        for btn in _BUTTON_ORDER:
            if btn not in still_held:
                continue

            track = self._tracks[btn]
            track.held_ms += dt_ms

            if track.consumed_by_chord:
                continue

            if btn in self._hold_buttons:
                if not track.hold_triggered:
                    if ready_hold_ms <= 0 or track.held_ms >= ready_hold_ms:
                        track.hold_triggered = True
                        track.hold_progress_emitted = True
                        events.append(ButtonHoldProgress(button=btn, progress=1.0))
                        events.append(ButtonHoldTriggered(button=btn))
                    elif track.held_ms > 0.0:
                        progress = min(1.0, track.held_ms / float(ready_hold_ms))
                        track.hold_progress_emitted = True
                        events.append(ButtonHoldProgress(button=btn, progress=progress))
            elif btn in self._repeat_buttons:
                while track.held_ms >= track.next_repeat_ms:
                    events.append(ButtonRepeat(button=btn))
                    track.next_repeat_ms += float(repeat_interval_ms)

        # 3. 处理本帧松开的按键
        for btn in _BUTTON_ORDER:
            if btn not in newly_released:
                continue

            track = self._tracks[btn]

            if btn in self._hold_buttons:
                if not track.consumed_by_chord:
                    if not track.hold_triggered:
                        # 未满长按阈值即松开：先发 progress=0.0 清除 HUD 进度环，再发单键 Tap
                        events.append(ButtonHoldProgress(button=btn, progress=0.0))
                        events.append(ButtonTap(button=btn))
                    else:
                        # 已触发长按后松开：同样发送 0.0 归零以便 HUD 收起进度环，但不补发 Tap
                        events.append(ButtonHoldProgress(button=btn, progress=0.0))
            elif btn in _MODIFIER_SET:
                if not track.consumed_by_chord:
                    # 修饰键独立按下并松开（期间未组合其他键）：触发单独 Tap（如 BACK 切层）
                    events.append(ButtonTap(button=btn))
                if track.modifier_notified:
                    events.append(ModifierStateChanged(modifier=btn, active=False))

            track.reset()

        # 4. 处理左右摇杆位移积分（带 gamma 曲线与亚像素余量累加）与左摇杆离散方向导航
        left_motion = self._process_stick("left", state.lx, state.ly, dt_sec)
        if left_motion is not None:
            events.append(left_motion)

        if getattr(self._keymap, "left_stick_mode", "dpad") == "dpad":
            events.extend(
                self._process_left_stick_direction(
                    state.lx,
                    state.ly,
                    dt_ms,
                    repeat_initial_ms,
                    repeat_interval_ms,
                )
            )
        else:
            self._left_dpad_dir = None
            self._left_dpad_held_ms = 0.0
            self._left_dpad_next_repeat_ms = 0.0

        right_motion = self._process_stick("right", state.rx, state.ry, dt_sec)
        if right_motion is not None:
            events.append(right_motion)

        self._prev_buttons = frozenset(current_buttons)
        return events

    def _extract_buttons(self, state: GamepadState) -> set[GamepadButton]:
        """合并离散按钮集合与模拟扳机超阈值状态。"""
        threshold = (
            self._trigger_threshold_override
            if self._trigger_threshold_override is not None
            else float(getattr(self._keymap, "trigger_threshold", 0.45))
        )
        buttons = set(state.buttons)
        if state.lt >= threshold:
            buttons.add(GamepadButton.LT)
        if state.rt >= threshold:
            buttons.add(GamepadButton.RT)
        return buttons

    def _resolve_active_modifier(
        self,
        still_held: set[GamepadButton],
        newly_pressed: set[GamepadButton],
    ) -> GamepadButton | None:
        """按既定优先级解析本帧生效的修饰键。"""
        for mod in MODIFIER_BUTTONS:
            if mod in still_held:
                return mod

        if len(newly_pressed) >= 2:
            for mod in MODIFIER_BUTTONS:
                if mod in newly_pressed:
                    return mod

        return None

    def _try_absorb_pending_hold_chord(
        self,
        modifier: GamepadButton,
        still_held: set[GamepadButton],
        events: list[GestureEvent],
    ) -> None:
        """容错处理：若长按按键（如 START）刚按下极短时间便按下修饰键（如 BACK），合并为组合键。"""
        mod_track = self._tracks[modifier]
        for btn in _BUTTON_ORDER:
            if btn not in still_held or btn not in self._hold_buttons:
                continue
            track = self._tracks[btn]
            if (
                not track.consumed_by_chord
                and not track.hold_triggered
                and track.held_ms <= _REVERSE_CHORD_GRACE_MS
            ):
                if track.hold_progress_emitted:
                    events.append(ButtonHoldProgress(button=btn, progress=0.0))
                    track.hold_progress_emitted = False
                track.consumed_by_chord = True
                mod_track.consumed_by_chord = True
                events.append(ChordTriggered(modifier=modifier, button=btn))

    def _process_stick(
        self,
        stick: Literal["left", "right"],
        x: float,
        y: float,
        dt_sec: float,
    ) -> StickMotion | None:
        """计算摇杆屏幕像素位移增量，并维护亚像素余量累加器。"""
        remainder = self._stick_remainder[stick]
        magnitude = math.hypot(x, y)

        if magnitude <= 1e-6:
            remainder[0] = 0.0
            remainder[1] = 0.0
            return None

        clamped_mag = min(1.0, magnitude)
        gamma = float(getattr(self._keymap, "cursor_gamma", 2.2))
        max_speed = float(getattr(self._keymap, "cursor_max_speed_px", 1400.0))

        # 径向响应曲线：微推时极低速精确微调，推满时全向达到 max_speed
        speed_px = (clamped_mag**gamma) * max_speed * dt_sec
        raw_dx = (x / magnitude) * speed_px
        # 摇杆 y 向上为正，屏幕像素 y 向下为正，因此对 y 取反
        raw_dy = (-y / magnitude) * speed_px

        # 反向推杆时立即清零异号残余，避免换向迟滞
        if raw_dx * remainder[0] < 0.0:
            remainder[0] = 0.0
        if raw_dy * remainder[1] < 0.0:
            remainder[1] = 0.0

        remainder[0] += raw_dx
        remainder[1] += raw_dy

        dx_px = float(math.trunc(remainder[0]))
        dy_px = float(math.trunc(remainder[1]))
        remainder[0] -= dx_px
        remainder[1] -= dy_px

        return StickMotion(
            stick=stick,
            x=float(x),
            y=float(y),
            dx_px=dx_px,
            dy_px=dy_px,
        )

    def _process_left_stick_direction(
        self,
        x: float,
        y: float,
        dt_ms: float,
        repeat_initial_ms: int,
        repeat_interval_ms: int,
    ) -> list[StickDirection]:
        """将左摇杆模拟向量转换为带滞回防抖与连发节拍的离散四向导航事件。"""
        mag = math.hypot(x, y)
        trigger_thresh = float(getattr(self._keymap, "left_stick_dpad_threshold", 0.50))
        release_thresh = float(getattr(self._keymap, "left_stick_dpad_release_threshold", 0.30))

        if mag <= release_thresh:
            self._left_dpad_dir = None
            self._left_dpad_held_ms = 0.0
            self._left_dpad_next_repeat_ms = 0.0
            return []

        if mag < trigger_thresh:
            # 位于滞回带内：保持已有方向不提前复位，但不触发新方向或连发
            return []

        cardinal: Literal["up", "down", "left", "right"]
        if abs(x) >= abs(y):
            cardinal = "right" if x >= 0.0 else "left"
        else:
            cardinal = "up" if y >= 0.0 else "down"

        emitted: list[StickDirection] = []
        if cardinal != self._left_dpad_dir:
            self._left_dpad_dir = cardinal
            self._left_dpad_held_ms = 0.0
            self._left_dpad_next_repeat_ms = float(repeat_initial_ms)
            emitted.append(StickDirection(stick="left", direction=cardinal, is_repeat=False))
            return emitted

        self._left_dpad_held_ms += dt_ms
        while self._left_dpad_held_ms >= self._left_dpad_next_repeat_ms:
            emitted.append(StickDirection(stick="left", direction=cardinal, is_repeat=True))
            self._left_dpad_next_repeat_ms += float(repeat_interval_ms)
        return emitted
