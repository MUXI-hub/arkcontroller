"""XInput 驱动与手势状态机单元测试。"""

from __future__ import annotations

import ctypes
import math
import time
from typing import Any

import pytest

from arkcontroller.config.models import KeymapConfig
from arkcontroller.input.gesture_engine import (
    ButtonHoldProgress,
    ButtonHoldTriggered,
    ButtonRepeat,
    ButtonTap,
    ChordTriggered,
    GestureEngine,
    ModifierStateChanged,
    StickMotion,
)
from arkcontroller.input.xinput_driver import (
    ERROR_DEVICE_NOT_CONNECTED,
    ERROR_SUCCESS,
    XINPUT_GAMEPAD_A,
    XINPUT_GAMEPAD_DPAD_UP,
    XINPUT_GAMEPAD_START,
    GamepadButton,
    GamepadState,
    XInputDriver,
    apply_radial_deadzone,
)


class _FakeXInputDLL:
    """用于单测的 XInput DLL 替身，模拟手柄轮询与马达震动写入。"""

    def __init__(self) -> None:
        self.connected: bool = True
        self.packet_number: int = 1
        self.buttons_mask: int = 0
        self.left_trigger: int = 0
        self.right_trigger: int = 0
        self.thumb_lx: int = 0
        self.thumb_ly: int = 0
        self.thumb_rx: int = 0
        self.thumb_ry: int = 0
        self.vibrations: list[tuple[int, int, int]] = []

    def XInputGetState(self, user_index: int, state_ptr: Any) -> int:
        if not self.connected:
            return ERROR_DEVICE_NOT_CONNECTED
        state = state_ptr._obj
        state.dwPacketNumber = self.packet_number
        state.Gamepad.wButtons = self.buttons_mask
        state.Gamepad.bLeftTrigger = self.left_trigger
        state.Gamepad.bRightTrigger = self.right_trigger
        state.Gamepad.sThumbLX = self.thumb_lx
        state.Gamepad.sThumbLY = self.thumb_ly
        state.Gamepad.sThumbRX = self.thumb_rx
        state.Gamepad.sThumbRY = self.thumb_ry
        return ERROR_SUCCESS

    def XInputSetState(self, user_index: int, vib_ptr: Any) -> int:
        vib = vib_ptr._obj
        self.vibrations.append(
            (user_index, int(vib.wLeftMotorSpeed), int(vib.wRightMotorSpeed))
        )
        return ERROR_SUCCESS


def _make_state(
    *buttons: GamepadButton,
    connected: bool = True,
    lx: float = 0.0,
    ly: float = 0.0,
    rx: float = 0.0,
    ry: float = 0.0,
    lt: float = 0.0,
    rt: float = 0.0,
    packet: int = 1,
) -> GamepadState:
    return GamepadState(
        connected=connected,
        packet_number=packet,
        buttons=frozenset(buttons),
        lx=lx,
        ly=ly,
        rx=rx,
        ry=ry,
        lt=lt,
        rt=rt,
    )


class TestRadialDeadzoneAndXInputDriver:
    """校验径向缩放死区算法与 XInputDriver 硬件解析。"""

    def test_radial_deadzone_inside_and_on_boundary_returns_zero(self) -> None:
        assert apply_radial_deadzone(0.0, 0.0, 0.14) == (0.0, 0.0)
        assert apply_radial_deadzone(0.10, 0.0, 0.14) == (0.0, 0.0)
        assert apply_radial_deadzone(0.14, 0.0, 0.14) == (0.0, 0.0)
        # 对角线半径 sqrt(0.09^2 + 0.09^2) ≈ 0.127 < 0.14 应归零
        assert apply_radial_deadzone(0.09, 0.09, 0.14) == (0.0, 0.0)

    def test_radial_deadzone_continuity_and_direction_preservation(self) -> None:
        # 略超死区应平滑从接近 0 起步，无跳变
        out_x, out_y = apply_radial_deadzone(0.14 + 0.0086, 0.0, 0.14)
        assert out_x == pytest.approx(0.01, abs=1e-4)
        assert out_y == 0.0

        # 对角线方向 45 度保持严格不变，不存在十字轴吸附
        diag_x, diag_y = apply_radial_deadzone(0.5, 0.5, 0.14)
        assert diag_x == pytest.approx(diag_y, abs=1e-6)
        expected_mag = (math.hypot(0.5, 0.5) - 0.14) / (1.0 - 0.14)
        assert math.hypot(diag_x, diag_y) == pytest.approx(expected_mag, abs=1e-6)

        # 角落满推 (1.0, 1.0) 应钳制到单位圆内，不超过 1.0
        corner_x, corner_y = apply_radial_deadzone(1.0, 1.0, 0.14)
        assert math.hypot(corner_x, corner_y) == pytest.approx(1.0, abs=1e-6)
        assert corner_x == pytest.approx(corner_y, abs=1e-6)

    def test_xinput_driver_poll_parses_buttons_triggers_and_sticks(self) -> None:
        fake_dll = _FakeXInputDLL()
        fake_dll.packet_number = 42
        fake_dll.buttons_mask = XINPUT_GAMEPAD_A | XINPUT_GAMEPAD_START | XINPUT_GAMEPAD_DPAD_UP
        fake_dll.left_trigger = 128  # ~0.502 >= 0.45 -> 应置入 LT
        fake_dll.right_trigger = 80  # ~0.314 < 0.45 -> 不应置入 RT
        fake_dll.thumb_lx = 32767
        fake_dll.thumb_ly = 0
        fake_dll.thumb_rx = 0
        fake_dll.thumb_ry = -32768  # 向下满推 -> -1.0

        driver = XInputDriver(
            user_index=0,
            left_deadzone=0.14,
            right_deadzone=0.16,
            dll=fake_dll,
        )
        state = driver.poll()

        assert state.connected is True
        assert state.packet_number == 42
        assert GamepadButton.A in state.buttons
        assert GamepadButton.START in state.buttons
        assert GamepadButton.DPAD_UP in state.buttons
        assert GamepadButton.LT in state.buttons
        assert GamepadButton.RT not in state.buttons
        assert state.lt == pytest.approx(128 / 255.0, abs=1e-4)
        assert state.rt == pytest.approx(80 / 255.0, abs=1e-4)
        assert state.lx == pytest.approx(1.0, abs=1e-4)
        assert state.ly == pytest.approx(0.0, abs=1e-4)
        assert state.rx == pytest.approx(0.0, abs=1e-4)
        assert state.ry == pytest.approx(-1.0, abs=1e-4)

    def test_xinput_driver_disconnected_and_invalid_user_index(self) -> None:
        with pytest.raises(ValueError):
            XInputDriver(user_index=4)

        fake_dll = _FakeXInputDLL()
        fake_dll.connected = False
        driver = XInputDriver(dll=fake_dll)
        state = driver.poll()
        assert state.connected is False
        assert state.buttons == frozenset()
        assert driver.is_connected is False

    def test_xinput_driver_rumble_auto_stops_and_manual_stop(self) -> None:
        fake_dll = _FakeXInputDLL()
        driver = XInputDriver(dll=fake_dll)

        driver.rumble(0.5, 1.0, duration_ms=40)
        assert fake_dll.vibrations[0] == (0, int(round(0.5 * 65535)), 65535)

        # 等待异步定时器自动关闭马达
        deadline = time.monotonic() + 1.0
        while len(fake_dll.vibrations) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert len(fake_dll.vibrations) >= 2
        assert fake_dll.vibrations[-1] == (0, 0, 0)

        # 立即停止震动
        driver.rumble(0.8, 0.2, duration_ms=5000)
        driver.stop_rumble()
        assert fake_dll.vibrations[-1] == (0, 0, 0)

        # 强度为 0 时直接调用停震，且支持运行时更新死区与 close()
        driver.rumble(0.0, 0.0, duration_ms=50)
        assert fake_dll.vibrations[-1] == (0, 0, 0)
        driver.set_deadzones(0.20, 0.22)
        assert driver.left_deadzone == 0.20
        assert driver.right_deadzone == 0.22

        # 已连接状态下发生断线时应取消挂起震动定时器并更新连接状态
        driver.poll()
        assert driver.is_connected is True
        driver.rumble(0.4, 0.4, duration_ms=5000)
        fake_dll.connected = False
        assert driver.poll().connected is False
        assert driver.is_connected is False
        driver.close()


class TestGestureEngineButtonsAndChords:
    """校验单键点击、长按防误触读条、方向连发与组合键抑制。"""

    def test_regular_button_tap_fires_immediately_on_press_only(self) -> None:
        engine = GestureEngine(KeymapConfig())

        ev_press = engine.update(_make_state(GamepadButton.A), dt=0.008)
        assert ev_press == [ButtonTap(GamepadButton.A)]

        ev_hold = engine.update(_make_state(GamepadButton.A), dt=0.100)
        assert ev_hold == []

        ev_release = engine.update(_make_state(), dt=0.008)
        assert ev_release == []

    def test_start_hold_progress_and_trigger_suppresses_release_tap(self) -> None:
        keymap = KeymapConfig(ready_hold_ms=350)
        engine = GestureEngine(keymap)

        # 按下当帧不立即触发 Tap
        assert engine.update(_make_state(GamepadButton.START), dt=0.010) == []

        # 持续按住 175ms -> 进度 0.5
        ev_mid = engine.update(_make_state(GamepadButton.START), dt=0.175)
        assert len(ev_mid) == 1
        assert isinstance(ev_mid[0], ButtonHoldProgress)
        assert ev_mid[0].button == GamepadButton.START
        assert ev_mid[0].progress == pytest.approx(0.5, abs=1e-4)

        # 再按住 175ms 达到 350ms -> 进度 1.0 + ButtonHoldTriggered
        ev_done = engine.update(_make_state(GamepadButton.START), dt=0.175)
        assert ev_done == [
            ButtonHoldProgress(GamepadButton.START, 1.0),
            ButtonHoldTriggered(GamepadButton.START),
        ]

        # 继续按住不重复触发
        assert engine.update(_make_state(GamepadButton.START), dt=0.200) == []

        # 松开时仅归零 HUD 进度环，不补发 ButtonTap
        ev_rel = engine.update(_make_state(), dt=0.010)
        assert ev_rel == [ButtonHoldProgress(GamepadButton.START, 0.0)]

    def test_start_early_release_resets_progress_and_emits_tap(self) -> None:
        engine = GestureEngine(KeymapConfig(ready_hold_ms=350))

        engine.update(_make_state(GamepadButton.START), dt=0.010)
        ev_mid = engine.update(_make_state(GamepadButton.START), dt=0.140)
        assert ev_mid == [ButtonHoldProgress(GamepadButton.START, pytest.approx(0.4))]

        # 未满阈值提前松开 -> 先发 progress=0.0 清除 HUD 进度环，再触发单键 Tap
        ev_rel = engine.update(_make_state(), dt=0.010)
        assert ev_rel == [
            ButtonHoldProgress(GamepadButton.START, 0.0),
            ButtonTap(GamepadButton.START),
        ]

    def test_dpad_and_bumper_repeat_cadence(self) -> None:
        keymap = KeymapConfig(repeat_initial_ms=260, repeat_interval_ms=95)
        engine = GestureEngine(keymap)

        # 按下瞬间触发 Tap
        assert engine.update(_make_state(GamepadButton.DPAD_RIGHT), dt=0.010) == [
            ButtonTap(GamepadButton.DPAD_RIGHT)
        ]

        # 250ms < 260ms，尚未连发
        assert engine.update(_make_state(GamepadButton.DPAD_RIGHT), dt=0.250) == []

        # 再过 10ms 到达 260ms -> 首次连发
        assert engine.update(_make_state(GamepadButton.DPAD_RIGHT), dt=0.010) == [
            ButtonRepeat(GamepadButton.DPAD_RIGHT)
        ]

        # 再过 190ms (2 * 95ms) -> 精准连发 2 次
        assert engine.update(_make_state(GamepadButton.DPAD_RIGHT), dt=0.190) == [
            ButtonRepeat(GamepadButton.DPAD_RIGHT),
            ButtonRepeat(GamepadButton.DPAD_RIGHT),
        ]

        # RB 同样支持按住连发
        assert engine.update(_make_state(GamepadButton.RB), dt=0.010) == [
            ButtonTap(GamepadButton.RB)
        ]
        assert engine.update(_make_state(GamepadButton.RB), dt=0.260) == [
            ButtonRepeat(GamepadButton.RB)
        ]

    def test_chord_lt_plus_x_suppresses_both_modifier_and_child_taps(self) -> None:
        engine = GestureEngine(KeymapConfig())

        # 按下 LT -> 发出修饰键激活事件
        ev1 = engine.update(_make_state(GamepadButton.LT, lt=0.9), dt=0.010)
        assert ev1 == [ModifierStateChanged(GamepadButton.LT, True)]

        # 保持 LT 同时按下 X -> 触发组合键，不触发 ButtonTap(X)
        ev2 = engine.update(
            _make_state(GamepadButton.LT, GamepadButton.X, lt=0.9), dt=0.010
        )
        assert ev2 == [ChordTriggered(GamepadButton.LT, GamepadButton.X)]

        # 松开 X -> 被抑制，无事件
        ev3 = engine.update(_make_state(GamepadButton.LT, lt=0.9), dt=0.010)
        assert ev3 == []

        # 松开 LT -> 仅发出修饰键释放，不补发 ButtonTap(LT)
        ev4 = engine.update(_make_state(), dt=0.010)
        assert ev4 == [ModifierStateChanged(GamepadButton.LT, False)]

    def test_rt_hold_rapid_buy_multiple_a_presses(self) -> None:
        engine = GestureEngine(KeymapConfig())

        assert engine.update(_make_state(rt=0.8), dt=0.010) == [
            ModifierStateChanged(GamepadButton.RT, True)
        ]
        # 连续按两次 A（极速连买）
        assert engine.update(_make_state(GamepadButton.A, rt=0.8), dt=0.010) == [
            ChordTriggered(GamepadButton.RT, GamepadButton.A)
        ]
        assert engine.update(_make_state(rt=0.8), dt=0.010) == []
        assert engine.update(_make_state(GamepadButton.A, rt=0.8), dt=0.010) == [
            ChordTriggered(GamepadButton.RT, GamepadButton.A)
        ]
        assert engine.update(_make_state(), dt=0.010) == [
            ModifierStateChanged(GamepadButton.RT, False)
        ]

    def test_back_plus_start_suppresses_start_hold_and_back_tap(self) -> None:
        engine = GestureEngine(KeymapConfig(ready_hold_ms=350))

        # 1) 先按 BACK 再按 START 并长按超过 500ms
        assert engine.update(_make_state(GamepadButton.BACK), dt=0.010) == [
            ModifierStateChanged(GamepadButton.BACK, True)
        ]
        assert engine.update(
            _make_state(GamepadButton.BACK, GamepadButton.START), dt=0.010
        ) == [ChordTriggered(GamepadButton.BACK, GamepadButton.START)]
        # 长按期间严禁出现 START 读条或 HoldTriggered
        assert engine.update(
            _make_state(GamepadButton.BACK, GamepadButton.START), dt=0.500
        ) == []
        # 全部松开时严禁触发 START 或 BACK 单键 Tap
        assert engine.update(_make_state(), dt=0.010) == [
            ModifierStateChanged(GamepadButton.BACK, False)
        ]

        # 2) 同帧同时按下 BACK + START
        ev_same = engine.update(
            _make_state(GamepadButton.BACK, GamepadButton.START), dt=0.010
        )
        assert ModifierStateChanged(GamepadButton.BACK, True) in ev_same
        assert ChordTriggered(GamepadButton.BACK, GamepadButton.START) in ev_same
        assert not any(isinstance(e, ButtonTap) for e in ev_same)
        assert engine.update(_make_state(), dt=0.010) == [
            ModifierStateChanged(GamepadButton.BACK, False)
        ]

        # 独立轻按 BACK 后松开 -> 正常触发切层 Tap
        assert engine.update(_make_state(GamepadButton.BACK), dt=0.010) == [
            ModifierStateChanged(GamepadButton.BACK, True)
        ]
        assert engine.update(_make_state(), dt=0.010) == [
            ButtonTap(GamepadButton.BACK),
            ModifierStateChanged(GamepadButton.BACK, False),
        ]

    @pytest.mark.parametrize(
        ("modifier", "child"),
        [
            (GamepadButton.RT, GamepadButton.A),
            (GamepadButton.LT, GamepadButton.RB),
            (GamepadButton.BACK, GamepadButton.A),
            (GamepadButton.LT, GamepadButton.BACK),
        ],
    )
    def test_same_frame_chord_preserves_order_and_suppresses_all_taps(
        self,
        modifier: GamepadButton,
        child: GamepadButton,
    ) -> None:
        """回归测试：子键在枚举顺序中位于修饰键之前时，同帧组合仍须严格先发修饰键激活再发组合键，且松开后不产生任何 ButtonTap。"""
        engine = GestureEngine(KeymapConfig())

        ev_press = engine.update(_make_state(modifier, child), dt=0.010)
        assert ev_press == [
            ModifierStateChanged(modifier, True),
            ChordTriggered(modifier, child),
        ]

        # 持续按住超过连发阈值也不应触发 ButtonRepeat 或 ButtonTap
        ev_hold = engine.update(_make_state(modifier, child), dt=0.400)
        assert ev_hold == []

        # 全部松开时仅释放修饰键状态，严禁补发修饰键或子键的 ButtonTap
        ev_release = engine.update(_make_state(), dt=0.010)
        assert ev_release == [ModifierStateChanged(modifier, False)]
        assert not any(isinstance(e, ButtonTap) for e in ev_press + ev_hold + ev_release)

    def test_lt_plus_back_and_lt_peek_modal_choice(self) -> None:
        engine = GestureEngine(KeymapConfig())

        # LT + BACK 暂停组合键
        assert engine.update(_make_state(GamepadButton.LT), dt=0.010) == [
            ModifierStateChanged(GamepadButton.LT, True)
        ]
        assert engine.update(
            _make_state(GamepadButton.LT, GamepadButton.BACK), dt=0.010
        ) == [ChordTriggered(GamepadButton.LT, GamepadButton.BACK)]
        assert engine.update(_make_state(), dt=0.010) == [
            ModifierStateChanged(GamepadButton.LT, False)
        ]

        # 单独按住 LT 查看战场再松开
        assert engine.update(_make_state(GamepadButton.LT), dt=0.010) == [
            ModifierStateChanged(GamepadButton.LT, True)
        ]
        assert engine.update(_make_state(GamepadButton.LT), dt=1.0) == []
        ev_rel = engine.update(_make_state(), dt=0.010)
        assert ModifierStateChanged(GamepadButton.LT, False) in ev_rel

    def test_disconnect_mid_hold_emits_cleanup_events(self) -> None:
        engine = GestureEngine(KeymapConfig(ready_hold_ms=350))
        engine.update(_make_state(GamepadButton.START, GamepadButton.LT), dt=0.010)
        # 独立长按 START 并激活 LT 后断线
        engine.update(_make_state(), dt=0.010)
        engine.update(_make_state(GamepadButton.START), dt=0.010)
        engine.update(_make_state(GamepadButton.START), dt=0.100)

        ev_disc = engine.update(_make_state(connected=False), dt=0.010)
        assert ButtonHoldProgress(GamepadButton.START, 0.0) in ev_disc

    def test_start_hold_zero_threshold_and_reverse_skew_back_start_chord(self) -> None:
        # 1) ready_hold_ms=0 时按下即触发准备就绪，无除零异常
        instant_engine = GestureEngine(KeymapConfig(ready_hold_ms=0))
        ev_instant = instant_engine.update(_make_state(GamepadButton.START), dt=0.010)
        assert ev_instant == [
            ButtonHoldProgress(GamepadButton.START, 1.0),
            ButtonHoldTriggered(GamepadButton.START),
        ]

        # 2) START 比 BACK 早按极短时间（人手错帧容差内）仍自动合并为 BACK + START 并归零进度环
        engine = GestureEngine(KeymapConfig(ready_hold_ms=350))
        engine.update(_make_state(GamepadButton.START), dt=0.010)
        ev_prog = engine.update(_make_state(GamepadButton.START), dt=0.040)
        assert len(ev_prog) == 1 and isinstance(ev_prog[0], ButtonHoldProgress)

        ev_skew = engine.update(
            _make_state(GamepadButton.START, GamepadButton.BACK), dt=0.010
        )
        assert ModifierStateChanged(GamepadButton.BACK, True) in ev_skew
        assert ButtonHoldProgress(GamepadButton.START, 0.0) in ev_skew
        assert ChordTriggered(GamepadButton.BACK, GamepadButton.START) in ev_skew
        # 后续松开不触发单键 Tap
        ev_end = engine.update(_make_state(), dt=0.010)
        assert ev_end == [ModifierStateChanged(GamepadButton.BACK, False)]


class TestGestureEngineStickIntegration:
    """校验摇杆坐标方向约定、gamma 曲线与亚像素余量累加器。"""

    def test_stick_screen_direction_convention(self) -> None:
        keymap = KeymapConfig(cursor_max_speed_px=1000.0, cursor_gamma=2.0)
        engine = GestureEngine(keymap)

        # 向右满推 -> dx_px > 0
        ev_right = engine.update(_make_state(lx=1.0, ly=0.0), dt=0.010)
        assert len(ev_right) == 1
        assert isinstance(ev_right[0], StickMotion)
        assert ev_right[0].stick == "left"
        assert ev_right[0].dx_px == 10.0
        assert ev_right[0].dy_px == 0.0

        # 向上满推 (ly = +1.0) -> 屏幕坐标向上 dy_px < 0
        ev_up = engine.update(_make_state(lx=0.0, ly=1.0), dt=0.010)
        assert ev_up[0].dx_px == 0.0
        assert ev_up[0].dy_px == -10.0

        # 向下满推 (ly = -1.0) -> 屏幕坐标向下 dy_px > 0
        ev_down = engine.update(_make_state(lx=0.0, ly=-1.0), dt=0.010)
        assert ev_down[0].dx_px == 0.0
        assert ev_down[0].dy_px == 10.0

        # 归中不产生 StickMotion 事件
        assert engine.update(_make_state(lx=0.0, ly=0.0), dt=0.010) == []

    def test_stick_gamma_curve_and_subpixel_accumulator(self) -> None:
        keymap = KeymapConfig(cursor_max_speed_px=1400.0, cursor_gamma=2.2)
        engine = GestureEngine(keymap)

        # 极微小推杆：单帧位移约 (0.1 ** 2.2) * 1400 * 0.01 ≈ 0.0883 px
        # 连续 100 帧后应累计稳定步进约 8 px（绝不会因逐帧截断丢失成 0）
        total_dx = 0.0
        step_frames = 0
        dt = 0.010
        frames = 100
        for _ in range(frames):
            events = engine.update(_make_state(lx=0.1, ly=0.0), dt=dt)
            assert len(events) == 1
            motion = events[0]
            assert isinstance(motion, StickMotion)
            total_dx += motion.dx_px
            if motion.dx_px > 0:
                step_frames += 1

        expected_exact = (0.1**2.2) * 1400.0 * dt * frames
        assert total_dx == pytest.approx(math.floor(expected_exact), abs=1.0)
        assert step_frames == int(total_dx)  # 每次恰好丝滑步进 1 像素

    def test_update_keymap_hot_reload_takes_effect_immediately(self) -> None:
        engine = GestureEngine(KeymapConfig(cursor_max_speed_px=1000.0, cursor_gamma=1.0))
        ev1 = engine.update(_make_state(lx=1.0), dt=0.010)
        assert isinstance(ev1[0], StickMotion)
        assert ev1[0].dx_px == 10.0

        engine.update_keymap(KeymapConfig(cursor_max_speed_px=2000.0, cursor_gamma=1.0))
        ev2 = engine.update(_make_state(lx=1.0), dt=0.010)
        assert isinstance(ev2[0], StickMotion)
        assert ev2[0].dx_px == 20.0

    def test_right_stick_motion_and_reverse_direction_clears_remainder(self) -> None:
        engine = GestureEngine(KeymapConfig(cursor_max_speed_px=100.0, cursor_gamma=1.0))
        # 向右微推积累 0.6 亚像素余量后突然反推向左，应清零异号余量避免迟滞
        ev_pos = engine.update(_make_state(lx=0.6, ry=-0.8), dt=0.010)
        assert any(isinstance(e, StickMotion) and e.stick == "left" and e.dx_px == 0.0 for e in ev_pos)
        right_events = [e for e in ev_pos if isinstance(e, StickMotion) and e.stick == "right"]
        assert len(right_events) == 1
        assert right_events[0].y == pytest.approx(-0.8)
        assert right_events[0].dy_px == 0.0

        # 立即反向全推，确保位移为负而非被旧正余量抵消
        ev_neg = engine.update(_make_state(lx=-1.0, ry=1.0), dt=0.010)
        assert StickMotion(stick="left", x=-1.0, y=0.0, dx_px=-1.0, dy_px=0.0) in ev_neg
        assert StickMotion(stick="right", x=0.0, y=1.0, dx_px=0.0, dy_px=-1.0) in ev_neg
