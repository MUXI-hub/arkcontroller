"""单元测试：两段式部署状态机与全流程操控调度中枢（Mock VirtualHID）。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import pytest

from arkcontroller.config.loader import ConfigLoader
from arkcontroller.config.models import KeymapConfig, LayoutProfile, LayoutViewState, LinearZone
from arkcontroller.control import (
    ControlDispatcher,
    DeployState,
    FocusZone,
    LinearZoneMapper,
    PerspectiveBoardMapper,
)
from arkcontroller.input.gesture_engine import (
    ButtonHoldProgress,
    ButtonHoldTriggered,
    ButtonRepeat,
    ButtonTap,
    ChordTriggered,
    GestureEngine,
    ModifierStateChanged,
    StickDirection,
    StickMotion,
)
from arkcontroller.input.xinput_driver import GamepadButton, GamepadState
from arkcontroller.platform.window_tracker import ViewportRect
from arkcontroller.vision.scene_detector import GameScene, SceneDetectionResult


class MockVirtualHID:
    """记录全部键鼠模拟调用序列的测试替身。"""

    def __init__(self, *, cursor_pos: tuple[int, int] | None = None) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.cursor_pos: tuple[int, int] | None = cursor_pos

    def clear(self) -> None:
        self.calls.clear()

    def get_cursor_pos(self) -> tuple[int, int] | None:
        return self.cursor_pos

    def move_cursor(self, screen_x: int, screen_y: int) -> None:
        self.calls.append(("move_cursor", screen_x, screen_y))
        if self.cursor_pos is not None:
            self.cursor_pos = (screen_x, screen_y)

    def click(
        self,
        screen_x: int,
        screen_y: int,
        button: str = "left",
        hold_ms: int = 30,
    ) -> None:
        self.calls.append(("click", screen_x, screen_y, button))
        if self.cursor_pos is not None:
            self.cursor_pos = (screen_x, screen_y)

    def double_click(
        self,
        screen_x: int,
        screen_y: int,
        interval_ms: int = 55,
    ) -> None:
        self.calls.append(("double_click", screen_x, screen_y))
        if self.cursor_pos is not None:
            self.cursor_pos = (screen_x, screen_y)

    def smooth_drag(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        duration_ms: int = 110,
        steps: int = 10,
        release_at_end: bool = True,
    ) -> None:
        self.calls.append(("smooth_drag", start, end, release_at_end))

    def smooth_drag_loopback(
        self,
        origin: tuple[int, int],
        offset: tuple[int, int],
        duration_ms: int = 130,
        steps_each_way: int = 6,
        release_at_end: bool = True,
    ) -> None:
        self.calls.append(("smooth_drag_loopback", origin, offset, release_at_end))

    def mouse_up(self, button: str = "left") -> None:
        self.calls.append(("mouse_up", button))

    def directional_flick(
        self,
        origin: tuple[int, int],
        direction: Literal["up", "down", "left", "right"],
        distance_px: int,
    ) -> None:
        self.calls.append(("directional_flick", origin, direction, distance_px))

    def tap_key(self, key_name: str, hold_ms: int = 30) -> None:
        self.calls.append(("tap_key", key_name))

    def key_down(self, key_name: str) -> None:
        self.calls.append(("key_down", key_name))

    def key_up(self, key_name: str) -> None:
        self.calls.append(("key_up", key_name))

    def scroll(self, wheel_delta: int, horizontal: bool = False) -> None:
        self.calls.append(("scroll", wheel_delta, horizontal))


def _make_viewport(*, is_foreground: bool = True) -> ViewportRect:
    return ViewportRect(
        hwnd=1001,
        client_left=0,
        client_top=0,
        client_width=1920,
        client_height=1080,
        active_left=0,
        active_top=0,
        active_width=1920,
        active_height=1080,
        is_foreground=is_foreground,
        window_title="Arknights",
    )


def _make_env(
    *,
    use_native_shortcuts: bool = True,
    allow_retreat_chord: bool = True,
    left_stick_mode: str = "dpad",
    auto_toggle_shop: bool = True,
) -> tuple[
    ControlDispatcher,
    MockVirtualHID,
    list[tuple[float, float, int]],
    KeymapConfig,
    LayoutProfile,
    ViewportRect,
]:
    keymap, layout = ConfigLoader(use_user_override=False).load()
    keymap = replace(
        keymap,
        use_native_keyboard_shortcuts=use_native_shortcuts,
        allow_retreat_chord=allow_retreat_chord,
        left_stick_mode=left_stick_mode,
        auto_toggle_shop=auto_toggle_shop,
    )
    hid = MockVirtualHID()
    rumbles: list[tuple[float, float, int]] = []
    dispatcher = ControlDispatcher(
        keymap=keymap,
        layout=layout,
        hid=hid,  # type: ignore[arg-type]
        rumble_cb=lambda l, r, d: rumbles.append((l, r, d)),
    )
    return dispatcher, hid, rumbles, keymap, layout, _make_viewport(is_foreground=True)


class TestForegroundAndGlobalControls:
    """验证窗口前景安全守卫、暂停、校准呼出、手动锁层与自由光标混合模式。"""

    def test_background_or_missing_window_blocks_all_hid_output(self) -> None:
        dispatcher, hid, _, _, _, _ = _make_env()
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.95, shop_open=True)

        # 未找到游戏窗口
        snap_none = dispatcher.process_tick(
            None, prep, [ButtonTap(GamepadButton.A), ButtonTap(GamepadButton.X)]
        )
        assert snap_none.visible is False
        assert hid.calls == []

        # 游戏窗口处于后台
        bg_vp = _make_viewport(is_foreground=False)
        snap_bg = dispatcher.process_tick(
            bg_vp,
            prep,
            [
                ButtonTap(GamepadButton.A),
                ChordTriggered(GamepadButton.RT, GamepadButton.A),
                StickMotion("left", 0.9, 0.0, 40.0, 0.0),
            ],
        )
        assert snap_bg.visible is False
        assert hid.calls == []

    def test_global_chords_calibration_pause_and_manual_lock(self) -> None:
        dispatcher, hid, _, _, _, vp = _make_env()
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.95, shop_open=True)

        # 1. Back + Start 呼出校准向导
        snap = dispatcher.process_tick(
            vp, prep, [ChordTriggered(GamepadButton.BACK, GamepadButton.START)]
        )
        assert snap.visible is True
        assert snap.calibration_requested is True
        dispatcher.acknowledge_calibration_request()
        snap_after_ack = dispatcher.process_tick(vp, prep, [])
        assert snap_after_ack.calibration_requested is False

        # 2. LT + Back 切换暂停，暂停期间所有常规输入均不发往 HID
        snap_paused = dispatcher.process_tick(
            vp,
            prep,
            [
                ChordTriggered(GamepadButton.LT, GamepadButton.BACK),
                ButtonTap(GamepadButton.A),
                ButtonTap(GamepadButton.X),
            ],
        )
        assert snap_paused.paused is True
        assert snap_paused.visible is True
        assert hid.calls == []

        # 再次 LT + Back 恢复操控
        snap_resumed = dispatcher.process_tick(
            vp, prep, [ChordTriggered(GamepadButton.LT, GamepadButton.BACK)]
        )
        assert snap_resumed.paused is False

        # 3. Back 单击手动循环锁层：BATTLE_PREP -> MODAL_CHOICE -> MODAL_CONFIRM -> UNKNOWN_FREE_CURSOR -> AUTO
        snap_lock1 = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.BACK)])
        assert snap_lock1.scene_locked_manual is True
        assert snap_lock1.scene == GameScene.BATTLE_PREP

        # 切到 MODAL_CHOICE 后视觉仍报 BATTLE_PREP，因手动锁层不会被覆盖
        snap_lock2 = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.BACK)])
        assert snap_lock2.scene_locked_manual is True
        assert snap_lock2.scene == GameScene.MODAL_CHOICE
        snap_still_locked = dispatcher.process_tick(vp, prep, [])
        assert snap_still_locked.scene == GameScene.MODAL_CHOICE

        # 继续按 Back 依次走完 MODAL_CONFIRM -> UNKNOWN_FREE_CURSOR
        for expected in (GameScene.MODAL_CONFIRM, GameScene.UNKNOWN_FREE_CURSOR):
            snap_step = dispatcher.process_tick(
                vp, prep, [ButtonTap(GamepadButton.BACK)]
            )
            assert snap_step.scene == expected
            assert snap_step.scene_locked_manual is True

        # 再按一次 Back 回到自动识别模式，并立即同步回 prep 场景
        snap_auto = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.BACK)])
        assert snap_auto.scene_locked_manual is False
        snap_synced = dispatcher.process_tick(vp, prep, [])
        assert snap_synced.scene == GameScene.BATTLE_PREP

    def test_left_stick_free_cursor_and_dpad_snap_recovery(self) -> None:
        # 1. 默认 left_stick_mode="dpad"：按 LS 一键切入自由光标，移动并点击，再按十字键恢复吸附
        dispatcher, hid, _, _, _, vp = _make_env(left_stick_mode="dpad")
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.95, shop_open=True)
        dispatcher.process_tick(vp, prep, [])
        hid.clear()

        # 未切自由光标前，纯左摇杆 StickMotion 不强制踢入自由光标模式
        snap_stay = dispatcher.process_tick(
            vp, prep, [StickMotion("left", 0.8, -0.4, 96.0, 54.0)]
        )
        assert snap_stay.free_cursor_active is False
        assert hid.calls == []

        # 按 LS 切入自由光标模式（立即将真实光标落到当前吸附槽位），随后左摇杆平滑移动鼠标并按 A 点击
        snap = dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.LS),
                StickMotion("left", 0.8, -0.4, 96.0, 54.0),
                ButtonTap(GamepadButton.A),
            ],
        )
        assert snap.free_cursor_active is True
        assert snap.focus_zone == FocusZone.FREE_CURSOR
        assert snap.focus_polygon_norm is None
        assert hid.calls[0][0] == "move_cursor"
        assert hid.calls[1][0] == "move_cursor"
        assert hid.calls[2][0] == "click"
        assert hid.calls[2][1:3] == hid.calls[1][1:3]

        # 右摇杆触发垂直与水平滚轮
        hid.clear()
        dispatcher.process_tick(
            vp,
            prep,
            [
                StickMotion("right", 0.0, 0.8, 0.0, -48.0),
                StickMotion("right", 0.8, 0.0, 48.0, 0.0),
            ],
        )
        assert ("scroll", 240, False) in hid.calls
        assert ("scroll", 240, True) in hid.calls

        # 按十字键或 LS 无缝恢复槽位吸附模式
        snap_back = dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.DPAD_RIGHT)]
        )
        assert snap_back.free_cursor_active is False
        assert snap_back.focus_zone == FocusZone.SHOP
        assert snap_back.focus_polygon_norm is not None

        # 2. 兼容模式 left_stick_mode="cursor"：推动左摇杆仍可立即切入自由光标
        dispatcher_cur, hid_cur, _, _, _, vp_cur = _make_env(left_stick_mode="cursor")
        dispatcher_cur.process_tick(vp_cur, prep, [])
        hid_cur.clear()
        snap_cur = dispatcher_cur.process_tick(
            vp_cur, prep, [StickMotion("left", 0.8, 0.0, 60.0, 0.0)]
        )
        assert snap_cur.free_cursor_active is True
        assert hid_cur.calls[0][0] == "move_cursor"

    def test_right_stick_scroll_125hz_gesture_engine_no_subpixel_burst(self) -> None:
        """回归测试（#15）：125Hz 下微推右摇杆亚像素帧 dy_px==0 不得暴增滚轮，且倾角越大滚动越快。"""
        dispatcher_light, hid_light, _, keymap, _, vp = _make_env()
        dispatcher_med, hid_med, _, _, _, _ = _make_env()
        ge_light = GestureEngine(keymap)
        ge_med = GestureEngine(keymap)
        free_scene = SceneDetectionResult(GameScene.UNKNOWN_FREE_CURSOR, 0.0)
        dt = 0.008  # 125Hz 主循环帧间隔

        # 1. 显式亚像素零增量帧不得触发任何滚轮事件
        dispatcher_light.process_tick(
            vp,
            free_scene,
            [StickMotion("right", 0.0, 0.30, 0.0, 0.0) for _ in range(20)],
        )
        assert hid_light.calls == []

        # 2. 联动真实 GestureEngine 运行 50 帧（0.4s）：比较微推 ry=0.30 与中推 ry=0.45
        for i in range(50):
            st_light = GamepadState(
                connected=True,
                packet_number=i + 1,
                buttons=frozenset(),
                lx=0.0,
                ly=0.0,
                rx=0.0,
                ry=0.30,
                lt=0.0,
                rt=0.0,
            )
            st_med = GamepadState(
                connected=True,
                packet_number=i + 1,
                buttons=frozenset(),
                lx=0.0,
                ly=0.0,
                rx=0.0,
                ry=0.45,
                lt=0.0,
                rt=0.0,
            )
            dispatcher_light.process_tick(vp, free_scene, ge_light.update(st_light, dt))
            dispatcher_med.process_tick(vp, free_scene, ge_med.update(st_med, dt))

        light_v_scroll = sum(
            call[1] for call in hid_light.calls if call[0] == "scroll" and not call[2]
        )
        med_v_scroll = sum(
            call[1] for call in hid_med.calls if call[0] == "scroll" and not call[2]
        )
        # 微推 0.4s 约累计 39px（仅 1 次 +120），严格小于中推（约 96px -> 4 次 +480），绝不暴增
        assert 0 < light_v_scroll <= 240
        assert med_v_scroll > light_v_scroll

        # 3. 横向推满伴随极小垂直偏移（rx=1.0, ry=0.05）运行 10 帧：仅触发水平滚轮，无垂直误滚
        dispatcher_h, hid_h, _, _, _, _ = _make_env()
        ge_h = GestureEngine(keymap)
        for i in range(10):
            st_h = GamepadState(
                connected=True,
                packet_number=i + 1,
                buttons=frozenset(),
                lx=0.0,
                ly=0.0,
                rx=1.0,
                ry=0.05,
                lt=0.0,
                rt=0.0,
            )
            dispatcher_h.process_tick(vp, free_scene, ge_h.update(st_h, dt))

        h_scrolls = [c for c in hid_h.calls if c[0] == "scroll" and c[2] is True]
        v_scrolls = [c for c in hid_h.calls if c[0] == "scroll" and c[2] is False]
        assert len(h_scrolls) >= 2
        assert v_scrolls == []


class TestDeployFSMAndBattlePrep:
    """验证卫戍协议核心两段式部署、换位、穿装备、出售、赠卡与商店运营闭环。"""

    def test_bench_grab_board_place_and_right_stick_directional_flick(self) -> None:
        """核心流程：整备区抓人 -> 棋盘落位 -> 推右摇杆向右滑动定朝向并微震动。"""
        dispatcher, hid, rumbles, keymap, layout, vp = _make_env()
        bench_mapper = LinearZoneMapper(layout.bench_zone)
        board_mapper = PerspectiveBoardMapper(layout.board_quad)
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.96, shop_open=False)

        # 进入休整期（商店未展开 -> 默认焦点在整备区 0 号槽）
        snap0 = dispatcher.process_tick(vp, prep, [])
        assert snap0.focus_zone == FocusZone.BENCH
        assert snap0.deploy_state == DeployState.IDLE

        # 右移 2 格到整备区第 3 槽（index=2），按 A 抓取干员
        snap_hold = dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.DPAD_RIGHT),
                ButtonRepeat(GamepadButton.DPAD_RIGHT),
                ButtonTap(GamepadButton.A),
            ],
        )
        bench2_norm = bench_mapper.slot_center(2)
        bench2_xy = vp.norm_to_screen(*bench2_norm)

        assert snap_hold.deploy_state == DeployState.HOLDING_UNIT
        assert snap_hold.holding_source_norm == pytest.approx(bench2_norm)
        # 抓取干员时不提前单击源位置，避免触发选取角色界面缩放错位
        assert hid.calls == []
        assert len(rumbles) == 1

        # 抓取过程中即使视觉短暂波动为 BATTLE_COMBAT 也不打断抓取态
        snap_protected = dispatcher.process_tick(
            vp, SceneDetectionResult(GameScene.BATTLE_COMBAT, 0.85), []
        )
        assert snap_protected.scene == GameScene.BATTLE_PREP
        assert snap_protected.deploy_state == DeployState.HOLDING_UNIT

        # 按 DPAD_UP 直接跨区上棋盘底行，再上移 2 行落至目标棋盘格，按 A 执行第一段拖拽落位
        hid.clear()
        snap_aim = dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.DPAD_UP),
                ButtonTap(GamepadButton.DPAD_UP),
                ButtonTap(GamepadButton.DPAD_UP),
                ButtonTap(GamepadButton.A),
            ],
        )
        assert snap_aim.focus_zone == FocusZone.BOARD
        assert snap_aim.deploy_state == DeployState.AIMING_DIRECTION
        assert snap_aim.aim_origin_norm is not None

        target_cell = dispatcher.deploy_fsm.aim_cell
        assert target_cell is not None
        assert target_cell[0] == board_mapper.rows - 3
        board_xy = vp.norm_to_screen(*board_mapper.cell_center(*target_cell))
        assert hid.calls == [("smooth_drag", bench2_xy, board_xy, True)]
        assert len(rumbles) == 2

        # 轻推右摇杆向右（幅度 0.40）：仅预览朝向高亮，尚未提交滑动
        hid.clear()
        snap_preview = dispatcher.process_tick(
            vp, prep, [StickMotion("right", 0.40, 0.05, 10.0, 0.0)]
        )
        assert snap_preview.deploy_state == DeployState.AIMING_DIRECTION
        assert snap_preview.aim_preview_dir == "right"
        assert hid.calls == []

        # 推满右摇杆向右（幅度 0.90）：执行第二段 directional_flick 定朝向并触发确认震动
        snap_done = dispatcher.process_tick(
            vp, prep, [StickMotion("right", 0.90, 0.0, 35.0, 0.0)]
        )
        expected_dist = int(round(vp.active_width * keymap.flick_distance_ratio))
        assert snap_done.deploy_state == DeployState.IDLE
        assert snap_done.aim_origin_norm is None
        assert snap_done.aim_preview_dir is None
        assert hid.calls == [("directional_flick", board_xy, "right", expected_dist)]
        assert len(rumbles) == 3

    def test_aiming_dpad_confirm_and_b_safe_cancel(self) -> None:
        dispatcher, hid, _, keymap, layout, vp = _make_env()
        board_mapper = PerspectiveBoardMapper(layout.board_quad)
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.95, shop_open=False)
        dispatcher.process_tick(vp, prep, [])

        # 抓取 -> 落位棋盘 -> 十字键向左直接完成定朝向
        dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.A),
                ButtonTap(GamepadButton.RB),  # 抓取态 LB/RB 在整备区<->棋盘间快速切换
                ButtonTap(GamepadButton.A),
            ],
        )
        assert dispatcher.deploy_fsm.state == DeployState.AIMING_DIRECTION
        aim_xy = vp.norm_to_screen(*dispatcher.deploy_fsm.aim_origin_norm)  # type: ignore[misc]
        hid.clear()

        snap_done = dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.DPAD_LEFT)]
        )
        expected_dist = int(round(vp.active_width * keymap.flick_distance_ratio))
        assert snap_done.deploy_state == DeployState.IDLE
        assert hid.calls == [("directional_flick", aim_xy, "left", expected_dist)]

        # 再次抓棋盘干员移至邻格后按 B 安全取消落位（官方 PC 端不响应右键，应左键点击菱形左上红色 ×）
        dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.A),
                ButtonTap(GamepadButton.DPAD_RIGHT),
                ButtonTap(GamepadButton.A),
            ],
        )
        assert dispatcher.deploy_fsm.state == DeployState.AIMING_DIRECTION
        aim_u, aim_v = dispatcher.deploy_fsm.aim_origin_norm  # type: ignore[misc]
        cancel_xy = vp.norm_to_screen(max(0.02, aim_u - 0.10), max(0.02, aim_v - 0.15))
        hid.clear()

        snap_cancelled = dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.B)]
        )
        assert snap_cancelled.deploy_state == DeployState.IDLE
        assert hid.calls == [("click", cancel_xy[0], cancel_xy[1], "left")]
        assert board_mapper.rows == 4
        assert board_mapper.cols == 7

    def test_bench_internal_swap_and_self_slot_noop(self) -> None:
        """验证整备区内部换位及同槽二次按 A 取消不发冗余拖拽。"""
        dispatcher, hid, _, _, layout, vp = _make_env()
        bench_mapper = LinearZoneMapper(layout.bench_zone)
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.95, shop_open=False)
        dispatcher.process_tick(vp, prep, [])

        # 抓取 0 号整备槽 -> 移到 4 号整备槽 -> 按 A 平滑拖拽互换
        dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.A),
                ButtonTap(GamepadButton.DPAD_RIGHT),
                ButtonTap(GamepadButton.DPAD_RIGHT),
                ButtonTap(GamepadButton.DPAD_RIGHT),
                ButtonTap(GamepadButton.DPAD_RIGHT),
                ButtonTap(GamepadButton.A),
            ],
        )
        s0_xy = vp.norm_to_screen(*bench_mapper.slot_center(0))
        s4_xy = vp.norm_to_screen(*bench_mapper.slot_center(4))
        assert hid.calls == [
            ("smooth_drag", s0_xy, s4_xy, True),
        ]
        assert dispatcher.deploy_fsm.state == DeployState.IDLE

        # 在同一槽位抓起再按 A，不应发出任何多余点击或拖拽
        hid.clear()
        dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.A), ButtonTap(GamepadButton.A)]
        )
        assert hid.calls == []
        assert dispatcher.deploy_fsm.state == DeployState.IDLE

    def test_equipment_drag_to_board_and_bench_units(self) -> None:
        """验证装备栏抓取并拖拽穿戴至场上干员与整备区干员。"""
        dispatcher, hid, _, _, layout, vp = _make_env()
        equip_mapper = LinearZoneMapper(layout.equip_zone)
        board_mapper = PerspectiveBoardMapper(layout.board_quad)
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.95, shop_open=True)
        dispatcher.process_tick(vp, prep, [])

        # 从 SHOP 按 LB 一步循环到横向 EQUIPMENT，右移至 1 号装备格，按 A 抓取装备（同时自动收起商店恢复标准视角）
        snap_eq = dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.LB),
                ButtonTap(GamepadButton.DPAD_RIGHT),
                ButtonTap(GamepadButton.A),
            ],
        )
        assert snap_eq.focus_zone == FocusZone.EQUIPMENT
        assert snap_eq.deploy_state == DeployState.HOLDING_EQUIP
        assert snap_eq.shop_open is False

        eq1_xy = vp.norm_to_screen(*equip_mapper.slot_center(1))
        hid.clear()

        # 按 RB 直接切到 BOARD，按 A 将装备平滑拖拽到当前焦点干员完成穿戴
        snap_equipped = dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.RB),
                ButtonTap(GamepadButton.A),
            ],
        )
        assert snap_equipped.focus_zone == FocusZone.BOARD
        assert snap_equipped.deploy_state == DeployState.IDLE
        board_center_xy = vp.norm_to_screen(
            *board_mapper.cell_center(board_mapper.rows // 2, board_mapper.cols // 2)
        )
        assert hid.calls == [("smooth_drag", eq1_xy, board_center_xy, True)]

    def test_quick_sell_native_vs_mouse_dropzone_and_no_coop_gift(self) -> None:
        """验证抓取态按 X 出售（官方实机整备/场上/装备统一 X，关闭快捷键拖 sell_dropzone），且按 Y 不再误触发赠送队友。"""
        # 1. 原生快捷键开启：整备区、棋盘、装备栏均发送官方出售键 X
        dispatcher, hid, _, _, layout, vp = _make_env(use_native_shortcuts=True)
        assert not hasattr(dispatcher.deploy_fsm, "gift_holding")
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.95, shop_open=False)
        dispatcher.process_tick(vp, prep, [])

        # 整备区抓起按 X -> x
        hid.clear()
        dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.A), ButtonTap(GamepadButton.X)]
        )
        assert ("tap_key", "x") in hid.calls
        assert dispatcher.deploy_fsm.state == DeployState.IDLE

        # 切到棋盘抓起按 X -> x
        hid.clear()
        dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.RB),
                ButtonTap(GamepadButton.A),
                ButtonTap(GamepadButton.X),
            ],
        )
        assert ("tap_key", "x") in hid.calls
        assert dispatcher.deploy_fsm.state == DeployState.IDLE

        # 切到装备栏抓起按 X -> x
        hid.clear()
        dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.RB),
                ButtonTap(GamepadButton.A),
                ButtonTap(GamepadButton.X),
            ],
        )
        assert ("tap_key", "x") in hid.calls

        # 2. 关闭原生快捷键：按 X 平滑拖拽至 sell_dropzone
        dispatcher_mouse, hid_mouse, _, _, layout_m, vp_m = _make_env(
            use_native_shortcuts=False
        )
        bench_mapper = LinearZoneMapper(layout_m.bench_zone)
        dispatcher_mouse.process_tick(vp_m, prep, [])
        hid_mouse.clear()
        dispatcher_mouse.process_tick(
            vp_m, prep, [ButtonTap(GamepadButton.A), ButtonTap(GamepadButton.X)]
        )
        bench0_xy = vp_m.norm_to_screen(*bench_mapper.slot_center(0))
        sell_pt = layout_m.buttons["sell_dropzone"]
        sell_xy = vp_m.norm_to_screen(sell_pt.u, sell_pt.v, sell_pt.anchor)
        assert ("smooth_drag", bench0_xy, sell_xy, True) in hid_mouse.calls

        # 3. 联机查看队友仍正常工作，但抓取干员后按 Y 严禁产生任何赠卡拖拽
        hid_mouse.clear()
        dispatcher_mouse.process_tick(
            vp_m,
            prep,
            [
                ChordTriggered(GamepadButton.LT, GamepadButton.DPAD_RIGHT),
                ChordTriggered(GamepadButton.LT, GamepadButton.DPAD_RIGHT),
            ],
        )
        assert dispatcher_mouse.selected_teammate == 2
        mate2_pt = layout_m.buttons["teammate_2"]
        mate2_xy = vp_m.norm_to_screen(mate2_pt.u, mate2_pt.v, mate2_pt.anchor)
        assert ("click", mate2_xy[0], mate2_xy[1], "left") in hid_mouse.calls

        hid_mouse.clear()
        snap_hold_y = dispatcher_mouse.process_tick(
            vp_m, prep, [ButtonTap(GamepadButton.A), ButtonTap(GamepadButton.Y)]
        )
        assert hid_mouse.calls == []
        assert snap_hold_y.deploy_state == DeployState.HOLDING_UNIT
        assert all(p.button != "Y" and "赠" not in p.label for p in snap_hold_y.prompts)

    def test_shop_economy_ready_hold_and_lt_chords(self) -> None:
        """验证商店连按A购买/RT+A查详情/刷新/升本/冻结、Start长按开战及全部 LT 局内组合键。"""
        dispatcher, hid, _, _, layout, vp = _make_env(use_native_shortcuts=True)
        shop_mapper = LinearZoneMapper(layout.shop_zone)
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.98, shop_open=True)
        dispatcher.process_tick(vp, prep, [])
        hid.clear()

        # 商店防误触连按两次 A 触发双击购买；RT+A 单击查商店干员详情后按 B 退出；随后刷新 X(D)、升本 Y(G)、冻结 RS(S)
        dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.A),
                ButtonTap(GamepadButton.A),
                ChordTriggered(GamepadButton.RT, GamepadButton.A),
                ButtonTap(GamepadButton.B),
                ButtonTap(GamepadButton.X),
                ButtonTap(GamepadButton.Y),
                ButtonTap(GamepadButton.RS),
            ],
        )
        shop0_xy = vp.norm_to_screen(*shop_mapper.slot_center(0))
        assert ("double_click", shop0_xy[0], shop0_xy[1]) in hid.calls
        assert ("click", shop0_xy[0], shop0_xy[1], "left") in hid.calls
        assert ("tap_key", "d") in hid.calls
        assert ("tap_key", "g") in hid.calls
        assert ("tap_key", "s") in hid.calls

        # 商店收起态按 X / RS 会自动先按 A 展开商店
        dispatcher_closed, hid_closed, _, _, _, vp_closed = _make_env(
            use_native_shortcuts=True, auto_toggle_shop=True
        )
        prep_closed = SceneDetectionResult(GameScene.BATTLE_PREP, 0.98, shop_open=False)
        dispatcher_closed.process_tick(vp_closed, prep_closed, [])
        hid_closed.clear()
        snap_auto_open = dispatcher_closed.process_tick(
            vp_closed, prep_closed, [ButtonTap(GamepadButton.RS)]
        )
        assert snap_auto_open.shop_open is True
        assert hid_closed.calls == [("tap_key", "a"), ("tap_key", "s")]

        # Start 长按进度条与防误触开战 C
        hid.clear()
        snap_prog = dispatcher.process_tick(
            vp, prep, [ButtonHoldProgress(GamepadButton.START, 0.65)]
        )
        assert snap_prog.ready_hold_progress == pytest.approx(0.65)
        snap_ready = dispatcher.process_tick(
            vp, prep, [ButtonHoldTriggered(GamepadButton.START)]
        )
        assert snap_ready.ready_hold_progress == pytest.approx(0.0)
        assert ("tap_key", "c") in hid.calls

        # 局内 LT 信息速查与联机组合键（LT+DPAD_DOWN 点击左下交流按钮而非发无效 z 键）
        hid.clear()
        dispatcher.process_tick(
            vp,
            prep,
            [
                ChordTriggered(GamepadButton.LT, GamepadButton.DPAD_LEFT),  # 盟约按钮
                ChordTriggered(GamepadButton.LT, GamepadButton.DPAD_UP),  # 敌方路线 (W)
                ChordTriggered(GamepadButton.LT, GamepadButton.DPAD_DOWN),  # 左下交流表情按钮
                ButtonTap(GamepadButton.B),  # 关闭交流表情面板
                ChordTriggered(GamepadButton.LT, GamepadButton.RB),  # 撤退 (Q)
                ChordTriggered(GamepadButton.LT, GamepadButton.START),  # 暂停菜单
            ],
        )
        cov_pt = layout.buttons["covenant_panel"]
        cov_xy = vp.norm_to_screen(cov_pt.u, cov_pt.v, cov_pt.anchor)
        emote_pt = layout.buttons["emote_wheel"]
        emote_xy = vp.norm_to_screen(emote_pt.u, emote_pt.v, emote_pt.anchor)
        pause_pt = layout.buttons["pause_menu"]
        pause_xy = vp.norm_to_screen(pause_pt.u, pause_pt.v, pause_pt.anchor)
        assert ("click", cov_xy[0], cov_xy[1], "left") in hid.calls
        assert ("tap_key", "w") in hid.calls
        assert ("click", emote_xy[0], emote_xy[1], "left") in hid.calls
        assert ("tap_key", "z") not in hid.calls
        assert ("tap_key", "q") in hid.calls
        assert ("click", pause_xy[0], pause_xy[1], "left") in hid.calls

        # 关闭 allow_retreat_chord 时阻止 LT + RB 误撤退
        dispatcher_safe, hid_safe, _, _, _, vp_safe = _make_env(
            allow_retreat_chord=False
        )
        dispatcher_safe.process_tick(vp_safe, prep, [])
        hid_safe.clear()
        dispatcher_safe.process_tick(
            vp_safe, prep, [ChordTriggered(GamepadButton.LT, GamepadButton.RB)]
        )
        assert ("tap_key", "q") not in hid_safe.calls


class TestModalChoiceAndOutOfBattleScenes:
    """验证三选一按住 LT 查看战场与局外全流程（活动主界面、分队准备、编队、结算页）。"""

    def test_modal_choice_selection_and_lt_hold_peek_battlefield(self) -> None:
        """验证三选一选卡、极速双击确认与按住 LT 点击小眼睛隐藏/松开恢复。"""
        dispatcher, hid, _, _, layout, vp = _make_env()
        choice_mapper = LinearZoneMapper(layout.choice_zone)
        modal = SceneDetectionResult(GameScene.MODAL_CHOICE, 0.92, choice_count=3)

        snap = dispatcher.process_tick(vp, modal, [])
        assert snap.focus_zone == FocusZone.CHOICE_CARDS

        # 按住 LT -> 点击 choice_toggle_eye 隐藏弹窗查看战场，此时选卡多边形隐藏且禁止误点卡片
        eye_pt = layout.buttons["choice_toggle_eye"]
        eye_xy = vp.norm_to_screen(eye_pt.u, eye_pt.v, eye_pt.anchor)
        hid.clear()
        snap_peek = dispatcher.process_tick(
            vp,
            modal,
            [
                ModifierStateChanged(GamepadButton.LT, active=True),
                ButtonTap(GamepadButton.A),
            ],
        )
        assert snap_peek.focus_polygon_norm is None
        assert hid.calls == [("click", eye_xy[0], eye_xy[1], "left")]

        # 松开 LT -> 再次点击 choice_toggle_eye 恢复弹窗
        hid.clear()
        snap_restored = dispatcher.process_tick(
            vp, modal, [ModifierStateChanged(GamepadButton.LT, active=False)]
        )
        assert snap_restored.focus_polygon_norm is not None
        assert hid.calls == [("click", eye_xy[0], eye_xy[1], "left")]

        # 左移选第 1 张卡并 RT + A 一键选中卡片并点击底部白色确认条
        hid.clear()
        dispatcher.process_tick(
            vp,
            modal,
            [
                ButtonTap(GamepadButton.DPAD_LEFT),
                ChordTriggered(GamepadButton.RT, GamepadButton.A),
            ],
        )
        card0_xy = vp.norm_to_screen(*layout.resolve_choice_card_center(0, 3))
        confirm0_xy = vp.norm_to_screen(*layout.resolve_choice_confirm_point(0, 3))
        assert hid.calls == [
            ("click", card0_xy[0], card0_xy[1], "left"),
            ("click", confirm0_xy[0], confirm0_xy[1], "left"),
        ]

    def test_out_of_battle_unknown_free_cursor_mode_and_state_retention(self) -> None:
        """验证默认与局外 UNKNOWN_FREE_CURSOR 仅使用模拟自由鼠标，且不清空局内记住的槽数与队友状态。"""
        dispatcher, hid, _, keymap, layout, vp = _make_env(left_stick_mode="dpad")

        # 1. 初始即处于局外自由鼠标态（无需局外菜单节点）
        snap_init = dispatcher.process_tick(vp, None, [])
        assert snap_init.scene == GameScene.UNKNOWN_FREE_CURSOR
        assert snap_init.focus_zone == FocusZone.FREE_CURSOR
        assert snap_init.free_cursor_active is True

        # 2. 在 UNKNOWN_FREE_CURSOR 下，即便 left_stick_mode="dpad"，推左摇杆也始终直接平滑驱动鼠标
        unknown = SceneDetectionResult(GameScene.UNKNOWN_FREE_CURSOR, 0.0)
        hid.clear()
        snap_move = dispatcher.process_tick(
            vp,
            unknown,
            [
                StickMotion("left", 0.85, -0.30, 120.0, 40.0),
                ButtonTap(GamepadButton.A),
                ChordTriggered(GamepadButton.RT, GamepadButton.A),
                ButtonTap(GamepadButton.B),
                StickMotion("right", 0.0, -0.8, 0.0, 48.0),
            ],
        )
        assert snap_move.free_cursor_active is True
        assert hid.calls[0][0] == "move_cursor"
        assert hid.calls[1] == ("click", hid.calls[0][1], hid.calls[0][2], "left")
        assert hid.calls[2] == ("double_click", hid.calls[0][1], hid.calls[0][2])
        assert ("tap_key", "esc") in hid.calls
        assert ("scroll", -240, False) in hid.calls

        # 3. 局内进入 6 槽联机休整期后再短暂回到 UNKNOWN_FREE_CURSOR（如打开覆盖页），保留槽数与队友状态
        prep6_coop = SceneDetectionResult(
            GameScene.BATTLE_PREP,
            0.96,
            shop_open=True,
            shop_slot_count=6,
            has_teammates=True,
        )
        dispatcher.process_tick(vp, prep6_coop, [])
        assert dispatcher.shop_slot_count == 6
        assert dispatcher.has_teammates is True

        dispatcher.process_tick(vp, unknown, [])
        assert dispatcher.current_scene == GameScene.UNKNOWN_FREE_CURSOR
        assert dispatcher.shop_slot_count == 6
        assert dispatcher.has_teammates is True

        # 4. 热重载配置不报错且保持状态健康
        dispatcher.update_configs(keymap, layout)
        snap_reloaded = dispatcher.process_tick(vp, unknown, [])
        assert snap_reloaded.visible is True

    def test_combat_scene_cross_zones_and_extra_shortcuts(self) -> None:
        """补充验证作战期交互、全方向跨区跳格、预选朝向按 A 确认与 RT+Start 快捷开战。"""
        dispatcher, hid, _, keymap, layout, vp = _make_env()

        # 1. 作战期（BATTLE_COMBAT）：棋盘浏览、点击、右键关闭、LT+下 打开/B关闭表情面板与 LT+X 直达 3 号队友
        combat = SceneDetectionResult(GameScene.BATTLE_COMBAT, 0.94)
        snap_c = dispatcher.process_tick(
            vp,
            combat,
            [
                ButtonTap(GamepadButton.DPAD_LEFT),
                ButtonTap(GamepadButton.A),
                ButtonTap(GamepadButton.B),
                ChordTriggered(GamepadButton.LT, GamepadButton.DPAD_DOWN),
                ButtonTap(GamepadButton.B),
                ChordTriggered(GamepadButton.LT, GamepadButton.X),
            ],
        )
        assert snap_c.scene == GameScene.BATTLE_COMBAT
        assert snap_c.focus_zone == FocusZone.BOARD
        assert dispatcher.selected_teammate == 3
        emote_pt = layout.buttons["emote_wheel"]
        emote_xy = vp.norm_to_screen(emote_pt.u, emote_pt.v, emote_pt.anchor)
        assert ("click", emote_xy[0], emote_xy[1], "left") in hid.calls
        assert ("tap_key", "z") not in hid.calls
        mate3_pt = layout.buttons["teammate_3"]
        mate3_xy = vp.norm_to_screen(mate3_pt.u, mate3_pt.v, mate3_pt.anchor)
        assert ("click", mate3_xy[0], mate3_xy[1], "left") in hid.calls

        # 2. 休整期十字键跨区链路（实机底部商店+底部整备+棋盘下方横向装备栏）：
        #    SHOP(上) -> BENCH(上) -> EQUIPMENT(上) -> BOARD(自动收起商店) -> 下出界 -> EQUIPMENT(下) -> BENCH(下) -> SHOP(自动展开商店)
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.96, shop_open=True)
        snap_p = dispatcher.process_tick(vp, prep, [])
        assert snap_p.focus_zone == FocusZone.SHOP

        snap_bench_up = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.DPAD_UP)])
        assert snap_bench_up.focus_zone == FocusZone.BENCH
        assert snap_bench_up.shop_open is True  # 切至整备区时仍保持商店展开并套用缩放补偿

        snap_eq_up = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.DPAD_UP)])
        assert snap_eq_up.focus_zone == FocusZone.EQUIPMENT

        snap_board = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.DPAD_UP)])
        assert snap_board.focus_zone == FocusZone.BOARD
        assert snap_board.shop_open is False  # 切入棋盘自动收起商店

        # 棋盘最底行向下跨入横向装备栏 -> 再向下进入整备区 -> 再向下展开商店
        snap_eq_down = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.DPAD_DOWN)])
        assert snap_eq_down.focus_zone == FocusZone.EQUIPMENT
        snap_bench = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.DPAD_DOWN)])
        assert snap_bench.focus_zone == FocusZone.BENCH
        snap_shop = dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.DPAD_DOWN)]
        )
        assert snap_shop.focus_zone == FocusZone.SHOP
        assert snap_shop.shop_open is True  # 切入商店自动展开商店

        # 3. 装备穿戴至整备区干员 + 预选朝向按 A 确认 + RT+Start 快捷开战
        hid.clear()
        dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.LB),  # -> EQUIPMENT
                ButtonTap(GamepadButton.A),  # 抓装备（自动收起商店恢复标准视角）
                ButtonTap(GamepadButton.RB),  # -> BOARD
                ButtonTap(GamepadButton.RB),  # -> BENCH
                ButtonTap(GamepadButton.A),  # 给整备区干员穿戴装备
            ],
        )
        assert any(call[0] == "smooth_drag" for call in hid.calls)
        assert dispatcher.deploy_fsm.state == DeployState.IDLE

        # 预选朝向后按 A 确认落位
        hid.clear()
        dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.A),  # 抓整备区干员
                ButtonTap(GamepadButton.RB),  # -> BOARD
                ButtonTap(GamepadButton.A),  # 落位进入定朝向
                StickMotion("right", 0.0, 0.40, 0.0, -10.0),  # 仅预选“上”
                ButtonTap(GamepadButton.A),  # 按 A 确认“上”朝向
            ],
        )
        expected_dist = int(round(vp.active_width * keymap.flick_distance_ratio))
        assert any(
            call[0] == "directional_flick"
            and call[2] == "up"
            and call[3] == expected_dist
            for call in hid.calls
        )
        assert dispatcher.deploy_fsm.state == DeployState.IDLE

        # RT + Start 快捷开战（官方快捷键 C）
        hid.clear()
        dispatcher.process_tick(
            vp, prep, [ChordTriggered(GamepadButton.RT, GamepadButton.START)]
        )
        assert ("tap_key", "c") in hid.calls


class TestLeftStickNavShopToggleAndZoomCompensation:
    """验证左摇杆替代十字键、商店自动/手动开合、三态镜头缩放补偿与横向装备栏导航。"""

    def test_left_stick_replaces_dpad_across_scenes_and_ls_toggles_cursor(self) -> None:
        dispatcher, hid, _, keymap, layout, vp = _make_env(left_stick_mode="dpad")
        ge = GestureEngine(keymap)
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.96, shop_open=False)
        dispatcher.process_tick(vp, prep, [])
        hid.clear()

        # 局内休整期：左摇杆向右推动经 GestureEngine 生成 StickDirection 直接整备区跳格
        st_right = GamepadState(
            connected=True,
            packet_number=1,
            buttons=frozenset(),
            lx=0.85,
            ly=0.0,
            rx=0.0,
            ry=0.0,
            lt=0.0,
            rt=0.0,
        )
        ev1 = ge.update(st_right, dt=0.016)
        assert any(isinstance(e, StickDirection) and e.direction == "right" for e in ev1)
        snap_b1 = dispatcher.process_tick(vp, prep, ev1)
        assert snap_b1.free_cursor_active is False
        assert snap_b1.focus_zone == FocusZone.BENCH
        assert snap_b1.focus_label == "整备区 [2/10]"

        snap_b = dispatcher.process_tick(
            vp,
            prep,
            [StickDirection("left", "right")],
        )
        assert snap_b.focus_zone == FocusZone.BENCH
        assert snap_b.focus_label == "整备区 [3/10]"

        # 按 LS 切自由光标再按 LS 恢复吸附
        snap_fc = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.LS)])
        assert snap_fc.free_cursor_active is True
        snap_snap = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.LS)])
        assert snap_snap.free_cursor_active is False

    def test_shop_auto_toggle_manual_toggle_and_zoom_view_state(self) -> None:
        dispatcher, hid, _, _, layout, vp = _make_env(
            use_native_shortcuts=True, auto_toggle_shop=True
        )
        toggle_pt = layout.buttons["shop_toggle"]
        toggle_xy = vp.norm_to_screen(toggle_pt.u, toggle_pt.v, toggle_pt.anchor)

        # 1. 进入休整期（商店关闭 -> NORMAL 视图态）
        prep_closed = SceneDetectionResult(GameScene.BATTLE_PREP, 0.95, shop_open=False)
        snap0 = dispatcher.process_tick(vp, prep_closed, [])
        assert snap0.focus_zone == FocusZone.BENCH
        assert snap0.shop_open is False
        assert snap0.view_state == LayoutViewState.NORMAL.value

        # 2. 按 LB 切入 SHOP：自动触发官方快捷键 A 展开商店并切到 SHOP_OPEN 缩放视图态
        hid.clear()
        snap_shop = dispatcher.process_tick(vp, prep_closed, [ButtonTap(GamepadButton.LB)])
        assert snap_shop.focus_zone == FocusZone.SHOP
        assert snap_shop.shop_open is True
        assert snap_shop.view_state == LayoutViewState.SHOP_OPEN.value
        assert ("tap_key", "a") in hid.calls

        # 3. 此时即便视觉旧帧仍报 shop_open=False，防抖保护窗内也不应回退覆盖
        snap_hold_open = dispatcher.process_tick(vp, prep_closed, [])
        assert snap_hold_open.shop_open is True
        assert snap_hold_open.view_state == LayoutViewState.SHOP_OPEN.value

        # 4. 在商店区按 B 手动收起商店并退回整备区，视图态回到 NORMAL
        hid.clear()
        snap_b = dispatcher.process_tick(vp, prep_closed, [ButtonTap(GamepadButton.B)])
        assert snap_b.shop_open is False
        assert snap_b.focus_zone == FocusZone.BENCH
        assert snap_b.view_state == LayoutViewState.NORMAL.value
        assert ("tap_key", "a") in hid.calls

        # 5. LT + LB 独立手动展开商店
        hid.clear()
        snap_lt_open = dispatcher.process_tick(
            vp, prep_closed, [ChordTriggered(GamepadButton.LT, GamepadButton.LB)]
        )
        assert snap_lt_open.shop_open is True
        assert snap_lt_open.focus_zone == FocusZone.SHOP
        assert snap_lt_open.view_state == LayoutViewState.SHOP_OPEN.value

        # 切到棋盘时自动收起商店恢复标准棋盘坐标（SHOP -> BENCH -> BOARD）
        hid.clear()
        snap_board = dispatcher.process_tick(
            vp,
            prep_closed,
            [ButtonTap(GamepadButton.RB), ButtonTap(GamepadButton.RB)],
        )
        assert snap_board.focus_zone == FocusZone.BOARD
        assert snap_board.shop_open is False
        assert snap_board.view_state == LayoutViewState.NORMAL.value
        assert ("tap_key", "a") in hid.calls

        # 6. 关闭原生快捷键时自动回退为左键点击 shop_toggle 按钮
        dispatcher_btn, hid_btn, _, _, _, vp_btn = _make_env(
            use_native_shortcuts=False, auto_toggle_shop=True
        )
        dispatcher_btn.process_tick(vp_btn, prep_closed, [])
        hid_btn.clear()
        snap_btn = dispatcher_btn.process_tick(
            vp_btn, prep_closed, [ButtonTap(GamepadButton.LB)]
        )
        assert snap_btn.shop_open is True
        assert ("click", toggle_xy[0], toggle_xy[1], "left") in hid_btn.calls

    def test_unit_selected_zoom_compensation_and_clean_drag(self) -> None:
        dispatcher, hid, _, _, layout, vp = _make_env()
        normal_board = PerspectiveBoardMapper(layout.resolve_board_quad(LayoutViewState.NORMAL))
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.96, shop_open=False)
        dispatcher.process_tick(vp, prep, [])

        # 切到棋盘中心格：先确认处于 NORMAL 坐标
        snap_norm = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.RB)])
        assert snap_norm.focus_zone == FocusZone.BOARD
        assert snap_norm.view_state == LayoutViewState.NORMAL.value
        assert snap_norm.cursor_norm == pytest.approx(
            normal_board.cell_center(normal_board.rows // 2, normal_board.cols // 2)
        )

        # 按 RT+A 单击查看角色详情 -> 自动切入 UNIT_SELECTED 专属详情模式，吸附左侧第 1 栏「特质」
        hid.clear()
        snap_sel = dispatcher.process_tick(
            vp, prep, [ChordTriggered(GamepadButton.RT, GamepadButton.A)]
        )
        assert snap_sel.view_state == LayoutViewState.UNIT_SELECTED.value
        assert snap_sel.cursor_norm == pytest.approx((0.045, 0.515))
        assert "特质" in snap_sel.focus_label
        assert hid.calls[0][0] == "click"

        # 详情态下右移切至第 2 栏「技能」并点击
        hid.clear()
        snap_tab2 = dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.DPAD_RIGHT)]
        )
        assert snap_tab2.cursor_norm == pytest.approx((0.135, 0.515))
        assert "技能" in snap_tab2.focus_label
        assert hid.calls == [("click", *vp.norm_to_screen(0.135, 0.515), "left")]

        # 按 B 左键点击安全空白区退出角色详情，立即恢复 NORMAL 棋盘坐标
        hid.clear()
        snap_back = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.B)])
        safe_pt = layout.buttons["safe_blank"]
        safe_xy = vp.norm_to_screen(safe_pt.u, safe_pt.v, safe_pt.anchor)
        assert snap_back.view_state == LayoutViewState.NORMAL.value
        assert snap_back.cursor_norm == pytest.approx(snap_norm.cursor_norm)
        assert hid.calls == [("click", safe_xy[0], safe_xy[1], "left")]

        # 详情态下按 X 直接触发官方 X 键出售并退出，按 Y 触发官方 Q 键撤回并退出
        dispatcher.process_tick(
            vp, prep, [ChordTriggered(GamepadButton.RT, GamepadButton.A)]
        )
        hid.clear()
        snap_sold = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.X)])
        assert snap_sold.view_state == LayoutViewState.NORMAL.value
        assert hid.calls == [("tap_key", "x")]

        dispatcher.process_tick(
            vp, prep, [ChordTriggered(GamepadButton.RT, GamepadButton.A)]
        )
        hid.clear()
        snap_retreat = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.Y)])
        assert snap_retreat.view_state == LayoutViewState.NORMAL.value
        assert hid.calls == [("tap_key", "q")]

    def test_horizontal_equip_zone_navigation_matches_bottom_layout(self) -> None:
        """验证底部横向装备栏与底部商店布局（贴合实机 user_override 结构）的跨区导航。"""
        dispatcher, hid, _, keymap, layout, vp = _make_env(auto_toggle_shop=False)
        custom_layout = replace(
            layout,
            shop_zone=LinearZone("shop", 5, 0.28, 0.845, 0.79, 0.845, 0.11, 0.19),
            bench_zone=LinearZone("bench", 10, 0.115, 0.845, 0.82, 0.845, 0.072, 0.135),
            equip_zone=LinearZone("equipment", 5, 0.425, 0.715, 0.717, 0.715, 0.068, 0.088),
        )
        dispatcher.update_configs(keymap, custom_layout)
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.95, shop_open=False)
        dispatcher.process_tick(vp, prep, [])

        # 空闲态：整备区向上 -> 横向装备栏 -> 向上 -> 棋盘底行 -> 向下 -> 横向装备栏 -> 向下 -> 整备区
        snap1 = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.DPAD_UP)])
        assert snap1.focus_zone == FocusZone.EQUIPMENT
        snap2 = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.DPAD_UP)])
        assert snap2.focus_zone == FocusZone.BOARD
        snap3 = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.DPAD_DOWN)])
        assert snap3.focus_zone == FocusZone.EQUIPMENT
        snap4 = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.DPAD_DOWN)])
        assert snap4.focus_zone == FocusZone.BENCH

        # 抓取干员态：从整备区向上直接跳过装备栏直达棋盘，方便一步落位
        snap_hold = dispatcher.process_tick(
            vp,
            prep,
            [ButtonTap(GamepadButton.A), ButtonTap(GamepadButton.DPAD_UP)],
        )
        assert snap_hold.deploy_state == DeployState.HOLDING_UNIT
        assert snap_hold.focus_zone == FocusZone.BOARD

    def test_dynamic_shop_slot_expansion_cursor_stability_and_upgrade_button(self) -> None:
        """验证商店升至二本(5槽)/四本(6槽)向左延伸时焦点顺移保稳、原生 G 键/纯鼠标升本与自由层记忆保留。"""
        # 1. 原生快捷键开启：按 Y 发送官方升本快捷键 G（绝不误触商品卡）
        dispatcher_key, hid_key, _, _, _, vp_key = _make_env(use_native_shortcuts=True)
        prep5_key = SceneDetectionResult(
            GameScene.BATTLE_PREP, 0.96, shop_open=True, shop_slot_count=5
        )
        dispatcher_key.process_tick(vp_key, prep5_key, [])
        hid_key.clear()
        dispatcher_key.process_tick(vp_key, prep5_key, [ButtonTap(GamepadButton.Y)])
        assert hid_key.calls == [("tap_key", "g")]

        # 2. 纯鼠标模式（use_native_shortcuts=False）：点击随栏位数同步左移的升级卡坐标
        dispatcher, hid, _, _, layout, vp = _make_env(use_native_shortcuts=False)

        # 1 本 4 槽：进入商店并右移到索引 1（第 2 张卡）
        prep4 = SceneDetectionResult(
            GameScene.BATTLE_PREP, 0.96, shop_open=True, shop_slot_count=4
        )
        dispatcher.process_tick(vp, prep4, [])
        snap4 = dispatcher.process_tick(vp, prep4, [ButtonTap(GamepadButton.DPAD_RIGHT)])
        assert snap4.focus_zone == FocusZone.SHOP
        assert snap4.focus_label == "商店 [2/4]"
        card_before = snap4.cursor_norm

        # 升至二本（5 槽向左侧延伸）：焦点顺移至索引 2，光标牢牢咬住同一张卡不乱跳
        prep5 = SceneDetectionResult(
            GameScene.BATTLE_PREP, 0.96, shop_open=True, shop_slot_count=5
        )
        snap5 = dispatcher.process_tick(vp, prep5, [])
        assert dispatcher.shop_slot_count == 5
        assert snap5.focus_label == "商店 [3/5]"
        assert snap5.cursor_norm == pytest.approx(card_before)

        # 左移两步可直达二本新增的最左商品卡（槽 1/5）
        snap5_left = dispatcher.process_tick(
            vp,
            prep5,
            [ButtonTap(GamepadButton.DPAD_LEFT), ButtonTap(GamepadButton.DPAD_LEFT)],
        )
        z5 = layout.resolve_shop_zone(LayoutViewState.SHOP_OPEN, slot_count=5)
        assert snap5_left.focus_label == "商店 [1/5]"
        assert snap5_left.cursor_norm == pytest.approx(z5.slot_center(0))

        # 纯鼠标模式下按 Y 升级商店：应点击同步左移后的升级卡坐标，绝不误点新商品卡
        hid.clear()
        dispatcher.process_tick(vp, prep5, [ButtonTap(GamepadButton.Y)])
        upg5 = layout.resolve_shop_upgrade_point(LayoutViewState.SHOP_OPEN, slot_count=5)
        assert upg5 is not None
        assert hid.calls == [("click", *vp.norm_to_screen(upg5.u, upg5.v, upg5.anchor), "left")]

        # 升至四本（6 槽再向左侧延伸）：按 Y 升级再左移一格，且收起后保留 6 槽记忆
        prep6 = SceneDetectionResult(
            GameScene.BATTLE_PREP, 0.96, shop_open=True, shop_slot_count=6
        )
        snap6 = dispatcher.process_tick(vp, prep6, [])
        assert dispatcher.shop_slot_count == 6
        assert snap6.focus_label == "商店 [2/6]"
        hid.clear()
        dispatcher.process_tick(vp, prep6, [ButtonTap(GamepadButton.Y)])
        upg6 = layout.resolve_shop_upgrade_point(LayoutViewState.SHOP_OPEN, slot_count=6)
        assert upg6 is not None
        assert hid.calls == [("click", *vp.norm_to_screen(upg6.u, upg6.v, upg6.anchor), "left")]

        # 商店收起（shop_slot_count=None）期间保持已识别的 6 槽状态不回退
        prep_closed = SceneDetectionResult(
            GameScene.BATTLE_PREP, 0.96, shop_open=False, shop_slot_count=None
        )
        dispatcher.process_tick(vp, prep_closed, [ButtonTap(GamepadButton.B)])
        assert dispatcher.shop_slot_count == 6

        # 短暂退回 UNKNOWN_FREE_CURSOR 时不清空记住的 6 槽状态
        dispatcher.process_tick(vp, SceneDetectionResult(GameScene.UNKNOWN_FREE_CURSOR, 0.0), [])
        assert dispatcher.shop_slot_count == 6


class TestModal6ChoiceEquipReorderCovenantAndTeammates:
    """专项覆盖：三/六选一两段白条确认、装备区内部挪动、右侧安全空白、顶部放大镜/对称羁绊与左侧队友栏。"""

    def test_modal_3_and_6_choice_two_step_white_bar_confirm_and_2d_nav(self) -> None:
        dispatcher, hid, _, _, layout, vp = _make_env()

        # 1. 三选一两段式确认：首次按 A 点击卡片中心并聚焦底部白条，二次按 A 点击底部白色确认条
        modal3 = SceneDetectionResult(GameScene.MODAL_CHOICE, 0.93, choice_count=3)
        snap0 = dispatcher.process_tick(vp, modal3, [])
        assert snap0.focus_label == "三选一 [2/3]"
        card1_xy = vp.norm_to_screen(*layout.resolve_choice_card_center(1, 3))
        confirm1_xy = vp.norm_to_screen(*layout.resolve_choice_confirm_point(1, 3))

        hid.clear()
        snap_sel = dispatcher.process_tick(vp, modal3, [ButtonTap(GamepadButton.A)])
        assert dispatcher.selected_choice_index == 1
        assert "待确认" in snap_sel.focus_label
        assert snap_sel.cursor_norm == pytest.approx(
            layout.resolve_choice_confirm_point(1, 3)
        )
        assert hid.calls == [("click", card1_xy[0], card1_xy[1], "left")]

        # 中途改主意右移到第 3 张卡按 A：先选中第 3 张卡主体，再按 A 才点第 3 张卡白条
        card2_xy = vp.norm_to_screen(*layout.resolve_choice_card_center(2, 3))
        confirm2_xy = vp.norm_to_screen(*layout.resolve_choice_confirm_point(2, 3))
        hid.clear()
        dispatcher.process_tick(
            vp,
            modal3,
            [
                ButtonTap(GamepadButton.DPAD_RIGHT),
                ButtonTap(GamepadButton.A),
                ButtonTap(GamepadButton.A),
            ],
        )
        assert hid.calls == [
            ("click", card2_xy[0], card2_xy[1], "left"),
            ("click", confirm2_xy[0], confirm2_xy[1], "left"),
        ]
        assert dispatcher.selected_choice_index is None

        # 2. 多人六选一（2x3 网格）：验证同行左右循环、上下跨排切换、按住 LT 防切场景与预选中直点白条
        dispatcher2, hid2, _, _, _, _ = _make_env()
        modal6 = SceneDetectionResult(
            GameScene.MODAL_CHOICE,
            0.94,
            choice_count=6,
            selected_choice_index=4,
            has_teammates=True,
        )
        snap6 = dispatcher2.process_tick(vp, modal6, [])
        assert snap6.focus_label == "六选一 [5/6] · 待确认"
        assert dispatcher2.has_teammates is True

        # 已处于视觉识别出的待确认态（idx=4 下排中卡），直接按 A 一步点击底部白色确认条
        confirm4_xy = vp.norm_to_screen(*layout.resolve_choice_confirm_point(4, 6))
        hid2.clear()
        dispatcher2.process_tick(vp, modal6, [ButtonTap(GamepadButton.A)])
        assert hid2.calls == [("click", confirm4_xy[0], confirm4_xy[1], "left")]

        # 下排左移 -> idx=3(下排左)，上移 -> idx=0(上排左)，左移同行循环 -> idx=2(上排右)
        s_nav = dispatcher2.process_tick(
            vp,
            SceneDetectionResult(GameScene.MODAL_CHOICE, 0.94, choice_count=6),
            [
                ButtonTap(GamepadButton.DPAD_LEFT),
                ButtonTap(GamepadButton.DPAD_UP),
                ButtonTap(GamepadButton.DPAD_LEFT),
            ],
        )
        assert s_nav.focus_label == "六选一 [3/6]"

        # 六选一按住 LT 点击右侧收起按钮 choice_toggle_eye_6，且期间即使视觉报 BATTLE_PREP 也不丢层
        eye6 = layout.buttons["choice_toggle_eye_6"]
        eye6_xy = vp.norm_to_screen(eye6.u, eye6.v, eye6.anchor)
        hid2.clear()
        dispatcher2.process_tick(
            vp,
            modal6,
            [ModifierStateChanged(GamepadButton.LT, active=True)],
        )
        assert hid2.calls == [("click", eye6_xy[0], eye6_xy[1], "left")]
        snap_hold_lt = dispatcher2.process_tick(
            vp, SceneDetectionResult(GameScene.BATTLE_PREP, 0.90), []
        )
        assert snap_hold_lt.scene == GameScene.MODAL_CHOICE
        hid2.clear()
        dispatcher2.process_tick(
            vp, modal6, [ModifierStateChanged(GamepadButton.LT, active=False)]
        )
        assert hid2.calls == [("click", eye6_xy[0], eye6_xy[1], "left")]

    def test_equipment_zone_internal_reorder_and_same_slot_cancel(self) -> None:
        dispatcher, hid, _, _, layout, vp = _make_env()
        equip_mapper = LinearZoneMapper(layout.equip_zone)
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.96, shop_open=False)
        dispatcher.process_tick(vp, prep, [])

        # 从整备区上移至装备栏 0 号槽，按 A 抓取 -> 右移 3 格至 3 号槽 -> 按 A 平滑拖拽挪动装备
        snap_eq = dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.DPAD_UP),
                ButtonTap(GamepadButton.A),
                ButtonTap(GamepadButton.DPAD_RIGHT),
                ButtonTap(GamepadButton.DPAD_RIGHT),
                ButtonTap(GamepadButton.DPAD_RIGHT),
                ButtonTap(GamepadButton.A),
            ],
        )
        eq0_xy = vp.norm_to_screen(*equip_mapper.slot_center(0))
        eq3_xy = vp.norm_to_screen(*equip_mapper.slot_center(3))
        assert snap_eq.focus_zone == FocusZone.EQUIPMENT
        assert snap_eq.deploy_state == DeployState.IDLE
        assert hid.calls == [("smooth_drag", eq0_xy, eq3_xy, True)]

        # 同槽位二次按 A 取消选中，不产生多余拖拽；且支持抓取装备后上棋盘再下移回装备栏换位
        hid.clear()
        dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.A), ButtonTap(GamepadButton.A)]
        )
        assert hid.calls == []
        assert dispatcher.deploy_fsm.state == DeployState.IDLE

        hid.clear()
        dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.A),          # 抓 3 号槽装备
                ButtonTap(GamepadButton.DPAD_UP),    # 上棋盘
                ButtonTap(GamepadButton.DPAD_DOWN),  # 退回装备栏
                ButtonTap(GamepadButton.DPAD_LEFT),  # 移至邻槽
                ButtonTap(GamepadButton.A),          # 挪动装备
            ],
        )
        assert len(hid.calls) == 1 and hid.calls[0][0] == "smooth_drag"
        assert dispatcher.deploy_fsm.state == DeployState.IDLE

    def test_safe_blank_never_overlaps_shop_cards(self) -> None:
        dispatcher, hid, _, _, layout, vp = _make_env()
        safe_pt = layout.buttons["safe_blank"]
        shop6 = layout.resolve_shop_zone(LayoutViewState.SHOP_OPEN, slot_count=6)

        # 验证 safe_blank 位于左上角退出按钮右侧延迟区，与 6 槽全展开商店、4 人队友抽屉及右侧策略头像彻底隔离
        shop_top_edge = shop6.start_v - shop6.slot_h * 0.5
        assert safe_pt.anchor == "left_edge"
        assert safe_pt.u == pytest.approx(0.1172, abs=1e-4)
        assert safe_pt.v == pytest.approx(0.0903, abs=1e-4)
        assert safe_pt.v < shop_top_edge

        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.96, shop_open=False)
        dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.RB),
                ChordTriggered(GamepadButton.RT, GamepadButton.A),
            ],
        )
        hid.clear()
        dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.B)])
        assert hid.calls == [
            ("click", *vp.norm_to_screen(safe_pt.u, safe_pt.v, safe_pt.anchor), "left")
        ]

    def test_top_magnifiers_symmetric_covenants_and_multiplayer_teammates(self) -> None:
        dispatcher, hid, _, _, layout, vp = _make_env()
        prep_coop = SceneDetectionResult(
            GameScene.BATTLE_PREP,
            0.96,
            shop_open=False,
            covenant_count=2,
            has_teammates=True,
        )
        dispatcher.process_tick(vp, prep_coop, [])
        assert dispatcher.covenant_count == 2
        assert dispatcher.has_teammates is True

        # 1. 从整备区 -> 装备栏 -> 棋盘一路向上跨入 TOP_INFO（顶部放大镜与居中对称羁绊栏）
        snap = dispatcher.process_tick(
            vp,
            prep_coop,
            [
                ButtonTap(GamepadButton.DPAD_UP),  # -> EQUIPMENT
                ButtonTap(GamepadButton.DPAD_UP),  # -> BOARD row 3
                ButtonTap(GamepadButton.DPAD_UP),  # row 2
                ButtonTap(GamepadButton.DPAD_UP),  # row 1
                ButtonTap(GamepadButton.DPAD_UP),  # row 0
                ButtonTap(GamepadButton.DPAD_UP),  # -> TOP_INFO
            ],
        )
        assert snap.focus_zone == FocusZone.TOP_INFO

        # 切到第 0 项「查看本局信息」（左放大镜），按 A 打开，期间视觉暂报 UNKNOWN 不丢层，按 B 点右上 X 关闭
        while "查看本局信息" not in snap.focus_label:
            snap = dispatcher.process_tick(
                vp, prep_coop, [ButtonTap(GamepadButton.DPAD_LEFT)]
            )
        hid.clear()
        dispatcher.process_tick(vp, prep_coop, [ButtonTap(GamepadButton.A)])
        left_mag = layout.buttons["covenant_panel"]
        assert hid.calls == [
            ("click", *vp.norm_to_screen(left_mag.u, left_mag.v, left_mag.anchor), "left")
        ]
        # 面板打开期间视觉返回 UNKNOWN_FREE_CURSOR 仍保持 BATTLE_PREP 上下文
        snap_keep = dispatcher.process_tick(
            vp, SceneDetectionResult(GameScene.UNKNOWN_FREE_CURSOR, 0.0), []
        )
        assert snap_keep.scene == GameScene.BATTLE_PREP
        hid.clear()
        dispatcher.process_tick(vp, prep_coop, [ButtonTap(GamepadButton.B)])
        close_pt = layout.buttons["covenant_close"]
        assert hid.calls == [
            ("click", *vp.norm_to_screen(close_pt.u, close_pt.v, close_pt.anchor), "left")
        ]

        # 右移依次点击偶数居中对称的 2 个羁绊图标与最右侧「查看本轮出怪」放大镜
        hid.clear()
        s_cov1 = dispatcher.process_tick(
            vp,
            prep_coop,
            [ButtonTap(GamepadButton.DPAD_RIGHT), ButtonTap(GamepadButton.A)],
        )
        s_cov2 = dispatcher.process_tick(
            vp,
            prep_coop,
            [ButtonTap(GamepadButton.DPAD_RIGHT), ButtonTap(GamepadButton.A)],
        )
        assert "羁绊 [1/2]" in s_cov1.focus_label
        assert "羁绊 [2/2]" in s_cov2.focus_label
        c0_xy = vp.norm_to_screen(*layout.covenant_zone.slot_center(0, count=2))
        c1_xy = vp.norm_to_screen(*layout.covenant_zone.slot_center(1, count=2))
        assert hid.calls == [
            ("click", c0_xy[0], c0_xy[1], "left"),
            ("click", c1_xy[0], c1_xy[1], "left"),
        ]

        # 再右移到「查看本轮出怪」，按 A 触发 W，按 B 再次触发 W 关闭预览
        hid.clear()
        s_wave = dispatcher.process_tick(
            vp,
            prep_coop,
            [
                ButtonTap(GamepadButton.DPAD_RIGHT),
                ButtonTap(GamepadButton.A),
                ButtonTap(GamepadButton.B),
            ],
        )
        assert "查看本轮出怪" in s_wave.focus_label
        assert hid.calls == [("tap_key", "w"), ("tap_key", "w")]

        # 2. 下移回棋盘，并从棋盘最左列继续按左跨入左侧联机队友头像栏 TEAMMATES
        s_board = dispatcher.process_tick(
            vp, prep_coop, [ButtonTap(GamepadButton.DPAD_DOWN)]
        )
        assert s_board.focus_zone == FocusZone.BOARD
        while s_board.focus_zone != FocusZone.TEAMMATES:
            s_board = dispatcher.process_tick(
                vp, prep_coop, [ButtonTap(GamepadButton.DPAD_LEFT)]
            )
        assert s_board.focus_zone == FocusZone.TEAMMATES

        # 下移选中下一位队友并按 A 点击左侧队友头像查看，再按右返回棋盘第 1 列
        hid.clear()
        s_mate = dispatcher.process_tick(
            vp,
            prep_coop,
            [ButtonTap(GamepadButton.DPAD_DOWN), ButtonTap(GamepadButton.A)],
        )
        mate_num = dispatcher.selected_teammate
        mate_pt = layout.buttons[f"teammate_{mate_num}"]
        assert hid.calls == [
            ("click", *vp.norm_to_screen(mate_pt.u, mate_pt.v, mate_pt.anchor), "left")
        ]
        assert f"[{mate_num}/4]" in s_mate.focus_label

        s_back_board = dispatcher.process_tick(
            vp, prep_coop, [ButtonTap(GamepadButton.DPAD_RIGHT)]
        )
        assert s_back_board.focus_zone == FocusZone.BOARD


class TestPhase2InplaceTurnConfirmStrategyAndLockSync:
    """阶段二专项验证：棋盘原地回环转向、回合确认弹窗、右侧策略头像与手动锁层槽数同步。"""

    def test_board_operator_inplace_loopback_reorient_and_fallback(self) -> None:
        """需求 1：棋盘干员同格二次按 A 执行回环拖拽进入 AIMING_DIRECTION 完成原地转向，右列向左偏移防越界。"""
        dispatcher, hid, rumbles, keymap, layout, vp = _make_env()
        board_mapper = PerspectiveBoardMapper(layout.board_quad)
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.96, shop_open=False)

        # 切到棋盘中心格（u <= 0.65 -> 正向偏移 dx > 0）
        dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.RB)])
        center_cell = (board_mapper.rows // 2, board_mapper.cols // 2)
        center_norm = board_mapper.cell_center(*center_cell)
        center_xy = vp.norm_to_screen(*center_norm)
        expected_offset = max(36, int(round(vp.active_width * 0.035)))
        expected_dx = -expected_offset if center_norm[0] > 0.65 else expected_offset

        hid.clear()
        snap_hold = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.A)])
        assert snap_hold.deploy_state == DeployState.HOLDING_UNIT
        assert any(p.label == "原地转向" for p in snap_hold.prompts)

        # 在同一棋盘格再按 A：不再 reset，而是调用 smooth_drag_loopback 并进入 AIMING_DIRECTION
        snap_aim = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.A)])
        assert snap_aim.deploy_state == DeployState.AIMING_DIRECTION
        assert snap_aim.aim_origin_norm == pytest.approx(center_norm)
        assert hid.calls == [
            ("smooth_drag_loopback", center_xy, (expected_dx, 0), True)
        ]

        # 推摇杆向上确认朝向
        hid.clear()
        snap_done = dispatcher.process_tick(
            vp, prep, [StickMotion("left", 0.0, 0.90, 0.0, -30.0)]
        )
        expected_flick = max(48, int(round(vp.active_width * keymap.flick_distance_ratio)))
        assert snap_done.deploy_state == DeployState.IDLE
        assert hid.calls == [("directional_flick", center_xy, "up", expected_flick)]

        # 最右列干员（u > 0.65）原地转向：应向左偏移 dx < 0 防止拖出棋盘右边界
        steps_to_rightmost = (board_mapper.cols - 1) - (board_mapper.cols // 2)
        snap_rightmost = dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.DPAD_RIGHT)] * steps_to_rightmost
        )
        assert snap_rightmost.focus_zone == FocusZone.BOARD
        right_cell = (board_mapper.rows // 2, board_mapper.cols - 1)
        right_norm = board_mapper.cell_center(*right_cell)
        assert right_norm[0] > 0.65
        right_xy = vp.norm_to_screen(*right_norm)
        hid.clear()
        dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.A), ButtonTap(GamepadButton.A)]
        )
        assert hid.calls == [
            ("smooth_drag_loopback", right_xy, (-expected_offset, 0), True)
        ]

        # 取消刚唤出的定朝向态，再验证按 B 可在 HOLDING_UNIT 态无点击取消原地抓取
        dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.B)])
        hid.clear()
        dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.A), ButtonTap(GamepadButton.B)]
        )
        assert hid.calls == []
        assert dispatcher.deploy_fsm.state == DeployState.IDLE

        # 缺少 smooth_drag_loopback 的简化 HID 自动回退两段 smooth_drag
        class MinimalDragHID:
            def __init__(self) -> None:
                self.calls: list[tuple[Any, ...]] = []

            def click(self, x: int, y: int, button: str = "left", hold_ms: int = 30) -> None:
                self.calls.append(("click", x, y, button))

            def smooth_drag(
                self,
                start: tuple[int, int],
                end: tuple[int, int],
                duration_ms: int = 110,
                steps: int = 10,
                release_at_end: bool = True,
            ) -> None:
                self.calls.append(("smooth_drag", start, end, release_at_end))

            def directional_flick(
                self, origin: tuple[int, int], direction: str, distance_px: int
            ) -> None:
                self.calls.append(("directional_flick", origin, direction, distance_px))

            def tap_key(self, key_name: str, hold_ms: int = 30) -> None:
                self.calls.append(("tap_key", key_name))

        min_hid = MinimalDragHID()
        min_disp = ControlDispatcher(keymap, layout, min_hid)  # type: ignore[arg-type]
        min_disp.process_tick(vp, prep, [ButtonTap(GamepadButton.RB)])
        min_hid.calls.clear()
        min_disp.process_tick(
            vp, prep, [ButtonTap(GamepadButton.A), ButtonTap(GamepadButton.A)]
        )
        assert min_disp.deploy_fsm.state == DeployState.AIMING_DIRECTION
        wp_xy = (center_xy[0] + expected_dx, center_xy[1])
        assert min_hid.calls == [
            ("smooth_drag", center_xy, wp_xy, False),
            ("smooth_drag", wp_xy, center_xy, True),
        ]
        assert len(rumbles) >= 3

    def test_modal_confirm_dialog_navigation_and_shortcuts(self) -> None:
        """需求 2：回合结束未用完资金/策略确认弹窗（MODAL_CONFIRM）导航、默认聚焦确认与快捷按键。"""
        dispatcher, hid, _, _, layout, vp = _make_env()
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.96, shop_open=False)
        confirm = SceneDetectionResult(GameScene.MODAL_CONFIRM, 0.95, shop_open=False)

        yes_pt = layout.buttons["confirm_yes"]
        no_pt = layout.buttons["confirm_no"]
        dont_pt = layout.buttons["confirm_dont_ask"]
        yes_xy = vp.norm_to_screen(yes_pt.u, yes_pt.v, yes_pt.anchor)
        no_xy = vp.norm_to_screen(no_pt.u, no_pt.v, no_pt.anchor)
        dont_xy = vp.norm_to_screen(dont_pt.u, dont_pt.v, dont_pt.anchor)

        # 在休整期触发开战（即使此前处于抓取态），视觉弹出 MODAL_CONFIRM 时立即无阻塞切入
        dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.A)])
        assert dispatcher.deploy_fsm.state == DeployState.HOLDING_UNIT
        snap_confirm = dispatcher.process_tick(vp, confirm, [])
        assert snap_confirm.scene == GameScene.MODAL_CONFIRM
        assert snap_confirm.focus_zone == FocusZone.CONFIRM_DIALOG
        assert snap_confirm.focus_label == "回合确认 · 确认"
        assert snap_confirm.cursor_norm == pytest.approx((yes_pt.u, yes_pt.v))
        assert snap_confirm.focus_polygon_norm == [
            (0.50, 0.623),
            (1.0, 0.623),
            (1.0, 0.723),
            (0.50, 0.723),
        ]

        # 左选取消、上选本局不再提示、按 A 勾选后焦点自动回移至确认
        snap_left = dispatcher.process_tick(vp, confirm, [ButtonTap(GamepadButton.DPAD_LEFT)])
        assert snap_left.focus_label == "回合确认 · 取消"
        assert snap_left.focus_polygon_norm == [
            (0.0, 0.623),
            (0.50, 0.623),
            (0.50, 0.723),
            (0.0, 0.723),
        ]

        hid.clear()
        snap_up = dispatcher.process_tick(vp, confirm, [ButtonTap(GamepadButton.DPAD_UP)])
        assert snap_up.focus_label == "回合确认 · 本局不再提示"
        snap_after_dont = dispatcher.process_tick(vp, confirm, [ButtonTap(GamepadButton.A)])
        assert hid.calls == [("click", dont_xy[0], dont_xy[1], "left")]
        assert snap_after_dont.focus_label == "回合确认 · 确认"

        # 验证 B 直接点取消、START / RT+A 直接点确认、X 直接点本局不再提示
        hid.clear()
        dispatcher.process_tick(
            vp,
            confirm,
            [
                ButtonTap(GamepadButton.B),
                ButtonTap(GamepadButton.START),
                ChordTriggered(GamepadButton.RT, GamepadButton.A),
                ButtonTap(GamepadButton.X),
                ButtonHoldTriggered(GamepadButton.START),
            ],
        )
        assert hid.calls == [
            ("click", no_xy[0], no_xy[1], "left"),
            ("click", yes_xy[0], yes_xy[1], "left"),
            ("click", yes_xy[0], yes_xy[1], "left"),
            ("click", dont_xy[0], dont_xy[1], "left"),
            ("click", yes_xy[0], yes_xy[1], "left"),
        ]

    def test_strategy_avatar_spatial_nav_click_inspect_and_drag_targets(self) -> None:
        """需求 5：右侧策略头像空间导航、RT+A 直接点击、X 查看说明、拖拽至棋盘/整备/商店与安全取消。"""
        dispatcher, hid, _, _, layout, vp = _make_env(auto_toggle_shop=True)
        board_mapper = PerspectiveBoardMapper(layout.board_quad)
        bench_mapper = LinearZoneMapper(layout.bench_zone)
        shop_open_mapper = LinearZoneMapper(
            layout.resolve_shop_zone(LayoutViewState.SHOP_OPEN, slot_count=4)
        )
        av_pt = layout.buttons["strategy_avatar"]
        ins_pt = layout.buttons["strategy_inspect"]
        safe_pt = layout.buttons["safe_blank"]
        av_xy = vp.norm_to_screen(av_pt.u, av_pt.v, av_pt.anchor)
        ins_xy = vp.norm_to_screen(ins_pt.u, ins_pt.v, ins_pt.anchor)
        safe_xy = vp.norm_to_screen(safe_pt.u, safe_pt.v, safe_pt.anchor)

        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.96, shop_open=False)
        dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.RB)])  # -> BOARD

        # 1. 从棋盘最右列按右跨入 FocusZone.STRATEGY
        snap_st = dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.DPAD_RIGHT)] * board_mapper.cols
        )
        assert snap_st.focus_zone == FocusZone.STRATEGY
        assert snap_st.focus_label == "右侧策略"

        # 上进 TOP_INFO 右放大镜「查看本轮出怪」
        snap_top = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.DPAD_UP)])
        assert snap_top.focus_zone == FocusZone.TOP_INFO
        assert "查看本轮出怪" in snap_top.focus_label

        # 回棋盘最右列 -> 右进 STRATEGY -> 下进 EQUIPMENT -> 上回棋盘 -> 右回 STRATEGY -> 左回棋盘最右列
        dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.DPAD_DOWN),
                *[ButtonTap(GamepadButton.DPAD_RIGHT)] * board_mapper.cols,
            ],
        )
        snap_eq = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.DPAD_DOWN)])
        assert snap_eq.focus_zone == FocusZone.EQUIPMENT

        dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.DPAD_UP),
                *[ButtonTap(GamepadButton.DPAD_RIGHT)] * board_mapper.cols,
            ],
        )
        snap_back_board = dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.DPAD_LEFT)]
        )
        assert snap_back_board.focus_zone == FocusZone.BOARD

        # 回到 STRATEGY
        dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.DPAD_RIGHT)])

        # 2. STRATEGY 空闲态：RT+A 一键点击使用；X 点击左侧查看按钮（不刷新商店）；B 点安全空白关闭
        hid.clear()
        dispatcher.process_tick(
            vp,
            prep,
            [
                ChordTriggered(GamepadButton.RT, GamepadButton.A),
                ButtonTap(GamepadButton.X),
                ButtonTap(GamepadButton.B),
            ],
        )
        assert hid.calls == [
            ("click", av_xy[0], av_xy[1], "left"),
            ("click", ins_xy[0], ins_xy[1], "left"),
            ("click", safe_xy[0], safe_xy[1], "left"),
        ]

        # 3. 按 A 锁定 HOLDING_STRATEGY：按 B 无点击取消；二次按 A 直接点击使用
        hid.clear()
        snap_lock = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.A)])
        assert snap_lock.deploy_state == DeployState.HOLDING_STRATEGY
        assert hid.calls == []
        snap_cancel = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.B)])
        assert snap_cancel.deploy_state == DeployState.IDLE
        assert hid.calls == []

        dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.A), ButtonTap(GamepadButton.A)]
        )
        assert hid.calls == [("click", av_xy[0], av_xy[1], "left")]
        assert dispatcher.deploy_fsm.state == DeployState.IDLE

        # 4. 锁定策略后经 LB/RB（STRATEGY -> BOARD -> BENCH -> SHOP）拖至棋盘干员格
        hid.clear()
        dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.A),  # 锁定策略
                ButtonTap(GamepadButton.RB),  # -> BOARD
                ButtonTap(GamepadButton.A),  # 拖至棋盘干员释放
            ],
        )
        assert len(hid.calls) == 1
        assert hid.calls[0][0] == "smooth_drag"
        assert hid.calls[0][1] == av_xy
        assert dispatcher.deploy_fsm.state == DeployState.IDLE

        # 5. 锁定策略后经 RB -> BOARD -> BENCH 拖至整备区干员槽
        dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.DPAD_RIGHT)] * board_mapper.cols
        )
        hid.clear()
        dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.A),  # 锁定策略
                ButtonTap(GamepadButton.RB),  # -> BOARD
                ButtonTap(GamepadButton.RB),  # -> BENCH
                ButtonTap(GamepadButton.A),  # 拖至整备槽释放
            ],
        )
        bench0_xy = vp.norm_to_screen(*bench_mapper.slot_center(0))
        assert hid.calls == [("smooth_drag", av_xy, bench0_xy, True)]

        # 6. 锁定策略后经 RB 切至 SHOP（商店原为收起态，自动按 A 展开后从固定 strategy_avatar 拖向商店卡槽）
        dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.RB),  # BENCH -> BOARD
                *[ButtonTap(GamepadButton.DPAD_RIGHT)] * board_mapper.cols,  # -> STRATEGY
            ],
        )
        assert dispatcher.shop_open is False
        hid.clear()
        snap_shop_used = dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.A),   # 锁定策略
                ButtonTap(GamepadButton.LB),  # 循环回 SHOP 并自动按 A 展开商店
                ButtonTap(GamepadButton.A),   # 从固定右侧 strategy_avatar 拖至商店 0 号卡
            ],
        )
        shop0_xy = vp.norm_to_screen(*shop_open_mapper.slot_center(0))
        assert snap_shop_used.deploy_state == DeployState.IDLE
        assert snap_shop_used.shop_open is True
        assert hid.calls == [
            ("tap_key", "a"),
            ("smooth_drag", av_xy, shop0_xy, True),
        ]

        # 7. 作战期（BATTLE_COMBAT）从棋盘最右列也可按右进入 STRATEGY，且自动切换作战期右下策略坐标
        combat = SceneDetectionResult(GameScene.BATTLE_COMBAT, 0.94)
        av_cb_pt = layout.buttons["strategy_avatar_combat"]
        ins_cb_pt = layout.buttons["strategy_inspect_combat"]
        av_cb_xy = vp.norm_to_screen(av_cb_pt.u, av_cb_pt.v, av_cb_pt.anchor)
        ins_cb_xy = vp.norm_to_screen(ins_cb_pt.u, ins_cb_pt.v, ins_cb_pt.anchor)
        dispatcher.process_tick(vp, combat, [])
        hid.clear()
        snap_cb_st = dispatcher.process_tick(
            vp,
            combat,
            [
                *[ButtonTap(GamepadButton.DPAD_RIGHT)] * board_mapper.cols,
                ButtonTap(GamepadButton.X),
                ButtonTap(GamepadButton.A),
            ],
        )
        assert snap_cb_st.focus_zone == FocusZone.STRATEGY
        assert hid.calls == [
            ("click", ins_cb_xy[0], ins_cb_xy[1], "left"),
            ("click", av_cb_xy[0], av_cb_xy[1], "left"),
        ]

    def test_manual_lock_battle_prep_still_syncs_shop_slots_and_metadata(self) -> None:
        """需求 6：手动锁层在 BATTLE_PREP 时仍实时同步 shop_open / shop_slot_count / unit_selected / covenant / teammates。"""
        dispatcher, _, _, _, _, vp = _make_env()
        prep4 = SceneDetectionResult(
            GameScene.BATTLE_PREP, 0.95, shop_open=False, shop_slot_count=4
        )
        dispatcher.process_tick(vp, prep4, [])

        # Back 单击进入手动锁层（BATTLE_PREP）
        snap_locked = dispatcher.process_tick(vp, prep4, [ButtonTap(GamepadButton.BACK)])
        assert snap_locked.scene_locked_manual is True
        assert snap_locked.scene == GameScene.BATTLE_PREP

        # 锁层期间视觉检测到商店展开且升至 6 槽、3 羁绊、联机有队友 -> 必须实时同步！
        prep6_coop = SceneDetectionResult(
            GameScene.BATTLE_PREP,
            0.96,
            shop_open=True,
            shop_slot_count=6,
            unit_selected=False,
            covenant_count=3,
            has_teammates=True,
        )
        snap_synced = dispatcher.process_tick(vp, prep6_coop, [])
        assert snap_synced.scene_locked_manual is True
        assert snap_synced.shop_open is True
        assert snap_synced.view_state == LayoutViewState.SHOP_OPEN.value
        assert dispatcher.shop_slot_count == 6
        assert dispatcher.covenant_count == 3
        assert dispatcher.has_teammates is True

        # 锁层期间即使视觉短暂判为 UNKNOWN_FREE_CURSOR，场景仍锁在 BATTLE_PREP 不丢失
        snap_hold = dispatcher.process_tick(
            vp, SceneDetectionResult(GameScene.UNKNOWN_FREE_CURSOR, 0.0), []
        )
        assert snap_hold.scene == GameScene.BATTLE_PREP
        assert snap_hold.scene_locked_manual is True

    def test_modal_2_choice_navigation_labels_white_bar_and_lt_peek(self) -> None:
        """v0.2.1：二选一（1x2）文案、choice_2_zone 精准落点、初始白条吸附、两段 A / RT+A / B 重选与按住 LT 看场。"""
        dispatcher, hid, _, _, layout, vp = _make_env()

        # 1. 无预选中进入二选一：默认聚焦左卡（索引 0），标签显示「二选一 [1/2]」，坐标对齐 choice_2_zone
        modal2 = SceneDetectionResult(GameScene.MODAL_CHOICE, 0.94, choice_count=2)
        snap0 = dispatcher.process_tick(vp, modal2, [])
        assert snap0.scene == GameScene.MODAL_CHOICE
        assert snap0.focus_zone == FocusZone.CHOICE_CARDS
        assert snap0.focus_label == "二选一 [1/2]"
        assert snap0.cursor_norm == pytest.approx(layout.resolve_choice_card_center(0, 2))
        assert snap0.cursor_norm == pytest.approx((0.3809, 0.5000), abs=1e-3)

        # 左右 / LB / RB 在两张卡之间循环切换
        snap_r = dispatcher.process_tick(vp, modal2, [ButtonTap(GamepadButton.DPAD_RIGHT)])
        assert snap_r.focus_label == "二选一 [2/2]"
        assert snap_r.cursor_norm == pytest.approx(layout.resolve_choice_card_center(1, 2))
        snap_wrap = dispatcher.process_tick(vp, modal2, [ButtonTap(GamepadButton.RB)])
        assert snap_wrap.focus_label == "二选一 [1/2]"
        snap_lb = dispatcher.process_tick(vp, modal2, [ButtonTap(GamepadButton.LB)])
        assert snap_lb.focus_label == "二选一 [2/2]"

        # 两段式按 A：首次点击右卡主体进入待确认态（吸附右卡白条），按 B 取消待确认重选，再次两段 A 确认
        card1_xy = vp.norm_to_screen(*layout.resolve_choice_card_center(1, 2))
        confirm1_xy = vp.norm_to_screen(*layout.resolve_choice_confirm_point(1, 2))
        hid.clear()
        snap_sel = dispatcher.process_tick(vp, modal2, [ButtonTap(GamepadButton.A)])
        assert snap_sel.focus_label == "二选一 [2/2] · 待确认"
        assert snap_sel.cursor_norm == pytest.approx(layout.resolve_choice_confirm_point(1, 2))
        assert hid.calls == [("click", card1_xy[0], card1_xy[1], "left")]

        snap_reselect = dispatcher.process_tick(vp, modal2, [ButtonTap(GamepadButton.B)])
        assert dispatcher.selected_choice_index is None
        assert snap_reselect.focus_label == "二选一 [2/2]"

        hid.clear()
        dispatcher.process_tick(
            vp, modal2, [ButtonTap(GamepadButton.A), ButtonTap(GamepadButton.A)]
        )
        assert hid.calls == [
            ("click", card1_xy[0], card1_xy[1], "left"),
            ("click", confirm1_xy[0], confirm1_xy[1], "left"),
        ]

        # 左移到第 1 卡，RT + A 一键选中并点击底部白条确认
        card0_xy = vp.norm_to_screen(*layout.resolve_choice_card_center(0, 2))
        confirm0_xy = vp.norm_to_screen(*layout.resolve_choice_confirm_point(0, 2))
        hid.clear()
        dispatcher.process_tick(
            vp,
            modal2,
            [
                ButtonTap(GamepadButton.DPAD_LEFT),
                ChordTriggered(GamepadButton.RT, GamepadButton.A),
            ],
        )
        assert hid.calls == [
            ("click", card0_xy[0], card0_xy[1], "left"),
            ("click", confirm0_xy[0], confirm0_xy[1], "left"),
        ]

        # 按住 LT 收起看场、松开 LT 恢复（复用 choice_toggle_eye）
        eye_pt = layout.resolve_choice_toggle_point(2)
        assert eye_pt is not None
        eye_xy = vp.norm_to_screen(eye_pt.u, eye_pt.v, eye_pt.anchor)
        hid.clear()
        snap_hide = dispatcher.process_tick(
            vp, modal2, [ModifierStateChanged(GamepadButton.LT, active=True)]
        )
        assert snap_hide.focus_polygon_norm is None
        snap_show = dispatcher.process_tick(
            vp, modal2, [ModifierStateChanged(GamepadButton.LT, active=False)]
        )
        assert snap_show.focus_polygon_norm is not None
        assert hid.calls == [
            ("click", eye_xy[0], eye_xy[1], "left"),
            ("click", eye_xy[0], eye_xy[1], "left"),
        ]

        # 2. 初始已由视觉识别出右卡白条（selected_choice_index=1）：进入即吸附白条，单按 A 直接确认
        dispatcher2, hid2, _, _, _, _ = _make_env()
        modal2_presel = SceneDetectionResult(
            GameScene.MODAL_CHOICE, 0.95, choice_count=2, selected_choice_index=1
        )
        snap_pre = dispatcher2.process_tick(vp, modal2_presel, [])
        assert snap_pre.focus_label == "二选一 [2/2] · 待确认"
        assert snap_pre.cursor_norm == pytest.approx(layout.resolve_choice_confirm_point(1, 2))
        hid2.clear()
        dispatcher2.process_tick(vp, modal2_presel, [ButtonTap(GamepadButton.A)])
        assert hid2.calls == [("click", confirm1_xy[0], confirm1_xy[1], "left")]

    def test_startup_free_cursor_physical_sync_and_subpixel_micro_push(self) -> None:
        """v0.2.1：启动与自由鼠标态同步物理光标、慢速微推摇杆亚像素不锁死、手动移鼠跟手与 LS 切入防拉回。"""
        keymap, layout = ConfigLoader(use_user_override=False).load()
        hid = MockVirtualHID(cursor_pos=(480, 270))
        dispatcher = ControlDispatcher(keymap=keymap, layout=layout, hid=hid)  # type: ignore[arg-type]
        vp = _make_viewport(is_foreground=True)
        unknown = SceneDetectionResult(GameScene.UNKNOWN_FREE_CURSOR, 0.0)

        # 1. 启动首帧同步当前物理光标 (480, 270) -> (0.25, 0.25)
        snap0 = dispatcher.process_tick(vp, unknown, [])
        assert snap0.scene == GameScene.UNKNOWN_FREE_CURSOR
        assert snap0.free_cursor_active is True
        assert snap0.cursor_norm == pytest.approx((480 / 1920, 270 / 1080))
        assert hid.calls == []

        # 2. 连续 10 帧极慢速微推摇杆（每帧 0.3px, 0.2px < 0.5px）：浮点亚像素累加不被整数读回抹零
        for _ in range(10):
            dispatcher.process_tick(
                vp, unknown, [StickMotion("left", 0.20, -0.15, 0.3, 0.2)]
            )
        assert hid.cursor_pos == (483, 272)
        assert hid.calls[-1] == ("move_cursor", 483, 272)

        # 3. 用户用物理鼠标移动 >= 2px 到 (1200, 600)：下一帧立刻无损同步 HUD 准星
        hid.cursor_pos = (1200, 600)
        snap_hand = dispatcher.process_tick(vp, unknown, [])
        assert snap_hand.cursor_norm == pytest.approx((1200 / 1920, 600 / 1080))

        # 4. 物理鼠标移到游戏窗口客户区外（例如副屏 -100, 300）：忽略不污染准星
        hid.cursor_pos = (-100, 300)
        snap_out = dispatcher.process_tick(vp, unknown, [])
        assert snap_out.cursor_norm == pytest.approx((1200 / 1920, 600 / 1080))

        # 5. 游戏切到后台再回前台：重置 _last_cmd_cursor_px 并在回前台首帧重同步
        bg_vp = _make_viewport(is_foreground=False)
        dispatcher.process_tick(bg_vp, unknown, [])
        hid.cursor_pos = (1201, 601)  # 即使只偏移 1px，重回前台也必须同步
        snap_refocus = dispatcher.process_tick(vp, unknown, [])
        assert snap_refocus.cursor_norm == pytest.approx((1201 / 1920, 601 / 1080))

        # 6. 局内按 LS 切至自由光标：立刻将真实鼠标移到吸附槽位，且下一帧不被旧物理光标拉走
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.96, shop_open=False)
        snap_prep = dispatcher.process_tick(vp, prep, [])
        assert snap_prep.focus_zone == FocusZone.BENCH
        bench0_xy = vp.norm_to_screen(*LinearZoneMapper(layout.bench_zone).slot_center(0))
        hid.cursor_pos = (100, 100)  # 模拟局内操作前物理鼠标停在左上角
        hid.clear()
        snap_ls = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.LS)])
        assert snap_ls.free_cursor_active is True
        assert hid.calls == [("move_cursor", bench0_xy[0], bench0_xy[1])]
        assert hid.cursor_pos == bench0_xy

        # 下一帧无新输入时准星与物理光标稳停在 bench0，不会被拉回旧位置
        snap_stay = dispatcher.process_tick(vp, prep, [])
        assert snap_stay.cursor_norm == pytest.approx(
            LinearZoneMapper(layout.bench_zone).slot_center(0)
        )


class TestV022ShopEmoteManualSyncAndConfirmAdapt:
    """v0.2.2 专项验证：商店防误触连按A买+RT+A查详情、交流表情面板、六选一手动锁层、商店收起防残框与ESC确认窗。"""

    def test_shop_double_tap_a_buy_and_rt_a_inspect_detail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """需求 1：商店首按 A 零点击仅预选、2s内二次 A 双击购买、B 仅取消预选、RT+A 单击查详情且屏蔽出售/撤回。"""
        now = 1000.0
        monkeypatch.setattr("arkcontroller.control.dispatcher.time.monotonic", lambda: now)

        dispatcher, hid, _, _, layout, vp = _make_env(use_native_shortcuts=True)
        shop_open_mapper = LinearZoneMapper(
            layout.resolve_shop_zone(LayoutViewState.SHOP_OPEN, slot_count=4)
        )
        prep_open = SceneDetectionResult(GameScene.BATTLE_PREP, 0.96, shop_open=True)
        dispatcher.process_tick(vp, prep_open, [])
        hid.clear()

        # 1. 第 1 次按 A：严禁发送任何鼠标点击，HUD focus_label 显示「再按A购买」
        snap_pre = dispatcher.process_tick(vp, prep_open, [ButtonTap(GamepadButton.A)])
        assert hid.calls == []
        assert dispatcher.shop_pending_buy_index == 0
        assert snap_pre.focus_zone == FocusZone.SHOP
        assert snap_pre.focus_label == "商店 [1/4] · 再按A购买"
        assert any(p.button == "A" and p.label == "确认购买" for p in snap_pre.prompts)

        # 2. 预选态按 B：仅取消预选，不收起商店！再按一次 B 才收起商店并退回整备区
        snap_cancel = dispatcher.process_tick(vp, prep_open, [ButtonTap(GamepadButton.B)])
        assert hid.calls == []
        assert dispatcher.shop_pending_buy_index is None
        assert snap_cancel.shop_open is True
        assert snap_cancel.focus_zone == FocusZone.SHOP
        assert snap_cancel.focus_label == "商店 [1/4]"

        # 3. 预选后换槽（DPAD_RIGHT）：自动清空原槽预选
        dispatcher.process_tick(vp, prep_open, [ButtonTap(GamepadButton.A)])
        assert dispatcher.shop_pending_buy_index == 0
        snap_move = dispatcher.process_tick(
            vp, prep_open, [ButtonTap(GamepadButton.DPAD_RIGHT)]
        )
        assert dispatcher.shop_pending_buy_index is None
        assert snap_move.focus_label == "商店 [2/4]"

        # 4. 预选超时（> 2.0s）后按 A：重新进入预选而不直接双击购买
        dispatcher.process_tick(vp, prep_open, [ButtonTap(GamepadButton.A)])
        assert dispatcher.shop_pending_buy_index == 1
        now += 2.1
        snap_expired = dispatcher.process_tick(vp, prep_open, [ButtonTap(GamepadButton.A)])
        assert hid.calls == []
        assert dispatcher.shop_pending_buy_index == 1
        assert snap_expired.focus_label == "商店 [2/4] · 再按A购买"

        # 5. 有效期内（+0.5s）再次对 2 号槽按 A：触发 double_click 购买并清空预选
        now += 0.5
        snap_bought = dispatcher.process_tick(vp, prep_open, [ButtonTap(GamepadButton.A)])
        shop1_xy = vp.norm_to_screen(*shop_open_mapper.slot_center(1))
        assert hid.calls == [("double_click", shop1_xy[0], shop1_xy[1])]
        assert dispatcher.shop_pending_buy_index is None
        assert snap_bought.focus_label == "商店 [2/4]"

        # 6. 按 X / Y / RS 刷新/升本/冻结前自动清空预选
        dispatcher.process_tick(vp, prep_open, [ButtonTap(GamepadButton.A)])
        assert dispatcher.shop_pending_buy_index == 1
        dispatcher.process_tick(vp, prep_open, [ButtonTap(GamepadButton.X)])
        assert dispatcher.shop_pending_buy_index is None

        # 7. 在商店按 RT+A：单击商店卡进入角色详情特写态，屏蔽 X(出售) 与 Y(撤回) 并隐藏提示
        hid.clear()
        snap_detail = dispatcher.process_tick(
            vp, prep_open, [ChordTriggered(GamepadButton.RT, GamepadButton.A)]
        )
        assert hid.calls == [("click", shop1_xy[0], shop1_xy[1], "left")]
        assert dispatcher.unit_selected is True
        assert dispatcher.unit_detail_from_shop is True
        assert snap_detail.view_state == LayoutViewState.UNIT_SELECTED.value
        assert "商店干员详情" in snap_detail.focus_label
        prompt_buttons = [p.button for p in snap_detail.prompts]
        assert "X" not in prompt_buttons
        assert "Y" not in prompt_buttons

        # 详情态按 X / Y 均被静默屏蔽，左右仍可切详情标签，按 B 点 safe_blank 退出并留在原商店 2 号槽
        hid.clear()
        dispatcher.process_tick(
            vp,
            prep_open,
            [
                ButtonTap(GamepadButton.X),
                ButtonTap(GamepadButton.Y),
                ButtonTap(GamepadButton.DPAD_RIGHT),
            ],
        )
        assert ("tap_key", "x") not in hid.calls
        assert ("tap_key", "q") not in hid.calls
        assert len(hid.calls) == 1 and hid.calls[0][0] == "click"

        hid.clear()
        safe_pt = layout.buttons["safe_blank"]
        safe_xy = vp.norm_to_screen(safe_pt.u, safe_pt.v, safe_pt.anchor)
        snap_exit = dispatcher.process_tick(vp, prep_open, [ButtonTap(GamepadButton.B)])
        assert hid.calls == [("click", safe_xy[0], safe_xy[1], "left")]
        assert dispatcher.unit_selected is False
        assert dispatcher.unit_detail_from_shop is False
        assert snap_exit.focus_zone == FocusZone.SHOP
        assert snap_exit.focus_label == "商店 [2/4]"

    def test_emote_panel_toggle_2x3_grid_scroll_and_visual_sync(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """需求 2：左下「交流」按钮（不发 z）、商店先收起、2x3 网格上下跨排、RS/LB/RB 翻页、A 发送、B 关闭与视觉同步。"""
        now = 2000.0
        monkeypatch.setattr("arkcontroller.control.dispatcher.time.monotonic", lambda: now)

        dispatcher, hid, _, _, layout, vp = _make_env(use_native_shortcuts=True)
        emote_btn = layout.buttons["emote_wheel"]
        emote_btn_xy = vp.norm_to_screen(emote_btn.u, emote_btn.v, emote_btn.anchor)
        center_btn = layout.buttons["emote_panel_center"]
        center_xy = vp.norm_to_screen(center_btn.u, center_btn.v, center_btn.anchor)

        # 1. 商店展开态按 LT + DPAD_DOWN：先按 A 收起商店，再点击左下「交流」按钮，严禁发 z 键
        prep_open = SceneDetectionResult(GameScene.BATTLE_PREP, 0.96, shop_open=True)
        dispatcher.process_tick(vp, prep_open, [])
        hid.clear()
        snap_open = dispatcher.process_tick(
            vp, prep_open, [ChordTriggered(GamepadButton.LT, GamepadButton.DPAD_DOWN)]
        )
        slot0_xy = vp.norm_to_screen(*layout.resolve_emote_slot_center(0), anchor="left_edge")
        assert snap_open.focus_zone == FocusZone.EMOTE_PANEL
        assert snap_open.shop_open is False
        assert snap_open.focus_label == "交流表情 [1/6]"
        assert dispatcher.emote_panel_open is True
        assert ("tap_key", "z") not in hid.calls
        assert hid.calls == [
            ("tap_key", "a"),
            ("click", emote_btn_xy[0], emote_btn_xy[1], "left"),
            ("move_cursor", slot0_xy[0], slot0_xy[1]),
        ]

        # 2. 验证 2x3 网格精确上下跨排映射（0<->3, 1<->4, 2<->5）与左右同行循环 + 光标同步跟随
        hid.clear()
        # 0 -> 右 -> 1 -> 下 -> 4（校验不踩 1^3==2 的异或坑！）-> 右 -> 5 -> 右循环 -> 3 -> 上 -> 0
        s1 = dispatcher.process_tick(vp, prep_open, [ButtonTap(GamepadButton.DPAD_RIGHT)])
        assert dispatcher.emote_index == 1 and s1.focus_label == "交流表情 [2/6]"
        s4 = dispatcher.process_tick(vp, prep_open, [ButtonTap(GamepadButton.DPAD_DOWN)])
        assert dispatcher.emote_index == 4 and s4.focus_label == "交流表情 [5/6]"
        s5 = dispatcher.process_tick(vp, prep_open, [ButtonTap(GamepadButton.DPAD_RIGHT)])
        assert dispatcher.emote_index == 5 and s5.focus_label == "交流表情 [6/6]"
        s3 = dispatcher.process_tick(vp, prep_open, [ButtonTap(GamepadButton.DPAD_RIGHT)])
        assert dispatcher.emote_index == 3 and s3.focus_label == "交流表情 [4/6]"
        s0 = dispatcher.process_tick(vp, prep_open, [ButtonTap(GamepadButton.DPAD_UP)])
        assert dispatcher.emote_index == 0 and s0.focus_label == "交流表情 [1/6]"
        slot4_xy = vp.norm_to_screen(*layout.resolve_emote_slot_center(4), anchor="left_edge")
        assert ("move_cursor", slot4_xy[0], slot4_xy[1]) in hid.calls

        # 3. 推右摇杆 RS 滚轮翻页：先自动移至 emote_panel_center 再滚动；LB/RB 快捷上下翻页
        hid.clear()
        dispatcher.process_tick(
            vp, prep_open, [StickMotion("right", 0.0, -0.8, 0.0, 26.0)]
        )
        assert hid.calls == [
            ("move_cursor", center_xy[0], center_xy[1]),
            ("scroll", -120, False),
        ]

        hid.clear()
        dispatcher.process_tick(
            vp,
            prep_open,
            [ButtonTap(GamepadButton.LB), ButtonTap(GamepadButton.RB)],
        )
        assert hid.calls == [
            ("move_cursor", center_xy[0], center_xy[1]),
            ("scroll", 240, False),
            ("move_cursor", center_xy[0], center_xy[1]),
            ("scroll", -240, False),
        ]

        # 4. 按 A 点击当前表情槽位发送，按 B 点击「交流」关闭并安全退回整备区（而非已关闭商店！）
        hid.clear()
        dispatcher.process_tick(vp, prep_open, [ButtonTap(GamepadButton.A)])
        assert hid.calls == [("click", slot0_xy[0], slot0_xy[1], "left")]

        hid.clear()
        snap_closed = dispatcher.process_tick(vp, prep_open, [ButtonTap(GamepadButton.B)])
        assert hid.calls == [("click", emote_btn_xy[0], emote_btn_xy[1], "left")]
        assert dispatcher.emote_panel_open is False
        assert snap_closed.focus_zone == FocusZone.BENCH

        # 5. 从联机队友栏 TEAMMATES 最底第 4 槽按 DPAD_DOWN 呼出交流表情面板
        prep_coop = SceneDetectionResult(
            GameScene.BATTLE_PREP, 0.96, shop_open=False, has_teammates=True
        )
        now += 1.0
        dispatcher.process_tick(vp, prep_coop, [ButtonTap(GamepadButton.RB)])  # BENCH -> BOARD
        for _ in range(12):
            snap_nav = dispatcher.process_tick(
                vp, prep_coop, [ButtonTap(GamepadButton.DPAD_LEFT)]
            )
            if snap_nav.focus_zone == FocusZone.TEAMMATES:
                break
        assert snap_nav.focus_zone == FocusZone.TEAMMATES
        while dispatcher.selected_teammate < 4:
            dispatcher.process_tick(vp, prep_coop, [ButtonTap(GamepadButton.DPAD_DOWN)])
        hid.clear()
        snap_from_mates = dispatcher.process_tick(
            vp, prep_coop, [ButtonTap(GamepadButton.DPAD_DOWN)]
        )
        assert snap_from_mates.focus_zone == FocusZone.EMOTE_PANEL
        assert ("click", emote_btn_xy[0], emote_btn_xy[1], "left") in hid.calls

        # 6. 视觉同步 emote_panel_open=False：1.8s 保护窗内与超窗后连续 3 帧 False 均不退出，第 4 帧 False 才自动退出 EMOTE_PANEL
        now += 1.0  # 仍在 1.8s 保护窗内
        snap_in_lock = dispatcher.process_tick(
            vp,
            SceneDetectionResult(
                GameScene.BATTLE_PREP, 0.96, shop_open=False, emote_panel_open=False
            ),
            [],
        )
        assert dispatcher.emote_panel_open is True
        assert snap_in_lock.focus_zone == FocusZone.EMOTE_PANEL

        now += 1.0  # 超过 1.8s 保护窗，需要连续 4 帧 False 才退出
        for _ in range(3):
            snap_hold = dispatcher.process_tick(
                vp,
                SceneDetectionResult(
                    GameScene.BATTLE_PREP, 0.96, shop_open=False, emote_panel_open=False
                ),
                [],
            )
            assert dispatcher.emote_panel_open is True
            assert snap_hold.focus_zone == FocusZone.EMOTE_PANEL

        snap_vis_close = dispatcher.process_tick(
            vp,
            SceneDetectionResult(
                GameScene.BATTLE_PREP, 0.96, shop_open=False, emote_panel_open=False
            ),
            [],
        )
        assert dispatcher.emote_panel_open is False
        assert snap_vis_close.focus_zone == FocusZone.TEAMMATES

        # 视觉同步 emote_panel_open=True 自动进入 EMOTE_PANEL
        snap_vis_open = dispatcher.process_tick(
            vp,
            SceneDetectionResult(
                GameScene.BATTLE_PREP, 0.96, shop_open=False, emote_panel_open=True
            ),
            [],
        )
        assert dispatcher.emote_panel_open is True
        assert snap_vis_open.focus_zone == FocusZone.EMOTE_PANEL

    def test_manual_lock_modal_choice_syncs_6_cards_and_selected_index(self) -> None:
        """需求 3：手动锁层 MODAL_CHOICE 时实时同步 choice_count=6 与 selected_choice_index。"""
        dispatcher, hid, _, _, layout, vp = _make_env()
        unknown = SceneDetectionResult(GameScene.UNKNOWN_FREE_CURSOR, 0.0)
        dispatcher.process_tick(vp, unknown, [])

        # Back 按 2 次：自动 -> BATTLE_PREP -> MODAL_CHOICE（手动锁层，默认 3 卡）
        dispatcher.process_tick(vp, unknown, [ButtonTap(GamepadButton.BACK)])
        snap_locked = dispatcher.process_tick(vp, unknown, [ButtonTap(GamepadButton.BACK)])
        assert snap_locked.scene_locked_manual is True
        assert snap_locked.scene == GameScene.MODAL_CHOICE
        assert "三选一" in snap_locked.focus_label

        # 视觉检测到多人六选一且第 5 张卡（索引 4）已预选白条 -> 手动锁层必须实时同步！
        modal6 = SceneDetectionResult(
            GameScene.MODAL_CHOICE, 0.95, choice_count=6, selected_choice_index=4
        )
        snap_sync6 = dispatcher.process_tick(vp, modal6, [])
        assert snap_sync6.scene_locked_manual is True
        assert "六选一" in snap_sync6.focus_label
        assert dispatcher.selected_choice_index == 4

        # 在 2x3 六选一中上下跨排跳到第 5 卡（初始 idx=1 -> 下移至 idx=4），此时正命中白条待确认，按 A 直接点白条
        snap_down = dispatcher.process_tick(vp, modal6, [ButtonTap(GamepadButton.DPAD_DOWN)])
        assert snap_down.focus_label == "六选一 [5/6] · 待确认"
        confirm4_xy = vp.norm_to_screen(*layout.resolve_choice_confirm_point(4, 6))
        hid.clear()
        dispatcher.process_tick(vp, modal6, [ButtonTap(GamepadButton.A)])
        assert hid.calls == [("click", confirm4_xy[0], confirm4_xy[1], "left")]

    def test_shop_closed_fallback_prevents_ghost_highlight_auto_and_manual(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """需求 4：自动与手动锁层下视觉检测到 shop_open=False 时，商店残框自动退回整备区并清预选。"""
        now = 3000.0
        monkeypatch.setattr("arkcontroller.control.dispatcher.time.monotonic", lambda: now)

        dispatcher, _, _, _, _, vp = _make_env()
        prep_open = SceneDetectionResult(GameScene.BATTLE_PREP, 0.96, shop_open=True)
        prep_closed = SceneDetectionResult(GameScene.BATTLE_PREP, 0.96, shop_open=False)

        # 1. 自动模式：在商店预选购买后，商店被外部关闭 -> 自动退回 BENCH 且清空预选
        dispatcher.process_tick(vp, prep_open, [ButtonTap(GamepadButton.A)])
        assert dispatcher.shop_pending_buy_index == 0
        now += 1.0
        snap_auto = dispatcher.process_tick(vp, prep_closed, [])
        assert snap_auto.shop_open is False
        assert snap_auto.focus_zone == FocusZone.BENCH
        assert dispatcher.shop_pending_buy_index is None

        # 2. 手动锁层 BATTLE_PREP：商店展开时停在 SHOP 并预选，随后视觉报告 shop_open=False -> 同样退回 BENCH
        snap_lock_shop = dispatcher.process_tick(
            vp, prep_open, [ButtonTap(GamepadButton.BACK)]
        )
        assert snap_lock_shop.scene_locked_manual is True
        assert snap_lock_shop.shop_open is True
        assert snap_lock_shop.focus_zone == FocusZone.SHOP
        dispatcher.process_tick(vp, prep_open, [ButtonTap(GamepadButton.A)])
        assert dispatcher.shop_pending_buy_index is not None

        now += 1.0
        snap_lock_fallback = dispatcher.process_tick(vp, prep_closed, [])
        assert snap_lock_fallback.scene_locked_manual is True
        assert snap_lock_fallback.shop_open is False
        assert snap_lock_fallback.focus_zone == FocusZone.BENCH
        assert dispatcher.shop_pending_buy_index is None

        # 3. 拖拽策略态（HOLDING_STRATEGY）停在 SHOP 时若视觉暂报 shop_open=False，不被强行拉离
        dispatcher2, _, _, _, layout2, _ = _make_env(auto_toggle_shop=True)
        board_cols = PerspectiveBoardMapper(layout2.board_quad).cols
        dispatcher2.process_tick(
            vp,
            prep_closed,
            [
                ButtonTap(GamepadButton.RB),  # BENCH -> BOARD
                *[ButtonTap(GamepadButton.DPAD_RIGHT)] * board_cols,  # -> STRATEGY
                ButtonTap(GamepadButton.A),  # 锁定 HOLDING_STRATEGY
                ButtonTap(GamepadButton.LB),  # -> SHOP（展开商店）
            ],
        )
        assert dispatcher2.deploy_fsm.state == DeployState.HOLDING_STRATEGY
        now += 1.0
        snap_drag_keep = dispatcher2.process_tick(vp, prep_closed, [])
        assert snap_drag_keep.deploy_state == DeployState.HOLDING_STRATEGY
        assert snap_drag_keep.focus_zone == FocusZone.SHOP

    def test_modal_confirm_without_dont_ask_checkbox_adaptation(self) -> None:
        """需求 5：ESC 退出确认弹窗 confirm_has_dont_ask=False 时仅保留取消/确认并屏蔽 UP 与 X。"""
        dispatcher, hid, _, _, layout, vp = _make_env()
        yes_pt = layout.buttons["confirm_yes"]
        no_pt = layout.buttons["confirm_no"]
        yes_xy = vp.norm_to_screen(yes_pt.u, yes_pt.v, yes_pt.anchor)
        no_xy = vp.norm_to_screen(no_pt.u, no_pt.v, no_pt.anchor)

        esc_confirm = SceneDetectionResult(
            GameScene.MODAL_CONFIRM,
            0.95,
            shop_open=False,
            confirm_has_dont_ask=False,
        )
        snap0 = dispatcher.process_tick(vp, esc_confirm, [])
        assert snap0.scene == GameScene.MODAL_CONFIRM
        assert dispatcher.confirm_has_dont_ask is False
        assert snap0.focus_label == "确认弹窗 · 确认"
        assert all(p.button != "X" for p in snap0.prompts)

        # 按 UP 不会切到第 3 项，按 X 不会产生任何误点击
        hid.clear()
        snap_up = dispatcher.process_tick(
            vp,
            esc_confirm,
            [ButtonTap(GamepadButton.DPAD_UP), ButtonTap(GamepadButton.X)],
        )
        assert hid.calls == []
        assert snap_up.focus_label == "确认弹窗 · 确认"

        # LB / RB 仅在 0(取消) <-> 1(确认) 两项间循环
        snap_rb = dispatcher.process_tick(vp, esc_confirm, [ButtonTap(GamepadButton.RB)])
        assert snap_rb.focus_label == "确认弹窗 · 取消"
        snap_rb2 = dispatcher.process_tick(vp, esc_confirm, [ButtonTap(GamepadButton.RB)])
        assert snap_rb2.focus_label == "确认弹窗 · 确认"
        snap_lb = dispatcher.process_tick(vp, esc_confirm, [ButtonTap(GamepadButton.LB)])
        assert snap_lb.focus_label == "确认弹窗 · 取消"

        # 按 A 点击取消，按 START 点击确认
        hid.clear()
        dispatcher.process_tick(
            vp,
            esc_confirm,
            [ButtonTap(GamepadButton.A), ButtonTap(GamepadButton.START)],
        )
        assert hid.calls == [
            ("click", no_xy[0], no_xy[1], "left"),
            ("click", yes_xy[0], yes_xy[1], "left"),
        ]

        # 手动锁层 MODAL_CONFIRM 时，先停在「本局不再提示」(idx=2)，视觉切换 confirm_has_dont_ask=False 自动钳回确认
        dispatcher2, hid2, _, _, _, _ = _make_env()
        confirm_with = SceneDetectionResult(
            GameScene.MODAL_CONFIRM, 0.95, confirm_has_dont_ask=True
        )
        dispatcher2.process_tick(vp, confirm_with, [ButtonTap(GamepadButton.DPAD_UP)])
        # 手动锁层循环至 MODAL_CONFIRM（自动 -> BATTLE_PREP -> MODAL_CHOICE -> MODAL_CONFIRM）
        for _ in range(3):
            dispatcher2.process_tick(vp, confirm_with, [ButtonTap(GamepadButton.BACK)])
        dispatcher2.process_tick(vp, confirm_with, [ButtonTap(GamepadButton.DPAD_UP)])
        snap_clamped = dispatcher2.process_tick(vp, esc_confirm, [])
        assert snap_clamped.scene_locked_manual is True
        assert dispatcher2.confirm_has_dont_ask is False
        assert snap_clamped.focus_label == "确认弹窗 · 确认"
        hid2.clear()
        dispatcher2.process_tick(vp, esc_confirm, [ButtonTap(GamepadButton.X)])
        assert hid2.calls == []


class TestV032CovenantOverflowAndEmoteLock:
    """v0.3.2 专项验证：顶部 >7 盟约溢出左对齐坐标与 RS/LB/RB 平滑拖拽滑动、表情面板全链路锁焦防逃逸。"""

    def test_covenant_overflow_left_aligned_coords_and_smooth_drag_scroll(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """需求 2：同步 covenant_overflow、左对齐 7 槽坐标与 7+ 标签、RS 横纵推杆与作战期 LB/RB 平滑拖拽盟约栏。"""
        now = 4000.0
        monkeypatch.setattr("arkcontroller.control.dispatcher.time.monotonic", lambda: now)

        dispatcher, hid, _, _, layout, vp = _make_env()
        prep_overflow = SceneDetectionResult(
            GameScene.BATTLE_PREP,
            0.96,
            shop_open=False,
            covenant_count=7,
            covenant_overflow=True,
        )
        # 进入休整期并切至棋盘最顶行再上跨进入 FocusZone.TOP_INFO
        dispatcher.process_tick(vp, prep_overflow, [ButtonTap(GamepadButton.RB)])  # BENCH -> BOARD
        assert dispatcher.covenant_count == 7
        assert dispatcher.covenant_overflow is True

        snap_top = dispatcher.process_tick(
            vp, prep_overflow, [ButtonTap(GamepadButton.DPAD_UP)] * 6
        )
        assert snap_top.focus_zone == FocusZone.TOP_INFO
        assert any(p.button == "RS" and "滑动盟约" in p.label for p in snap_top.prompts)

        # 导航到最左端「查看本局信息」后右移 1 格命中第 1 枚溢出羁绊，核对左对齐坐标与「7+」标签
        for _ in range(10):
            snap_top = dispatcher.process_tick(
                vp, prep_overflow, [ButtonTap(GamepadButton.DPAD_LEFT)]
            )
            if "查看本局信息" in snap_top.focus_label:
                break
        snap_cov0 = dispatcher.process_tick(
            vp, prep_overflow, [ButtonTap(GamepadButton.DPAD_RIGHT)]
        )
        expected_overflow_u0, expected_v = layout.covenant_zone.slot_center(
            0, count=7, overflow=True
        )
        centered_u0, _ = layout.covenant_zone.slot_center(0, count=7, overflow=False)
        assert expected_overflow_u0 == pytest.approx(0.2961, abs=1e-4)
        assert expected_overflow_u0 != pytest.approx(centered_u0, abs=1e-3)
        assert snap_cov0.cursor_norm == pytest.approx((expected_overflow_u0, expected_v))
        assert "羁绊 [1/7+]" in snap_cov0.focus_label

        # 安全带平滑拖拽坐标：right -> 0.62 拖向 0.38；left -> 0.38 拖向 0.62（v=0.1208）
        left_xy = vp.norm_to_screen(0.38, 0.1208)
        right_xy = vp.norm_to_screen(0.62, 0.1208)

        # 1. RS 水平向右推 -> 从 right_xy 拖向 left_xy
        hid.clear()
        dispatcher.process_tick(
            vp, prep_overflow, [StickMotion("right", 0.85, 0.0, 26.0, 0.0)]
        )
        assert hid.calls == [("smooth_drag", right_xy, left_xy, True)]

        # 0.30s 冷却内再次推杆不重复拖拽，也不穿透发 scroll
        hid.clear()
        now += 0.10
        dispatcher.process_tick(
            vp, prep_overflow, [StickMotion("right", 0.85, 0.0, 26.0, 0.0)]
        )
        assert hid.calls == []

        # 冷却结束后水平向左推 -> 从 left_xy 拖向 right_xy
        now += 0.30
        dispatcher.process_tick(
            vp, prep_overflow, [StickMotion("right", -0.85, 0.0, -26.0, 0.0)]
        )
        assert hid.calls == [("smooth_drag", left_xy, right_xy, True)]

        # 垂直向下推（dy_px > 0 -> 映射向右滑）与垂直向上推（dy_px < 0 -> 映射向左滑）
        hid.clear()
        now += 0.35
        dispatcher.process_tick(
            vp, prep_overflow, [StickMotion("right", 0.0, -0.85, 0.0, 26.0)]
        )
        now += 0.35
        dispatcher.process_tick(
            vp, prep_overflow, [StickMotion("right", 0.0, 0.85, 0.0, -26.0)]
        )
        assert hid.calls == [
            ("smooth_drag", right_xy, left_xy, True),
            ("smooth_drag", left_xy, right_xy, True),
        ]

        # 2. 手动锁层 BATTLE_PREP 同样实时同步 covenant_overflow
        dispatcher.process_tick(vp, prep_overflow, [ButtonTap(GamepadButton.BACK)])
        assert dispatcher.scene_locked_manual is True
        prep_no_overflow = SceneDetectionResult(
            GameScene.BATTLE_PREP,
            0.96,
            shop_open=False,
            covenant_count=5,
            covenant_overflow=False,
        )
        dispatcher.process_tick(vp, prep_no_overflow, [])
        assert dispatcher.covenant_overflow is False
        assert dispatcher.covenant_count == 5
        dispatcher.process_tick(vp, prep_overflow, [])
        assert dispatcher.covenant_overflow is True
        assert dispatcher.covenant_count == 7

        # 3. 作战期 BATTLE_COMBAT 在 TOP_INFO 下按 RB / LB 亦可向右/向左平滑拖拽
        dispatcher_c, hid_c, _, _, _, _ = _make_env()
        combat_overflow = SceneDetectionResult(
            GameScene.BATTLE_COMBAT,
            0.95,
            covenant_count=7,
            covenant_overflow=True,
        )
        snap_c_top = dispatcher_c.process_tick(
            vp, combat_overflow, [ButtonTap(GamepadButton.DPAD_UP)] * 6
        )
        assert snap_c_top.focus_zone == FocusZone.TOP_INFO
        assert any(p.button == "RS" and "滑动盟约" in p.label for p in snap_c_top.prompts)
        hid_c.clear()
        now += 0.35
        dispatcher_c.process_tick(vp, combat_overflow, [ButtonTap(GamepadButton.RB)])
        now += 0.35
        dispatcher_c.process_tick(vp, combat_overflow, [ButtonTap(GamepadButton.LB)])
        assert hid_c.calls == [
            ("smooth_drag", right_xy, left_xy, True),
            ("smooth_drag", left_xy, right_xy, True),
        ]

        # 4. 兼容旧式不接收 overflow 参数的自定义 covenant_zone（TypeError 安全兜底）
        class LegacyCovenantZone:
            default_count = 3

            def slot_center(self, index: int, count: int | None = None) -> tuple[float, float]:
                return (0.4 + 0.1 * index, 0.12)

            def slot_polygon(
                self, index: int, count: int | None = None
            ) -> list[tuple[float, float]]:
                return [(0.38, 0.08), (0.42, 0.08), (0.42, 0.16), (0.38, 0.16)]

        dispatcher_c.layout.covenant_zone = LegacyCovenantZone()  # type: ignore[assignment]
        snap_legacy = dispatcher_c.process_tick(vp, combat_overflow, [])
        assert snap_legacy.focus_zone == FocusZone.TOP_INFO

    def test_emote_panel_lock_rs_both_axes_l3_cursor_mode_and_4frame_debounce(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """需求 4：表情面板内 RS 横纵翻页、保护锁续期、L3 误按忽略、cursor 模式锁死与连续 4 帧 False 防抖。"""
        now = 5000.0
        monkeypatch.setattr("arkcontroller.control.dispatcher.time.monotonic", lambda: now)

        # 1. left_stick_mode="cursor" 下从 BOARD 打开 EMOTE_PANEL：左摇杆连续运动严禁激活自由光标、离散四向仍可 2x3 跳格、LS 误按忽略
        dispatcher_cur, hid_cur, _, _, layout, vp = _make_env(left_stick_mode="cursor")
        center_btn = layout.buttons["emote_panel_center"]
        center_xy = vp.norm_to_screen(center_btn.u, center_btn.v, center_btn.anchor)
        emote_btn = layout.buttons["emote_wheel"]
        emote_btn_xy = vp.norm_to_screen(emote_btn.u, emote_btn.v, emote_btn.anchor)

        prep_closed = SceneDetectionResult(
            GameScene.BATTLE_PREP, 0.96, shop_open=False, emote_panel_open=False
        )
        dispatcher_cur.process_tick(vp, prep_closed, [ButtonTap(GamepadButton.RB)])  # BENCH -> BOARD
        snap_open = dispatcher_cur.process_tick(
            vp, prep_closed, [ChordTriggered(GamepadButton.LT, GamepadButton.DPAD_DOWN)]
        )
        assert snap_open.focus_zone == FocusZone.EMOTE_PANEL
        assert snap_open.free_cursor_active is False

        # 推左摇杆连续运动（cursor 模式下原本会激活自由光标）+ 误按 L3（LS） -> 全部稳锁在 EMOTE_PANEL！
        hid_cur.clear()
        snap_locked = dispatcher_cur.process_tick(
            vp,
            prep_closed,
            [
                StickMotion("left", 0.9, 0.1, 20.0, 2.0),
                ButtonTap(GamepadButton.LS),
            ],
        )
        assert snap_locked.focus_zone == FocusZone.EMOTE_PANEL
        assert snap_locked.free_cursor_active is False
        assert hid_cur.calls == []

        # cursor 模式下左摇杆离散四向仍驱动 2x3 表情选格
        snap_dir = dispatcher_cur.process_tick(
            vp, prep_closed, [StickDirection("left", "right")]
        )
        assert snap_dir.focus_zone == FocusZone.EMOTE_PANEL
        assert dispatcher_cur.emote_index == 1

        # 2. left_stick_mode="dpad" 下验证 RS 横纵双轴均映射垂直翻页、续期 1.8s 保护锁、L3 忽略及 4 帧连续 False 防抖
        dispatcher, hid, _, _, _, _ = _make_env(left_stick_mode="dpad")
        dispatcher.process_tick(vp, prep_closed, [ButtonTap(GamepadButton.RB)])  # -> BOARD
        dispatcher.process_tick(
            vp, prep_closed, [ChordTriggered(GamepadButton.LT, GamepadButton.DPAD_DOWN)]
        )
        assert dispatcher.emote_panel_open is True

        # 误按 LS（dpad 模式下原本会切自由光标）-> 必须完全忽略，留在 EMOTE_PANEL
        hid.clear()
        snap_ls = dispatcher.process_tick(vp, prep_closed, [ButtonTap(GamepadButton.LS)])
        assert snap_ls.focus_zone == FocusZone.EMOTE_PANEL
        assert snap_ls.free_cursor_active is False
        assert hid.calls == []

        # RS 直觉向右推（下一页 -> scroll -120, horizontal=False）与向左推（上一页 -> scroll +120, horizontal=False）
        now += 1.2  # 即将到期前执行翻页 -> 刷新 1.8s 保护锁！
        hid.clear()
        dispatcher.process_tick(
            vp, prep_closed, [StickMotion("right", 0.85, 0.0, 26.0, 0.0)]
        )
        assert hid.calls == [
            ("move_cursor", center_xy[0], center_xy[1]),
            ("scroll", -120, False),
        ]

        hid.clear()
        dispatcher.process_tick(
            vp, prep_closed, [StickMotion("right", -0.85, 0.0, -26.0, 0.0)]
        )
        assert hid.calls == [
            ("move_cursor", center_xy[0], center_xy[1]),
            ("scroll", 120, False),
        ]

        # 续期后过 1.0s（总耗时距打开 2.2s，但距上次翻页仅 1.0s < 1.8s）：连续多帧 False 均被保护锁拦下！
        now += 1.0
        for _ in range(5):
            snap_protected = dispatcher.process_tick(
                vp,
                SceneDetectionResult(
                    GameScene.BATTLE_PREP, 0.96, shop_open=False, emote_panel_open=False
                ),
                [],
            )
            assert snap_protected.focus_zone == FocusZone.EMOTE_PANEL
            assert dispatcher.emote_panel_open is True

        # 超过 1.8s 保护窗后：先模拟 125Hz 主循环（每 8ms 复用同一个 False 视觉帧 10 次）-> 仅计为 1 个视觉帧，绝不退出！
        now += 1.0
        shared_false_frame = SceneDetectionResult(
            GameScene.BATTLE_PREP, 0.96, shop_open=False, emote_panel_open=False
        )
        for _ in range(10):
            now += 0.008
            snap_125hz = dispatcher.process_tick(vp, shared_false_frame, [])
            assert snap_125hz.focus_zone == FocusZone.EMOTE_PANEL
            assert dispatcher.emote_panel_open is True

        # 再来 2 个新视觉帧 False（累计 3 帧 < 4 帧）仍不逃逸入 BOARD
        for _ in range(2):
            now += 0.20
            snap_f = dispatcher.process_tick(
                vp,
                SceneDetectionResult(
                    GameScene.BATTLE_PREP, 0.96, shop_open=False, emote_panel_open=False
                ),
                [],
            )
            assert snap_f.focus_zone == FocusZone.EMOTE_PANEL

        # 中途来 1 帧 True -> 防抖计数立即归零！随后再来 3 个新视觉帧 False 依然稳锁 EMOTE_PANEL
        now += 0.20
        dispatcher.process_tick(
            vp,
            SceneDetectionResult(
                GameScene.BATTLE_PREP, 0.96, shop_open=False, emote_panel_open=True
            ),
            [],
        )
        for _ in range(3):
            now += 0.20
            snap_reset = dispatcher.process_tick(
                vp,
                SceneDetectionResult(
                    GameScene.BATTLE_PREP, 0.96, shop_open=False, emote_panel_open=False
                ),
                [],
            )
            assert snap_reset.focus_zone == FocusZone.EMOTE_PANEL

        # 真正连续第 4 个视觉帧 False -> 安全退回原区域 BOARD
        now += 0.20
        snap_exit = dispatcher.process_tick(
            vp,
            SceneDetectionResult(
                GameScene.BATTLE_PREP, 0.96, shop_open=False, emote_panel_open=False
            ),
            [],
        )
        assert dispatcher.emote_panel_open is False
        assert snap_exit.focus_zone == FocusZone.BOARD

        # 3. 显式 LT+DPAD_DOWN 打开后再按 LT+DPAD_DOWN 显式关闭，点击 emote_wheel 并回到 BOARD
        dispatcher.process_tick(
            vp, prep_closed, [ChordTriggered(GamepadButton.LT, GamepadButton.DPAD_DOWN)]
        )
        assert dispatcher.emote_panel_open is True
        hid.clear()
        snap_lt_close = dispatcher.process_tick(
            vp, prep_closed, [ChordTriggered(GamepadButton.LT, GamepadButton.DPAD_DOWN)]
        )
        assert snap_lt_close.focus_zone == FocusZone.BOARD
        assert dispatcher.emote_panel_open is False
        assert hid.calls == [("click", emote_btn_xy[0], emote_btn_xy[1], "left")]

    def test_teammate_panel_b_closes_via_top_left_latency_safe_blank(self) -> None:
        """v0.3.2 需求 1：多人查看队友后在休整/作战/自由光标按 B 均优先点击左上延迟区 safe_blank 关闭队友抽屉。"""
        dispatcher, hid, _, _, layout, vp = _make_env()
        safe_pt = layout.buttons["safe_blank"]
        safe_xy = vp.norm_to_screen(safe_pt.u, safe_pt.v, safe_pt.anchor)
        assert safe_pt.u == pytest.approx(0.1172, abs=1e-4)
        assert safe_pt.v == pytest.approx(0.0903, abs=1e-4)
        assert safe_pt.anchor == "left_edge"

        # 1. 休整期商店展开态按 LT+A 打开 1 号队友抽屉后按 B：优先点左上延迟区关闭队友抽屉，不先收起商店！
        prep_shop = SceneDetectionResult(
            GameScene.BATTLE_PREP, 0.96, shop_open=True, has_teammates=True
        )
        dispatcher.process_tick(vp, prep_shop, [])
        dispatcher.process_tick(
            vp, prep_shop, [ChordTriggered(GamepadButton.LT, GamepadButton.A)]
        )
        assert dispatcher.teammate_panel_open is True
        hid.clear()
        snap_close_shop = dispatcher.process_tick(
            vp, prep_shop, [ButtonTap(GamepadButton.B)]
        )
        assert hid.calls == [("click", safe_xy[0], safe_xy[1], "left")]
        assert dispatcher.teammate_panel_open is False
        assert snap_close_shop.shop_open is True
        assert snap_close_shop.focus_zone == FocusZone.SHOP

        # 2. 休整期从棋盘左跨进入 FocusZone.TEAMMATES 按 A 查看队友后，再按 B 必点 safe_blank 并安全切回棋盘
        prep_closed = SceneDetectionResult(
            GameScene.BATTLE_PREP, 0.96, shop_open=False, has_teammates=True
        )
        dispatcher.process_tick(vp, prep_closed, [ButtonTap(GamepadButton.RB)])  # -> BOARD
        for _ in range(12):
            snap = dispatcher.process_tick(
                vp, prep_closed, [ButtonTap(GamepadButton.DPAD_LEFT)]
            )
            if snap.focus_zone == FocusZone.TEAMMATES:
                break
        assert snap.focus_zone == FocusZone.TEAMMATES
        dispatcher.process_tick(vp, prep_closed, [ButtonTap(GamepadButton.A)])
        assert dispatcher.teammate_panel_open is True

        hid.clear()
        snap_back_prep = dispatcher.process_tick(
            vp, prep_closed, [ButtonTap(GamepadButton.B)]
        )
        assert hid.calls == [("click", safe_xy[0], safe_xy[1], "left")]
        assert dispatcher.teammate_panel_open is False
        assert snap_back_prep.focus_zone == FocusZone.BOARD

        # 3. 作战期（BATTLE_COMBAT）在 FocusZone.TEAMMATES 打开队友面板后按 B，必点 safe_blank 并返回棋盘
        combat_coop = SceneDetectionResult(
            GameScene.BATTLE_COMBAT, 0.95, has_teammates=True
        )
        dispatcher.process_tick(vp, combat_coop, [])
        for _ in range(12):
            snap = dispatcher.process_tick(
                vp, combat_coop, [ButtonTap(GamepadButton.DPAD_LEFT)]
            )
            if snap.focus_zone == FocusZone.TEAMMATES:
                break
        assert snap.focus_zone == FocusZone.TEAMMATES
        dispatcher.process_tick(vp, combat_coop, [ButtonTap(GamepadButton.A)])
        assert dispatcher.teammate_panel_open is True
        hid.clear()
        snap_back_combat = dispatcher.process_tick(
            vp, combat_coop, [ButtonTap(GamepadButton.B)]
        )
        assert hid.calls == [("click", safe_xy[0], safe_xy[1], "left")]
        assert dispatcher.teammate_panel_open is False
        assert snap_back_combat.focus_zone == FocusZone.BOARD

        # 4. 局内自由光标态下若队友抽屉已打开，按 B 优先点击左上延迟区关闭队友抽屉而非发 ESC
        dispatcher.process_tick(
            vp,
            combat_coop,
            [
                ChordTriggered(GamepadButton.LT, GamepadButton.DPAD_RIGHT),
                ButtonTap(GamepadButton.LS),  # 切至自由光标
            ],
        )
        assert dispatcher.teammate_panel_open is True
        hid.clear()
        dispatcher.process_tick(vp, combat_coop, [ButtonTap(GamepadButton.B)])
        assert hid.calls == [("click", safe_xy[0], safe_xy[1], "left")]
        assert ("tap_key", "esc") not in hid.calls
        assert dispatcher.teammate_panel_open is False

        # 5. 缺少 safe_blank 配置键时兜底仍精准落在 (0.1172, 0.0903, left_edge)
        del dispatcher.layout.buttons["safe_blank"]
        dispatcher.process_tick(
            vp, combat_coop, [ChordTriggered(GamepadButton.LT, GamepadButton.B)]
        )
        hid.clear()
        dispatcher.process_tick(vp, combat_coop, [ButtonTap(GamepadButton.B)])
        assert hid.calls == [
            ("click", *vp.norm_to_screen(0.1172, 0.0903, anchor="left_edge"), "left")
        ]

    def test_holding_unit_and_equip_removed_gift_teammate_completely(self) -> None:
        """v0.3.2 需求 2：彻底移除抓取干员/装备赠送队友（无 gift_holding、按 Y 无操作、抓取态棋盘最左列不入 TEAMMATES）。"""
        dispatcher, hid, _, _, layout, vp = _make_env()
        board_cols = PerspectiveBoardMapper(layout.board_quad).cols
        prep_coop = SceneDetectionResult(
            GameScene.BATTLE_PREP, 0.96, shop_open=False, has_teammates=True
        )
        dispatcher.process_tick(vp, prep_coop, [])
        assert not hasattr(dispatcher.deploy_fsm, "gift_holding")

        # 1. 抓取整备干员 -> 切棋盘最左列继续按左：严禁跨入 FocusZone.TEAMMATES；按 Y 零调用
        snap_hold_unit = dispatcher.process_tick(
            vp,
            prep_coop,
            [
                ButtonTap(GamepadButton.A),  # 抓整备干员
                ButtonTap(GamepadButton.RB),  # -> BOARD
                *[ButtonTap(GamepadButton.DPAD_LEFT)] * (board_cols + 2),
            ],
        )
        assert snap_hold_unit.deploy_state == DeployState.HOLDING_UNIT
        assert snap_hold_unit.focus_zone == FocusZone.BOARD
        assert all(p.button != "Y" and "赠" not in p.label for p in snap_hold_unit.prompts)
        hid.clear()
        dispatcher.process_tick(vp, prep_coop, [ButtonTap(GamepadButton.Y)])
        assert hid.calls == []
        dispatcher.process_tick(vp, prep_coop, [ButtonTap(GamepadButton.B)])

        # 2. 抓取装备 -> 切棋盘最左列继续按左同样不入 TEAMMATES，按 Y 零调用
        dispatcher.process_tick(vp, prep_coop, [ButtonTap(GamepadButton.RB)])  # BOARD -> EQUIPMENT
        snap_hold_eq = dispatcher.process_tick(
            vp,
            prep_coop,
            [
                ButtonTap(GamepadButton.A),  # 抓装备
                ButtonTap(GamepadButton.RB),  # -> BOARD
                *[ButtonTap(GamepadButton.DPAD_LEFT)] * (board_cols + 2),
            ],
        )
        assert snap_hold_eq.deploy_state == DeployState.HOLDING_EQUIP
        assert snap_hold_eq.focus_zone == FocusZone.BOARD
        assert all(p.button != "Y" and "赠" not in p.label for p in snap_hold_eq.prompts)
        hid.clear()
        dispatcher.process_tick(vp, prep_coop, [ButtonTap(GamepadButton.Y)])
        assert hid.calls == []

    def test_lt_rt_unordered_chord_and_dual_state_strategy_coords(self) -> None:
        """v0.3.2 需求 3：休整/作战双态策略头像坐标 + 无序 LT+RT（含 GestureEngine 端到端）一键点击本局策略。"""
        dispatcher, hid, rumbles, keymap, layout, vp = _make_env()
        prep_av = layout.buttons["strategy_avatar"]
        combat_av = layout.buttons["strategy_avatar_combat"]
        combat_ins = layout.buttons["strategy_inspect_combat"]
        prep_av_xy = vp.norm_to_screen(prep_av.u, prep_av.v, prep_av.anchor)
        combat_av_xy = vp.norm_to_screen(combat_av.u, combat_av.v, combat_av.anchor)
        combat_ins_xy = vp.norm_to_screen(combat_ins.u, combat_ins.v, combat_ins.anchor)

        # 1. 休整期焦点停在整备区 BENCH：先 LT 后 RT 与先 RT 后 LT 均直接点击休整期策略头像
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.96, shop_open=False)
        dispatcher.process_tick(vp, prep, [])
        hid.clear()
        rumbles.clear()
        snap_p1 = dispatcher.process_tick(
            vp,
            prep,
            [
                ChordTriggered(GamepadButton.LT, GamepadButton.RT),
                ChordTriggered(GamepadButton.RT, GamepadButton.LT),
            ],
        )
        assert snap_p1.focus_zone == FocusZone.BENCH
        assert hid.calls == [
            ("click", prep_av_xy[0], prep_av_xy[1], "left"),
            ("click", prep_av_xy[0], prep_av_xy[1], "left"),
        ]
        assert len(rumbles) == 2

        # 2. HOLDING_UNIT / HOLDING_EQUIP / AIMING_DIRECTION 中途按 LT+RT 被安全拦截不打断部署
        dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.A)])
        assert dispatcher.deploy_fsm.state == DeployState.HOLDING_UNIT
        hid.clear()
        dispatcher.process_tick(
            vp, prep, [ChordTriggered(GamepadButton.LT, GamepadButton.RT)]
        )
        assert hid.calls == []
        assert dispatcher.deploy_fsm.state == DeployState.HOLDING_UNIT

        # 落位进入 AIMING_DIRECTION 按 LT+RT 同样安全拦截
        dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.RB), ButtonTap(GamepadButton.A)]
        )
        assert dispatcher.deploy_fsm.state == DeployState.AIMING_DIRECTION
        hid.clear()
        dispatcher.process_tick(
            vp, prep, [ChordTriggered(GamepadButton.RT, GamepadButton.LT)]
        )
        assert hid.calls == []
        assert dispatcher.deploy_fsm.state == DeployState.AIMING_DIRECTION
        dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.B)])

        # 3. HOLDING_STRATEGY 态下按 LT+RT 直接原地点击释放并重置为 IDLE
        board_cols = PerspectiveBoardMapper(layout.board_quad).cols
        dispatcher.process_tick(
            vp,
            prep,
            [
                *[ButtonTap(GamepadButton.DPAD_RIGHT)] * board_cols,
                ButtonTap(GamepadButton.A),
            ],
        )
        assert dispatcher.deploy_fsm.state == DeployState.HOLDING_STRATEGY
        hid.clear()
        dispatcher.process_tick(
            vp, prep, [ChordTriggered(GamepadButton.LT, GamepadButton.RT)]
        )
        assert hid.calls == [("click", prep_av_xy[0], prep_av_xy[1], "left")]
        assert dispatcher.deploy_fsm.state == DeployState.IDLE

        # 4. 作战期（BATTLE_COMBAT）：无论焦点在 BOARD 还是 STRATEGY，LT+RT 均点击作战期右下策略头像；X 点击作战期查看按钮
        combat = SceneDetectionResult(GameScene.BATTLE_COMBAT, 0.95)
        dispatcher.process_tick(vp, combat, [])
        hid.clear()
        dispatcher.process_tick(
            vp,
            combat,
            [
                ChordTriggered(GamepadButton.LT, GamepadButton.RT),
                ChordTriggered(GamepadButton.RT, GamepadButton.LT),
            ],
        )
        assert hid.calls == [
            ("click", combat_av_xy[0], combat_av_xy[1], "left"),
            ("click", combat_av_xy[0], combat_av_xy[1], "left"),
        ]

        # 作战期移至 STRATEGY 校验选框中心与查看按钮均为作战期坐标
        hid.clear()
        snap_cb_s = dispatcher.process_tick(
            vp,
            combat,
            [
                *[ButtonTap(GamepadButton.DPAD_RIGHT)] * board_cols,
                ButtonTap(GamepadButton.X),
            ],
        )
        assert snap_cb_s.focus_zone == FocusZone.STRATEGY
        assert snap_cb_s.cursor_norm == pytest.approx((combat_av.u, combat_av.v))
        assert hid.calls == [("click", combat_ins_xy[0], combat_ins_xy[1], "left")]

        # 5. MODAL_CHOICE 决策弹窗下按 LT+RT 不误点策略
        modal3 = SceneDetectionResult(GameScene.MODAL_CHOICE, 0.95, choice_count=3)
        dispatcher.process_tick(vp, modal3, [])
        hid.clear()
        dispatcher.process_tick(
            vp, modal3, [ChordTriggered(GamepadButton.LT, GamepadButton.RT)]
        )
        assert hid.calls == []

        # 6. 端到端集成 GestureEngine：验证先 RT 后 LT、先 LT 后 RT 与同帧双按均精准触发策略点击
        engine = GestureEngine(keymap)
        dispatcher.process_tick(vp, combat, [])
        hid.clear()
        # 6a. 先按 RT 再按 LT（错帧 1 帧）
        ev1 = engine.update(
            GamepadState(connected=True, rt=0.9, buttons=frozenset({GamepadButton.RT})),
            0.016,
        )
        dispatcher.process_tick(vp, combat, ev1)
        ev2 = engine.update(
            GamepadState(
                connected=True,
                lt=0.9,
                rt=0.9,
                buttons=frozenset({GamepadButton.LT, GamepadButton.RT}),
            ),
            0.016,
        )
        dispatcher.process_tick(vp, combat, ev2)
        ev3 = engine.update(GamepadState(connected=True), 0.016)
        dispatcher.process_tick(vp, combat, ev3)
        # 6b. 先按 LT 再按 RT（错帧 1 帧）
        ev4 = engine.update(
            GamepadState(connected=True, lt=0.9, buttons=frozenset({GamepadButton.LT})),
            0.016,
        )
        dispatcher.process_tick(vp, combat, ev4)
        ev5 = engine.update(
            GamepadState(
                connected=True,
                lt=0.9,
                rt=0.9,
                buttons=frozenset({GamepadButton.LT, GamepadButton.RT}),
            ),
            0.016,
        )
        dispatcher.process_tick(vp, combat, ev5)
        ev6 = engine.update(GamepadState(connected=True), 0.016)
        dispatcher.process_tick(vp, combat, ev6)
        # 6c. 同帧同时按 LT + RT
        ev7 = engine.update(
            GamepadState(
                connected=True,
                lt=0.9,
                rt=0.9,
                buttons=frozenset({GamepadButton.LT, GamepadButton.RT}),
            ),
            0.016,
        )
        dispatcher.process_tick(vp, combat, ev7)
        engine.update(GamepadState(connected=True), 0.016)
        assert hid.calls == [
            ("click", combat_av_xy[0], combat_av_xy[1], "left"),
            ("click", combat_av_xy[0], combat_av_xy[1], "left"),
            ("click", combat_av_xy[0], combat_av_xy[1], "left"),
        ]
