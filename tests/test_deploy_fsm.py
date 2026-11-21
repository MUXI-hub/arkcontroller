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

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def clear(self) -> None:
        self.calls.clear()

    def move_cursor(self, screen_x: int, screen_y: int) -> None:
        self.calls.append(("move_cursor", screen_x, screen_y))

    def click(
        self,
        screen_x: int,
        screen_y: int,
        button: str = "left",
        hold_ms: int = 30,
    ) -> None:
        self.calls.append(("click", screen_x, screen_y, button))

    def double_click(
        self,
        screen_x: int,
        screen_y: int,
        interval_ms: int = 55,
    ) -> None:
        self.calls.append(("double_click", screen_x, screen_y))

    def smooth_drag(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        duration_ms: int = 110,
        steps: int = 10,
        release_at_end: bool = True,
    ) -> None:
        self.calls.append(("smooth_drag", start, end, release_at_end))

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

        # 3. Back 单击手动循环锁层，锁层后忽略自动场景覆盖，循环一圈后恢复自动
        snap_lock1 = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.BACK)])
        assert snap_lock1.scene_locked_manual is True
        assert snap_lock1.scene == GameScene.ACTIVITY_HUB

        # 视觉仍报 BATTLE_PREP，但因为手动锁层不会被覆盖
        snap_still_locked = dispatcher.process_tick(vp, prep, [])
        assert snap_still_locked.scene == GameScene.ACTIVITY_HUB

        # 继续按 Back 依次走完 SQUAD_EDIT -> BATTLE_PREP -> MODAL_CHOICE -> UNKNOWN_FREE_CURSOR -> AUTO
        expected_scenes = [
            GameScene.SQUAD_EDIT,
            GameScene.BATTLE_PREP,
            GameScene.MODAL_CHOICE,
            GameScene.UNKNOWN_FREE_CURSOR,
        ]
        for expected in expected_scenes:
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

        # 按 LS 切入自由光标模式，随后左摇杆平滑移动鼠标并按 A 点击
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
        assert hid.calls[1][0] == "click"
        assert hid.calls[1][1:3] == hid.calls[0][1:3]

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
        squad = SceneDetectionResult(GameScene.SQUAD_EDIT, 0.95)
        dt = 0.008  # 125Hz 主循环帧间隔

        # 1. 显式亚像素零增量帧不得触发任何滚轮事件
        dispatcher_light.process_tick(
            vp,
            squad,
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
            dispatcher_light.process_tick(vp, squad, ge_light.update(st_light, dt))
            dispatcher_med.process_tick(vp, squad, ge_med.update(st_med, dt))

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
            dispatcher_h.process_tick(vp, squad, ge_h.update(st_h, dt))

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

        # 再次抓棋盘干员移至邻格后按 B 安全取消落位（应发送右键取消而非 Esc）
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
        cancel_xy = vp.norm_to_screen(*dispatcher.deploy_fsm.aim_origin_norm)  # type: ignore[misc]
        hid.clear()

        snap_cancelled = dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.B)]
        )
        assert snap_cancelled.deploy_state == DeployState.IDLE
        assert hid.calls == [("click", cancel_xy[0], cancel_xy[1], "right")]
        assert board_mapper.rows == 5

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

        # 从 SHOP 按 LB 一步循环到 EQUIPMENT，下移至 1 号装备格，按 A 抓取装备
        snap_eq = dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.LB),
                ButtonTap(GamepadButton.DPAD_DOWN),
                ButtonTap(GamepadButton.A),
            ],
        )
        assert snap_eq.focus_zone == FocusZone.EQUIPMENT
        assert snap_eq.deploy_state == DeployState.HOLDING_EQUIP

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

    def test_quick_sell_native_vs_mouse_dropzone_and_coop_gift(self) -> None:
        """验证抓取态按 X 出售（场上 E / 整备与装备 Q / 关闭快捷键拖 sell_dropzone）与按 Y 赠卡。"""
        # 1. 原生快捷键开启：整备区按 Q，棋盘按 E
        dispatcher, hid, _, _, layout, vp = _make_env(use_native_shortcuts=True)
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.95, shop_open=False)
        dispatcher.process_tick(vp, prep, [])

        # 整备区抓起按 X -> Q
        hid.clear()
        dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.A), ButtonTap(GamepadButton.X)]
        )
        assert ("tap_key", "q") in hid.calls
        assert dispatcher.deploy_fsm.state == DeployState.IDLE

        # 切到棋盘抓起按 X -> E
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
        assert ("tap_key", "e") in hid.calls
        assert dispatcher.deploy_fsm.state == DeployState.IDLE

        # 切到装备栏抓起按 X -> Q
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
        assert ("tap_key", "q") in hid.calls

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

        # 3. 联机赠卡：先用 LT + D-Pad Right 切至队友 1 -> 2，再抓取整备干员按 Y 拖向 teammate_2
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
        dispatcher_mouse.process_tick(
            vp_m, prep, [ButtonTap(GamepadButton.A), ButtonTap(GamepadButton.Y)]
        )
        assert ("smooth_drag", bench0_xy, mate2_xy, True) in hid_mouse.calls

    def test_shop_economy_ready_hold_and_lt_chords(self) -> None:
        """验证商店买牌/极速连买/刷新/升本/冻结、Start长按开战及全部 LT 局内组合键。"""
        dispatcher, hid, _, _, layout, vp = _make_env(use_native_shortcuts=True)
        shop_mapper = LinearZoneMapper(layout.shop_zone)
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.98, shop_open=True)
        dispatcher.process_tick(vp, prep, [])
        hid.clear()

        # 商店选买 A、极速连买 RT+A、刷新 X(D)、升本 Y(点按钮)、冻结 RS(R)
        dispatcher.process_tick(
            vp,
            prep,
            [
                ButtonTap(GamepadButton.A),
                ChordTriggered(GamepadButton.RT, GamepadButton.A),
                ButtonTap(GamepadButton.X),
                ButtonTap(GamepadButton.Y),
                ButtonTap(GamepadButton.RS),
            ],
        )
        shop0_xy = vp.norm_to_screen(*shop_mapper.slot_center(0))
        upg_pt = layout.buttons["shop_upgrade"]
        upg_xy = vp.norm_to_screen(upg_pt.u, upg_pt.v, upg_pt.anchor)
        assert ("click", shop0_xy[0], shop0_xy[1], "left") in hid.calls
        assert ("double_click", shop0_xy[0], shop0_xy[1]) in hid.calls
        assert ("tap_key", "d") in hid.calls
        assert ("click", upg_xy[0], upg_xy[1], "left") in hid.calls
        assert ("tap_key", "r") in hid.calls

        # Start 长按进度条与防误触开战 Space
        hid.clear()
        snap_prog = dispatcher.process_tick(
            vp, prep, [ButtonHoldProgress(GamepadButton.START, 0.65)]
        )
        assert snap_prog.ready_hold_progress == pytest.approx(0.65)
        snap_ready = dispatcher.process_tick(
            vp, prep, [ButtonHoldTriggered(GamepadButton.START)]
        )
        assert snap_ready.ready_hold_progress == pytest.approx(0.0)
        assert ("tap_key", "space") in hid.calls

        # 局内 LT 信息速查与联机组合键
        hid.clear()
        dispatcher.process_tick(
            vp,
            prep,
            [
                ChordTriggered(GamepadButton.LT, GamepadButton.DPAD_LEFT),  # 盟约 (~)
                ChordTriggered(GamepadButton.LT, GamepadButton.DPAD_UP),  # 敌方路线 (Tab)
                ChordTriggered(GamepadButton.LT, GamepadButton.DPAD_DOWN),  # 表情 (Z)
                ChordTriggered(GamepadButton.LT, GamepadButton.RB),  # 撤退 (A)
                ChordTriggered(GamepadButton.LT, GamepadButton.START),  # 暂停菜单
            ],
        )
        pause_pt = layout.buttons["pause_menu"]
        pause_xy = vp.norm_to_screen(pause_pt.u, pause_pt.v, pause_pt.anchor)
        assert ("tap_key", "grave") in hid.calls
        assert ("tap_key", "tab") in hid.calls
        assert ("tap_key", "z") in hid.calls
        assert ("tap_key", "a") in hid.calls
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
        assert ("tap_key", "a") not in hid_safe.calls


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

        # 左移选第 1 张卡并 RT + A 极速双击拿牌
        hid.clear()
        dispatcher.process_tick(
            vp,
            modal,
            [
                ButtonTap(GamepadButton.DPAD_LEFT),
                ChordTriggered(GamepadButton.RT, GamepadButton.A),
            ],
        )
        card0_xy = vp.norm_to_screen(*choice_mapper.slot_center(0, count=3))
        assert hid.calls == [("double_click", card0_xy[0], card0_xy[1])]

    def test_full_out_of_battle_scenes_flow(self) -> None:
        """验证活动主界面 -> 模式分队 -> 战前编队（技能/模组/助战/滚轮） -> 结算页全流程。"""
        dispatcher, hid, _, keymap, layout, vp = _make_env()

        # 1. 活动主界面：X 跳编队、Y 跳奖励、Start 点击开始模拟、B 返回
        hub = SceneDetectionResult(GameScene.ACTIVITY_HUB, 0.95)
        snap_hub = dispatcher.process_tick(
            vp,
            hub,
            [
                ButtonTap(GamepadButton.DPAD_RIGHT),
                ButtonTap(GamepadButton.X),
                ButtonTap(GamepadButton.Y),
                ButtonTap(GamepadButton.START),
                ButtonTap(GamepadButton.B),
            ],
        )
        assert snap_hub.scene == GameScene.ACTIVITY_HUB
        squad_node = layout.activity_hub_nodes["squad_entry"]
        reward_node = layout.activity_hub_nodes["rewards"]
        start_pt = layout.buttons["start_action"]
        assert (
            "click",
            *vp.norm_to_screen(squad_node.u, squad_node.v),
            "left",
        ) in hid.calls
        assert (
            "click",
            *vp.norm_to_screen(reward_node.u, reward_node.v),
            "left",
        ) in hid.calls
        assert (
            "click",
            *vp.norm_to_screen(start_pt.u, start_pt.v, start_pt.anchor),
            "left",
        ) in hid.calls
        assert ("tap_key", "esc") in hid.calls

        # 2. 模式与分队准备界面
        hid.clear()
        pre_match = SceneDetectionResult(GameScene.PRE_MATCH_SETUP, 0.93)
        dispatcher.process_tick(
            vp,
            pre_match,
            [
                ButtonTap(GamepadButton.DPAD_RIGHT),
                ButtonTap(GamepadButton.A),
                ButtonTap(GamepadButton.START),
            ],
        )
        faction2 = layout.pre_match_nodes["faction_2"]
        assert ("click", *vp.norm_to_screen(faction2.u, faction2.v), "left") in hid.calls

        # 3. 战前编队：LB/RB 循环切换技能 1/2/3 与模组槽位、Y 好友助战、RS 滚轮浏览卡池、Start 确认出击
        hid.clear()
        squad_edit = SceneDetectionResult(GameScene.SQUAD_EDIT, 0.96)
        dispatcher.process_tick(
            vp,
            squad_edit,
            [
                ButtonTap(GamepadButton.RB),  # 技能 2
                ButtonTap(GamepadButton.RB),  # 技能 3
                ButtonTap(GamepadButton.RB),  # 模组
                ButtonTap(GamepadButton.Y),  # 好友助战
                StickMotion("right", 0.0, -0.9, 0.0, 48.0),  # 向下滚动卡池
                ButtonTap(GamepadButton.START),  # 确认编队出击
            ],
        )
        skill2_pt = layout.buttons["squad_skill_2"]
        skill3_pt = layout.buttons["squad_skill_3"]
        mod_pt = layout.buttons["squad_module"]
        friend_node = layout.squad_edit_nodes["friend_support"]
        assert (
            "click",
            *vp.norm_to_screen(skill2_pt.u, skill2_pt.v, skill2_pt.anchor),
            "left",
        ) in hid.calls
        assert (
            "click",
            *vp.norm_to_screen(skill3_pt.u, skill3_pt.v, skill3_pt.anchor),
            "left",
        ) in hid.calls
        assert (
            "click",
            *vp.norm_to_screen(mod_pt.u, mod_pt.v, mod_pt.anchor),
            "left",
        ) in hid.calls
        assert (
            "click",
            *vp.norm_to_screen(friend_node.u, friend_node.v),
            "left",
        ) in hid.calls
        assert ("scroll", -240, False) in hid.calls

        # 4. 结算页：继续确认与返回
        hid.clear()
        settlement = SceneDetectionResult(GameScene.SETTLEMENT, 0.94)
        dispatcher.process_tick(
            vp,
            settlement,
            [
                ButtonTap(GamepadButton.START),
                ButtonTap(GamepadButton.B),
            ],
        )
        claim_node = layout.settlement_nodes["claim_continue"]
        assert (
            "click",
            *vp.norm_to_screen(claim_node.u, claim_node.v),
            "left",
        ) in hid.calls
        assert ("tap_key", "esc") in hid.calls

        # 5. 热重载配置不报错且保持状态健康
        dispatcher.update_configs(keymap, layout)
        snap_reloaded = dispatcher.process_tick(vp, settlement, [])
        assert snap_reloaded.visible is True

    def test_combat_scene_cross_zones_and_extra_shortcuts(self) -> None:
        """补充验证作战期交互、全方向跨区跳格、预选朝向按 A 确认与 RT+Start 快捷开战。"""
        dispatcher, hid, _, keymap, layout, vp = _make_env()

        # 1. 作战期（BATTLE_COMBAT）：棋盘浏览、点击、右键关闭、LT+下 表情与 LT+X 直达 3 号队友
        combat = SceneDetectionResult(GameScene.BATTLE_COMBAT, 0.94)
        snap_c = dispatcher.process_tick(
            vp,
            combat,
            [
                ButtonTap(GamepadButton.DPAD_LEFT),
                ButtonTap(GamepadButton.A),
                ButtonTap(GamepadButton.B),
                ChordTriggered(GamepadButton.LT, GamepadButton.DPAD_DOWN),
                ChordTriggered(GamepadButton.LT, GamepadButton.X),
            ],
        )
        assert snap_c.scene == GameScene.BATTLE_COMBAT
        assert snap_c.focus_zone == FocusZone.BOARD
        assert dispatcher.selected_teammate == 3
        assert ("tap_key", "z") in hid.calls
        mate3_pt = layout.buttons["teammate_3"]
        mate3_xy = vp.norm_to_screen(mate3_pt.u, mate3_pt.v, mate3_pt.anchor)
        assert ("click", mate3_xy[0], mate3_xy[1], "left") in hid.calls

        # 2. 休整期十字键跨区链路：SHOP(下) -> BOARD(右出界) -> EQUIPMENT(左) -> BOARD(下出界) -> BENCH(下) -> SHOP
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.96, shop_open=True)
        snap_p = dispatcher.process_tick(vp, prep, [])
        assert snap_p.focus_zone == FocusZone.SHOP

        snap_board = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.DPAD_DOWN)])
        assert snap_board.focus_zone == FocusZone.BOARD
        assert snap_board.shop_open is False  # 切入棋盘自动收起商店

        # 连续向右越过棋盘右边界进入装备栏
        snap_eq = dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.DPAD_RIGHT)] * 10
        )
        assert snap_eq.focus_zone == FocusZone.EQUIPMENT

        # 装备栏向左回棋盘（row=0），再连续向下 5 次越过第 5 行进入整备区，整备区再向下回商店
        snap_back_board = dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.DPAD_LEFT)]
        )
        assert snap_back_board.focus_zone == FocusZone.BOARD
        snap_bench = dispatcher.process_tick(
            vp, prep, [ButtonTap(GamepadButton.DPAD_DOWN)] * 5
        )
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

        # RT + Start 快捷开战
        hid.clear()
        dispatcher.process_tick(
            vp, prep, [ChordTriggered(GamepadButton.RT, GamepadButton.START)]
        )
        assert ("tap_key", "space") in hid.calls


class TestLeftStickNavShopToggleAndZoomCompensation:
    """验证左摇杆替代十字键、商店自动/手动开合、三态镜头缩放补偿与横向装备栏导航。"""

    def test_left_stick_replaces_dpad_across_scenes_and_ls_toggles_cursor(self) -> None:
        dispatcher, hid, _, keymap, layout, vp = _make_env(left_stick_mode="dpad")
        ge = GestureEngine(keymap)
        hub = SceneDetectionResult(GameScene.ACTIVITY_HUB, 0.95)
        dispatcher.process_tick(vp, hub, [])
        hid.clear()

        # 活动主界面：左摇杆向右推动直接跳转至下一个菜单节点并按 A 点击
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
        snap_hub = dispatcher.process_tick(vp, hub, ev1 + [ButtonTap(GamepadButton.A)])
        assert snap_hub.free_cursor_active is False
        assert snap_hub.focus_label == layout.activity_hub_nodes["mode_coop"].label
        coop = layout.activity_hub_nodes["mode_coop"]
        assert ("click", *vp.norm_to_screen(coop.u, coop.v), "left") in hid.calls

        # 局内休整期：左摇杆在整备区跳格并跨上棋盘
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.96, shop_open=False)
        dispatcher.process_tick(vp, prep, [])
        snap_b = dispatcher.process_tick(
            vp,
            prep,
            [
                StickDirection("left", "right"),
                StickDirection("left", "right"),
            ],
        )
        assert snap_b.focus_zone == FocusZone.BENCH
        assert snap_b.focus_label == "整备区 [3/10]"

        # 按 LS 切自由光标再按 LS 恢复吸附
        snap_fc = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.LS)])
        assert snap_fc.free_cursor_active is True
        snap_snap = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.LS)])
        assert snap_snap.free_cursor_active is False

    def test_shop_auto_toggle_manual_toggle_and_zoom_view_state(self) -> None:
        dispatcher, hid, _, _, layout, vp = _make_env(auto_toggle_shop=True)
        toggle_pt = layout.buttons["shop_toggle"]
        toggle_xy = vp.norm_to_screen(toggle_pt.u, toggle_pt.v, toggle_pt.anchor)

        # 1. 进入休整期（商店关闭 -> NORMAL 视图态）
        prep_closed = SceneDetectionResult(GameScene.BATTLE_PREP, 0.95, shop_open=False)
        snap0 = dispatcher.process_tick(vp, prep_closed, [])
        assert snap0.focus_zone == FocusZone.BENCH
        assert snap0.shop_open is False
        assert snap0.view_state == LayoutViewState.NORMAL.value

        # 2. 按 LB 切入 SHOP：自动点击 shop_toggle 展开商店并切到 SHOP_OPEN 缩放视图态
        hid.clear()
        snap_shop = dispatcher.process_tick(vp, prep_closed, [ButtonTap(GamepadButton.LB)])
        assert snap_shop.focus_zone == FocusZone.SHOP
        assert snap_shop.shop_open is True
        assert snap_shop.view_state == LayoutViewState.SHOP_OPEN.value
        assert ("click", toggle_xy[0], toggle_xy[1], "left") in hid.calls

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
        assert ("click", toggle_xy[0], toggle_xy[1], "left") in hid.calls

        # 5. LT + LB 独立手动展开/收起商店
        hid.clear()
        snap_lt_open = dispatcher.process_tick(
            vp, prep_closed, [ChordTriggered(GamepadButton.LT, GamepadButton.LB)]
        )
        assert snap_lt_open.shop_open is True
        assert snap_lt_open.focus_zone == FocusZone.SHOP
        assert snap_lt_open.view_state == LayoutViewState.SHOP_OPEN.value

        # 切到棋盘时自动收起商店恢复标准棋盘坐标
        hid.clear()
        snap_board = dispatcher.process_tick(
            vp, prep_closed, [ButtonTap(GamepadButton.DPAD_DOWN)]
        )
        assert snap_board.focus_zone == FocusZone.BOARD
        assert snap_board.shop_open is False
        assert snap_board.view_state == LayoutViewState.NORMAL.value
        assert ("click", toggle_xy[0], toggle_xy[1], "left") in hid.calls

    def test_unit_selected_zoom_compensation_and_clean_drag(self) -> None:
        dispatcher, hid, _, _, layout, vp = _make_env()
        normal_board = PerspectiveBoardMapper(layout.resolve_board_quad(LayoutViewState.NORMAL))
        selected_board = PerspectiveBoardMapper(
            layout.resolve_board_quad(LayoutViewState.UNIT_SELECTED)
        )
        prep = SceneDetectionResult(GameScene.BATTLE_PREP, 0.96, shop_open=False)
        dispatcher.process_tick(vp, prep, [])

        # 切到棋盘中心格：先确认处于 NORMAL 坐标
        snap_norm = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.RB)])
        assert snap_norm.focus_zone == FocusZone.BOARD
        assert snap_norm.view_state == LayoutViewState.NORMAL.value
        assert snap_norm.cursor_norm == pytest.approx(
            normal_board.cell_center(normal_board.rows // 2, normal_board.cols // 2)
        )

        # 按 RT+A 单击查看角色详情 -> 切换为 UNIT_SELECTED 缩放补偿坐标，框选自动跟随偏移
        hid.clear()
        snap_sel = dispatcher.process_tick(
            vp, prep, [ChordTriggered(GamepadButton.RT, GamepadButton.A)]
        )
        assert snap_sel.view_state == LayoutViewState.UNIT_SELECTED.value
        assert snap_sel.cursor_norm == pytest.approx(
            selected_board.cell_center(selected_board.rows // 2, selected_board.cols // 2)
        )
        assert snap_sel.cursor_norm != pytest.approx(snap_norm.cursor_norm)
        assert hid.calls[0][0] == "click"

        # 按 B 退出角色详情，立即恢复 NORMAL 坐标
        hid.clear()
        snap_back = dispatcher.process_tick(vp, prep, [ButtonTap(GamepadButton.B)])
        assert snap_back.view_state == LayoutViewState.NORMAL.value
        assert snap_back.cursor_norm == pytest.approx(snap_norm.cursor_norm)
        assert hid.calls[0][3] == "right"

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
