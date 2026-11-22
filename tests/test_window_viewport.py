"""窗口视口坐标双向解算、窗口追踪优先级、配置系统与 UAC 单元测试。"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

import pytest

from arkcontroller.config.loader import ConfigLoader
from arkcontroller.config.models import (
    REQUIRED_BUTTON_KEYS,
    KeymapConfig,
    LayoutProfile,
    LinearZone,
    NavNode,
    NormalizedPoint,
    NormalizedRect,
    PerspectiveQuad,
    SymmetricCovenantZone,
)
from arkcontroller.platform import uac
from arkcontroller.platform.window_tracker import ViewportRect, WindowTracker


class _FakeWindowBackend:
    """用于离线模拟多窗口/模拟器层级与前台状态的测试桩后端。"""

    def __init__(self) -> None:
        self.top_windows: list[int] = []
        self.children: dict[int, list[int]] = {}
        self.titles: dict[int, str] = {}
        self.classes: dict[int, str] = {}
        self.pids: dict[int, int] = {}
        self.proc_names: dict[int, str] = {}
        self.rects: dict[int, tuple[int, int, int, int]] = {}
        self.visible: set[int] = set()
        self.iconic: set[int] = set()
        self.roots: dict[int, int] = {}
        self.foreground_hwnd: int = 0

    def add_window(
        self,
        hwnd: int,
        *,
        title: str,
        class_name: str,
        pid: int,
        proc_name: str,
        rect: tuple[int, int, int, int],
        parent: int | None = None,
        visible: bool = True,
        iconic: bool = False,
    ) -> None:
        self.titles[hwnd] = title
        self.classes[hwnd] = class_name
        self.pids[hwnd] = pid
        self.proc_names[pid] = proc_name
        self.rects[hwnd] = rect
        if visible:
            self.visible.add(hwnd)
        if iconic:
            self.iconic.add(hwnd)
        if parent is None:
            self.top_windows.append(hwnd)
            self.roots[hwnd] = hwnd
        else:
            self.children.setdefault(parent, []).append(hwnd)
            self.roots[hwnd] = self.roots.get(parent, parent)

    def is_window_valid(self, hwnd: int) -> bool:
        return hwnd in self.rects

    def is_window_visible(self, hwnd: int) -> bool:
        return hwnd in self.visible

    def is_iconic(self, hwnd: int) -> bool:
        return hwnd in self.iconic

    def get_foreground_window(self) -> int:
        return self.foreground_hwnd

    def get_root_window(self, hwnd: int) -> int:
        return self.roots.get(hwnd, hwnd)

    def get_window_title(self, hwnd: int) -> str:
        return self.titles.get(hwnd, "")

    def get_class_name(self, hwnd: int) -> str:
        return self.classes.get(hwnd, "")

    def get_window_pid(self, hwnd: int) -> int:
        return self.pids.get(hwnd, 0)

    def get_process_name(self, pid: int) -> str:
        return self.proc_names.get(pid, "")

    def get_client_rect_screen(self, hwnd: int) -> tuple[int, int, int, int] | None:
        return self.rects.get(hwnd)

    def enum_top_windows(self) -> list[int]:
        return list(self.top_windows)

    def enum_child_windows(self, parent_hwnd: int) -> list[int]:
        return list(self.children.get(parent_hwnd, []))


class TestViewportRect:
    """验证 16:9、16:10、21:9 及带偏移窗口的坐标映射与黑边裁剪。"""

    def test_standard_16_9_1080p(self) -> None:
        vp = ViewportRect.from_client_rect(
            hwnd=1001,
            client_left=0,
            client_top=0,
            client_width=1920,
            client_height=1080,
            is_foreground=True,
            window_title="明日方舟",
        )
        assert (vp.active_left, vp.active_top, vp.active_width, vp.active_height) == (
            0,
            0,
            1920,
            1080,
        )
        assert vp.norm_to_screen(0.5, 0.5) == (960, 540)
        assert vp.norm_to_screen(0.0, 0.0) == (0, 0)
        assert vp.norm_to_screen(1.0, 1.0) == (1920, 1080)
        assert vp.norm_to_screen(0.05, 0.06, anchor="left_edge") == vp.norm_to_screen(0.05, 0.06)
        assert vp.norm_to_screen(0.95, 0.06, anchor="right_edge") == vp.norm_to_screen(0.95, 0.06)

        u, v = vp.screen_to_norm(960, 540)
        assert u == pytest.approx(0.5, abs=1e-6)
        assert v == pytest.approx(0.5, abs=1e-6)

        assert vp.norm_rect_to_local(0.25, 0.5, 0.5, 0.25) == (480, 540, 960, 270)

    def test_16_10_letterbox_2560x1600(self) -> None:
        vp = ViewportRect.from_client_rect(
            hwnd=1002,
            client_left=0,
            client_top=0,
            client_width=2560,
            client_height=1600,
        )
        # 2560 / (16/9) = 1440，上下各 80px 黑边
        assert (vp.active_left, vp.active_top, vp.active_width, vp.active_height) == (
            0,
            80,
            2560,
            1440,
        )
        assert vp.norm_to_screen(0.5, 0.0) == (1280, 80)
        assert vp.norm_to_screen(0.5, 0.5) == (1280, 800)
        assert vp.norm_to_screen(0.5, 1.0) == (1280, 1520)
        assert vp.norm_to_screen(0.5, 0.0, anchor="top_edge") == (1280, 0)
        assert vp.norm_to_screen(0.5, 1.0, anchor="bottom_edge") == (1280, 1600)

        sx, sy = vp.norm_to_screen(0.25, 0.75)
        u, v = vp.screen_to_norm(sx, sy)
        assert u == pytest.approx(0.25, abs=1e-3)
        assert v == pytest.approx(0.75, abs=1e-3)

    def test_ultrawide_21_9_3440x1440(self) -> None:
        vp = ViewportRect.from_client_rect(
            hwnd=1003,
            client_left=0,
            client_top=0,
            client_width=3440,
            client_height=1440,
        )
        # 1440 * 16 / 9 = 2560，左右各 440px 扩展区/黑边
        assert (vp.active_left, vp.active_top, vp.active_width, vp.active_height) == (
            440,
            0,
            2560,
            1440,
        )
        assert vp.norm_to_screen(0.0, 0.5, anchor="center_16_9") == (440, 720)
        assert vp.norm_to_screen(1.0, 0.5, anchor="center_16_9") == (3000, 720)
        # 贴边锚点应相对整个 3440 宽客户区左右边界对齐
        assert vp.norm_to_screen(0.05, 0.1, anchor="left_edge") == (int(round(0.05 * 2560)), 144)
        assert vp.norm_to_screen(0.95, 0.1, anchor="right_edge") == (
            int(round(3440 - 0.05 * 2560)),
            144,
        )
        assert vp.norm_to_screen(0.5, 0.5, anchor="client") == (1720, 720)

        # 黑边区域 screen_to_norm 正确反映负值或大于 1 的相对坐标
        u_left, _ = vp.screen_to_norm(0, 720)
        u_right, _ = vp.screen_to_norm(3440, 720)
        assert u_left < 0.0
        assert u_right > 1.0

    def test_emulator_offset_render_subwindow(self) -> None:
        # 模拟扣除模拟器顶部 36px 标题栏与左侧 120px 偏移后的 1280x720 渲染子窗口
        vp = ViewportRect.from_client_rect(
            hwnd=2002,
            client_left=120,
            client_top=36,
            client_width=1280,
            client_height=720,
            window_title="MuMu模拟器12 - 明日方舟",
        )
        assert (vp.active_left, vp.active_top, vp.active_width, vp.active_height) == (
            120,
            36,
            1280,
            720,
        )
        sx, sy = vp.norm_to_screen(0.125, 0.895)
        assert (sx, sy) == (120 + 160, 36 + int(round(0.895 * 720)))
        u, v = vp.screen_to_norm(sx, sy)
        assert u == pytest.approx(0.125, abs=1e-3)
        assert v == pytest.approx(0.895, abs=2e-3)

    def test_viewport_is_frozen_and_validates_anchor(self) -> None:
        vp = ViewportRect.from_client_rect(1, 0, 0, 1920, 1080)
        with pytest.raises(FrozenInstanceError):
            vp.hwnd = 2  # type: ignore[misc]
        with pytest.raises(ValueError, match="不支持的锚点类型"):
            vp.norm_to_screen(0.5, 0.5, anchor="invalid_anchor")

    def test_zero_size_viewport_safe(self) -> None:
        vp = ViewportRect.from_client_rect(1, 10, 20, 0, 0)
        assert vp.screen_to_norm(10, 20) == (0.0, 0.0)
        assert vp.norm_rect_to_local(0.1, 0.2, 0.3, 0.4) == (0, 0, 0, 0)


class TestWindowTracker:
    """验证官方 PC 端优先匹配、模拟器渲染子窗口下钻与前台焦点判定。"""

    def test_prefers_official_pc_client_over_emulator(self) -> None:
        backend = _FakeWindowBackend()
        backend.add_window(
            100,
            title="MuMu模拟器 - 明日方舟",
            class_name="Qt5156QWindowIcon",
            pid=2001,
            proc_name="mumuplayer.exe",
            rect=(0, 0, 1660, 940),
        )
        backend.add_window(
            101,
            title="nemuwin",
            class_name="nemuwin",
            pid=2001,
            proc_name="mumuplayer.exe",
            rect=(10, 40, 1600, 900),
            parent=100,
        )
        backend.add_window(
            200,
            title="明日方舟",
            class_name="UnityWndClass",
            pid=2002,
            proc_name="arknights.exe",
            rect=(0, 0, 1920, 1080),
        )
        backend.foreground_hwnd = 200

        tracker = WindowTracker(backend=backend)
        vp = tracker.update()
        assert vp is not None
        assert vp.hwnd == 200
        assert vp.active_width == 1920
        assert vp.is_foreground is True

    def test_mumu_emulator_drills_into_nemuwin_and_tracks_root_foreground(self) -> None:
        backend = _FakeWindowBackend()
        backend.add_window(
            300,
            title="MuMu模拟器12",
            class_name="Qt5156QWindowIcon",
            pid=3001,
            proc_name="mumuplayer.exe",
            rect=(50, 20, 1340, 760),
        )
        backend.add_window(
            301,
            title="nemuwin",
            class_name="nemuwin",
            pid=3001,
            proc_name="mumuplayer.exe",
            rect=(50, 56, 1280, 720),
            parent=300,
        )
        # Windows 前台焦点在顶层主窗口 300 上，子窗口 301 仍应判定为前台
        backend.foreground_hwnd = 300

        tracker = WindowTracker(backend=backend)
        vp = tracker.update()
        assert vp is not None
        assert vp.hwnd == 301
        assert (vp.client_left, vp.client_top, vp.client_width, vp.client_height) == (
            50,
            56,
            1280,
            720,
        )
        assert vp.is_foreground is True

        # 切换到其他应用后 is_foreground 变为 False
        backend.foreground_hwnd = 9999
        vp_bg = tracker.update()
        assert vp_bg is not None
        assert vp_bg.is_foreground is False

    def test_ldplayer_emulator_drills_into_therender(self) -> None:
        backend = _FakeWindowBackend()
        backend.add_window(
            400,
            title="雷电模拟器 - 明日方舟",
            class_name="LDPlayerMainFrame",
            pid=4001,
            proc_name="dnplayer.exe",
            rect=(100, 100, 1340, 770),
        )
        backend.add_window(
            401,
            title="TheRender",
            class_name="TheRender",
            pid=4001,
            proc_name="dnplayer.exe",
            rect=(102, 136, 1280, 720),
            parent=400,
        )
        backend.foreground_hwnd = 400

        tracker = WindowTracker(backend=backend)
        vp = tracker.update()
        assert vp is not None
        assert vp.hwnd == 401
        assert vp.active_width == 1280
        assert vp.active_height == 720
        assert vp.is_foreground is True


class TestConfigSystem:
    """验证默认配置加载、模型校验以及 user_override.json 合并与热重载。"""

    def test_load_default_configs_from_repo(self) -> None:
        loader = ConfigLoader(use_user_override=False)
        keymap, layout = loader.load()

        assert isinstance(keymap, KeymapConfig)
        assert isinstance(layout, LayoutProfile)
        assert keymap.left_deadzone == pytest.approx(0.14)
        assert keymap.right_deadzone == pytest.approx(0.16)
        assert keymap.cursor_max_speed_px == pytest.approx(1400.0)
        assert keymap.cursor_gamma == pytest.approx(2.2)
        assert keymap.ready_hold_ms == 350
        assert keymap.repeat_initial_ms == 260
        assert keymap.repeat_interval_ms == 95
        assert keymap.use_native_keyboard_shortcuts is True
        assert keymap.flick_distance_ratio == pytest.approx(0.09)
        assert keymap.vibration_enabled is True
        assert keymap.left_stick_mode == "dpad"
        assert keymap.auto_toggle_shop is True
        assert keymap.hide_hud_from_capture is True
        assert keymap.to_dict()["hide_hud_from_capture"] is True
        assert KeymapConfig.from_dict({}).hide_hud_from_capture is True
        assert KeymapConfig.from_dict({"hide_hud_from_capture": False}).hide_hud_from_capture is False
        assert keymap.keyboard_shortcuts["shop_upgrade"] == "g"
        assert keymap.keyboard_shortcuts == {
            "shop_toggle": "a",
            "shop_freeze": "s",
            "shop_refresh": "d",
            "shop_upgrade": "g",
            "battle_ready": "c",
            "enemy_preview": "w",
            "sell_board": "x",
            "sell_bench": "x",
            "retreat_unit": "q",
            "covenant_panel": "",
            "emote_wheel": "",
            "back": "esc",
            "pause_menu": "esc",
        }
        assert "menu" not in keymap.bindings
        assert keymap.bindings["modal_confirm"] == {
            "confirm": "A",
            "cancel": "B",
            "quick_confirm": "START",
            "toggle_dont_ask": "X",
        }

        for key in REQUIRED_BUTTON_KEYS:
            assert key in layout.buttons
            pt = layout.buttons[key]
            assert 0.0 <= pt.u <= 1.0
            assert 0.0 <= pt.v <= 1.0

        # 验证已删除所有局外按钮与局外节点表
        for removed_btn in (
            "start_action",
            "squad_skill_1",
            "squad_skill_2",
            "squad_skill_3",
            "squad_module",
        ):
            assert removed_btn not in layout.buttons
        for removed_attr in (
            "activity_hub_nodes",
            "pre_match_nodes",
            "squad_edit_nodes",
            "settlement_nodes",
        ):
            assert not hasattr(layout, removed_attr)
            assert removed_attr not in layout.to_dict()

        # 验证新增策略头像、交流面板与回合确认弹窗实测按钮锚点
        assert layout.buttons["strategy_avatar"] == NormalizedPoint(0.9605, 0.3243, "right_edge")
        assert layout.buttons["strategy_inspect"] == NormalizedPoint(0.9047, 0.3236, "right_edge")
        assert layout.buttons["confirm_no"] == NormalizedPoint(0.2496, 0.6729, "center_16_9")
        assert layout.buttons["confirm_yes"] == NormalizedPoint(0.7496, 0.6729, "center_16_9")
        assert layout.buttons["confirm_dont_ask"] == NormalizedPoint(0.4992, 0.5597, "center_16_9")
        assert layout.buttons["emote_wheel"] == NormalizedPoint(0.0684, 0.9181, "left_edge")
        assert layout.buttons["emote_panel_center"] == NormalizedPoint(0.1290, 0.7400, "left_edge")
        assert layout.buttons["choice_toggle_eye_6"] == NormalizedPoint(0.968, 0.500, "right_edge")

        assert "shop_toggle" in layout.buttons
        assert "safe_blank" in layout.buttons
        assert "choice_toggle_eye_6" in layout.buttons
        # 验证安全空白区已移至右侧策略头像正下方安全带，绝不落入底部商店卡牌区
        assert layout.buttons["safe_blank"].u >= 0.90
        assert 0.39 <= layout.buttons["safe_blank"].v <= 0.50
        # 验证二选一、三选一与多人六选一卡片及底部白色确认条几何关系
        assert layout.choice_2_zone.count == 2
        assert "choice_2_zone" in layout.to_dict()
        c2_left = layout.resolve_choice_card_center(0, 2)
        c2_right = layout.resolve_choice_card_center(1, 2)
        assert c2_left == pytest.approx((0.3809, 0.5000))
        assert c2_right == pytest.approx((0.6477, 0.5000))
        assert layout.resolve_choice_card_size(0, 2) == pytest.approx((0.2188, 0.3306))
        c2_confirm_right = layout.resolve_choice_confirm_point(1, 2)
        assert c2_confirm_right[0] == pytest.approx(0.6477)
        assert c2_confirm_right[1] == pytest.approx(0.666953, abs=1e-4)
        c2_poly = layout.resolve_choice_card_polygon(0, 2)
        assert len(c2_poly) == 4
        assert c2_poly[0] == pytest.approx((0.3809 - 0.1094, 0.5000 - 0.1653))
        assert c2_poly[2] == pytest.approx((0.3809 + 0.1094, 0.5000 + 0.1653))
        c2_confirm_poly = layout.resolve_choice_confirm_polygon(1, 2)
        assert len(c2_confirm_poly) == 4
        assert (c2_confirm_poly[0][0] + c2_confirm_poly[2][0]) * 0.5 == pytest.approx(0.6477)
        assert (c2_confirm_poly[0][1] + c2_confirm_poly[2][1]) * 0.5 == pytest.approx(
            c2_confirm_right[1]
        )
        assert layout.resolve_choice_toggle_point(2) == layout.buttons["choice_toggle_eye"]

        c3_center = layout.resolve_choice_card_center(0, 3)
        c3_confirm = layout.resolve_choice_confirm_point(0, 3)
        assert c3_confirm[0] == pytest.approx(c3_center[0])
        assert c3_confirm[1] == pytest.approx(0.674, abs=0.006)
        # 官方 PC 16:9 六选一三列对齐与上下排白条确认点
        assert layout.resolve_choice_card_center(0, 6) == pytest.approx((0.2473, 0.2931))
        assert layout.resolve_choice_card_center(2, 6) == pytest.approx((0.7816, 0.2931))
        assert layout.resolve_choice_card_center(3, 6) == pytest.approx((0.2473, 0.7083))
        assert layout.resolve_choice_card_center(5, 6) == pytest.approx((0.7816, 0.7083))
        assert layout.resolve_choice_card_size(0, 6) == pytest.approx((0.2180, 0.3306))
        c6_top_confirm = layout.resolve_choice_confirm_point(1, 6)
        c6_bot_confirm = layout.resolve_choice_confirm_point(4, 6)
        assert c6_top_confirm[0] == pytest.approx(c6_bot_confirm[0])
        assert c6_top_confirm[1] == pytest.approx(0.4601, abs=0.003)
        assert c6_bot_confirm[1] == pytest.approx(0.8753, abs=0.003)
        assert layout.resolve_choice_toggle_point(3) == layout.buttons["choice_toggle_eye"]
        assert layout.resolve_choice_toggle_point(6) == layout.buttons["choice_toggle_eye_6"]

        # 验证左下 2x3 表情面板几何解析 (0..5) 与序列化
        assert layout.emote_top_zone.count == 3
        assert layout.emote_bottom_zone.count == 3
        assert "emote_top_zone" in layout.to_dict()
        assert "emote_bottom_zone" in layout.to_dict()
        expected_emote_centers = [
            (0.0582, 0.6771),
            (0.1289, 0.6771),
            (0.1996, 0.6771),
            (0.0582, 0.8016),
            (0.1289, 0.8016),
            (0.1996, 0.8016),
        ]
        for idx, exp in enumerate(expected_emote_centers):
            cu, cv = layout.resolve_emote_slot_center(idx)
            assert (cu, cv) == pytest.approx(exp, abs=1e-4)
            poly = layout.resolve_emote_slot_polygon(idx)
            assert len(poly) == 4
            assert poly[0] == pytest.approx((cu - 0.032, cv - 0.056), abs=1e-4)
            assert poly[2] == pytest.approx((cu + 0.032, cv + 0.056), abs=1e-4)
        # 验证顶部下方居中对称羁绊坐标解算（奇数、偶数、7 枚满屏与 >7 溢出左对齐）
        assert isinstance(layout.covenant_zone, SymmetricCovenantZone)
        assert layout.covenant_zone.v == pytest.approx(0.1208)
        assert layout.covenant_zone.step_u == pytest.approx(0.0625)
        assert layout.covenant_zone.overflow_start_u == pytest.approx(0.2961)
        assert layout.covenant_zone.to_dict()["overflow_start_u"] == pytest.approx(0.2961)
        assert layout.covenant_zone.slot_center(0, count=1) == pytest.approx((0.50, 0.1208))
        u_even_0, _ = layout.covenant_zone.slot_center(0, count=2)
        u_even_1, _ = layout.covenant_zone.slot_center(1, count=2)
        assert (u_even_0 + u_even_1) * 0.5 == pytest.approx(0.50)
        assert u_even_1 - u_even_0 == pytest.approx(0.0625)
        # 7 枚居中满屏：0.50 - 3 * 0.0625 = 0.3125 .. 0.6875
        assert layout.covenant_zone.slot_center(0, count=7) == pytest.approx((0.3125, 0.1208))
        assert layout.covenant_zone.slot_center(3, count=7) == pytest.approx((0.5000, 0.1208))
        assert layout.covenant_zone.slot_center(6, count=7) == pytest.approx((0.6875, 0.1208))
        # >7 溢出左对齐 7 槽：overflow_start_u + i * step_u
        for i in range(7):
            exp_u = 0.2961 + i * 0.0625
            cu, cv = layout.covenant_zone.slot_center(i, count=7, overflow=True)
            assert (cu, cv) == pytest.approx((exp_u, 0.1208), abs=1e-6)
            poly = layout.covenant_zone.slot_polygon(i, count=7, overflow=True)
            assert len(poly) == 4
            assert (poly[0][0] + poly[2][0]) * 0.5 == pytest.approx(exp_u, abs=1e-6)
            assert (poly[0][1] + poly[2][1]) * 0.5 == pytest.approx(0.1208, abs=1e-6)
        # 旧版缺少 overflow_start_u 的 JSON 字典向后兼容
        legacy_cov = SymmetricCovenantZone.from_dict({"center_u": 0.5, "v": 0.12, "step_u": 0.0625})
        assert legacy_cov.overflow_start_u == pytest.approx(0.2961)
        assert "shop_open" in layout.view_overrides
        assert "unit_selected" in layout.view_overrides
        assert layout.resolve_board_quad("shop_open") != layout.board_quad
        assert layout.resolve_bench_zone("shop_open") != layout.bench_zone
        assert layout.resolve_equip_zone("shop_open") != layout.equip_zone

        assert layout.shop_zone.count == 4
        assert layout.bench_zone.count == 10
        assert layout.equip_zone.count == 5
        assert layout.choice_zone.count == 3
        assert layout.board_quad.rows == 4
        assert layout.board_quad.cols == 7

        # 验证商店升至二本(5槽) / 四本(6槽)时间距与尺寸不变、向左侧延伸且升级按钮同步左移
        z4 = layout.resolve_shop_zone("shop_open", slot_count=4)
        z5 = layout.resolve_shop_zone("shop_open", slot_count=5)
        z6 = layout.resolve_shop_zone("shop_open", slot_count=6)
        step4 = (z4.end_u - z4.start_u) / 3
        assert z5.count == 5 and z6.count == 6
        assert z5.end_u == pytest.approx(z4.end_u)
        assert z6.end_u == pytest.approx(z4.end_u)
        assert z5.slot_w == pytest.approx(z4.slot_w)
        assert z6.slot_h == pytest.approx(z4.slot_h)
        assert z5.start_u == pytest.approx(z4.start_u - step4)
        assert z6.start_u == pytest.approx(z4.start_u - 2 * step4)
        # 扩展后右侧原有槽位中心严格重合
        for i in range(4):
            assert z5.slot_center(i + 1) == pytest.approx(z4.slot_center(i))
            assert z6.slot_center(i + 2) == pytest.approx(z4.slot_center(i))
        # 升级按钮同步左移相同步距
        upg4 = layout.resolve_shop_upgrade_point("shop_open", slot_count=4)
        upg5 = layout.resolve_shop_upgrade_point("shop_open", slot_count=5)
        upg6 = layout.resolve_shop_upgrade_point("shop_open", slot_count=6)
        assert upg4 is not None and upg5 is not None and upg6 is not None
        assert upg5.u == pytest.approx(upg4.u - step4)
        assert upg6.u == pytest.approx(upg4.u - 2 * step4)

    def test_layout_from_dict_ignores_legacy_nodes_and_fills_default_new_buttons(
        self, tmp_path: Path
    ) -> None:
        repo_configs = Path(__file__).resolve().parents[1] / "configs"
        raw_layout = json.loads(
            (repo_configs / "stronghold_layout_16_9.json").read_text(encoding="utf-8")
        )
        # 模拟旧配置字典：残留局外节点键且缺少 choice_2_zone 与新增的 5 个策略/确认弹窗按钮
        raw_layout["activity_hub_nodes"] = {
            "back": {"id": "back", "label": "返回", "u": 0.05, "v": 0.06}
        }
        raw_layout["pre_match_nodes"] = {}
        raw_layout["squad_edit_nodes"] = {}
        raw_layout["settlement_nodes"] = {}
        raw_layout.pop("choice_2_zone", None)
        raw_layout.pop("emote_top_zone", None)
        raw_layout.pop("emote_bottom_zone", None)
        for new_key in (
            "strategy_avatar",
            "strategy_inspect",
            "confirm_no",
            "confirm_yes",
            "confirm_dont_ask",
            "emote_panel_center",
        ):
            raw_layout["buttons"].pop(new_key, None)

        layout = LayoutProfile.from_dict(raw_layout)
        assert layout.choice_2_zone.start_u == pytest.approx(0.3809)
        assert layout.choice_2_zone.end_u == pytest.approx(0.6477)
        assert layout.resolve_choice_card_center(0, 2) == pytest.approx((0.3809, 0.5000))
        assert layout.resolve_choice_card_center(1, 2) == pytest.approx((0.6477, 0.5000))
        assert layout.resolve_emote_slot_center(0) == pytest.approx((0.0582, 0.6771))
        assert layout.resolve_emote_slot_center(5) == pytest.approx((0.1996, 0.8016))
        assert layout.buttons["emote_panel_center"] == NormalizedPoint(0.1290, 0.7400, "left_edge")
        assert layout.buttons["strategy_avatar"] == NormalizedPoint(0.9605, 0.3243, "right_edge")
        assert layout.buttons["confirm_yes"] == NormalizedPoint(0.7496, 0.6729, "center_16_9")

        # 模拟旧 user_override.json 残留局外节点键时 ConfigLoader 正常加载不报错
        (tmp_path / "default_bindings.json").write_text(
            (repo_configs / "default_bindings.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (tmp_path / "stronghold_layout_16_9.json").write_text(
            (repo_configs / "stronghold_layout_16_9.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (tmp_path / "user_override.json").write_text(
            json.dumps(
                {
                    "layout": {
                        "activity_hub_nodes": {"back": {"id": "back", "u": 0.1, "v": 0.1}},
                        "settlement_nodes": {},
                    }
                }
            ),
            encoding="utf-8",
        )
        loader = ConfigLoader(config_dir=tmp_path)
        _, loaded_layout = loader.load()
        assert loaded_layout.buttons["confirm_dont_ask"] == NormalizedPoint(
            0.4992, 0.5597, "center_16_9"
        )

    def test_save_user_override_merge_and_reload(self, tmp_path: Path) -> None:
        repo_configs = Path(__file__).resolve().parents[1] / "configs"
        (tmp_path / "default_bindings.json").write_text(
            (repo_configs / "default_bindings.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (tmp_path / "stronghold_layout_16_9.json").write_text(
            (repo_configs / "stronghold_layout_16_9.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        loader = ConfigLoader(config_dir=tmp_path)
        assert loader.keymap_path.name == "default_bindings.json"
        override_path = loader.save_user_override(
            {
                "keymap": {"ready_hold_ms": 500, "use_native_keyboard_shortcuts": False},
                "layout": {
                    "board_quad": {
                        "top_left": [0.21, 0.27],
                        "rows": 5,
                        "cols": 9,
                    }
                },
            }
        )
        assert override_path == tmp_path / "user_override.json"
        assert override_path.exists()

        # 第二次增量保存（使用扁平格式）不应覆盖前一次已保存的字段
        loader.save_user_override({"left_deadzone": 0.18})

        keymap, layout = loader.reload()
        assert keymap.ready_hold_ms == 500
        assert keymap.use_native_keyboard_shortcuts is False
        assert keymap.left_deadzone == pytest.approx(0.18)
        assert layout.board_quad.top_left == pytest.approx((0.21, 0.27))
        # 未覆盖的顶点保持实机校准后的默认值
        assert layout.board_quad.top_right == pytest.approx((0.76823, 0.28519))

        raw_override = json.loads(override_path.read_text(encoding="utf-8"))
        assert raw_override["keymap"]["ready_hold_ms"] == 500
        assert raw_override["keymap"]["left_deadzone"] == 0.18

    def test_legacy_keymap_filename_fallback(self, tmp_path: Path) -> None:
        repo_configs = Path(__file__).resolve().parents[1] / "configs"
        (tmp_path / "default_keymap.json").write_text(
            (repo_configs / "default_bindings.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (tmp_path / "stronghold_layout_16_9.json").write_text(
            (repo_configs / "stronghold_layout_16_9.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        loader = ConfigLoader(config_dir=tmp_path)
        assert loader.keymap_path.name == "default_keymap.json"
        keymap, _ = loader.load()
        assert keymap.ready_hold_ms == 350

    def test_model_validation_guards(self) -> None:
        with pytest.raises(ValueError):
            NormalizedPoint(u=1.2, v=0.5)
        with pytest.raises(ValueError):
            NormalizedPoint(u=0.5, v=0.5, anchor="unknown")
        with pytest.raises(ValueError):
            NormalizedRect(u=0.1, v=0.1, w=0.0, h=0.2)
        with pytest.raises(ValueError):
            PerspectiveQuad((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9), rows=0, cols=9)
        with pytest.raises(ValueError):
            LinearZone("empty", 0, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1)
        with pytest.raises(ValueError):
            NavNode(id="", label="空", u=0.5, v=0.5)
        with pytest.raises(ValueError):
            KeymapConfig(left_deadzone=0.99)


class TestPlatformUAC:
    """验证 DPI 感知与 UAC 辅助函数的正常分支与回退分支。"""

    def test_enable_dpi_awareness_v2_and_fallback(self) -> None:
        fake_user32 = mock.MagicMock()
        fake_user32.SetProcessDpiAwarenessContext.return_value = 1

        with (
            mock.patch.object(uac.sys, "platform", "win32"),
            mock.patch.object(uac.ctypes, "WinDLL", return_value=fake_user32),
        ):
            assert uac.enable_per_monitor_dpi_awareness() is True

        # 当 SetProcessDpiAwarenessContext 失败且错误码非 ACCESS_DENIED 时回退至 Shcore
        fake_user32_fail = mock.MagicMock()
        fake_user32_fail.SetProcessDpiAwarenessContext.return_value = 0
        fake_shcore = mock.MagicMock()
        fake_shcore.SetProcessDpiAwareness.return_value = 0

        def _fake_windll(name: str, use_last_error: bool = False) -> mock.MagicMock:
            return fake_user32_fail if name == "user32" else fake_shcore

        with (
            mock.patch.object(uac.sys, "platform", "win32"),
            mock.patch.object(uac.ctypes, "WinDLL", side_effect=_fake_windll),
            mock.patch.object(uac.ctypes, "get_last_error", return_value=0),
        ):
            assert uac.enable_per_monitor_dpi_awareness() is True
            fake_shcore.SetProcessDpiAwareness.assert_called_once_with(
                uac.PROCESS_PER_MONITOR_DPI_AWARE
            )

    def test_relaunch_as_admin_handles_user_cancel(self) -> None:
        fake_shell32 = mock.MagicMock()
        fake_shell32.ShellExecuteW.return_value = 5  # 用户取消 UAC (SE_ERR_ACCESSDENIED)

        with (
            mock.patch.object(uac.sys, "platform", "win32"),
            mock.patch.object(uac, "is_admin", return_value=False),
            mock.patch.object(uac.ctypes, "WinDLL", return_value=fake_shell32),
        ):
            assert uac.relaunch_as_admin(["--calibrate"]) is False

        fake_shell32.ShellExecuteW.return_value = 42  # > 32 表示成功
        with (
            mock.patch.object(uac.sys, "platform", "win32"),
            mock.patch.object(uac, "is_admin", return_value=False),
            mock.patch.object(uac.ctypes, "WinDLL", return_value=fake_shell32),
        ):
            assert uac.relaunch_as_admin(["--calibrate"]) is True
