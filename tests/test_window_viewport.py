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
        loader = ConfigLoader()
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

        for key in REQUIRED_BUTTON_KEYS:
            assert key in layout.buttons
            pt = layout.buttons[key]
            assert 0.0 <= pt.u <= 1.0
            assert 0.0 <= pt.v <= 1.0

        assert layout.bench_zone.count == 10
        assert layout.equip_zone.count == 5
        assert layout.choice_zone.count == 3
        assert layout.board_quad.rows == 5
        assert layout.board_quad.cols == 9
        assert len(layout.activity_hub_nodes) >= 5
        assert len(layout.pre_match_nodes) >= 4
        assert len(layout.squad_edit_nodes) >= 12
        assert len(layout.settlement_nodes) >= 2

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
        # 未覆盖的顶点保持默认值
        assert layout.board_quad.top_right == pytest.approx((0.775, 0.285))

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
