"""罗德岛 PRTS 风格 HUD 重绘与桌面主控台前端的离屏渲染与交互单元测试。

全部用例运行于 ``QT_QPA_PLATFORM=offscreen`` 无头平台，覆盖：
    - 透明穿透 HUD 的导出符号契约、穿透窗口属性与全元素离屏绘制；
    - PRTSDashboardWindow 的普通窗口层级、关闭隐藏至托盘、~16Hz 可见节流遥测、开关/调校信号双向联动；
    - 托盘「打开 PRTS 主控台」入口与单击/双击唤出；
    - main.py 装配：--minimized 参数、参数热保存至 user_override.json 与运行态热刷新；
    - 全程零 ``QFont::setPointSize`` 警告。
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path
import shutil

import pytest
from PySide6.QtCore import QRectF, Qt, QtMsgType, qInstallMessageHandler
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication, QSystemTrayIcon

from arkcontroller.config.models import KeymapConfig
from arkcontroller.control.deploy_fsm import DeployState
from arkcontroller.control.dispatcher import FocusZone, HUDStateSnapshot, PromptItem
from arkcontroller.input.xinput_driver import GamepadButton, GamepadState
from arkcontroller.overlay import hud_window
from arkcontroller.overlay.dashboard_window import (
    KEYMAP_BRIEFS,
    TUNING_SPECS,
    DashboardRuntimeStatus,
    PRTSDashboardWindow,
)
from arkcontroller.overlay.hud_window import (
    COLOR_AMBER,
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_RED,
    SCENE_DISPLAY_NAMES,
    HUDWindow,
    align_widget_to_physical_rect,
    apply_win32_click_through,
    chamfer_polygon,
)
from arkcontroller.overlay.tray import ControllerTray, create_prts_icon
from arkcontroller.vision.scene_detector import GameScene

REPO_CONFIGS = Path(__file__).resolve().parents[1] / "configs"

_QT_MESSAGES: list[str] = []


def _qt_message_handler(_mode: QtMsgType, _context, message: str) -> None:
    _QT_MESSAGES.append(str(message))


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication(["test_rhodes_ui"])
    previous = qInstallMessageHandler(_qt_message_handler)
    yield app  # type: ignore[misc]
    qInstallMessageHandler(previous)


@pytest.fixture(autouse=True)
def _release_top_level_widgets(qapp):
    """每个用例结束后关闭并 deleteLater 全部顶层窗口，杜绝解释器退出期与 QApplication 析构竞争崩溃。"""
    yield
    for w in list(qapp.topLevelWidgets()):
        try:
            if hasattr(w, "force_close"):
                w.force_close()
            else:
                w.close()
            w.deleteLater()
        except RuntimeError:
            pass
    qapp.processEvents()


@pytest.fixture(autouse=True)
def _no_font_warnings():
    start = len(_QT_MESSAGES)
    yield
    new_msgs = _QT_MESSAGES[start:]
    bad = [m for m in new_msgs if "setPointSize" in m or "Point size <= 0" in m]
    assert not bad, f"检测到 Qt 字体点字号警告：{bad}"


def _dispose(qapp, *widgets) -> None:
    """关闭并安全释放顶层窗口，避免解释器退出时与 QApplication 析构竞争。"""
    for w in widgets:
        if w is None:
            continue
        if hasattr(w, "force_close"):
            w.force_close()
        else:
            w.close()
        w.deleteLater()
    qapp.processEvents()


def _snapshot(**overrides) -> HUDStateSnapshot:
    base = dict(
        visible=True,
        paused=False,
        calibration_requested=False,
        scene=GameScene.BATTLE_PREP,
        scene_locked_manual=False,
        focus_zone=FocusZone.BOARD,
        focus_label="棋盘 R2C5",
        focus_polygon_norm=[(0.40, 0.40), (0.48, 0.40), (0.49, 0.48), (0.39, 0.48)],
        holding_source_norm=None,
        cursor_norm=(0.5, 0.5),
        free_cursor_active=False,
        deploy_state=DeployState.IDLE,
        aim_origin_norm=None,
        aim_preview_dir=None,
        ready_hold_progress=0.0,
        prompts=[PromptItem("A", "确认"), PromptItem("B", "取消"), PromptItem("LT+RB", "撤退")],
        toast_message=None,
        view_state="normal",
    )
    base.update(overrides)
    return HUDStateSnapshot(**base)


def _render(hud: HUDWindow, snapshot: HUDStateSnapshot, w: int = 1280, h: int = 720) -> QImage:
    img = QImage(w, h, QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))
    painter = QPainter(img)
    try:
        hud.render_snapshot(painter, float(w), float(h), snapshot)
    finally:
        painter.end()
    return img


def _non_transparent_pixels(img: QImage, step: int = 4) -> int:
    count = 0
    for y in range(0, img.height(), step):
        for x in range(0, img.width(), step):
            if img.pixelColor(x, y).alpha() > 0:
                count += 1
    return count


# ----------------------------------------------------------------------
# 透明穿透 HUD
# ----------------------------------------------------------------------
def test_hud_exports_and_palette_contract(qapp) -> None:
    assert hud_window.WDA_EXCLUDEFROMCAPTURE == 0x00000011
    assert hud_window.WS_EX_TRANSPARENT == 0x00000020
    assert hud_window.WS_EX_NOACTIVATE == 0x08000000
    assert hud_window.WS_EX_LAYERED == 0x00080000
    assert hud_window.WS_EX_TOPMOST == 0x00000008
    assert COLOR_CYAN.name().upper() == "#0098DC"
    assert COLOR_AMBER.name().upper() == "#F59E0B"
    assert COLOR_RED.name().upper() == "#DC2626"
    assert isinstance(COLOR_GREEN, QColor)
    assert SCENE_DISPLAY_NAMES[GameScene.BATTLE_PREP] == "休整运营"
    assert SCENE_DISPLAY_NAMES[GameScene.UNKNOWN_FREE_CURSOR] == "自由鼠标"
    assert callable(align_widget_to_physical_rect)


def test_hud_window_is_click_through_and_never_takes_focus(qapp) -> None:
    hud = HUDWindow()
    flags = hud.windowFlags()
    assert flags & Qt.WindowType.WindowTransparentForInput
    assert flags & Qt.WindowType.WindowStaysOnTopHint
    assert flags & Qt.WindowType.FramelessWindowHint
    assert hud.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert hud.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert hud.focusPolicy() == Qt.FocusPolicy.NoFocus
    # 离屏平台不应调用 Win32 API
    assert apply_win32_click_through(12345) is False
    for f in (hud._font_title, hud._font_badge, hud._font_prompt, hud._font_small, hud._font_micro):
        assert f.pointSize() > 0


def test_format_top_status_text_dedupes_free_cursor(qapp) -> None:
    snap = _snapshot(scene=GameScene.UNKNOWN_FREE_CURSOR, focus_label="自由光标", free_cursor_active=True)
    assert HUDWindow._format_top_status_text(snap) == "PRTS // 自由鼠标"
    snap2 = _snapshot(view_state="shop_open", scene_locked_manual=True, paused=True)
    text = HUDWindow._format_top_status_text(snap2)
    assert text.startswith("PRTS // 休整运营")
    assert "商店展开" in text and "手动锁层" in text and "已暂停" in text


def test_chamfer_polygon_geometry(qapp) -> None:
    poly = chamfer_polygon(QRectF(0, 0, 100, 40), tl=10, br=10)
    assert poly.count() == 6
    xs = [poly.at(i).x() for i in range(poly.count())]
    ys = [poly.at(i).y() for i in range(poly.count())]
    assert min(xs) == 0 and max(xs) == 100 and min(ys) == 0 and max(ys) == 40
    # 左上角被斜切：不存在 (0,0) 顶点
    assert all(not (poly.at(i).x() == 0 and poly.at(i).y() == 0) for i in range(poly.count()))


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"holding_source_norm": (0.3, 0.7), "deploy_state": DeployState.HOLDING_UNIT, "toast_message": "已锁定抓取源"},
        {"holding_source_norm": (0.9, 0.3), "deploy_state": DeployState.HOLDING_STRATEGY},
        {"holding_source_norm": (0.6, 0.8), "deploy_state": DeployState.HOLDING_EQUIP},
        {"deploy_state": DeployState.AIMING_DIRECTION, "aim_origin_norm": (0.5, 0.45), "aim_preview_dir": "left"},
        {"deploy_state": DeployState.AIMING_DIRECTION, "aim_origin_norm": (0.5, 0.95), "aim_preview_dir": None},
        {"free_cursor_active": True, "cursor_norm": (0.7, 0.3), "scene": GameScene.UNKNOWN_FREE_CURSOR},
        {"ready_hold_progress": 0.55},
        {"paused": True, "scene_locked_manual": True, "toast_message": "手柄映射已暂停"},
        {"prompts": [PromptItem(f"RT+{b}", f"超长提示标签{b}" * 3) for b in "ABXY" * 4]},
        {"focus_polygon_norm": [(0.2, 0.2), (0.3, 0.25), (0.25, 0.35)]},
    ],
)
def test_hud_renders_all_tactical_elements(qapp, overrides) -> None:
    hud = HUDWindow()
    img = _render(hud, _snapshot(**overrides))
    assert _non_transparent_pixels(img) > 200


def test_hud_paint_event_offscreen_widget_grab(qapp) -> None:
    hud = HUDWindow()
    hud.resize(960, 540)
    hud._snapshot = _snapshot(ready_hold_progress=0.3, holding_source_norm=(0.2, 0.8),
                              deploy_state=DeployState.HOLDING_UNIT)
    pix = hud.grab()
    assert not pix.isNull()
    assert pix.width() == 960


def test_hud_update_state_hides_without_viewport(qapp) -> None:
    hud = HUDWindow()
    hud.update_state(None, _snapshot())
    assert not hud.isVisible()
    hud.set_overlay_enabled(False)
    assert hud.overlay_enabled is False


# ----------------------------------------------------------------------
# PRTS 桌面主控台
# ----------------------------------------------------------------------
@pytest.fixture()
def dashboard(qapp):
    win = PRTSDashboardWindow()
    yield win
    _dispose(qapp, win)


def test_dashboard_is_normal_window_not_topmost(dashboard) -> None:
    flags = dashboard.windowFlags()
    assert not (flags & Qt.WindowType.WindowStaysOnTopHint)
    assert not (flags & Qt.WindowType.WindowTransparentForInput)
    assert dashboard.windowTitle().startswith("ArkController")


def test_dashboard_close_hides_to_tray_and_notifies_once(qapp, dashboard) -> None:
    hits: list[int] = []
    dashboard.hidden_to_tray.connect(lambda: hits.append(1))
    dashboard.show()
    qapp.processEvents()
    assert dashboard.isVisible()
    dashboard.close()
    qapp.processEvents()
    assert not dashboard.isVisible()
    assert hits == [1]
    dashboard.show()
    dashboard.close()
    qapp.processEvents()
    assert hits == [1]  # 仅首次提示


def test_dashboard_quit_button_emits_quit(dashboard) -> None:
    hits: list[int] = []
    dashboard.quit_requested.connect(lambda: hits.append(1))
    dashboard._btn_quit.click()
    assert hits == [1]


def test_dashboard_telemetry_throttled_only_when_visible(qapp, dashboard) -> None:
    assert not dashboard.refresh_active
    state = GamepadState(
        connected=True,
        buttons=frozenset({GamepadButton.A, GamepadButton.LB}),
        lx=0.5, ly=-0.25, rx=0.0, ry=0.8, lt=0.9, rt=0.1,
    )
    dashboard.push_gamepad_state(state)
    # 隐藏状态下仅缓存，不应用
    assert dashboard.telemetry.state != state
    dashboard.show()
    qapp.processEvents()
    assert dashboard.refresh_active
    assert dashboard.telemetry.state == state
    labels = dashboard.telemetry.pressed_labels()
    assert "A" in labels and "LB" in labels and "LT" in labels and "RT" not in labels
    dashboard.hide()
    qapp.processEvents()
    assert not dashboard.refresh_active


def test_dashboard_runtime_status_applied(qapp, dashboard) -> None:
    dashboard.push_runtime_status(
        DashboardRuntimeStatus(
            gamepad_connected=True,
            window_found=True,
            window_title="明日方舟",
            client_rect=(0, 0, 1920, 1200),
            active_rect=(0, 60, 1920, 1080),
            is_foreground=True,
            scene_name="决策弹窗",
            shop_open=True,
            shop_slot_count=6,
            paused=True,
        )
    )
    dashboard._flush_pending()
    grid = dashboard._status_grid
    assert "1920 × 1080" in grid.value("viewport")
    assert grid.value("scene") == "决策弹窗"
    assert "6 槽" in grid.value("shop")
    assert grid.value("mapping") == "已暂停"
    assert dashboard.toggle_pause.isChecked()


def test_dashboard_toggles_emit_and_sync_is_silent(dashboard) -> None:
    km = KeymapConfig()
    dashboard.sync_settings(keymap=km, overlay_enabled=True, paused=False, refresh_tuning=True)
    events: list[tuple] = []
    dashboard.overlay_toggled.connect(lambda v: events.append(("overlay", v)))
    dashboard.native_shortcuts_toggled.connect(lambda v: events.append(("native", v)))
    dashboard.left_stick_dpad_toggled.connect(lambda v: events.append(("dpad", v)))
    dashboard.keymap_flag_toggled.connect(lambda k, v: events.append((k, v)))
    dashboard.toggle_pause_requested.connect(lambda: events.append(("pause",)))

    dashboard.sync_settings(keymap=km, overlay_enabled=False, paused=True)
    assert events == []
    assert dashboard.toggle_overlay.isChecked() is False

    dashboard.toggle_overlay.click()
    dashboard.toggle_native.click()
    dashboard.toggle_dpad.click()
    dashboard.toggle_auto_shop.click()
    dashboard.toggle_vibration.click()
    dashboard.toggle_retreat.click()
    dashboard.toggle_pause.click()
    assert ("overlay", True) in events
    assert ("native", False) in events
    assert ("dpad", False) in events
    assert ("auto_toggle_shop", False) in events
    assert ("vibration_enabled", False) in events
    assert ("allow_retreat_chord", False) in events
    assert ("pause",) in events
    # 暂停开关由主控回写，点击后保持真实状态
    assert dashboard.toggle_pause.isChecked() is True


def test_dashboard_action_buttons_emit(dashboard) -> None:
    hits: list[str] = []
    dashboard.open_calibration_requested.connect(lambda: hits.append("cal"))
    dashboard.reload_config_requested.connect(lambda: hits.append("reload"))
    dashboard.btn_calibrate.click()
    dashboard.btn_reload.click()
    assert hits == ["cal", "reload"]


def test_dashboard_tuning_apply_and_reset(dashboard) -> None:
    km = KeymapConfig()
    dashboard.sync_settings(keymap=km, overlay_enabled=True, paused=False, refresh_tuning=True)
    assert set(dashboard.tuning_rows) == {s.key for s in TUNING_SPECS}
    payloads: list[dict] = []
    dashboard.tuning_apply_requested.connect(lambda d: payloads.append(d))

    dashboard.tuning_rows["left_deadzone"].set_value(0.22, silent=False)
    dashboard.tuning_rows["ready_hold_ms"].set_value(500, silent=False)
    dashboard.tuning_rows["cursor_gamma"].set_value(2.6, silent=False)
    assert dashboard._tuning_dirty
    dashboard.btn_apply_tuning.click()
    assert len(payloads) == 1
    p = payloads[0]
    assert p["left_deadzone"] == pytest.approx(0.22)
    assert p["ready_hold_ms"] == 500 and isinstance(p["ready_hold_ms"], int)
    assert p["cursor_gamma"] == pytest.approx(2.6)
    KeymapConfig(**{**km.to_dict(), **p})  # 可通过模型校验

    # 未保存修改时 sync 不覆盖滑块
    dashboard.sync_settings(keymap=km, overlay_enabled=True, paused=False)
    assert dashboard.tuning_rows["left_deadzone"].value() == pytest.approx(0.22)

    dashboard.btn_reset_tuning.click()
    defaults = KeymapConfig()
    assert payloads[-1]["left_deadzone"] == pytest.approx(defaults.left_deadzone)
    assert payloads[-1]["cursor_max_speed_px"] == pytest.approx(defaults.cursor_max_speed_px)
    assert payloads[-1]["ready_hold_ms"] == defaults.ready_hold_ms


def test_dashboard_keymap_manual_covers_v022_mappings(dashboard) -> None:
    assert len(dashboard.keymap_cards) == len(KEYMAP_BRIEFS) == 9
    all_rows = [(k, desc) for _c, _t, _s, rows in KEYMAP_BRIEFS for k, desc in rows]
    all_keys = [k for k, _ in all_rows]
    all_text = " ".join(desc for _, desc in all_rows)
    for keyword in ("双击 A 买牌", "查看商店干员详情", "升本", "刷新商店", "冻结", "原地转向",
                    "白条", "不再提示", "表情", "撤退", "队友", "本局策略"):
        assert keyword in all_text, keyword
    assert "赠予" not in all_text
    assert "LT + RT" in all_keys
    assert ("LT + RT", "无需移焦点，一键点击右侧本局策略") in all_rows


def test_dashboard_pages_render_offscreen(qapp, dashboard) -> None:
    dashboard.resize(1280, 820)
    dashboard.show()
    qapp.processEvents()
    for idx in range(4):
        dashboard.set_current_page(idx)
        qapp.processEvents()
        assert dashboard.current_page_index() == idx
        assert dashboard._nav_buttons[idx].isChecked()
        pix = dashboard.grab()
        assert not pix.isNull() and pix.width() == 1280


def test_dashboard_system_info(dashboard) -> None:
    from arkcontroller.config.loader import ConfigLoader

    loader = ConfigLoader(config_dir=REPO_CONFIGS)
    _km, layout = loader.load()
    dashboard.set_system_info(version="9.9.9", is_admin=False, config_dir="C:/cfg",
                              override_path="C:/cfg/user_override.json", template_count=3, layout=layout)
    assert dashboard._sys_grid.value("version") == "v9.9.9"
    assert "标准权限" in dashboard._sys_grid.value("admin")
    assert f"{layout.board_quad.rows} 行 × {layout.board_quad.cols} 列" in dashboard._cfg_grid.value("board")
    assert dashboard._matrix_grid.value("normal").startswith("基准坐标系")


# ----------------------------------------------------------------------
# 系统托盘
# ----------------------------------------------------------------------
def test_tray_open_dashboard_entry_and_activation(qapp) -> None:
    tray = ControllerTray()
    actions = tray.menu.actions()
    assert "PRTS 主控台" in actions[0].text()
    hits: list[int] = []
    tray.open_dashboard_requested.connect(lambda: hits.append(1))
    actions[0].trigger()
    tray._on_activated(QSystemTrayIcon.ActivationReason.Trigger)
    tray._on_activated(QSystemTrayIcon.ActivationReason.DoubleClick)
    tray._on_activated(QSystemTrayIcon.ActivationReason.Context)
    assert hits == [1, 1, 1]
    assert "QMenu" in tray.menu.styleSheet()
    for active, paused in ((True, False), (False, False), (True, True)):
        assert not create_prts_icon(active=active, paused=paused).isNull()


# ----------------------------------------------------------------------
# 校准向导
# ----------------------------------------------------------------------
def test_calibration_dialog_renders_with_rhodes_style(qapp, tmp_path) -> None:
    from arkcontroller.config.loader import ConfigLoader
    from arkcontroller.overlay.calibration_ui import CalibrationDialog

    loader = ConfigLoader(config_dir=REPO_CONFIGS)
    _km, layout = loader.load()
    dlg = CalibrationDialog(layout=layout, viewport=None, config_loader=loader)
    dlg.resize(1280, 720)
    pix = dlg.grab()
    assert not pix.isNull()
    assert "#F59E0B" in dlg._dock.styleSheet()
    dlg.close()


# ----------------------------------------------------------------------
# main.py 装配
# ----------------------------------------------------------------------
@pytest.fixture()
def tmp_configs(tmp_path) -> Path:
    cfg = tmp_path / "configs"
    cfg.mkdir()
    for name in ("default_bindings.json", "stronghold_layout_16_9.json"):
        shutil.copy(REPO_CONFIGS / name, cfg / name)
    return cfg


def test_main_parser_minimized_and_relaunch_args() -> None:
    from arkcontroller.main import _build_relaunch_args, build_arg_parser

    args = build_arg_parser().parse_args(["--minimized", "--no-elevate"])
    assert args.minimized is True
    assert "--minimized" in _build_relaunch_args(args)
    assert build_arg_parser().parse_args([]).minimized is False


def test_main_app_dashboard_wiring_and_hot_save(qapp, tmp_configs) -> None:
    from arkcontroller.main import ArkControllerApp

    app = ArkControllerApp(qt_app=qapp, config_dir=tmp_configs, dry_run=True)
    try:
        assert app._show_dashboard_on_start is False  # dry-run 不弹窗
        assert not app.dashboard.isVisible()

        ok = app.apply_tuning({"left_deadzone": 0.2, "ready_hold_ms": 480.0, "cursor_gamma": 2.5, "bogus": 1})
        assert ok is True
        assert app.keymap.left_deadzone == pytest.approx(0.2)
        assert app.keymap.ready_hold_ms == 480
        saved = json.loads((tmp_configs / "user_override.json").read_text(encoding="utf-8"))
        assert saved["keymap"]["left_deadzone"] == pytest.approx(0.2)
        assert saved["keymap"]["ready_hold_ms"] == 480
        assert "bogus" not in saved["keymap"]
        assert app.dashboard.tuning_rows["left_deadzone"].value() == pytest.approx(0.2)

        bad = app.apply_tuning({"trigger_threshold": 5.0})
        assert bad is False
        assert app.keymap.trigger_threshold == pytest.approx(KeymapConfig().trigger_threshold)

        app.set_keymap_flag("vibration_enabled", False)
        assert app.keymap.vibration_enabled is False
        assert app.dashboard.toggle_vibration.isChecked() is False
        saved = json.loads((tmp_configs / "user_override.json").read_text(encoding="utf-8"))
        assert saved["keymap"]["vibration_enabled"] is False

        app.set_keymap_flag("not_a_flag", True)  # 安全忽略

        app.toggle_pause()
        assert app.dashboard.toggle_pause.isChecked() is True
        app.toggle_pause()
        assert app.dashboard.toggle_pause.isChecked() is False

        app.set_overlay_enabled(False)
        assert app.dashboard.toggle_overlay.isChecked() is False

        status = app._build_dashboard_status()
        assert isinstance(status, DashboardRuntimeStatus)
        app.show_dashboard()
        qapp.processEvents()
        assert app.dashboard.isVisible()
    finally:
        app.shutdown()
    assert not app.dashboard.isVisible()
    _dispose(qapp, app.dashboard, app.hud)


def test_main_app_shows_dashboard_unless_minimized(qapp, tmp_configs) -> None:
    from arkcontroller.main import ArkControllerApp

    app = ArkControllerApp(qt_app=qapp, config_dir=tmp_configs, dry_run=False, show_dashboard=False)
    try:
        assert app._show_dashboard_on_start is False
    finally:
        app.shutdown()
        _dispose(qapp, app.dashboard, app.hud)
    app2 = ArkControllerApp(qt_app=qapp, config_dir=tmp_configs, dry_run=False, show_dashboard=True)
    try:
        assert app2._show_dashboard_on_start is True
    finally:
        app2.shutdown()
        _dispose(qapp, app2.dashboard, app2.hud)


# ----------------------------------------------------------------------
# 纯白极简方案（方案 B v2 + H1/H2 混搭）专项
# ----------------------------------------------------------------------
DISPATCHER_BUTTON_SPECS = (
    "A", "B", "X", "Y", "A/RT+A", "Back+Start", "LB/RB", "LS", "RS", "RS/LB/RB",
    "LT+Back", "LT+LB", "LT+RB", "LT+Start", "LT+十字键", "LT + RT", "RT+A", "Start/RT+A",
    "按住 LT", "松开 LT", "下", "摇杆/十字键", "右/B", "长按Start", "左/上", "左/上/下",
    "LS/D-Pad", "LS/十字键", "D-Pad", "双击A",
)


def _kinds(spec: str) -> list[tuple[str, str]]:
    from arkcontroller.overlay.hud_window import parse_button_spec

    return [(g.kind, g.value) for g in parse_button_spec(spec)]


def test_button_spec_parser_maps_dispatcher_strings(qapp) -> None:
    assert _kinds("长按Start") == [("prefix", "长按"), ("start", "")]
    assert _kinds("Start/RT+A") == [("start", ""), ("sep", "/"), ("cap", "RT"), ("sep", "+"), ("face", "A")]
    assert _kinds("LS/D-Pad") == [("stick", "LS"), ("sep", "/"), ("dpad", "")]
    assert _kinds("LS/十字键") == [("stick", "LS"), ("sep", "/"), ("dpad", "")]
    assert _kinds("摇杆/十字键")[0] == ("stick", "LS")
    assert _kinds("RS") == [("stick", "RS")]
    assert _kinds("右/B") == [("dpad", "right"), ("sep", "/"), ("face", "B")]
    assert _kinds("左/上/下") == [("dpad", "left"), ("sep", "/"), ("dpad", "up"), ("sep", "/"), ("dpad", "down")]
    assert _kinds("按住 LT") == [("prefix", "按住"), ("cap", "LT")]
    assert _kinds("松开 LT") == [("prefix", "松开"), ("cap", "LT")]
    assert _kinds("Back+Start") == [("back", ""), ("sep", "+"), ("start", "")]
    assert _kinds("LT + RT") == [("cap", "LT"), ("sep", "+"), ("cap", "RT")]
    assert _kinds("双击A") == [("prefix", "双击"), ("face", "A")]
    assert _kinds("A>A") == [("face", "A"), ("sep", "›"), ("face", "A")]
    # 未知 token 兜底为文字键帽，不丢失信息
    assert _kinds("Esc") == [("text", "Esc")]


def test_all_dispatcher_prompt_specs_render(qapp) -> None:
    from PySide6.QtGui import QFont

    from arkcontroller.overlay.hud_window import measure_button_spec, paint_button_spec

    font = QFont("Segoe UI", 8, QFont.Weight.DemiBold)
    img = QImage(400, 40, QImage.Format.Format_ARGB32)
    for spec in DISPATCHER_BUTTON_SPECS:
        img.fill(QColor(0, 0, 0, 0))
        p = QPainter(img)
        try:
            end = paint_button_spec(p, spec, 4.0, 20.0, font)
        finally:
            p.end()
        width = measure_button_spec(spec, font)
        assert width > 0 and abs((end - 4.0) - width) < 0.01, spec
        assert width < 260, spec
        assert _non_transparent_pixels(img, step=2) > 10, spec


def test_hud_bottom_bar_with_real_dispatcher_prompts(qapp) -> None:
    hud = HUDWindow()
    prompts = [PromptItem(s, "操作") for s in DISPATCHER_BUTTON_SPECS[:10]]
    img = _render(hud, _snapshot(prompts=prompts))
    assert _non_transparent_pixels(img) > 200


def test_hud_pending_focus_detection(qapp) -> None:
    assert HUDWindow._is_pending(_snapshot(focus_label="商店 3 · 再按A购买"))
    assert HUDWindow._is_pending(_snapshot(focus_label="卡片 2 · 待确认"))
    assert HUDWindow._is_pending(_snapshot(holding_source_norm=(0.3, 0.7), deploy_state=DeployState.HOLDING_UNIT))
    assert not HUDWindow._is_pending(_snapshot())


def test_dashboard_white_theme_controls(qapp, dashboard) -> None:
    from arkcontroller.overlay.dashboard_window import AmberCheckBox, CheckRow, LineButton

    # 右上两颗按钮统一为白底细边 LineButton
    assert isinstance(dashboard.btn_reload, LineButton)
    assert isinstance(dashboard.btn_calibrate, LineButton)
    assert type(dashboard.btn_reload) is type(dashboard.btn_calibrate)
    assert "启动校准向导" in dashboard.btn_calibrate.text()
    # 快捷开关为空心框黄勾 CheckRow
    for t in (dashboard.toggle_pause, dashboard.toggle_overlay, dashboard.toggle_native,
              dashboard.toggle_dpad, dashboard.toggle_auto_shop, dashboard.toggle_vibration,
              dashboard.toggle_retreat):
        assert isinstance(t, CheckRow)
    # 勾选框渲染：勾选后框内出现琥珀色像素，且框内其余为白色（无大面积填充）
    row = CheckRow("测试", "说明")
    row.resize(240, 44)
    row.setChecked(True)
    img = row.grab().toImage()
    amber = sum(
        1 for y in range(7, 21) for x in range(1, 15)
        if img.pixelColor(x, y).red() > 200 and 100 < img.pixelColor(x, y).green() < 190
        and img.pixelColor(x, y).blue() < 80
    )
    white = sum(1 for y in range(7, 21) for x in range(1, 15) if img.pixelColor(x, y).lightness() > 245)
    assert amber > 3
    assert white > amber
    cb = AmberCheckBox("棋盘透视")
    cb.setChecked(True)
    assert not cb.grab().isNull()


def test_dashboard_header_status_texts(qapp, dashboard) -> None:
    dashboard.push_runtime_status(DashboardRuntimeStatus(gamepad_connected=True, window_found=True,
                                                         is_foreground=True, paused=False))
    dashboard._flush_pending()
    texts = dashboard.header_status_texts()
    assert "手柄在线" in texts
    assert any("明日方舟" in t for t in texts)


# ----------------------------------------------------------------------
# v0.3.2：截屏 / 屏幕共享可见性热切换与充能环避让盟约栏
# ----------------------------------------------------------------------
def test_display_affinity_constants_and_offscreen_safe(qapp) -> None:
    assert hud_window.WDA_NONE == 0x00000000
    assert hud_window.WDA_EXCLUDEFROMCAPTURE == 0x00000011
    assert hud_window.apply_win32_display_affinity(12345, True) is False
    assert hud_window.apply_win32_display_affinity(12345, False) is False
    assert apply_win32_click_through(12345, exclude_from_capture=False) is False


def test_hud_hide_from_capture_hot_switch(qapp, monkeypatch) -> None:
    calls: list[tuple[int, bool]] = []
    monkeypatch.setattr(hud_window, "_is_windows_qpa", lambda: True)
    monkeypatch.setattr(hud_window, "apply_win32_display_affinity",
                        lambda hwnd, exclude=True: calls.append((hwnd, exclude)) or True)
    hud = HUDWindow()
    assert hud.hide_from_capture is True
    hud.set_hide_from_capture(False)  # 句柄未创建：仅记录状态
    assert hud.hide_from_capture is False and calls == []
    hwnd = int(hud.winId())
    hud.set_hide_from_capture(True)
    hud.set_hide_from_capture(False)
    assert calls == [(hwnd, True), (hwnd, False)]


def test_ready_ring_below_covenant_bar(qapp) -> None:
    hud = HUDWindow()
    img = _render(hud, _snapshot(ready_hold_progress=0.5, prompts=[], focus_polygon_norm=None))
    # 顶签 (y<40) 与充能环 (cy>=142) 之间的盟约栏区域保持全透明
    band = sum(1 for y in range(48, 118, 2) for x in range(480, 800, 4) if img.pixelColor(x, y).alpha() > 0)
    assert band == 0
    ring = sum(1 for y in range(122, 166, 2) for x in range(480, 800, 4) if img.pixelColor(x, y).alpha() > 0)
    assert ring > 20


def test_dashboard_hide_capture_switch_and_protocol_text(dashboard) -> None:
    events: list[tuple[str, bool]] = []
    dashboard.keymap_flag_toggled.connect(lambda k, v: events.append((k, v)))
    km = KeymapConfig()
    dashboard.sync_settings(keymap=km, overlay_enabled=True, paused=False, hide_hud_from_capture=True)
    assert dashboard.toggle_hide_capture.isChecked()
    assert "WDA_EXCLUDEFROMCAPTURE" in dashboard.capture_mode_text()
    assert events == []
    dashboard.toggle_hide_capture.click()
    assert events == [("hide_hud_from_capture", False)]
    dashboard.sync_settings(keymap=km, overlay_enabled=True, paused=False, hide_hud_from_capture=False)
    assert not dashboard.toggle_hide_capture.isChecked()
    assert "WDA_NONE" in dashboard.capture_mode_text()


def test_tray_hide_capture_action(qapp) -> None:
    tray = ControllerTray()
    hits: list[bool] = []
    tray.toggle_hide_capture_requested.connect(lambda v: hits.append(v))
    tray.sync_settings(overlay_enabled=True, use_native_shortcuts=True, paused=False, hide_hud_from_capture=False)
    assert tray._action_hide_capture.isChecked() is False
    assert hits == []
    tray._action_hide_capture.trigger()
    assert hits == [True]
    tray.sync_settings(overlay_enabled=True, use_native_shortcuts=True, paused=False)  # 向后兼容
    assert tray._action_hide_capture.isChecked() is True


def test_main_app_hide_capture_persist_and_hot_apply(qapp, tmp_configs) -> None:
    from arkcontroller.main import DASHBOARD_KEYMAP_FLAGS, ArkControllerApp

    assert "hide_hud_from_capture" in DASHBOARD_KEYMAP_FLAGS
    app = ArkControllerApp(qt_app=qapp, config_dir=tmp_configs, dry_run=True)
    try:
        assert app.hud.hide_from_capture is True
        app.set_keymap_flag("hide_hud_from_capture", False)
        assert app.hud.hide_from_capture is False
        assert app.dashboard.toggle_hide_capture.isChecked() is False
        assert app.tray._action_hide_capture.isChecked() is False
        saved = json.loads((tmp_configs / "user_override.json").read_text(encoding="utf-8"))
        assert saved["keymap"]["hide_hud_from_capture"] is False
        app._on_tray_hide_capture(True)
        assert app.hud.hide_from_capture is True
        assert app.dashboard.toggle_hide_capture.isChecked() is True
    finally:
        app.shutdown()
        _dispose(qapp, app.dashboard, app.hud)
