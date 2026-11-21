"""ArkController 命令行主入口与全模块实时事件循环装配器。

负责初始化 Windows Per-Monitor DPI V2 感知、检测/请求管理员权限（UAC）、
装配配置加载、窗口追踪、XInput 手柄驱动、手势状态机、Win32 SendInput 合成器、
异步屏幕采样与场景识别器、核心操控调度器、PySide6 透明穿透 HUD 与系统托盘。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import logging
from pathlib import Path
import signal
import sys
import time
from typing import Sequence

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QApplication, QSystemTrayIcon

from arkcontroller.config.loader import ConfigLoader
from arkcontroller.config.models import KeymapConfig, LayoutProfile
from arkcontroller.control.dispatcher import ControlDispatcher, HUDStateSnapshot
from arkcontroller.input.gesture_engine import GestureEngine
from arkcontroller.input.virtual_hid import VirtualHID
from arkcontroller.input.xinput_driver import GamepadState, XInputDriver
from arkcontroller.overlay.calibration_ui import CalibrationDialog
from arkcontroller.overlay.hud_window import HUDWindow
from arkcontroller.overlay.tray import ControllerTray
from arkcontroller.platform.uac import (
    enable_per_monitor_dpi_awareness,
    is_admin,
    relaunch_as_admin,
)
from arkcontroller.platform.window_tracker import ViewportRect, WindowTracker
from arkcontroller.vision.calibrator import VisionCalibrator
from arkcontroller.vision.scene_detector import SceneDetectionResult, SceneDetector
from arkcontroller.vision.screen_capture import ScreenCapture

logger = logging.getLogger("arkcontroller")

INPUT_TICK_INTERVAL_MS = 8      # ~125Hz 手柄轮询与调度
WINDOW_POLL_INTERVAL_MS = 250   # 4Hz 窗口句柄与 16:9 视口刷新
VISION_POLL_INTERVAL_MS = 200   # 5Hz 异步场景视觉识别读取


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="arkcontroller",
        description="《明日方舟》卫戍协议零注入 Xbox 手柄全流程操控系统",
    )
    parser.add_argument(
        "--config-dir",
        "--config",
        dest="config_dir",
        type=Path,
        default=None,
        help="自定义配置文件目录路径（默认自动定位项目根目录下 configs/）",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="启动时关闭 PySide6 透明 HUD 悬浮提示层（后续仍可在托盘中重新开启）",
    )
    parser.add_argument(
        "--no-elevate",
        action="store_true",
        help="跳过 UAC 管理员权限自动提权检测（调试或模拟器环境适用）",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="启动后立即打开可视化透视棋盘与模板校准向导",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="自动化冒烟测试模式：完成全模块装配并运行数个主循环 tick 后优雅退出",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="日志输出级别（默认 INFO）",
    )
    return parser


def _resolve_templates_dir(config_loader: ConfigLoader) -> Path:
    """定位场景识别模板存储目录（assets/templates）。"""
    candidate = config_loader.config_dir.resolve().parent / "assets" / "templates"
    if candidate.parent.exists():
        return candidate
    return Path(__file__).resolve().parents[2] / "assets" / "templates"


class ArkControllerApp(QObject):
    """ArkController 主控装配与 PySide6 定时器驱动中枢。"""

    def __init__(
        self,
        *,
        qt_app: QApplication,
        config_dir: Path | None = None,
        no_overlay: bool = False,
        dry_run: bool = False,
        dry_run_ticks: int = 5,
    ) -> None:
        super().__init__()
        self._qt_app = qt_app
        self._dry_run = bool(dry_run)
        self._dry_run_target_ticks = max(1, int(dry_run_ticks))
        self._tick_count: int = 0
        self._running: bool = False
        self._shutdown_done: bool = False

        # 1. 加载默认配置与用户校准覆盖
        self.config_loader = ConfigLoader(config_dir=config_dir)
        self.keymap, self.layout = self.config_loader.load()

        # 2. 初始化平台窗口追踪、手柄输入与键鼠合成
        self.window_tracker = WindowTracker()
        self.xinput = XInputDriver(
            user_index=0,
            left_deadzone=self.keymap.left_deadzone,
            right_deadzone=self.keymap.right_deadzone,
            trigger_threshold=self.keymap.trigger_threshold,
        )
        self.gesture_engine = GestureEngine(self.keymap)
        self.hid = VirtualHID()

        # 3. 初始化非侵入式异步屏幕采样与场景识别
        self.templates_dir = _resolve_templates_dir(self.config_loader)
        self.screen_capture = ScreenCapture()
        self.scene_detector = SceneDetector(templates_dir=self.templates_dir)
        self.calibrator = VisionCalibrator(
            templates_dir=self.templates_dir,
            loader=self.config_loader,
        )

        # 4. 初始化核心操控调度器
        self.dispatcher = ControlDispatcher(
            keymap=self.keymap,
            layout=self.layout,
            hid=self.hid,
            rumble_cb=self._handle_rumble,
        )

        # 5. 初始化 PySide6 透明穿透 HUD 与系统托盘
        self.hud = HUDWindow()
        self.hud.set_overlay_enabled(not no_overlay)

        self.tray = ControllerTray(
            overlay_enabled=not no_overlay,
            use_native_shortcuts=self.keymap.use_native_keyboard_shortcuts,
        )
        self.tray.toggle_pause_requested.connect(self.toggle_pause)
        self.tray.toggle_overlay_requested.connect(self.set_overlay_enabled)
        self.tray.toggle_shortcuts_mode_requested.connect(self.set_use_native_shortcuts)
        self.tray.open_calibration_requested.connect(self.open_calibration)
        self.tray.reload_config_requested.connect(self.reload_configs)
        self.tray.quit_requested.connect(self.request_quit)

        # 运行时状态缓存
        self._viewport: ViewportRect | None = None
        self._scene_result: SceneDetectionResult | None = None
        self._last_gamepad_state: GamepadState | None = None
        self._last_snapshot: HUDStateSnapshot | None = None
        self._last_tick_ts: float = time.perf_counter()
        self._calibration_dialog: CalibrationDialog | None = None
        self._paused_before_calibration: bool | None = None

        # 多频异步定时器（125Hz 输入采样、4Hz 窗口追踪、5Hz 场景视觉判别）
        self._input_timer = QTimer(self)
        self._input_timer.setInterval(INPUT_TICK_INTERVAL_MS)
        self._input_timer.timeout.connect(self._on_input_tick)

        self._window_timer = QTimer(self)
        self._window_timer.setInterval(WINDOW_POLL_INTERVAL_MS)
        self._window_timer.timeout.connect(self._on_window_tick)

        self._vision_timer = QTimer(self)
        self._vision_timer.setInterval(VISION_POLL_INTERVAL_MS)
        self._vision_timer.timeout.connect(self._on_vision_tick)

    def start(self, *, open_calibration_on_start: bool = False) -> None:
        """启动系统托盘、后台视觉采样线程与主事件定时器。"""
        if self._running:
            return
        self._running = True
        self._last_tick_ts = time.perf_counter()

        if not self._dry_run and QSystemTrayIcon.isSystemTrayAvailable():
            self.tray.show()

        # 先同步执行一次窗口检测并启动 5Hz 异步截屏线程
        self._on_window_tick()
        self.screen_capture.start_async(fps=5.0)
        self._on_vision_tick()

        self._input_timer.start()
        self._window_timer.start()
        self._vision_timer.start()

        logger.info(
            "ArkController 已启动（HUD=%s，快捷键模式=%s，模板数=%d，dry_run=%s）",
            "开启" if self.hud.overlay_enabled else "关闭",
            "官方PC快捷键" if self.keymap.use_native_keyboard_shortcuts else "纯鼠标坐标",
            getattr(self.scene_detector, "loaded_template_count", 0),
            self._dry_run,
        )

        if open_calibration_on_start and not self._dry_run:
            QTimer.singleShot(120, self.open_calibration)

        if self._dry_run:
            # 防止极端环境下 dry-run 卡死的安全兜底定时器
            QTimer.singleShot(2500, self._finish_dry_run)

    def shutdown(self) -> None:
        """优雅释放定时器、后台采图线程、手柄震动与界面资源。"""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._running = False

        self._input_timer.stop()
        self._window_timer.stop()
        self._vision_timer.stop()

        if self._calibration_dialog is not None:
            try:
                self._calibration_dialog.close()
            except Exception:
                pass
            self._calibration_dialog = None

        try:
            self.screen_capture.close()
        except Exception:
            try:
                self.screen_capture.stop_async()
            except Exception:
                pass

        try:
            self.xinput.stop_rumble()
        except Exception:
            pass

        try:
            self.hud.hide()
            self.hud.close()
        except Exception:
            pass

        try:
            self.tray.hide()
        except Exception:
            pass

        logger.info("ArkController 已安全关闭（累计运行 %d 个输入 tick）。", self._tick_count)

    def request_quit(self) -> None:
        """响应托盘退出或中断信号并退出 Qt 主循环。"""
        self.shutdown()
        self._qt_app.quit()

    def _finish_dry_run(self) -> None:
        """冒烟测试达标后优雅退出。"""
        if not self._shutdown_done:
            logger.info("Dry-run 冒烟测试完成（共执行 %d 个 tick），准备退出。", self._tick_count)
            self.shutdown()
            self._qt_app.quit()

    def _handle_rumble(self, left_motor: float, right_motor: float, duration_ms: int) -> None:
        """按配置开关触发手柄异步微震动反馈。"""
        if not self.keymap.vibration_enabled or self._dry_run:
            return
        try:
            self.xinput.rumble(left_motor, right_motor, duration_ms)
        except Exception:
            logger.debug("发送手柄震动反馈时忽略异常", exc_info=True)

    def _on_window_tick(self) -> None:
        """每 250ms 刷新一次《明日方舟》窗口句柄、16:9 有效视口及托盘状态。"""
        try:
            self._viewport = self.window_tracker.update()
        except Exception:
            logger.debug("窗口追踪刷新失败", exc_info=True)
            self._viewport = None

        self.screen_capture.update_viewport(self._viewport)
        if self._viewport is None:
            self._scene_result = None

        self._refresh_tray_status()

    def _on_vision_tick(self) -> None:
        """每 200ms (~5Hz) 读取最新后台帧并刷新场景阶段识别结果。"""
        if self._viewport is None or not self._viewport.is_foreground:
            return
        try:
            frame = self.screen_capture.get_latest_frame()
            if frame is not None:
                self._scene_result = self.scene_detector.detect(frame)
        except Exception:
            logger.debug("场景视觉检测发生异常", exc_info=True)

    def _on_input_tick(self) -> None:
        """125Hz (8ms) 主循环：采样 XInput -> 解析手势 -> 调度控制 -> 刷新透明 HUD。"""
        now = time.perf_counter()
        dt = max(0.001, min(0.1, now - self._last_tick_ts))
        self._last_tick_ts = now

        try:
            state = self.xinput.poll()
            self._last_gamepad_state = state
            events = self.gesture_engine.update(state, dt)

            # 校准面板打开期间暂停输出，避免手柄误操作游戏
            active_viewport = None if self._calibration_dialog is not None else self._viewport
            snapshot = self.dispatcher.process_tick(active_viewport, self._scene_result, events)
            self._last_snapshot = snapshot

            if self._calibration_dialog is None:
                self.hud.update_state(self._viewport, snapshot)
            else:
                self.hud.update_state(None, None)

            if snapshot.calibration_requested and not self._dry_run:
                self.dispatcher.acknowledge_calibration_request()
                self.open_calibration()
        except Exception:
            logger.exception("主输入循环执行出现异常")
            if self._dry_run:
                self.shutdown()
                self._qt_app.exit(1)
                return

        self._tick_count += 1
        if self._dry_run and self._tick_count >= self._dry_run_target_ticks:
            self._finish_dry_run()

    def _refresh_tray_status(self) -> None:
        """同步手柄与游戏窗口状态至系统托盘。"""
        gp_connected = bool(
            self._last_gamepad_state is not None and self._last_gamepad_state.connected
        )
        win_title = self._viewport.window_title if self._viewport is not None else None
        is_fg = bool(self._viewport is not None and self._viewport.is_foreground)
        paused = bool(
            self._last_snapshot.paused
            if self._last_snapshot is not None
            else getattr(self.dispatcher, "paused", False)
        )
        self.tray.update_status(
            gamepad_connected=gp_connected,
            window_title=win_title,
            is_foreground=is_fg,
            paused=paused,
        )

    def toggle_pause(self) -> None:
        """切换手柄映射暂停/恢复状态。"""
        paused = self.dispatcher.toggle_pause()
        self.tray.sync_settings(
            overlay_enabled=self.hud.overlay_enabled,
            use_native_shortcuts=self.keymap.use_native_keyboard_shortcuts,
            paused=paused,
        )
        self._refresh_tray_status()
        logger.info("手柄控制状态已切换：%s", "已暂停" if paused else "已恢复")

    def set_overlay_enabled(self, enabled: bool) -> None:
        """开关透明 HUD 悬浮显示。"""
        self.hud.set_overlay_enabled(enabled)
        paused = bool(
            self._last_snapshot.paused if self._last_snapshot is not None else False
        )
        self.tray.sync_settings(
            overlay_enabled=self.hud.overlay_enabled,
            use_native_shortcuts=self.keymap.use_native_keyboard_shortcuts,
            paused=paused,
        )

    def set_use_native_shortcuts(self, enabled: bool) -> None:
        """切换官方 PC 原生快捷键模式 / 纯鼠标坐标模式并立即生效。"""
        new_keymap = replace(self.keymap, use_native_keyboard_shortcuts=bool(enabled))
        self._apply_updated_configs(new_keymap, self.layout)
        try:
            self.config_loader.save_user_override(
                {"keymap": {"use_native_keyboard_shortcuts": bool(enabled)}}
            )
        except Exception:
            logger.warning("保存快捷键模式切换至 user_override.json 失败", exc_info=True)

        mode_label = "官方 PC 原生快捷键模式" if enabled else "纯鼠标坐标点击模式（模拟器兼容）"
        logger.info("已切换操控输出模式：%s", mode_label)
        self.tray.show_notification("ArkController 模式切换", f"当前已切换为：{mode_label}")

    def reload_configs(self) -> None:
        """从磁盘热重载配置文件与场景识别模板。"""
        try:
            new_keymap, new_layout = self.config_loader.reload()
            self._apply_updated_configs(new_keymap, new_layout)
            self.scene_detector.reload_templates()
            logger.info("配置文件与视觉识别模板热重载完成。")
            self.tray.show_notification("ArkController 配置已重载", "键位、布局与视觉模板已即时生效。")
        except Exception as exc:
            logger.error("重新加载配置文件失败：%s", exc, exc_info=True)
            self.tray.show_notification("配置文件重载失败", str(exc))

    def _apply_updated_configs(self, keymap: KeymapConfig, layout: LayoutProfile) -> None:
        """将新加载的 KeymapConfig 与 LayoutProfile 同步至所有运行时子模块。"""
        self.keymap = keymap
        self.layout = layout
        self.gesture_engine.update_keymap(keymap)
        self.dispatcher.update_configs(keymap, layout)
        if hasattr(self.xinput, "set_deadzones"):
            self.xinput.set_deadzones(keymap.left_deadzone, keymap.right_deadzone)
        if hasattr(self.xinput, "trigger_threshold"):
            self.xinput.trigger_threshold = keymap.trigger_threshold

        paused = bool(
            self._last_snapshot.paused if self._last_snapshot is not None else False
        )
        self.tray.sync_settings(
            overlay_enabled=self.hud.overlay_enabled,
            use_native_shortcuts=keymap.use_native_keyboard_shortcuts,
            paused=paused,
        )

    def open_calibration(self) -> None:
        """弹出交互式 16:9 可视化校准面板，关闭后自动恢复穿透操控。"""
        if self._calibration_dialog is not None:
            self._calibration_dialog.raise_()
            self._calibration_dialog.activateWindow()
            return

        # 校准期间隐藏透明 HUD，防止视觉重叠
        self.hud.update_state(None, None)

        dialog = CalibrationDialog(
            layout=self.layout,
            viewport=self._viewport,
            config_loader=self.config_loader,
            calibrator=self.calibrator,
            scene_detector=self.scene_detector,
            screen_capture=self.screen_capture,
            on_saved=self._on_calibration_saved,
        )
        dialog.finished.connect(self._on_calibration_closed)
        self._calibration_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_calibration_saved(self, keymap: KeymapConfig, layout: LayoutProfile) -> None:
        """校准面板点击「保存并生效」后的热刷新回调。"""
        self._apply_updated_configs(keymap, layout)
        logger.info("可视化校准配置已保存并热刷新至调度器。")
        self.tray.show_notification("校准已保存生效", "棋盘透视网格与槽位坐标已立即应用。")

    def _on_calibration_closed(self, _result: int) -> None:
        """校准窗口关闭后释放引用，立即恢复游戏穿透操控。"""
        self._calibration_dialog = None


def _configure_logging(level_name: str) -> None:
    """配置简体中文友好的控制台日志格式。"""
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=False,
    )
    logger.setLevel(level)


def _build_relaunch_args(args: argparse.Namespace) -> list[str]:
    """构建 UAC 提权重启时透传的命令行参数列表。"""
    forwarded: list[str] = []
    if args.config_dir is not None:
        forwarded.extend(["--config-dir", str(args.config_dir)])
    if args.no_overlay:
        forwarded.append("--no-overlay")
    if args.calibrate:
        forwarded.append("--calibrate")
    if args.log_level and args.log_level != "INFO":
        forwarded.extend(["--log-level", args.log_level])
    return forwarded


def main(argv: Sequence[str] | None = None) -> int:
    """ArkController CLI 主入口函数。"""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _configure_logging(args.log_level)

    # 1. 开启 Windows Per-Monitor DPI V2 感知（必须在创建 QApplication 之前）
    dpi_ok = enable_per_monitor_dpi_awareness()
    logger.debug("Per-Monitor DPI V2 初始化状态：%s", dpi_ok)

    # 2. UAC 管理员权限检测与自动提权（dry-run 与 --no-elevate 跳过）
    if not args.no_elevate and not args.dry_run and not is_admin():
        print(
            "[ArkController] 检测到当前未以管理员权限运行。"
            "由于《明日方舟》官方 PC 客户端以高权限启动，正在请求 UAC 提权以绕过 Windows UIPI 隔离……"
        )
        elevated = relaunch_as_admin(
            extra_args=None if argv is None else _build_relaunch_args(args)
        )
        if elevated:
            print("[ArkController] 已拉起管理员权限实例，当前普通进程即将退出。")
            return 0
        print(
            "[ArkController] 提示：管理员提权未完成或被取消，将继续以标准权限降级模式运行。\n"
            "  - 安卓模拟器（MuMu / 雷电等普通权限进程）与可视化校准功能可正常使用；\n"
            "  - 若需操控官方 PC 端（Arknights.exe），请右键以管理员身份重新启动本程序。"
        )

    # 3. 初始化或复用 QApplication
    existing_app = QApplication.instance()
    if isinstance(existing_app, QApplication):
        qt_app = existing_app
    else:
        qt_app = QApplication(["arkcontroller"])
    qt_app.setApplicationName("ArkController")
    qt_app.setQuitOnLastWindowClosed(False)

    # 4. 装配全链路控制中枢
    controller = ArkControllerApp(
        qt_app=qt_app,
        config_dir=args.config_dir,
        no_overlay=args.no_overlay,
        dry_run=args.dry_run,
    )

    # 允许终端 Ctrl+C 优雅退出 Qt 事件循环
    sigint_timer = QTimer()
    sigint_timer.setInterval(200)
    sigint_timer.timeout.connect(lambda: None)
    prev_sigint = signal.getsignal(signal.SIGINT)

    def _on_sigint(_signum: int, _frame: object) -> None:
        controller.request_quit()

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except (ValueError, OSError):
        prev_sigint = None

    sigint_timer.start()
    controller.start(open_calibration_on_start=args.calibrate)

    try:
        exit_code = int(qt_app.exec())
    finally:
        sigint_timer.stop()
        controller.shutdown()
        if prev_sigint is not None:
            try:
                signal.signal(signal.SIGINT, prev_sigint)
            except (ValueError, OSError):
                pass

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
