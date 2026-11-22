"""ArkController 系统托盘图标与常驻控制菜单。

内置纯代码绘制的方案 B 纯白极简矢量托盘图标（白底细边 + 琥珀状态方点，不依赖外部 .ico 文件），
右键菜单采用白底细灰边浅底深字 QSS；支持单击 / 双击托盘图标唤出 PRTS 主控台、实时展示手柄/游戏窗口
绑定状态、一键暂停恢复、开关 HUD、切换原生快捷键/纯鼠标模式、打开校准向导与热重载配置。
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QMenu, QSystemTrayIcon, QWidget

TRAY_MENU_QSS = """
QMenu {
    background-color: #FFFFFF;
    color: #1A1A1A;
    border: 1px solid #D9D9D9;
    padding: 6px 0px;
    font-family: "Microsoft YaHei UI";
    font-size: 9pt;
}
QMenu::item {
    padding: 7px 28px 7px 28px;
    border-left: 2px solid transparent;
    background: transparent;
    color: #1A1A1A;
}
QMenu::item:selected {
    background-color: #FFF7E6;
    border-left: 2px solid #F59E0B;
    color: #1A1A1A;
}
QMenu::item:disabled {
    color: #8C8C8C;
}
QMenu::separator {
    height: 1px;
    background: #EDEDED;
    margin: 5px 12px;
}
QMenu::indicator {
    width: 12px;
    height: 12px;
    left: 9px;
}
QMenu::indicator:checked {
    background-color: #FFFFFF;
    border: 1px solid #BDBDBD;
}
QMenu::indicator:checked:selected {
    background-color: #FFF7E6;
}
QMenu::indicator:unchecked {
    background-color: #FFFFFF;
    border: 1px solid #BDBDBD;
}
"""


def _menu_check_qss() -> str:
    """菜单勾选项：白底细框内绘制琥珀线条小勾（矢量 SVG）。"""
    try:
        from arkcontroller.overlay.dashboard_window import ui_asset_path

        path = ui_asset_path("check")
    except Exception:
        path = ""
    if not path:
        return ""
    return f'\nQMenu::indicator:checked {{ image: url("{path}"); }}\n'


def create_prts_icon(active: bool = True, paused: bool = False) -> QIcon:
    """动态绘制 64x64 纯白极简托盘图标：白底细炭灰边方块 + 中央状态方点。

    运行中为琥珀方点，待机为浅灰方点，暂停为两道深炭灰竖条。
    """
    size = 64
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    try:
        # 白底方块 + 深炭灰细边（任务栏深/浅色背景均清晰）
        painter.setPen(QPen(QColor("#2B2B2B"), 3.0))
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.drawRect(QRectF(5.5, 5.5, 53.0, 53.0))
        # 左上琥珀角标（呼应方案 B 品牌方点）
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#F59E0B")))
        painter.drawRect(QRectF(12.0, 12.0, 8.0, 8.0))
        if paused:
            painter.setBrush(QBrush(QColor("#2B2B2B")))
            painter.drawRect(QRectF(23.0, 22.0, 6.0, 20.0))
            painter.drawRect(QRectF(35.0, 22.0, 6.0, 20.0))
        else:
            painter.setBrush(QBrush(QColor("#F59E0B") if active else QColor("#BDBDBD")))
            painter.drawRect(QRectF(23.0, 23.0, 18.0, 18.0))
            painter.setPen(QPen(QColor("#2B2B2B"), 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawLine(QPointF(32.0, 46.0), QPointF(32.0, 52.0))
    finally:
        painter.end()

    return QIcon(pixmap)


class ControllerTray(QSystemTrayIcon):
    """ArkController 系统托盘控制器。"""

    open_dashboard_requested = Signal()
    toggle_pause_requested = Signal()
    toggle_overlay_requested = Signal(bool)
    toggle_shortcuts_mode_requested = Signal(bool)
    toggle_left_stick_dpad_requested = Signal(bool)
    toggle_hide_capture_requested = Signal(bool)
    open_calibration_requested = Signal()
    reload_config_requested = Signal()
    quit_requested = Signal()

    def __init__(
        self,
        *,
        overlay_enabled: bool = True,
        use_native_shortcuts: bool = True,
        left_stick_as_dpad: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        self._icon_active = create_prts_icon(active=True, paused=False)
        self._icon_idle = create_prts_icon(active=False, paused=False)
        self._icon_paused = create_prts_icon(active=True, paused=True)
        super().__init__(self._icon_idle, parent)

        self._menu = QMenu(parent)
        self._menu.setStyleSheet(TRAY_MENU_QSS + _menu_check_qss())
        self.setToolTip("ArkController — 卫戍协议手柄控制")

        # 顶部：打开 PRTS 主控台
        self._action_dashboard = QAction("打开 PRTS 主控台", self._menu)
        bold = QFont("Microsoft YaHei UI", 9, QFont.Weight.Bold)
        self._action_dashboard.setFont(bold)
        self._action_dashboard.triggered.connect(lambda _checked=False: self.open_dashboard_requested.emit())
        self._menu.addAction(self._action_dashboard)
        self._menu.addSeparator()

        # 状态展示项（只读）
        self._action_gamepad_status = QAction("手柄状态：未连接", self._menu)
        self._action_gamepad_status.setEnabled(False)
        self._menu.addAction(self._action_gamepad_status)

        self._action_window_status = QAction("游戏窗口：等待《明日方舟》启动", self._menu)
        self._action_window_status.setEnabled(False)
        self._menu.addAction(self._action_window_status)

        self._menu.addSeparator()

        # 暂停 / 恢复
        self._action_pause = QAction("暂停手柄控制", self._menu)
        self._action_pause.setCheckable(True)
        self._action_pause.setChecked(False)
        self._action_pause.triggered.connect(lambda _checked: self.toggle_pause_requested.emit())
        self._menu.addAction(self._action_pause)

        # 显示 / 隐藏悬浮 HUD
        self._action_overlay = QAction("显示悬浮 HUD", self._menu)
        self._action_overlay.setCheckable(True)
        self._action_overlay.setChecked(overlay_enabled)
        self._action_overlay.triggered.connect(self.toggle_overlay_requested.emit)
        self._menu.addAction(self._action_overlay)

        # 切换原生 PC 快捷键 / 纯鼠标坐标模式
        self._action_shortcuts = QAction("官方 PC 原生快捷键模式（取消为纯鼠标模式）", self._menu)
        self._action_shortcuts.setCheckable(True)
        self._action_shortcuts.setChecked(use_native_shortcuts)
        self._action_shortcuts.triggered.connect(self.toggle_shortcuts_mode_requested.emit)
        self._menu.addAction(self._action_shortcuts)

        # 左摇杆代替十字键导航模式
        self._action_left_stick_dpad = QAction("左摇杆代替十字键导航（按 LS 切换自由光标）", self._menu)
        self._action_left_stick_dpad.setCheckable(True)
        self._action_left_stick_dpad.setChecked(left_stick_as_dpad)
        self._action_left_stick_dpad.triggered.connect(self.toggle_left_stick_dpad_requested.emit)
        self._menu.addAction(self._action_left_stick_dpad)

        # 截屏 / 屏幕共享时隐藏 HUD
        self._action_hide_capture = QAction("截屏/共享时隐藏 HUD（关闭后共享可见）", self._menu)
        self._action_hide_capture.setCheckable(True)
        self._action_hide_capture.setChecked(True)
        self._action_hide_capture.triggered.connect(self._on_hide_capture_triggered)
        self._menu.addAction(self._action_hide_capture)

        self._menu.addSeparator()

        # 可视化校准向导
        self._action_calibrate = QAction("打开可视化校准向导 (Back + Start)", self._menu)
        self._action_calibrate.triggered.connect(lambda _checked: self.open_calibration_requested.emit())
        self._menu.addAction(self._action_calibrate)

        # 配置热重载
        self._action_reload = QAction("重新加载配置文件", self._menu)
        self._action_reload.triggered.connect(lambda _checked: self.reload_config_requested.emit())
        self._menu.addAction(self._action_reload)

        self._menu.addSeparator()

        # 退出
        self._action_quit = QAction("退出 ArkController", self._menu)
        self._action_quit.triggered.connect(lambda _checked: self.quit_requested.emit())
        self._menu.addAction(self._action_quit)

        self.setContextMenu(self._menu)
        self._last_icon_state: tuple[bool, bool] | None = None

        # 单击 / 双击托盘图标直接唤出 PRTS 主控台
        self.activated.connect(self._on_activated)

    @property
    def menu(self) -> QMenu:
        """返回托盘右键菜单（供测试与外部扩展）。"""
        return self._menu

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """处理托盘图标激活：左键单击或双击唤出主控台，右键交由系统弹出菜单。"""
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.open_dashboard_requested.emit()

    def _on_hide_capture_triggered(self, _checked: bool = False) -> None:
        self.toggle_hide_capture_requested.emit(self._action_hide_capture.isChecked())

    def update_status(
        self,
        *,
        gamepad_connected: bool,
        window_title: str | None,
        is_foreground: bool,
        paused: bool,
    ) -> None:
        """同步刷新托盘状态栏文本、图标颜色与 Tooltip。"""
        gp_text = "已连接 (Xbox XInput)" if gamepad_connected else "未连接"
        self._action_gamepad_status.setText(f"手柄状态：{gp_text}")

        if window_title:
            fg_tag = "前台激活" if is_foreground else "后台挂起"
            short_title = window_title if len(window_title) <= 20 else window_title[:20] + "…"
            win_text = f"{short_title} ({fg_tag})"
        else:
            win_text = "未检测到《明日方舟》"
        self._action_window_status.setText(f"游戏窗口：{win_text}")

        self._action_pause.blockSignals(True)
        self._action_pause.setChecked(paused)
        self._action_pause.setText("恢复手柄控制" if paused else "暂停手柄控制")
        self._action_pause.blockSignals(False)

        active_flag = gamepad_connected and bool(window_title)
        icon_state = (active_flag, paused)
        if icon_state != self._last_icon_state:
            self._last_icon_state = icon_state
            if paused:
                self.setIcon(self._icon_paused)
            elif active_flag:
                self.setIcon(self._icon_active)
            else:
                self.setIcon(self._icon_idle)

        state_summary = "已暂停" if paused else ("运行中" if active_flag else "待机中")
        self.setToolTip(f"ArkController ({state_summary})\n手柄：{gp_text}\n窗口：{win_text}")

    def sync_settings(
        self,
        *,
        overlay_enabled: bool,
        use_native_shortcuts: bool,
        paused: bool,
        left_stick_as_dpad: bool | None = None,
        hide_hud_from_capture: bool | None = None,
    ) -> None:
        """从主控状态反向同步托盘菜单勾选框状态（不触发回环信号）。"""
        if hide_hud_from_capture is not None:
            self._action_hide_capture.blockSignals(True)
            self._action_hide_capture.setChecked(bool(hide_hud_from_capture))
            self._action_hide_capture.blockSignals(False)

        self._action_overlay.blockSignals(True)
        self._action_overlay.setChecked(overlay_enabled)
        self._action_overlay.blockSignals(False)

        self._action_shortcuts.blockSignals(True)
        self._action_shortcuts.setChecked(use_native_shortcuts)
        self._action_shortcuts.blockSignals(False)

        if left_stick_as_dpad is not None:
            self._action_left_stick_dpad.blockSignals(True)
            self._action_left_stick_dpad.setChecked(left_stick_as_dpad)
            self._action_left_stick_dpad.blockSignals(False)

        self._action_pause.blockSignals(True)
        self._action_pause.setChecked(paused)
        self._action_pause.setText("恢复手柄控制" if paused else "暂停手柄控制")
        self._action_pause.blockSignals(False)

    def show_notification(self, title: str, message: str, duration_ms: int = 2500) -> None:
        """安全发送系统托盘气泡提示。"""
        try:
            if self.isVisible() and self.supportsMessages():
                self.showMessage(title, message, self.icon(), duration_ms)
        except Exception:
            pass
