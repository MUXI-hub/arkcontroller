"""ArkController 系统托盘图标与常驻控制菜单。

内置纯代码绘制的罗德岛 PRTS 菱形矢量托盘图标（不依赖外部 .ico 文件），菜单采用明日方舟深色高对比风格，
支持单击 / 双击托盘图标唤出 PRTS 主控台、实时展示手柄/游戏窗口绑定状态、一键暂停恢复、开关 HUD、
切换原生快捷键/纯鼠标模式、打开校准向导与热重载配置。
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QIcon,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import QMenu, QSystemTrayIcon, QWidget

TRAY_MENU_QSS = """
QMenu {
    background-color: #0B0E14;
    color: #F4F6F9;
    border: 1px solid #00C8FF;
    padding: 6px 0px;
    font-family: "Microsoft YaHei UI";
    font-size: 9pt;
}
QMenu::item {
    padding: 7px 28px 7px 26px;
    border-left: 3px solid transparent;
    background: transparent;
}
QMenu::item:selected {
    background-color: rgba(0, 200, 255, 46);
    border-left: 3px solid #FF9800;
    color: #FFFFFF;
}
QMenu::item:disabled {
    color: #8A99AD;
}
QMenu::separator {
    height: 1px;
    background: #1F2A38;
    margin: 5px 10px;
}
QMenu::indicator {
    width: 12px;
    height: 12px;
    left: 8px;
}
QMenu::indicator:checked {
    background-color: #00C8FF;
    border: 1px solid #F4F6F9;
}
QMenu::indicator:unchecked {
    background-color: #121822;
    border: 1px solid #2C3A4D;
}
"""


def _diamond(cx: float, cy: float, r: float) -> QPolygonF:
    return QPolygonF([QPointF(cx, cy - r), QPointF(cx + r, cy), QPointF(cx, cy + r), QPointF(cx - r, cy)])


def create_prts_icon(active: bool = True, paused: bool = False) -> QIcon:
    """动态绘制 64x64 罗德岛 PRTS 菱形矢量托盘图标（无需外部图标资源）。"""
    size = 64
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    try:
        accent = (
            QColor("#FF9800")
            if paused
            else (QColor("#00C8FF") if active else QColor("#64748B"))
        )

        # 斜切角黑曜石底座（左上 / 右下 45° 切角）
        base = QPolygonF(
            [
                QPointF(16, 3),
                QPointF(61, 3),
                QPointF(61, 48),
                QPointF(48, 61),
                QPointF(3, 61),
                QPointF(3, 16),
            ]
        )
        painter.setPen(QPen(accent, 2.6))
        painter.setBrush(QBrush(QColor("#0B0E14")))
        painter.drawPolygon(base)

        # 外层菱形骨架
        outer_pen = QPen(accent, 2.4)
        outer_pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(outer_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(_diamond(32, 32, 22))

        # 内层渐变实心菱形
        grad = QLinearGradient(QPointF(32, 18), QPointF(32, 46))
        grad.setColorAt(0.0, accent.lighter(130))
        grad.setColorAt(1.0, accent.darker(150))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(grad))
        painter.drawPolygon(_diamond(32, 32, 13))

        # 黑芯 + 白色十字核
        painter.setBrush(QBrush(QColor("#0B0E14")))
        painter.drawPolygon(_diamond(32, 32, 6))
        cross_pen = QPen(QColor("#F4F6F9"), 2.2)
        cross_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(cross_pen)
        painter.drawLine(QPointF(28, 32), QPointF(36, 32))
        painter.drawLine(QPointF(32, 28), QPointF(32, 36))

        # 右下状态角标
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(accent))
        painter.drawPolygon(QPolygonF([QPointF(61, 50), QPointF(61, 61), QPointF(50, 61)]))
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
        self._menu.setStyleSheet(TRAY_MENU_QSS)
        self.setToolTip("ArkController — 卫戍协议手柄控制")

        # 顶部：打开 PRTS 主控台
        self._action_dashboard = QAction("◆  打开 PRTS 主控台", self._menu)
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
    ) -> None:
        """从主控状态反向同步托盘菜单勾选框状态（不触发回环信号）。"""
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
