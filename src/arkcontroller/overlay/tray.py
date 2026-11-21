"""ArkController 系统托盘图标与常驻控制菜单。

内置纯代码绘制的 PRTS 工业风格矢量托盘图标（不依赖外部 .ico 文件），
支持实时展示手柄/游戏窗口绑定状态、一键暂停恢复、开关 HUD、切换原生快捷键/纯鼠标模式、打开校准向导与热重载配置。
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QMenu, QSystemTrayIcon, QWidget


def create_prts_icon(active: bool = True, paused: bool = False) -> QIcon:
    """动态绘制 64x64 PRTS 科技风菱形准星托盘图标（无需外部图标资源）。"""
    size = 64
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    try:
        accent = (
            QColor("#FFB300")
            if paused
            else (QColor("#00E5FF") if active else QColor("#64748B"))
        )

        # 深色八角底座
        bg_poly = QPolygonF(
            [
                QPointF(18, 4),
                QPointF(46, 4),
                QPointF(60, 18),
                QPointF(60, 46),
                QPointF(46, 60),
                QPointF(18, 60),
                QPointF(4, 46),
                QPointF(4, 18),
            ]
        )
        painter.setPen(QPen(accent, 3.0))
        painter.setBrush(QBrush(QColor("#0F172A")))
        painter.drawPolygon(bg_poly)

        # 内层 PRTS 菱形骨架
        diamond = QPolygonF(
            [
                QPointF(32, 12),
                QPointF(52, 32),
                QPointF(32, 52),
                QPointF(12, 32),
            ]
        )
        inner_fill = QColor(accent)
        inner_fill.setAlpha(42)
        painter.setPen(QPen(accent, 2.5))
        painter.setBrush(QBrush(inner_fill))
        painter.drawPolygon(diamond)

        # 十字手柄方向键核心标识
        cross_pen = QPen(QColor("#F8FAFC"), 4.2)
        cross_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(cross_pen)
        painter.drawLine(QPointF(23, 32), QPointF(41, 32))
        painter.drawLine(QPointF(32, 23), QPointF(32, 41))

        # 中心核心亮点
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(accent))
        painter.drawEllipse(QPointF(32, 32), 3.6, 3.6)
    finally:
        painter.end()

    return QIcon(pixmap)


class ControllerTray(QSystemTrayIcon):
    """ArkController 系统托盘控制器。"""

    toggle_pause_requested = Signal()
    toggle_overlay_requested = Signal(bool)
    toggle_shortcuts_mode_requested = Signal(bool)
    open_calibration_requested = Signal()
    reload_config_requested = Signal()
    quit_requested = Signal()

    def __init__(
        self,
        *,
        overlay_enabled: bool = True,
        use_native_shortcuts: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        self._icon_active = create_prts_icon(active=True, paused=False)
        self._icon_idle = create_prts_icon(active=False, paused=False)
        self._icon_paused = create_prts_icon(active=True, paused=True)
        super().__init__(self._icon_idle, parent)

        self._menu = QMenu(parent)
        self.setToolTip("ArkController — 卫戍协议手柄控制")

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
    ) -> None:
        """从主控状态反向同步托盘菜单勾选框状态（不触发回环信号）。"""
        self._action_overlay.blockSignals(True)
        self._action_overlay.setChecked(overlay_enabled)
        self._action_overlay.blockSignals(False)

        self._action_shortcuts.blockSignals(True)
        self._action_shortcuts.setChecked(use_native_shortcuts)
        self._action_shortcuts.blockSignals(False)

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
