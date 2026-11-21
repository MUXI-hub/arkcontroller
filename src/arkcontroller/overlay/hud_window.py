"""PRTS 工业科技风透明鼠标穿透 HUD 悬浮层。

基于 PySide6 构建无边框、置顶、完全不抢夺游戏焦点、不拦截鼠标输入的 OSD 叠加层，
实时渲染焦点高亮框、透视棋盘格选区、抓取源连线、四向定朝向轮盘、准备防误触进度环及底部 Xbox 键位提示栏。
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import math
import sys
import time
from typing import Sequence

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QGuiApplication,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QShowEvent,
)
from PySide6.QtWidgets import QWidget

from arkcontroller.control.deploy_fsm import DeployState
from arkcontroller.control.dispatcher import HUDStateSnapshot, PromptItem
from arkcontroller.platform.window_tracker import ViewportRect
from arkcontroller.vision.scene_detector import GameScene

# Win32 扩展窗口样式常量
GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x00000008
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000

HWND_TOPMOST = ctypes.c_void_p(-1)
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
SWP_NOOWNERZORDER = 0x0200
WDA_EXCLUDEFROMCAPTURE = 0x00000011

# PRTS 工业科技风调色板
COLOR_CYAN = QColor("#00E5FF")
COLOR_AMBER = QColor("#FFB300")
COLOR_GREEN = QColor("#4CAF50")
COLOR_RED = QColor("#F44336")
COLOR_BLUE = QColor("#2196F3")
COLOR_YELLOW = QColor("#FFC107")
COLOR_TEXT_PRIMARY = QColor("#F1F5F9")
COLOR_TEXT_MUTED = QColor("#94A3B8")
COLOR_CAPSULE_BG = QColor(12, 16, 22, 212)
COLOR_CAPSULE_BORDER = QColor(0, 229, 255, 110)

SCENE_DISPLAY_NAMES: dict[GameScene, str] = {
    GameScene.BATTLE_PREP: "休整运营",
    GameScene.MODAL_CHOICE: "决策弹窗",
    GameScene.MODAL_CONFIRM: "回合确认",
    GameScene.BATTLE_COMBAT: "作战中",
    GameScene.UNKNOWN_FREE_CURSOR: "自由鼠标",
}

_FREE_CURSOR_SYNONYM_LABELS: frozenset[str] = frozenset({"自由鼠标", "自由光标"})

FACE_BUTTON_COLORS: dict[str, tuple[QColor, QColor]] = {
    "A": (COLOR_GREEN, QColor("#FFFFFF")),
    "B": (COLOR_RED, QColor("#FFFFFF")),
    "X": (COLOR_BLUE, QColor("#FFFFFF")),
    "Y": (COLOR_YELLOW, QColor("#111827")),
}


def _is_windows_qpa() -> bool:
    """判断当前是否运行在真实 Windows 桌面 QPA（排除 offscreen 等无头测试平台）。"""
    if sys.platform != "win32":
        return False
    app = QGuiApplication.instance()
    if app is None:
        return False
    return app.platformName().lower() == "windows"


def align_widget_to_physical_rect(
    widget: QWidget,
    left: int,
    top: int,
    width: int,
    height: int,
    *,
    topmost: bool = True,
    no_activate: bool = True,
) -> None:
    """将 Qt 窗口精确对齐至 Win32 物理像素区域（兼容高分屏 DPI 缩放与多显示器）。"""
    width = max(1, int(width))
    height = max(1, int(height))
    left = int(left)
    top = int(top)

    screen = widget.screen() or QGuiApplication.primaryScreen()
    dpr = float(screen.devicePixelRatio()) if screen is not None else 1.0
    if dpr <= 0.0:
        dpr = 1.0

    logical_x = int(round(left / dpr))
    logical_y = int(round(top / dpr))
    logical_w = max(1, int(round(width / dpr)))
    logical_h = max(1, int(round(height / dpr)))
    widget.setGeometry(logical_x, logical_y, logical_w, logical_h)

    if _is_windows_qpa() and widget.isVisible():
        try:
            hwnd = int(widget.winId())
            if hwnd:
                user32 = ctypes.windll.user32
                user32.SetWindowPos.argtypes = [
                    wintypes.HWND,
                    wintypes.HWND,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    wintypes.UINT,
                ]
                user32.SetWindowPos.restype = wintypes.BOOL
                flags = SWP_NOOWNERZORDER
                if no_activate:
                    flags |= SWP_NOACTIVATE
                insert_after = HWND_TOPMOST if topmost else ctypes.c_void_p(0)
                user32.SetWindowPos(
                    wintypes.HWND(hwnd),
                    insert_after,
                    left,
                    top,
                    width,
                    height,
                    flags,
                )
        except Exception:
            # 平台 API 失败时静默回退至 Qt setGeometry
            pass


def apply_win32_click_through(hwnd: int) -> bool:
    """对给定 Win32 窗口句柄双重加固透明鼠标穿透与不抢焦点样式。"""
    if not hwnd or not _is_windows_qpa():
        return False
    try:
        user32 = ctypes.windll.user32
        user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetWindowLongW.restype = ctypes.c_long
        user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
        user32.SetWindowLongW.restype = ctypes.c_long
        user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        user32.SetWindowPos.restype = wintypes.BOOL

        ex_style = user32.GetWindowLongW(wintypes.HWND(hwnd), GWL_EXSTYLE)
        new_style = (
            ex_style
            | WS_EX_LAYERED
            | WS_EX_TRANSPARENT
            | WS_EX_NOACTIVATE
            | WS_EX_TOPMOST
        )
        user32.SetWindowLongW(wintypes.HWND(hwnd), GWL_EXSTYLE, new_style)
        user32.SetWindowPos(
            wintypes.HWND(hwnd),
            HWND_TOPMOST,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED,
        )
        try:
            user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
            user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
            user32.SetWindowDisplayAffinity(wintypes.HWND(hwnd), WDA_EXCLUDEFROMCAPTURE)
        except Exception:
            # 兼容不支持 WDA_EXCLUDEFROMCAPTURE 的旧版系统或受限测试环境
            pass
        return True
    except Exception:
        return False


class HUDWindow(QWidget):
    """PRTS 风格透明鼠标穿透 HUD 悬浮层窗口。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._overlay_enabled: bool = True
        self._viewport: ViewportRect | None = None
        self._snapshot: HUDStateSnapshot | None = None
        self._last_geom_rect: tuple[int, int, int, int] | None = None
        self._last_repaint_ts: float = 0.0
        self._win32_styled: bool = False

        self.setWindowTitle("ArkController HUD")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._font_title = QFont("Microsoft YaHei UI", 10, QFont.Weight.Bold)
        self._font_badge = QFont("Segoe UI", 9, QFont.Weight.Bold)
        self._font_prompt = QFont("Microsoft YaHei UI", 9, QFont.Weight.Medium)
        self._font_small = QFont("Microsoft YaHei UI", 8, QFont.Weight.Bold)

    @property
    def overlay_enabled(self) -> bool:
        """返回用户是否开启了 HUD 悬浮显示。"""
        return self._overlay_enabled

    def set_overlay_enabled(self, enabled: bool) -> None:
        """开关悬浮层显示状态。"""
        self._overlay_enabled = bool(enabled)
        if not self._overlay_enabled and self.isVisible():
            self.hide()
        elif self._overlay_enabled and self._viewport and self._snapshot and self._snapshot.visible:
            self._ensure_shown_and_aligned(self._viewport)
            self.update()

    def showEvent(self, event: QShowEvent) -> None:
        """窗口显示时通过 Win32 SetWindowLongW 双重加固鼠标穿透与不抢焦点属性。"""
        super().showEvent(event)
        try:
            hwnd = int(self.winId())
        except Exception:
            hwnd = 0
        if hwnd:
            self._win32_styled = apply_win32_click_through(hwnd)
        if self._viewport is not None:
            align_widget_to_physical_rect(
                self,
                self._viewport.active_left,
                self._viewport.active_top,
                self._viewport.active_width,
                self._viewport.active_height,
                topmost=True,
                no_activate=True,
            )

    def update_state(
        self,
        viewport: ViewportRect | None,
        snapshot: HUDStateSnapshot | None,
    ) -> None:
        """接收最新视口与调度器快照并刷新透明悬浮层绘制。"""
        prev_snapshot = self._snapshot
        self._viewport = viewport
        self._snapshot = snapshot

        should_show = (
            self._overlay_enabled
            and viewport is not None
            and snapshot is not None
            and snapshot.visible
            and viewport.active_width > 0
            and viewport.active_height > 0
        )

        if not should_show:
            if self.isVisible():
                self.hide()
            return

        assert viewport is not None
        assert snapshot is not None
        self._ensure_shown_and_aligned(viewport)

        # 判断是否需要重绘：快照变化立即重绘；存在动态虚线/读条动画时以 ~60Hz 节流重绘
        now = time.monotonic()
        has_animation = (
            snapshot.holding_source_norm is not None
            or snapshot.deploy_state == DeployState.AIMING_DIRECTION
            or snapshot.ready_hold_progress > 0.0
        )
        if snapshot != prev_snapshot or (has_animation and (now - self._last_repaint_ts) >= 0.016):
            self._last_repaint_ts = now
            self.update()

    def _ensure_shown_and_aligned(self, viewport: ViewportRect) -> None:
        """按需显示并对齐至游戏 16:9 有效视口区域。"""
        target_rect = (
            viewport.active_left,
            viewport.active_top,
            viewport.active_width,
            viewport.active_height,
        )
        geom_changed = target_rect != self._last_geom_rect
        if geom_changed or not self.isVisible():
            self._last_geom_rect = target_rect
            align_widget_to_physical_rect(
                self,
                *target_rect,
                topmost=True,
                no_activate=True,
            )
            if not self.isVisible():
                self.show()
                align_widget_to_physical_rect(
                    self,
                    *target_rect,
                    topmost=True,
                    no_activate=True,
                )

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        """抗锯齿绘制 PRTS 科技风 HUD 元素。"""
        snapshot = self._snapshot
        if snapshot is None or not snapshot.visible:
            return

        w = float(self.width())
        h = float(self.height())
        if w <= 1.0 or h <= 1.0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            # 1. 抓取源琥珀色虚线框 + 动态引导连线
            if snapshot.holding_source_norm is not None:
                self._draw_holding_source_and_trajectory(painter, w, h, snapshot)

            # 2. 焦点多边形高亮框（支持标准矩形与 3D 透视棋盘梯形格）
            if (
                snapshot.focus_polygon_norm
                and len(snapshot.focus_polygon_norm) >= 3
                and not snapshot.free_cursor_active
            ):
                accent = (
                    COLOR_AMBER
                    if snapshot.holding_source_norm is not None
                    else COLOR_CYAN
                )
                self._draw_focus_polygon(painter, w, h, snapshot.focus_polygon_norm, accent)

            # 3. 自由光标模式准星环
            if snapshot.free_cursor_active:
                self._draw_free_cursor_reticle(painter, w, h, snapshot.cursor_norm)

            # 4. 两段式部署：四向定朝向十字轮盘
            if (
                snapshot.deploy_state == DeployState.AIMING_DIRECTION
                and snapshot.aim_origin_norm is not None
            ):
                self._draw_direction_wheel(
                    painter,
                    w,
                    h,
                    snapshot.aim_origin_norm,
                    snapshot.aim_preview_dir,
                )

            # 5. 顶部居中磨砂状态胶囊 + Toast 提示
            self._draw_top_status_capsule(painter, w, h, snapshot)

            # 6. 准备开战防误触长按进度环
            if snapshot.ready_hold_progress > 0.0:
                self._draw_ready_hold_ring(painter, w, h, snapshot.ready_hold_progress)

            # 7. 底部紧凑 Xbox 彩色键位提示条
            if snapshot.prompts:
                self._draw_bottom_prompt_bar(painter, w, h, snapshot.prompts)
        finally:
            painter.end()

    def _draw_focus_polygon(
        self,
        painter: QPainter,
        w: float,
        h: float,
        polygon_norm: Sequence[tuple[float, float]],
        color: QColor,
    ) -> None:
        """绘制青蓝/琥珀半透明发光焦点多边形与 PRTS 工业角标。"""
        pts = [QPointF(u * w, v * h) for u, v in polygon_norm]
        poly = QPolygonF(pts)

        # 内填半透明高光
        fill_color = QColor(color)
        fill_color.setAlpha(30)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(fill_color))
        painter.drawPolygon(poly)

        # 外层柔和发光描边
        glow_color = QColor(color)
        glow_color.setAlpha(68)
        glow_pen = QPen(glow_color, 6.0)
        glow_pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(glow_pen)
        painter.drawPolygon(poly)

        # 主轮廓锐利描边
        main_color = QColor(color)
        main_color.setAlpha(235)
        main_pen = QPen(main_color, 2.0)
        main_pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(main_pen)
        painter.drawPolygon(poly)

        # 沿多边形边向量绘制 4 个锐利角标（精准适配透视倾斜格子）
        bracket_pen = QPen(QColor("#FFFFFF"), 2.6)
        bracket_pen.setCapStyle(Qt.PenCapStyle.SquareCap)
        painter.setPen(bracket_pen)
        n = len(pts)
        for i in range(n):
            p_curr = pts[i]
            p_prev = pts[(i - 1) % n]
            p_next = pts[(i + 1) % n]
            for neighbor in (p_prev, p_next):
                dx = neighbor.x() - p_curr.x()
                dy = neighbor.y() - p_curr.y()
                seg_len = math.hypot(dx, dy)
                if seg_len < 1e-3:
                    continue
                arm = min(13.0, seg_len * 0.28)
                end_pt = QPointF(
                    p_curr.x() + (dx / seg_len) * arm,
                    p_curr.y() + (dy / seg_len) * arm,
                )
                painter.drawLine(p_curr, end_pt)

    def _draw_holding_source_and_trajectory(
        self,
        painter: QPainter,
        w: float,
        h: float,
        snapshot: HUDStateSnapshot,
    ) -> None:
        """绘制琥珀色抓取源虚线槽位框及指向当前焦点的动态虚线轨迹。"""
        assert snapshot.holding_source_norm is not None
        su, sv = snapshot.holding_source_norm
        sx, sy = su * w, sv * h

        box_w = max(44.0, min(96.0, w * 0.068))
        box_h = max(44.0, min(96.0, h * 0.11))
        src_rect = QRectF(sx - box_w / 2.0, sy - box_h / 2.0, box_w, box_h)

        # 源槽位半透明底色
        fill = QColor(COLOR_AMBER)
        fill.setAlpha(36)
        painter.setBrush(QBrush(fill))
        dash_pen = QPen(COLOR_AMBER, 2.2, Qt.PenStyle.DashLine)
        dash_pen.setDashPattern([6.0, 4.0])
        dash_pen.setDashOffset((time.monotonic() * 24.0) % 10.0)
        painter.setPen(dash_pen)
        painter.drawRoundedRect(src_rect, 6.0, 6.0)

        # 源槽位顶部标签
        if snapshot.deploy_state == DeployState.HOLDING_STRATEGY:
            tag_text = "策略源"
        elif snapshot.deploy_state == DeployState.HOLDING_EQUIP:
            tag_text = "装备源"
        else:
            tag_text = "抓取源"
        painter.setFont(self._font_small)
        fm = QFontMetrics(self._font_small)
        tw = fm.horizontalAdvance(tag_text) + 10
        th = fm.height() + 4
        tag_rect = QRectF(sx - tw / 2.0, src_rect.top() - th - 4.0, tw, th)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(24, 18, 8, 220)))
        painter.drawRoundedRect(tag_rect, 4.0, 4.0)
        painter.setPen(QPen(COLOR_AMBER))
        painter.drawText(tag_rect, Qt.AlignmentFlag.AlignCenter, tag_text)

        # 计算目标中心点（焦点多边形质心，否则使用光标坐标）
        if snapshot.focus_polygon_norm and not snapshot.free_cursor_active:
            tx = sum(u for u, _ in snapshot.focus_polygon_norm) / len(snapshot.focus_polygon_norm) * w
            ty = sum(v for _, v in snapshot.focus_polygon_norm) / len(snapshot.focus_polygon_norm) * h
        else:
            tx = snapshot.cursor_norm[0] * w
            ty = snapshot.cursor_norm[1] * h

        dist = math.hypot(tx - sx, ty - sy)
        if dist >= 16.0:
            # 发光底线
            glow_pen = QPen(QColor(255, 179, 0, 75), 6.0)
            painter.setPen(glow_pen)
            painter.drawLine(QPointF(sx, sy), QPointF(tx, ty))

            # 动态流光虚线
            traj_pen = QPen(COLOR_AMBER, 2.5, Qt.PenStyle.CustomDashLine)
            traj_pen.setDashPattern([8.0, 6.0])
            traj_pen.setDashOffset(-(time.monotonic() * 48.0) % 14.0)
            painter.setPen(traj_pen)
            painter.drawLine(QPointF(sx, sy), QPointF(tx, ty))

            # 终点方向小箭头
            angle = math.atan2(ty - sy, tx - sx)
            arrow_len = 11.0
            p1 = QPointF(
                tx - arrow_len * math.cos(angle - math.pi / 6.0),
                ty - arrow_len * math.sin(angle - math.pi / 6.0),
            )
            p2 = QPointF(
                tx - arrow_len * math.cos(angle + math.pi / 6.0),
                ty - arrow_len * math.sin(angle + math.pi / 6.0),
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(COLOR_AMBER))
            painter.drawPolygon(QPolygonF([QPointF(tx, ty), p1, p2]))

    def _draw_free_cursor_reticle(
        self,
        painter: QPainter,
        w: float,
        h: float,
        cursor_norm: tuple[float, float],
    ) -> None:
        """自由光标模式下的轻量 PRTS 空心准星。"""
        cx, cy = cursor_norm[0] * w, cursor_norm[1] * h
        radius = 12.0
        ring_pen = QPen(QColor(0, 229, 255, 190), 1.8)
        painter.setPen(ring_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

        tick = 5.0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            painter.drawLine(
                QPointF(cx + dx * (radius + 2.0), cy + dy * (radius + 2.0)),
                QPointF(cx + dx * (radius + 2.0 + tick), cy + dy * (radius + 2.0 + tick)),
            )

    def _draw_direction_wheel(
        self,
        painter: QPainter,
        w: float,
        h: float,
        origin_norm: tuple[float, float],
        preview_dir: str | None,
    ) -> None:
        """绘制两段式定朝向十字四向轮盘。"""
        cx = origin_norm[0] * w
        cy = origin_norm[1] * h
        outer_r = max(52.0, min(94.0, min(w, h) * 0.085))
        inner_r = outer_r * 0.34

        # 背景磨砂圆盘
        painter.setPen(QPen(QColor(0, 229, 255, 130), 1.8))
        painter.setBrush(QBrush(QColor(10, 15, 22, 196)))
        painter.drawEllipse(QPointF(cx, cy), outer_r, outer_r)

        # 内核中心圆
        painter.setPen(QPen(QColor(255, 255, 255, 160), 1.5))
        painter.setBrush(QBrush(QColor(0, 229, 255, 55)))
        painter.drawEllipse(QPointF(cx, cy), inner_r * 0.62, inner_r * 0.62)

        directions: list[tuple[str, float, float, str]] = [
            ("up", 0.0, -1.0, "上"),
            ("down", 0.0, 1.0, "下"),
            ("left", -1.0, 0.0, "左"),
            ("right", 1.0, 0.0, "右"),
        ]
        for dir_key, dx, dy, label in directions:
            active = preview_dir == dir_key
            tip = QPointF(cx + dx * (outer_r * 0.86), cy + dy * (outer_r * 0.86))
            base_c = QPointF(cx + dx * (outer_r * 0.44), cy + dy * (outer_r * 0.44))
            px, py = -dy, dx  # 垂直法向量
            half_w = outer_r * (0.24 if active else 0.19)
            wing1 = QPointF(base_c.x() + px * half_w, base_c.y() + py * half_w)
            wing2 = QPointF(base_c.x() - px * half_w, base_c.y() - py * half_w)
            arrow_poly = QPolygonF([tip, wing1, wing2])

            if active:
                glow_pen = QPen(QColor(255, 179, 0, 120), 6.0)
                painter.setPen(glow_pen)
                painter.setBrush(QBrush(COLOR_AMBER))
                painter.drawPolygon(arrow_poly)
                painter.setPen(QPen(QColor("#FFFFFF"), 1.8))
                painter.drawPolygon(arrow_poly)
            else:
                painter.setPen(QPen(QColor(0, 229, 255, 180), 1.4))
                painter.setBrush(QBrush(QColor(0, 229, 255, 60)))
                painter.drawPolygon(arrow_poly)

            # 方向微文字
            painter.setFont(self._font_small)
            txt_pos = QPointF(cx + dx * (outer_r * 0.61), cy + dy * (outer_r * 0.61))
            txt_rect = QRectF(txt_pos.x() - 12.0, txt_pos.y() - 10.0, 24.0, 20.0)
            painter.setPen(QPen(QColor("#111827") if active else COLOR_TEXT_PRIMARY))
            painter.drawText(txt_rect, Qt.AlignmentFlag.AlignCenter, label)

        # 底部引导说明条
        hint = "推摇杆 / 十字键定朝向  ·  B 取消部署"
        painter.setFont(self._font_small)
        fm = QFontMetrics(self._font_small)
        hw = fm.horizontalAdvance(hint) + 20.0
        hh = fm.height() + 8.0
        hint_rect = QRectF(cx - hw / 2.0, cy + outer_r + 8.0, hw, hh)
        painter.setPen(QPen(QColor(255, 179, 0, 170), 1.2))
        painter.setBrush(QBrush(COLOR_CAPSULE_BG))
        painter.drawRoundedRect(hint_rect, 6.0, 6.0)
        painter.setPen(QPen(COLOR_AMBER))
        painter.drawText(hint_rect, Qt.AlignmentFlag.AlignCenter, hint)

    @staticmethod
    def _format_top_status_text(snapshot: HUDStateSnapshot) -> str:
        """格式化顶部状态胶囊文案，消除自由鼠标与自由光标同义重复段。"""
        scene_name = SCENE_DISPLAY_NAMES.get(snapshot.scene, snapshot.scene.value)
        segments: list[str] = [f"PRTS // {scene_name}"]
        focus_label = (snapshot.focus_label or "").strip()

        if snapshot.scene == GameScene.UNKNOWN_FREE_CURSOR:
            if focus_label and focus_label not in _FREE_CURSOR_SYNONYM_LABELS:
                segments.append(focus_label)
        else:
            if focus_label and not (
                snapshot.free_cursor_active and focus_label in _FREE_CURSOR_SYNONYM_LABELS
            ):
                segments.append(focus_label)
            view_state = getattr(snapshot, "view_state", "normal")
            if view_state == "shop_open":
                segments.append("商店展开")
            elif view_state == "unit_selected":
                segments.append("角色详情缩放")
            if snapshot.free_cursor_active:
                segments.append("自由光标")

        if snapshot.scene_locked_manual:
            segments.append("手动锁层")
        if snapshot.paused:
            segments.append("已暂停")

        return "  ·  ".join(segments)

    def _draw_top_status_capsule(
        self,
        painter: QPainter,
        w: float,
        h: float,
        snapshot: HUDStateSnapshot,
    ) -> None:
        """绘制顶部居中微型磨砂黑胶囊及可选 Toast 提示。"""
        text = self._format_top_status_text(snapshot)
        painter.setFont(self._font_title)
        fm = QFontMetrics(self._font_title)
        text_w = fm.horizontalAdvance(text)
        capsule_h = max(30.0, fm.height() + 12.0)
        capsule_w = min(w * 0.88, text_w + 46.0)
        top_y = max(8.0, h * 0.014)
        capsule_rect = QRectF((w - capsule_w) / 2.0, top_y, capsule_w, capsule_h)

        # 胶囊底板
        border_col = COLOR_RED if snapshot.paused else (
            COLOR_AMBER if snapshot.scene_locked_manual else COLOR_CAPSULE_BORDER
        )
        painter.setPen(QPen(border_col, 1.4))
        painter.setBrush(QBrush(COLOR_CAPSULE_BG))
        painter.drawRoundedRect(capsule_rect, capsule_h / 2.0, capsule_h / 2.0)

        # 状态指示灯点
        dot_col = COLOR_RED if snapshot.paused else (
            COLOR_AMBER
            if snapshot.scene_locked_manual or snapshot.deploy_state != DeployState.IDLE
            else COLOR_CYAN
        )
        dot_center = QPointF(capsule_rect.left() + 16.0, capsule_rect.center().y())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(dot_col))
        painter.drawEllipse(dot_center, 4.5, 4.5)

        # 文字内容
        text_rect = QRectF(
            capsule_rect.left() + 28.0,
            capsule_rect.top(),
            capsule_rect.width() - 38.0,
            capsule_rect.height(),
        )
        painter.setPen(QPen(COLOR_TEXT_PRIMARY))
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            text,
        )

        # Toast 次级消息胶囊
        if snapshot.toast_message:
            painter.setFont(self._font_prompt)
            tfm = QFontMetrics(self._font_prompt)
            tw = min(w * 0.8, tfm.horizontalAdvance(snapshot.toast_message) + 32.0)
            th = tfm.height() + 10.0
            toast_rect = QRectF((w - tw) / 2.0, capsule_rect.bottom() + 6.0, tw, th)
            painter.setPen(QPen(QColor(0, 229, 255, 160), 1.2))
            painter.setBrush(QBrush(QColor(12, 20, 30, 225)))
            painter.drawRoundedRect(toast_rect, 6.0, 6.0)
            # 左侧亮条
            accent_bar = QRectF(toast_rect.left() + 2.0, toast_rect.top() + 4.0, 3.0, th - 8.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(COLOR_CYAN))
            painter.drawRoundedRect(accent_bar, 1.5, 1.5)
            painter.setPen(QPen(COLOR_TEXT_PRIMARY))
            painter.drawText(toast_rect, Qt.AlignmentFlag.AlignCenter, snapshot.toast_message)

    def _draw_ready_hold_ring(
        self,
        painter: QPainter,
        w: float,
        h: float,
        progress: float,
    ) -> None:
        """绘制准备开战防误触长按琥珀色进度环。"""
        clamped = max(0.0, min(1.0, float(progress)))
        radius = max(32.0, min(46.0, min(w, h) * 0.055))
        cx = w / 2.0
        cy = max(94.0, h * 0.16)

        # 半透明圆形衬底
        painter.setPen(QPen(QColor(255, 179, 0, 75), 5.0))
        painter.setBrush(QBrush(QColor(14, 18, 24, 215)))
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

        # 进度圆弧（从 12 点方向顺时针增长）
        arc_rect = QRectF(cx - radius, cy - radius, radius * 2.0, radius * 2.0)
        arc_pen = QPen(COLOR_AMBER, 5.5)
        arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(arc_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        span_angle = int(round(-clamped * 360.0 * 16.0))
        painter.drawArc(arc_rect, 90 * 16, span_angle)

        # 百分比读数
        painter.setFont(self._font_title)
        painter.setPen(QPen(COLOR_AMBER))
        painter.drawText(arc_rect, Qt.AlignmentFlag.AlignCenter, f"{int(clamped * 100)}%")

        # 下方提示文案
        label = "长按 Start 准备开战"
        painter.setFont(self._font_small)
        fm = QFontMetrics(self._font_small)
        lw = fm.horizontalAdvance(label) + 18.0
        lh = fm.height() + 6.0
        lrect = QRectF(cx - lw / 2.0, cy + radius + 8.0, lw, lh)
        painter.setPen(QPen(QColor(255, 179, 0, 170), 1.0))
        painter.setBrush(QBrush(COLOR_CAPSULE_BG))
        painter.drawRoundedRect(lrect, 5.0, 5.0)
        painter.setPen(QPen(COLOR_TEXT_PRIMARY))
        painter.drawText(lrect, Qt.AlignmentFlag.AlignCenter, label)

    def _draw_bottom_prompt_bar(
        self,
        painter: QPainter,
        w: float,
        h: float,
        prompts: Sequence[PromptItem],
    ) -> None:
        """绘制底部紧凑 Xbox 彩色键位提示条。"""
        badge_fm = QFontMetrics(self._font_badge)
        label_fm = QFontMetrics(self._font_prompt)

        item_layouts: list[tuple[list[str], str, float]] = []
        item_gap = 16.0
        for item in prompts:
            tokens = [tok.strip() for tok in item.button.split("+") if tok.strip()]
            if not tokens:
                tokens = [item.button]
            tokens_w = self._measure_tokens_width(tokens, badge_fm)
            label_w = label_fm.horizontalAdvance(item.label)
            total_item_w = tokens_w + 6.0 + label_w
            item_layouts.append((tokens, item.label, total_item_w))

        pad_x = 18.0
        bar_h = 34.0
        content_w = sum(x[2] for x in item_layouts) + item_gap * max(0, len(item_layouts) - 1)
        bar_w = min(w * 0.96, content_w + pad_x * 2.0)
        bottom_margin = max(8.0, h * 0.015)
        bar_rect = QRectF((w - bar_w) / 2.0, h - bar_h - bottom_margin, bar_w, bar_h)

        # 背景磨砂底条
        path = QPainterPath()
        path.addRoundedRect(bar_rect, 10.0, 10.0)
        painter.setPen(QPen(QColor(255, 255, 255, 42), 1.2))
        painter.setBrush(QBrush(COLOR_CAPSULE_BG))
        painter.drawPath(path)

        # 绘制每一个按键项（若总宽超出容器则安全截断不溢出）
        cursor_x = bar_rect.left() + pad_x
        center_y = bar_rect.center().y()
        max_x = bar_rect.right() - pad_x + 2.0
        for tokens, label, item_w in item_layouts:
            if cursor_x + item_w > max_x and cursor_x > bar_rect.left() + pad_x:
                break
            cursor_x = self._draw_button_tokens(painter, tokens, cursor_x, center_y, badge_fm)
            cursor_x += 6.0
            painter.setFont(self._font_prompt)
            painter.setPen(QPen(COLOR_TEXT_PRIMARY))
            lw = label_fm.horizontalAdvance(label)
            lrect = QRectF(cursor_x, bar_rect.top(), lw + 2.0, bar_h)
            painter.drawText(
                lrect,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                label,
            )
            cursor_x += lw + item_gap

    def _measure_tokens_width(self, tokens: Sequence[str], fm: QFontMetrics) -> float:
        """计算组合按键徽章组合的水平宽度。"""
        total = 0.0
        for idx, tok in enumerate(tokens):
            if idx > 0:
                total += 10.0  # '+' 间距
            upper = tok.upper()
            if upper in FACE_BUTTON_COLORS:
                total += 22.0
            else:
                total += max(26.0, fm.horizontalAdvance(tok) + 12.0)
        return total

    def _draw_button_tokens(
        self,
        painter: QPainter,
        tokens: Sequence[str],
        start_x: float,
        center_y: float,
        fm: QFontMetrics,
    ) -> float:
        """绘制 Xbox 彩色圆形按键徽章或银灰肩键/扳机胶囊，返回结束 X 坐标。"""
        x = start_x
        painter.setFont(self._font_badge)
        for idx, tok in enumerate(tokens):
            if idx > 0:
                plus_rect = QRectF(x, center_y - 9.0, 10.0, 18.0)
                painter.setPen(QPen(COLOR_TEXT_MUTED))
                painter.drawText(plus_rect, Qt.AlignmentFlag.AlignCenter, "+")
                x += 10.0

            upper = tok.upper()
            if upper in FACE_BUTTON_COLORS:
                bg_col, fg_col = FACE_BUTTON_COLORS[upper]
                d = 22.0
                circle_rect = QRectF(x, center_y - d / 2.0, d, d)
                painter.setPen(QPen(QColor(255, 255, 255, 90), 1.0))
                painter.setBrush(QBrush(bg_col))
                painter.drawEllipse(circle_rect)
                painter.setPen(QPen(fg_col))
                painter.drawText(circle_rect, Qt.AlignmentFlag.AlignCenter, upper)
                x += d
            else:
                pill_w = max(26.0, fm.horizontalAdvance(tok) + 12.0)
                pill_h = 20.0
                pill_rect = QRectF(x, center_y - pill_h / 2.0, pill_w, pill_h)
                painter.setPen(QPen(QColor("#9CA3AF"), 1.1))
                painter.setBrush(QBrush(QColor("#374151")))
                painter.drawRoundedRect(pill_rect, 6.0, 6.0)
                painter.setPen(QPen(QColor("#F8FAFC")))
                painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, tok)
                x += pill_w
        return x
