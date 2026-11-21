"""罗德岛 PRTS 工业战术风透明鼠标穿透 HUD 悬浮层。

基于 PySide6 构建无边框、置顶、完全不抢夺游戏焦点、不拦截鼠标输入的 OSD 叠加层，
以《明日方舟》罗德岛终端的 45° 斜切角几何美学实时渲染：战术终端状态栏、干员部署锁定框、
抓取源战术航线、菱形四向战术罗盘、开战充能仪及底部手柄战术指令条。
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
    QLinearGradient,
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

# 罗德岛 PRTS 工业战术调色板
COLOR_CYAN = QColor("#00C8FF")          # 罗德岛战术青
COLOR_AMBER = QColor("#FF9800")         # 源石琥珀橙
COLOR_GREEN = QColor("#3FB950")         # 作战绿（A 键 / 正常态）
COLOR_RED = QColor("#EF4444")           # 危机绯红
COLOR_BLUE = QColor("#2F81F7")
COLOR_YELLOW = QColor("#F5C518")
COLOR_TEXT_PRIMARY = QColor("#F4F6F9")  # 纯净战术白
COLOR_TEXT_MUTED = QColor("#8A99AD")    # 冷灰副标
COLOR_OBSIDIAN = QColor(11, 14, 20, 225)  # 黑曜石磨砂底板 #0B0E14
COLOR_CAPSULE_BG = COLOR_OBSIDIAN
COLOR_CAPSULE_BORDER = QColor(0, 200, 255, 120)

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

_FACE_BADGE_SIZE = 22.0
_KEYCAP_MIN_W = 28.0
_KEYCAP_H = 20.0
_TOKEN_PLUS_W = 12.0


def _alpha(color: QColor, alpha: int) -> QColor:
    """返回指定透明度的颜色副本。"""
    c = QColor(color)
    c.setAlpha(max(0, min(255, int(alpha))))
    return c


def chamfer_polygon(
    rect: QRectF,
    *,
    tl: float = 0.0,
    tr: float = 0.0,
    br: float = 0.0,
    bl: float = 0.0,
) -> QPolygonF:
    """构造带 45° 斜切角的矩形多边形（罗德岛工业切角底板通用几何）。"""
    x0, y0, x1, y1 = rect.left(), rect.top(), rect.right(), rect.bottom()
    lim = max(0.0, min(rect.width(), rect.height()) / 2.0)
    tl, tr, br, bl = (min(lim, max(0.0, v)) for v in (tl, tr, br, bl))
    pts: list[QPointF] = []
    pts.append(QPointF(x0 + tl, y0))
    pts.append(QPointF(x1 - tr, y0))
    if tr > 0.0:
        pts.append(QPointF(x1, y0 + tr))
    pts.append(QPointF(x1, y1 - br))
    if br > 0.0:
        pts.append(QPointF(x1 - br, y1))
    pts.append(QPointF(x0 + bl, y1))
    if bl > 0.0:
        pts.append(QPointF(x0, y1 - bl))
    pts.append(QPointF(x0, y0 + tl))
    return QPolygonF(pts)


def diamond_polygon(cx: float, cy: float, rx: float, ry: float | None = None) -> QPolygonF:
    """构造以 (cx, cy) 为中心的 45° 旋转菱形。"""
    ry = rx if ry is None else ry
    return QPolygonF(
        [
            QPointF(cx, cy - ry),
            QPointF(cx + rx, cy),
            QPointF(cx, cy + ry),
            QPointF(cx - rx, cy),
        ]
    )


def measure_button_tokens(tokens: Sequence[str], fm: QFontMetrics) -> float:
    """计算组合按键徽章序列的水平总宽度。"""
    total = 0.0
    for idx, tok in enumerate(tokens):
        if idx > 0:
            total += _TOKEN_PLUS_W
        upper = tok.upper()
        if upper in FACE_BUTTON_COLORS:
            total += _FACE_BADGE_SIZE
        else:
            total += max(_KEYCAP_MIN_W, fm.horizontalAdvance(tok) + 14.0)
    return total


def paint_button_tokens(
    painter: QPainter,
    tokens: Sequence[str],
    start_x: float,
    center_y: float,
    font: QFont,
    *,
    pressed: frozenset[str] | None = None,
) -> float:
    """绘制硬朗切角的 Xbox 面键徽章与斜角金属肩键/摇杆键帽，返回结束 X 坐标。

    该函数同时被透明 HUD 与 PRTS 桌面主控台复用，确保两端按键视觉完全一致。
    """
    fm = QFontMetrics(font)
    x = start_x
    painter.setFont(font)
    for idx, tok in enumerate(tokens):
        if idx > 0:
            plus_rect = QRectF(x, center_y - 9.0, _TOKEN_PLUS_W, 18.0)
            painter.setPen(QPen(COLOR_TEXT_MUTED))
            painter.drawText(plus_rect, Qt.AlignmentFlag.AlignCenter, "+")
            x += _TOKEN_PLUS_W

        upper = tok.upper()
        if upper in FACE_BUTTON_COLORS:
            bg_col, fg_col = FACE_BUTTON_COLORS[upper]
            d = _FACE_BADGE_SIZE
            rect = QRectF(x, center_y - d / 2.0, d, d)
            poly = chamfer_polygon(rect, tl=5.0, tr=5.0, br=5.0, bl=5.0)
            grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            grad.setColorAt(0.0, bg_col.lighter(118))
            grad.setColorAt(1.0, bg_col.darker(135))
            painter.setPen(QPen(QColor(255, 255, 255, 120), 1.0))
            painter.setBrush(QBrush(grad))
            painter.drawPolygon(poly)
            # 工业镶边：内侧细线
            inner = chamfer_polygon(rect.adjusted(2.5, 2.5, -2.5, -2.5), tl=3.5, tr=3.5, br=3.5, bl=3.5)
            painter.setPen(QPen(QColor(0, 0, 0, 70), 1.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolygon(inner)
            painter.setPen(QPen(fg_col))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, upper)
            x += d
        else:
            cap_w = max(_KEYCAP_MIN_W, fm.horizontalAdvance(tok) + 14.0)
            rect = QRectF(x, center_y - _KEYCAP_H / 2.0, cap_w, _KEYCAP_H)
            poly = chamfer_polygon(rect, tl=5.0, br=5.0)
            is_pressed = bool(pressed and upper in pressed)
            grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            if is_pressed:
                grad.setColorAt(0.0, QColor("#FFB74D"))
                grad.setColorAt(1.0, QColor("#E07B00"))
            else:
                grad.setColorAt(0.0, QColor("#2E3846"))
                grad.setColorAt(1.0, QColor("#141A23"))
            painter.setPen(QPen(QColor("#6B7C93") if not is_pressed else QColor("#FFE0B2"), 1.0))
            painter.setBrush(QBrush(grad))
            painter.drawPolygon(poly)
            # 顶部金属高光
            painter.setPen(QPen(QColor(255, 255, 255, 55), 1.0))
            painter.drawLine(
                QPointF(rect.left() + 6.0, rect.top() + 1.5),
                QPointF(rect.right() - 2.0, rect.top() + 1.5),
            )
            painter.setPen(QPen(QColor("#111827") if is_pressed else QColor("#F4F6F9")))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, tok)
            x += cap_w
    return x


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
    """罗德岛 PRTS 工业战术风透明鼠标穿透 HUD 悬浮层窗口。"""

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

        # 全部字体显式给定正整数点字号，杜绝 QFont::setPointSize(-1) 警告
        self._font_title = QFont("Microsoft YaHei UI", 10, QFont.Weight.Bold)
        self._font_badge = QFont("Segoe UI", 9, QFont.Weight.Bold)
        self._font_prompt = QFont("Microsoft YaHei UI", 9, QFont.Weight.Medium)
        self._font_small = QFont("Microsoft YaHei UI", 8, QFont.Weight.Bold)
        self._font_micro = QFont("Consolas", 7, QFont.Weight.Bold)

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
        """抗锯齿绘制罗德岛 PRTS 工业战术风 HUD 元素。"""
        snapshot = self._snapshot
        if snapshot is None or not snapshot.visible:
            return

        w = float(self.width())
        h = float(self.height())
        if w <= 1.0 or h <= 1.0:
            return

        painter = QPainter(self)
        try:
            self.render_snapshot(painter, w, h, snapshot)
        finally:
            painter.end()

    def render_snapshot(
        self,
        painter: QPainter,
        w: float,
        h: float,
        snapshot: HUDStateSnapshot,
    ) -> None:
        """在任意 QPainter 设备上绘制完整 HUD 图层（供 paintEvent 与离屏预览复用）。"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. 抓取源斜切虚线框 + 双轨战术航线
        if snapshot.holding_source_norm is not None:
            self._draw_holding_source_and_trajectory(painter, w, h, snapshot)

        # 2. 干员部署锁定框（支持标准矩形与 3D 透视棋盘梯形格）
        if (
            snapshot.focus_polygon_norm
            and len(snapshot.focus_polygon_norm) >= 3
            and not snapshot.free_cursor_active
        ):
            accent = COLOR_AMBER if snapshot.holding_source_norm is not None else COLOR_CYAN
            self._draw_focus_polygon(painter, w, h, snapshot.focus_polygon_norm, accent)

        # 3. 自由光标菱形准星
        if snapshot.free_cursor_active:
            self._draw_free_cursor_reticle(painter, w, h, snapshot.cursor_norm)

        # 4. 两段式部署：菱形四向战术罗盘
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

        # 5. 顶部战术终端状态栏 + Toast 提示
        self._draw_top_status_capsule(painter, w, h, snapshot)

        # 6. 开战长按充能仪
        if snapshot.ready_hold_progress > 0.0:
            self._draw_ready_hold_ring(painter, w, h, snapshot.ready_hold_progress)

        # 7. 底部战术指令条
        if snapshot.prompts:
            self._draw_bottom_prompt_bar(painter, w, h, snapshot.prompts)

    # ------------------------------------------------------------------
    # 部署锁定框
    # ------------------------------------------------------------------
    def _draw_focus_polygon(
        self,
        painter: QPainter,
        w: float,
        h: float,
        polygon_norm: Sequence[tuple[float, float]],
        color: QColor,
    ) -> None:
        """绘制明日方舟干员部署锁定框：扫描内衬 + 锐利主框 + 白色战术折角与菱形锚点。"""
        pts = [QPointF(u * w, v * h) for u, v in polygon_norm]
        poly = QPolygonF(pts)
        n = len(pts)
        cx = sum(p.x() for p in pts) / n
        cy = sum(p.y() for p in pts) / n

        # 半透明内衬 + 水平扫描线（裁剪至多边形内部）
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(_alpha(color, 30)))
        painter.drawPolygon(poly)

        clip_path = QPainterPath()
        clip_path.addPolygon(poly)
        clip_path.closeSubpath()
        bounds = poly.boundingRect()
        painter.save()
        painter.setClipPath(clip_path)
        painter.setPen(QPen(_alpha(color, 34), 1.0))
        y = bounds.top() + 2.0
        while y < bounds.bottom():
            painter.drawLine(QPointF(bounds.left(), y), QPointF(bounds.right(), y))
            y += 4.0
        painter.restore()

        # 外层柔光 + 锐利主框
        glow_pen = QPen(_alpha(color, 60), 5.0)
        glow_pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(glow_pen)
        painter.drawPolygon(poly)

        main_pen = QPen(_alpha(color, 240), 1.8)
        main_pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(main_pen)
        painter.drawPolygon(poly)

        # 边中点微刻度（沿外法线方向）
        tick_pen = QPen(_alpha(color, 230), 1.6)
        tick_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(tick_pen)
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            mx, my = (a.x() + b.x()) / 2.0, (a.y() + b.y()) / 2.0
            dx, dy = b.x() - a.x(), b.y() - a.y()
            seg = math.hypot(dx, dy)
            if seg < 18.0:
                continue
            nx, ny = -dy / seg, dx / seg
            if (mx + nx - cx) ** 2 + (my + ny - cy) ** 2 < (mx - cx) ** 2 + (my - cy) ** 2:
                nx, ny = -nx, -ny
            painter.drawLine(QPointF(mx + nx * 2.0, my + ny * 2.0), QPointF(mx + nx * 6.5, my + ny * 6.5))

        # 四角白色战术 L 折角（沿多边形边向量，精准适配透视倾斜格子）
        bracket_pen = QPen(QColor("#FFFFFF"), 2.6)
        bracket_pen.setCapStyle(Qt.PenCapStyle.SquareCap)
        bracket_pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        for i in range(n):
            p_curr = pts[i]
            arms: list[QPointF] = []
            for neighbor in (pts[(i - 1) % n], pts[(i + 1) % n]):
                dx = neighbor.x() - p_curr.x()
                dy = neighbor.y() - p_curr.y()
                seg_len = math.hypot(dx, dy)
                if seg_len < 1e-3:
                    continue
                arm = min(14.0, seg_len * 0.28)
                arms.append(QPointF(p_curr.x() + dx / seg_len * arm, p_curr.y() + dy / seg_len * arm))
            if len(arms) == 2:
                painter.setPen(bracket_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                path = QPainterPath(arms[0])
                path.lineTo(p_curr)
                path.lineTo(arms[1])
                painter.drawPath(path)

            # 角部外侧小菱形锚点
            ox, oy = p_curr.x() - cx, p_curr.y() - cy
            olen = math.hypot(ox, oy)
            if olen > 1e-3:
                ax = p_curr.x() + ox / olen * 7.0
                ay = p_curr.y() + oy / olen * 7.0
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(_alpha(color, 235)))
                painter.drawPolygon(diamond_polygon(ax, ay, 2.6))

    # ------------------------------------------------------------------
    # 抓取源与战术航线
    # ------------------------------------------------------------------
    def _draw_holding_source_and_trajectory(
        self,
        painter: QPainter,
        w: float,
        h: float,
        snapshot: HUDStateSnapshot,
    ) -> None:
        """绘制源石橙斜切虚线源槽位框、斜角战术铭牌及双轨流光航线与菱形箭头。"""
        assert snapshot.holding_source_norm is not None
        su, sv = snapshot.holding_source_norm
        sx, sy = su * w, sv * h
        now = time.monotonic()

        box_w = max(44.0, min(96.0, w * 0.068))
        box_h = max(44.0, min(96.0, h * 0.11))
        src_rect = QRectF(sx - box_w / 2.0, sy - box_h / 2.0, box_w, box_h)
        src_poly = chamfer_polygon(src_rect, tl=9.0, br=9.0)

        painter.setBrush(QBrush(_alpha(COLOR_AMBER, 34)))
        dash_pen = QPen(COLOR_AMBER, 2.0, Qt.PenStyle.CustomDashLine)
        dash_pen.setDashPattern([5.0, 3.0])
        dash_pen.setDashOffset((now * 24.0) % 8.0)
        dash_pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(dash_pen)
        painter.drawPolygon(src_poly)

        # 源中心菱形
        painter.setPen(QPen(QColor("#FFFFFF"), 1.2))
        painter.setBrush(QBrush(_alpha(COLOR_AMBER, 200)))
        painter.drawPolygon(diamond_polygon(sx, sy, 4.0))

        # 源槽位顶部斜角黑底橙字战术铭牌
        if snapshot.deploy_state == DeployState.HOLDING_STRATEGY:
            tag_text = "策略源"
        elif snapshot.deploy_state == DeployState.HOLDING_EQUIP:
            tag_text = "装备源"
        else:
            tag_text = "抓取源"
        fm = QFontMetrics(self._font_small)
        tw = fm.horizontalAdvance(tag_text) + 20.0
        th = fm.height() + 4.0
        tag_rect = QRectF(sx - tw / 2.0, src_rect.top() - th - 5.0, tw, th)
        painter.setPen(QPen(_alpha(COLOR_AMBER, 200), 1.0))
        painter.setBrush(QBrush(QColor(10, 9, 6, 232)))
        painter.drawPolygon(chamfer_polygon(tag_rect, tl=5.0, br=5.0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(COLOR_AMBER))
        painter.drawRect(QRectF(tag_rect.left() + 3.0, tag_rect.top() + 3.0, 2.5, th - 6.0))
        painter.setFont(self._font_small)
        painter.setPen(QPen(COLOR_AMBER))
        painter.drawText(tag_rect.adjusted(5.0, 0.0, 0.0, 0.0), Qt.AlignmentFlag.AlignCenter, tag_text)

        # 计算目标中心点（焦点多边形质心，否则使用光标坐标）
        if snapshot.focus_polygon_norm and not snapshot.free_cursor_active:
            tx = sum(u for u, _ in snapshot.focus_polygon_norm) / len(snapshot.focus_polygon_norm) * w
            ty = sum(v for _, v in snapshot.focus_polygon_norm) / len(snapshot.focus_polygon_norm) * h
        else:
            tx = snapshot.cursor_norm[0] * w
            ty = snapshot.cursor_norm[1] * h

        dist = math.hypot(tx - sx, ty - sy)
        if dist < 16.0:
            return

        ux, uy = (tx - sx) / dist, (ty - sy) / dist
        px, py = -uy, ux
        head_len = 14.0
        end_x, end_y = tx - ux * head_len * 0.6, ty - uy * head_len * 0.6

        # 柔光底轨
        glow_pen = QPen(_alpha(COLOR_AMBER, 60), 8.0)
        glow_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(glow_pen)
        painter.drawLine(QPointF(sx, sy), QPointF(end_x, end_y))

        # 双轨动态流光虚线
        offset = (now * 48.0) % 14.0
        for sign in (-1.0, 1.0):
            traj_pen = QPen(COLOR_AMBER, 1.6, Qt.PenStyle.CustomDashLine)
            traj_pen.setDashPattern([7.0, 5.0])
            traj_pen.setDashOffset(-offset)
            traj_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            painter.setPen(traj_pen)
            painter.drawLine(
                QPointF(sx + px * 2.6 * sign, sy + py * 2.6 * sign),
                QPointF(end_x + px * 2.6 * sign, end_y + py * 2.6 * sign),
            )

        # 锐利菱形箭头（风筝形，尾部内凹）
        tip = QPointF(tx, ty)
        back = QPointF(tx - ux * head_len, ty - uy * head_len)
        notch = QPointF(tx - ux * head_len * 0.68, ty - uy * head_len * 0.68)
        wing1 = QPointF(back.x() + px * 6.5, back.y() + py * 6.5)
        wing2 = QPointF(back.x() - px * 6.5, back.y() - py * 6.5)
        painter.setPen(QPen(QColor("#FFFFFF"), 1.2))
        painter.setBrush(QBrush(COLOR_AMBER))
        painter.drawPolygon(QPolygonF([tip, wing1, notch, wing2]))

    # ------------------------------------------------------------------
    # 自由光标准星
    # ------------------------------------------------------------------
    def _draw_free_cursor_reticle(
        self,
        painter: QPainter,
        w: float,
        h: float,
        cursor_norm: tuple[float, float],
    ) -> None:
        """自由光标模式下精致轻盈的空心细菱形 + 四向分离微刻度准星。"""
        cx, cy = cursor_norm[0] * w, cursor_norm[1] * h
        r = 10.0

        # 暗色描底保障在亮背景上的可读性
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(0, 0, 0, 120), 3.2))
        painter.drawPolygon(diamond_polygon(cx, cy, r))
        painter.setPen(QPen(_alpha(COLOR_CYAN, 235), 1.4))
        painter.drawPolygon(diamond_polygon(cx, cy, r))

        tick_pen = QPen(_alpha(COLOR_CYAN, 225), 1.6)
        tick_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        shadow_pen = QPen(QColor(0, 0, 0, 110), 3.4)
        shadow_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a = QPointF(cx + dx * (r + 4.0), cy + dy * (r + 4.0))
            b = QPointF(cx + dx * (r + 10.0), cy + dy * (r + 10.0))
            painter.setPen(shadow_pen)
            painter.drawLine(a, b)
            painter.setPen(tick_pen)
            painter.drawLine(a, b)

    # ------------------------------------------------------------------
    # 菱形四向战术罗盘
    # ------------------------------------------------------------------
    def _draw_direction_wheel(
        self,
        painter: QPainter,
        w: float,
        h: float,
        origin_norm: tuple[float, float],
        preview_dir: str | None,
    ) -> None:
        """绘制明日方舟标志性 45° 正交菱形战术罗盘（Diamond Tactical Rose）。"""
        cx = origin_norm[0] * w
        cy = origin_norm[1] * h
        r = max(56.0, min(98.0, min(w, h) * 0.09))

        # 外层菱形骨架底板
        outer = diamond_polygon(cx, cy, r)
        painter.setPen(QPen(_alpha(COLOR_CYAN, 70), 6.0))
        painter.setBrush(QBrush(QColor(11, 14, 20, 205)))
        painter.drawPolygon(outer)
        outer_pen = QPen(_alpha(COLOR_CYAN, 210), 1.6)
        outer_pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(outer_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(outer)

        # 内层菱形骨架线
        inner_pen = QPen(_alpha(COLOR_CYAN, 110), 1.0, Qt.PenStyle.CustomDashLine)
        inner_pen.setDashPattern([3.0, 3.0])
        painter.setPen(inner_pen)
        painter.drawPolygon(diamond_polygon(cx, cy, r * 0.62))

        # 外框四条斜边中点刻度
        painter.setPen(QPen(_alpha(COLOR_TEXT_PRIMARY, 170), 1.4))
        for sx_, sy_ in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            mx, my = cx + sx_ * r / 2.0, cy + sy_ * r / 2.0
            painter.drawLine(
                QPointF(mx - sx_ * 3.0, my - sy_ * 3.0),
                QPointF(mx + sx_ * 3.0, my + sy_ * 3.0),
            )

        directions: list[tuple[str, float, float, str]] = [
            ("up", 0.0, -1.0, "上"),
            ("down", 0.0, 1.0, "下"),
            ("left", -1.0, 0.0, "左"),
            ("right", 1.0, 0.0, "右"),
        ]
        painter.setFont(self._font_small)
        for dir_key, dx, dy, label in directions:
            active = preview_dir == dir_key
            px, py = -dy, dx
            tip = QPointF(cx + dx * r * 0.93, cy + dy * r * 0.93)
            wing_d = r * 0.50
            half_w = r * (0.21 if active else 0.17)
            wing1 = QPointF(cx + dx * wing_d + px * half_w, cy + dy * wing_d + py * half_w)
            wing2 = QPointF(cx + dx * wing_d - px * half_w, cy + dy * wing_d - py * half_w)
            back = QPointF(cx + dx * r * 0.30, cy + dy * r * 0.30)
            blade = QPolygonF([tip, wing1, back, wing2])

            if active:
                painter.setPen(QPen(_alpha(COLOR_AMBER, 110), 7.0))
                painter.setBrush(QBrush(COLOR_AMBER))
                painter.drawPolygon(blade)
                white_pen = QPen(QColor("#FFFFFF"), 1.8)
                white_pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
                painter.setPen(white_pen)
                painter.drawPolygon(blade)
            else:
                blade_pen = QPen(_alpha(COLOR_CYAN, 190), 1.2)
                blade_pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
                painter.setPen(blade_pen)
                painter.setBrush(QBrush(_alpha(COLOR_CYAN, 52)))
                painter.drawPolygon(blade)

            txt_c = QPointF(cx + dx * r * 0.56, cy + dy * r * 0.56)
            txt_rect = QRectF(txt_c.x() - 11.0, txt_c.y() - 9.0, 22.0, 18.0)
            painter.setPen(QPen(QColor("#111827") if active else COLOR_TEXT_PRIMARY))
            painter.drawText(txt_rect, Qt.AlignmentFlag.AlignCenter, label)

        # 中心十字核
        core_pen = QPen(_alpha(COLOR_TEXT_PRIMARY, 200), 1.2)
        painter.setPen(core_pen)
        painter.drawLine(QPointF(cx - r * 0.2, cy), QPointF(cx + r * 0.2, cy))
        painter.drawLine(QPointF(cx, cy - r * 0.2), QPointF(cx, cy + r * 0.2))
        painter.setPen(QPen(QColor("#FFFFFF"), 1.2))
        painter.setBrush(QBrush(COLOR_AMBER if preview_dir else COLOR_CYAN))
        painter.drawPolygon(diamond_polygon(cx, cy, 4.5))

        # 底部切角操作铭牌
        hint = "推摇杆 / 十字键定朝向   B 取消部署"
        fm = QFontMetrics(self._font_small)
        hw = fm.horizontalAdvance(hint) + 28.0
        hh = fm.height() + 8.0
        hint_rect = QRectF(cx - hw / 2.0, cy + r + 10.0, hw, hh)
        if hint_rect.bottom() > h - 4.0:
            hint_rect.moveBottom(cy - r - 10.0)
        self._draw_nameplate(painter, hint_rect, hint, COLOR_AMBER, COLOR_AMBER)

    def _draw_nameplate(
        self,
        painter: QPainter,
        rect: QRectF,
        text: str,
        accent: QColor,
        text_color: QColor,
    ) -> None:
        """绘制左上/右下斜切的黑晶操作铭牌。"""
        painter.setPen(QPen(_alpha(accent, 180), 1.0))
        painter.setBrush(QBrush(COLOR_OBSIDIAN))
        painter.drawPolygon(chamfer_polygon(rect, tl=6.0, br=6.0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(accent))
        painter.drawRect(QRectF(rect.left() + 4.0, rect.top() + 4.0, 2.5, rect.height() - 8.0))
        painter.setFont(self._font_small)
        painter.setPen(QPen(text_color))
        painter.drawText(rect.adjusted(6.0, 0.0, 0.0, 0.0), Qt.AlignmentFlag.AlignCenter, text)

    # ------------------------------------------------------------------
    # 顶部战术终端状态栏
    # ------------------------------------------------------------------
    @staticmethod
    def _format_top_status_text(snapshot: HUDStateSnapshot) -> str:
        """格式化顶部状态栏文案，消除自由鼠标与自由光标同义重复段。"""
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

    def _status_accent(self, snapshot: HUDStateSnapshot) -> QColor:
        """根据运行态返回顶栏状态色。"""
        if snapshot.paused:
            return COLOR_RED
        if snapshot.scene_locked_manual or snapshot.deploy_state != DeployState.IDLE:
            return COLOR_AMBER
        return COLOR_CYAN

    def _draw_top_status_capsule(
        self,
        painter: QPainter,
        w: float,
        h: float,
        snapshot: HUDStateSnapshot,
    ) -> None:
        """绘制顶部 45° 斜切战术终端状态栏及可选罗德岛 Toast 提示条。"""
        text = self._format_top_status_text(snapshot)
        accent = self._status_accent(snapshot)

        fm = QFontMetrics(self._font_title)
        micro_fm = QFontMetrics(self._font_micro)
        bar_h = max(32.0, fm.height() + 14.0)
        tab_h = float(micro_fm.height() + 3)
        lead_w = 34.0
        text_w = fm.horizontalAdvance(text)
        bar_w = min(w * 0.9, text_w + lead_w + 30.0)
        top_y = max(6.0, h * 0.010) + tab_h
        bar_rect = QRectF((w - bar_w) / 2.0, top_y, bar_w, bar_h)
        cut = 10.0

        # 顶部 PRTS // TERMINAL 微型标签页与装饰刻度
        tab_text = "PRTS // TERMINAL"
        tab_w = micro_fm.horizontalAdvance(tab_text) + 18.0
        tab_rect = QRectF(bar_rect.left() + cut, bar_rect.top() - tab_h, tab_w, tab_h)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(_alpha(accent, 210)))
        painter.drawPolygon(chamfer_polygon(tab_rect, tr=tab_h * 0.8))
        painter.setFont(self._font_micro)
        painter.setPen(QPen(QColor("#0B0E14")))
        painter.drawText(
            tab_rect.adjusted(6.0, 0.0, 0.0, 0.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            tab_text,
        )
        tick_pen = QPen(_alpha(accent, 150), 1.0)
        painter.setPen(tick_pen)
        tick_x = tab_rect.right() + 8.0
        tick_end = bar_rect.right() - 4.0
        idx = 0
        while tick_x < tick_end:
            tick_h = 5.0 if idx % 5 == 0 else 2.5
            painter.drawLine(
                QPointF(tick_x, bar_rect.top() - 1.5),
                QPointF(tick_x, bar_rect.top() - 1.5 - tick_h),
            )
            tick_x += 6.0
            idx += 1

        # 斜切黑曜石底板
        poly = chamfer_polygon(bar_rect, tl=cut, br=cut)
        border = _alpha(accent, 170) if accent != COLOR_CYAN else COLOR_CAPSULE_BORDER
        painter.setPen(QPen(border, 1.2))
        painter.setBrush(QBrush(COLOR_OBSIDIAN))
        painter.drawPolygon(poly)

        # 左侧粗竖向状态色条（顺斜切角裁形）
        strip = QPolygonF(
            [
                QPointF(bar_rect.left() + cut, bar_rect.top()),
                QPointF(bar_rect.left() + cut + 5.0, bar_rect.top()),
                QPointF(bar_rect.left() + 5.0, bar_rect.top() + cut + 5.0),
                QPointF(bar_rect.left() + 5.0, bar_rect.bottom()),
                QPointF(bar_rect.left(), bar_rect.bottom()),
                QPointF(bar_rect.left(), bar_rect.top() + cut),
            ]
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(accent))
        painter.drawPolygon(strip)

        # 旋转 45° 菱形信标
        beacon_c = QPointF(bar_rect.left() + 20.0, bar_rect.center().y())
        painter.setPen(QPen(_alpha(accent, 90), 4.0))
        painter.setBrush(QBrush(accent))
        painter.drawPolygon(diamond_polygon(beacon_c.x(), beacon_c.y(), 4.8))

        # 右下角斜纹装饰
        painter.setPen(QPen(_alpha(accent, 120), 1.2))
        for k in range(3):
            ox = bar_rect.right() - cut - 6.0 - k * 5.0
            painter.drawLine(
                QPointF(ox, bar_rect.bottom() - 2.0),
                QPointF(ox + 5.0, bar_rect.bottom() - 7.0),
            )

        # 主状态文案
        text_rect = QRectF(
            bar_rect.left() + lead_w,
            bar_rect.top(),
            bar_rect.width() - lead_w - 16.0,
            bar_rect.height(),
        )
        painter.setFont(self._font_title)
        painter.setPen(QPen(COLOR_TEXT_PRIMARY))
        elided = fm.elidedText(text, Qt.TextElideMode.ElideRight, int(text_rect.width()))
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            elided,
        )

        # Toast 次级消息：黑底 + 左侧亮立柱 + 右侧斜切纹样
        if snapshot.toast_message:
            tfm = QFontMetrics(self._font_prompt)
            tw = min(w * 0.8, tfm.horizontalAdvance(snapshot.toast_message) + 64.0)
            th = tfm.height() + 12.0
            toast_rect = QRectF((w - tw) / 2.0, bar_rect.bottom() + 6.0, tw, th)
            toast_accent = COLOR_AMBER if (snapshot.paused or snapshot.scene_locked_manual) else COLOR_CYAN
            painter.setPen(QPen(_alpha(toast_accent, 110), 1.0))
            painter.setBrush(QBrush(QColor(6, 8, 12, 236)))
            painter.drawPolygon(chamfer_polygon(toast_rect, br=th * 0.5))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(toast_accent))
            painter.drawRect(QRectF(toast_rect.left(), toast_rect.top(), 4.0, th))
            painter.setBrush(QBrush(_alpha(toast_accent, 90)))
            for k in range(3):
                ox = toast_rect.right() - th * 0.5 - 10.0 - k * 7.0
                painter.drawPolygon(
                    QPolygonF(
                        [
                            QPointF(ox, toast_rect.bottom() - 3.0),
                            QPointF(ox + 3.0, toast_rect.bottom() - 3.0),
                            QPointF(ox + 9.0, toast_rect.top() + 3.0),
                            QPointF(ox + 6.0, toast_rect.top() + 3.0),
                        ]
                    )
                )
            painter.setFont(self._font_prompt)
            painter.setPen(QPen(COLOR_TEXT_PRIMARY))
            msg_rect = toast_rect.adjusted(14.0, 0.0, -th * 0.5 - 30.0, 0.0)
            painter.drawText(
                msg_rect,
                Qt.AlignmentFlag.AlignCenter,
                tfm.elidedText(snapshot.toast_message, Qt.TextElideMode.ElideRight, int(msg_rect.width())),
            )

    # ------------------------------------------------------------------
    # 开战长按充能仪
    # ------------------------------------------------------------------
    def _draw_ready_hold_ring(
        self,
        painter: QPainter,
        w: float,
        h: float,
        progress: float,
    ) -> None:
        """绘制双环刻度盘 + 源石橙充能圆弧 + 中心百分比读数与切角铭牌。"""
        clamped = max(0.0, min(1.0, float(progress)))
        radius = max(34.0, min(48.0, min(w, h) * 0.058))
        cx = w / 2.0
        cy = max(108.0, h * 0.18)

        # 黑曜石衬底
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(11, 14, 20, 220)))
        painter.drawEllipse(QPointF(cx, cy), radius + 6.0, radius + 6.0)

        # 外环刻度
        for i in range(48):
            ang = math.radians(i * 7.5 - 90.0)
            major = i % 6 == 0
            lit = (i / 48.0) < clamped
            r0 = radius + (0.0 if major else 2.0)
            r1 = radius + 5.0
            col = COLOR_AMBER if lit else _alpha(COLOR_TEXT_MUTED, 150)
            painter.setPen(QPen(col, 1.6 if major else 1.0))
            painter.drawLine(
                QPointF(cx + math.cos(ang) * r0, cy + math.sin(ang) * r0),
                QPointF(cx + math.cos(ang) * r1, cy + math.sin(ang) * r1),
            )

        # 内环细线
        painter.setPen(QPen(_alpha(COLOR_CYAN, 120), 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), radius * 0.70, radius * 0.70)

        # 充能圆弧轨道 + 进度（从 12 点方向顺时针增长）
        arc_r = radius - 5.0
        arc_rect = QRectF(cx - arc_r, cy - arc_r, arc_r * 2.0, arc_r * 2.0)
        painter.setPen(QPen(_alpha(COLOR_AMBER, 45), 5.0))
        painter.drawEllipse(arc_rect)
        arc_pen = QPen(COLOR_AMBER, 5.0)
        arc_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(arc_pen)
        span_angle = int(round(-clamped * 360.0 * 16.0))
        painter.drawArc(arc_rect, 90 * 16, span_angle)

        # 百分比读数
        painter.setFont(self._font_title)
        painter.setPen(QPen(COLOR_TEXT_PRIMARY))
        painter.drawText(arc_rect, Qt.AlignmentFlag.AlignCenter, f"{int(clamped * 100)}%")

        # 切角铭牌
        label = "长按 START 准备开战"
        fm = QFontMetrics(self._font_small)
        lw = fm.horizontalAdvance(label) + 28.0
        lh = fm.height() + 8.0
        lrect = QRectF(cx - lw / 2.0, cy + radius + 12.0, lw, lh)
        self._draw_nameplate(painter, lrect, label, COLOR_AMBER, COLOR_TEXT_PRIMARY)

    # ------------------------------------------------------------------
    # 底部战术指令条
    # ------------------------------------------------------------------
    def _draw_bottom_prompt_bar(
        self,
        painter: QPainter,
        w: float,
        h: float,
        prompts: Sequence[PromptItem],
    ) -> None:
        """绘制切角黑晶战术底栏、顶部青色高光分割线与工业风手柄按键徽章。"""
        badge_fm = QFontMetrics(self._font_badge)
        label_fm = QFontMetrics(self._font_prompt)

        item_layouts: list[tuple[list[str], str, float]] = []
        item_gap = 22.0
        for item in prompts:
            tokens = [tok.strip() for tok in item.button.split("+") if tok.strip()]
            if not tokens:
                tokens = [item.button]
            tokens_w = self._measure_tokens_width(tokens, badge_fm)
            label_w = label_fm.horizontalAdvance(item.label)
            total_item_w = tokens_w + 7.0 + label_w
            item_layouts.append((tokens, item.label, total_item_w))

        pad_x = 22.0
        bar_h = 36.0
        cut = 12.0
        content_w = sum(x[2] for x in item_layouts) + item_gap * max(0, len(item_layouts) - 1)
        bar_w = min(w * 0.96, content_w + pad_x * 2.0)
        bottom_margin = max(8.0, h * 0.015)
        bar_rect = QRectF((w - bar_w) / 2.0, h - bar_h - bottom_margin, bar_w, bar_h)

        poly = chamfer_polygon(bar_rect, tl=cut, br=cut)
        painter.setPen(QPen(QColor(255, 255, 255, 36), 1.0))
        painter.setBrush(QBrush(COLOR_OBSIDIAN))
        painter.drawPolygon(poly)

        # 顶部 1px 青色高光分割线 + 左右端微型标记
        painter.setPen(QPen(_alpha(COLOR_CYAN, 200), 1.0))
        painter.drawLine(
            QPointF(bar_rect.left() + cut, bar_rect.top() + 0.5),
            QPointF(bar_rect.right() - 1.0, bar_rect.top() + 0.5),
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(COLOR_CYAN))
        painter.drawRect(QRectF(bar_rect.right() - 26.0, bar_rect.top(), 24.0, 2.5))
        painter.setBrush(QBrush(_alpha(COLOR_CYAN, 160)))
        painter.drawRect(QRectF(bar_rect.left() + 4.0, bar_rect.top() + cut, 2.0, bar_h - cut - 6.0))

        # 绘制每一个按键项（若总宽超出容器则安全截断不溢出）
        cursor_x = bar_rect.left() + pad_x
        center_y = bar_rect.center().y() + 0.5
        max_x = bar_rect.right() - pad_x + 2.0
        for idx, (tokens, label, item_w) in enumerate(item_layouts):
            if cursor_x + item_w > max_x and cursor_x > bar_rect.left() + pad_x:
                break
            if idx > 0:
                # 指令项间菱形分隔点
                sep_x = cursor_x - item_gap / 2.0
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(_alpha(COLOR_TEXT_MUTED, 150)))
                painter.drawPolygon(diamond_polygon(sep_x, center_y, 2.2))
            cursor_x = self._draw_button_tokens(painter, tokens, cursor_x, center_y, badge_fm)
            cursor_x += 7.0
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
        return measure_button_tokens(tokens, fm)

    def _draw_button_tokens(
        self,
        painter: QPainter,
        tokens: Sequence[str],
        start_x: float,
        center_y: float,
        fm: QFontMetrics,
    ) -> float:
        """绘制切角面键徽章或斜角金属肩键/摇杆键帽，返回结束 X 坐标。"""
        del fm
        return paint_button_tokens(painter, tokens, start_x, center_y, self._font_badge)
