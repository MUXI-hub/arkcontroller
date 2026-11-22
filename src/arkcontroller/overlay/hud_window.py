"""罗德岛 PRTS 纯白极简风透明鼠标穿透 HUD 悬浮层。

基于 PySide6 构建无边框、置顶、完全不抢夺游戏焦点、不拦截鼠标输入的 OSD 叠加层。
视觉采用「以白为主、克制细线、琥珀微点缀」的 H1 极简细线基底：
白底细边顶部状态签、四角细折标选择框、白色细虚线抓取源与连线、分离式细线准星、
半透白细带底部提示条；定朝向组件与左右摇杆图标采用 H2 风格（十字细轴 + 折线箭头 / 双层细圆环）。
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import math
import re
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
WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011

# 纯白极简调色板（以白为主，琥珀微点缀）
COLOR_WHITE = QColor("#FFFFFF")
COLOR_AMBER = QColor("#F59E0B")         # 源石琥珀（唯一强调色）
COLOR_CYAN = QColor("#0098DC")          # 罗德岛天青（校准网格等辅助色）
COLOR_GREEN = QColor("#16A34A")
COLOR_RED = QColor("#DC2626")
COLOR_INK = QColor("#1A1A1A")           # 主文字深炭黑
COLOR_CHARCOAL = QColor("#2B2B2B")      # 键帽线框
COLOR_EDGE = QColor("#BDBDBD")          # 细灰描边
COLOR_TEXT_PRIMARY = COLOR_INK
COLOR_TEXT_MUTED = QColor("#7A7A7A")
COLOR_CAPSULE_BG = QColor(255, 255, 255, 245)
COLOR_CAPSULE_BORDER = COLOR_EDGE
COLOR_HALO = QColor(0, 0, 0, 72)        # 线条外衬暗影（雪地等亮背景防融色）

SCENE_DISPLAY_NAMES: dict[GameScene, str] = {
    GameScene.BATTLE_PREP: "休整运营",
    GameScene.MODAL_CHOICE: "决策弹窗",
    GameScene.MODAL_CONFIRM: "回合确认",
    GameScene.BATTLE_COMBAT: "作战中",
    GameScene.UNKNOWN_FREE_CURSOR: "自由鼠标",
}

_FREE_CURSOR_SYNONYM_LABELS: frozenset[str] = frozenset({"自由鼠标", "自由光标"})
_PENDING_KEYWORDS: tuple[str, ...] = ("再按", "待确认")

# 面键在手柄上的物理方位（用于在细圆环上点一粒琥珀微点）
FACE_BUTTON_POSITIONS: dict[str, tuple[int, int]] = {
    "A": (0, 1),
    "B": (1, 0),
    "X": (-1, 0),
    "Y": (0, -1),
}
# 兼容旧接口：面键 -> (底色, 字色)
FACE_BUTTON_COLORS: dict[str, tuple[QColor, QColor]] = {
    k: (COLOR_WHITE, COLOR_INK) for k in FACE_BUTTON_POSITIONS
}

_ICON_R = 10.0
_CAP_H = 18.0
_SEP_W = 11.0

_DIR_ALIASES: dict[str, str] = {
    "上": "up", "下": "down", "左": "left", "右": "right",
    "↑": "up", "↓": "down", "←": "left", "→": "right",
    "DPAD_UP": "up", "DPAD_DOWN": "down", "DPAD_LEFT": "left", "DPAD_RIGHT": "right",
    "UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right",
}
_DPAD_ALIASES = frozenset({"D-PAD", "DPAD", "十字键", "方向键"})
_STICK_ALIASES: dict[str, str] = {"LS": "LS", "RS": "RS", "摇杆": "LS", "左摇杆": "LS", "右摇杆": "RS"}
_CAP_KEYS = frozenset({"LB", "RB", "LT", "RT"})
_PREFIX_RE = re.compile(r"^(长按|按住|按下|松开|双击|连按|再按)\s*(.+)$")


@dataclass(frozen=True)
class ButtonGlyph:
    """按钮图示解析单元：kind ∈ face/cap/start/back/stick/dpad/text/sep/prefix。"""

    kind: str
    value: str = ""


def _alpha(color: QColor, alpha: int) -> QColor:
    c = QColor(color)
    c.setAlpha(max(0, min(255, int(alpha))))
    return c


def _pen(color: QColor, width: float, style: Qt.PenStyle = Qt.PenStyle.SolidLine,
         cap: Qt.PenCapStyle = Qt.PenCapStyle.SquareCap) -> QPen:
    pn = QPen(color, width, style)
    pn.setCapStyle(cap)
    pn.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    return pn


def chamfer_polygon(
    rect: QRectF,
    *,
    tl: float = 0.0,
    tr: float = 0.0,
    br: float = 0.0,
    bl: float = 0.0,
) -> QPolygonF:
    """构造带 45° 斜切角的矩形多边形。"""
    x0, y0, x1, y1 = rect.left(), rect.top(), rect.right(), rect.bottom()
    lim = max(0.0, min(rect.width(), rect.height()) / 2.0)
    tl, tr, br, bl = (min(lim, max(0.0, v)) for v in (tl, tr, br, bl))
    pts: list[QPointF] = [QPointF(x0 + tl, y0), QPointF(x1 - tr, y0)]
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
    return QPolygonF([QPointF(cx, cy - ry), QPointF(cx + rx, cy), QPointF(cx, cy + ry), QPointF(cx - rx, cy)])


# ----------------------------------------------------------------------
# 手柄按钮图示解析与绘制（HUD 底栏与主控台键位手册共用）
# ----------------------------------------------------------------------
def _parse_atom(atom: str) -> list[ButtonGlyph]:
    raw = atom.strip()
    if not raw:
        return []
    m = _PREFIX_RE.match(raw)
    if m and m.group(2).strip():
        rest = _parse_atom(m.group(2))
        if rest and all(g.kind != "text" for g in rest):
            return [ButtonGlyph("prefix", m.group(1)), *rest]
    up = raw.upper()
    if up in FACE_BUTTON_POSITIONS:
        return [ButtonGlyph("face", up)]
    if up in _CAP_KEYS:
        return [ButtonGlyph("cap", up)]
    if up == "START":
        return [ButtonGlyph("start")]
    if up in ("BACK", "VIEW"):
        return [ButtonGlyph("back")]
    if raw in _STICK_ALIASES or up in _STICK_ALIASES:
        return [ButtonGlyph("stick", _STICK_ALIASES.get(raw, _STICK_ALIASES.get(up, "LS")))]
    if up in _DPAD_ALIASES or raw in _DPAD_ALIASES:
        return [ButtonGlyph("dpad", "")]
    if raw in _DIR_ALIASES or up in _DIR_ALIASES:
        return [ButtonGlyph("dpad", _DIR_ALIASES.get(raw, _DIR_ALIASES.get(up, "")))]
    return [ButtonGlyph("text", raw)]


def parse_button_spec(spec: str) -> list[ButtonGlyph]:
    """解析调度器 PromptItem.button 字符串为按钮图示序列。

    语法：'/' 表示「或」，'+' 表示同时按下，'>' 表示依次连按；
    支持「长按Start」「按住 LT」「松开 LT」「双击A」等中文前缀，
    以及「LS/D-Pad」「摇杆/十字键」「左/上/下」等组合写法。
    """
    glyphs: list[ButtonGlyph] = []
    seq_parts = [s for s in str(spec).split(">")]
    for si, seq in enumerate(seq_parts):
        if si > 0:
            glyphs.append(ButtonGlyph("sep", "›"))
        alts = seq.split("/")
        for ai, alt in enumerate(alts):
            if ai > 0:
                glyphs.append(ButtonGlyph("sep", "/"))
            chords = alt.split("+")
            for ci, chord in enumerate(chords):
                if ci > 0:
                    glyphs.append(ButtonGlyph("sep", "+"))
                glyphs.extend(_parse_atom(chord))
    # 清理首尾与连续分隔符
    cleaned: list[ButtonGlyph] = []
    for g in glyphs:
        if g.kind == "sep" and (not cleaned or cleaned[-1].kind == "sep"):
            continue
        cleaned.append(g)
    while cleaned and cleaned[-1].kind == "sep":
        cleaned.pop()
    return cleaned or [ButtonGlyph("text", str(spec))]


def _glyph_width(g: ButtonGlyph, font: QFont) -> float:
    fm = QFontMetrics(font)
    if g.kind in ("face", "start", "back", "stick"):
        return _ICON_R * 2.0 + (0.5 if g.kind == "stick" else 0.0)
    if g.kind == "dpad":
        return 21.0
    if g.kind == "cap":
        return max(26.0, fm.horizontalAdvance(g.value) + 12.0)
    if g.kind == "sep":
        return _SEP_W
    if g.kind == "prefix":
        return QFontMetrics(_prefix_font(font)).horizontalAdvance(g.value) + 4.0
    return max(24.0, fm.horizontalAdvance(g.value) + 12.0)


def _prefix_font(font: QFont) -> QFont:
    return QFont("Microsoft YaHei UI", max(1, font.pointSize()))


def measure_button_spec(spec: str | Sequence[ButtonGlyph], font: QFont) -> float:
    """测量按钮图示序列总宽度。"""
    glyphs = parse_button_spec(spec) if isinstance(spec, str) else list(spec)
    total = 0.0
    for i, g in enumerate(glyphs):
        if i > 0:
            total += 2.0
        total += _glyph_width(g, font)
    return total


def _halo_circle(painter: QPainter, c: QPointF, r: float) -> None:
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(0, 0, 0, 42)))
    painter.drawEllipse(c + QPointF(0.0, 1.0), r + 0.6, r + 0.6)


def paint_button_glyph(
    painter: QPainter,
    g: ButtonGlyph,
    x: float,
    cy: float,
    font: QFont,
    *,
    highlight: bool = False,
) -> float:
    """绘制单个按钮图示，返回结束 X 坐标。highlight=True 时线框改为琥珀色（实时按下反馈）。"""
    line = COLOR_AMBER if highlight else COLOR_CHARCOAL
    w = _glyph_width(g, font)
    r = _ICON_R
    if g.kind == "face":
        c = QPointF(x + r, cy)
        _halo_circle(painter, c, r)
        painter.setPen(_pen(line, 1.8 if highlight else 1.5))
        painter.setBrush(QBrush(COLOR_WHITE))
        painter.drawEllipse(c, r, r)
        bold = QFont(font)
        bold.setBold(True)
        painter.setFont(bold)
        painter.setPen(QPen(COLOR_INK))
        painter.drawText(QRectF(c.x() - r, cy - r, 2 * r, 2 * r), Qt.AlignmentFlag.AlignCenter, g.value)
        dx, dy = FACE_BUTTON_POSITIONS.get(g.value, (0, 1))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(COLOR_AMBER))
        painter.drawEllipse(QPointF(c.x() + dx * r, cy + dy * r), 2.0, 2.0)
    elif g.kind == "cap":
        rect = QRectF(x, cy - _CAP_H / 2.0, w, _CAP_H)
        path = QPainterPath()
        path.addRoundedRect(rect, 3.0, 3.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 38)))
        painter.drawPath(path.translated(0.0, 1.0))
        painter.setPen(_pen(line, 1.6 if highlight else 1.3))
        painter.setBrush(QBrush(COLOR_WHITE))
        painter.drawPath(path)
        painter.setFont(font)
        painter.setPen(QPen(COLOR_INK))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, g.value)
    elif g.kind == "start":
        c = QPointF(x + r, cy)
        _halo_circle(painter, c, r)
        painter.setPen(_pen(line, 1.5))
        painter.setBrush(QBrush(COLOR_WHITE))
        painter.drawEllipse(c, r, r)
        bar = _pen(COLOR_INK, 1.4, cap=Qt.PenCapStyle.FlatCap)
        painter.setPen(bar)
        half = r * 0.46
        for k in (-1, 0, 1):
            yy = cy + k * r * 0.34
            painter.drawLine(QPointF(c.x() - half, yy), QPointF(c.x() + half, yy))
    elif g.kind == "back":
        c = QPointF(x + r, cy)
        _halo_circle(painter, c, r)
        painter.setPen(_pen(line, 1.5))
        painter.setBrush(QBrush(COLOR_WHITE))
        painter.drawEllipse(c, r, r)
        painter.setPen(_pen(COLOR_INK, 1.1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        s = r * 0.5
        painter.drawRect(QRectF(c.x() - s * 0.95, cy - s * 0.75, s * 1.15, s * 0.95))
        painter.setBrush(QBrush(COLOR_WHITE))
        painter.drawRect(QRectF(c.x() - s * 0.25, cy - s * 0.15, s * 1.15, s * 0.95))
    elif g.kind == "stick":
        # H2：外圈深炭灰实线 + 内圈极细浅灰环 + 居中 LS/RS 字标
        c = QPointF(x + r + 0.25, cy)
        _halo_circle(painter, c, r)
        painter.setPen(_pen(line, 1.5))
        painter.setBrush(QBrush(COLOR_WHITE))
        painter.drawEllipse(c, r, r)
        painter.setPen(_pen(COLOR_EDGE, 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(c, r * 0.68, r * 0.68)
        tiny = QFont("Segoe UI", max(1, font.pointSize() - 3), QFont.Weight.Bold)
        painter.setFont(tiny)
        painter.setPen(QPen(COLOR_INK))
        painter.drawText(QRectF(c.x() - r, cy - r, 2 * r, 2 * r), Qt.AlignmentFlag.AlignCenter, g.value or "LS")
    elif g.kind == "dpad":
        c = QPointF(x + w / 2.0, cy)
        s, a = 6.8, 3.5
        cross = QPolygonF([
            QPointF(c.x() - a, cy - s - a), QPointF(c.x() + a, cy - s - a), QPointF(c.x() + a, cy - a),
            QPointF(c.x() + s + a, cy - a), QPointF(c.x() + s + a, cy + a), QPointF(c.x() + a, cy + a),
            QPointF(c.x() + a, cy + s + a), QPointF(c.x() - a, cy + s + a), QPointF(c.x() - a, cy + a),
            QPointF(c.x() - s - a, cy + a), QPointF(c.x() - s - a, cy - a), QPointF(c.x() - a, cy - a),
        ])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 38)))
        painter.drawPolygon(cross.translated(0.0, 1.0))
        painter.setPen(_pen(line, 1.2))
        painter.setBrush(QBrush(COLOR_WHITE))
        painter.drawPolygon(cross)
        if g.value:
            d = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[g.value]
            painter.setPen(_pen(COLOR_AMBER, 2.0, cap=Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPointF(c.x() + d[0] * 2.4, cy + d[1] * 2.4),
                             QPointF(c.x() + d[0] * (s + 1.0), cy + d[1] * (s + 1.0)))
    elif g.kind == "sep":
        painter.setFont(font)
        painter.setPen(QPen(COLOR_TEXT_MUTED))
        painter.drawText(QRectF(x, cy - 9.0, w, 18.0), Qt.AlignmentFlag.AlignCenter, g.value)
    elif g.kind == "prefix":
        painter.setFont(_prefix_font(font))
        painter.setPen(QPen(QColor("#444444")))
        painter.drawText(QRectF(x, cy - 9.0, w, 18.0),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, g.value)
    else:  # text 兜底键帽
        rect = QRectF(x, cy - _CAP_H / 2.0, w, _CAP_H)
        path = QPainterPath()
        path.addRoundedRect(rect, 3.0, 3.0)
        painter.setPen(_pen(line, 1.1))
        painter.setBrush(QBrush(COLOR_WHITE))
        painter.drawPath(path)
        f = QFont("Microsoft YaHei UI", max(1, font.pointSize() - 1))
        painter.setFont(f)
        painter.setPen(QPen(COLOR_INK))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, g.value)
    return x + w


def paint_button_spec(
    painter: QPainter,
    spec: str | Sequence[ButtonGlyph],
    x: float,
    cy: float,
    font: QFont,
    *,
    highlight: bool = False,
) -> float:
    """绘制完整按钮图示序列（如「长按Start」「LS/D-Pad」「RT+A」），返回结束 X 坐标。"""
    glyphs = parse_button_spec(spec) if isinstance(spec, str) else list(spec)
    for i, g in enumerate(glyphs):
        if i > 0:
            x += 2.0
        x = paint_button_glyph(painter, g, x, cy, font, highlight=highlight)
    return x


def measure_button_tokens(tokens: Sequence[str], fm: QFontMetrics) -> float:
    """兼容旧接口：以 '+' 连接的组合按键宽度。"""
    font = QFont("Segoe UI", 8, QFont.Weight.DemiBold)
    del fm
    return measure_button_spec("+".join(tokens), font)


def paint_button_tokens(
    painter: QPainter,
    tokens: Sequence[str],
    start_x: float,
    center_y: float,
    font: QFont,
    *,
    pressed: frozenset[str] | None = None,
) -> float:
    """兼容旧接口：绘制以 '+' 连接的组合按键。"""
    hl = bool(pressed and any(t.upper() in pressed for t in tokens))
    return paint_button_spec(painter, "+".join(tokens), start_x, center_y, font, highlight=hl)


# ----------------------------------------------------------------------
# Win32 穿透与对齐
# ----------------------------------------------------------------------
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


def apply_win32_display_affinity(hwnd: int, exclude_from_capture: bool = True) -> bool:
    """设置窗口截屏/屏幕共享可见性：True 为 WDA_EXCLUDEFROMCAPTURE（隐藏），False 为 WDA_NONE（可见）。"""
    if not hwnd or not _is_windows_qpa():
        return False
    try:
        user32 = ctypes.windll.user32
        user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
        user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
        affinity = WDA_EXCLUDEFROMCAPTURE if exclude_from_capture else WDA_NONE
        return bool(user32.SetWindowDisplayAffinity(wintypes.HWND(hwnd), affinity))
    except Exception:
        # 兼容不支持 WDA_EXCLUDEFROMCAPTURE 的旧版系统或受限测试环境
        return False


def apply_win32_click_through(hwnd: int, *, exclude_from_capture: bool = True) -> bool:
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
        apply_win32_display_affinity(hwnd, exclude_from_capture)
        return True
    except Exception:
        return False


def _corner_segments(pts: Sequence[QPointF], arm: float, ratio: float) -> list[tuple[QPointF, QPointF, QPointF]]:
    """计算多边形每个顶点的两条折标端点（沿边向量，适配透视梯形格）。"""
    n = len(pts)
    segs: list[tuple[QPointF, QPointF, QPointF]] = []
    for i in range(n):
        c = pts[i]
        ends: list[QPointF] = []
        for nb in (pts[i - 1], pts[(i + 1) % n]):
            dx, dy = nb.x() - c.x(), nb.y() - c.y()
            seg = math.hypot(dx, dy)
            if seg < 1e-3:
                ends.append(c)
                continue
            a = min(arm, seg * ratio)
            ends.append(QPointF(c.x() + dx / seg * a, c.y() + dy / seg * a))
        segs.append((ends[0], c, ends[1]))
    return segs


class HUDWindow(QWidget):
    """罗德岛 PRTS 纯白极简风透明鼠标穿透 HUD 悬浮层窗口。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._overlay_enabled: bool = True
        self._viewport: ViewportRect | None = None
        self._snapshot: HUDStateSnapshot | None = None
        self._last_geom_rect: tuple[int, int, int, int] | None = None
        self._last_repaint_ts: float = 0.0
        self._win32_styled: bool = False
        self._hide_from_capture: bool = True

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
        self._font_title = QFont("Microsoft YaHei UI", 9, QFont.Weight.Bold)
        self._font_badge = QFont("Segoe UI", 8, QFont.Weight.DemiBold)
        self._font_prompt = QFont("Microsoft YaHei UI", 9)
        self._font_small = QFont("Microsoft YaHei UI", 8, QFont.Weight.Bold)
        self._font_micro = QFont("Segoe UI", 7, QFont.Weight.Bold)

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

    @property
    def hide_from_capture(self) -> bool:
        """是否在截屏 / 录屏 / 屏幕共享中隐藏 HUD（默认 True，防止 mss 采样自污染）。"""
        return self._hide_from_capture

    def set_hide_from_capture(self, enabled: bool) -> None:
        """热切换截屏可见性：已创建窗口句柄时立即在 WDA_EXCLUDEFROMCAPTURE 与 WDA_NONE 间切换。"""
        self._hide_from_capture = bool(enabled)
        if not _is_windows_qpa() or not self.testAttribute(Qt.WidgetAttribute.WA_WState_Created):
            return
        try:
            hwnd = int(self.winId())
        except Exception:
            hwnd = 0
        if hwnd:
            apply_win32_display_affinity(hwnd, self._hide_from_capture)

    def showEvent(self, event: QShowEvent) -> None:
        """窗口显示时通过 Win32 SetWindowLongW 双重加固鼠标穿透与不抢焦点属性。"""
        super().showEvent(event)
        try:
            hwnd = int(self.winId())
        except Exception:
            hwnd = 0
        if hwnd:
            self._win32_styled = apply_win32_click_through(
                hwnd, exclude_from_capture=self._hide_from_capture
            )
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
        """抗锯齿绘制纯白极简 HUD 元素。"""
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

    @staticmethod
    def _is_pending(snapshot: HUDStateSnapshot) -> bool:
        """当前焦点是否处于待二次确认 / 抓取目标态。"""
        label = snapshot.focus_label or ""
        return snapshot.holding_source_norm is not None or any(k in label for k in _PENDING_KEYWORDS)

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

        # 1. 抓取源细虚线框 + 连线
        if snapshot.holding_source_norm is not None:
            self._draw_holding_source_and_trajectory(painter, w, h, snapshot)

        # 2. 四角细折标选择框（支持标准矩形与 3D 透视棋盘梯形格）
        if (
            snapshot.focus_polygon_norm
            and len(snapshot.focus_polygon_norm) >= 3
            and not snapshot.free_cursor_active
        ):
            pending = self._is_pending(snapshot)
            self._draw_focus_polygon(
                painter, w, h, snapshot.focus_polygon_norm,
                COLOR_AMBER if pending else COLOR_WHITE, pending=pending,
            )

        # 3. 自由光标细线准星
        if snapshot.free_cursor_active:
            self._draw_free_cursor_reticle(painter, w, h, snapshot.cursor_norm)

        # 4. 两段式部署：H2 十字细轴定朝向
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

        # 5. 顶部白底细边状态签 + Toast
        self._draw_top_status_capsule(painter, w, h, snapshot)

        # 6. 开战长按细环
        if snapshot.ready_hold_progress > 0.0:
            self._draw_ready_hold_ring(painter, w, h, snapshot.ready_hold_progress)

        # 7. 底部半透白细带提示条
        if snapshot.prompts:
            self._draw_bottom_prompt_bar(painter, w, h, snapshot.prompts)

    # ------------------------------------------------------------------
    # 选择框
    # ------------------------------------------------------------------
    def _draw_focus_polygon(
        self,
        painter: QPainter,
        w: float,
        h: float,
        polygon_norm: Sequence[tuple[float, float]],
        color: QColor,
        *,
        pending: bool = False,
    ) -> None:
        """H1：四角 10px 纯白 2px 细折标（外衬暗描边，内部全留空）；待确认态在四角内侧点琥珀微点。"""
        del color
        pts = [QPointF(u * w, v * h) for u, v in polygon_norm]
        segs = _corner_segments(pts, 10.0, 0.3)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for pn in (_pen(COLOR_HALO, 3.6), _pen(COLOR_WHITE, 2.0)):
            painter.setPen(pn)
            for a, c, b in segs:
                path = QPainterPath(a)
                path.lineTo(c)
                path.lineTo(b)
                painter.drawPath(path)
        if pending:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(COLOR_AMBER))
            for a, c, b in segs:
                ix = a.x() + b.x() - c.x()
                iy = a.y() + b.y() - c.y()
                # 折标内侧（两臂合成方向的一半处）
                painter.drawEllipse(QPointF((c.x() + ix) / 2.0, (c.y() + iy) / 2.0), 2.0, 2.0)

    # ------------------------------------------------------------------
    # 抓取源与连线
    # ------------------------------------------------------------------
    def _draw_tag(self, painter: QPainter, x: float, y: float, text: str, accent: QColor = COLOR_AMBER) -> QRectF:
        """白底细边小签（左侧 2px 强调竖线）。"""
        fm = QFontMetrics(self._font_small)
        rect = QRectF(x, y, fm.horizontalAdvance(text) + 18.0, fm.height() + 4.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 40)))
        painter.drawRect(rect.translated(0.0, 1.0))
        painter.setPen(_pen(COLOR_EDGE, 1.0))
        painter.setBrush(QBrush(COLOR_CAPSULE_BG))
        painter.drawRect(rect)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(accent))
        painter.drawRect(QRectF(rect.left(), rect.top(), 2.0, rect.height()))
        painter.setFont(self._font_small)
        painter.setPen(QPen(COLOR_INK))
        painter.drawText(rect.adjusted(3.0, 0.0, 0.0, 0.0), Qt.AlignmentFlag.AlignCenter, text)
        return rect

    def _draw_holding_source_and_trajectory(
        self,
        painter: QPainter,
        w: float,
        h: float,
        snapshot: HUDStateSnapshot,
    ) -> None:
        """H1：白色细虚线源框 + 白底细边小签 + 纤细白色虚线连线与空心小菱形终点。"""
        assert snapshot.holding_source_norm is not None
        su, sv = snapshot.holding_source_norm
        sx, sy = su * w, sv * h
        now = time.monotonic()

        box_w = max(44.0, min(96.0, w * 0.068))
        box_h = max(44.0, min(96.0, h * 0.11))
        src_rect = QRectF(sx - box_w / 2.0, sy - box_h / 2.0, box_w, box_h)
        offset = (now * 16.0) % 9.0
        painter.setBrush(Qt.BrushStyle.NoBrush)
        halo = _pen(COLOR_HALO, 3.0, Qt.PenStyle.CustomDashLine, Qt.PenCapStyle.FlatCap)
        halo.setDashPattern([2.0, 1.0])
        halo.setDashOffset(offset / 3.0)
        dash = _pen(COLOR_WHITE, 1.4, Qt.PenStyle.CustomDashLine, Qt.PenCapStyle.FlatCap)
        dash.setDashPattern([4.0, 2.4])
        dash.setDashOffset(offset)
        painter.setPen(halo)
        painter.drawRect(src_rect)
        painter.setPen(dash)
        painter.drawRect(src_rect)

        if snapshot.deploy_state == DeployState.HOLDING_STRATEGY:
            tag_text = "策略源"
        elif snapshot.deploy_state == DeployState.HOLDING_EQUIP:
            tag_text = "装备源"
        else:
            tag_text = "抓取源"
        fm = QFontMetrics(self._font_small)
        self._draw_tag(painter, src_rect.left(), src_rect.top() - fm.height() - 9.0, tag_text)

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
        end = QPointF(tx - ux * 5.5, ty - uy * 5.5)
        painter.setPen(_pen(COLOR_HALO, 2.8, cap=Qt.PenCapStyle.FlatCap))
        painter.drawLine(QPointF(sx, sy), end)
        line = _pen(COLOR_WHITE, 1.2, Qt.PenStyle.CustomDashLine, Qt.PenCapStyle.FlatCap)
        line.setDashPattern([5.0, 4.0])
        line.setDashOffset(-(now * 30.0) % 9.0)
        painter.setPen(line)
        painter.drawLine(QPointF(sx, sy), end)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(_pen(COLOR_HALO, 3.2))
        painter.drawPolygon(diamond_polygon(tx, ty, 5.0))
        painter.setPen(_pen(COLOR_WHITE, 1.5))
        painter.drawPolygon(diamond_polygon(tx, ty, 5.0))

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
        """H1：中心留空的分离式正交细白短线准星（不遮挡指针处游戏文字）。"""
        cx, cy = cursor_norm[0] * w, cursor_norm[1] * h
        for pn in (_pen(COLOR_HALO, 3.2), _pen(COLOR_WHITE, 1.4)):
            painter.setPen(pn)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                painter.drawLine(QPointF(cx + dx * 5.0, cy + dy * 5.0), QPointF(cx + dx * 12.0, cy + dy * 12.0))

    # ------------------------------------------------------------------
    # H2 定朝向
    # ------------------------------------------------------------------
    def _draw_direction_wheel(
        self,
        painter: QPainter,
        w: float,
        h: float,
        origin_norm: tuple[float, float],
        preview_dir: str | None,
    ) -> None:
        """H2：中心小圆环 + 十字细轴线 + 四端空心折线箭头；预选方向点亮为双层琥珀折线箭头。"""
        cx = origin_norm[0] * w
        cy = origin_norm[1] * h
        L = max(30.0, min(46.0, min(w, h) * 0.05))
        directions = (("up", 0, -1), ("right", 1, 0), ("down", 0, 1), ("left", -1, 0))
        for key, dx, dy in directions:
            on = preview_dir == key
            for pn in (_pen(COLOR_HALO, 3.0), _pen(COLOR_WHITE, 1.1)):
                painter.setPen(pn)
                painter.drawLine(QPointF(cx + dx * 6.0, cy + dy * 6.0), QPointF(cx + dx * L, cy + dy * L))
            for off in ((0.0, 7.0) if on else (0.0,)):
                tip = QPointF(cx + dx * (L + 6.0 + off), cy + dy * (L + 6.0 + off))
                a = QPointF(tip.x() - dx * 7.0 - dy * 6.0, tip.y() - dy * 7.0 + dx * 6.0)
                b = QPointF(tip.x() - dx * 7.0 + dy * 6.0, tip.y() - dy * 7.0 - dx * 6.0)
                path = QPainterPath(a)
                path.lineTo(tip)
                path.lineTo(b)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(_pen(QColor(0, 0, 0, 85), 3.8))
                painter.drawPath(path)
                painter.setPen(_pen(COLOR_AMBER if on else COLOR_WHITE, 2.2 if on else 1.4))
                painter.drawPath(path)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for pn in (_pen(COLOR_HALO, 3.0), _pen(COLOR_WHITE, 1.2)):
            painter.setPen(pn)
            painter.drawEllipse(QPointF(cx, cy), 3.2, 3.2)

        hint = "推摇杆定朝向 · B 取消"
        fm = QFontMetrics(self._font_small)
        tw = fm.horizontalAdvance(hint) + 18.0
        ty = cy + L + 22.0
        if ty + fm.height() + 6.0 > h - 4.0:
            ty = cy - L - 22.0 - fm.height() - 4.0
        self._draw_tag(painter, cx - tw / 2.0, ty, hint)

    def _draw_nameplate(self, painter: QPainter, rect: QRectF, text: str, accent: QColor, text_color: QColor) -> None:
        """兼容旧接口：白底细边铭牌。"""
        del text_color
        self._draw_tag(painter, rect.left(), rect.top(), text, accent)

    # ------------------------------------------------------------------
    # 顶部状态签
    # ------------------------------------------------------------------
    @staticmethod
    def _format_top_status_text(snapshot: HUDStateSnapshot) -> str:
        """格式化顶部状态签文案，消除自由鼠标与自由光标同义重复段。"""
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
        """顶签左侧竖线颜色：暂停为红，其余为琥珀。"""
        return COLOR_RED if snapshot.paused else COLOR_AMBER

    def _draw_top_status_capsule(
        self,
        painter: QPainter,
        w: float,
        h: float,
        snapshot: HUDStateSnapshot,
    ) -> None:
        """H1：纯白底 1px 细灰描边纤薄横签 + 左侧 2px 琥珀竖线 + 深炭黑文字；Toast 为同风格次级签。"""
        text = self._format_top_status_text(snapshot)
        fm = QFontMetrics(self._font_title)
        bar_h = max(26.0, fm.height() + 10.0)
        bar_w = min(w * 0.9, fm.horizontalAdvance(text) + 34.0)
        top_y = max(8.0, h * 0.012)
        rect = QRectF((w - bar_w) / 2.0, top_y, bar_w, bar_h)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 45)))
        painter.drawRect(rect.translated(0.0, 1.0))
        painter.setPen(_pen(COLOR_CAPSULE_BORDER, 1.0))
        painter.setBrush(QBrush(COLOR_CAPSULE_BG))
        painter.drawRect(rect)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self._status_accent(snapshot)))
        painter.drawRect(QRectF(rect.left() + 1.0, rect.top() + 5.0, 2.0, rect.height() - 10.0))
        text_rect = rect.adjusted(14.0, 0.0, -10.0, 0.0)
        painter.setFont(self._font_title)
        painter.setPen(QPen(COLOR_INK))
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            fm.elidedText(text, Qt.TextElideMode.ElideRight, int(text_rect.width())),
        )

        if snapshot.toast_message:
            tfm = QFontMetrics(self._font_prompt)
            tw = min(w * 0.8, tfm.horizontalAdvance(snapshot.toast_message) + 28.0)
            th = tfm.height() + 8.0
            trect = QRectF((w - tw) / 2.0, rect.bottom() + 5.0, tw, th)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 35)))
            painter.drawRect(trect.translated(0.0, 1.0))
            painter.setPen(_pen(COLOR_CAPSULE_BORDER, 1.0))
            painter.setBrush(QBrush(QColor(255, 255, 255, 236)))
            painter.drawRect(trect)
            painter.setFont(self._font_prompt)
            painter.setPen(QPen(QColor("#333333")))
            inner = trect.adjusted(10.0, 0.0, -10.0, 0.0)
            painter.drawText(
                inner,
                Qt.AlignmentFlag.AlignCenter,
                tfm.elidedText(snapshot.toast_message, Qt.TextElideMode.ElideRight, int(inner.width())),
            )

    # ------------------------------------------------------------------
    # 开战长按细环
    # ------------------------------------------------------------------
    def _draw_ready_hold_ring(
        self,
        painter: QPainter,
        w: float,
        h: float,
        progress: float,
    ) -> None:
        """H1：白色圆片细灰边底衬 + 琥珀细圆弧进度 + 百分比；右侧白底细边签配圆圈三横线 START 图示。"""
        clamped = max(0.0, min(1.0, float(progress)))
        r = 14.0
        fm = QFontMetrics(self._font_prompt)
        label = "长按开战"
        tag_w = _ICON_R * 2.0 + 8.0 + fm.horizontalAdvance(label) + 20.0
        total = r * 2.0 + 10.0 + tag_w
        cx = (w - total) / 2.0 + r
        # 位于顶部盟约图标与名称（y≈128）正下方，避免遮挡盟约及其视觉检测
        cy = max(142.0, h * 0.012 + 134.0)

        painter.setPen(_pen(COLOR_EDGE, 1.0))
        painter.setBrush(QBrush(QColor(255, 255, 255, 236)))
        painter.drawEllipse(QPointF(cx, cy), r + 4.0, r + 4.0)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(_pen(QColor("#E3E3E3"), 2.2))
        painter.drawEllipse(QPointF(cx, cy), r, r)
        arc = _pen(COLOR_AMBER, 2.4, cap=Qt.PenCapStyle.FlatCap)
        painter.setPen(arc)
        painter.drawArc(QRectF(cx - r, cy - r, 2 * r, 2 * r), 90 * 16, int(round(-clamped * 360.0 * 16.0)))
        painter.setFont(self._font_micro)
        painter.setPen(QPen(COLOR_INK))
        painter.drawText(QRectF(cx - r, cy - r, 2 * r, 2 * r), Qt.AlignmentFlag.AlignCenter, f"{int(clamped * 100)}%")

        tag = QRectF(cx + r + 10.0, cy - 12.0, tag_w, 24.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 40)))
        painter.drawRect(tag.translated(0.0, 1.0))
        painter.setPen(_pen(COLOR_EDGE, 1.0))
        painter.setBrush(QBrush(COLOR_CAPSULE_BG))
        painter.drawRect(tag)
        x = paint_button_glyph(painter, ButtonGlyph("start"), tag.left() + 8.0, cy, self._font_badge)
        painter.setFont(self._font_prompt)
        painter.setPen(QPen(COLOR_INK))
        painter.drawText(QRectF(x + 8.0, tag.top(), tag.right() - x - 8.0, tag.height()),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)

    # ------------------------------------------------------------------
    # 底部提示条
    # ------------------------------------------------------------------
    def _draw_bottom_prompt_bar(
        self,
        painter: QPainter,
        w: float,
        h: float,
        prompts: Sequence[PromptItem],
    ) -> None:
        """H1：纤薄半透纯白细带 + 1px 细灰边 + 矢量手柄按钮图示 + 深色文字。"""
        label_fm = QFontMetrics(self._font_prompt)
        item_gap = 20.0
        layouts: list[tuple[list[ButtonGlyph], str, float, float]] = []
        for item in prompts:
            glyphs = parse_button_spec(item.button)
            gw = measure_button_spec(glyphs, self._font_badge)
            lw = float(label_fm.horizontalAdvance(item.label))
            layouts.append((glyphs, item.label, gw, lw))

        pad_x = 18.0
        bar_h = 32.0
        content_w = sum(gw + 6.0 + lw for _g, _l, gw, lw in layouts) + item_gap * max(0, len(layouts) - 1)
        bar_w = min(w * 0.96, content_w + pad_x * 2.0)
        bottom_margin = max(8.0, h * 0.014)
        bar = QRectF((w - bar_w) / 2.0, h - bar_h - bottom_margin, bar_w, bar_h)
        painter.setPen(_pen(QColor(189, 189, 189, 200), 1.0))
        painter.setBrush(QBrush(QColor(255, 255, 255, 212)))
        painter.drawRect(bar)

        x = bar.left() + pad_x
        cy = bar.center().y()
        max_x = bar.right() - pad_x + 2.0
        for glyphs, label, gw, lw in layouts:
            if x + gw + 6.0 + lw > max_x and x > bar.left() + pad_x:
                break
            x = paint_button_spec(painter, glyphs, x, cy, self._font_badge)
            x += 6.0
            painter.setFont(self._font_prompt)
            painter.setPen(QPen(QColor("#333333")))
            painter.drawText(QRectF(x, bar.top(), lw + 2.0, bar_h),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)
            x += lw + item_gap

    def _measure_tokens_width(self, tokens: Sequence[str], fm: QFontMetrics) -> float:
        """兼容旧接口：计算组合按键宽度。"""
        del fm
        return measure_button_spec("+".join(tokens), self._font_badge)

    def _draw_button_tokens(
        self,
        painter: QPainter,
        tokens: Sequence[str],
        start_x: float,
        center_y: float,
        fm: QFontMetrics,
    ) -> float:
        """兼容旧接口：绘制组合按键图示，返回结束 X 坐标。"""
        del fm
        return paint_button_spec(painter, "+".join(tokens), start_x, center_y, self._font_badge)
