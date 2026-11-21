"""罗德岛 PRTS 桌面主控台前端（PRTS Desktop Frontend）。

以《明日方舟》罗德岛终端的深色工业切角卡片、左侧斜切战术导航与纯 QPainter 矢量徽标，
提供四大真实双向联动分区：
    ① OVERVIEW // 终端监控：实时手柄遥测仪、游戏视口与视觉状态卡、战术快捷控制矩阵；
    ② TUNING   // 参数调校：KeymapConfig 核心手感参数滑块与持久化保存；
    ③ KEYMAP   // 卫戍协议战术手册：v0.2.2 全部操作映射任务简报卡片；
    ④ SYSTEM   // 档案与诊断：版本、权限、配置档案与多视图补偿矩阵摘要。

窗口保持普通层级（不置顶），关闭按钮默认隐藏至系统托盘；遥测刷新仅在窗口可见时以 ~16Hz 节流运行。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import platform
from typing import Any, Sequence

from PySide6 import __version__ as PYSIDE_VERSION
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QBrush,
    QCloseEvent,
    QColor,
    QDesktopServices,
    QFont,
    QFontMetrics,
    QHideEvent,
    QLinearGradient,
    QPainter,
    QPaintEvent,
    QPen,
    QPolygonF,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from arkcontroller import __version__ as APP_VERSION
from arkcontroller.config.models import KeymapConfig, LayoutProfile, LayoutViewState
from arkcontroller.input.xinput_driver import GamepadButton, GamepadState
from arkcontroller.overlay.hud_window import (
    COLOR_AMBER,
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    FACE_BUTTON_COLORS,
    chamfer_polygon,
    diamond_polygon,
    measure_button_tokens,
    paint_button_tokens,
)

# ----------------------------------------------------------------------
# 罗德岛终端主题常量
# ----------------------------------------------------------------------
BG_DEEP = QColor("#07090D")
BG_PANEL = QColor("#0E131B")
BG_CARD = QColor(16, 21, 30, 238)
BG_CARD_HEAD = QColor("#131A25")
LINE_DIM = QColor("#1F2A38")
LINE_MID = QColor("#2C3A4D")

FONT_CN = "Microsoft YaHei UI"
FONT_EN = "Bahnschrift"
FONT_MONO = "Consolas"

TELEMETRY_INTERVAL_MS = 60  # ~16Hz 节流刷新

PAGE_TITLES: tuple[tuple[str, str, str], ...] = (
    ("01", "OVERVIEW", "终端监控"),
    ("02", "TUNING", "参数调校"),
    ("03", "KEYMAP", "卫戍协议战术手册"),
    ("04", "SYSTEM", "档案与诊断"),
)

VIEW_STATE_LABELS: dict[str, str] = {
    LayoutViewState.NORMAL.value: "默认态",
    LayoutViewState.SHOP_OPEN.value: "商店展开态",
    LayoutViewState.UNIT_SELECTED.value: "角色特写态",
}


def _font(family: str, size: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    """创建显式正整数点字号的字体，杜绝 QFont::setPointSize(-1) 警告。"""
    f = QFont(family, max(1, int(size)), weight)
    f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return f


def _hex(color: QColor) -> str:
    return color.name(QColor.NameFormat.HexRgb)


def _alpha(color: QColor, alpha: int) -> QColor:
    c = QColor(color)
    c.setAlpha(max(0, min(255, int(alpha))))
    return c


def paint_prts_emblem(painter: QPainter, cx: float, cy: float, size: float, accent: QColor) -> None:
    """纯 QPainter 绘制罗德岛 / PRTS 几何菱形矢量徽标（不依赖任何外部图片）。"""
    r = size / 2.0
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    # 外层菱形骨架
    outer_pen = QPen(accent, max(1.2, size * 0.045))
    outer_pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    painter.setPen(outer_pen)
    painter.setBrush(QBrush(QColor(8, 12, 18, 230)))
    painter.drawPolygon(diamond_polygon(cx, cy, r * 0.96))
    # 外框缺口刻痕
    painter.setPen(QPen(BG_DEEP, max(1.5, size * 0.07)))
    for sx, sy in ((1, 1), (-1, -1)):
        mx, my = cx + sx * r * 0.48, cy + sy * r * 0.48
        painter.drawLine(QPointF(mx - sy * size * 0.05, my + sx * size * 0.05),
                         QPointF(mx + sy * size * 0.05, my - sx * size * 0.05))
    # 内层实心菱形
    grad = QLinearGradient(QPointF(cx, cy - r), QPointF(cx, cy + r))
    grad.setColorAt(0.0, accent.lighter(130))
    grad.setColorAt(1.0, accent.darker(150))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(grad))
    painter.drawPolygon(diamond_polygon(cx, cy, r * 0.58))
    # 中心十字核与黑芯
    painter.setBrush(QBrush(BG_DEEP))
    painter.drawPolygon(diamond_polygon(cx, cy, r * 0.24))
    cross = QPen(QColor("#F4F6F9"), max(1.0, size * 0.04))
    painter.setPen(cross)
    painter.drawLine(QPointF(cx - r * 0.16, cy), QPointF(cx + r * 0.16, cy))
    painter.drawLine(QPointF(cx, cy - r * 0.16), QPointF(cx, cy + r * 0.16))
    # 四角外侧小菱形信标
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(accent))
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        painter.drawPolygon(diamond_polygon(cx + dx * r * 1.14, cy + dy * r * 1.14, size * 0.05))
    painter.restore()


def paint_tactical_texture(painter: QPainter, rect: QRectF, *, spacing: float = 26.0, alpha: int = 9) -> None:
    """绘制背景微弱斜线与网格战术肌理。"""
    painter.save()
    painter.setClipRect(rect)
    painter.setPen(QPen(QColor(120, 190, 255, alpha), 1.0))
    x = rect.left() - rect.height()
    while x < rect.right():
        painter.drawLine(QPointF(x, rect.bottom()), QPointF(x + rect.height(), rect.top()))
        x += spacing
    painter.setPen(QPen(QColor(0, 200, 255, alpha + 6), 1.0))
    gx = rect.left() + 8.0
    while gx < rect.right():
        gy = rect.top() + 8.0
        while gy < rect.bottom():
            painter.drawPoint(QPointF(gx, gy))
            gy += spacing * 2.0
        gx += spacing * 2.0
    painter.restore()


# ----------------------------------------------------------------------
# 运行时数据结构
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class DashboardRuntimeStatus:
    """主控台「游戏视口与视觉状态卡」所需的只读运行态快照。"""

    gamepad_connected: bool = False
    window_found: bool = False
    window_title: str = ""
    client_rect: tuple[int, int, int, int] | None = None
    active_rect: tuple[int, int, int, int] | None = None
    is_foreground: bool = False
    scene_name: str = "自由鼠标"
    scene_locked_manual: bool = False
    paused: bool = False
    shop_open: bool = False
    shop_slot_count: int = 0
    focus_label: str = ""
    deploy_state_name: str = "空闲"
    template_count: int = 0


@dataclass(frozen=True)
class TuningSpec:
    """单个手感参数滑块的定义。"""

    key: str
    title: str
    desc: str
    minimum: float
    maximum: float
    step: float
    decimals: int
    unit: str = ""
    is_int: bool = False


TUNING_SPECS: tuple[TuningSpec, ...] = (
    TuningSpec("left_deadzone", "左摇杆死区", "过滤左摇杆中心漂移，数值越大越不灵敏", 0.0, 0.50, 0.01, 2),
    TuningSpec("right_deadzone", "右摇杆死区", "过滤右摇杆（滚轮）中心漂移", 0.0, 0.50, 0.01, 2),
    TuningSpec("cursor_max_speed_px", "自由光标最高移速", "摇杆推满时虚拟光标每秒移动像素", 200.0, 4000.0, 50.0, 0, " px/s"),
    TuningSpec("cursor_gamma", "摇杆精准度曲线指数", "越大越易微调，越小越跟手", 0.5, 4.0, 0.05, 2),
    TuningSpec("scroll_speed", "滚轮灵敏度", "右摇杆模拟鼠标滚轮的滚动倍率", 0.1, 5.0, 0.1, 1, " ×"),
    TuningSpec("ready_hold_ms", "开战防误触长按时长", "长按 START 读满后才触发开战（0 为单击）", 0.0, 1500.0, 10.0, 0, " ms", True),
    TuningSpec("trigger_threshold", "扳机触发阈值", "LT / RT 按下深度超过该比例才视为按下", 0.05, 1.0, 0.01, 2),
)


# ----------------------------------------------------------------------
# 基础工业风控件
# ----------------------------------------------------------------------
class PrtsCard(QFrame):
    """左上 / 右下 45° 斜切的深色工业卡片，带英文代号 + 中文标题抬头。"""

    HEAD_H = 38

    def __init__(
        self,
        title_en: str,
        title_cn: str,
        *,
        accent: QColor = COLOR_CYAN,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._title_en = title_en
        self._title_cn = title_cn
        self._accent = QColor(accent)
        self._font_en = _font(FONT_EN, 10, QFont.Weight.Bold)
        self._font_cn = _font(FONT_CN, 10, QFont.Weight.Bold)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(16, self.HEAD_H + 8, 16, 14)
        self.body.setSpacing(8)

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            cut = 14.0
            poly = chamfer_polygon(r, tl=cut, br=cut)
            painter.setPen(QPen(LINE_DIM, 1.0))
            painter.setBrush(QBrush(BG_CARD))
            painter.drawPolygon(poly)

            # 抬头条
            head = QRectF(r.left(), r.top(), r.width(), float(self.HEAD_H))
            head_poly = chamfer_polygon(head, tl=cut)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(BG_CARD_HEAD))
            painter.drawPolygon(head_poly)
            painter.setPen(QPen(_alpha(self._accent, 110), 1.0))
            painter.drawLine(QPointF(head.left() + cut, head.bottom()), QPointF(head.right(), head.bottom()))
            painter.setPen(QPen(self._accent, 2.0))
            painter.drawLine(QPointF(head.left() + cut + 2.0, head.bottom()),
                             QPointF(head.left() + cut + 62.0, head.bottom()))

            # 左侧状态立柱 + 菱形信标
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(self._accent))
            painter.drawRect(QRectF(head.left() + cut + 2.0, head.top() + 11.0, 3.0, head.height() - 22.0))
            painter.drawPolygon(diamond_polygon(head.left() + cut + 15.0, head.center().y(), 3.5))

            x = head.left() + cut + 26.0
            painter.setFont(self._font_en)
            fm_en = QFontMetrics(self._font_en)
            painter.setPen(QPen(self._accent))
            en_text = f"{self._title_en} //"
            painter.drawText(QRectF(x, head.top(), fm_en.horizontalAdvance(en_text) + 4.0, head.height()),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, en_text)
            x += fm_en.horizontalAdvance(en_text) + 8.0
            painter.setFont(self._font_cn)
            painter.setPen(QPen(COLOR_TEXT_PRIMARY))
            painter.drawText(QRectF(x, head.top(), head.right() - x - 60.0, head.height()),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._title_cn)

            # 右上微型刻度装饰
            painter.setPen(QPen(_alpha(COLOR_TEXT_MUTED, 120), 1.0))
            tx = head.right() - 12.0
            for k in range(8):
                th = 7.0 if k % 4 == 0 else 3.5
                painter.drawLine(QPointF(tx - k * 5.0, head.center().y() - th / 2.0),
                                 QPointF(tx - k * 5.0, head.center().y() + th / 2.0))

            # 右下斜切角高光
            painter.setPen(QPen(_alpha(self._accent, 160), 1.4))
            painter.drawLine(QPointF(r.right() - cut, r.bottom()), QPointF(r.right(), r.bottom() - cut))
        finally:
            painter.end()


class PrtsButton(QPushButton):
    """纯 QPainter 绘制的斜切角工业按钮（amber 主按钮 / cyan / ghost / danger）。"""

    def __init__(self, text: str, variant: str = "ghost", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._variant = variant
        self._font = _font(FONT_CN, 10, QFont.Weight.Bold)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(40 if variant in ("amber", "cyan") else 34)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        fm = QFontMetrics(self._font)
        return QSize(fm.horizontalAdvance(self.text()) + 48, self.minimumHeight())

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        try:
            r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            hover = self.underMouse() and self.isEnabled()
            down = self.isDown()
            cut = 9.0
            poly = chamfer_polygon(r, tl=cut, br=cut)
            if self._variant == "amber":
                grad = QLinearGradient(r.topLeft(), r.bottomLeft())
                base = COLOR_AMBER.lighter(112) if hover else COLOR_AMBER
                if down:
                    base = COLOR_AMBER.darker(115)
                grad.setColorAt(0.0, base.lighter(110))
                grad.setColorAt(1.0, base.darker(118))
                painter.setPen(QPen(QColor("#FFD08A"), 1.0))
                painter.setBrush(QBrush(grad))
                fg = QColor("#120C02")
            elif self._variant == "cyan":
                painter.setPen(QPen(COLOR_CYAN, 1.2))
                painter.setBrush(QBrush(_alpha(COLOR_CYAN, 70 if hover else 38)))
                fg = COLOR_TEXT_PRIMARY
            elif self._variant == "danger":
                painter.setPen(QPen(COLOR_RED, 1.2))
                painter.setBrush(QBrush(_alpha(COLOR_RED, 90 if hover else 40)))
                fg = QColor("#FFE4E4")
            else:
                painter.setPen(QPen(COLOR_CYAN if hover else LINE_MID, 1.0))
                painter.setBrush(QBrush(QColor("#151C27") if not hover else QColor("#1B2533")))
                fg = COLOR_TEXT_PRIMARY
            if not self.isEnabled():
                painter.setOpacity(0.45)
            painter.drawPolygon(poly)
            # 左侧斜纹装饰
            stripe_col = QColor(0, 0, 0, 70) if self._variant == "amber" else _alpha(COLOR_TEXT_MUTED, 70)
            painter.setPen(QPen(stripe_col, 1.4))
            for k in range(3):
                ox = r.left() + cut + 6.0 + k * 5.0
                painter.drawLine(QPointF(ox, r.bottom() - 6.0), QPointF(ox + 5.0, r.top() + 6.0))
            painter.setFont(self._font)
            painter.setPen(QPen(fg))
            painter.drawText(r.adjusted(22.0, 0.0, -6.0, 0.0), Qt.AlignmentFlag.AlignCenter, self.text())
        finally:
            painter.end()

    def enterEvent(self, event) -> None:  # type: ignore[override]
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        super().leaveEvent(event)
        self.update()


class PrtsToggle(QAbstractButton):
    """战术快捷控制矩阵中的切角开关块（标题 + 英文提示 + 菱形滑钮）。"""

    def __init__(
        self,
        title: str,
        hint: str,
        *,
        accent: QColor = COLOR_CYAN,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setText(title)
        self._hint = hint
        self._accent = QColor(accent)
        self._font_title = _font(FONT_CN, 9, QFont.Weight.Bold)
        self._font_hint = _font(FONT_MONO, 8)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(52)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(190, 52)

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        try:
            r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            on = self.isChecked()
            hover = self.underMouse()
            cut = 8.0
            painter.setPen(QPen(_alpha(self._accent, 200) if on else (LINE_MID if not hover else _alpha(self._accent, 120)), 1.0))
            painter.setBrush(QBrush(_alpha(self._accent, 30) if on else QColor("#121822")))
            painter.drawPolygon(chamfer_polygon(r, tl=cut, br=cut))
            # 左侧状态立柱
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(self._accent if on else LINE_MID))
            painter.drawRect(QRectF(r.left() + 3.0, r.top() + cut + 2.0, 3.0, r.height() - cut * 2.0 - 4.0))

            # 右侧开关轨道 + 菱形滑钮
            track = QRectF(r.right() - 48.0, r.center().y() - 8.0, 38.0, 16.0)
            painter.setPen(QPen(self._accent if on else LINE_MID, 1.0))
            painter.setBrush(QBrush(_alpha(self._accent, 90) if on else QColor("#0B0F15")))
            painter.drawPolygon(chamfer_polygon(track, tl=4.0, br=4.0))
            knob_x = track.right() - 9.0 if on else track.left() + 9.0
            painter.setPen(QPen(QColor("#FFFFFF") if on else COLOR_TEXT_MUTED, 1.0))
            painter.setBrush(QBrush(self._accent if on else QColor("#2A3444")))
            painter.drawPolygon(diamond_polygon(knob_x, track.center().y(), 7.0))

            text_rect = QRectF(r.left() + 14.0, r.top() + 6.0, track.left() - r.left() - 18.0, r.height() * 0.5)
            painter.setFont(self._font_title)
            painter.setPen(QPen(COLOR_TEXT_PRIMARY if on else QColor("#C5CEDB")))
            fm = QFontMetrics(self._font_title)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             fm.elidedText(self.text(), Qt.TextElideMode.ElideRight, int(text_rect.width())))
            hint_rect = QRectF(text_rect.left(), r.center().y() + 1.0, text_rect.width(), r.height() * 0.4)
            painter.setFont(self._font_hint)
            painter.setPen(QPen(_alpha(self._accent, 230) if on else COLOR_TEXT_MUTED))
            hfm = QFontMetrics(self._font_hint)
            painter.drawText(hint_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             hfm.elidedText(self._hint, Qt.TextElideMode.ElideRight, int(hint_rect.width())))
        finally:
            painter.end()

    def enterEvent(self, event) -> None:  # type: ignore[override]
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        super().leaveEvent(event)
        self.update()

    def set_checked_silently(self, checked: bool) -> None:
        """不触发 toggled 信号地同步勾选状态。"""
        self.blockSignals(True)
        self.setChecked(bool(checked))
        self.blockSignals(False)
        self.update()


class _NavButton(QAbstractButton):
    """左侧斜切战术导航按钮。"""

    def __init__(self, index: str, code: str, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self._index = index
        self._code = code
        self._title = title
        self._font_idx = _font(FONT_MONO, 9, QFont.Weight.Bold)
        self._font_code = _font(FONT_EN, 12, QFont.Weight.Bold)
        self._font_title = _font(FONT_CN, 9)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(62)

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        try:
            r = QRectF(self.rect()).adjusted(0.5, 2.5, -0.5, -2.5)
            on = self.isChecked()
            hover = self.underMouse()
            cut = 12.0
            if on:
                grad = QLinearGradient(r.topLeft(), r.topRight())
                grad.setColorAt(0.0, _alpha(COLOR_CYAN, 70))
                grad.setColorAt(1.0, _alpha(COLOR_CYAN, 8))
                painter.setPen(QPen(_alpha(COLOR_CYAN, 150), 1.0))
                painter.setBrush(QBrush(grad))
            else:
                painter.setPen(QPen(LINE_DIM if not hover else LINE_MID, 1.0))
                painter.setBrush(QBrush(QColor("#10161F") if not hover else QColor("#141C27")))
            painter.drawPolygon(chamfer_polygon(r, tr=cut, bl=cut * 0.6))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(COLOR_AMBER if on else LINE_MID))
            painter.drawRect(QRectF(r.left(), r.top() + 6.0, 4.0 if on else 2.0, r.height() - 12.0 - cut * 0.6))

            painter.setFont(self._font_idx)
            painter.setPen(QPen(COLOR_AMBER if on else COLOR_TEXT_MUTED))
            painter.drawText(QRectF(r.left() + 14.0, r.top() + 8.0, 26.0, 18.0),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._index)
            painter.setFont(self._font_code)
            painter.setPen(QPen(COLOR_TEXT_PRIMARY if on else QColor("#AEB9C8")))
            painter.drawText(QRectF(r.left() + 40.0, r.top() + 6.0, r.width() - 52.0, 22.0),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._code)
            painter.setFont(self._font_title)
            painter.setPen(QPen(COLOR_CYAN if on else COLOR_TEXT_MUTED))
            fm = QFontMetrics(self._font_title)
            title_rect = QRectF(r.left() + 40.0, r.top() + 29.0, r.width() - 50.0, 20.0)
            painter.drawText(title_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             fm.elidedText(f"// {self._title}", Qt.TextElideMode.ElideRight, int(title_rect.width())))
            if on:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(COLOR_CYAN))
                painter.drawPolygon(diamond_polygon(r.right() - 14.0, r.center().y() + 4.0, 3.5))
        finally:
            painter.end()

    def enterEvent(self, event) -> None:  # type: ignore[override]
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        super().leaveEvent(event)
        self.update()


class _HeaderBar(QWidget):
    """顶部 RHODES ISLAND // PRTS CONTROLLER TERMINAL 抬头（矢量徽标 + 状态芯片）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(84)
        self._chips: list[tuple[str, str, QColor]] = []
        self._font_title = _font(FONT_EN, 16, QFont.Weight.Bold)
        self._font_sub = _font(FONT_CN, 9)
        self._font_chip_k = _font(FONT_CN, 8)
        self._font_chip_v = _font(FONT_CN, 9, QFont.Weight.Bold)
        self._font_micro = _font(FONT_MONO, 7)

    def set_chips(self, chips: Sequence[tuple[str, str, QColor]]) -> None:
        new = list(chips)
        if new != self._chips:
            self._chips = new
            self.update()

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            w, h = float(self.width()), float(self.height())
            grad = QLinearGradient(0.0, 0.0, 0.0, h)
            grad.setColorAt(0.0, QColor("#0F151E"))
            grad.setColorAt(1.0, QColor("#0A0E14"))
            painter.fillRect(self.rect(), QBrush(grad))
            paint_tactical_texture(painter, QRectF(0.0, 0.0, w, h), spacing=18.0, alpha=10)

            # 底部分割线：青色主线 + 源石橙缺口段
            painter.setPen(QPen(_alpha(COLOR_CYAN, 120), 1.0))
            painter.drawLine(QPointF(0.0, h - 1.0), QPointF(w, h - 1.0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(COLOR_AMBER))
            painter.drawPolygon(QPolygonF([
                QPointF(220.0, h - 4.0), QPointF(330.0, h - 4.0), QPointF(334.0, h), QPointF(216.0, h)]))
            painter.setBrush(QBrush(COLOR_CYAN))
            painter.drawRect(QRectF(0.0, h - 3.0, 200.0, 3.0))

            paint_prts_emblem(painter, 46.0, h / 2.0 - 1.0, 44.0, COLOR_CYAN)

            # 主抬头
            x0 = 88.0
            painter.setFont(self._font_micro)
            painter.setPen(QPen(_alpha(COLOR_CYAN, 200)))
            painter.drawText(QRectF(x0, 12.0, 400.0, 12.0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             "PRTS TACTICAL TERMINAL  ·  NODE-RI-0721  ·  SECURE LINK")
            painter.setFont(self._font_title)
            painter.setPen(QPen(COLOR_TEXT_PRIMARY))
            title = "RHODES ISLAND // PRTS CONTROLLER TERMINAL"
            painter.drawText(QRectF(x0, 24.0, 620.0, 28.0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)
            painter.setFont(self._font_sub)
            painter.setPen(QPen(COLOR_TEXT_MUTED))
            painter.drawText(QRectF(x0, 52.0, 620.0, 20.0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             f"罗德岛 · 卫戍协议 Xbox 手柄零注入战术终端    v{APP_VERSION}")

            # 右侧状态芯片
            chip_w, chip_h, gap = 132.0, 44.0, 8.0
            x = w - 20.0 - len(self._chips) * (chip_w + gap) + gap
            y = (h - chip_h) / 2.0 - 1.0
            for key, value, color in self._chips:
                cr = QRectF(x, y, chip_w, chip_h)
                painter.setPen(QPen(_alpha(color, 150), 1.0))
                painter.setBrush(QBrush(QColor(12, 16, 23, 235)))
                painter.drawPolygon(chamfer_polygon(cr, tl=8.0, br=8.0))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(color))
                painter.drawRect(QRectF(cr.left() + 4.0, cr.top() + 10.0, 3.0, chip_h - 20.0))
                painter.drawPolygon(diamond_polygon(cr.right() - 14.0, cr.top() + 13.0, 3.5))
                painter.setFont(self._font_chip_k)
                painter.setPen(QPen(COLOR_TEXT_MUTED))
                painter.drawText(QRectF(cr.left() + 14.0, cr.top() + 4.0, chip_w - 34.0, 18.0),
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, key)
                painter.setFont(self._font_chip_v)
                painter.setPen(QPen(color))
                fm = QFontMetrics(self._font_chip_v)
                painter.drawText(QRectF(cr.left() + 14.0, cr.top() + 21.0, chip_w - 20.0, 18.0),
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                                 fm.elidedText(value, Qt.TextElideMode.ElideRight, int(chip_w - 20.0)))
                x += chip_w + gap
        finally:
            painter.end()


class _NavPanel(QWidget):
    """左侧斜切战术导航栏背景。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(228)
        self._font_micro = _font(FONT_MONO, 7)

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        try:
            w, h = float(self.width()), float(self.height())
            painter.fillRect(self.rect(), BG_DEEP)
            poly = QPolygonF([
                QPointF(0.0, 0.0), QPointF(w - 1.0, 0.0), QPointF(w - 1.0, h - 40.0),
                QPointF(w - 41.0, h), QPointF(0.0, h)])
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(BG_PANEL))
            painter.drawPolygon(poly)
            paint_tactical_texture(painter, QRectF(0.0, 0.0, w, h), spacing=22.0, alpha=7)
            painter.setPen(QPen(LINE_DIM, 1.0))
            painter.drawLine(QPointF(w - 1.0, 0.0), QPointF(w - 1.0, h - 40.0))
            painter.setPen(QPen(_alpha(COLOR_CYAN, 140), 1.2))
            painter.drawLine(QPointF(w - 1.0, h - 40.0), QPointF(w - 41.0, h))
            painter.setFont(self._font_micro)
            painter.setPen(QPen(_alpha(COLOR_TEXT_MUTED, 200)))
            painter.drawText(QRectF(16.0, 10.0, w - 32.0, 14.0),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "NAVIGATION // 终端分区")
        finally:
            painter.end()


class _PageBackground(QWidget):
    """主内容区背景：深色底 + 斜线网格战术肌理。"""

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), BG_DEEP)
            paint_tactical_texture(painter, QRectF(self.rect()), spacing=28.0, alpha=8)
        finally:
            painter.end()


# ----------------------------------------------------------------------
# ① 作战总览：实时手柄遥测仪 / 视口示意 / 状态卡
# ----------------------------------------------------------------------
_TELEMETRY_ROWS: tuple[tuple[tuple[str, GamepadButton], ...], ...] = (
    (
        ("LB", GamepadButton.LB), ("RB", GamepadButton.RB), ("LT", GamepadButton.LT),
        ("RT", GamepadButton.RT), ("BACK", GamepadButton.BACK), ("START", GamepadButton.START),
        ("LS", GamepadButton.LS), ("RS", GamepadButton.RS),
    ),
    (
        ("↑", GamepadButton.DPAD_UP), ("↓", GamepadButton.DPAD_DOWN), ("←", GamepadButton.DPAD_LEFT),
        ("→", GamepadButton.DPAD_RIGHT), ("A", GamepadButton.A), ("B", GamepadButton.B),
        ("X", GamepadButton.X), ("Y", GamepadButton.Y),
    ),
)


class GamepadTelemetryWidget(QWidget):
    """实时手柄遥测仪：双摇杆 2D 十字靶盘、LT/RT 线性充能条与按键高亮阵列。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state: GamepadState = GamepadState()
        self._left_dz: float = 0.14
        self._right_dz: float = 0.16
        self._trigger_threshold: float = 0.45
        self._font_head = _font(FONT_CN, 9, QFont.Weight.Bold)
        self._font_label = _font(FONT_EN, 10, QFont.Weight.Bold)
        self._font_small = _font(FONT_CN, 8)
        self._font_mono = _font(FONT_MONO, 8, QFont.Weight.Bold)
        self._font_chip = _font("Segoe UI", 9, QFont.Weight.Bold)
        self.setMinimumSize(520, 320)

    @property
    def state(self) -> GamepadState:
        return self._state

    def set_state(self, state: GamepadState | None) -> None:
        new_state = state if state is not None else GamepadState()
        if new_state != self._state:
            self._state = new_state
            self.update()

    def set_calibration(self, left_deadzone: float, right_deadzone: float, trigger_threshold: float) -> None:
        self._left_dz = float(left_deadzone)
        self._right_dz = float(right_deadzone)
        self._trigger_threshold = float(trigger_threshold)
        self.update()

    def pressed_labels(self) -> list[str]:
        """返回当前高亮按键标签（供单测与状态展示）。"""
        labels: list[str] = []
        for row in _TELEMETRY_ROWS:
            for label, btn in row:
                if self._is_pressed(btn):
                    labels.append(label)
        return labels

    def _is_pressed(self, btn: GamepadButton) -> bool:
        s = self._state
        if not s.connected:
            return False
        if btn == GamepadButton.LT:
            return btn in s.buttons or s.lt >= self._trigger_threshold
        if btn == GamepadButton.RT:
            return btn in s.buttons or s.rt >= self._trigger_threshold
        return btn in s.buttons

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            self._paint(painter, float(self.width()), float(self.height()))
        finally:
            painter.end()

    def _paint(self, painter: QPainter, w: float, h: float) -> None:
        s = self._state
        connected = s.connected

        # 顶部连接状态行
        status_col = COLOR_GREEN if connected else COLOR_RED
        painter.setPen(QPen(_alpha(status_col, 90), 4.0))
        painter.setBrush(QBrush(status_col))
        painter.drawPolygon(diamond_polygon(10.0, 13.0, 4.5))
        painter.setFont(self._font_head)
        painter.setPen(QPen(COLOR_TEXT_PRIMARY))
        head = "XINPUT #0  ·  Xbox 手柄已连接" if connected else "XINPUT #0  ·  未检测到 Xbox 手柄"
        painter.drawText(QRectF(24.0, 2.0, w * 0.6, 22.0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, head)
        painter.setFont(self._font_mono)
        painter.setPen(QPen(COLOR_TEXT_MUTED))
        painter.drawText(QRectF(w * 0.45, 2.0, w * 0.55 - 2.0, 22.0),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                         f"PKT {s.packet_number:06d}  ·  16Hz")

        # 布局计算
        array_h = 2 * 26.0 + 8.0
        mid_top = 34.0
        mid_bottom = h - array_h - 14.0
        mid_h = max(60.0, mid_bottom - mid_top)
        trig_w = 26.0
        gap = 18.0
        pad = max(60.0, min(mid_h, (w - 2.0 * trig_w - 3.0 * gap - 8.0) / 2.0))
        total_w = 2.0 * trig_w + 2.0 * pad + 3.0 * gap
        x = (w - total_w) / 2.0
        y = mid_top + (mid_h - pad) / 2.0

        self._paint_trigger(painter, QRectF(x, y, trig_w, pad), "LT", s.lt, self._is_pressed(GamepadButton.LT))
        x += trig_w + gap
        self._paint_stick(painter, QRectF(x, y, pad, pad), "LS", "左摇杆", s.lx, s.ly, self._left_dz,
                          self._is_pressed(GamepadButton.LS))
        x += pad + gap
        self._paint_stick(painter, QRectF(x, y, pad, pad), "RS", "右摇杆", s.rx, s.ry, self._right_dz,
                          self._is_pressed(GamepadButton.RS))
        x += pad + gap
        self._paint_trigger(painter, QRectF(x, y, trig_w, pad), "RT", s.rt, self._is_pressed(GamepadButton.RT))

        # 按键实时高亮阵列
        ay = h - array_h
        n = len(_TELEMETRY_ROWS[0])
        cgap = 6.0
        chip_w = (w - (n - 1) * cgap) / n
        painter.setFont(self._font_chip)
        for ri, row in enumerate(_TELEMETRY_ROWS):
            cy = ay + ri * 30.0
            for ci, (label, btn) in enumerate(row):
                cr = QRectF(ci * (chip_w + cgap), cy, chip_w, 24.0)
                pressed = self._is_pressed(btn)
                if pressed and label in FACE_BUTTON_COLORS:
                    fill, fg = FACE_BUTTON_COLORS[label]
                    border = QColor("#FFFFFF")
                elif pressed:
                    fill, fg, border = COLOR_AMBER, QColor("#111827"), QColor("#FFE0B2")
                else:
                    fill = QColor("#121822")
                    fg = FACE_BUTTON_COLORS[label][0] if label in FACE_BUTTON_COLORS else QColor("#7F8EA3")
                    border = LINE_MID
                painter.setPen(QPen(border, 1.0))
                painter.setBrush(QBrush(fill))
                painter.drawPolygon(chamfer_polygon(cr, tl=5.0, br=5.0))
                painter.setPen(QPen(fg))
                painter.drawText(cr, Qt.AlignmentFlag.AlignCenter, label)

    def _paint_stick(
        self,
        painter: QPainter,
        rect: QRectF,
        code: str,
        title: str,
        sx: float,
        sy: float,
        deadzone: float,
        clicked: bool,
    ) -> None:
        connected = self._state.connected
        painter.setPen(QPen(COLOR_AMBER if clicked else LINE_MID, 1.2 if clicked else 1.0))
        painter.setBrush(QBrush(QColor("#0A0E14")))
        painter.drawPolygon(chamfer_polygon(rect, tl=10.0, br=10.0))
        cx, cy = rect.center().x(), rect.center().y()
        radius = rect.width() / 2.0 - 16.0

        # 网格环与十字线
        for frac in (0.25, 0.5, 0.75):
            painter.setPen(QPen(_alpha(COLOR_CYAN, 26), 1.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), radius * frac, radius * frac)
        painter.setPen(QPen(_alpha(COLOR_CYAN, 120), 1.2))
        painter.drawEllipse(QPointF(cx, cy), radius, radius)
        dash = QPen(_alpha(COLOR_CYAN, 60), 1.0, Qt.PenStyle.CustomDashLine)
        dash.setDashPattern([3.0, 3.0])
        painter.setPen(dash)
        painter.drawLine(QPointF(cx - radius, cy), QPointF(cx + radius, cy))
        painter.drawLine(QPointF(cx, cy - radius), QPointF(cx, cy + radius))
        # 外环刻度
        painter.setPen(QPen(_alpha(COLOR_TEXT_MUTED, 150), 1.0))
        for i in range(24):
            ang = math.radians(i * 15.0)
            r0 = radius + (2.0 if i % 6 else 0.0)
            painter.drawLine(QPointF(cx + math.cos(ang) * r0, cy + math.sin(ang) * r0),
                             QPointF(cx + math.cos(ang) * (radius + 5.0), cy + math.sin(ang) * (radius + 5.0)))

        # 死区圆环
        dz_r = max(2.0, radius * max(0.0, min(0.95, deadzone)))
        painter.setPen(QPen(_alpha(COLOR_RED, 170), 1.0, Qt.PenStyle.DashLine))
        painter.setBrush(QBrush(_alpha(COLOR_RED, 38)))
        painter.drawEllipse(QPointF(cx, cy), dz_r, dz_r)

        # 当前摇杆位置
        mag = math.hypot(sx, sy)
        px = cx + max(-1.0, min(1.0, sx)) * radius
        py = cy - max(-1.0, min(1.0, sy)) * radius
        active = connected and mag > deadzone
        dot_col = COLOR_AMBER if active else _alpha(COLOR_TEXT_MUTED, 200)
        if connected:
            painter.setPen(QPen(_alpha(dot_col, 160), 1.4))
            painter.drawLine(QPointF(cx, cy), QPointF(px, py))
            painter.setPen(QPen(_alpha(dot_col, 70), 1.0))
            painter.drawLine(QPointF(px, rect.top() + 8.0), QPointF(px, rect.bottom() - 8.0))
            painter.drawLine(QPointF(rect.left() + 8.0, py), QPointF(rect.right() - 8.0, py))
            painter.setPen(QPen(QColor("#FFFFFF"), 1.4))
            painter.setBrush(QBrush(dot_col))
            painter.drawEllipse(QPointF(px, py), 6.5, 6.5)

        # 标签与读数
        painter.setFont(self._font_label)
        painter.setPen(QPen(COLOR_AMBER if clicked else COLOR_CYAN))
        painter.drawText(QRectF(rect.left() + 12.0, rect.top() + 4.0, 40.0, 18.0),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, code)
        painter.setFont(self._font_small)
        painter.setPen(QPen(COLOR_TEXT_MUTED))
        painter.drawText(QRectF(rect.left() + 36.0, rect.top() + 4.0, 80.0, 18.0),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)
        painter.setFont(self._font_mono)
        painter.setPen(QPen(COLOR_TEXT_PRIMARY if connected else COLOR_TEXT_MUTED))
        painter.drawText(QRectF(rect.left() + 8.0, rect.bottom() - 22.0, rect.width() - 16.0, 18.0),
                         Qt.AlignmentFlag.AlignCenter, f"X {sx:+.2f}   Y {sy:+.2f}")
        painter.setPen(QPen(_alpha(COLOR_RED, 220)))
        painter.drawText(QRectF(rect.right() - 76.0, rect.top() + 4.0, 66.0, 18.0),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"DZ {deadzone:.2f}")

    def _paint_trigger(self, painter: QPainter, rect: QRectF, code: str, value: float, pressed: bool) -> None:
        value = max(0.0, min(1.0, float(value))) if self._state.connected else 0.0
        label_h = 18.0
        track = QRectF(rect.left() + 4.0, rect.top() + label_h, rect.width() - 8.0, rect.height() - label_h * 2.0)
        painter.setPen(QPen(COLOR_AMBER if pressed else LINE_MID, 1.0))
        painter.setBrush(QBrush(QColor("#0A0E14")))
        painter.drawPolygon(chamfer_polygon(track, tl=4.0, br=4.0))
        # 分段刻度
        painter.setPen(QPen(_alpha(COLOR_TEXT_MUTED, 60), 1.0))
        for k in range(1, 10):
            ty = track.bottom() - track.height() * k / 10.0
            painter.drawLine(QPointF(track.left() + 3.0, ty), QPointF(track.right() - 3.0, ty))
        fill_h = track.height() * value
        if fill_h > 0.5:
            fill = QRectF(track.left() + 2.0, track.bottom() - fill_h, track.width() - 4.0, fill_h - 1.0)
            grad = QLinearGradient(fill.bottomLeft(), fill.topLeft())
            grad.setColorAt(0.0, COLOR_AMBER.darker(140))
            grad.setColorAt(1.0, COLOR_AMBER.lighter(115))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(grad))
            painter.drawRect(fill)
        # 触发阈值标记
        th_y = track.bottom() - track.height() * self._trigger_threshold
        painter.setPen(QPen(COLOR_CYAN, 1.6))
        painter.drawLine(QPointF(track.left() - 4.0, th_y), QPointF(track.right() + 4.0, th_y))
        painter.setFont(self._font_label)
        painter.setPen(QPen(COLOR_AMBER if pressed else COLOR_CYAN))
        painter.drawText(QRectF(rect.left() - 6.0, rect.top(), rect.width() + 12.0, label_h),
                         Qt.AlignmentFlag.AlignCenter, code)
        painter.setFont(self._font_mono)
        painter.setPen(QPen(COLOR_TEXT_PRIMARY))
        painter.drawText(QRectF(rect.left() - 10.0, rect.bottom() - label_h, rect.width() + 20.0, label_h),
                         Qt.AlignmentFlag.AlignCenter, f"{int(round(value * 100))}%")


class _ViewportSchematic(QWidget):
    """游戏客户区与 16:9 有效视口矩形示意图。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client: tuple[int, int, int, int] | None = None
        self._active: tuple[int, int, int, int] | None = None
        self._foreground = False
        self._font = _font(FONT_CN, 8)
        self._font_mono = _font(FONT_MONO, 8, QFont.Weight.Bold)
        self.setMinimumHeight(118)

    def set_rects(
        self,
        client: tuple[int, int, int, int] | None,
        active: tuple[int, int, int, int] | None,
        foreground: bool,
    ) -> None:
        if (client, active, foreground) != (self._client, self._active, self._foreground):
            self._client, self._active, self._foreground = client, active, foreground
            self.update()

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        try:
            w, h = float(self.width()), float(self.height())
            area = QRectF(1.0, 1.0, w - 2.0, h - 2.0)
            painter.setPen(QPen(LINE_DIM, 1.0))
            painter.setBrush(QBrush(QColor("#0A0E14")))
            painter.drawPolygon(chamfer_polygon(area, tl=8.0, br=8.0))
            paint_tactical_texture(painter, area, spacing=12.0, alpha=12)
            if self._client is None or self._client[2] <= 0 or self._client[3] <= 0:
                dash = QPen(_alpha(COLOR_TEXT_MUTED, 160), 1.0, Qt.PenStyle.DashLine)
                painter.setPen(dash)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                inner = area.adjusted(40.0, 16.0, -40.0, -16.0)
                painter.drawRect(inner)
                painter.setFont(self._font)
                painter.setPen(QPen(COLOR_TEXT_MUTED))
                painter.drawText(inner, Qt.AlignmentFlag.AlignCenter, "等待《明日方舟》窗口接入……")
                return
            cl, ct, cw, ch = self._client
            avail = area.adjusted(14.0, 18.0, -14.0, -10.0)
            scale = min(avail.width() / cw, avail.height() / ch)
            dw, dh = cw * scale, ch * scale
            ox = avail.left() + (avail.width() - dw) / 2.0
            oy = avail.top() + (avail.height() - dh) / 2.0
            client_r = QRectF(ox, oy, dw, dh)
            painter.setPen(QPen(_alpha(COLOR_TEXT_MUTED, 200), 1.0))
            painter.setBrush(QBrush(QColor(40, 48, 60, 120)))
            painter.drawRect(client_r)
            if self._active is not None:
                al, at, aw, ah = self._active
                act_r = QRectF(ox + (al - cl) * scale, oy + (at - ct) * scale, aw * scale, ah * scale)
                accent = COLOR_CYAN if self._foreground else COLOR_AMBER
                painter.setPen(QPen(accent, 1.4))
                painter.setBrush(QBrush(_alpha(accent, 40)))
                painter.drawRect(act_r)
                painter.setPen(QPen(QColor("#FFFFFF"), 2.0))
                arm = 8.0
                for (px, py, dx, dy) in (
                    (act_r.left(), act_r.top(), 1, 1), (act_r.right(), act_r.top(), -1, 1),
                    (act_r.left(), act_r.bottom(), 1, -1), (act_r.right(), act_r.bottom(), -1, -1),
                ):
                    painter.drawLine(QPointF(px, py), QPointF(px + dx * arm, py))
                    painter.drawLine(QPointF(px, py), QPointF(px, py + dy * arm))
                painter.setFont(self._font_mono)
                painter.setPen(QPen(accent))
                painter.drawText(act_r, Qt.AlignmentFlag.AlignCenter, f"16:9  {aw}×{ah}")
            painter.setFont(self._font_mono)
            painter.setPen(QPen(COLOR_TEXT_MUTED))
            painter.drawText(QRectF(area.left() + 10.0, area.top() + 2.0, area.width() - 20.0, 15.0),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"CLIENT {cw}×{ch}")
        finally:
            painter.end()


class _KeyValueGrid(QWidget):
    """状态卡键值对网格（左侧冷灰标签 + 右侧彩色数值）。"""

    def __init__(self, keys: Sequence[tuple[str, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(5)
        self._values: dict[str, QLabel] = {}
        for row, (key, label) in enumerate(keys):
            k = QLabel(label)
            k.setObjectName("KvKey")
            v = QLabel("—")
            v.setObjectName("KvValue")
            v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            v.setWordWrap(False)
            v.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            grid.addWidget(k, row, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(v, row, 1)
            self._values[key] = v
        grid.setColumnStretch(1, 1)

    def set_value(self, key: str, text: str, color: QColor | None = None) -> None:
        label = self._values.get(key)
        if label is None:
            return
        if label.text() != text:
            label.setText(text)
            label.setToolTip(text)
        col = _hex(color) if color is not None else _hex(COLOR_TEXT_PRIMARY)
        style = f"color: {col};"
        if label.styleSheet() != style:
            label.setStyleSheet(style)

    def value(self, key: str) -> str:
        label = self._values.get(key)
        return label.text() if label is not None else ""


# ----------------------------------------------------------------------
# ② 操控精调：滑块行与曲线预览
# ----------------------------------------------------------------------
class TuningSliderRow(QWidget):
    """工业风参数滑块行：中文标题 + 英文键名 + 滑块 + 实时数值 + 说明。"""

    value_changed = Signal(str, float)

    def __init__(self, spec: TuningSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 2, 0, 2)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(1)

        title = QLabel(spec.title)
        title.setObjectName("TuneTitle")
        key = QLabel(spec.key)
        key.setObjectName("TuneKey")
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title_box.addWidget(title)
        title_box.addWidget(key)
        title_wrap = QWidget()
        title_wrap.setLayout(title_box)
        title_box.setContentsMargins(0, 0, 0, 0)
        title_wrap.setFixedWidth(170)

        self._steps = int(round((spec.maximum - spec.minimum) / spec.step))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._steps)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(max(1, self._steps // 10))
        self.slider.valueChanged.connect(self._on_slider)

        self.value_label = QLabel()
        self.value_label.setObjectName("TuneValue")
        self.value_label.setFixedWidth(104)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        desc = QLabel(spec.desc)
        desc.setObjectName("TuneDesc")

        grid.addWidget(title_wrap, 0, 0, 2, 1)
        grid.addWidget(self.slider, 0, 1)
        grid.addWidget(self.value_label, 0, 2)
        grid.addWidget(desc, 1, 1, 1, 2)
        grid.setColumnStretch(1, 1)
        self._refresh_label()

    def value(self) -> float:
        raw = self.spec.minimum + self.slider.value() * self.spec.step
        if self.spec.is_int:
            return float(int(round(raw)))
        return round(raw, max(self.spec.decimals, 2))

    def set_value(self, value: float, *, silent: bool = True) -> None:
        pos = int(round((float(value) - self.spec.minimum) / self.spec.step))
        pos = max(0, min(self._steps, pos))
        if silent:
            self.slider.blockSignals(True)
        self.slider.setValue(pos)
        if silent:
            self.slider.blockSignals(False)
        self._refresh_label()

    def _refresh_label(self) -> None:
        v = self.value()
        text = f"{int(v)}" if self.spec.is_int or self.spec.decimals == 0 else f"{v:.{self.spec.decimals}f}"
        self.value_label.setText(f"{text}{self.spec.unit}")

    def _on_slider(self, _pos: int) -> None:
        self._refresh_label()
        self.value_changed.emit(self.spec.key, self.value())


class _CurvePreview(QWidget):
    """自由光标摇杆响应曲线预览（死区 + gamma 指数 + 最高移速）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.deadzone = 0.14
        self.gamma = 2.2
        self.max_speed = 1400.0
        self._font = _font(FONT_MONO, 8)
        self._font_cn = _font(FONT_CN, 8)
        self.setMinimumHeight(210)

    def set_params(self, deadzone: float, gamma: float, max_speed: float) -> None:
        self.deadzone, self.gamma, self.max_speed = float(deadzone), float(gamma), float(max_speed)
        self.update()

    def response(self, x: float) -> float:
        """返回推杆幅度 x ∈ [0,1] 对应的归一化光标速度。"""
        if x <= self.deadzone:
            return 0.0
        t = (x - self.deadzone) / max(1e-6, 1.0 - self.deadzone)
        return max(0.0, min(1.0, t)) ** max(0.05, self.gamma)

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        try:
            w, h = float(self.width()), float(self.height())
            frame = QRectF(1.0, 1.0, w - 2.0, h - 2.0)
            painter.setPen(QPen(LINE_DIM, 1.0))
            painter.setBrush(QBrush(QColor("#0A0E14")))
            painter.drawPolygon(chamfer_polygon(frame, tl=8.0, br=8.0))
            plot = frame.adjusted(46.0, 14.0, -14.0, -28.0)
            painter.setPen(QPen(_alpha(COLOR_CYAN, 24), 1.0))
            for k in range(1, 5):
                gx = plot.left() + plot.width() * k / 5.0
                gy = plot.top() + plot.height() * k / 5.0
                painter.drawLine(QPointF(gx, plot.top()), QPointF(gx, plot.bottom()))
                painter.drawLine(QPointF(plot.left(), gy), QPointF(plot.right(), gy))
            painter.setPen(QPen(LINE_MID, 1.0))
            painter.drawLine(plot.bottomLeft(), plot.bottomRight())
            painter.drawLine(plot.bottomLeft(), plot.topLeft())
            # 死区阴影
            dz_w = plot.width() * self.deadzone
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(_alpha(COLOR_RED, 40)))
            painter.drawRect(QRectF(plot.left(), plot.top(), dz_w, plot.height()))
            # 线性参考
            ref = QPen(_alpha(COLOR_TEXT_MUTED, 90), 1.0, Qt.PenStyle.DashLine)
            painter.setPen(ref)
            painter.drawLine(QPointF(plot.left() + dz_w, plot.bottom()), plot.topRight())
            # 响应曲线
            pts = []
            for i in range(0, 101):
                x = i / 100.0
                pts.append(QPointF(plot.left() + plot.width() * x, plot.bottom() - plot.height() * self.response(x)))
            painter.setPen(QPen(COLOR_CYAN, 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolyline(QPolygonF(pts))
            # 半推杆标记点
            mid_y = plot.bottom() - plot.height() * self.response(0.5)
            painter.setPen(QPen(QColor("#FFFFFF"), 1.0))
            painter.setBrush(QBrush(COLOR_AMBER))
            painter.drawPolygon(diamond_polygon(plot.left() + plot.width() * 0.5, mid_y, 4.5))
            painter.setFont(self._font)
            painter.setPen(QPen(COLOR_TEXT_MUTED))
            painter.drawText(QRectF(4.0, plot.top() - 6.0, 40.0, 14.0), Qt.AlignmentFlag.AlignRight, f"{int(self.max_speed)}")
            painter.drawText(QRectF(4.0, plot.bottom() - 8.0, 40.0, 14.0), Qt.AlignmentFlag.AlignRight, "0")
            painter.drawText(QRectF(plot.left(), plot.bottom() + 6.0, plot.width(), 14.0),
                             Qt.AlignmentFlag.AlignRight, "STICK 100%")
            painter.setPen(QPen(COLOR_AMBER))
            painter.drawText(QRectF(plot.left() + plot.width() * 0.5 + 8.0, mid_y - 16.0, 140.0, 14.0),
                             Qt.AlignmentFlag.AlignLeft,
                             f"50% → {int(self.response(0.5) * self.max_speed)} px/s")
            painter.setFont(self._font_cn)
            painter.setPen(QPen(_alpha(COLOR_RED, 220)))
            painter.drawText(QRectF(plot.left() + 4.0, plot.bottom() + 6.0, 120.0, 14.0),
                             Qt.AlignmentFlag.AlignLeft, f"死区 {self.deadzone:.2f}")
        finally:
            painter.end()


# ----------------------------------------------------------------------
# ③ 键位图谱：任务简报卡片
# ----------------------------------------------------------------------
KEYMAP_BRIEFS: tuple[tuple[str, str, str, QColor, tuple[tuple[str, str], ...]], ...] = (
    ("GLOBAL", "全局热键", "全阶段生效", COLOR_CYAN, (
        ("LS", "推杆：局外移动鼠标 / 局内跳格"),
        ("LS", "按下：吸附导航 ↔ 自由光标"),
        ("RS", "推杆：鼠标滚轮滚动"),
        ("BACK", "循环手动切换场景层（锁层）"),
        ("BACK+START", "打开可视化校准向导"),
        ("LT+BACK", "暂停 / 恢复手柄映射"),
    )),
    ("FREE CURSOR", "局外模拟鼠标", "UNKNOWN_FREE_CURSOR", COLOR_CYAN, (
        ("LS", "平滑移动虚拟鼠标光标"),
        ("A", "鼠标左键单击"),
        ("RT+A", "鼠标左键双击"),
        ("B", "发送 Esc 返回上一级"),
        ("RS", "滚动编队 / 列表 / 结算页"),
    )),
    ("SHOP", "休整期商店运营", "BATTLE_PREP", COLOR_AMBER, (
        ("A>A", "双击 A 买牌（首下仅预选）"),
        ("RT+A", "查看商店干员详情"),
        ("Y", "升本（官方 G 键）"),
        ("X", "刷新商店（官方 D 键）"),
        ("RS", "冻结 / 解冻商店（官方 S 键）"),
        ("LB+RB", "LB / RB 切换四大运营区"),
        ("START", "长按准备开战（官方 C 键）"),
    )),
    ("DEPLOY", "棋盘部署与转向", "4x7 透视棋盘", COLOR_AMBER, (
        ("A", "整备区抓取干员 → 棋盘落位"),
        ("LS", "推杆 / 十字键选四向朝向"),
        ("A>A", "同格回环拖拽，原地转向"),
        ("X", "快速出售（官方 X 键）"),
        ("Y", "拖拽赠予联机队友"),
        ("B", "取消抓取 / 取消定朝向"),
    )),
    ("STRATEGY", "右侧策略头像", "FocusZone.STRATEGY", COLOR_CYAN, (
        ("RT+A", "一键点击释放策略"),
        ("A>A", "锁定策略源后原地释放"),
        ("X", "查看策略说明"),
        ("A", "抓取 → 拖拽至干员 / 商店卡槽"),
        ("B", "无点击安全取消抓取"),
    )),
    ("CHOICE", "2 / 3 / 6 选一决策", "MODAL_CHOICE", COLOR_GREEN, (
        ("LS", "选卡导航（1x2 / 1x3 / 2x3）"),
        ("A", "第 1 下：选中卡片弹出白条"),
        ("A>A", "第 2 下：点击白条确认选择"),
        ("RT+A", "一键选中并确认"),
        ("B", "退回卡片视图重新对比"),
        ("LT", "按住收起弹窗临时看战场"),
    )),
    ("CONFIRM", "回合 / ESC 退出确认窗", "MODAL_CONFIRM", COLOR_RED, (
        ("LB+RB", "左右切换「取消 / 确认」"),
        ("A", "点击当前聚焦按钮"),
        ("B", "直接点击「取消」"),
        ("START", "直接点击「确认」"),
        ("X", "勾选「本局不再提示」"),
    )),
    ("EMOTE", "左下 2x3 交流表情", "FocusZone.EMOTE_PANEL", COLOR_CYAN, (
        ("LT+↓", "打开 / 关闭表情面板"),
        ("LS", "在 2x3 表情间切换"),
        ("RS", "RS 滚轮 / LB·RB 翻页"),
        ("A", "发送当前高亮表情"),
        ("B", "关闭面板回归原分区"),
    )),
    ("LT CHORD", "LT 联机组合键", "休整期 / 作战期", COLOR_AMBER, (
        ("LT+←", "盟约容器详情面板"),
        ("LT+↑", "敌方波次与出怪路线（W）"),
        ("LT+→", "循环切换联机队友"),
        ("LT+A", "直达队友 1 号位（B/X/Y→2/3/4）"),
        ("LT+LB", "展开 / 收起底部商店"),
        ("LT+RB", "一键撤退干员（Q）"),
        ("LT+START", "暂停 / 设置菜单"),
    )),
)


class _KeyCapStrip(QWidget):
    """绘制组合键（'+' 同按）与连按序列（'>' 依次）的工业风键帽条。"""

    def __init__(self, combo: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.combo = combo
        self._font = _font("Segoe UI", 9, QFont.Weight.Bold)
        self.setFixedHeight(26)
        self.setFixedWidth(int(self.natural_width()) + 4)

    def _sequence(self) -> list[list[str]]:
        return [[t.strip() for t in part.split("+") if t.strip()] for part in self.combo.split(">")]

    def natural_width(self) -> float:
        fm = QFontMetrics(self._font)
        seq = self._sequence()
        return sum(measure_button_tokens(tokens, fm) for tokens in seq) + 14.0 * max(0, len(seq) - 1)

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            x = 1.0
            cy = self.height() / 2.0
            for idx, tokens in enumerate(self._sequence()):
                if idx > 0:
                    painter.setFont(self._font)
                    painter.setPen(QPen(COLOR_AMBER))
                    painter.drawText(QRectF(x, cy - 9.0, 14.0, 18.0), Qt.AlignmentFlag.AlignCenter, "›")
                    x += 14.0
                x = paint_button_tokens(painter, tokens, x, cy, self._font)
        finally:
            painter.end()


class KeymapBriefCard(PrtsCard):
    """明日方舟任务简报风格的键位卡片。"""

    def __init__(
        self,
        code: str,
        title: str,
        subtitle: str,
        accent: QColor,
        rows: Sequence[tuple[str, str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(code, title, accent=accent, parent=parent)
        self.rows = tuple(rows)
        sub = QLabel(subtitle)
        sub.setObjectName("BriefSub")
        self.body.addWidget(sub)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        strips = [_KeyCapStrip(combo) for combo, _ in rows]
        strip_w = max((s.width() for s in strips), default=40)
        for i, ((_combo, desc), strip) in enumerate(zip(rows, strips)):
            strip.setFixedWidth(strip_w)
            label = QLabel(desc)
            label.setObjectName("BriefDesc")
            label.setWordWrap(True)
            grid.addWidget(strip, i, 0, Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(label, i, 1)
        grid.setColumnStretch(1, 1)
        self.body.addLayout(grid)
        self.body.addStretch(1)


# ----------------------------------------------------------------------
# 主窗口
# ----------------------------------------------------------------------
_DASHBOARD_QSS = f"""
QWidget {{
    color: {_hex(COLOR_TEXT_PRIMARY)};
    font-family: "{FONT_CN}";
    font-size: 9pt;
}}
QLabel {{ background: transparent; }}
QLabel#KvKey {{ color: {_hex(COLOR_TEXT_MUTED)}; font-size: 9pt; }}
QLabel#KvValue {{ font-family: "{FONT_CN}"; font-size: 9pt; font-weight: bold; }}
QLabel#TuneTitle {{ font-size: 10pt; font-weight: bold; color: {_hex(COLOR_TEXT_PRIMARY)}; }}
QLabel#TuneKey {{ font-family: "{FONT_MONO}"; font-size: 8pt; color: {_hex(COLOR_TEXT_MUTED)}; }}
QLabel#TuneValue {{ font-family: "{FONT_MONO}"; font-size: 11pt; font-weight: bold; color: {_hex(COLOR_CYAN)}; }}
QLabel#TuneDesc {{ font-size: 8pt; color: {_hex(COLOR_TEXT_MUTED)}; }}
QLabel#BriefSub {{ font-family: "{FONT_MONO}"; font-size: 8pt; color: {_hex(COLOR_TEXT_MUTED)}; }}
QLabel#BriefDesc {{ font-size: 9pt; color: #D5DCE6; }}
QLabel#SectionNote {{ font-size: 8pt; color: {_hex(COLOR_TEXT_MUTED)}; }}
QLabel#Feedback {{ font-size: 9pt; font-weight: bold; }}
QLabel#PageTitle {{ font-family: "{FONT_EN}"; font-size: 15pt; font-weight: bold; color: {_hex(COLOR_TEXT_PRIMARY)}; }}
QLabel#PageSub {{ font-size: 9pt; color: {_hex(COLOR_TEXT_MUTED)}; }}
QLabel#NavFoot {{ font-family: "{FONT_MONO}"; font-size: 7pt; color: {_hex(COLOR_TEXT_MUTED)}; }}
QSlider {{ min-height: 22px; background: transparent; }}
QSlider::groove:horizontal {{ height: 4px; background: #1A2330; border: none; }}
QSlider::sub-page:horizontal {{ background: {_hex(COLOR_CYAN)}; }}
QSlider::add-page:horizontal {{ background: #1A2330; }}
QSlider::handle:horizontal {{
    background: {_hex(COLOR_AMBER)};
    border: 1px solid #FFE0B2;
    width: 10px;
    margin: -8px 0;
}}
QSlider::handle:horizontal:hover {{ background: #FFB547; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: #0B0F15; width: 8px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #2C3A4D; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {_hex(COLOR_CYAN)}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
QToolTip {{
    background-color: #0B0E14;
    color: {_hex(COLOR_TEXT_PRIMARY)};
    border: 1px solid {_hex(COLOR_CYAN)};
    padding: 4px;
}}
"""


class PRTSDashboardWindow(QMainWindow):
    """罗德岛 PRTS 桌面主控台窗口（普通层级，不置顶，关闭即隐藏至托盘）。"""

    toggle_pause_requested = Signal()
    overlay_toggled = Signal(bool)
    native_shortcuts_toggled = Signal(bool)
    left_stick_dpad_toggled = Signal(bool)
    keymap_flag_toggled = Signal(str, bool)
    open_calibration_requested = Signal()
    reload_config_requested = Signal()
    tuning_apply_requested = Signal(dict)
    quit_requested = Signal()
    hidden_to_tray = Signal()

    PAGE_OVERVIEW = 0
    PAGE_TUNING = 1
    PAGE_KEYMAP = 2
    PAGE_SYSTEM = 3

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ArkController — RHODES ISLAND // PRTS 主控台")
        # 普通窗口层级：显式不添加 WindowStaysOnTopHint，玩家切入游戏时自动退居后台
        self.setWindowFlags(Qt.WindowType.Window)
        self.setMinimumSize(1180, 760)
        self.resize(1280, 820)
        self.setStyleSheet(_DASHBOARD_QSS)
        try:
            from arkcontroller.overlay.tray import create_prts_icon

            self.setWindowIcon(create_prts_icon(active=True, paused=False))
        except Exception:
            pass

        self._allow_close = False
        self._tray_hint_shown = False
        self._pending_gamepad: GamepadState | None = None
        self._pending_status: DashboardRuntimeStatus | None = None
        self._status: DashboardRuntimeStatus = DashboardRuntimeStatus()
        self._paused = False
        self._keymap: KeymapConfig = KeymapConfig()
        self._tuning_dirty = False

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(TELEMETRY_INTERVAL_MS)
        self._refresh_timer.timeout.connect(self._on_refresh_tick)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self._header = _HeaderBar()
        root_layout.addWidget(self._header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        root_layout.addLayout(body, 1)

        self._nav = self._build_nav()
        body.addWidget(self._nav)

        self._content_bg = _PageBackground()
        content_layout = QVBoxLayout(self._content_bg)
        content_layout.setContentsMargins(20, 16, 20, 16)
        content_layout.setSpacing(12)
        self._page_title = QLabel()
        self._page_title.setObjectName("PageTitle")
        self._page_sub = QLabel()
        self._page_sub.setObjectName("PageSub")
        title_row = QHBoxLayout()
        title_row.addWidget(self._page_title)
        title_row.addSpacing(12)
        title_row.addWidget(self._page_sub, 1, Qt.AlignmentFlag.AlignBottom)
        content_layout.addLayout(title_row)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_overview_page())
        self._stack.addWidget(self._build_tuning_page())
        self._stack.addWidget(self._build_keymap_page())
        self._stack.addWidget(self._build_system_page())
        content_layout.addWidget(self._stack, 1)
        body.addWidget(self._content_bg, 1)

        self.setCentralWidget(root)
        self.set_current_page(self.PAGE_OVERVIEW)
        self._apply_status(self._status)

    # ------------------------------------------------------------------
    # 构建：导航
    # ------------------------------------------------------------------
    def _build_nav(self) -> QWidget:
        nav = _NavPanel()
        layout = QVBoxLayout(nav)
        layout.setContentsMargins(12, 32, 14, 46)
        layout.setSpacing(6)
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        self._nav_buttons: list[_NavButton] = []
        subtitles = ("终端监控", "参数调校", "战术手册", "档案与诊断")
        for idx, ((num, code, _full), sub) in enumerate(zip(PAGE_TITLES, subtitles)):
            btn = _NavButton(num, code, sub)
            self._nav_group.addButton(btn, idx)
            self._nav_buttons.append(btn)
            layout.addWidget(btn)
        self._nav_group.idClicked.connect(self.set_current_page)
        layout.addStretch(1)

        foot = QLabel("RHODES ISLAND PHARMACEUTICALS\nPRTS ACCESS GRANTED · DR.")
        foot.setObjectName("NavFoot")
        layout.addWidget(foot)
        layout.addSpacing(6)
        self._btn_hide = PrtsButton("隐藏至托盘", "ghost")
        self._btn_hide.clicked.connect(self.hide_to_tray)
        layout.addWidget(self._btn_hide)
        self._btn_quit = PrtsButton("退出程序", "danger")
        self._btn_quit.clicked.connect(self.quit_requested.emit)
        layout.addWidget(self._btn_quit)
        return nav

    # ------------------------------------------------------------------
    # 构建：① 作战总览
    # ------------------------------------------------------------------
    def _build_overview_page(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)

        tele_card = PrtsCard("TELEMETRY", "实时手柄遥测仪")
        self.telemetry = GamepadTelemetryWidget()
        tele_card.body.addWidget(self.telemetry, 1)
        grid.addWidget(tele_card, 0, 0)

        vis_card = PrtsCard("VIEWPORT", "游戏视口与视觉状态")
        self._viewport_schematic = _ViewportSchematic()
        vis_card.body.addWidget(self._viewport_schematic)
        self._status_grid = _KeyValueGrid(
            (
                ("window", "游戏窗口"),
                ("resolution", "客户区分辨率"),
                ("viewport", "16:9 有效视口"),
                ("foreground", "前台状态"),
                ("scene", "识别场景"),
                ("shop", "商店 / 动态槽位"),
                ("focus", "当前焦点"),
                ("mapping", "映射状态"),
            )
        )
        vis_card.body.addWidget(self._status_grid)
        vis_card.body.addStretch(1)
        vis_card.setMinimumWidth(360)
        grid.addWidget(vis_card, 0, 1)

        ctrl_card = PrtsCard("QUICK CONTROL", "战术快捷控制矩阵", accent=COLOR_AMBER)
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(14)
        toggles = QGridLayout()
        toggles.setHorizontalSpacing(10)
        toggles.setVerticalSpacing(10)
        self.toggle_pause = PrtsToggle("暂停手柄映射", "LT + BACK", accent=COLOR_RED)
        self.toggle_overlay = PrtsToggle("局内透明 HUD", "OVERLAY · 穿透")
        self.toggle_native = PrtsToggle("PC 原生快捷键", "G/D/S/A/C/W/X/Q")
        self.toggle_dpad = PrtsToggle("左摇杆代替十字键", "LS → D-PAD 跳格")
        self.toggle_auto_shop = PrtsToggle("跨区自动开关商店", "AUTO SHOP")
        self.toggle_vibration = PrtsToggle("震动反馈", "RUMBLE")
        self.toggle_retreat = PrtsToggle("允许 LT+RB 撤退", "RETREAT · Q", accent=COLOR_AMBER)
        self.toggle_pause.clicked.connect(lambda _c: self._on_pause_clicked())
        self.toggle_overlay.toggled.connect(self.overlay_toggled.emit)
        self.toggle_native.toggled.connect(self.native_shortcuts_toggled.emit)
        self.toggle_dpad.toggled.connect(self.left_stick_dpad_toggled.emit)
        self.toggle_auto_shop.toggled.connect(lambda v: self.keymap_flag_toggled.emit("auto_toggle_shop", bool(v)))
        self.toggle_vibration.toggled.connect(lambda v: self.keymap_flag_toggled.emit("vibration_enabled", bool(v)))
        self.toggle_retreat.toggled.connect(lambda v: self.keymap_flag_toggled.emit("allow_retreat_chord", bool(v)))
        for i, t in enumerate((
            self.toggle_pause, self.toggle_overlay, self.toggle_native, self.toggle_dpad,
            self.toggle_auto_shop, self.toggle_vibration, self.toggle_retreat,
        )):
            toggles.addWidget(t, i // 4, i % 4)
        ctrl_row.addLayout(toggles, 1)

        actions = QVBoxLayout()
        actions.setSpacing(10)
        self.btn_calibrate = PrtsButton("启动 16:9 可视化透视校准向导", "amber")
        self.btn_calibrate.setMinimumHeight(52)
        self.btn_calibrate.clicked.connect(self.open_calibration_requested.emit)
        self.btn_reload = PrtsButton("热重载配置与模板", "cyan")
        self.btn_reload.setMinimumHeight(52)
        self.btn_reload.clicked.connect(self.reload_config_requested.emit)
        actions.addWidget(self.btn_calibrate)
        actions.addWidget(self.btn_reload)
        act_wrap = QWidget()
        act_wrap.setLayout(actions)
        actions.setContentsMargins(0, 0, 0, 0)
        act_wrap.setFixedWidth(272)
        ctrl_row.addWidget(act_wrap)
        ctrl_card.body.addLayout(ctrl_row)

        grid.addWidget(ctrl_card, 1, 0, 1, 2)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        grid.setRowStretch(0, 1)
        return page

    # ------------------------------------------------------------------
    # 构建：② 操控精调
    # ------------------------------------------------------------------
    def _build_tuning_page(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(14)

        slider_card = PrtsCard("HANDLING", "摇杆与光标手感参数")
        self.tuning_rows: dict[str, TuningSliderRow] = {}
        for spec in TUNING_SPECS:
            tr = TuningSliderRow(spec)
            tr.value_changed.connect(self._on_tuning_changed)
            self.tuning_rows[spec.key] = tr
            slider_card.body.addWidget(tr)
            if spec.key in ("right_deadzone", "scroll_speed"):
                sep = QFrame()
                sep.setFixedHeight(1)
                sep.setStyleSheet(f"background-color: {_hex(LINE_DIM)};")
                slider_card.body.addWidget(sep)
        slider_card.body.addStretch(1)
        row.addWidget(slider_card, 3)

        side = QVBoxLayout()
        side.setSpacing(14)
        curve_card = PrtsCard("RESPONSE", "自由光标响应曲线")
        self.curve_preview = _CurvePreview()
        curve_card.body.addWidget(self.curve_preview, 1)
        note = QLabel("横轴为摇杆推动幅度，纵轴为光标速度；红色区域为死区，琥珀菱形标记半推杆速度。")
        note.setObjectName("SectionNote")
        note.setWordWrap(True)
        curve_card.body.addWidget(note)
        side.addWidget(curve_card, 1)

        act_card = PrtsCard("COMMIT", "保存与恢复", accent=COLOR_AMBER)
        self.btn_apply_tuning = PrtsButton("应用并持久化保存", "amber")
        self.btn_apply_tuning.clicked.connect(self._on_apply_tuning)
        self.btn_reset_tuning = PrtsButton("恢复默认手感", "ghost")
        self.btn_reset_tuning.clicked.connect(self._on_reset_tuning)
        act_card.body.addWidget(self.btn_apply_tuning)
        act_card.body.addWidget(self.btn_reset_tuning)
        self.tuning_feedback = QLabel("调整滑块后点击「应用并持久化保存」，参数将写入 user_override.json 并立即热刷新。")
        self.tuning_feedback.setObjectName("Feedback")
        self.tuning_feedback.setWordWrap(True)
        self.tuning_feedback.setStyleSheet(f"color: {_hex(COLOR_TEXT_MUTED)};")
        act_card.body.addWidget(self.tuning_feedback)
        act_card.body.addStretch(1)
        side.addWidget(act_card)
        side_wrap = QWidget()
        side_wrap.setLayout(side)
        side.setContentsMargins(0, 0, 0, 0)
        row.addWidget(side_wrap, 2)
        self._load_tuning_from_keymap(self._keymap)
        return page

    # ------------------------------------------------------------------
    # 构建：③ 键位图谱
    # ------------------------------------------------------------------
    def _build_keymap_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setContentsMargins(0, 0, 6, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        self.keymap_cards: list[KeymapBriefCard] = []
        for i, (code, title, sub, accent, rows) in enumerate(KEYMAP_BRIEFS):
            card = KeymapBriefCard(code, title, sub, accent, rows)
            self.keymap_cards.append(card)
            grid.addWidget(card, i // 3, i % 3)
        for c in range(3):
            grid.setColumnStretch(c, 1)
        scroll.setWidget(inner)
        return scroll

    # ------------------------------------------------------------------
    # 构建：④ 系统与配置
    # ------------------------------------------------------------------
    def _build_system_page(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)

        ver_card = PrtsCard("ARCHIVE", "版本与权限档案")
        self._sys_grid = _KeyValueGrid(
            (
                ("version", "ArkController 版本"),
                ("admin", "管理员权限（UAC）"),
                ("python", "Python 运行时"),
                ("qt", "PySide6 / Qt"),
                ("platform", "操作系统"),
            )
        )
        ver_card.body.addWidget(self._sys_grid)
        ver_card.body.addStretch(1)
        grid.addWidget(ver_card, 0, 0)

        cfg_card = PrtsCard("CONFIG", "配置档案与视觉模板")
        self._cfg_grid = _KeyValueGrid(
            (
                ("config_dir", "配置文件目录"),
                ("override", "用户覆盖文件"),
                ("templates", "已加载视觉模板"),
                ("board", "棋盘网格维度"),
            )
        )
        cfg_card.body.addWidget(self._cfg_grid)
        cfg_card.body.addStretch(1)
        grid.addWidget(cfg_card, 0, 1)

        mat_card = PrtsCard("VIEW MATRIX", "多视图缩放补偿矩阵")
        self._matrix_grid = _KeyValueGrid(
            tuple((vs.value, VIEW_STATE_LABELS.get(vs.value, vs.value)) for vs in LayoutViewState)
        )
        mat_card.body.addWidget(self._matrix_grid)
        mnote = QLabel("补偿矩阵由 SIFT 实机标定或可视化校准向导生成，用于商店展开与角色特写时的镜头缩放坐标修正。")
        mnote.setObjectName("SectionNote")
        mnote.setWordWrap(True)
        mat_card.body.addWidget(mnote)
        mat_card.body.addStretch(1)
        grid.addWidget(mat_card, 1, 0)

        op_card = PrtsCard("OPERATIONS", "诊断与操作入口", accent=COLOR_AMBER)
        b1 = PrtsButton("打开 16:9 可视化校准向导", "amber")
        b1.clicked.connect(self.open_calibration_requested.emit)
        b2 = PrtsButton("热重载配置与模板", "cyan")
        b2.clicked.connect(self.reload_config_requested.emit)
        b3 = PrtsButton("打开配置文件目录", "ghost")
        b3.clicked.connect(self._open_config_dir)
        b4 = PrtsButton("完全退出 ArkController", "danger")
        b4.clicked.connect(self.quit_requested.emit)
        for b in (b1, b2, b3, b4):
            op_card.body.addWidget(b)
        op_card.body.addStretch(1)
        grid.addWidget(op_card, 1, 1)

        sec_card = PrtsCard("PROTOCOL", "零注入安全协议自检", accent=COLOR_GREEN)
        self._sec_grid = _KeyValueGrid(
            (
                ("input", "输入合成通道"),
                ("memory", "游戏进程内存"),
                ("hud", "透明 HUD 穿透"),
                ("capture", "截屏隔离"),
                ("frontend", "主控台窗口层级"),
            )
        )
        self._sec_grid.set_value("input", "Win32 SendInput 硬件扫描码 / 鼠标合成", COLOR_GREEN)
        self._sec_grid.set_value("memory", "零读取 · 零写入 · 零 DLL 注入", COLOR_GREEN)
        self._sec_grid.set_value("hud", "WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_LAYERED | WS_EX_TOPMOST",
                                 COLOR_CYAN)
        self._sec_grid.set_value("capture", "SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)", COLOR_CYAN)
        self._sec_grid.set_value("frontend", "普通层级（不置顶）· 关闭即隐藏至托盘 · 隐藏时零遥测开销")
        sec_card.body.addWidget(self._sec_grid)
        grid.addWidget(sec_card, 2, 0, 1, 2)
        grid.setRowStretch(3, 1)
        self._config_dir_path: str = ""
        self._sys_grid.set_value("version", f"v{APP_VERSION}", COLOR_CYAN)
        self._sys_grid.set_value("python", platform.python_version())
        self._sys_grid.set_value("qt", f"PySide6 {PYSIDE_VERSION}")
        self._sys_grid.set_value("platform", f"{platform.system()} {platform.release()}")
        return page

    # ------------------------------------------------------------------
    # 页面切换
    # ------------------------------------------------------------------
    def set_current_page(self, index: int) -> None:
        """切换至指定分区页面（0 总览 / 1 调校 / 2 键位 / 3 系统）。"""
        index = max(0, min(self._stack.count() - 1, int(index)))
        self._stack.setCurrentIndex(index)
        btn = self._nav_buttons[index]
        if not btn.isChecked():
            btn.setChecked(True)
        num, code, title = PAGE_TITLES[index]
        self._page_title.setText(f"{code} // {title}")
        subs = (
            "实时遥测 · 视口状态 · 战术快捷控制",
            "KeymapConfig 核心手感参数调校与持久化",
            f"v{APP_VERSION} 全阶段操作映射任务简报",
            "版本 · 权限 · 配置档案 · 多视图补偿诊断",
        )
        self._page_sub.setText(f"SECTION {num}  ·  {subs[index]}")

    def current_page_index(self) -> int:
        return self._stack.currentIndex()

    # ------------------------------------------------------------------
    # 数据输入接口
    # ------------------------------------------------------------------
    def push_gamepad_state(self, state: GamepadState | None) -> None:
        """缓存最新手柄状态（零开销，仅在可见刷新节拍内应用）。"""
        self._pending_gamepad = state

    def push_runtime_status(self, status: DashboardRuntimeStatus) -> None:
        """缓存最新运行态快照（零开销，仅在可见刷新节拍内应用）。"""
        self._pending_status = status

    def _flush_pending(self) -> None:
        """立即应用缓存的遥测与运行态数据。"""
        if self._pending_gamepad is not None:
            self.telemetry.set_state(self._pending_gamepad)
            self._pending_gamepad = None
        if self._pending_status is not None:
            status = self._pending_status
            self._pending_status = None
            if status != self._status:
                self._status = status
                self._apply_status(status)

    def _on_refresh_tick(self) -> None:
        if not self.isVisible():
            self._refresh_timer.stop()
            return
        self._flush_pending()

    @property
    def refresh_active(self) -> bool:
        """遥测节流刷新定时器是否正在运行。"""
        return self._refresh_timer.isActive()

    def _apply_status(self, s: DashboardRuntimeStatus) -> None:
        g = self._status_grid
        if s.window_found:
            title = s.window_title or "明日方舟"
            g.set_value("window", f"已接入 · {title}", COLOR_GREEN)
        else:
            g.set_value("window", "未检测到《明日方舟》", COLOR_RED)
        if s.client_rect is not None:
            g.set_value("resolution", f"{s.client_rect[2]} × {s.client_rect[3]}")
        else:
            g.set_value("resolution", "—", COLOR_TEXT_MUTED)
        if s.active_rect is not None:
            al, at, aw, ah = s.active_rect
            g.set_value("viewport", f"{aw} × {ah}  @ ({al}, {at})", COLOR_CYAN)
        else:
            g.set_value("viewport", "—", COLOR_TEXT_MUTED)
        if s.window_found:
            g.set_value("foreground", "前台激活" if s.is_foreground else "后台挂起（自动静默）",
                        COLOR_GREEN if s.is_foreground else COLOR_AMBER)
        else:
            g.set_value("foreground", "—", COLOR_TEXT_MUTED)
        scene = s.scene_name + ("  · 手动锁层" if s.scene_locked_manual else "")
        g.set_value("scene", scene, COLOR_AMBER if s.scene_locked_manual else COLOR_CYAN)
        shop = f"{'展开' if s.shop_open else '收起'} · {s.shop_slot_count} 槽" if s.shop_slot_count else (
            "展开" if s.shop_open else "收起")
        g.set_value("shop", shop)
        focus = s.focus_label or "—"
        if s.deploy_state_name and s.deploy_state_name != "空闲":
            focus = f"{focus} · {s.deploy_state_name}"
        g.set_value("focus", focus)
        g.set_value("mapping", "已暂停" if s.paused else "运行中", COLOR_RED if s.paused else COLOR_GREEN)
        self._viewport_schematic.set_rects(s.client_rect, s.active_rect, s.is_foreground)
        self._set_paused_ui(s.paused)
        self._header.set_chips(
            [
                ("XBOX 手柄", "已连接" if s.gamepad_connected else "未连接",
                 COLOR_GREEN if s.gamepad_connected else COLOR_RED),
                ("游戏窗口", ("前台" if s.is_foreground else "后台") if s.window_found else "未接入",
                 (COLOR_CYAN if s.is_foreground else COLOR_AMBER) if s.window_found else COLOR_TEXT_MUTED),
                ("手柄映射", "已暂停" if s.paused else "运行中", COLOR_RED if s.paused else COLOR_GREEN),
            ]
        )

    def _set_paused_ui(self, paused: bool) -> None:
        self._paused = bool(paused)
        self.toggle_pause.set_checked_silently(self._paused)

    def sync_settings(
        self,
        *,
        keymap: KeymapConfig,
        overlay_enabled: bool,
        paused: bool,
        refresh_tuning: bool = False,
    ) -> None:
        """从主控运行态反向同步开关、遥测标定与调校滑块（不触发回环信号）。"""
        self._keymap = keymap
        self.toggle_overlay.set_checked_silently(overlay_enabled)
        self.toggle_native.set_checked_silently(keymap.use_native_keyboard_shortcuts)
        self.toggle_dpad.set_checked_silently(keymap.left_stick_mode == "dpad")
        self.toggle_auto_shop.set_checked_silently(keymap.auto_toggle_shop)
        self.toggle_vibration.set_checked_silently(keymap.vibration_enabled)
        self.toggle_retreat.set_checked_silently(keymap.allow_retreat_chord)
        self._set_paused_ui(paused)
        self.telemetry.set_calibration(keymap.left_deadzone, keymap.right_deadzone, keymap.trigger_threshold)
        if refresh_tuning or not self._tuning_dirty:
            self._load_tuning_from_keymap(keymap)

    def set_system_info(
        self,
        *,
        version: str | None = None,
        is_admin: bool | None = None,
        config_dir: str | None = None,
        override_path: str | None = None,
        template_count: int | None = None,
        layout: LayoutProfile | None = None,
    ) -> None:
        """刷新「系统与配置」页档案信息。"""
        if version is not None:
            self._sys_grid.set_value("version", f"v{version}", COLOR_CYAN)
        if is_admin is not None:
            self._sys_grid.set_value(
                "admin",
                "已提权（可操控官方 PC 客户端）" if is_admin else "标准权限（官方 PC 端需管理员运行）",
                COLOR_GREEN if is_admin else COLOR_AMBER,
            )
        if config_dir is not None:
            self._config_dir_path = config_dir
            self._cfg_grid.set_value("config_dir", config_dir)
        if override_path is not None:
            self._cfg_grid.set_value("override", override_path)
        if template_count is not None:
            self._cfg_grid.set_value("templates", f"{int(template_count)} 个场景模板",
                                     COLOR_CYAN if template_count else COLOR_TEXT_MUTED)
        if layout is not None:
            q = layout.board_quad
            self._cfg_grid.set_value("board", f"{q.rows} 行 × {q.cols} 列 · 4 点单应性透视", COLOR_CYAN)
            for vs in LayoutViewState:
                self._matrix_grid.set_value(vs.value, self._describe_view(layout, vs))

    @staticmethod
    def _describe_view(layout: LayoutProfile, vs: LayoutViewState) -> str:
        if vs == LayoutViewState.NORMAL:
            return "基准坐标系（恒等变换）"
        override = layout.view_overrides.get(vs.value)
        if override is None:
            return "未配置（沿用基准坐标）"
        parts: list[str] = []
        t = override.board_transform
        if override.board_quad is not None:
            parts.append("棋盘独立四角")
        elif t is not None:
            parts.append(f"棋盘 s=({t.scale_x:.3f}, {t.scale_y:.3f}) Δ=({t.offset_u:+.3f}, {t.offset_v:+.3f})")
        for name, zone, tf in (
            ("整备", override.bench_zone, override.bench_transform),
            ("装备", override.equip_zone, override.equip_transform),
        ):
            if zone is not None:
                parts.append(f"{name}独立槽位")
            elif tf is not None:
                parts.append(f"{name} s={tf.scale_x:.3f}")
        if override.shop_zone is not None:
            parts.append("商店独立槽位")
        return " · ".join(parts) if parts else "已配置（恒等）"

    def set_tuning_feedback(self, text: str, ok: bool = True) -> None:
        """显示参数保存结果反馈。"""
        self.tuning_feedback.setText(text)
        self.tuning_feedback.setStyleSheet(f"color: {_hex(COLOR_GREEN if ok else COLOR_RED)};")
        if ok:
            self._tuning_dirty = False

    # ------------------------------------------------------------------
    # 调校逻辑
    # ------------------------------------------------------------------
    def tuning_values(self) -> dict[str, Any]:
        """返回当前滑块对应的 KeymapConfig 字段字典。"""
        values: dict[str, Any] = {}
        for spec in TUNING_SPECS:
            v = self.tuning_rows[spec.key].value()
            values[spec.key] = int(v) if spec.is_int else float(v)
        return values

    def _load_tuning_from_keymap(self, keymap: KeymapConfig) -> None:
        for spec in TUNING_SPECS:
            row = self.tuning_rows.get(spec.key)
            if row is not None:
                row.set_value(float(getattr(keymap, spec.key)), silent=True)
        self._tuning_dirty = False
        self._refresh_curve()

    def _refresh_curve(self) -> None:
        self.curve_preview.set_params(
            self.tuning_rows["left_deadzone"].value(),
            self.tuning_rows["cursor_gamma"].value(),
            self.tuning_rows["cursor_max_speed_px"].value(),
        )

    def _on_tuning_changed(self, _key: str, _value: float) -> None:
        self._tuning_dirty = True
        self._refresh_curve()
        self.tuning_feedback.setText("存在未保存的参数修改。")
        self.tuning_feedback.setStyleSheet(f"color: {_hex(COLOR_AMBER)};")

    def _on_apply_tuning(self) -> None:
        self.tuning_apply_requested.emit(self.tuning_values())

    def _on_reset_tuning(self) -> None:
        defaults = KeymapConfig()
        self._load_tuning_from_keymap(defaults)
        self.tuning_apply_requested.emit(self.tuning_values())

    def _on_pause_clicked(self) -> None:
        # 由主控切换暂停状态并通过 sync 回写；先恢复为当前真实状态避免闪烁不一致
        self.toggle_pause.set_checked_silently(self._paused)
        self.toggle_pause_requested.emit()

    def _open_config_dir(self) -> None:
        if self._config_dir_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._config_dir_path))

    # ------------------------------------------------------------------
    # 显示 / 隐藏 / 关闭
    # ------------------------------------------------------------------
    def show_and_raise(self) -> None:
        """唤出并前置主控台窗口。"""
        if self.isMinimized():
            self.showNormal()
        else:
            self.show()
        self.raise_()
        self.activateWindow()

    def hide_to_tray(self) -> None:
        """隐藏至系统托盘（首次隐藏发出提示信号）。"""
        self.hide()
        if not self._tray_hint_shown:
            self._tray_hint_shown = True
            self.hidden_to_tray.emit()

    def force_close(self) -> None:
        """程序退出时真正关闭窗口。"""
        self._allow_close = True
        self._refresh_timer.stop()
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        if self._allow_close:
            self._refresh_timer.stop()
            event.accept()
            return
        event.ignore()
        self.hide_to_tray()

    def showEvent(self, event: QShowEvent) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._flush_pending()
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def hideEvent(self, event: QHideEvent) -> None:  # type: ignore[override]
        super().hideEvent(event)
        self._refresh_timer.stop()


__all__ = [
    "DashboardRuntimeStatus",
    "GamepadTelemetryWidget",
    "KEYMAP_BRIEFS",
    "PRTSDashboardWindow",
    "PrtsButton",
    "PrtsCard",
    "PrtsToggle",
    "TUNING_SPECS",
    "paint_prts_emblem",
    "paint_tactical_texture",
]
