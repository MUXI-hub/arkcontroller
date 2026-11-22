"""罗德岛 PRTS 桌面主控台前端（方案 B v2 · 终末地 / 莱茵纯白实验风）。

瓷白主底 + 顶部水平极简导航 + 1px 浅灰几何分割线大留白网格排版，琥珀 #F59E0B 单点强调：
    ① 总览：实时手柄遥测（可见时 ~16Hz 节流）、游戏视口大字号信息、空心黄勾快捷开关、手感参数速览；
    ② 调校：KeymapConfig 七项手感滑块、自由光标响应曲线、应用并持久化保存 / 恢复默认；
    ③ 键位：复用 HUD 手柄矢量按钮绘制器渲染 v0.2.2 全部操作简报；
    ④ 系统：版本、UAC、配置路径、多视图补偿矩阵与零注入安全协议自检。

窗口保持普通层级（不置顶），关闭按钮默认隐藏至系统托盘并停止遥测定时器。
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
    QPainter,
    QPaintEvent,
    QPen,
    QPolygonF,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QCheckBox,
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
    COLOR_GREEN,
    COLOR_RED,
    ButtonGlyph,
    measure_button_spec,
    paint_button_glyph,
    paint_button_spec,
    parse_button_spec,
)

# ----------------------------------------------------------------------
# 方案 B 主题常量
# ----------------------------------------------------------------------
BG = QColor("#FAFAFA")
WHITE = QColor("#FFFFFF")
LINE = QColor("#E6E6E6")
EDGE = QColor("#BDBDBD")
INK = QColor("#1A1A1A")
SUB = QColor("#8C8C8C")
AMBER_TEXT = QColor("#B45309")   # 浅底上可读的琥珀深色文字

FONT_CN = "Microsoft YaHei UI"
FONT_EN = "Segoe UI"
FONT_MONO = "Consolas"

TELEMETRY_INTERVAL_MS = 60  # ~16Hz 节流刷新

PAGE_TITLES: tuple[tuple[str, str, str], ...] = (
    ("总览", "OVERVIEW", "手柄、游戏窗口与核心参数的实时概况"),
    ("调校", "TUNING", "摇杆与光标手感参数，保存后写入 user_override.json 并立即生效"),
    ("键位", "KEYMAP", f"卫戍协议 v0.2.2 全阶段 Xbox 手柄操作手册"),
    ("系统", "SYSTEM", "版本、权限、配置档案与安全协议自检"),
)

VIEW_STATE_LABELS: dict[str, str] = {
    LayoutViewState.NORMAL.value: "默认态",
    LayoutViewState.SHOP_OPEN.value: "商店展开态",
    LayoutViewState.UNIT_SELECTED.value: "角色特写态",
}


def _font(family: str, size: int, weight: QFont.Weight = QFont.Weight.Normal, spacing: float = 0.0) -> QFont:
    """创建显式正整数点字号的字体，杜绝 QFont::setPointSize(-1) 警告。"""
    f = QFont(family, max(1, int(size)), weight)
    f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    if spacing:
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return f


def _hex(color: QColor) -> str:
    return color.name(QColor.NameFormat.HexRgb)


def paint_amber_check(painter: QPainter, box: QRectF, checked: bool, *, enabled: bool = True) -> None:
    """白底 1px 细灰空心方框；勾选时框内绘制约 2px 琥珀线条小勾（无背景填充）。"""
    painter.setPen(QPen(EDGE if enabled else LINE, 1.0))
    painter.setBrush(QBrush(WHITE))
    painter.drawRect(box)
    if checked:
        pen = QPen(COLOR_AMBER, 2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        w, h = box.width(), box.height()
        painter.drawPolyline(QPolygonF([
            QPointF(box.left() + w * 0.23, box.top() + h * 0.52),
            QPointF(box.left() + w * 0.43, box.top() + h * 0.74),
            QPointF(box.left() + w * 0.80, box.top() + h * 0.27),
        ]))


_UI_ASSET_SVGS: dict[str, str] = {
    "check": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">'
        '<polyline points="2.6,6.3 5,8.8 9.6,3.2" fill="none" stroke="#F59E0B" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg>'
    ),
    "arrow_down": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 10 10">'
        '<polyline points="2,3.5 5,6.5 8,3.5" fill="none" stroke="#7A7A7A" stroke-width="1.4"/></svg>'
    ),
    "arrow_up": (
        '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 10 10">'
        '<polyline points="2,6.5 5,3.5 8,6.5" fill="none" stroke="#7A7A7A" stroke-width="1.4"/></svg>'
    ),
}
_UI_ASSET_CACHE: dict[str, str] = {}


def ui_asset_path(name: str) -> str:
    """返回 QSS 可引用的极简矢量 SVG 资源路径（首次调用时写入系统临时目录，不依赖仓库外部图片）。"""
    if name in _UI_ASSET_CACHE:
        return _UI_ASSET_CACHE[name]
    import os
    import tempfile

    folder = os.path.join(tempfile.gettempdir(), "arkcontroller_ui")
    path = os.path.join(folder, f"{name}.svg")
    try:
        os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            fp.write(_UI_ASSET_SVGS[name])
    except OSError:
        return ""
    _UI_ASSET_CACHE[name] = path.replace("\\", "/")
    return _UI_ASSET_CACHE[name]


def form_control_qss() -> str:
    """下拉框 / 数值框箭头与菜单勾选标记的白底细线 QSS（矢量 SVG）。"""
    down, up, check = ui_asset_path("arrow_down"), ui_asset_path("arrow_up"), ui_asset_path("check")
    return f"""
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 22px;
                border: none;
                border-left: 1px solid #EDEDED;
                background: #FFFFFF;
            }}
            QComboBox::down-arrow {{ image: url("{down}"); width: 10px; height: 10px; }}
            QSpinBox::up-button {{
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 18px;
                border: none;
                border-left: 1px solid #EDEDED;
                background: #FFFFFF;
            }}
            QSpinBox::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 18px;
                border: none;
                border-left: 1px solid #EDEDED;
                background: #FFFFFF;
            }}
            QSpinBox::up-arrow {{ image: url("{up}"); width: 9px; height: 9px; }}
            QSpinBox::down-arrow {{ image: url("{down}"); width: 9px; height: 9px; }}
            QMenu::indicator:checked {{ image: url("{check}"); }}
            """


def paint_prts_emblem(painter: QPainter, cx: float, cy: float, size: float, accent: QColor = COLOR_AMBER) -> None:
    """方案 B 品牌标：琥珀小方点（兼容旧接口名）。"""
    s = size * 0.2
    painter.save()
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(accent))
    painter.drawRect(QRectF(cx - s / 2.0, cy - s / 2.0, s, s))
    painter.restore()


def paint_tactical_texture(painter: QPainter, rect: QRectF, **_kwargs: Any) -> None:
    """兼容旧接口：方案 B 不使用背景肌理，仅铺瓷白底。"""
    painter.fillRect(rect, BG)


# ----------------------------------------------------------------------
# 运行时数据结构
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class DashboardRuntimeStatus:
    """主控台「游戏窗口」区所需的只读运行态快照。"""

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
    TuningSpec("cursor_max_speed_px", "光标最高移速", "摇杆推满时虚拟光标每秒移动像素", 200.0, 4000.0, 50.0, 0, " px/s"),
    TuningSpec("cursor_gamma", "精准度曲线", "越大越易微调，越小越跟手", 0.5, 4.0, 0.05, 2),
    TuningSpec("scroll_speed", "滚轮灵敏度", "右摇杆模拟鼠标滚轮的滚动倍率", 0.1, 5.0, 0.1, 1, " ×"),
    TuningSpec("ready_hold_ms", "长按开战时长", "长按 START 读满后才触发开战（0 为单击）", 0.0, 1500.0, 10.0, 0, " ms", True),
    TuningSpec("trigger_threshold", "扳机触发阈值", "LT / RT 按下深度超过该比例才视为按下", 0.05, 1.0, 0.01, 2),
)


def _format_value(spec: TuningSpec, v: float) -> str:
    text = f"{int(v)}" if spec.is_int or spec.decimals == 0 else f"{v:.{spec.decimals}f}"
    return f"{text}{spec.unit}"


# ----------------------------------------------------------------------
# 基础控件
# ----------------------------------------------------------------------
class LineButton(QPushButton):
    """方案 B 统一按钮：白底 + 1px 细描边 + 深色文字（无大面积色块）。"""

    def __init__(self, text: str, *, danger: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._danger = danger
        self._font = _font(FONT_CN, 10)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(38)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(QFontMetrics(self._font).horizontalAdvance(self.text()) + 48, self.minimumHeight())

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            hover = self.underMouse() and self.isEnabled()
            p.setPen(QPen(EDGE if hover else LINE, 1.0))
            p.setBrush(QBrush(QColor("#F5F5F5") if self.isDown() else WHITE))
            p.drawRect(r)
            if not self.isEnabled():
                p.setOpacity(0.45)
            p.setFont(self._font)
            p.setPen(QPen(COLOR_RED if self._danger else INK))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.text())
        finally:
            p.end()

    def enterEvent(self, event) -> None:  # type: ignore[override]
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        super().leaveEvent(event)
        self.update()


# 兼容旧名称
PrtsButton = LineButton


class CheckRow(QAbstractButton):
    """快捷开关行：白底细灰空心框 + 琥珀线条小勾 + 标题与说明。"""

    def __init__(self, title: str, hint: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setText(title)
        self._hint = hint
        self._font_title = _font(FONT_CN, 10)
        self._font_hint = _font(FONT_CN, 8)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(44)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    @property
    def hint(self) -> str:
        return self._hint

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(240, 44)

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            h = float(self.height())
            box = QRectF(0.5, 6.5, 14.0, 14.0)
            paint_amber_check(p, box, self.isChecked())
            tx = 26.0
            avail = int(self.width() - tx - 4)
            p.setFont(self._font_title)
            p.setPen(QPen(INK))
            fm = QFontMetrics(self._font_title)
            p.drawText(QRectF(tx, 2.0, avail, 22.0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       fm.elidedText(self.text(), Qt.TextElideMode.ElideRight, avail))
            p.setFont(self._font_hint)
            p.setPen(QPen(SUB))
            hfm = QFontMetrics(self._font_hint)
            p.drawText(QRectF(tx, 23.0, avail, h - 25.0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                       hfm.elidedText(self._hint, Qt.TextElideMode.ElideRight, avail))
        finally:
            p.end()

    def set_checked_silently(self, checked: bool) -> None:
        """不触发 toggled 信号地同步勾选状态。"""
        self.blockSignals(True)
        self.setChecked(bool(checked))
        self.blockSignals(False)
        self.update()


# 兼容旧名称
PrtsToggle = CheckRow


class AmberCheckBox(QCheckBox):
    """方案 B 风格复选框（白底细灰空心框 + 琥珀线条小勾），供校准向导等普通表单复用。"""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._font = _font(FONT_CN, 9)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(QFontMetrics(self._font).horizontalAdvance(self.text()) + 26, 22)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return self.sizeHint()

    def hitButton(self, pos) -> bool:  # type: ignore[override]
        return self.rect().contains(pos)

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            cy = self.height() / 2.0
            paint_amber_check(p, QRectF(0.5, cy - 6.5, 13.0, 13.0), self.isChecked(), enabled=self.isEnabled())
            p.setFont(self._font)
            p.setPen(QPen(INK))
            p.drawText(QRectF(20.0, 0.0, self.width() - 20.0, self.height()),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text())
        finally:
            p.end()


class SectionLabel(QWidget):
    """区块标题：英文小号字距标 + 中文粗体标题。"""

    def __init__(self, en: str, cn: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.en, self.cn = en, cn
        self._f_en = _font(FONT_EN, 7, QFont.Weight.DemiBold, 1.6)
        self._f_cn = _font(FONT_CN, 11, QFont.Weight.Bold)
        self.setFixedHeight(40)

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            p.setFont(self._f_en)
            p.setPen(QPen(SUB))
            p.drawText(QRectF(0, 0, self.width(), 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.en)
            p.setFont(self._f_cn)
            p.setPen(QPen(INK))
            p.drawText(QRectF(0, 14, self.width(), 24), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.cn)
        finally:
            p.end()


class Cell(QWidget):
    """网格单元：按需绘制右 / 下 1px 浅灰分割线。"""

    def __init__(self, *, right: bool = False, bottom: bool = False, top: bool = False,
                 margins: tuple[int, int, int, int] = (0, 24, 32, 24), parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._right, self._bottom, self._top = right, bottom, top
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(*margins)
        self.body.setSpacing(10)

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        p = QPainter(self)
        try:
            p.setPen(QPen(LINE, 1.0))
            w, h = self.width(), self.height()
            if self._right:
                p.drawLine(w - 1, 0, w - 1, h)
            if self._bottom:
                p.drawLine(0, h - 1, w, h - 1)
            if self._top:
                p.drawLine(0, 0, w, 0)
        finally:
            p.end()


# 兼容旧名称
PrtsCard = Cell


class _TabButton(QAbstractButton):
    """顶部水平极简 Tab：选中深色粗体 + 2px 琥珀下划线。"""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self._f_on = _font(FONT_CN, 11, QFont.Weight.Bold)
        self._f_off = _font(FONT_CN, 11)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(QFontMetrics(self._f_on).horizontalAdvance(text) + 8, 72)

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            on = self.isChecked()
            hover = self.underMouse()
            p.setFont(self._f_on if on else self._f_off)
            p.setPen(QPen(INK if (on or hover) else SUB))
            p.drawText(QRectF(self.rect()), Qt.AlignmentFlag.AlignCenter, self.text())
            if on:
                p.fillRect(QRectF(4, self.height() - 2, self.width() - 8, 2), COLOR_AMBER)
        finally:
            p.end()

    def enterEvent(self, event) -> None:  # type: ignore[override]
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        super().leaveEvent(event)
        self.update()


class _Brand(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._font = _font(FONT_EN, 12, QFont.Weight.DemiBold, 3.0)
        self.setFixedSize(QFontMetrics(self._font).horizontalAdvance("ARKCONTROLLER") + 40, 72)

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            cy = self.height() / 2.0
            p.fillRect(QRectF(0, cy - 4, 8, 8), COLOR_AMBER)
            p.setFont(self._font)
            p.setPen(QPen(INK))
            p.drawText(QRectF(18, 0, self.width() - 18, self.height()),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "ARKCONTROLLER")
        finally:
            p.end()


class _StatusDots(QWidget):
    """右上状态圆点 + 在线状态文字。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items: list[tuple[str, QColor]] = []
        self._font = _font(FONT_CN, 9)
        self.setFixedHeight(72)
        self.setMinimumWidth(300)

    def set_items(self, items: Sequence[tuple[str, QColor]]) -> None:
        new = list(items)
        if new != self._items:
            self._items = new
            self.update()

    def texts(self) -> list[str]:
        return [t for t, _c in self._items]

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            fm = QFontMetrics(self._font)
            x = float(self.width())
            cy = self.height() / 2.0
            p.setFont(self._font)
            for label, color in reversed(self._items):
                tw = fm.horizontalAdvance(label)
                x -= tw
                p.setPen(QPen(INK))
                p.drawText(QRectF(x, 0, tw + 2, self.height()), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
                x -= 12
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(color))
                p.drawEllipse(QPointF(x + 4, cy), 3.5, 3.5)
                x -= 22
        finally:
            p.end()


class _HeaderBar(QWidget):
    """顶部白色导航栏：品牌 + 水平 Tab + 状态圆点。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(72)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(40, 0, 40, 0)
        lay.setSpacing(0)
        self.brand = _Brand()
        lay.addWidget(self.brand)
        lay.addStretch(1)
        self.tabs: list[_TabButton] = []
        for i, (title, _en, _sub) in enumerate(PAGE_TITLES):
            btn = _TabButton(title)
            self.tabs.append(btn)
            lay.addWidget(btn)
            if i < len(PAGE_TITLES) - 1:
                lay.addSpacing(44)
        lay.addStretch(1)
        self.status = _StatusDots()
        lay.addWidget(self.status)

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        p = QPainter(self)
        try:
            p.fillRect(self.rect(), WHITE)
            p.setPen(QPen(LINE, 1.0))
            p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        finally:
            p.end()


class _Background(QWidget):
    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        p = QPainter(self)
        try:
            p.fillRect(self.rect(), BG)
        finally:
            p.end()


# ----------------------------------------------------------------------
# ① 总览：手柄遥测
# ----------------------------------------------------------------------
_TELEMETRY_KEYS: tuple[tuple[str, GamepadButton, ButtonGlyph], ...] = (
    ("A", GamepadButton.A, ButtonGlyph("face", "A")),
    ("B", GamepadButton.B, ButtonGlyph("face", "B")),
    ("X", GamepadButton.X, ButtonGlyph("face", "X")),
    ("Y", GamepadButton.Y, ButtonGlyph("face", "Y")),
    ("LB", GamepadButton.LB, ButtonGlyph("cap", "LB")),
    ("RB", GamepadButton.RB, ButtonGlyph("cap", "RB")),
    ("LT", GamepadButton.LT, ButtonGlyph("cap", "LT")),
    ("RT", GamepadButton.RT, ButtonGlyph("cap", "RT")),
    ("LS", GamepadButton.LS, ButtonGlyph("stick", "LS")),
    ("RS", GamepadButton.RS, ButtonGlyph("stick", "RS")),
    ("↑", GamepadButton.DPAD_UP, ButtonGlyph("dpad", "up")),
    ("↓", GamepadButton.DPAD_DOWN, ButtonGlyph("dpad", "down")),
    ("←", GamepadButton.DPAD_LEFT, ButtonGlyph("dpad", "left")),
    ("→", GamepadButton.DPAD_RIGHT, ButtonGlyph("dpad", "right")),
    ("BACK", GamepadButton.BACK, ButtonGlyph("back")),
    ("START", GamepadButton.START, ButtonGlyph("start")),
)


class GamepadTelemetryWidget(QWidget):
    """实时手柄遥测：双摇杆圆靶盘（死区虚线环）、LT / RT 细进度条与按键实时反馈阵列。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state: GamepadState = GamepadState()
        self._left_dz: float = 0.14
        self._right_dz: float = 0.16
        self._trigger_threshold: float = 0.45
        self._f_mono = _font(FONT_MONO, 8)
        self._f_key = _font(FONT_EN, 8, QFont.Weight.DemiBold)
        self._f_label = _font(FONT_EN, 8, QFont.Weight.DemiBold)
        self._f_cn = _font(FONT_CN, 9)
        self.setMinimumSize(420, 300)

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
        """返回当前按下的按键标签（供单测与状态展示）。"""
        return [label for label, btn, _g in _TELEMETRY_KEYS if self._is_pressed(btn)]

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
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            self._paint(p, float(self.width()), float(self.height()))
        finally:
            p.end()

    def _paint(self, p: QPainter, w: float, h: float) -> None:
        s = self._state
        keys_h = 64.0
        trig_h = 26.0
        area_h = h - keys_h - trig_h - 30.0
        r = max(40.0, min(area_h / 2.0, (w - 40.0) / 4.4))
        cy = r + 4.0
        for i, (code, x, y, dz) in enumerate((("LS", s.lx, s.ly, self._left_dz), ("RS", s.rx, s.ry, self._right_dz))):
            cx = w * (0.25 if i == 0 else 0.72)
            self._stick(p, cx, cy, r, x, y, dz)
            p.setFont(self._f_mono)
            p.setPen(QPen(SUB))
            p.drawText(QRectF(cx - r - 30, cy + r + 8, 2 * r + 60, 16), Qt.AlignmentFlag.AlignCenter,
                       f"{code}  {x:+.2f} / {y:+.2f}" if s.connected else f"{code}  —")
        ty = cy + r + 40.0
        half = w / 2.0
        for i, (code, v) in enumerate((("LT", s.lt), ("RT", s.rt))):
            v = max(0.0, min(1.0, v)) if s.connected else 0.0
            x0 = 0.0 if i == 0 else half + 10.0
            bar_w = half - 70.0
            p.setFont(self._f_label)
            p.setPen(QPen(INK))
            p.drawText(QRectF(x0, ty - 8, 26, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, code)
            p.fillRect(QRectF(x0 + 28, ty, bar_w, 1), LINE)
            p.fillRect(QRectF(x0 + 28, ty - 0.5, bar_w * v, 2), COLOR_AMBER)
            tx = x0 + 28 + bar_w * self._trigger_threshold
            p.fillRect(QRectF(tx, ty - 4, 1, 9), EDGE)
            p.setFont(self._f_mono)
            p.setPen(QPen(SUB))
            p.drawText(QRectF(x0 + 32 + bar_w, ty - 8, 34, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       f"{int(round(v * 100))}")
        # 按键实时反馈阵列（按下时线框转琥珀）
        ky = h - keys_h + 14.0
        n = len(_TELEMETRY_KEYS)
        per_row = 8
        cell = w / per_row
        for idx, (label, btn, glyph) in enumerate(_TELEMETRY_KEYS):
            row, col = divmod(idx, per_row)
            gw = measure_button_spec([glyph], self._f_key)
            x = col * cell + (cell - gw) / 2.0
            y = ky + row * 32.0
            pressed = self._is_pressed(btn)
            paint_button_glyph(p, glyph, x, y, self._f_key, highlight=pressed)
            if pressed:
                p.fillRect(QRectF(x, y + 13.0, gw, 2.0), COLOR_AMBER)
        del n
        if not s.connected:
            p.setFont(self._f_cn)
            p.setPen(QPen(SUB))
            p.drawText(QRectF(0, cy - 10, w, 20), Qt.AlignmentFlag.AlignCenter, "未检测到 Xbox 手柄")

    def _stick(self, p: QPainter, cx: float, cy: float, r: float, x: float, y: float, dz: float) -> None:
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor("#D9D9D9"), 1.0))
        p.drawEllipse(QPointF(cx, cy), r, r)
        p.setPen(QPen(QColor("#EFEFEF"), 1.0))
        p.drawLine(QPointF(cx - r, cy), QPointF(cx + r, cy))
        p.drawLine(QPointF(cx, cy - r), QPointF(cx, cy + r))
        p.setPen(QPen(QColor("#CFCFCF"), 1.0, Qt.PenStyle.DashLine))
        dz_r = max(2.0, r * max(0.0, min(0.95, dz)))
        p.drawEllipse(QPointF(cx, cy), dz_r, dz_r)
        if not self._state.connected:
            return
        px = cx + max(-1.0, min(1.0, x)) * r
        py = cy - max(-1.0, min(1.0, y)) * r
        active = math.hypot(x, y) > dz
        col = COLOR_AMBER if active else EDGE
        p.setPen(QPen(col, 1.2))
        p.drawLine(QPointF(cx, cy), QPointF(px, py))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(col))
        p.drawEllipse(QPointF(px, py), 5.0, 5.0)


class _KeyValueGrid(QWidget):
    """键值对行：左侧冷灰标签 + 右侧右对齐深色粗体数值。"""

    def __init__(self, keys: Sequence[tuple[str, str]], *, value_align_right: bool = True,
                 key_width: int = 110, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)
        self._values: dict[str, QLabel] = {}
        for row, (key, label) in enumerate(keys):
            k = QLabel(label)
            k.setObjectName("KvKey")
            k.setMinimumWidth(key_width)
            v = QLabel("—")
            v.setObjectName("KvValue")
            v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            v.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            if value_align_right:
                v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            else:
                v.setWordWrap(True)
                v.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
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
        style = f"color: {_hex(color if color is not None else INK)};"
        if label.styleSheet() != style:
            label.setStyleSheet(style)

    def value(self, key: str) -> str:
        label = self._values.get(key)
        return label.text() if label is not None else ""


class _ParamView(QWidget):
    """参数速览：小号灰标签 + 大号细体数值 + 琥珀细进度线。"""

    def __init__(self, spec: TuningSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.value = spec.minimum
        self._f_label = _font(FONT_CN, 9)
        self._f_value = _font(FONT_EN, 17, QFont.Weight.Light)
        self.setFixedHeight(76)

    def set_value(self, v: float) -> None:
        self.value = float(v)
        self.update()

    def text(self) -> str:
        return _format_value(self.spec, self.value)

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            w = float(self.width())
            p.setFont(self._f_label)
            p.setPen(QPen(SUB))
            p.drawText(QRectF(0, 0, w, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.spec.title)
            p.setFont(self._f_value)
            p.setPen(QPen(INK))
            p.drawText(QRectF(0, 18, w, 36), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text())
            frac = (self.value - self.spec.minimum) / max(1e-6, self.spec.maximum - self.spec.minimum)
            p.fillRect(QRectF(0, 60, w, 1), LINE)
            p.fillRect(QRectF(0, 59, w * max(0.0, min(1.0, frac)), 3), COLOR_AMBER)
        finally:
            p.end()


# ----------------------------------------------------------------------
# ② 调校：滑块行与曲线
# ----------------------------------------------------------------------
class TuningSliderRow(QWidget):
    """参数滑块行：中文标题 + 英文键名 + 琥珀细滑轨 + 大号细体数值 + 说明。"""

    value_changed = Signal(str, float)

    def __init__(self, spec: TuningSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 4, 0, 4)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(0)

        title = QLabel(spec.title)
        title.setObjectName("TuneTitle")
        key = QLabel(spec.key)
        key.setObjectName("TuneKey")
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        box.addWidget(title)
        box.addWidget(key)
        wrap = QWidget()
        wrap.setLayout(box)
        wrap.setFixedWidth(150)

        self._steps = int(round((spec.maximum - spec.minimum) / spec.step))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._steps)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(max(1, self._steps // 10))
        self.slider.valueChanged.connect(self._on_slider)

        self.value_label = QLabel()
        self.value_label.setObjectName("TuneValue")
        self.value_label.setFixedWidth(120)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        desc = QLabel(spec.desc)
        desc.setObjectName("TuneDesc")

        grid.addWidget(wrap, 0, 0, 2, 1)
        grid.addWidget(self.slider, 0, 1)
        grid.addWidget(self.value_label, 0, 2, 2, 1)
        grid.addWidget(desc, 1, 1)
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
        self.value_label.setText(_format_value(self.spec, self.value()))

    def _on_slider(self, _pos: int) -> None:
        self._refresh_label()
        self.value_changed.emit(self.spec.key, self.value())


class _CurvePreview(QWidget):
    """自由光标摇杆响应曲线（白底极简：浅灰网格 + 琥珀曲线 + 浅灰死区）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.deadzone = 0.14
        self.gamma = 2.2
        self.max_speed = 1400.0
        self._f_mono = _font(FONT_MONO, 8)
        self._f_cn = _font(FONT_CN, 8)
        self.setMinimumHeight(240)

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
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            w, h = float(self.width()), float(self.height())
            plot = QRectF(46.0, 10.0, w - 60.0, h - 40.0)
            p.fillRect(QRectF(plot.left(), plot.top(), plot.width() * self.deadzone, plot.height()), QColor("#F2F2F2"))
            p.setPen(QPen(QColor("#F0F0F0"), 1.0))
            for k in range(1, 5):
                gx = plot.left() + plot.width() * k / 5.0
                gy = plot.top() + plot.height() * k / 5.0
                p.drawLine(QPointF(gx, plot.top()), QPointF(gx, plot.bottom()))
                p.drawLine(QPointF(plot.left(), gy), QPointF(plot.right(), gy))
            p.setPen(QPen(LINE, 1.0))
            p.drawLine(plot.bottomLeft(), plot.bottomRight())
            p.drawLine(plot.bottomLeft(), plot.topLeft())
            p.setPen(QPen(EDGE, 1.0, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(plot.left() + plot.width() * self.deadzone, plot.bottom()), plot.topRight())
            pts = [QPointF(plot.left() + plot.width() * i / 100.0,
                           plot.bottom() - plot.height() * self.response(i / 100.0)) for i in range(101)]
            p.setPen(QPen(COLOR_AMBER, 2.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(QPolygonF(pts))
            mx = plot.left() + plot.width() * 0.5
            my = plot.bottom() - plot.height() * self.response(0.5)
            p.setPen(QPen(COLOR_AMBER, 1.5))
            p.setBrush(QBrush(WHITE))
            p.drawEllipse(QPointF(mx, my), 4.0, 4.0)
            p.setFont(self._f_mono)
            p.setPen(QPen(SUB))
            p.drawText(QRectF(0, plot.top() - 6, 40, 14), Qt.AlignmentFlag.AlignRight, f"{int(self.max_speed)}")
            p.drawText(QRectF(0, plot.bottom() - 8, 40, 14), Qt.AlignmentFlag.AlignRight, "0")
            p.drawText(QRectF(plot.left(), plot.bottom() + 8, plot.width(), 14), Qt.AlignmentFlag.AlignRight, "推杆 100%")
            p.setPen(QPen(AMBER_TEXT))
            label = f"50% → {int(self.response(0.5) * self.max_speed)} px/s"
            if self.gamma >= 1.0:
                # 下凸曲线：圆点左上方为空白区
                lrect = QRectF(mx - 166, my - 24, 156, 16)
                align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            else:
                # 上凸曲线：圆点右下方为空白区
                lrect = QRectF(mx + 10, my + 8, 156, 16)
                align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            lrect.moveTop(max(plot.top(), min(plot.bottom() - lrect.height(), lrect.top())))
            tw_ = QFontMetrics(self._f_mono).horizontalAdvance(label) + 8
            bg_rect = QRectF(lrect.right() - tw_ if self.gamma >= 1.0 else lrect.left() - 4, lrect.top(), tw_, lrect.height())
            p.fillRect(bg_rect, BG)
            p.drawText(lrect, align, label)
            p.setFont(self._f_cn)
            p.setPen(QPen(SUB))
            p.drawText(QRectF(plot.left(), plot.bottom() + 8, 140, 14), Qt.AlignmentFlag.AlignLeft,
                       f"死区 {self.deadzone:.2f}")
        finally:
            p.end()


# ----------------------------------------------------------------------
# ③ 键位：操作简报
# ----------------------------------------------------------------------
KEYMAP_BRIEFS: tuple[tuple[str, str, str, tuple[tuple[str, str], ...]], ...] = (
    ("GLOBAL", "全局热键", "全阶段生效", (
        ("LS", "推杆：局外移动鼠标 / 局内跳格"),
        ("按下LS", "吸附导航 ↔ 自由光标"),
        ("RS", "推杆：鼠标滚轮滚动"),
        ("BACK", "循环手动切换场景层（锁层）"),
        ("BACK+START", "打开可视化校准向导"),
        ("LT+BACK", "暂停 / 恢复手柄映射"),
    )),
    ("FREE CURSOR", "局外模拟鼠标", "UNKNOWN_FREE_CURSOR", (
        ("LS", "平滑移动虚拟鼠标光标"),
        ("A", "鼠标左键单击"),
        ("RT+A", "鼠标左键双击"),
        ("B", "发送 Esc 返回上一级"),
        ("RS", "滚动编队 / 列表 / 结算页"),
    )),
    ("SHOP", "休整期商店运营", "BATTLE_PREP", (
        ("双击A", "双击 A 买牌（首下仅预选）"),
        ("RT+A", "查看商店干员详情"),
        ("Y", "升本（官方 G 键）"),
        ("X", "刷新商店（官方 D 键）"),
        ("RS", "按下冻结 / 解冻商店（官方 S 键）"),
        ("LB/RB", "切换四大运营区"),
        ("长按START", "准备开战（官方 C 键）"),
    )),
    ("DEPLOY", "棋盘部署与转向", "4x7 透视棋盘", (
        ("A", "整备区抓取干员 → 棋盘落位"),
        ("LS/D-PAD", "选择四向朝向"),
        ("双击A", "同格回环拖拽，原地转向"),
        ("X", "快速出售（官方 X 键）"),
        ("Y", "拖拽赠予联机队友"),
        ("B", "取消抓取 / 取消定朝向"),
    )),
    ("STRATEGY", "右侧策略头像", "FocusZone.STRATEGY", (
        ("RT+A", "一键点击释放策略"),
        ("A>A", "锁定策略源后原地释放"),
        ("X", "查看策略说明"),
        ("A", "抓取 → 拖拽至干员 / 商店卡槽"),
        ("B", "无点击安全取消抓取"),
    )),
    ("CHOICE", "2 / 3 / 6 选一决策", "MODAL_CHOICE", (
        ("LS/D-PAD", "选卡导航（1x2 / 1x3 / 2x3）"),
        ("A", "第 1 下：选中卡片弹出白条"),
        ("A>A", "第 2 下：点击白条确认选择"),
        ("RT+A", "一键选中并确认"),
        ("B", "退回卡片视图重新对比"),
        ("按住LT", "收起弹窗临时看战场"),
    )),
    ("CONFIRM", "回合 / ESC 退出确认窗", "MODAL_CONFIRM", (
        ("左/右", "切换「取消 / 确认」"),
        ("A", "点击当前聚焦按钮"),
        ("B", "直接点击「取消」"),
        ("START/RT+A", "直接点击「确认」"),
        ("X", "勾选「本局不再提示」"),
    )),
    ("EMOTE", "左下 2x3 交流表情", "FocusZone.EMOTE_PANEL", (
        ("LT+下", "打开 / 关闭表情面板"),
        ("LS/D-PAD", "在 2x3 表情间切换"),
        ("RS/LB/RB", "滚轮翻页"),
        ("A", "发送当前高亮表情"),
        ("B", "关闭面板回归原分区"),
    )),
    ("LT CHORD", "LT 联机组合键", "休整期 / 作战期", (
        ("LT+左", "盟约容器详情面板"),
        ("LT+上", "敌方波次与出怪路线（W）"),
        ("LT+右", "循环切换联机队友"),
        ("LT+A", "直达队友 1 号位（B/X/Y→2/3/4）"),
        ("LT+LB", "展开 / 收起底部商店"),
        ("LT+RB", "一键撤退干员（Q）"),
        ("LT+START", "暂停 / 设置菜单"),
    )),
)


class _KeyCapStrip(QWidget):
    """以 HUD 共用矢量绘制器渲染按钮图示序列。"""

    def __init__(self, spec: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.combo = spec
        self._font = _font(FONT_EN, 8, QFont.Weight.DemiBold)
        self.glyphs = parse_button_spec(spec)
        self.setFixedHeight(26)
        self.setFixedWidth(int(self.natural_width()) + 4)

    def natural_width(self) -> float:
        return measure_button_spec(self.glyphs, self._font)

    def paintEvent(self, _event: QPaintEvent) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            paint_button_spec(p, self.glyphs, 1.0, self.height() / 2.0, self._font)
        finally:
            p.end()


class KeymapBriefCard(Cell):
    """操作简报单元：英文小标 + 中文标题 + 场景代号 + 按钮图示行。"""

    def __init__(self, code: str, title: str, subtitle: str, rows: Sequence[tuple[str, str]],
                 *, right: bool, bottom: bool, parent: QWidget | None = None) -> None:
        super().__init__(right=right, bottom=bottom, margins=(24, 20, 24, 20), parent=parent)
        self.rows = tuple(rows)
        self.body.addWidget(SectionLabel(code, title))
        sub = QLabel(subtitle)
        sub.setObjectName("BriefSub")
        self.body.addWidget(sub)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        strips = [_KeyCapStrip(spec) for spec, _ in rows]
        strip_w = max((s.width() for s in strips), default=40)
        for i, ((_spec, desc), strip) in enumerate(zip(rows, strips)):
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
    color: {_hex(INK)};
    font-family: "{FONT_CN}";
    font-size: 9pt;
}}
QLabel {{ background: transparent; }}
QLabel#KvKey {{ color: {_hex(SUB)}; font-size: 9pt; }}
QLabel#KvValue {{ font-size: 10pt; font-weight: bold; }}
QLabel#PageTitle {{ font-size: 22pt; font-weight: 300; color: {_hex(INK)}; }}
QLabel#PageSub {{ font-size: 9pt; color: {_hex(SUB)}; }}
QLabel#BigValue {{ font-family: "{FONT_EN}"; font-size: 28pt; font-weight: 300; color: {_hex(INK)}; }}
QLabel#BigSub {{ font-size: 9pt; color: {_hex(SUB)}; }}
QLabel#TuneTitle {{ font-size: 10pt; font-weight: bold; }}
QLabel#TuneKey {{ font-family: "{FONT_MONO}"; font-size: 8pt; color: {_hex(SUB)}; }}
QLabel#TuneValue {{ font-family: "{FONT_EN}"; font-size: 15pt; font-weight: 300; }}
QLabel#TuneDesc {{ font-size: 8pt; color: {_hex(SUB)}; }}
QLabel#BriefSub {{ font-family: "{FONT_MONO}"; font-size: 8pt; color: {_hex(SUB)}; }}
QLabel#BriefDesc {{ font-size: 9pt; color: #333333; }}
QLabel#Note {{ font-size: 8pt; color: {_hex(SUB)}; }}
QLabel#Feedback {{ font-size: 9pt; }}
QLabel#Footer {{ font-size: 8pt; color: {_hex(SUB)}; }}
QSlider {{ min-height: 22px; background: transparent; }}
QSlider::groove:horizontal {{ height: 2px; background: {_hex(LINE)}; border: none; }}
QSlider::sub-page:horizontal {{ background: {_hex(COLOR_AMBER)}; }}
QSlider::add-page:horizontal {{ background: {_hex(LINE)}; }}
QSlider::handle:horizontal {{
    background: #FFFFFF;
    border: 1px solid {_hex(EDGE)};
    width: 10px;
    margin: -8px 0;
}}
QSlider::handle:horizontal:hover {{ border: 1px solid {_hex(COLOR_AMBER)}; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 6px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #D9D9D9; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {_hex(EDGE)}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
QToolTip {{
    background-color: #FFFFFF;
    color: {_hex(INK)};
    border: 1px solid {_hex(EDGE)};
    padding: 4px;
}}
"""


class PRTSDashboardWindow(QMainWindow):
    """罗德岛 PRTS 桌面主控台窗口（方案 B v2；普通层级，不置顶，关闭即隐藏至托盘）。"""

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
        self.setWindowTitle("ArkController — PRTS 主控台")
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
        self._config_dir_path: str = ""

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(TELEMETRY_INTERVAL_MS)
        self._refresh_timer.timeout.connect(self._on_refresh_tick)

        root = _Background()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self._header = _HeaderBar()
        root_layout.addWidget(self._header)

        body = QVBoxLayout()
        body.setContentsMargins(40, 22, 40, 14)
        body.setSpacing(0)
        root_layout.addLayout(body, 1)

        # 标题行 + 右上统一白底细边按钮
        title_row = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        self._page_title = QLabel()
        self._page_title.setObjectName("PageTitle")
        self._page_sub = QLabel()
        self._page_sub.setObjectName("PageSub")
        title_box.addWidget(self._page_title)
        title_box.addWidget(self._page_sub)
        title_row.addLayout(title_box)
        title_row.addStretch(1)
        self.btn_reload = LineButton("重载配置")
        self.btn_reload.setFixedWidth(118)
        self.btn_reload.clicked.connect(self.reload_config_requested.emit)
        self.btn_calibrate = LineButton("启动校准向导  →")
        self.btn_calibrate.setFixedWidth(200)
        self.btn_calibrate.clicked.connect(self.open_calibration_requested.emit)
        title_row.addWidget(self.btn_reload, 0, Qt.AlignmentFlag.AlignTop)
        title_row.addSpacing(12)
        title_row.addWidget(self.btn_calibrate, 0, Qt.AlignmentFlag.AlignTop)
        body.addLayout(title_row)
        body.addSpacing(20)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_overview_page())
        self._stack.addWidget(self._build_tuning_page())
        self._stack.addWidget(self._build_keymap_page())
        self._stack.addWidget(self._build_system_page())
        body.addWidget(self._stack, 1)

        # 页脚：版本 + 隐藏至托盘 / 退出程序
        foot = QHBoxLayout()
        foot.setContentsMargins(0, 10, 0, 0)
        ver = QLabel(f"v{APP_VERSION}  ·  卫戍协议手柄助手")
        ver.setObjectName("Footer")
        foot.addWidget(ver)
        foot.addStretch(1)
        self._btn_hide = LineButton("隐藏至托盘")
        self._btn_hide.setFixedSize(108, 30)
        self._btn_hide.clicked.connect(self.hide_to_tray)
        self._btn_quit = LineButton("退出程序", danger=True)
        self._btn_quit.setFixedSize(96, 30)
        self._btn_quit.clicked.connect(self.quit_requested.emit)
        foot.addWidget(self._btn_hide)
        foot.addSpacing(8)
        foot.addWidget(self._btn_quit)
        body.addLayout(foot)

        self.setCentralWidget(root)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        self._nav_buttons = self._header.tabs
        for i, btn in enumerate(self._nav_buttons):
            self._nav_group.addButton(btn, i)
        self._nav_group.idClicked.connect(self.set_current_page)

        self.set_current_page(self.PAGE_OVERVIEW)
        self._apply_status(self._status)
        self._sync_param_views(self._keymap)

    # ------------------------------------------------------------------
    # 构建：① 总览
    # ------------------------------------------------------------------
    def _build_overview_page(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        tele = Cell(right=True, bottom=True, top=True, margins=(0, 22, 32, 20))
        tele.body.addWidget(SectionLabel("CONTROLLER", "手柄输入"))
        self.telemetry = GamepadTelemetryWidget()
        tele.body.addWidget(self.telemetry, 1)
        grid.addWidget(tele, 0, 0)

        win = Cell(right=True, bottom=True, top=True, margins=(32, 22, 32, 20))
        win.body.addWidget(SectionLabel("WINDOW", "游戏窗口"))
        self._big_resolution = QLabel("—")
        self._big_resolution.setObjectName("BigValue")
        self._big_sub = QLabel("等待《明日方舟》窗口接入")
        self._big_sub.setObjectName("BigSub")
        win.body.addWidget(self._big_resolution)
        win.body.addWidget(self._big_sub)
        win.body.addSpacing(8)
        self._status_grid = _KeyValueGrid(
            (
                ("window", "游戏窗口"),
                ("viewport", "有效视口"),
                ("scene", "识别场景"),
                ("shop", "商店"),
                ("focus", "当前焦点"),
                ("mapping", "映射状态"),
            ),
            key_width=72,
        )
        win.body.addWidget(self._status_grid)
        win.body.addStretch(1)
        grid.addWidget(win, 0, 1)

        params = Cell(top=True, margins=(32, 22, 0, 20))
        params.body.addWidget(SectionLabel("HANDLING", "手感参数"))
        self._param_views: dict[str, _ParamView] = {}
        for spec in TUNING_SPECS:
            if spec.key in ("scroll_speed", "trigger_threshold"):
                continue
            view = _ParamView(spec)
            self._param_views[spec.key] = view
            params.body.addWidget(view)
        params.body.addStretch(1)
        grid.addWidget(params, 0, 2, 2, 1)

        sw = Cell(right=True, margins=(0, 22, 32, 10))
        sw.body.addWidget(SectionLabel("SWITCHES", "快捷开关"))
        toggles = QGridLayout()
        toggles.setHorizontalSpacing(24)
        toggles.setVerticalSpacing(4)
        self.toggle_pause = CheckRow("暂停手柄映射", "LT + Back")
        self.toggle_overlay = CheckRow("局内透明 HUD", "鼠标穿透，不挡游戏")
        self.toggle_native = CheckRow("PC 原生快捷键", "G / D / S / A / C / W / X / Q")
        self.toggle_dpad = CheckRow("左摇杆代替十字键", "局内吸附跳格，按 LS 切自由光标")
        self.toggle_auto_shop = CheckRow("跨区自动开关商店", "切区自动展开 / 收起")
        self.toggle_vibration = CheckRow("震动反馈", "落位与确认时轻震")
        self.toggle_retreat = CheckRow("允许 LT+RB 撤退", "官方 Q 键一键撤退")
        self.toggle_hide_capture = CheckRow("截屏/共享时隐藏 HUD", "关闭后截图、录屏与屏幕共享可见悬浮窗")
        self.toggle_pause.clicked.connect(self._on_pause_clicked)
        self.toggle_overlay.toggled.connect(self.overlay_toggled.emit)
        self.toggle_native.toggled.connect(self.native_shortcuts_toggled.emit)
        self.toggle_dpad.toggled.connect(self.left_stick_dpad_toggled.emit)
        self.toggle_auto_shop.toggled.connect(self._on_auto_shop_toggled)
        self.toggle_vibration.toggled.connect(self._on_vibration_toggled)
        self.toggle_retreat.toggled.connect(self._on_retreat_toggled)
        self.toggle_hide_capture.toggled.connect(self._on_hide_capture_toggled)
        for i, t in enumerate((
            self.toggle_pause, self.toggle_overlay, self.toggle_native, self.toggle_dpad,
            self.toggle_auto_shop, self.toggle_vibration, self.toggle_retreat, self.toggle_hide_capture,
        )):
            toggles.addWidget(t, i // 3, i % 3)
        sw.body.addLayout(toggles)
        sw.body.addStretch(1)
        grid.addWidget(sw, 1, 0, 1, 2)

        grid.setColumnStretch(0, 47)
        grid.setColumnStretch(1, 36)
        grid.setColumnStretch(2, 30)
        grid.setRowStretch(0, 3)
        grid.setRowStretch(1, 2)
        return page

    # ------------------------------------------------------------------
    # 构建：② 调校
    # ------------------------------------------------------------------
    def _build_tuning_page(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        left = Cell(right=True, top=True, margins=(0, 22, 32, 10))
        left.body.addWidget(SectionLabel("HANDLING", "摇杆与光标手感"))
        self.tuning_rows: dict[str, TuningSliderRow] = {}
        for spec in TUNING_SPECS:
            tr = TuningSliderRow(spec)
            tr.value_changed.connect(self._on_tuning_changed)
            self.tuning_rows[spec.key] = tr
            left.body.addWidget(tr)
        left.body.addStretch(1)
        row.addWidget(left, 3)

        right = Cell(top=True, margins=(32, 22, 0, 10))
        right.body.addWidget(SectionLabel("RESPONSE", "自由光标响应曲线"))
        self.curve_preview = _CurvePreview()
        right.body.addWidget(self.curve_preview, 1)
        note = QLabel("横轴为摇杆推动幅度，纵轴为光标速度；浅灰区域为死区，圆点为半推杆速度。")
        note.setObjectName("Note")
        note.setWordWrap(True)
        right.body.addWidget(note)
        right.body.addSpacing(12)
        btns = QHBoxLayout()
        btns.setSpacing(12)
        self.btn_apply_tuning = LineButton("应用并持久化保存")
        self.btn_apply_tuning.clicked.connect(self._on_apply_tuning)
        self.btn_reset_tuning = LineButton("恢复默认手感")
        self.btn_reset_tuning.clicked.connect(self._on_reset_tuning)
        btns.addWidget(self.btn_apply_tuning, 3)
        btns.addWidget(self.btn_reset_tuning, 2)
        right.body.addLayout(btns)
        self.tuning_feedback = QLabel("调整滑块后点击「应用并持久化保存」，参数将写入 user_override.json 并立即热刷新。")
        self.tuning_feedback.setObjectName("Feedback")
        self.tuning_feedback.setWordWrap(True)
        self.tuning_feedback.setStyleSheet(f"color: {_hex(SUB)};")
        right.body.addWidget(self.tuning_feedback)
        row.addWidget(right, 2)
        self._load_tuning_from_keymap(self._keymap)
        return page

    # ------------------------------------------------------------------
    # 构建：③ 键位
    # ------------------------------------------------------------------
    def _build_keymap_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setContentsMargins(0, 0, 8, 0)
        grid.setSpacing(0)
        self.keymap_cards: list[KeymapBriefCard] = []
        rows_total = (len(KEYMAP_BRIEFS) + 2) // 3
        for i, (code, title, sub, rows) in enumerate(KEYMAP_BRIEFS):
            r, c = divmod(i, 3)
            card = KeymapBriefCard(code, title, sub, rows, right=(c < 2), bottom=(r < rows_total - 1))
            self.keymap_cards.append(card)
            grid.addWidget(card, r, c)
        for c in range(3):
            grid.setColumnStretch(c, 1)
        wrap = Cell(top=True, margins=(0, 0, 0, 0))
        wrap.body.addWidget(inner)
        scroll.setWidget(wrap)
        return scroll

    # ------------------------------------------------------------------
    # 构建：④ 系统
    # ------------------------------------------------------------------
    def _build_system_page(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        ver = Cell(right=True, bottom=True, top=True, margins=(0, 22, 32, 20))
        ver.body.addWidget(SectionLabel("ARCHIVE", "版本与权限"))
        self._sys_grid = _KeyValueGrid(
            (
                ("version", "ArkController 版本"),
                ("admin", "管理员权限（UAC）"),
                ("python", "Python 运行时"),
                ("qt", "PySide6 / Qt"),
                ("platform", "操作系统"),
            ),
            key_width=130,
        )
        ver.body.addWidget(self._sys_grid)
        ver.body.addStretch(1)
        grid.addWidget(ver, 0, 0)

        cfg = Cell(right=True, bottom=True, top=True, margins=(32, 22, 32, 20))
        cfg.body.addWidget(SectionLabel("CONFIG", "配置档案"))
        self._cfg_grid = _KeyValueGrid(
            (
                ("config_dir", "配置目录"),
                ("override", "用户覆盖"),
                ("templates", "视觉模板"),
                ("board", "棋盘网格"),
            ),
            key_width=72,
        )
        cfg.body.addWidget(self._cfg_grid)
        cfg.body.addStretch(1)
        grid.addWidget(cfg, 0, 1)

        ops = Cell(top=True, bottom=True, margins=(32, 22, 0, 20))
        ops.body.addWidget(SectionLabel("OPERATIONS", "操作入口"))
        b1 = LineButton("启动校准向导  →")
        b1.clicked.connect(self.open_calibration_requested.emit)
        b2 = LineButton("热重载配置与模板")
        b2.clicked.connect(self.reload_config_requested.emit)
        b3 = LineButton("打开配置文件目录")
        b3.clicked.connect(self._open_config_dir)
        b4 = LineButton("完全退出 ArkController", danger=True)
        b4.clicked.connect(self.quit_requested.emit)
        for b in (b1, b2, b3, b4):
            ops.body.addWidget(b)
        ops.body.addStretch(1)
        grid.addWidget(ops, 0, 2)

        mat = Cell(right=True, margins=(0, 22, 32, 20))
        mat.body.addWidget(SectionLabel("VIEW MATRIX", "多视图缩放补偿矩阵"))
        self._matrix_grid = _KeyValueGrid(
            tuple((vs.value, VIEW_STATE_LABELS.get(vs.value, vs.value)) for vs in LayoutViewState),
            value_align_right=False,
            key_width=72,
        )
        mat.body.addWidget(self._matrix_grid)
        mnote = QLabel("补偿矩阵由 SIFT 实机标定或可视化校准向导生成，用于商店展开与角色特写时的镜头缩放坐标修正。")
        mnote.setObjectName("Note")
        mnote.setWordWrap(True)
        mat.body.addWidget(mnote)
        mat.body.addStretch(1)
        grid.addWidget(mat, 1, 0)

        sec = Cell(margins=(32, 22, 0, 20))
        sec.body.addWidget(SectionLabel("PROTOCOL", "零注入安全协议"))
        self._sec_grid = _KeyValueGrid(
            (
                ("input", "输入合成"),
                ("memory", "游戏内存"),
                ("hud", "HUD 穿透"),
                ("capture", "截屏隔离"),
                ("frontend", "主控台层级"),
            ),
            value_align_right=False,
            key_width=72,
        )
        self._sec_grid.set_value("input", "Win32 SendInput 硬件扫描码 / 鼠标合成")
        self._sec_grid.set_value("memory", "零读取 · 零写入 · 零 DLL 注入", COLOR_GREEN)
        self._sec_grid.set_value("hud", "WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_LAYERED | WS_EX_TOPMOST")
        self._sec_grid.set_value("frontend", "普通层级（不置顶）· 关闭即隐藏至托盘 · 隐藏时零遥测开销")
        self._set_capture_mode_text(True)
        sec.body.addWidget(self._sec_grid)
        sec.body.addStretch(1)
        grid.addWidget(sec, 1, 1, 1, 2)

        grid.setColumnStretch(0, 34)
        grid.setColumnStretch(1, 40)
        grid.setColumnStretch(2, 26)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        self._sys_grid.set_value("version", f"v{APP_VERSION}")
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
        title, _en, sub = PAGE_TITLES[index]
        self._page_title.setText(title)
        self._page_sub.setText(sub)

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

    def header_status_texts(self) -> list[str]:
        """返回右上状态文字（供单测）。"""
        return self._header.status.texts()

    def _apply_status(self, s: DashboardRuntimeStatus) -> None:
        g = self._status_grid
        if s.window_found:
            g.set_value("window", f"{s.window_title or '明日方舟'} · {'前台' if s.is_foreground else '后台'}")
        else:
            g.set_value("window", "未检测到《明日方舟》", SUB)
        if s.active_rect is not None:
            al, at, aw, ah = s.active_rect
            g.set_value("viewport", f"{aw} × {ah}  @ ({al}, {at})")
            self._big_resolution.setText(f"{aw}×{ah}")
            self._big_sub.setText(f"16:9 有效视口 · {'前台激活' if s.is_foreground else '后台挂起（自动静默）'}")
        else:
            g.set_value("viewport", "—", SUB)
            self._big_resolution.setText("—")
            self._big_sub.setText("等待《明日方舟》窗口接入")
        g.set_value("scene", s.scene_name + ("  · 手动锁层" if s.scene_locked_manual else ""),
                    AMBER_TEXT if s.scene_locked_manual else INK)
        if s.shop_slot_count:
            shop = f"{'展开' if s.shop_open else '收起'} · {s.shop_slot_count} 槽"
        else:
            shop = "展开" if s.shop_open else "收起"
        g.set_value("shop", shop)
        focus = s.focus_label or "—"
        if s.deploy_state_name and s.deploy_state_name != "空闲":
            focus = f"{focus} · {s.deploy_state_name}"
        g.set_value("focus", focus)
        g.set_value("mapping", "已暂停" if s.paused else "运行中", COLOR_RED if s.paused else COLOR_GREEN)
        self._set_paused_ui(s.paused)
        self._header.status.set_items(
            [
                ("手柄在线" if s.gamepad_connected else "手柄未连接", COLOR_AMBER if s.gamepad_connected else EDGE),
                ((f"明日方舟 · {'前台' if s.is_foreground else '后台'}") if s.window_found else "游戏未接入",
                 COLOR_AMBER if s.window_found else EDGE),
                ("映射已暂停" if s.paused else "映射运行中", COLOR_RED if s.paused else COLOR_AMBER),
            ]
        )

    def _set_paused_ui(self, paused: bool) -> None:
        self._paused = bool(paused)
        self.toggle_pause.set_checked_silently(self._paused)

    def _set_capture_mode_text(self, hide: bool) -> None:
        if hide:
            self._sec_grid.set_value("capture", "WDA_EXCLUDEFROMCAPTURE (默认隐藏防自污染)")
        else:
            self._sec_grid.set_value("capture", "WDA_NONE (截图与屏幕共享可见)", AMBER_TEXT)

    def capture_mode_text(self) -> str:
        return self._sec_grid.value("capture")

    def _sync_param_views(self, keymap: KeymapConfig) -> None:
        for key, view in self._param_views.items():
            view.set_value(float(getattr(keymap, key)))

    def sync_settings(
        self,
        *,
        keymap: KeymapConfig,
        overlay_enabled: bool,
        paused: bool,
        refresh_tuning: bool = False,
        hide_hud_from_capture: bool | None = None,
    ) -> None:
        """从主控运行态反向同步开关、遥测标定与调校滑块（不触发回环信号）。"""
        if hide_hud_from_capture is None:
            hide_hud_from_capture = bool(getattr(keymap, "hide_hud_from_capture", True))
        self.toggle_hide_capture.set_checked_silently(hide_hud_from_capture)
        self._set_capture_mode_text(bool(hide_hud_from_capture))
        self._keymap = keymap
        self.toggle_overlay.set_checked_silently(overlay_enabled)
        self.toggle_native.set_checked_silently(keymap.use_native_keyboard_shortcuts)
        self.toggle_dpad.set_checked_silently(keymap.left_stick_mode == "dpad")
        self.toggle_auto_shop.set_checked_silently(keymap.auto_toggle_shop)
        self.toggle_vibration.set_checked_silently(keymap.vibration_enabled)
        self.toggle_retreat.set_checked_silently(keymap.allow_retreat_chord)
        self._set_paused_ui(paused)
        self.telemetry.set_calibration(keymap.left_deadzone, keymap.right_deadzone, keymap.trigger_threshold)
        self._sync_param_views(keymap)
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
        """刷新「系统」页档案信息。"""
        if version is not None:
            self._sys_grid.set_value("version", f"v{version}")
        if is_admin is not None:
            self._sys_grid.set_value(
                "admin",
                "已提权（可操控官方 PC 客户端）" if is_admin else "标准权限（官方 PC 端需管理员运行）",
                COLOR_GREEN if is_admin else AMBER_TEXT,
            )
        if config_dir is not None:
            self._config_dir_path = config_dir
            self._cfg_grid.set_value("config_dir", config_dir)
        if override_path is not None:
            self._cfg_grid.set_value("override", override_path)
        if template_count is not None:
            self._cfg_grid.set_value("templates", f"{int(template_count)} 个场景模板", INK if template_count else SUB)
        if layout is not None:
            q = layout.board_quad
            self._cfg_grid.set_value("board", f"{q.rows} 行 × {q.cols} 列 · 4 点单应性透视")
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
        self.tuning_feedback.setStyleSheet(f"color: {_hex(AMBER_TEXT)};")

    def _on_apply_tuning(self) -> None:
        self.tuning_apply_requested.emit(self.tuning_values())

    def _on_reset_tuning(self) -> None:
        self._load_tuning_from_keymap(KeymapConfig())
        self.tuning_apply_requested.emit(self.tuning_values())

    def _on_auto_shop_toggled(self, checked: bool) -> None:
        self.keymap_flag_toggled.emit("auto_toggle_shop", bool(checked))

    def _on_vibration_toggled(self, checked: bool) -> None:
        self.keymap_flag_toggled.emit("vibration_enabled", bool(checked))

    def _on_retreat_toggled(self, checked: bool) -> None:
        self.keymap_flag_toggled.emit("allow_retreat_chord", bool(checked))

    def _on_hide_capture_toggled(self, checked: bool) -> None:
        self.keymap_flag_toggled.emit("hide_hud_from_capture", bool(checked))

    def _on_pause_clicked(self, _checked: bool = False) -> None:
        # 由主控切换暂停状态并通过 sync 回写；先恢复为当前真实状态避免不一致
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
    "AmberCheckBox",
    "CheckRow",
    "DashboardRuntimeStatus",
    "GamepadTelemetryWidget",
    "KEYMAP_BRIEFS",
    "LineButton",
    "PRTSDashboardWindow",
    "TUNING_SPECS",
    "paint_amber_check",
]
