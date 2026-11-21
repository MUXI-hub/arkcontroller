"""卫戍协议可视化交互校准面板。

按下手柄 Back + Start 或从系统托盘触发时弹出，贴合游戏 16:9 视口：
- 支持鼠标直接拖拽棋盘透视四角控制点（board_quad）并实时预览 5x9 单应性透视网格；
- 支持拖拽调整商店槽位（shop_zone）、底部 10 格整备区（bench_zone）与侧边 5 格装备栏（equip_zone）首尾控制点；
- 支持选择 GameScene 一键截取当前画面保存为 OpenCV 场景识别模板并即时热重载；
- 点击「保存并生效」持久化至 configs/user_override.json 并立即刷新 ControlDispatcher。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QGuiApplication,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPolygonF,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from arkcontroller.config.loader import ConfigLoader
from arkcontroller.config.models import (
    KeymapConfig,
    LayoutProfile,
    LayoutViewState,
    LinearZone,
    PerspectiveQuad,
)
from arkcontroller.control.grid_mapper import LinearZoneMapper, PerspectiveBoardMapper
from arkcontroller.overlay.hud_window import (
    COLOR_AMBER,
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_RED,
    SCENE_DISPLAY_NAMES,
    align_widget_to_physical_rect,
    chamfer_polygon,
    diamond_polygon,
)
from arkcontroller.platform.window_tracker import ViewportRect
from arkcontroller.vision.calibrator import VisionCalibrator
from arkcontroller.vision.scene_detector import GameScene, SceneDetector
from arkcontroller.vision.screen_capture import ScreenCapture

COLOR_VIOLET = QColor("#C084FC")

CALIBRATABLE_SCENES: tuple[GameScene, ...] = (
    GameScene.BATTLE_PREP,
    GameScene.MODAL_CHOICE,
    GameScene.MODAL_CONFIRM,
    GameScene.BATTLE_COMBAT,
)

VIEW_STATE_DISPLAY_NAMES: dict[LayoutViewState, str] = {
    LayoutViewState.NORMAL: "默认态（商店收起 / 未选角色）",
    LayoutViewState.SHOP_OPEN: "商店展开缩放态 (shop_open)",
    LayoutViewState.UNIT_SELECTED: "选取角色缩放态 (unit_selected)",
}


@dataclass
class _DragHandle:
    """可拖拽归一化控制点元数据。"""

    key: str
    group: str
    label: str
    u: float
    v: float
    color: QColor


class _CalibrationCanvas(QWidget):
    """覆盖全视口的交互式透视网格与槽位锚点拖拽画布。"""

    def __init__(self, layout: LayoutProfile, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._base_layout = layout
        self._view_state: LayoutViewState = LayoutViewState.NORMAL
        self._board_rows: int = layout.board_quad.rows
        self._board_cols: int = layout.board_quad.cols
        self._visible_groups: dict[str, bool] = {
            "board": True,
            "shop": True,
            "bench": True,
            "equip": True,
        }
        self._handles: dict[str, _DragHandle] = {}
        self._active_key: str | None = None
        self._hover_key: str | None = None

        self._font_label = QFont("Microsoft YaHei UI", 8, QFont.Weight.Bold)
        self._font_coord = QFont("Consolas", 8, QFont.Weight.Bold)

        self.reset_from_layout(layout, view_state=self._view_state)

    @property
    def view_state(self) -> LayoutViewState:
        return self._view_state

    def reset_from_layout(
        self,
        layout: LayoutProfile,
        *,
        view_state: LayoutViewState | None = None,
    ) -> None:
        """从给定 LayoutProfile 与指定界面缩放态重置全部控制点与棋盘行列维度。"""
        self._base_layout = layout
        if view_state is not None:
            self._view_state = view_state
        q = layout.resolve_board_quad(self._view_state)
        sz = layout.resolve_shop_zone(self._view_state)
        bz = layout.resolve_bench_zone(self._view_state)
        ez = layout.resolve_equip_zone(self._view_state)
        self._board_rows = q.rows
        self._board_cols = q.cols

        self._handles = {
            "board_tl": _DragHandle("board_tl", "board", "棋盘左上", q.top_left[0], q.top_left[1], COLOR_CYAN),
            "board_tr": _DragHandle("board_tr", "board", "棋盘右上", q.top_right[0], q.top_right[1], COLOR_CYAN),
            "board_br": _DragHandle("board_br", "board", "棋盘右下", q.bottom_right[0], q.bottom_right[1], COLOR_CYAN),
            "board_bl": _DragHandle("board_bl", "board", "棋盘左下", q.bottom_left[0], q.bottom_left[1], COLOR_CYAN),
            "shop_start": _DragHandle("shop_start", "shop", "商店起点", sz.start_u, sz.start_v, COLOR_GREEN),
            "shop_end": _DragHandle("shop_end", "shop", "商店终点", sz.end_u, sz.end_v, COLOR_GREEN),
            "bench_start": _DragHandle("bench_start", "bench", "整备起点", bz.start_u, bz.start_v, COLOR_AMBER),
            "bench_end": _DragHandle("bench_end", "bench", "整备终点", bz.end_u, bz.end_v, COLOR_AMBER),
            "equip_start": _DragHandle("equip_start", "equip", "装备起点", ez.start_u, ez.start_v, COLOR_VIOLET),
            "equip_end": _DragHandle("equip_end", "equip", "装备终点", ez.end_u, ez.end_v, COLOR_VIOLET),
        }
        self._active_key = None
        self.update()

    def set_group_visible(self, group: str, visible: bool) -> None:
        """切换某类锚点图层可见性。"""
        self._visible_groups[group] = bool(visible)
        self.update()

    def set_board_dimensions(self, rows: int, cols: int) -> None:
        """更新实时预览与保存使用的棋盘行列数。"""
        self._board_rows = max(1, int(rows))
        self._board_cols = max(1, int(cols))
        self.update()

    def build_board_quad(self) -> PerspectiveQuad:
        """根据当前控制点构造最新棋盘 PerspectiveQuad。"""
        tl = self._handles["board_tl"]
        tr = self._handles["board_tr"]
        br = self._handles["board_br"]
        bl = self._handles["board_bl"]
        return PerspectiveQuad(
            top_left=(round(tl.u, 5), round(tl.v, 5)),
            top_right=(round(tr.u, 5), round(tr.v, 5)),
            bottom_right=(round(br.u, 5), round(br.v, 5)),
            bottom_left=(round(bl.u, 5), round(bl.v, 5)),
            rows=self._board_rows,
            cols=self._board_cols,
        )

    def build_linear_zone(self, group: str, base_zone: LinearZone) -> LinearZone:
        """根据当前首尾控制点构造更新后的 LinearZone。"""
        s = self._handles[f"{group}_start"]
        e = self._handles[f"{group}_end"]
        return LinearZone(
            name=base_zone.name,
            count=base_zone.count,
            start_u=round(s.u, 5),
            start_v=round(s.v, 5),
            end_u=round(e.u, 5),
            end_v=round(e.v, 5),
            slot_w=base_zone.slot_w,
            slot_h=base_zone.slot_h,
        )

    def export_layout_override(self) -> dict[str, object]:
        """导出与 ConfigLoader.save_user_override 兼容的布局覆盖字典（支持三态缩放覆盖）。"""
        quad = self.build_board_quad()
        # 校验单应性可逆，避免保存退化四角导致主循环异常
        PerspectiveBoardMapper(quad)

        shop = self.build_linear_zone("shop", self._base_layout.resolve_shop_zone(self._view_state))
        bench = self.build_linear_zone("bench", self._base_layout.resolve_bench_zone(self._view_state))
        equip = self.build_linear_zone("equip", self._base_layout.resolve_equip_zone(self._view_state))

        if self._view_state == LayoutViewState.NORMAL:
            return {
                "layout": {
                    "board_quad": quad.to_dict(),
                    "shop_zone": shop.to_dict(),
                    "bench_zone": bench.to_dict(),
                    "equip_zone": equip.to_dict(),
                }
            }

        state_payload: dict[str, object] = {
            "board_quad": quad.to_dict(),
            "bench_zone": bench.to_dict(),
            "equip_zone": equip.to_dict(),
        }
        if self._view_state == LayoutViewState.SHOP_OPEN:
            state_payload["shop_zone"] = shop.to_dict()

        return {
            "layout": {
                "view_overrides": {
                    self._view_state.value: state_payload,
                }
            }
        }

    def _hit_test(self, pos: QPointF) -> str | None:
        """检测鼠标位置是否命中任意可见控制点手柄。"""
        w = max(1.0, float(self.width()))
        h = max(1.0, float(self.height()))
        best_key: str | None = None
        best_dist = 16.0
        for key, handle in self._handles.items():
            if not self._visible_groups.get(handle.group, True):
                continue
            hx, hy = handle.u * w, handle.v * h
            dist = math.hypot(pos.x() - hx, pos.y() - hy)
            if dist <= best_dist:
                best_dist = dist
                best_key = key
        return best_key

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            hit = self._hit_test(event.position())
            if hit is not None:
                self._active_key = hit
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                self.update()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._active_key is not None:
            w = max(1.0, float(self.width()))
            h = max(1.0, float(self.height()))
            handle = self._handles[self._active_key]
            handle.u = max(0.0, min(1.0, pos.x() / w))
            handle.v = max(0.0, min(1.0, pos.y() / h))
            self.update()
            event.accept()
            return

        hit = self._hit_test(pos)
        if hit != self._hover_key:
            self._hover_key = hit
            self.setCursor(
                Qt.CursorShape.OpenHandCursor if hit is not None else Qt.CursorShape.ArrowCursor
            )
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._active_key is not None:
            self._active_key = None
            hit = self._hit_test(event.position())
            self.setCursor(
                Qt.CursorShape.OpenHandCursor if hit is not None else Qt.CursorShape.ArrowCursor
            )
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        w = float(self.width())
        h = float(self.height())
        if w <= 1.0 or h <= 1.0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            # 半透明校准暗幕，保留下方游戏画面清晰可见
            painter.fillRect(self.rect(), QColor(8, 12, 18, 128))

            # 16:9 视口校准参考外框：细虚线 + 四角罗德岛战术 L 折角
            painter.setPen(QPen(QColor(0, 200, 255, 110), 1.2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(1.0, 1.0, w - 2.0, h - 2.0))
            corner_pen = QPen(COLOR_CYAN, 3.0)
            corner_pen.setCapStyle(Qt.PenCapStyle.SquareCap)
            painter.setPen(corner_pen)
            arm = 26.0
            for cx, cy, dx, dy in (
                (2.0, 2.0, 1, 1), (w - 2.0, 2.0, -1, 1),
                (2.0, h - 2.0, 1, -1), (w - 2.0, h - 2.0, -1, -1),
            ):
                painter.drawLine(QPointF(cx, cy), QPointF(cx + dx * arm, cy))
                painter.drawLine(QPointF(cx, cy), QPointF(cx, cy + dy * arm))

            # 1. 实时 5x9 透视棋盘网格预览
            if self._visible_groups.get("board", True):
                self._draw_board_preview(painter, w, h)

            # 2. 商店槽位预览
            if self._visible_groups.get("shop", True):
                self._draw_linear_zone_preview(
                    painter,
                    w,
                    h,
                    self.build_linear_zone(
                        "shop", self._base_layout.resolve_shop_zone(self._view_state)
                    ),
                    COLOR_GREEN,
                    prefix="S",
                )

            # 3. 底部 10 格整备区预览
            if self._visible_groups.get("bench", True):
                self._draw_linear_zone_preview(
                    painter,
                    w,
                    h,
                    self.build_linear_zone(
                        "bench", self._base_layout.resolve_bench_zone(self._view_state)
                    ),
                    COLOR_AMBER,
                    prefix="B",
                )

            # 4. 侧边/底部 5 格装备栏预览
            if self._visible_groups.get("equip", True):
                self._draw_linear_zone_preview(
                    painter,
                    w,
                    h,
                    self.build_linear_zone(
                        "equip", self._base_layout.resolve_equip_zone(self._view_state)
                    ),
                    COLOR_VIOLET,
                    prefix="E",
                )

            # 5. 绘制全部可见拖拽手柄
            for key, handle in self._handles.items():
                if not self._visible_groups.get(handle.group, True):
                    continue
                self._draw_handle(
                    painter,
                    w,
                    h,
                    handle,
                    is_active=(key == self._active_key),
                    is_hover=(key == self._hover_key),
                )
        finally:
            painter.end()

    def _draw_board_preview(self, painter: QPainter, w: float, h: float) -> None:
        """实时计算单应性矩阵并绘制 5x9 透视棋盘网格线与格心点。"""
        quad = self.build_board_quad()
        poly_pts = [QPointF(u * w, v * h) for u, v in quad.as_points()]
        outer_poly = QPolygonF(poly_pts)

        try:
            mapper = PerspectiveBoardMapper(quad)
        except ValueError:
            # 四角共线或交叉时绘制红色警告框，不崩溃
            painter.setPen(QPen(COLOR_RED, 2.5, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(QColor(244, 67, 54, 45)))
            painter.drawPolygon(outer_poly)
            return

        # 棋盘淡青底色
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 229, 255, 22)))
        painter.drawPolygon(outer_poly)

        # 绘制横向与纵向透视网格线
        horiz_lines, vert_lines = mapper.grid_lines()
        grid_pen = QPen(QColor(0, 229, 255, 165), 1.4)
        painter.setPen(grid_pen)
        for p1, p2 in horiz_lines:
            painter.drawLine(QPointF(p1[0] * w, p1[1] * h), QPointF(p2[0] * w, p2[1] * h))
        for p1, p2 in vert_lines:
            painter.drawLine(QPointF(p1[0] * w, p1[1] * h), QPointF(p2[0] * w, p2[1] * h))

        # 外框加粗强调
        painter.setPen(QPen(COLOR_CYAN, 2.4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(outer_poly)

        # 绘制各格中心锚点微点
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 229, 255, 190)))
        for r in range(mapper.rows):
            for c in range(mapper.cols):
                cu, cv = mapper.cell_center(r, c)
                painter.drawEllipse(QPointF(cu * w, cv * h), 2.3, 2.3)

    def _draw_linear_zone_preview(
        self,
        painter: QPainter,
        w: float,
        h: float,
        zone: LinearZone,
        color: QColor,
        *,
        prefix: str,
    ) -> None:
        """实时预览线性槽位区的所有等距插值槽位矩形。"""
        mapper = LinearZoneMapper(zone)
        fill = QColor(color)
        fill.setAlpha(30)
        border = QColor(color)
        border.setAlpha(200)

        # 首尾连接参考轴线
        axis_pen = QPen(QColor(color.red(), color.green(), color.blue(), 120), 1.4, Qt.PenStyle.DotLine)
        painter.setPen(axis_pen)
        painter.drawLine(
            QPointF(zone.start_u * w, zone.start_v * h),
            QPointF(zone.end_u * w, zone.end_v * h),
        )

        painter.setFont(self._font_label)
        for idx in range(mapper.count):
            poly = mapper.slot_polygon(idx)
            qpoly = QPolygonF([QPointF(u * w, v * h) for u, v in poly])
            painter.setPen(QPen(border, 1.5))
            painter.setBrush(QBrush(fill))
            painter.drawPolygon(qpoly)

            cu, cv = mapper.slot_center(idx)
            tag_rect = QRectF(cu * w - 18.0, cv * h - 10.0, 36.0, 20.0)
            painter.setPen(QPen(QColor("#F8FAFC")))
            painter.drawText(tag_rect, Qt.AlignmentFlag.AlignCenter, f"{prefix}{idx + 1}")

    def _draw_handle(
        self,
        painter: QPainter,
        w: float,
        h: float,
        handle: _DragHandle,
        *,
        is_active: bool,
        is_hover: bool,
    ) -> None:
        """绘制罗德岛工业风可拖拽菱形锚点（外环菱形 + 实心菱形核）及斜切坐标铭牌。"""
        hx, hy = handle.u * w, handle.v * h
        r = 10.0 if (is_active or is_hover) else 8.0

        # 外层菱形光晕框
        if is_active or is_hover:
            halo = QColor(handle.color)
            halo.setAlpha(80)
            painter.setPen(QPen(handle.color, 1.2))
            painter.setBrush(QBrush(halo))
            painter.drawPolygon(diamond_polygon(hx, hy, r + 7.0))

        # 主菱形控制点 + 黑芯十字
        painter.setPen(QPen(QColor("#FFFFFF"), 2.0))
        painter.setBrush(QBrush(handle.color))
        painter.drawPolygon(diamond_polygon(hx, hy, r))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#0B0E14")))
        painter.drawPolygon(diamond_polygon(hx, hy, r * 0.38))

        # 标签与归一化坐标提示（斜切黑晶铭牌 + 左侧色条）
        text = f"{handle.label} ({handle.u:.3f}, {handle.v:.3f})"
        painter.setFont(self._font_coord)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text) + 20.0
        th = fm.height() + 6.0
        tx = hx + 14.0
        if tx + tw > w - 8.0:
            tx = hx - tw - 14.0
        ty = max(6.0, min(h - th - 6.0, hy - th / 2.0))

        badge_rect = QRectF(tx, ty, tw, th)
        painter.setPen(QPen(handle.color, 1.0))
        painter.setBrush(QBrush(QColor(11, 14, 20, 232)))
        painter.drawPolygon(chamfer_polygon(badge_rect, tl=5.0, br=5.0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(handle.color))
        painter.drawRect(QRectF(badge_rect.left() + 3.0, badge_rect.top() + 3.0, 2.5, th - 6.0))
        painter.setPen(QPen(QColor("#F4F6F9")))
        painter.drawText(badge_rect.adjusted(6.0, 0.0, 0.0, 0.0), Qt.AlignmentFlag.AlignCenter, text)


class CalibrationDialog(QDialog):
    """卫戍协议可视化校准交互悬浮窗。"""

    calibration_saved = Signal(object, object)

    def __init__(
        self,
        *,
        layout: LayoutProfile,
        viewport: ViewportRect | None,
        config_loader: ConfigLoader,
        calibrator: VisionCalibrator | None = None,
        scene_detector: SceneDetector | None = None,
        screen_capture: ScreenCapture | None = None,
        on_saved: Callable[[KeymapConfig, LayoutProfile], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._viewport = viewport
        self._config_loader = config_loader
        self._calibrator = calibrator
        self._scene_detector = scene_detector
        self._screen_capture = screen_capture
        self._on_saved = on_saved
        self._layout = layout

        self.setWindowTitle("ArkController — 卫戍协议可视化校准向导")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Dialog
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # 底层全视口交互画布
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        self._canvas = _CalibrationCanvas(layout, self)
        root_layout.addWidget(self._canvas)

        # 左上角悬浮控制面板（作为画布子控件浮动停靠，不遮挡棋盘中央）
        self._dock = self._build_control_dock(self._canvas)
        self._dock.move(16, 16)
        self._dock.adjustSize()

        self._apply_viewport_geometry()

    def _build_control_dock(self, parent: QWidget) -> QFrame:
        """构建 PRTS 风格半透明悬浮控制卡片。"""
        dock = QFrame(parent)
        dock.setObjectName("CalibrationDock")
        dock.setFixedWidth(332)
        dock.setStyleSheet(
            """
            QFrame#CalibrationDock {
                background-color: rgba(11, 14, 20, 236);
                border: 1px solid rgba(0, 200, 255, 170);
                border-left: 4px solid #00C8FF;
                border-radius: 0px;
            }
            QLabel {
                color: #E2E8F0;
                font-family: "Microsoft YaHei UI";
                font-size: 9pt;
            }
            QCheckBox {
                color: #E2E8F0;
                font-family: "Microsoft YaHei UI";
                font-size: 9pt;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 11px;
                height: 11px;
                background-color: #121822;
                border: 1px solid #2C3A4D;
            }
            QCheckBox::indicator:checked {
                background-color: #00C8FF;
                border: 1px solid #F4F6F9;
            }
            QComboBox, QSpinBox {
                background-color: #121822;
                color: #F4F6F9;
                border: 1px solid #2C3A4D;
                border-radius: 0px;
                padding: 4px 8px;
                font-family: "Microsoft YaHei UI";
                font-size: 9pt;
            }
            QComboBox:hover, QSpinBox:hover {
                border-color: #00C8FF;
            }
            QComboBox QAbstractItemView {
                background-color: #0B0E14;
                color: #F4F6F9;
                selection-background-color: rgba(0, 200, 255, 70);
                border: 1px solid #00C8FF;
            }
            QPushButton {
                background-color: #151C27;
                color: #F4F6F9;
                border: 1px solid #2C3A4D;
                border-radius: 0px;
                padding: 6px 10px;
                font-family: "Microsoft YaHei UI";
                font-size: 9pt;
                font-weight: bold;
            }
            QPushButton:hover {
                border-color: #00C8FF;
                background-color: #1B2533;
            }
            QPushButton#SaveBtn {
                background-color: #FF9800;
                color: #120C02;
                border: 1px solid #FFD08A;
            }
            QPushButton#SaveBtn:hover {
                background-color: #FFB547;
            }
            QPushButton#CaptureBtn {
                background-color: rgba(0, 200, 255, 40);
                color: #F4F6F9;
                border: 1px solid #00C8FF;
            }
            QPushButton#CaptureBtn:hover {
                background-color: rgba(0, 200, 255, 80);
            }
            """
        )

        vbox = QVBoxLayout(dock)
        vbox.setContentsMargins(14, 12, 14, 12)
        vbox.setSpacing(8)

        title = QLabel("PRTS // CALIBRATION  卫戍协议可视化校准向导")
        title.setStyleSheet("color: #00C8FF; font-size: 10pt; font-weight: bold; border-bottom: 1px solid #1F2A38; padding-bottom: 4px;")
        vbox.addWidget(title)

        desc = QLabel("先选择界面缩放态，再拖动菱形控制点对齐游戏画面并保存。")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #8A99AD; font-size: 8pt;")
        vbox.addWidget(desc)

        # 校准视图态切换（默认态 / 商店展开态 / 选中角色态）
        view_label = QLabel("当前校准视图态（应对界面缩放错位）：")
        view_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        vbox.addWidget(view_label)

        self._view_state_combo = QComboBox()
        for vs in LayoutViewState:
            self._view_state_combo.addItem(VIEW_STATE_DISPLAY_NAMES[vs], userData=vs)
        self._view_state_combo.currentIndexChanged.connect(self._on_view_state_changed)
        vbox.addWidget(self._view_state_combo)

        # 图层开关行
        layer_row1 = QHBoxLayout()
        cb_board = QCheckBox("棋盘透视")
        cb_board.setChecked(True)
        cb_board.toggled.connect(lambda v: self._canvas.set_group_visible("board", v))
        cb_shop = QCheckBox("商店槽位")
        cb_shop.setChecked(True)
        cb_shop.toggled.connect(lambda v: self._canvas.set_group_visible("shop", v))
        layer_row1.addWidget(cb_board)
        layer_row1.addWidget(cb_shop)
        vbox.addLayout(layer_row1)

        layer_row2 = QHBoxLayout()
        cb_bench = QCheckBox("整备区 (10格)")
        cb_bench.setChecked(True)
        cb_bench.toggled.connect(lambda v: self._canvas.set_group_visible("bench", v))
        cb_equip = QCheckBox("装备栏 (5格)")
        cb_equip.setChecked(True)
        cb_equip.toggled.connect(lambda v: self._canvas.set_group_visible("equip", v))
        layer_row2.addWidget(cb_bench)
        layer_row2.addWidget(cb_equip)
        vbox.addLayout(layer_row2)

        # 棋盘行列维度微调
        dim_row = QHBoxLayout()
        dim_row.addWidget(QLabel("棋盘网格维度:"))
        self._spin_rows = QSpinBox()
        self._spin_rows.setRange(1, 12)
        self._spin_rows.setValue(self._layout.board_quad.rows)
        self._spin_rows.setPrefix("行 ")
        self._spin_rows.valueChanged.connect(self._on_board_dim_changed)

        self._spin_cols = QSpinBox()
        self._spin_cols.setRange(1, 16)
        self._spin_cols.setValue(self._layout.board_quad.cols)
        self._spin_cols.setPrefix("列 ")
        self._spin_cols.valueChanged.connect(self._on_board_dim_changed)
        dim_row.addWidget(self._spin_rows)
        dim_row.addWidget(self._spin_cols)
        vbox.addLayout(dim_row)

        # 分割线
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("color: #1F2A38;")
        vbox.addWidget(sep1)

        # 场景识别模板采样区域
        tpl_title = QLabel("场景视觉模板采样（提升自动切层精准度）")
        tpl_title.setStyleSheet("color: #FF9800; font-weight: bold;")
        vbox.addWidget(tpl_title)

        self._scene_combo = QComboBox()
        for scene in CALIBRATABLE_SCENES:
            label = f"{SCENE_DISPLAY_NAMES.get(scene, scene.value)} ({scene.value})"
            self._scene_combo.addItem(label, userData=scene)
        self._scene_combo.addItem("辅助状态：商店已展开 (shop_open)", userData="shop_open")
        self._scene_combo.addItem("辅助状态：选取角色缩放 (unit_selected)", userData="unit_selected")
        vbox.addWidget(self._scene_combo)

        self._btn_capture = QPushButton("截取当前画面作为场景识别模板")
        self._btn_capture.setObjectName("CaptureBtn")
        self._btn_capture.clicked.connect(self._on_capture_template_clicked)
        vbox.addWidget(self._btn_capture)

        # 状态提示栏
        self._status_label = QLabel("提示：按 Esc 或点击关闭可立即返回游戏穿透操控。")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #8A99AD; font-size: 8pt;")
        vbox.addWidget(self._status_label)

        # 底部操作按钮
        btn_row = QHBoxLayout()
        btn_reset = QPushButton("重置")
        btn_reset.clicked.connect(self._on_reset_clicked)
        btn_save = QPushButton("保存并生效")
        btn_save.setObjectName("SaveBtn")
        btn_save.clicked.connect(self._on_save_clicked)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.reject)
        btn_row.addWidget(btn_reset)
        btn_row.addWidget(btn_save, stretch=1)
        btn_row.addWidget(btn_close)
        vbox.addLayout(btn_row)

        return dock

    def _apply_viewport_geometry(self) -> None:
        """贴合游戏 16:9 视口，无游戏窗口时回退至屏幕居中 1280x720。"""
        if (
            self._viewport is not None
            and self._viewport.active_width > 0
            and self._viewport.active_height > 0
        ):
            align_widget_to_physical_rect(
                self,
                self._viewport.active_left,
                self._viewport.active_top,
                self._viewport.active_width,
                self._viewport.active_height,
                topmost=True,
                no_activate=False,
            )
            return

        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            w = min(1280, max(960, int(avail.width() * 0.8)))
            h = int(round(w * 9.0 / 16.0))
            if h > avail.height() - 40:
                h = max(540, avail.height() - 40)
                w = int(round(h * 16.0 / 9.0))
            x = avail.left() + (avail.width() - w) // 2
            y = avail.top() + (avail.height() - h) // 2
            self.setGeometry(x, y, w, h)
        else:
            self.setGeometry(100, 100, 1280, 720)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._apply_viewport_geometry()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    def _on_view_state_changed(self) -> None:
        raw_vs = self._view_state_combo.currentData()
        if isinstance(raw_vs, LayoutViewState):
            vs = raw_vs
        elif isinstance(raw_vs, str):
            try:
                vs = LayoutViewState(raw_vs)
            except ValueError:
                return
        else:
            return
        resolved_quad = self._layout.resolve_board_quad(vs)
        self._spin_rows.blockSignals(True)
        self._spin_cols.blockSignals(True)
        self._spin_rows.setValue(resolved_quad.rows)
        self._spin_cols.setValue(resolved_quad.cols)
        self._spin_rows.blockSignals(False)
        self._spin_cols.blockSignals(False)
        self._canvas.reset_from_layout(self._layout, view_state=vs)
        self._set_status(f"已切换校准视图态：{VIEW_STATE_DISPLAY_NAMES[vs]}", "#00E5FF")

    def _on_board_dim_changed(self) -> None:
        self._canvas.set_board_dimensions(self._spin_rows.value(), self._spin_cols.value())

    def _on_reset_clicked(self) -> None:
        vs = self._canvas.view_state
        resolved_quad = self._layout.resolve_board_quad(vs)
        self._spin_rows.blockSignals(True)
        self._spin_cols.blockSignals(True)
        self._spin_rows.setValue(resolved_quad.rows)
        self._spin_cols.setValue(resolved_quad.cols)
        self._spin_rows.blockSignals(False)
        self._spin_cols.blockSignals(False)
        self._canvas.reset_from_layout(self._layout, view_state=vs)
        self._set_status("已重置为当前生效配置。", "#94A3B8")

    def _on_capture_template_clicked(self) -> None:
        """截取游戏原始干净画面并写入对应场景/状态识别模板，随即热重载检测器。"""
        if self._calibrator is None:
            self._set_status("校准器未初始化，无法保存场景模板。", "#F44336")
            return

        target = self._scene_combo.currentData()
        frame = None
        if self._screen_capture is not None and self._viewport is not None:
            # 暂时隐藏自身不透明度以避免校准遮罩污染截图模板
            prev_opacity = self.windowOpacity()
            try:
                self.setWindowOpacity(0.0)
                QApplication.processEvents()
                frame = self._screen_capture.grab_frame(self._viewport)
            finally:
                self.setWindowOpacity(prev_opacity)
                QApplication.processEvents()

        if frame is None and self._screen_capture is not None:
            frame = self._screen_capture.get_latest_frame()

        if frame is None:
            self._set_status(
                "未获取到游戏画面帧：请先确保《明日方舟》窗口可见且未被最小化。",
                "#FFB300",
            )
            return

        try:
            if target == "shop_open":
                saved_path = self._calibrator.capture_shop_open_template(frame)
            elif target == "unit_selected":
                saved_path = self._calibrator.capture_unit_selected_template(frame)
            else:
                scene_enum = target if isinstance(target, GameScene) else GameScene(str(target))
                saved_path = self._calibrator.capture_scene_template(frame, scene_enum)
            if self._scene_detector is not None:
                self._scene_detector.reload_templates()
            self._set_status(f"模板已保存并热重载：{saved_path.name}", "#4CAF50")
        except Exception as exc:
            self._set_status(f"模板截取失败：{exc}", "#F44336")

    def _on_save_clicked(self) -> None:
        """保存校准参数至 user_override.json 并立即刷新核心调度器。"""
        try:
            override_data = self._canvas.export_layout_override()
            self._config_loader.save_user_override(override_data)
            new_keymap, new_layout = self._config_loader.reload()
        except Exception as exc:
            self._set_status(f"保存失败（请检查棋盘四角是否交叉）：{exc}", "#F44336")
            return

        self._layout = new_layout
        if self._on_saved is not None:
            self._on_saved(new_keymap, new_layout)
        self.calibration_saved.emit(new_keymap, new_layout)
        self.accept()

    def _set_status(self, message: str, color_hex: str) -> None:
        self._status_label.setText(message)
        self._status_label.setStyleSheet(f"color: {color_hex}; font-size: 8pt;")
