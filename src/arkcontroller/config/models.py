"""ArkController 配置数据模型定义与合法性校验。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

VALID_ANCHORS: tuple[str, ...] = (
    "center_16_9",
    "left_edge",
    "right_edge",
    "top_edge",
    "bottom_edge",
    "client",
)

VALID_LEFT_STICK_MODES: tuple[str, ...] = ("dpad", "cursor")


class LayoutViewState(str, Enum):
    """卫戍协议局内界面镜头/缩放状态枚举。"""

    NORMAL = "normal"
    SHOP_OPEN = "shop_open"
    UNIT_SELECTED = "unit_selected"

REQUIRED_BUTTON_KEYS: tuple[str, ...] = (
    "back",
    "shop_refresh",
    "shop_upgrade",
    "shop_freeze",
    "battle_ready",
    "covenant_panel",
    "enemy_preview",
    "emote_wheel",
    "pause_menu",
    "choice_toggle_eye",
    "sell_dropzone",
    "teammate_1",
    "teammate_2",
    "teammate_3",
    "teammate_4",
    "strategy_avatar",
    "strategy_inspect",
    "confirm_no",
    "confirm_yes",
    "confirm_dont_ask",
)


def _validate_unit_coord(name: str, value: float) -> float:
    """校验归一化坐标处于 [0.0, 1.0] 合法区间并返回浮点数。"""
    val = float(value)
    if not (0.0 <= val <= 1.0):
        raise ValueError(f"{name} 必须位于 [0.0, 1.0] 范围内，当前值: {val}")
    return val


def _validate_positive_unit(name: str, value: float) -> float:
    """校验归一化尺寸处于 (0.0, 1.0] 合法区间并返回浮点数。"""
    val = float(value)
    if not (0.0 < val <= 1.0):
        raise ValueError(f"{name} 必须位于 (0.0, 1.0] 范围内，当前值: {val}")
    return val


def _default_keyboard_shortcuts() -> dict[str, str]:
    """返回《明日方舟》官方 PC 客户端卫戍协议实机原生快捷键映射。"""
    return {
        "shop_toggle": "a",
        "shop_freeze": "s",
        "shop_refresh": "d",
        "shop_upgrade": "g",
        "battle_ready": "c",
        "enemy_preview": "w",
        "sell_board": "x",
        "sell_bench": "x",
        "retreat_unit": "q",
        "covenant_panel": "",
        "emote_wheel": "",
        "back": "esc",
        "pause_menu": "esc",
    }


def _default_bindings() -> dict[str, dict[str, str]]:
    """返回各场景默认 Xbox 手柄按键功能映射表。"""
    return {
        "global": {
            "cycle_scene_layer": "BACK",
            "toggle_calibration": "BACK+START",
            "toggle_pause": "LT+BACK",
            "toggle_free_cursor": "LS",
        },
        "battle_prep": {
            "select_or_grab": "A",
            "cancel": "B",
            "shop_refresh": "X",
            "shop_upgrade": "Y",
            "shop_freeze": "RS",
            "fast_buy_or_inspect": "RT+A",
            "toggle_shop": "LT+LB",
            "prev_zone": "LB",
            "next_zone": "RB",
            "battle_ready_hold": "START",
        },
        "holding": {
            "place_or_confirm": "A",
            "cancel": "B",
            "quick_sell": "X",
            "toggle_bench_board": "LB/RB",
        },
        "modal_choice": {
            "confirm_choice": "A",
            "fast_confirm_choice": "RT+A",
            "peek_board_hold": "LT",
        },
        "modal_confirm": {
            "confirm": "A",
            "cancel": "B",
            "quick_confirm": "START",
            "toggle_dont_ask": "X",
        },
        "lt_chords": {
            "covenant_panel": "LT+DPAD_LEFT",
            "enemy_preview": "LT+DPAD_UP",
            "emote_wheel": "LT+DPAD_DOWN",
            "cycle_teammate": "LT+DPAD_RIGHT",
            "toggle_shop": "LT+LB",
            "retreat_unit": "LT+RB",
            "use_strategy": "LT+RT",
            "pause_menu": "LT+START",
        },
    }


@dataclass
class NormalizedPoint:
    """归一化屏幕锚点坐标，基于 16:9 核心视口或贴边锚点。"""

    u: float
    v: float
    anchor: str = "center_16_9"

    def __post_init__(self) -> None:
        self.u = _validate_unit_coord("u", self.u)
        self.v = _validate_unit_coord("v", self.v)
        if self.anchor not in VALID_ANCHORS:
            raise ValueError(
                f"无效的锚点模式 '{self.anchor}'，允许值: {', '.join(VALID_ANCHORS)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """序列化为可持久化 JSON 字典。"""
        return {"u": self.u, "v": self.v, "anchor": self.anchor}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | list[float] | tuple[float, float]) -> NormalizedPoint:
        """从字典或二元坐标序列反序列化归一化点。"""
        if isinstance(data, (list, tuple)) and len(data) == 2:
            return cls(u=float(data[0]), v=float(data[1]))
        if isinstance(data, Mapping):
            return cls(
                u=float(data["u"]),
                v=float(data["v"]),
                anchor=str(data.get("anchor", "center_16_9")),
            )
        raise TypeError(f"无法将 {type(data).__name__} 解析为 NormalizedPoint")


@dataclass
class NormalizedRect:
    """归一化矩形区域，以左上角 (u, v) 与宽高 (w, h) 表示。"""

    u: float
    v: float
    w: float
    h: float

    def __post_init__(self) -> None:
        self.u = _validate_unit_coord("u", self.u)
        self.v = _validate_unit_coord("v", self.v)
        self.w = _validate_positive_unit("w", self.w)
        self.h = _validate_positive_unit("h", self.h)

    @property
    def center(self) -> tuple[float, float]:
        """返回矩形中心归一化坐标。"""
        return (self.u + self.w * 0.5, self.v + self.h * 0.5)

    @property
    def right(self) -> float:
        """返回矩形右边界归一化坐标。"""
        return self.u + self.w

    @property
    def bottom(self) -> float:
        """返回矩形下边界归一化坐标。"""
        return self.v + self.h

    def to_dict(self) -> dict[str, float]:
        """序列化为 JSON 字典。"""
        return {"u": self.u, "v": self.v, "w": self.w, "h": self.h}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NormalizedRect:
        """从字典反序列化归一化矩形。"""
        return cls(
            u=float(data["u"]),
            v=float(data["v"]),
            w=float(data["w"]),
            h=float(data["h"]),
        )


@dataclass
class PerspectiveQuad:
    """塔防 3D 俯视棋盘 4 角归一化梯形与行列维度定义。"""

    top_left: tuple[float, float]
    top_right: tuple[float, float]
    bottom_right: tuple[float, float]
    bottom_left: tuple[float, float]
    rows: int = 5
    cols: int = 9

    def __post_init__(self) -> None:
        self.top_left = self._normalize_vertex("top_left", self.top_left)
        self.top_right = self._normalize_vertex("top_right", self.top_right)
        self.bottom_right = self._normalize_vertex("bottom_right", self.bottom_right)
        self.bottom_left = self._normalize_vertex("bottom_left", self.bottom_left)
        self.rows = int(self.rows)
        self.cols = int(self.cols)
        if self.rows < 1 or self.cols < 1:
            raise ValueError(f"棋盘行列数必须为正整数，当前: rows={self.rows}, cols={self.cols}")

    @staticmethod
    def _normalize_vertex(name: str, pt: Any) -> tuple[float, float]:
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise ValueError(f"{name} 必须为二元归一化坐标 (u, v)")
        return (
            _validate_unit_coord(f"{name}[0]", pt[0]),
            _validate_unit_coord(f"{name}[1]", pt[1]),
        )

    def as_points(self) -> list[tuple[float, float]]:
        """按顺时针顺序返回四角顶点列表。"""
        return [self.top_left, self.top_right, self.bottom_right, self.bottom_left]

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 字典。"""
        return {
            "top_left": list(self.top_left),
            "top_right": list(self.top_right),
            "bottom_right": list(self.bottom_right),
            "bottom_left": list(self.bottom_left),
            "rows": self.rows,
            "cols": self.cols,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PerspectiveQuad:
        """从字典反序列化棋盘透视四角配置。"""
        return cls(
            top_left=(float(data["top_left"][0]), float(data["top_left"][1])),
            top_right=(float(data["top_right"][0]), float(data["top_right"][1])),
            bottom_right=(float(data["bottom_right"][0]), float(data["bottom_right"][1])),
            bottom_left=(float(data["bottom_left"][0]), float(data["bottom_left"][1])),
            rows=int(data.get("rows", 5)),
            cols=int(data.get("cols", 9)),
        )


@dataclass
class NavNode:
    """不规则 UI 界面（活动主界面、编队、结算页）的导航图节点。"""

    id: str
    label: str
    u: float
    v: float
    w: float = 0.08
    h: float = 0.06
    up: str | None = None
    down: str | None = None
    left: str | None = None
    right: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError("NavNode.id 必须为非空字符串")
        self.u = _validate_unit_coord(f"NavNode({self.id}).u", self.u)
        self.v = _validate_unit_coord(f"NavNode({self.id}).v", self.v)
        self.w = _validate_positive_unit(f"NavNode({self.id}).w", self.w)
        self.h = _validate_positive_unit(f"NavNode({self.id}).h", self.h)

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 字典。"""
        return {
            "id": self.id,
            "label": self.label,
            "u": self.u,
            "v": self.v,
            "w": self.w,
            "h": self.h,
            "up": self.up,
            "down": self.down,
            "left": self.left,
            "right": self.right,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], node_id: str | None = None) -> NavNode:
        """从字典反序列化导航节点。"""
        resolved_id = str(data.get("id") or node_id or "")
        return cls(
            id=resolved_id,
            label=str(data.get("label", resolved_id)),
            u=float(data["u"]),
            v=float(data["v"]),
            w=float(data.get("w", 0.08)),
            h=float(data.get("h", 0.06)),
            up=data.get("up"),
            down=data.get("down"),
            left=data.get("left"),
            right=data.get("right"),
        )


@dataclass
class LinearZone:
    """线性等距排列槽位区（商店、整备区 10 槽、装备栏 5 槽、三选一卡片）。"""

    name: str
    count: int
    start_u: float
    start_v: float
    end_u: float
    end_v: float
    slot_w: float
    slot_h: float

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("LinearZone.name 必须为非空字符串")
        self.count = int(self.count)
        if self.count < 1:
            raise ValueError(f"LinearZone({self.name}).count 必须 >= 1，当前值: {self.count}")
        self.start_u = _validate_unit_coord(f"{self.name}.start_u", self.start_u)
        self.start_v = _validate_unit_coord(f"{self.name}.start_v", self.start_v)
        self.end_u = _validate_unit_coord(f"{self.name}.end_u", self.end_u)
        self.end_v = _validate_unit_coord(f"{self.name}.end_v", self.end_v)
        self.slot_w = _validate_positive_unit(f"{self.name}.slot_w", self.slot_w)
        self.slot_h = _validate_positive_unit(f"{self.name}.slot_h", self.slot_h)

    def slot_center(self, index: int) -> tuple[float, float]:
        """返回第 index 个槽位中心的归一化坐标 (u, v)。"""
        if not (0 <= index < self.count):
            raise IndexError(f"槽位索引越界: index={index}, count={self.count}")
        if self.count == 1:
            return (self.start_u, self.start_v)
        t = index / (self.count - 1)
        return (
            self.start_u + (self.end_u - self.start_u) * t,
            self.start_v + (self.end_v - self.start_v) * t,
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 字典。"""
        return {
            "name": self.name,
            "count": self.count,
            "start_u": self.start_u,
            "start_v": self.start_v,
            "end_u": self.end_u,
            "end_v": self.end_v,
            "slot_w": self.slot_w,
            "slot_h": self.slot_h,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LinearZone:
        """从字典反序列化线性槽位区配置。"""
        return cls(
            name=str(data["name"]),
            count=int(data["count"]),
            start_u=float(data["start_u"]),
            start_v=float(data["start_v"]),
            end_u=float(data["end_u"]),
            end_v=float(data["end_v"]),
            slot_w=float(data["slot_w"]),
            slot_h=float(data["slot_h"]),
        )


# 兼容设计文档中的别名引用
SlotZoneConfig = LinearZone


@dataclass
class SymmetricCovenantZone:
    """顶部栏下方居中对称分布（超过 7 个时左对齐滑动）的羁绊（盟约）图标区域配置。"""

    center_u: float = 0.500
    v: float = 0.1208
    step_u: float = 0.0625
    slot_w: float = 0.044
    slot_h: float = 0.082
    default_count: int = 5
    overflow_start_u: float = 0.2961

    def __post_init__(self) -> None:
        self.center_u = _validate_unit_coord("covenant_zone.center_u", self.center_u)
        self.v = _validate_unit_coord("covenant_zone.v", self.v)
        self.step_u = _validate_positive_unit("covenant_zone.step_u", self.step_u)
        self.slot_w = _validate_positive_unit("covenant_zone.slot_w", self.slot_w)
        self.slot_h = _validate_positive_unit("covenant_zone.slot_h", self.slot_h)
        self.overflow_start_u = _validate_unit_coord(
            "covenant_zone.overflow_start_u", self.overflow_start_u
        )
        self.default_count = int(self.default_count)
        if self.default_count < 0:
            raise ValueError(
                f"SymmetricCovenantZone.default_count 必须 >= 0，当前值: {self.default_count}"
            )

    def slot_center(
        self,
        index: int,
        count: int | None = None,
        overflow: bool = False,
    ) -> tuple[float, float]:
        """计算第 index 个盟约图标归一化中心坐标 (u, v)。

        - overflow=False：按总盟约数 count 居中对称排布；
        - overflow=True：超过 7 枚盟约溢出时按 overflow_start_u 左对齐等距排布。
        """
        n = self.default_count if count is None else int(count)
        if n < 1 or not (0 <= index < n):
            raise IndexError(f"羁绊槽位索引越界: index={index}, count={n}")
        if overflow:
            u = max(0.0, min(1.0, self.overflow_start_u + index * self.step_u))
            return (float(u), float(self.v))
        offset = (index - (n - 1) * 0.5) * self.step_u
        u = max(0.0, min(1.0, self.center_u + offset))
        return (float(u), float(self.v))

    def slot_polygon(
        self,
        index: int,
        count: int | None = None,
        overflow: bool = False,
    ) -> list[tuple[float, float]]:
        """返回第 index 个盟约图标的顺时针 4 顶点归一化矩形。"""
        cu, cv = self.slot_center(index, count=count, overflow=overflow)
        hw = self.slot_w * 0.5
        hh = self.slot_h * 0.5
        return [
            (max(0.0, cu - hw), max(0.0, cv - hh)),
            (min(1.0, cu + hw), max(0.0, cv - hh)),
            (min(1.0, cu + hw), min(1.0, cv + hh)),
            (max(0.0, cu - hw), min(1.0, cv + hh)),
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_u": self.center_u,
            "v": self.v,
            "step_u": self.step_u,
            "slot_w": self.slot_w,
            "slot_h": self.slot_h,
            "default_count": self.default_count,
            "overflow_start_u": self.overflow_start_u,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SymmetricCovenantZone:
        return cls(
            center_u=float(data.get("center_u", 0.500)),
            v=float(data.get("v", 0.1208)),
            step_u=float(data.get("step_u", 0.0625)),
            slot_w=float(data.get("slot_w", 0.044)),
            slot_h=float(data.get("slot_h", 0.082)),
            default_count=int(data.get("default_count", 5)),
            overflow_start_u=float(data.get("overflow_start_u", 0.2961)),
        )


def _default_choice_2_zone() -> LinearZone:
    return LinearZone(
        name="choice_2",
        count=2,
        start_u=0.3809,
        start_v=0.5000,
        end_u=0.6477,
        end_v=0.5000,
        slot_w=0.2188,
        slot_h=0.3306,
    )


def _default_choice_6_top_zone() -> LinearZone:
    return LinearZone(
        name="choice_6_top",
        count=3,
        start_u=0.2473,
        start_v=0.2931,
        end_u=0.7816,
        end_v=0.2931,
        slot_w=0.2180,
        slot_h=0.3306,
    )


def _default_choice_6_bottom_zone() -> LinearZone:
    return LinearZone(
        name="choice_6_bottom",
        count=3,
        start_u=0.2473,
        start_v=0.7083,
        end_u=0.7816,
        end_v=0.7083,
        slot_w=0.2180,
        slot_h=0.3306,
    )


def _default_emote_top_zone() -> LinearZone:
    return LinearZone(
        name="emote_top",
        count=3,
        start_u=0.0582,
        start_v=0.6771,
        end_u=0.1996,
        end_v=0.6771,
        slot_w=0.0640,
        slot_h=0.1120,
    )


def _default_emote_bottom_zone() -> LinearZone:
    return LinearZone(
        name="emote_bottom",
        count=3,
        start_u=0.0582,
        start_v=0.8016,
        end_u=0.1996,
        end_v=0.8016,
        slot_w=0.0640,
        slot_h=0.1120,
    )


def _default_mirrored_board_quad() -> PerspectiveQuad:
    """返回 Boss 战右侧分配时水平镜像的 4x7 棋盘透视四角（保持逻辑左→右列序）。"""
    return PerspectiveQuad(
        top_left=(0.17357, 0.28519),
        top_right=(0.59649, 0.27870),
        bottom_right=(0.61576, 0.66019),
        bottom_left=(0.12774, 0.66759),
        rows=4,
        cols=7,
    )


def _default_mirrored_equip_zone() -> LinearZone:
    """返回 Boss 战右侧分配时水平镜像的 5 槽装备台线性区域。"""
    return LinearZone(
        name="equip_mirrored",
        count=5,
        start_u=0.22475,
        start_v=0.71705,
        end_u=0.50715,
        end_v=0.71705,
        slot_w=0.068,
        slot_h=0.088,
    )


def _default_fallback_buttons() -> dict[str, NormalizedPoint]:
    """返回回合确认弹窗、安全空白区、策略头像与交流面板的默认实测按钮锚点（兼容精简配置字典）。"""
    return {
        "safe_blank": NormalizedPoint(u=0.1172, v=0.0903, anchor="left_edge"),
        "strategy_avatar": NormalizedPoint(u=0.9605, v=0.3243, anchor="right_edge"),
        "strategy_inspect": NormalizedPoint(u=0.9047, v=0.3236, anchor="right_edge"),
        "strategy_avatar_combat": NormalizedPoint(u=0.9602, v=0.7486, anchor="right_edge"),
        "strategy_inspect_combat": NormalizedPoint(u=0.9051, v=0.7493, anchor="right_edge"),
        "confirm_no": NormalizedPoint(u=0.2496, v=0.6729, anchor="center_16_9"),
        "confirm_yes": NormalizedPoint(u=0.7496, v=0.6729, anchor="center_16_9"),
        "confirm_dont_ask": NormalizedPoint(u=0.4992, v=0.5597, anchor="center_16_9"),
        "emote_panel_center": NormalizedPoint(u=0.1290, v=0.7400, anchor="left_edge"),
    }


@dataclass
class ViewportTransform:
    """界面缩放与平移变换参数（用于商店展开或选取角色触发镜头缩放后的坐标补偿）。"""

    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_u: float = 0.0
    offset_v: float = 0.0
    pivot_u: float = 0.5
    pivot_v: float = 0.5

    def __post_init__(self) -> None:
        self.scale_x = float(self.scale_x)
        self.scale_y = float(self.scale_y)
        self.offset_u = float(self.offset_u)
        self.offset_v = float(self.offset_v)
        self.pivot_u = _validate_unit_coord("pivot_u", self.pivot_u)
        self.pivot_v = _validate_unit_coord("pivot_v", self.pivot_v)
        if self.scale_x <= 0.0 or self.scale_y <= 0.0:
            raise ValueError(
                f"缩放因子必须大于 0，当前值: scale_x={self.scale_x}, scale_y={self.scale_y}"
            )

    def is_identity(self) -> bool:
        """判断当前变换是否为单位恒等变换。"""
        return (
            abs(self.scale_x - 1.0) < 1e-6
            and abs(self.scale_y - 1.0) < 1e-6
            and abs(self.offset_u) < 1e-6
            and abs(self.offset_v) < 1e-6
        )

    def apply_point(self, u: float, v: float) -> tuple[float, float]:
        """对单个归一化坐标点应用缩放与平移并安全裁剪至 [0.0, 1.0]。"""
        nu = self.pivot_u + (float(u) - self.pivot_u) * self.scale_x + self.offset_u
        nv = self.pivot_v + (float(v) - self.pivot_v) * self.scale_y + self.offset_v
        return (max(0.0, min(1.0, nu)), max(0.0, min(1.0, nv)))

    def apply_quad(self, quad: PerspectiveQuad) -> PerspectiveQuad:
        """对棋盘透视四角应用缩放与平移变换，返回新 PerspectiveQuad。"""
        if self.is_identity():
            return quad
        return PerspectiveQuad(
            top_left=self.apply_point(*quad.top_left),
            top_right=self.apply_point(*quad.top_right),
            bottom_right=self.apply_point(*quad.bottom_right),
            bottom_left=self.apply_point(*quad.bottom_left),
            rows=quad.rows,
            cols=quad.cols,
        )

    def apply_zone(self, zone: LinearZone) -> LinearZone:
        """对线性槽位区应用缩放与平移变换，返回新 LinearZone。"""
        if self.is_identity():
            return zone
        su, sv = self.apply_point(zone.start_u, zone.start_v)
        eu, ev = self.apply_point(zone.end_u, zone.end_v)
        sw = max(1e-4, min(1.0, zone.slot_w * self.scale_x))
        sh = max(1e-4, min(1.0, zone.slot_h * self.scale_y))
        return LinearZone(
            name=zone.name,
            count=zone.count,
            start_u=su,
            start_v=sv,
            end_u=eu,
            end_v=ev,
            slot_w=sw,
            slot_h=sh,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "offset_u": self.offset_u,
            "offset_v": self.offset_v,
            "pivot_u": self.pivot_u,
            "pivot_v": self.pivot_v,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ViewportTransform:
        return cls(
            scale_x=float(data.get("scale_x", 1.0)),
            scale_y=float(data.get("scale_y", 1.0)),
            offset_u=float(data.get("offset_u", 0.0)),
            offset_v=float(data.get("offset_v", 0.0)),
            pivot_u=float(data.get("pivot_u", 0.5)),
            pivot_v=float(data.get("pivot_v", 0.5)),
        )


@dataclass
class ViewLayoutOverride:
    """特定界面缩放态（如 shop_open / unit_selected）下的区域几何或缩放补偿覆盖。"""

    board_quad: PerspectiveQuad | None = None
    bench_zone: LinearZone | None = None
    equip_zone: LinearZone | None = None
    shop_zone: LinearZone | None = None
    board_transform: ViewportTransform | None = None
    bench_transform: ViewportTransform | None = None
    equip_transform: ViewportTransform | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.board_quad is not None:
            result["board_quad"] = self.board_quad.to_dict()
        if self.bench_zone is not None:
            result["bench_zone"] = self.bench_zone.to_dict()
        if self.equip_zone is not None:
            result["equip_zone"] = self.equip_zone.to_dict()
        if self.shop_zone is not None:
            result["shop_zone"] = self.shop_zone.to_dict()
        if self.board_transform is not None:
            result["board_transform"] = self.board_transform.to_dict()
        if self.bench_transform is not None:
            result["bench_transform"] = self.bench_transform.to_dict()
        if self.equip_transform is not None:
            result["equip_transform"] = self.equip_transform.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ViewLayoutOverride:
        def _parse_quad(raw: Any) -> PerspectiveQuad | None:
            if raw is None:
                return None
            return raw if isinstance(raw, PerspectiveQuad) else PerspectiveQuad.from_dict(raw)

        def _parse_zone(raw: Any) -> LinearZone | None:
            if raw is None:
                return None
            return raw if isinstance(raw, LinearZone) else LinearZone.from_dict(raw)

        def _parse_tf(raw: Any) -> ViewportTransform | None:
            if raw is None:
                return None
            return raw if isinstance(raw, ViewportTransform) else ViewportTransform.from_dict(raw)

        return cls(
            board_quad=_parse_quad(data.get("board_quad")),
            bench_zone=_parse_zone(data.get("bench_zone")),
            equip_zone=_parse_zone(data.get("equip_zone")),
            shop_zone=_parse_zone(data.get("shop_zone")),
            board_transform=_parse_tf(data.get("board_transform")),
            bench_transform=_parse_tf(data.get("bench_transform")),
            equip_transform=_parse_tf(data.get("equip_transform")),
        )


def _parse_view_overrides(raw: Any) -> dict[str, ViewLayoutOverride]:
    """解析多界面缩放态覆盖映射表。"""
    if not isinstance(raw, Mapping):
        return {}
    parsed: dict[str, ViewLayoutOverride] = {}
    for state_key, val in raw.items():
        if isinstance(val, ViewLayoutOverride):
            parsed[str(state_key)] = val
        elif isinstance(val, Mapping):
            parsed[str(state_key)] = ViewLayoutOverride.from_dict(val)
    return parsed


@dataclass
class LayoutProfile:
    """卫戍协议局内 16:9 归一化布局配置集。"""

    shop_zone: LinearZone
    bench_zone: LinearZone
    equip_zone: LinearZone
    choice_zone: LinearZone
    board_quad: PerspectiveQuad
    buttons: dict[str, NormalizedPoint]
    view_overrides: dict[str, ViewLayoutOverride] = field(default_factory=dict)
    choice_2_zone: LinearZone = field(default_factory=_default_choice_2_zone)
    choice_6_top_zone: LinearZone = field(default_factory=_default_choice_6_top_zone)
    choice_6_bottom_zone: LinearZone = field(default_factory=_default_choice_6_bottom_zone)
    emote_top_zone: LinearZone = field(default_factory=_default_emote_top_zone)
    emote_bottom_zone: LinearZone = field(default_factory=_default_emote_bottom_zone)
    covenant_zone: SymmetricCovenantZone = field(default_factory=SymmetricCovenantZone)
    mirrored_board_quad: PerspectiveQuad = field(default_factory=_default_mirrored_board_quad)
    mirrored_equip_zone: LinearZone = field(default_factory=_default_mirrored_equip_zone)

    def __post_init__(self) -> None:
        missing = [key for key in REQUIRED_BUTTON_KEYS if key not in self.buttons]
        if missing:
            raise ValueError(f"LayoutProfile.buttons 缺少必需按钮锚点: {', '.join(missing)}")

    def _get_view_override(
        self, view_state: LayoutViewState | str
    ) -> ViewLayoutOverride | None:
        key = view_state.value if isinstance(view_state, LayoutViewState) else str(view_state)
        if key == LayoutViewState.NORMAL.value:
            return None
        return self.view_overrides.get(key)

    def resolve_board_quad(
        self,
        view_state: LayoutViewState | str = LayoutViewState.NORMAL,
        *,
        mirrored: bool = False,
    ) -> PerspectiveQuad:
        """按当前镜头缩放态与 Boss 战镜像标志解析生效的棋盘透视四角。

        当 ``mirrored=True``（如 Boss 战被分配至右侧）时，始终以
        ``self.mirrored_board_quad`` 为基准，跳过仅针对非镜像布局的静态
        ``override.board_quad`` 覆盖，直接套用 ``override.board_transform``。
        """
        base_quad = self.mirrored_board_quad if mirrored else self.board_quad
        override = self._get_view_override(view_state)
        if override is None:
            return base_quad
        if not mirrored and override.board_quad is not None:
            return override.board_quad
        if override.board_transform is not None:
            return override.board_transform.apply_quad(base_quad)
        return base_quad

    def resolve_bench_zone(
        self,
        view_state: LayoutViewState | str = LayoutViewState.NORMAL,
        *,
        mirrored: bool = False,
    ) -> LinearZone:
        """按当前镜头缩放态解析生效的整备区线性槽位。

        实机测绘确认 Boss 战左右切换时 10 槽整备区与 2D UI 均不镜像，
        保留 ``mirrored`` 关键字参数以保持调用方接口一致。
        """
        _ = mirrored
        override = self._get_view_override(view_state)
        if override is None:
            return self.bench_zone
        if override.bench_zone is not None:
            return override.bench_zone
        if override.bench_transform is not None:
            return override.bench_transform.apply_zone(self.bench_zone)
        return self.bench_zone

    def resolve_equip_zone(
        self,
        view_state: LayoutViewState | str = LayoutViewState.NORMAL,
        *,
        mirrored: bool = False,
    ) -> LinearZone:
        """按当前镜头缩放态与 Boss 战镜像标志解析生效的装备栏线性槽位。

        当 ``mirrored=True`` 时以 ``self.mirrored_equip_zone`` 为基准，
        忽略仅针对非镜像布局的静态 ``override.equip_zone``，直接套用
        ``override.equip_transform``。
        """
        base_zone = self.mirrored_equip_zone if mirrored else self.equip_zone
        override = self._get_view_override(view_state)
        if override is None:
            return base_zone
        if not mirrored and override.equip_zone is not None:
            return override.equip_zone
        if override.equip_transform is not None:
            return override.equip_transform.apply_zone(base_zone)
        return base_zone

    def resolve_shop_zone(
        self,
        view_state: LayoutViewState | str = LayoutViewState.NORMAL,
        *,
        slot_count: int | None = None,
    ) -> LinearZone:
        """按当前镜头缩放态与商店栏位数解析生效的商店线性槽位。

        当商店升至二本（5 槽）或四本（6 槽）增加栏位时，严格保持卡片尺寸、
        槽位间距与最右侧锚点不变，向左侧等距延伸。
        """
        override = self._get_view_override(view_state)
        base_zone = (
            override.shop_zone
            if (override is not None and override.shop_zone is not None)
            else self.shop_zone
        )
        if slot_count is None:
            return base_zone

        target_count = max(1, int(slot_count))
        if target_count == base_zone.count:
            return base_zone

        if base_zone.count >= 2:
            step_u = (base_zone.end_u - base_zone.start_u) / (base_zone.count - 1)
            step_v = (base_zone.end_v - base_zone.start_v) / (base_zone.count - 1)
        else:
            step_u = base_zone.slot_w
            step_v = 0.0

        new_start_u = max(0.0, min(1.0, base_zone.end_u - step_u * (target_count - 1)))
        new_start_v = max(0.0, min(1.0, base_zone.end_v - step_v * (target_count - 1)))
        return LinearZone(
            name=base_zone.name,
            count=target_count,
            start_u=new_start_u,
            start_v=new_start_v,
            end_u=base_zone.end_u,
            end_v=base_zone.end_v,
            slot_w=base_zone.slot_w,
            slot_h=base_zone.slot_h,
        )

    def resolve_shop_upgrade_point(
        self,
        view_state: LayoutViewState | str = LayoutViewState.NORMAL,
        *,
        slot_count: int | None = None,
    ) -> NormalizedPoint | None:
        """解析随商店栏位向左延伸而同步左移的「升级商店」按钮归一化坐标。"""
        base_pt = self.buttons.get("shop_upgrade")
        if base_pt is None:
            return None
        if slot_count is None:
            return base_pt

        base_zone = self.resolve_shop_zone(view_state, slot_count=None)
        target_count = max(1, int(slot_count))
        delta_slots = target_count - base_zone.count
        if delta_slots == 0:
            return base_pt

        if base_zone.count >= 2:
            step_u = (base_zone.end_u - base_zone.start_u) / (base_zone.count - 1)
            step_v = (base_zone.end_v - base_zone.start_v) / (base_zone.count - 1)
        else:
            step_u = base_zone.slot_w
            step_v = 0.0

        new_u = max(0.0, min(1.0, base_pt.u - step_u * delta_slots))
        new_v = max(0.0, min(1.0, base_pt.v - step_v * delta_slots))
        return NormalizedPoint(u=new_u, v=new_v, anchor=base_pt.anchor)

    @staticmethod
    def _linear_slot_center(zone: LinearZone, index: int, count: int) -> tuple[float, float]:
        n = max(1, int(count))
        idx = max(0, min(n - 1, int(index)))
        if n == 1:
            return (
                (zone.start_u + zone.end_u) * 0.5,
                (zone.start_v + zone.end_v) * 0.5,
            )
        t = idx / (n - 1)
        return (
            float(zone.start_u + t * (zone.end_u - zone.start_u)),
            float(zone.start_v + t * (zone.end_v - zone.start_v)),
        )

    def resolve_choice_card_center(
        self, index: int, choice_count: int = 3
    ) -> tuple[float, float]:
        """解析二选一（1x2）、三选一（1x3）或多人六选一（2x3）第 index 张卡片的归一化中心坐标。"""
        total = max(1, int(choice_count))
        idx = max(0, min(total - 1, int(index)))
        if total == 2:
            return self._linear_slot_center(self.choice_2_zone, idx, 2)
        if total >= 5:
            top_cols = max(1, (total + 1) // 2)
            if idx < top_cols:
                return self._linear_slot_center(self.choice_6_top_zone, idx, top_cols)
            bot_cols = max(1, total - top_cols)
            return self._linear_slot_center(
                self.choice_6_bottom_zone, idx - top_cols, bot_cols
            )
        return self._linear_slot_center(self.choice_zone, idx, total)

    def resolve_choice_card_size(
        self, index: int, choice_count: int = 3
    ) -> tuple[float, float]:
        """解析二选一、三选一或六选一指定卡片的归一化宽高 (slot_w, slot_h)。"""
        total = max(1, int(choice_count))
        idx = max(0, min(total - 1, int(index)))
        if total == 2:
            return (self.choice_2_zone.slot_w, self.choice_2_zone.slot_h)
        if total >= 5:
            top_cols = max(1, (total + 1) // 2)
            zone = (
                self.choice_6_top_zone if idx < top_cols else self.choice_6_bottom_zone
            )
            return (zone.slot_w, zone.slot_h)
        return (self.choice_zone.slot_w, self.choice_zone.slot_h)

    def resolve_choice_confirm_point(
        self, index: int, choice_count: int = 3
    ) -> tuple[float, float]:
        """解析卡片被选中后底部弹出的白色「确认选择」条归一化中心点击坐标。"""
        cu, cv = self.resolve_choice_card_center(index, choice_count)
        _, sh = self.resolve_choice_card_size(index, choice_count)
        confirm_v = max(0.0, min(0.98, cv + sh * 0.505))
        return (float(cu), float(confirm_v))

    def resolve_choice_card_polygon(
        self, index: int, choice_count: int = 3
    ) -> list[tuple[float, float]]:
        """返回未确认阶段第 index 张决策卡片的 4 顶点归一化矩形。"""
        cu, cv = self.resolve_choice_card_center(index, choice_count)
        sw, sh = self.resolve_choice_card_size(index, choice_count)
        hw, hh = sw * 0.5, sh * 0.5
        return [
            (cu - hw, cv - hh),
            (cu + hw, cv - hh),
            (cu + hw, cv + hh),
            (cu - hw, cv + hh),
        ]

    def resolve_choice_confirm_polygon(
        self, index: int, choice_count: int = 3
    ) -> list[tuple[float, float]]:
        """返回卡片选中后底部白色「确认选择」横条的 4 顶点归一化矩形。"""
        cu, confirm_v = self.resolve_choice_confirm_point(index, choice_count)
        sw, _ = self.resolve_choice_card_size(index, choice_count)
        hw = sw * 0.51
        hh = 0.034
        return [
            (cu - hw, confirm_v - hh),
            (cu + hw, confirm_v - hh),
            (cu + hw, confirm_v + hh),
            (cu - hw, confirm_v + hh),
        ]

    def resolve_choice_toggle_point(
        self, choice_count: int = 3
    ) -> NormalizedPoint | None:
        """解析右侧「收起/展开」查看战场按钮坐标（兼容二选一、三选一与多人六选一）。"""
        if int(choice_count) >= 5 and "choice_toggle_eye_6" in self.buttons:
            return self.buttons["choice_toggle_eye_6"]
        return self.buttons.get("choice_toggle_eye")

    def resolve_emote_slot_center(self, index: int) -> tuple[float, float]:
        """解析左下 2x3 表情面板第 index (0..5) 个表情槽位的归一化中心坐标 (u, v)。"""
        idx = max(0, min(5, int(index)))
        if idx < 3:
            return self._linear_slot_center(self.emote_top_zone, idx, 3)
        return self._linear_slot_center(self.emote_bottom_zone, idx - 3, 3)

    def resolve_emote_slot_polygon(self, index: int) -> list[tuple[float, float]]:
        """返回左下 2x3 表情面板第 index (0..5) 个表情槽位的顺时针 4 顶点归一化矩形。"""
        idx = max(0, min(5, int(index)))
        cu, cv = self.resolve_emote_slot_center(idx)
        zone = self.emote_top_zone if idx < 3 else self.emote_bottom_zone
        hw = zone.slot_w * 0.5
        hh = zone.slot_h * 0.5
        return [
            (max(0.0, cu - hw), max(0.0, cv - hh)),
            (min(1.0, cu + hw), max(0.0, cv - hh)),
            (min(1.0, cu + hw), min(1.0, cv + hh)),
            (max(0.0, cu - hw), min(1.0, cv + hh)),
        ]

    def to_dict(self) -> dict[str, Any]:
        """序列化为完整 JSON 字典。"""
        data: dict[str, Any] = {
            "shop_zone": self.shop_zone.to_dict(),
            "bench_zone": self.bench_zone.to_dict(),
            "equip_zone": self.equip_zone.to_dict(),
            "choice_zone": self.choice_zone.to_dict(),
            "choice_2_zone": self.choice_2_zone.to_dict(),
            "choice_6_top_zone": self.choice_6_top_zone.to_dict(),
            "choice_6_bottom_zone": self.choice_6_bottom_zone.to_dict(),
            "emote_top_zone": self.emote_top_zone.to_dict(),
            "emote_bottom_zone": self.emote_bottom_zone.to_dict(),
            "covenant_zone": self.covenant_zone.to_dict(),
            "board_quad": self.board_quad.to_dict(),
            "mirrored_board_quad": self.mirrored_board_quad.to_dict(),
            "mirrored_equip_zone": self.mirrored_equip_zone.to_dict(),
            "buttons": {k: v.to_dict() for k, v in self.buttons.items()},
        }
        if self.view_overrides:
            data["view_overrides"] = {k: v.to_dict() for k, v in self.view_overrides.items()}
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LayoutProfile:
        """从合并后的字典反序列化 LayoutProfile 并校验完整性。

        自动静默忽略旧版残留的局外节点字段，并为新增的策略头像、镜像棋盘/装备区与回合确认按钮提供默认坐标兜底。
        """
        buttons_raw = data.get("buttons", {})
        buttons: dict[str, NormalizedPoint] = {
            k: (v if isinstance(v, NormalizedPoint) else NormalizedPoint.from_dict(v))
            for k, v in buttons_raw.items()
        }
        for key, default_pt in _default_fallback_buttons().items():
            buttons.setdefault(key, default_pt)
        raw_c2 = data.get("choice_2_zone")
        raw_c6_top = data.get("choice_6_top_zone")
        raw_c6_bot = data.get("choice_6_bottom_zone")
        raw_em_top = data.get("emote_top_zone")
        raw_em_bot = data.get("emote_bottom_zone")
        raw_cov = data.get("covenant_zone")
        raw_m_board = data.get("mirrored_board_quad")
        raw_m_equip = data.get("mirrored_equip_zone")
        return cls(
            shop_zone=(
                data["shop_zone"]
                if isinstance(data["shop_zone"], LinearZone)
                else LinearZone.from_dict(data["shop_zone"])
            ),
            bench_zone=(
                data["bench_zone"]
                if isinstance(data["bench_zone"], LinearZone)
                else LinearZone.from_dict(data["bench_zone"])
            ),
            equip_zone=(
                data["equip_zone"]
                if isinstance(data["equip_zone"], LinearZone)
                else LinearZone.from_dict(data["equip_zone"])
            ),
            choice_zone=(
                data["choice_zone"]
                if isinstance(data["choice_zone"], LinearZone)
                else LinearZone.from_dict(data["choice_zone"])
            ),
            board_quad=(
                data["board_quad"]
                if isinstance(data["board_quad"], PerspectiveQuad)
                else PerspectiveQuad.from_dict(data["board_quad"])
            ),
            buttons=buttons,
            view_overrides=_parse_view_overrides(data.get("view_overrides")),
            choice_2_zone=(
                raw_c2
                if isinstance(raw_c2, LinearZone)
                else (
                    LinearZone.from_dict(raw_c2)
                    if isinstance(raw_c2, Mapping)
                    else _default_choice_2_zone()
                )
            ),
            choice_6_top_zone=(
                raw_c6_top
                if isinstance(raw_c6_top, LinearZone)
                else (
                    LinearZone.from_dict(raw_c6_top)
                    if isinstance(raw_c6_top, Mapping)
                    else _default_choice_6_top_zone()
                )
            ),
            choice_6_bottom_zone=(
                raw_c6_bot
                if isinstance(raw_c6_bot, LinearZone)
                else (
                    LinearZone.from_dict(raw_c6_bot)
                    if isinstance(raw_c6_bot, Mapping)
                    else _default_choice_6_bottom_zone()
                )
            ),
            emote_top_zone=(
                raw_em_top
                if isinstance(raw_em_top, LinearZone)
                else (
                    LinearZone.from_dict(raw_em_top)
                    if isinstance(raw_em_top, Mapping)
                    else _default_emote_top_zone()
                )
            ),
            emote_bottom_zone=(
                raw_em_bot
                if isinstance(raw_em_bot, LinearZone)
                else (
                    LinearZone.from_dict(raw_em_bot)
                    if isinstance(raw_em_bot, Mapping)
                    else _default_emote_bottom_zone()
                )
            ),
            covenant_zone=(
                raw_cov
                if isinstance(raw_cov, SymmetricCovenantZone)
                else (
                    SymmetricCovenantZone.from_dict(raw_cov)
                    if isinstance(raw_cov, Mapping)
                    else SymmetricCovenantZone()
                )
            ),
            mirrored_board_quad=(
                raw_m_board
                if isinstance(raw_m_board, PerspectiveQuad)
                else (
                    PerspectiveQuad.from_dict(raw_m_board)
                    if isinstance(raw_m_board, Mapping)
                    else _default_mirrored_board_quad()
                )
            ),
            mirrored_equip_zone=(
                raw_m_equip
                if isinstance(raw_m_equip, LinearZone)
                else (
                    LinearZone.from_dict(raw_m_equip)
                    if isinstance(raw_m_equip, Mapping)
                    else _default_mirrored_equip_zone()
                )
            ),
        )


@dataclass
class KeymapConfig:
    """手柄手感参数、快捷键开关与自定义键位映射配置。"""

    left_deadzone: float = 0.14
    right_deadzone: float = 0.16
    cursor_max_speed_px: float = 1400.0
    cursor_gamma: float = 2.2
    ready_hold_ms: int = 350
    repeat_initial_ms: int = 260
    repeat_interval_ms: int = 95
    use_native_keyboard_shortcuts: bool = True
    flick_distance_ratio: float = 0.09
    vibration_enabled: bool = True
    trigger_threshold: float = 0.45
    scroll_speed: float = 1.0
    allow_retreat_chord: bool = True
    left_stick_mode: str = "dpad"
    left_stick_dpad_threshold: float = 0.50
    left_stick_dpad_release_threshold: float = 0.30
    auto_toggle_shop: bool = True
    hide_hud_from_capture: bool = True
    keyboard_shortcuts: dict[str, str] = field(default_factory=_default_keyboard_shortcuts)
    bindings: dict[str, dict[str, str]] = field(default_factory=_default_bindings)

    def __post_init__(self) -> None:
        self.left_deadzone = float(self.left_deadzone)
        self.right_deadzone = float(self.right_deadzone)
        self.cursor_max_speed_px = float(self.cursor_max_speed_px)
        self.cursor_gamma = float(self.cursor_gamma)
        self.ready_hold_ms = int(self.ready_hold_ms)
        self.repeat_initial_ms = int(self.repeat_initial_ms)
        self.repeat_interval_ms = int(self.repeat_interval_ms)
        self.use_native_keyboard_shortcuts = bool(self.use_native_keyboard_shortcuts)
        self.flick_distance_ratio = float(self.flick_distance_ratio)
        self.vibration_enabled = bool(self.vibration_enabled)
        self.trigger_threshold = float(self.trigger_threshold)
        self.scroll_speed = float(self.scroll_speed)
        self.allow_retreat_chord = bool(self.allow_retreat_chord)
        self.left_stick_mode = str(self.left_stick_mode).strip().lower()
        self.left_stick_dpad_threshold = float(self.left_stick_dpad_threshold)
        self.left_stick_dpad_release_threshold = float(self.left_stick_dpad_release_threshold)
        self.auto_toggle_shop = bool(self.auto_toggle_shop)
        self.hide_hud_from_capture = bool(self.hide_hud_from_capture)

        if not (0.0 <= self.left_deadzone < 0.95):
            raise ValueError(f"left_deadzone 必须位于 [0.0, 0.95) 范围内，当前值: {self.left_deadzone}")
        if not (0.0 <= self.right_deadzone < 0.95):
            raise ValueError(f"right_deadzone 必须位于 [0.0, 0.95) 范围内，当前值: {self.right_deadzone}")
        if self.cursor_max_speed_px <= 0.0:
            raise ValueError(f"cursor_max_speed_px 必须大于 0，当前值: {self.cursor_max_speed_px}")
        if self.cursor_gamma <= 0.0:
            raise ValueError(f"cursor_gamma 必须大于 0，当前值: {self.cursor_gamma}")
        if self.ready_hold_ms < 0:
            raise ValueError(f"ready_hold_ms 不可为负数，当前值: {self.ready_hold_ms}")
        if self.repeat_initial_ms < 0 or self.repeat_interval_ms <= 0:
            raise ValueError(
                f"连发时间参数非法: repeat_initial_ms={self.repeat_initial_ms}, "
                f"repeat_interval_ms={self.repeat_interval_ms}"
            )
        if not (0.0 < self.flick_distance_ratio <= 0.5):
            raise ValueError(
                f"flick_distance_ratio 必须位于 (0.0, 0.5] 范围内，当前值: {self.flick_distance_ratio}"
            )
        if not (0.0 < self.trigger_threshold <= 1.0):
            raise ValueError(
                f"trigger_threshold 必须位于 (0.0, 1.0] 范围内，当前值: {self.trigger_threshold}"
            )
        if self.scroll_speed <= 0.0:
            raise ValueError(f"scroll_speed 必须大于 0，当前值: {self.scroll_speed}")
        if self.left_stick_mode not in VALID_LEFT_STICK_MODES:
            raise ValueError(
                f"无效的 left_stick_mode '{self.left_stick_mode}'，允许值: {', '.join(VALID_LEFT_STICK_MODES)}"
            )
        if not (0.0 < self.left_stick_dpad_threshold <= 1.0):
            raise ValueError(
                f"left_stick_dpad_threshold 必须位于 (0.0, 1.0] 范围内，当前值: {self.left_stick_dpad_threshold}"
            )
        if not (0.0 <= self.left_stick_dpad_release_threshold < self.left_stick_dpad_threshold):
            raise ValueError(
                f"left_stick_dpad_release_threshold 必须位于 [0.0, {self.left_stick_dpad_threshold}) 范围内，"
                f"当前值: {self.left_stick_dpad_release_threshold}"
            )

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 字典。"""
        return {
            "left_deadzone": self.left_deadzone,
            "right_deadzone": self.right_deadzone,
            "cursor_max_speed_px": self.cursor_max_speed_px,
            "cursor_gamma": self.cursor_gamma,
            "ready_hold_ms": self.ready_hold_ms,
            "repeat_initial_ms": self.repeat_initial_ms,
            "repeat_interval_ms": self.repeat_interval_ms,
            "use_native_keyboard_shortcuts": self.use_native_keyboard_shortcuts,
            "flick_distance_ratio": self.flick_distance_ratio,
            "vibration_enabled": self.vibration_enabled,
            "trigger_threshold": self.trigger_threshold,
            "scroll_speed": self.scroll_speed,
            "allow_retreat_chord": self.allow_retreat_chord,
            "left_stick_mode": self.left_stick_mode,
            "left_stick_dpad_threshold": self.left_stick_dpad_threshold,
            "left_stick_dpad_release_threshold": self.left_stick_dpad_release_threshold,
            "auto_toggle_shop": self.auto_toggle_shop,
            "hide_hud_from_capture": self.hide_hud_from_capture,
            "keyboard_shortcuts": dict(self.keyboard_shortcuts),
            "bindings": {section: dict(items) for section, items in self.bindings.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KeymapConfig:
        """从字典反序列化 KeymapConfig，缺失字段回退为默认值。"""
        shortcuts = _default_keyboard_shortcuts()
        if isinstance(data.get("keyboard_shortcuts"), Mapping):
            shortcuts.update({str(k): str(v) for k, v in data["keyboard_shortcuts"].items()})

        bindings = _default_bindings()
        if isinstance(data.get("bindings"), Mapping):
            for section, mapping in data["bindings"].items():
                if isinstance(mapping, Mapping):
                    merged = dict(bindings.get(str(section), {}))
                    merged.update({str(k): str(v) for k, v in mapping.items()})
                    bindings[str(section)] = merged

        return cls(
            left_deadzone=float(data.get("left_deadzone", 0.14)),
            right_deadzone=float(data.get("right_deadzone", 0.16)),
            cursor_max_speed_px=float(data.get("cursor_max_speed_px", 1400.0)),
            cursor_gamma=float(data.get("cursor_gamma", 2.2)),
            ready_hold_ms=int(data.get("ready_hold_ms", 350)),
            repeat_initial_ms=int(data.get("repeat_initial_ms", 260)),
            repeat_interval_ms=int(data.get("repeat_interval_ms", 95)),
            use_native_keyboard_shortcuts=bool(data.get("use_native_keyboard_shortcuts", True)),
            flick_distance_ratio=float(data.get("flick_distance_ratio", 0.09)),
            vibration_enabled=bool(data.get("vibration_enabled", True)),
            trigger_threshold=float(data.get("trigger_threshold", 0.45)),
            scroll_speed=float(data.get("scroll_speed", 1.0)),
            allow_retreat_chord=bool(data.get("allow_retreat_chord", True)),
            left_stick_mode=str(data.get("left_stick_mode", "dpad")),
            left_stick_dpad_threshold=float(data.get("left_stick_dpad_threshold", 0.50)),
            left_stick_dpad_release_threshold=float(
                data.get("left_stick_dpad_release_threshold", 0.30)
            ),
            auto_toggle_shop=bool(data.get("auto_toggle_shop", True)),
            hide_hud_from_capture=bool(data.get("hide_hud_from_capture", True)),
            keyboard_shortcuts=shortcuts,
            bindings=bindings,
        )
