"""ArkController 配置数据模型定义与合法性校验。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

VALID_ANCHORS: tuple[str, ...] = (
    "center_16_9",
    "left_edge",
    "right_edge",
    "top_edge",
    "bottom_edge",
    "client",
)

REQUIRED_BUTTON_KEYS: tuple[str, ...] = (
    "back",
    "start_action",
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
    "squad_skill_1",
    "squad_skill_2",
    "squad_skill_3",
    "squad_module",
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
    """返回《明日方舟》官方 PC 客户端卫戍协议默认快捷键映射。"""
    return {
        "shop_refresh": "d",
        "shop_freeze": "r",
        "battle_ready": "space",
        "sell_board": "e",
        "sell_bench": "q",
        "retreat_unit": "a",
        "covenant_panel": "grave",
        "enemy_preview": "tab",
        "emote_wheel": "z",
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
        },
        "menu": {
            "confirm": "A",
            "back": "B",
            "quick_squad": "X",
            "quick_rewards": "Y",
            "prev_tab": "LB",
            "next_tab": "RB",
            "start_action": "START",
        },
        "battle_prep": {
            "select_or_grab": "A",
            "cancel": "B",
            "shop_refresh": "X",
            "shop_upgrade": "Y",
            "shop_freeze": "RS",
            "fast_buy": "RT+A",
            "prev_zone": "LB",
            "next_zone": "RB",
            "battle_ready_hold": "START",
        },
        "holding": {
            "place_or_confirm": "A",
            "cancel": "B",
            "quick_sell": "X",
            "gift_teammate": "Y",
            "toggle_bench_board": "LB/RB",
        },
        "modal_choice": {
            "confirm_choice": "A",
            "fast_confirm_choice": "RT+A",
            "peek_board_hold": "LT",
        },
        "lt_chords": {
            "covenant_panel": "LT+DPAD_LEFT",
            "enemy_preview": "LT+DPAD_UP",
            "emote_wheel": "LT+DPAD_DOWN",
            "cycle_teammate": "LT+DPAD_RIGHT",
            "retreat_unit": "LT+RB",
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


def _parse_node_map(raw: Mapping[str, Any]) -> dict[str, NavNode]:
    """解析导航节点映射表。"""
    nodes: dict[str, NavNode] = {}
    for key, val in raw.items():
        if isinstance(val, NavNode):
            nodes[key] = val
        elif isinstance(val, Mapping):
            node = NavNode.from_dict(val, node_id=key)
            nodes[node.id] = node
        else:
            raise TypeError(f"导航节点 '{key}' 类型错误: {type(val).__name__}")
    return nodes


@dataclass
class LayoutProfile:
    """卫戍协议全流程 16:9 归一化布局配置集。"""

    activity_hub_nodes: dict[str, NavNode]
    pre_match_nodes: dict[str, NavNode]
    squad_edit_nodes: dict[str, NavNode]
    settlement_nodes: dict[str, NavNode]
    shop_zone: LinearZone
    bench_zone: LinearZone
    equip_zone: LinearZone
    choice_zone: LinearZone
    board_quad: PerspectiveQuad
    buttons: dict[str, NormalizedPoint]

    def __post_init__(self) -> None:
        missing = [key for key in REQUIRED_BUTTON_KEYS if key not in self.buttons]
        if missing:
            raise ValueError(f"LayoutProfile.buttons 缺少必需按钮锚点: {', '.join(missing)}")

    def to_dict(self) -> dict[str, Any]:
        """序列化为完整 JSON 字典。"""
        return {
            "activity_hub_nodes": {k: v.to_dict() for k, v in self.activity_hub_nodes.items()},
            "pre_match_nodes": {k: v.to_dict() for k, v in self.pre_match_nodes.items()},
            "squad_edit_nodes": {k: v.to_dict() for k, v in self.squad_edit_nodes.items()},
            "settlement_nodes": {k: v.to_dict() for k, v in self.settlement_nodes.items()},
            "shop_zone": self.shop_zone.to_dict(),
            "bench_zone": self.bench_zone.to_dict(),
            "equip_zone": self.equip_zone.to_dict(),
            "choice_zone": self.choice_zone.to_dict(),
            "board_quad": self.board_quad.to_dict(),
            "buttons": {k: v.to_dict() for k, v in self.buttons.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LayoutProfile:
        """从合并后的字典反序列化 LayoutProfile 并校验完整性。"""
        buttons_raw = data.get("buttons", {})
        buttons: dict[str, NormalizedPoint] = {
            k: (v if isinstance(v, NormalizedPoint) else NormalizedPoint.from_dict(v))
            for k, v in buttons_raw.items()
        }
        return cls(
            activity_hub_nodes=_parse_node_map(data["activity_hub_nodes"]),
            pre_match_nodes=_parse_node_map(data["pre_match_nodes"]),
            squad_edit_nodes=_parse_node_map(data["squad_edit_nodes"]),
            settlement_nodes=_parse_node_map(data["settlement_nodes"]),
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
            keyboard_shortcuts=shortcuts,
            bindings=bindings,
        )
