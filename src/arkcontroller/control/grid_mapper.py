"""槽位线性映射、界面节点图导航与 4 点透视棋盘单应性映射器。"""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np

from arkcontroller.config.models import LinearZone, NavNode, PerspectiveQuad

Direction = Literal["up", "down", "left", "right"]

_DIRECTION_VECTORS: dict[Direction, tuple[float, float]] = {
    "up": (0.0, -1.0),
    "down": (0.0, 1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
}


def _rect_polygon(cu: float, cv: float, w: float, h: float) -> list[tuple[float, float]]:
    """根据中心点与宽高生成顺时针 4 顶点归一化多边形 [左上, 右上, 右下, 左下]。"""
    hw = w * 0.5
    hh = h * 0.5
    return [
        (cu - hw, cv - hh),
        (cu + hw, cv - hh),
        (cu + hw, cv + hh),
        (cu - hw, cv + hh),
    ]


class LinearZoneMapper:
    """线性槽位区域映射器（适用于商店、整备区 10 槽、侧边装备栏 5 槽及三选一卡片）。"""

    def __init__(self, zone: LinearZone) -> None:
        if zone.count < 1:
            raise ValueError(f"LinearZone '{zone.name}' 的槽位数量必须 >= 1，当前为 {zone.count}")
        self.zone = zone

    @property
    def count(self) -> int:
        return self.zone.count

    @property
    def is_vertical(self) -> bool:
        """判断该线性区域是否为纵向排列（例如侧边装备栏）。"""
        return abs(self.zone.end_v - self.zone.start_v) > abs(
            self.zone.end_u - self.zone.start_u
        )

    def _effective_count(self, count: int | None) -> int:
        n = self.zone.count if count is None else count
        if n < 1:
            raise ValueError(f"有效槽位数量必须 >= 1，当前为 {n}")
        return n

    def clamp_index(self, index: int, count: int | None = None) -> int:
        """将索引安全裁剪至 [0, count - 1]。"""
        n = self._effective_count(count)
        return max(0, min(n - 1, index))

    def slot_center(self, index: int, count: int | None = None) -> tuple[float, float]:
        """计算第 index 个槽位的归一化中心坐标 (u, v)。"""
        n = self._effective_count(count)
        if index < 0 or index >= n:
            raise IndexError(
                f"槽位索引 {index} 超出区域 '{self.zone.name}' 有效范围 [0, {n - 1}]"
            )
        if n == 1:
            return (
                (self.zone.start_u + self.zone.end_u) * 0.5,
                (self.zone.start_v + self.zone.end_v) * 0.5,
            )
        t = index / (n - 1)
        u = self.zone.start_u + t * (self.zone.end_u - self.zone.start_u)
        v = self.zone.start_v + t * (self.zone.end_v - self.zone.start_v)
        return (float(u), float(v))

    def slot_polygon(
        self, index: int, count: int | None = None
    ) -> list[tuple[float, float]]:
        """返回第 index 个槽位的 4 个归一化顶点 [左上, 右上, 右下, 左下]。"""
        cu, cv = self.slot_center(index, count=count)
        return _rect_polygon(cu, cv, self.zone.slot_w, self.zone.slot_h)

    def step(
        self,
        index: int,
        delta: int,
        *,
        wrap: bool = True,
        count: int | None = None,
    ) -> tuple[int, bool]:
        """按步长移动槽位索引，返回 (新索引, 是否触达边界)。"""
        n = self._effective_count(count)
        cur = self.clamp_index(index, count=n)
        target = cur + delta
        if 0 <= target < n:
            return (target, False)
        if wrap:
            return (target % n, True)
        return (self.clamp_index(target, count=n), True)

    def move(
        self,
        index: int,
        direction: Direction,
        *,
        wrap: bool = True,
        count: int | None = None,
    ) -> tuple[int, Direction | None]:
        """根据十字键方向在区域内移动或触发边界跨区。

        返回值 ``(new_index, exit_direction)`` 中，若 ``exit_direction`` 非 ``None``，
        表示触发了向该方向的跨区请求，由上层调度器切换至相邻焦点区域。
        """
        n = self._effective_count(count)
        cur = self.clamp_index(index, count=n)

        if not self.is_vertical:
            if direction in ("up", "down"):
                return (cur, direction)
            axis_sign = 1 if self.zone.end_u >= self.zone.start_u else -1
            delta = axis_sign if direction == "right" else -axis_sign
        else:
            if direction in ("left", "right"):
                return (cur, direction)
            axis_sign = 1 if self.zone.end_v >= self.zone.start_v else -1
            delta = axis_sign if direction == "down" else -axis_sign

        target = cur + delta
        if 0 <= target < n:
            return (target, None)
        if wrap:
            return (target % n, None)
        return (cur, direction)

    def nearest_slot(
        self, u: float, v: float, count: int | None = None
    ) -> int:
        """寻找距离归一化坐标 (u, v) 最近的槽位索引，用于跨区平滑衔接。"""
        n = self._effective_count(count)
        best_idx = 0
        best_dist = float("inf")
        for idx in range(n):
            su, sv = self.slot_center(idx, count=n)
            dist = math.hypot(su - u, sv - v)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx


class NodeGraphMapper:
    """不规则 UI 界面节点图导航器（活动主界面、模式配置、战前编队、结算页）。"""

    def __init__(
        self,
        nodes: dict[str, NavNode],
        *,
        cone_cos_threshold: float = 0.5,
    ) -> None:
        self.nodes: dict[str, NavNode] = dict(nodes)
        self.cone_cos_threshold = cone_cos_threshold

    @property
    def node_ids(self) -> list[str]:
        return list(self.nodes.keys())

    def default_node_id(self) -> str | None:
        """返回默认起始焦点节点 ID（若有非返回按钮则优先跳过左上角 back 节点）。"""
        first_id: str | None = None
        for node_id in self.nodes:
            if first_id is None:
                first_id = node_id
            if node_id.lower() != "back":
                return node_id
        return first_id

    def get_node(self, node_id: str) -> NavNode | None:
        return self.nodes.get(node_id)

    def node_center(self, node_id: str) -> tuple[float, float]:
        """返回指定节点的归一化中心坐标 (u, v)。"""
        node = self.nodes[node_id]
        return (float(node.u), float(node.v))

    def node_polygon(self, node_id: str) -> list[tuple[float, float]]:
        """返回指定节点的 4 顶点归一化多边形。"""
        node = self.nodes[node_id]
        return _rect_polygon(node.u, node.v, node.w, node.h)

    def navigate(self, current_id: str, direction: Direction) -> str:
        """从 current_id 沿 direction 跳转至下一节点。

        优先使用节点显式配置的邻接 ID；若为 None，则在目标方向锥内按加权距离
        自动搜索最符合直觉的最近邻节点。
        """
        current = self.nodes.get(current_id)
        if current is None:
            fallback = self.default_node_id()
            return fallback if fallback is not None else current_id

        explicit_target = getattr(current, direction, None)
        if explicit_target is not None and explicit_target in self.nodes:
            return explicit_target

        ux, uy = _DIRECTION_VECTORS[direction]
        best_id: str | None = None
        best_score = float("inf")

        for cand in self.nodes.values():
            if cand.id == current.id:
                continue
            dx = cand.u - current.u
            dy = cand.v - current.v
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue

            forward = dx * ux + dy * uy
            if forward <= 1e-5:
                continue

            cos_theta = forward / dist
            if cos_theta < self.cone_cos_threshold:
                continue

            # 正交偏移权重更高，确保优先选中与行进轴线对齐的按钮节点
            perp = abs(dx * (-uy) + dy * ux)
            score = forward + 2.0 * perp
            if score < best_score:
                best_score = score
                best_id = cand.id

        return best_id if best_id is not None else current_id

    def cycle(self, current_id: str, delta: int) -> str:
        """按节点定义顺序循环切换节点（常用于 LB/RB 切页或切换按钮）。"""
        ids = self.node_ids
        if not ids:
            return current_id
        if current_id not in self.nodes:
            return ids[0]
        idx = ids.index(current_id)
        return ids[(idx + delta) % len(ids)]

    def nearest_node(self, u: float, v: float) -> str | None:
        """查找距离指定归一化坐标最近的节点 ID。"""
        best_id: str | None = None
        best_dist = float("inf")
        for node_id, node in self.nodes.items():
            dist = math.hypot(node.u - u, node.v - v)
            if dist < best_dist:
                best_dist = dist
                best_id = node_id
        return best_id

    def find_by_keywords(self, keywords: tuple[str, ...]) -> str | None:
        """按节点 ID 或中文标签关键词查找目标快捷节点。"""
        lowered = tuple(k.lower() for k in keywords)
        for node_id, node in self.nodes.items():
            haystack = f"{node_id.lower()} {node.label.lower()}"
            if any(k in haystack for k in lowered):
                return node_id
        return None


class PerspectiveBoardMapper:
    """基于 4 点单应性变换的 3D 俯视透视棋盘网格映射器。"""

    def __init__(self, quad: PerspectiveQuad) -> None:
        if quad.rows < 1 or quad.cols < 1:
            raise ValueError(
                f"棋盘行列数必须 >= 1，当前为 rows={quad.rows}, cols={quad.cols}"
            )
        self.quad = quad

        pts = [
            quad.top_left,
            quad.top_right,
            quad.bottom_right,
            quad.bottom_left,
        ]
        # 校验四角构成非退化凸四边形（拒绝共线、凹多边形或自相交蝴蝶结形）
        crosses: list[float] = []
        for i in range(4):
            p0 = pts[i]
            p1 = pts[(i + 1) % 4]
            p2 = pts[(i + 2) % 4]
            e1x, e1y = p1[0] - p0[0], p1[1] - p0[1]
            e2x, e2y = p2[0] - p1[0], p2[1] - p1[1]
            crosses.append(e1x * e2y - e1y * e2x)
        if not (
            all(c > 1e-9 for c in crosses) or all(c < -1e-9 for c in crosses)
        ):
            raise ValueError("PerspectiveQuad 四顶点共线、凹陷或自相交，无法构造有效透视棋盘")

        src = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        dst = np.array(pts, dtype=np.float32)

        self.H: np.ndarray = cv2.getPerspectiveTransform(src, dst)
        det = float(np.linalg.det(self.H))
        if not math.isfinite(det) or abs(det) < 1e-12:
            raise ValueError("PerspectiveQuad 四顶点退化或共线，无法构造可逆单应性矩阵")
        self.H_inv: np.ndarray = cv2.getPerspectiveTransform(dst, src)

    @property
    def rows(self) -> int:
        return self.quad.rows

    @property
    def cols(self) -> int:
        return self.quad.cols

    @staticmethod
    def _project(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
        """对单个二维点执行 3x3 单应性齐次投影变换。"""
        vec = matrix @ np.array([x, y, 1.0], dtype=np.float64)
        w = float(vec[2])
        if abs(w) < 1e-12:
            raise ValueError("单应性投影分母趋近于零")
        return (float(vec[0] / w), float(vec[1] / w))

    def unit_to_norm(self, s: float, t: float) -> tuple[float, float]:
        """将棋盘单位正方形坐标 (s, t) ∈ [0, 1]^2 投影到屏幕归一化坐标 (u, v)。"""
        return self._project(self.H, s, t)

    def norm_to_unit(self, u: float, v: float) -> tuple[float, float]:
        """将屏幕归一化坐标 (u, v) 逆投影回棋盘单位正方形坐标 (s, t)。"""
        return self._project(self.H_inv, u, v)

    def _validate_cell(self, row: int, col: int) -> None:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError(
                f"棋盘格 ({row}, {col}) 超出有效范围 rows=[0,{self.rows - 1}], cols=[0,{self.cols - 1}]"
            )

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        """计算棋盘第 (row, col) 格的透视中心归一化坐标 (u, v)。"""
        self._validate_cell(row, col)
        s = (col + 0.5) / self.cols
        t = (row + 0.5) / self.rows
        return self.unit_to_norm(s, t)

    def cell_polygon(self, row: int, col: int) -> list[tuple[float, float]]:
        """返回棋盘第 (row, col) 格经透视变换后的 4 个归一化顶点 [左上, 右上, 右下, 左下]。"""
        self._validate_cell(row, col)
        s0 = col / self.cols
        s1 = (col + 1) / self.cols
        t0 = row / self.rows
        t1 = (row + 1) / self.rows
        return [
            self.unit_to_norm(s0, t0),
            self.unit_to_norm(s1, t0),
            self.unit_to_norm(s1, t1),
            self.unit_to_norm(s0, t1),
        ]

    def norm_to_cell(self, u: float, v: float) -> tuple[int, int] | None:
        """将屏幕归一化坐标 (u, v) 反算至所属棋盘格 (row, col)；若落在棋盘外则返回 None。"""
        s, t = self.norm_to_unit(u, v)
        eps = 1e-9
        if s < -eps or s > 1.0 + eps or t < -eps or t > 1.0 + eps:
            return None
        s_clamped = max(0.0, min(1.0, s))
        t_clamped = max(0.0, min(1.0, t))
        col = min(self.cols - 1, int(s_clamped * self.cols))
        row = min(self.rows - 1, int(t_clamped * self.rows))
        return (row, col)

    def nearest_cell(self, u: float, v: float) -> tuple[int, int]:
        """将任意归一化坐标投影并裁剪到最近的棋盘格 (row, col)，用于跨区焦点衔接。"""
        s, t = self.norm_to_unit(u, v)
        s_clamped = max(0.0, min(1.0, s))
        t_clamped = max(0.0, min(1.0, t))
        col = min(self.cols - 1, max(0, int(s_clamped * self.cols)))
        row = min(self.rows - 1, max(0, int(t_clamped * self.rows)))
        return (row, col)

    def move(
        self,
        row: int,
        col: int,
        direction: Direction,
        *,
        wrap_cols: bool = False,
    ) -> tuple[int, int, Direction | None]:
        """在棋盘内按方向移动，若越出边界则返回退出方向以触发跨区切换。"""
        r = max(0, min(self.rows - 1, row))
        c = max(0, min(self.cols - 1, col))

        if direction == "up":
            if r > 0:
                return (r - 1, c, None)
            return (r, c, "up")
        if direction == "down":
            if r < self.rows - 1:
                return (r + 1, c, None)
            return (r, c, "down")
        if direction == "left":
            if c > 0:
                return (r, c - 1, None)
            if wrap_cols:
                return (r, self.cols - 1, None)
            return (r, c, "left")
        # direction == "right"
        if c < self.cols - 1:
            return (r, c + 1, None)
        if wrap_cols:
            return (r, 0, None)
        return (r, c, "right")

    def grid_lines(
        self,
    ) -> tuple[
        list[tuple[tuple[float, float], tuple[float, float]]],
        list[tuple[tuple[float, float], tuple[float, float]]],
    ]:
        """生成所有透视网格的横向与纵向分割线段，便于校准面板与 HUD 预览绘制。"""
        horiz: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for r in range(self.rows + 1):
            t = r / self.rows
            horiz.append((self.unit_to_norm(0.0, t), self.unit_to_norm(1.0, t)))

        vert: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for c in range(self.cols + 1):
            s = c / self.cols
            vert.append((self.unit_to_norm(s, 0.0), self.unit_to_norm(s, 1.0)))
        return (horiz, vert)
