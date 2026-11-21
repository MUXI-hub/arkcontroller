"""单元测试：线性槽位映射、界面节点图导航与 4 点透视棋盘单应性变换。"""

from __future__ import annotations

import numpy as np
import pytest

from arkcontroller.config.loader import ConfigLoader
from arkcontroller.config.models import LinearZone, NavNode, PerspectiveQuad
from arkcontroller.control.grid_mapper import (
    LinearZoneMapper,
    NodeGraphMapper,
    PerspectiveBoardMapper,
)


class TestLinearZoneMapper:
    """验证商店、整备区、装备栏与三选一线性槽位映射及边界跨区行为。"""

    def test_horizontal_zone_centers_and_polygons(self) -> None:
        zone = LinearZone(
            name="bench",
            count=10,
            start_u=0.10,
            start_v=0.90,
            end_u=0.90,
            end_v=0.90,
            slot_w=0.07,
            slot_h=0.12,
        )
        mapper = LinearZoneMapper(zone)
        assert not mapper.is_vertical
        assert mapper.count == 10

        assert mapper.slot_center(0) == pytest.approx((0.10, 0.90))
        assert mapper.slot_center(9) == pytest.approx((0.90, 0.90))
        # 中间槽位线性等距
        u4, v4 = mapper.slot_center(4)
        u5, v5 = mapper.slot_center(5)
        assert (u4 + u5) * 0.5 == pytest.approx(0.50)
        assert v4 == pytest.approx(0.90)
        assert v5 == pytest.approx(0.90)

        poly = mapper.slot_polygon(0)
        assert len(poly) == 4
        assert poly[1][0] - poly[0][0] == pytest.approx(0.07)
        assert poly[2][1] - poly[1][1] == pytest.approx(0.12)

        with pytest.raises(IndexError):
            mapper.slot_center(-1)
        with pytest.raises(IndexError):
            mapper.slot_center(10)

    def test_dynamic_count_override_for_modal_choice(self) -> None:
        zone = LinearZone(
            name="choice",
            count=3,
            start_u=0.25,
            start_v=0.50,
            end_u=0.75,
            end_v=0.50,
            slot_w=0.20,
            slot_h=0.40,
        )
        mapper = LinearZoneMapper(zone)
        assert mapper.slot_center(0, count=1) == pytest.approx((0.50, 0.50))
        assert mapper.slot_center(0, count=2) == pytest.approx((0.25, 0.50))
        assert mapper.slot_center(1, count=2) == pytest.approx((0.75, 0.50))
        assert mapper.slot_center(1, count=3) == pytest.approx((0.50, 0.50))

    def test_horizontal_wrap_and_boundary_cross_zone(self) -> None:
        zone = LinearZone(
            name="shop",
            count=5,
            start_u=0.20,
            start_v=0.20,
            end_u=0.80,
            end_v=0.20,
            slot_w=0.10,
            slot_h=0.18,
        )
        mapper = LinearZoneMapper(zone)

        # 左右循环模式
        assert mapper.move(0, "left", wrap=True) == (4, None)
        assert mapper.move(4, "right", wrap=True) == (0, None)
        assert mapper.move(1, "right", wrap=True) == (2, None)

        # 左右不循环模式触发跨区退出信号
        assert mapper.move(0, "left", wrap=False) == (0, "left")
        assert mapper.move(4, "right", wrap=False) == (4, "right")

        # 横向区域的上下方向直接触发跨区
        assert mapper.move(2, "up", wrap=True) == (2, "up")
        assert mapper.move(2, "down", wrap=True) == (2, "down")

        # 最近槽位吸附
        assert mapper.nearest_slot(0.51, 0.85) == 2
        assert mapper.nearest_slot(0.18, 0.20) == 0

    def test_vertical_equipment_zone_navigation(self) -> None:
        zone = LinearZone(
            name="equipment",
            count=5,
            start_u=0.945,
            start_v=0.28,
            end_u=0.945,
            end_v=0.68,
            slot_w=0.068,
            slot_h=0.088,
        )
        mapper = LinearZoneMapper(zone)
        assert mapper.is_vertical

        assert mapper.move(0, "down", wrap=True) == (1, None)
        assert mapper.move(0, "up", wrap=True) == (4, None)
        assert mapper.move(4, "down", wrap=False) == (4, "down")
        # 纵向区域按左右直接触发跨回棋盘
        assert mapper.move(2, "left", wrap=True) == (2, "left")
        assert mapper.move(2, "right", wrap=True) == (2, "right")
        assert mapper.nearest_slot(0.90, 0.48) == 2


class TestNodeGraphMapper:
    """验证显式邻接与方向锥最近邻节点图导航。"""

    def test_explicit_and_cone_fallback_navigation(self) -> None:
        nodes = {
            "back": NavNode(id="back", label="返回", u=0.05, v=0.06),
            "top_left": NavNode(
                id="top_left", label="左上", u=0.20, v=0.30, right="custom_jump"
            ),
            "top_right": NavNode(id="top_right", label="右上", u=0.80, v=0.30),
            "bottom_left": NavNode(id="bottom_left", label="左下", u=0.20, v=0.75),
            "bottom_right": NavNode(id="bottom_right", label="右下", u=0.82, v=0.75),
            "custom_jump": NavNode(id="custom_jump", label="特指", u=0.50, v=0.52),
        }
        mapper = NodeGraphMapper(nodes)

        # 默认起始节点应跳过 back
        assert mapper.default_node_id() == "top_left"

        # 显式 right 优先于几何最近邻
        assert mapper.navigate("top_left", "right") == "custom_jump"

        # 显式属性为 None 时，自动按方向锥查找最近邻（同行精准对齐优先于斜向节点）
        assert mapper.navigate("top_left", "down") == "bottom_left"
        assert mapper.navigate("bottom_left", "right") == "bottom_right"
        assert mapper.navigate("top_right", "left") == "top_left"

        # 方向锥内的斜向节点在无同行节点时可被准确搜寻
        assert mapper.navigate("custom_jump", "right") in ("top_right", "bottom_right")
        assert mapper.navigate("custom_jump", "left") in ("top_left", "bottom_left")

        # 右上角继续按右：虽然 bottom_right 的 u=0.82 略大于 0.80，
        # 但垂直偏移过大落在 60° 方向锥之外，不应误跳到右下角
        assert mapper.navigate("top_right", "right") == "top_right"
        assert mapper.navigate("top_right", "down") == "bottom_right"

        # 关键词搜索、循环切节点与最近节点吸附
        assert mapper.find_by_keywords(("特指",)) == "custom_jump"
        assert mapper.cycle("top_left", 1) == "top_right"
        assert mapper.nearest_node(0.81, 0.74) == "bottom_right"
        assert len(mapper.node_polygon("top_left")) == 4

    def test_default_layout_menus_are_fully_navigable(self) -> None:
        _, layout = ConfigLoader(use_user_override=False).load()
        for node_map in (
            layout.activity_hub_nodes,
            layout.pre_match_nodes,
            layout.squad_edit_nodes,
            layout.settlement_nodes,
        ):
            mapper = NodeGraphMapper(node_map)
            start = mapper.default_node_id()
            assert start is not None
            for direction in ("up", "down", "left", "right"):
                target = mapper.navigate(start, direction)
                assert target in node_map


class TestPerspectiveBoardMapper:
    """验证 4 点透视棋盘单应性矩阵正逆变换、近大远小几何特征与退化校验。"""

    def test_rectangular_quad_exact_centers(self) -> None:
        quad = PerspectiveQuad(
            top_left=(0.20, 0.20),
            top_right=(0.80, 0.20),
            bottom_right=(0.80, 0.70),
            bottom_left=(0.20, 0.70),
            rows=5,
            cols=6,
        )
        mapper = PerspectiveBoardMapper(quad)

        # 中心点应精确对齐均分网格
        assert mapper.cell_center(0, 0) == pytest.approx((0.25, 0.25), abs=1e-6)
        assert mapper.cell_center(4, 5) == pytest.approx((0.75, 0.65), abs=1e-6)
        assert mapper.norm_to_cell(0.25, 0.25) == (0, 0)
        assert mapper.norm_to_cell(0.75, 0.65) == (4, 5)
        assert mapper.norm_to_cell(0.05, 0.05) is None

    def test_trapezoid_homography_roundtrip_and_foreshortening(self) -> None:
        # 使用真实卫戍协议 16:9 梯形透视棋盘（上窄下宽 5x9）
        _, layout = ConfigLoader(use_user_override=False).load()
        mapper = PerspectiveBoardMapper(layout.board_quad)

        assert mapper.H.shape == (3, 3)
        assert mapper.H_inv.shape == (3, 3)
        identity = mapper.H @ mapper.H_inv
        identity /= identity[2, 2]
        assert np.allclose(identity, np.eye(3), atol=1e-5)

        # 1. 全棋盘 45 格正逆映射 100% 精准闭合
        for r in range(mapper.rows):
            for c in range(mapper.cols):
                cu, cv = mapper.cell_center(r, c)
                assert mapper.norm_to_cell(cu, cv) == (r, c)

        # 2. 透视近大远小特征：第 0 行（远端）宽度和高度均严格小于第 4 行（近端）
        top_poly = mapper.cell_polygon(0, 4)
        bot_poly = mapper.cell_polygon(4, 4)
        top_width = top_poly[1][0] - top_poly[0][0]
        bot_width = bot_poly[1][0] - bot_poly[0][0]
        top_height = top_poly[3][1] - top_poly[0][1]
        bot_height = bot_poly[3][1] - bot_poly[0][1]

        assert top_width < bot_width
        assert top_height < bot_height

        # 3. 相邻棋盘格共边无缝衔接（无裂缝无错位）
        left_cell = mapper.cell_polygon(2, 3)
        right_cell = mapper.cell_polygon(2, 4)
        assert left_cell[1] == pytest.approx(right_cell[0], abs=1e-6)
        assert left_cell[2] == pytest.approx(right_cell[3], abs=1e-6)

        upper_cell = mapper.cell_polygon(1, 4)
        lower_cell = mapper.cell_polygon(2, 4)
        assert upper_cell[3] == pytest.approx(lower_cell[0], abs=1e-6)
        assert upper_cell[2] == pytest.approx(lower_cell[1], abs=1e-6)

        # 4. 棋盘外坐标检测与最近格裁剪
        assert mapper.norm_to_cell(0.02, 0.50) is None
        assert mapper.norm_to_cell(0.50, 0.95) is None
        assert mapper.nearest_cell(0.50, 0.95) == (4, 4)
        assert mapper.nearest_cell(0.50, 0.05) == (0, 4)

        # 5. 网格预览线段数量与端点吻合外框
        horiz, vert = mapper.grid_lines()
        assert len(horiz) == mapper.rows + 1
        assert len(vert) == mapper.cols + 1
        assert horiz[0][0] == pytest.approx(layout.board_quad.top_left, abs=1e-6)
        assert horiz[-1][1] == pytest.approx(layout.board_quad.bottom_right, abs=1e-6)

    def test_board_move_and_boundary_exits(self) -> None:
        _, layout = ConfigLoader(use_user_override=False).load()
        mapper = PerspectiveBoardMapper(layout.board_quad)

        assert mapper.move(0, 0, "up") == (0, 0, "up")
        assert mapper.move(0, 0, "left") == (0, 0, "left")
        assert mapper.move(0, 0, "left", wrap_cols=True) == (0, 8, None)
        assert mapper.move(4, 8, "down") == (4, 8, "down")
        assert mapper.move(4, 8, "right") == (4, 8, "right")
        assert mapper.move(2, 4, "up") == (1, 4, None)
        assert mapper.move(2, 4, "right") == (2, 5, None)

        with pytest.raises(IndexError):
            mapper.cell_center(5, 0)
        with pytest.raises(IndexError):
            mapper.cell_polygon(0, 9)

    def test_degenerate_and_crossed_quads_raise_value_error(self) -> None:
        # 四点共线
        collinear = PerspectiveQuad(
            top_left=(0.1, 0.5),
            top_right=(0.3, 0.5),
            bottom_right=(0.6, 0.5),
            bottom_left=(0.9, 0.5),
        )
        with pytest.raises(ValueError):
            PerspectiveBoardMapper(collinear)

        # 自相交蝴蝶结形（底边左右互换）
        bowtie = PerspectiveQuad(
            top_left=(0.2, 0.2),
            top_right=(0.8, 0.2),
            bottom_right=(0.2, 0.8),
            bottom_left=(0.8, 0.8),
        )
        with pytest.raises(ValueError):
            PerspectiveBoardMapper(bowtie)
