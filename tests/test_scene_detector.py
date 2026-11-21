"""视觉采样器、界面阶段识别器与模板校准器单元测试。"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from arkcontroller.config.loader import ConfigLoader
from arkcontroller.platform.window_tracker import ViewportRect
from arkcontroller.vision.calibrator import VisionCalibrator
from arkcontroller.vision.scene_detector import (
    DEFAULT_SCENE_ROIS,
    STANDARD_HEIGHT,
    STANDARD_WIDTH,
    GameScene,
    SceneDetectionResult,
    SceneDetector,
)
from arkcontroller.vision.screen_capture import ScreenCapture


@dataclass(frozen=True)
class _DummyViewport:
    """测试专用的轻量 ViewportRect 替身，与 platform.window_tracker.ViewportRect 字段完全一致。"""

    hwnd: int
    client_left: int
    client_top: int
    client_width: int
    client_height: int
    active_left: int
    active_top: int
    active_width: int
    active_height: int
    is_foreground: bool
    window_title: str = "Arknights"


def _draw_distinctive_badge(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    seed: int = 42,
) -> np.ndarray:
    """在指定矩形区域绘制富含边缘与色块纹理的标志性 UI 图案（避免低方差纯色）。"""
    rng = np.random.default_rng(seed)
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    # 背景棋盘格与斜向色带
    for row in range(0, h, 12):
        for col in range(0, w, 12):
            color = (
                int(rng.integers(30, 240)),
                int(rng.integers(30, 240)),
                int(rng.integers(30, 240)),
            )
            patch[row : row + 12, col : col + 12] = color
    cv2.rectangle(patch, (4, 4), (w - 5, h - 5), (255, 230, 40), 3)
    cv2.line(patch, (6, 6), (w - 7, h - 7), (40, 220, 255), 3)
    cv2.circle(patch, (w // 2, h // 2), max(6, min(w, h) // 4), (245, 245, 245), -1)
    cv2.circle(patch, (w // 2, h // 2), max(3, min(w, h) // 8), (20, 60, 220), -1)

    canvas[y : y + h, x : x + w] = patch
    return patch


def _make_modal_choice_frame(choice_count: int = 3) -> np.ndarray:
    """构造带四周半透明暗化遮罩与中央横向等距高亮卡片的合成弹窗帧。"""
    frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 24, dtype=np.uint8)
    # 模拟暗化后的微弱背景噪声
    frame[100:600, 140:1140] = 32

    if choice_count == 3:
        card_boxes = [
            (215, 200, 210, 300),
            (535, 200, 210, 300),
            (855, 200, 210, 300),
        ]
    else:
        card_boxes = [
            (360, 200, 220, 300),
            (700, 200, 220, 300),
        ]

    for idx, (x, y, w, h) in enumerate(card_boxes):
        # 卡片主亮面 + 顶部标题栏 + 内框线条
        frame[y : y + h, x : x + w] = (205, 212, 220)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (250, 250, 250), 3)
        cv2.rectangle(frame, (x + 12, y + 14), (x + w - 12, y + 54), (90 + idx * 30, 175, 235), -1)
        cv2.rectangle(frame, (x + 20, y + 76), (x + w - 20, y + h - 24), (170, 180, 195), 2)

    return frame


def _make_battle_prep_frame(shop_open: bool = True) -> np.ndarray:
    """构造具备底部 10 格整备槽条纹与右下角高亮准备按钮的局内休整期合成帧。"""
    frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 48, dtype=np.uint8)
    # 中部棋盘区域背景纹理
    cv2.rectangle(frame, (180, 110), (1080, 410), (75, 82, 92), -1)
    for gx in range(220, 1060, 90):
        cv2.line(frame, (gx, 120), (gx, 400), (105, 112, 122), 2)

    # 中下部商店展开面板
    if shop_open:
        cv2.rectangle(frame, (180, 415), (1025, 550), (110, 115, 125), -1)
        for sx in range(200, 980, 160):
            cv2.rectangle(frame, (sx, 425), (sx + 140, 540), (185, 195, 210), -1)
            cv2.rectangle(frame, (sx, 425), (sx + 140, 540), (245, 245, 245), 2)
    else:
        frame[410:555, 180:1030] = 22

    # 底部 10 格整备槽规则边框（y in [565, 685], x in [170, 1010]）
    cv2.rectangle(frame, (170, 570), (1010, 682), (35, 40, 48), -1)
    cv2.rectangle(frame, (170, 570), (1010, 682), (225, 230, 235), 3)
    slot_w = 84
    for i in range(10):
        sx = 170 + i * slot_w
        cv2.rectangle(frame, (sx + 4, 576), (sx + slot_w - 4, 676), (125, 135, 145), -1)
        cv2.rectangle(frame, (sx + 4, 576), (sx + slot_w - 4, 676), (240, 245, 250), 2)

    # 右下角高亮高饱和“准备就绪”行动按钮（BGR 亮青蓝/橙黄）
    cv2.rectangle(frame, (1055, 575), (1235, 675), (255, 200, 20), -1)
    cv2.rectangle(frame, (1055, 575), (1235, 675), (255, 255, 255), 3)

    return frame


def _make_battle_combat_frame() -> np.ndarray:
    """构造底部有整备槽、右下角无高亮准备按钮、顶部中央有波次 HUD 的作战期合成帧。"""
    frame = _make_battle_prep_frame(shop_open=False)
    # 清除右下角休整期准备按钮，改为暗灰背景
    frame[555:695, 1035:1255] = (32, 34, 38)

    # 顶部中央作战波次与剩余敌数 HUD 面板
    cv2.rectangle(frame, (475, 18), (805, 80), (30, 35, 42), -1)
    cv2.rectangle(frame, (475, 18), (805, 80), (235, 240, 245), 2)
    # 红/青状态进度条与数字高对比线条
    cv2.rectangle(frame, (492, 30), (630, 48), (40, 60, 240), -1)
    cv2.rectangle(frame, (648, 30), (788, 48), (245, 200, 30), -1)
    for tx in range(500, 780, 24):
        cv2.line(frame, (tx, 56), (tx + 10, 72), (245, 245, 245), 2)

    return frame


# =====================================================================
# 1. 接口契约与空目录 / 异常输入安全回退测试
# =====================================================================


def test_enum_and_result_contract() -> None:
    """校验精简后的局内 GameScene 枚举值（5 项）与 SceneDetectionResult 冻结属性契约。"""
    expected_values = {
        "battle_prep",
        "modal_choice",
        "modal_confirm",
        "battle_combat",
        "unknown_free_cursor",
    }
    assert {scene.value for scene in GameScene} == expected_values

    # 确认局外场景枚举已彻底移除
    for removed_name in ("ACTIVITY_HUB", "PRE_MATCH_SETUP", "SQUAD_EDIT", "SETTLEMENT"):
        assert not hasattr(GameScene, removed_name)

    assert GameScene.MODAL_CONFIRM in DEFAULT_SCENE_ROIS
    assert DEFAULT_SCENE_ROIS[GameScene.MODAL_CONFIRM] == (0.15, 0.28, 0.70, 0.44)

    res = SceneDetectionResult(scene=GameScene.BATTLE_PREP, confidence=0.95)
    assert res.shop_open is True
    assert res.choice_count == 3
    with pytest.raises(FrozenInstanceError):
        res.confidence = 0.5  # type: ignore[misc]


def test_empty_templates_dir_and_safe_fallback(tmp_path: Path) -> None:
    """验证空模板目录、不存在目录、None 帧、纯色帧与纯噪点帧均安全返回 UNKNOWN_FREE_CURSOR。"""
    missing_dir = tmp_path / "does_not_exist"
    detector = SceneDetector(templates_dir=missing_dir, debounce_frames=1)
    assert detector.loaded_template_count == 0

    # None 输入
    res_none = detector.detect(None)
    assert res_none.scene == GameScene.UNKNOWN_FREE_CURSOR
    assert res_none.confidence == 0.0

    # 纯黑帧、纯白帧、纯灰帧
    for fill_val in (0, 128, 255):
        solid = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), fill_val, dtype=np.uint8)
        res = detector.detect(solid)
        assert res.scene == GameScene.UNKNOWN_FREE_CURSOR
        assert res.confidence == 0.0

    # 纯随机白噪声帧不应误触任何启发式场景
    rng = np.random.default_rng(2026)
    noise = rng.integers(0, 256, size=(STANDARD_HEIGHT, STANDARD_WIDTH, 3), dtype=np.uint8)
    res_noise = detector.detect(noise)
    assert res_noise.scene == GameScene.UNKNOWN_FREE_CURSOR

    # 异常通道数帧也不抛异常
    bad_frame = np.zeros((100, 100, 5), dtype=np.uint8)
    res_bad = detector.detect(bad_frame)
    assert res_bad.scene == GameScene.UNKNOWN_FREE_CURSOR


# =====================================================================
# 2. 模板匹配、ROI 元数据空间裁剪与 VisionCalibrator 集成测试
# =====================================================================


def test_template_matching_plain_png_without_json(tmp_path: Path) -> None:
    """仅放置 <scene_value>.png（无 JSON 元数据）时应能在全帧准确匹配命中。"""
    frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 25, dtype=np.uint8)
    badge = _draw_distinctive_badge(frame, x=80, y=50, w=220, h=110, seed=101)

    png_path = tmp_path / f"{GameScene.BATTLE_COMBAT.value}.png"
    ok, buf = cv2.imencode(".png", badge)
    assert ok
    buf.tofile(str(png_path))

    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)
    assert detector.loaded_template_count == 1

    # 标准 1280x720 帧命中
    res = detector.detect(frame)
    assert res.scene == GameScene.BATTLE_COMBAT
    assert res.confidence >= 0.95

    # 非 1280x720（如 1920x1080）帧自动缩放后依然高置信度命中
    frame_1080p = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    res_1080p = detector.detect(frame_1080p)
    assert res_1080p.scene == GameScene.BATTLE_COMBAT
    assert res_1080p.confidence >= 0.85

    # 纯黑帧不会误命中已有模板
    black_frame = np.zeros((STANDARD_HEIGHT, STANDARD_WIDTH, 3), dtype=np.uint8)
    assert detector.detect(black_frame).scene == GameScene.UNKNOWN_FREE_CURSOR


def test_calibrator_roi_metadata_and_spatial_isolation(tmp_path: Path) -> None:
    """验证 VisionCalibrator 裁剪保存 .png + .json ROI 元数据，且检测器严格限定在对应 ROI 匹配。"""
    templates_dir = tmp_path / "templates"
    mock_loader = MagicMock()
    calibrator = VisionCalibrator(templates_dir=templates_dir, loader=mock_loader)

    # 在指定 ROI (u=0.05, v=0.05, w=0.20, h=0.15) 绘制唯一特征图案
    roi_norm = (0.05, 0.05, 0.20, 0.15)
    x = int(round(roi_norm[0] * STANDARD_WIDTH))
    y = int(round(roi_norm[1] * STANDARD_HEIGHT))
    w = int(round(roi_norm[2] * STANDARD_WIDTH))
    h = int(round(roi_norm[3] * STANDARD_HEIGHT))

    source_frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 20, dtype=np.uint8)
    badge = _draw_distinctive_badge(source_frame, x=x, y=y, w=w, h=h, seed=303)

    saved_png = calibrator.capture_scene_template(
        source_frame, GameScene.BATTLE_COMBAT, roi_norm=roi_norm
    )
    saved_json = saved_png.with_suffix(".json")
    assert saved_png.exists()
    assert saved_json.exists()

    meta = json.loads(saved_json.read_text(encoding="utf-8"))
    assert meta["scene"] == GameScene.BATTLE_COMBAT.value
    assert meta["frame_size"] == [STANDARD_WIDTH, STANDARD_HEIGHT]
    assert len(meta["roi_norm"]) == 4
    assert pytest.approx(meta["roi_norm"], abs=0.005) == list(roi_norm)

    # 先初始化为空检测器，随后 reload_templates 即刻生效
    detector = SceneDetector(templates_dir=templates_dir, debounce_frames=1)
    assert detector.loaded_template_count == 1
    res_hit = detector.detect(source_frame)
    assert res_hit.scene == GameScene.BATTLE_COMBAT
    assert res_hit.confidence >= 0.98

    # 将相同图案平移到遥远的右下角（超出原 ROI 搜索窗），验证不会被局部 ROI 误匹配
    relocated_frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 20, dtype=np.uint8)
    relocated_frame[520 : 520 + h, 920 : 920 + w] = badge
    res_miss = detector.detect(relocated_frame)
    assert res_miss.scene == GameScene.UNKNOWN_FREE_CURSOR

    # 测试使用 MODAL_CONFIRM 默认 ROI（roi_norm=None）也能正常裁剪并识别
    default_roi = DEFAULT_SCENE_ROIS[GameScene.MODAL_CONFIRM]
    dx = int(round(default_roi[0] * STANDARD_WIDTH))
    dy = int(round(default_roi[1] * STANDARD_HEIGHT))
    dw = int(round(default_roi[2] * STANDARD_WIDTH))
    dh = int(round(default_roi[3] * STANDARD_HEIGHT))
    confirm_tpl_frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 18, dtype=np.uint8)
    _draw_distinctive_badge(confirm_tpl_frame, x=dx, y=dy, w=dw, h=dh, seed=707)

    calibrator.capture_scene_template(confirm_tpl_frame, GameScene.MODAL_CONFIRM)
    detector.reload_templates()
    assert detector.loaded_template_count == 2
    res_confirm_tpl = detector.detect(confirm_tpl_frame)
    assert res_confirm_tpl.scene == GameScene.MODAL_CONFIRM
    assert res_confirm_tpl.shop_open is False

    # 禁止为 UNKNOWN_FREE_CURSOR 采集模板
    with pytest.raises(ValueError):
        calibrator.capture_scene_template(source_frame, GameScene.UNKNOWN_FREE_CURSOR)


def test_shop_open_template_detection(tmp_path: Path) -> None:
    """验证 battle_prep.png 与 shop_open.png 协同准确区分商店展开与收起状态。"""
    mock_loader = MagicMock()
    calibrator = VisionCalibrator(templates_dir=tmp_path, loader=mock_loader)

    prep_roi = (0.76, 0.78, 0.20, 0.16)
    shop_roi = (0.20, 0.60, 0.40, 0.16)

    px = int(round(prep_roi[0] * STANDARD_WIDTH))
    py = int(round(prep_roi[1] * STANDARD_HEIGHT))
    pw = int(round(prep_roi[2] * STANDARD_WIDTH))
    ph = int(round(prep_roi[3] * STANDARD_HEIGHT))

    sx = int(round(shop_roi[0] * STANDARD_WIDTH))
    sy = int(round(shop_roi[1] * STANDARD_HEIGHT))
    sw = int(round(shop_roi[2] * STANDARD_WIDTH))
    sh = int(round(shop_roi[3] * STANDARD_HEIGHT))

    frame_shop_open = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 30, dtype=np.uint8)
    _draw_distinctive_badge(frame_shop_open, px, py, pw, ph, seed=11)
    _draw_distinctive_badge(frame_shop_open, sx, sy, sw, sh, seed=22)

    calibrator.capture_scene_template(frame_shop_open, GameScene.BATTLE_PREP, roi_norm=prep_roi)
    calibrator.capture_shop_open_template(frame_shop_open, roi_norm=shop_roi)

    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    res_open = detector.detect(frame_shop_open)
    assert res_open.scene == GameScene.BATTLE_PREP
    assert res_open.shop_open is True

    # 清空商店面板区域，保留右下角休整期标志 -> 仍判定 BATTLE_PREP 但 shop_open=False
    frame_shop_closed = frame_shop_open.copy()
    frame_shop_closed[sy : sy + sh, sx : sx + sw] = 28
    res_closed = detector.detect(frame_shop_closed)
    assert res_closed.scene == GameScene.BATTLE_PREP
    assert res_closed.shop_open is False


def test_unit_selected_template_detection(tmp_path: Path) -> None:
    """验证 unit_selected.png 模板能够区分常规休整态与选取角色缩放态。"""
    mock_loader = MagicMock()
    calibrator = VisionCalibrator(templates_dir=tmp_path, loader=mock_loader)

    prep_roi = (0.76, 0.78, 0.20, 0.16)
    unit_roi = (0.02, 0.20, 0.20, 0.40)

    px = int(round(prep_roi[0] * STANDARD_WIDTH))
    py = int(round(prep_roi[1] * STANDARD_HEIGHT))
    pw = int(round(prep_roi[2] * STANDARD_WIDTH))
    ph = int(round(prep_roi[3] * STANDARD_HEIGHT))

    ux = int(round(unit_roi[0] * STANDARD_WIDTH))
    uy = int(round(unit_roi[1] * STANDARD_HEIGHT))
    uw = int(round(unit_roi[2] * STANDARD_WIDTH))
    uh = int(round(unit_roi[3] * STANDARD_HEIGHT))

    frame_unit_selected = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 30, dtype=np.uint8)
    _draw_distinctive_badge(frame_unit_selected, px, py, pw, ph, seed=31)
    _draw_distinctive_badge(frame_unit_selected, ux, uy, uw, uh, seed=42)

    calibrator.capture_scene_template(frame_unit_selected, GameScene.BATTLE_PREP, roi_norm=prep_roi)
    calibrator.capture_unit_selected_template(frame_unit_selected, roi_norm=unit_roi)

    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)
    res_sel = detector.detect(frame_unit_selected)
    assert res_sel.scene == GameScene.BATTLE_PREP
    assert res_sel.unit_selected is True

    frame_normal = frame_unit_selected.copy()
    frame_normal[uy : uy + uh, ux : ux + uw] = 26
    res_norm = detector.detect(frame_normal)
    assert res_norm.scene == GameScene.BATTLE_PREP
    assert res_norm.unit_selected is False


# =====================================================================
# 3. 无模板启发式视觉特征回退测试
# =====================================================================


def test_heuristic_modal_choice_detection(tmp_path: Path) -> None:
    """无模板时启发式准确检测三选一（3卡）与二选一（2卡）弹窗。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    frame_3 = _make_modal_choice_frame(choice_count=3)
    res_3 = detector.detect(frame_3)
    assert res_3.scene == GameScene.MODAL_CHOICE
    assert res_3.choice_count == 3
    assert res_3.shop_open is False
    assert res_3.confidence >= 0.70

    frame_2 = _make_modal_choice_frame(choice_count=2)
    res_2 = detector.detect(frame_2)
    assert res_2.scene == GameScene.MODAL_CHOICE
    assert res_2.choice_count == 2
    assert res_2.confidence >= 0.70


def test_heuristic_battle_prep_and_combat_detection(tmp_path: Path) -> None:
    """无模板时启发式准确区分局内休整期（含商店开关）与自动作战期。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    prep_open = _make_battle_prep_frame(shop_open=True)
    res_prep_open = detector.detect(prep_open)
    assert res_prep_open.scene == GameScene.BATTLE_PREP
    assert res_prep_open.shop_open is True
    assert res_prep_open.confidence >= 0.65

    prep_closed = _make_battle_prep_frame(shop_open=False)
    res_prep_closed = detector.detect(prep_closed)
    assert res_prep_closed.scene == GameScene.BATTLE_PREP
    assert res_prep_closed.shop_open is False

    combat_frame = _make_battle_combat_frame()
    res_combat = detector.detect(combat_frame)
    assert res_combat.scene == GameScene.BATTLE_COMBAT
    assert res_combat.shop_open is False
    assert res_combat.confidence >= 0.65


def test_heuristic_real_pc_stronghold_prep_shop_and_unit_selected(tmp_path: Path) -> None:
    """验证实机标定的零模板 HSV 强特征：右上粉红准备按钮、刷冻双色块与左侧角色血条（排除拖拽金框）。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    # 1. 默认休整期（右上角粉红准备按钮，无商店双按钮、无角色青蓝血条）
    base = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 45, dtype=np.uint8)
    # 铺一点灰阶方差避免纯色帧守卫
    cv2.rectangle(base, (200, 120), (1040, 500), (88, 92, 98), -1)
    # 右上角粉红准备按钮（BGR: (110, 70, 235) -> HSV H≈172, S≈179, V≈235）
    cv2.rectangle(base, (1120, 100), (1255, 140), (110, 70, 235), -1)

    res_normal = detector.detect(base)
    assert res_normal.scene == GameScene.BATTLE_PREP
    assert res_normal.shop_open is False
    assert res_normal.unit_selected is False
    assert res_normal.confidence >= 0.80

    # 2. 商店展开态（在右侧绘制黄色刷新与蓝色冻结双色块）
    shop_open_frame = base.copy()
    # 黄色刷新（BGR: (30, 210, 240) -> HSV H≈25, S≈223, V≈240）
    cv2.rectangle(shop_open_frame, (1125, 430), (1260, 480), (30, 210, 240), -1)
    # 蓝色冻结（BGR: (230, 140, 20) -> HSV H≈103, S≈233, V≈230）
    cv2.rectangle(shop_open_frame, (970, 430), (1105, 480), (230, 140, 20), -1)

    res_shop = detector.detect(shop_open_frame)
    assert res_shop.scene == GameScene.BATTLE_PREP
    assert res_shop.shop_open is True
    assert res_shop.unit_selected is False

    # 2b. 商店冻结态（按钮变为青绿色「解冻」，BGR: (170, 180, 40) -> HSV H≈88, S≈198, V≈180）
    shop_frozen_frame = shop_open_frame.copy()
    cv2.rectangle(shop_frozen_frame, (970, 430), (1105, 480), (170, 180, 40), -1)
    res_frozen = detector.detect(shop_frozen_frame)
    assert res_frozen.scene == GameScene.BATTLE_PREP
    assert res_frozen.shop_open is True
    assert res_frozen.unit_selected is False

    # 3. 选取角色特写详情态（左侧青蓝生命条，无右下拖拽售出金框）
    unit_sel_frame = base.copy()
    # 青蓝血条（BGR: (240, 185, 25) -> HSV H≈97, S≈228, V≈240）
    cv2.rectangle(unit_sel_frame, (10, 328), (420, 342), (240, 185, 25), -1)

    res_sel = detector.detect(unit_sel_frame)
    assert res_sel.scene == GameScene.BATTLE_PREP
    assert res_sel.shop_open is False
    assert res_sel.unit_selected is True

    # 4. 拖拽态保护：若右下角出现金黄色拖拽售出区，即使左侧残留青蓝条也不误判为角色详情态
    dragging_frame = unit_sel_frame.copy()
    cv2.rectangle(dragging_frame, (1115, 530), (1250, 690), (25, 190, 235), -1)
    res_drag = detector.detect(dragging_frame)
    assert res_drag.scene == GameScene.BATTLE_PREP
    assert res_drag.unit_selected is False


def test_heuristic_dynamic_shop_slot_count_4_5_6_and_max_level(tmp_path: Path) -> None:
    """验证商店升至二本(5槽)/四本(6槽)向左侧延伸及满级无升级卡时栏位数自动检测。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    base_shop = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 38, dtype=np.uint8)
    cv2.rectangle(base_shop, (200, 120), (1040, 480), (88, 92, 98), -1)
    # 右上角粉红准备按钮 + 右侧黄刷新与蓝冻结双按钮
    cv2.rectangle(base_shop, (1120, 100), (1255, 140), (110, 70, 235), -1)
    cv2.rectangle(base_shop, (1125, 430), (1260, 480), (30, 210, 240), -1)
    cv2.rectangle(base_shop, (970, 430), (1105, 480), (230, 140, 20), -1)

    # 1. 1 本 4 槽：升级卡位于 x=332..441，左侧 x<325 为暗背景
    f4 = base_shop.copy()
    cv2.rectangle(f4, (332, 535), (441, 695), (185, 185, 185), -1)
    res4 = detector.detect(f4)
    assert res4.scene == GameScene.BATTLE_PREP
    assert res4.shop_open is True
    assert res4.shop_slot_count == 4

    # 2. 2~3 本 5 槽：升级卡左移至 x=170..279（即使资金充足变为亮蓝/亮黄高亮色也能检出）
    f5 = base_shop.copy()
    cv2.rectangle(f5, (170, 535), (279, 695), (220, 210, 140), -1)
    cv2.rectangle(f5, (288, 535), (442, 695), (90, 80, 110), -1)
    res5 = detector.detect(f5)
    assert res5.shop_open is True
    assert res5.shop_slot_count == 5

    # 3. 4~5 本 6 槽：升级卡左移至 x=8..117
    f6 = base_shop.copy()
    cv2.rectangle(f6, (8, 535), (117, 695), (190, 190, 190), -1)
    cv2.rectangle(f6, (126, 535), (280, 695), (90, 80, 110), -1)
    res6 = detector.detect(f6)
    assert res6.shop_open is True
    assert res6.shop_slot_count == 6

    # 4. 6 本满级兜底：即便左侧升级卡消失，仅凭第 6 槽商品卡边框 [126..280] 仍精准检出 6 槽
    f6_max = base_shop.copy()
    cv2.rectangle(f6_max, (126, 535), (280, 695), (75, 70, 95), -1)
    cv2.rectangle(f6_max, (126, 535), (280, 695), (150, 140, 180), 2)
    res6_max = detector.detect(f6_max)
    assert res6_max.shop_open is True
    assert res6_max.shop_slot_count == 6


def _make_real_style_modal_frame(
    *,
    choice_count: int = 3,
    selected_index: int | None = None,
    teammate_count: int = 0,
) -> np.ndarray:
    """构造贴合《明日方舟》卫戍协议实机的暗底亮灰框三选一(1x3)/六选一(2x3)与白色确认条合成帧。"""
    frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 58, dtype=np.uint8)
    # 模拟半透明压暗后的冰面地板与左上角青色「悬赏决策」标题色带
    cv2.rectangle(frame, (80, 520), (1180, 705), (95, 92, 86), -1)
    cv2.rectangle(frame, (160, 26), (380, 76), (220, 215, 25), -1)
    # 右侧方形「收起」按钮
    btn_x = 1203 if choice_count <= 3 else 1147
    cv2.rectangle(frame, (btn_x, 325), (btn_x + 70, 395), (120, 120, 120), -1)
    cv2.rectangle(frame, (btn_x, 325), (btn_x + 70, 395), (235, 235, 235), 2)

    if choice_count == 6:
        cols = [(220, 478), (528, 785), (835, 1093)]
        boxes = [(xl, 120, xr, 337) for xl, xr in cols] + [
            (xl, 387, xr, 608) for xl, xr in cols
        ]
    else:
        cols = [(171, 460), (515, 801), (857, 1144)]
        boxes = [(xl, 242, xr, 483) for xl, xr in cols]

    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        if selected_index == idx:
            top_y, bot_y = y1 - 20, y2 + 26
            cv2.rectangle(frame, (x1, top_y), (x2, bot_y), (34, 38, 30), -1)
            cv2.rectangle(frame, (x1, bot_y - 50), (x2, bot_y), (242, 244, 242), -1)
            cv2.rectangle(frame, (x1, top_y), (x2, bot_y), (245, 248, 245), 3)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (24, 26, 26), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (215, 218, 218), 3)

    avatar_boxes = [
        (19, 160, 67, 66),
        (19, 235, 67, 67),
        (19, 311, 67, 66),
        (19, 387, 67, 66),
    ]
    for mx, my, mw, mh in avatar_boxes[: max(0, min(4, teammate_count))]:
        cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (95, 85, 80), -1)
        cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (235, 240, 240), 2)

    return frame


def _draw_covenant_badge(
    frame: np.ndarray, cx: float, *, activated: bool = True
) -> None:
    """在 1280x720 帧顶部栏下方 (cx, 85) 绘制已激活或未激活的圆形羁绊徽章及底部文字。"""
    ix = int(round(cx))
    if activated:
        cv2.circle(frame, (ix, 85), 24, (185, 230, 35), -1)
        cv2.circle(frame, (ix, 85), 24, (220, 250, 160), 2)
    else:
        cv2.circle(frame, (ix, 85), 24, (32, 34, 38), -1)
        cv2.ellipse(frame, (ix, 85), (22, 22), 0, 90, 260, (235, 95, 35), 3)
        cv2.circle(frame, (ix, 85), 22, (115, 120, 125), 1)
    # 底部层数与盟约名高对比暗影白字模拟
    cv2.rectangle(frame, (ix - 18, 96), (ix + 18, 122), (16, 18, 20), -1)
    cv2.line(frame, (ix - 12, 104), (ix + 12, 104), (225, 228, 230), 2)
    cv2.line(frame, (ix - 14, 114), (ix + 14, 114), (205, 210, 215), 2)


def test_real_style_3choice_and_6choice_modal_and_white_bar(tmp_path: Path) -> None:
    """验证实机暗底亮框三选一(1x3)与多人六选一(2x3)自动识别、白色确认条索引检测及左侧队友检测。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    # 1. 单人三选一：未选中任何卡片 vs 选中第 1 张弹出白色确认条
    f3_idle = _make_real_style_modal_frame(choice_count=3, selected_index=None)
    r3_idle = detector.detect(f3_idle)
    assert r3_idle.scene == GameScene.MODAL_CHOICE
    assert r3_idle.choice_count == 3
    assert r3_idle.selected_choice_index is None
    assert r3_idle.has_teammates is False

    f3_sel0 = _make_real_style_modal_frame(choice_count=3, selected_index=0)
    r3_sel0 = detector.detect(f3_sel0)
    assert r3_sel0.scene == GameScene.MODAL_CHOICE
    assert r3_sel0.choice_count == 3
    assert r3_sel0.selected_choice_index == 0

    # 2. 多人六选一（2x3）：上排选中(idx=1)与下排选中(idx=4)，且识别左侧 4 名联机队友头像
    f6_top = _make_real_style_modal_frame(
        choice_count=6, selected_index=1, teammate_count=4
    )
    r6_top = detector.detect(f6_top)
    assert r6_top.scene == GameScene.MODAL_CHOICE
    assert r6_top.choice_count == 6
    assert r6_top.selected_choice_index == 1
    assert r6_top.has_teammates is True

    f6_bot = _make_real_style_modal_frame(
        choice_count=6, selected_index=4, teammate_count=2
    )
    r6_bot = detector.detect(f6_bot)
    assert r6_bot.scene == GameScene.MODAL_CHOICE
    assert r6_bot.choice_count == 6
    assert r6_bot.selected_choice_index == 4
    assert r6_bot.has_teammates is True


def test_symmetric_covenant_count_and_teammate_detection_in_prep(tmp_path: Path) -> None:
    """验证休整期顶部下方居中对称羁绊图标数量（0~4，奇偶混合激活/未激活）与联机队友检测。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    base = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 46, dtype=np.uint8)
    cv2.rectangle(base, (200, 150), (1040, 500), (88, 92, 98), -1)
    cv2.rectangle(base, (1120, 100), (1255, 140), (110, 70, 235), -1)

    res0 = detector.detect(base)
    assert res0.scene == GameScene.BATTLE_PREP
    assert res0.covenant_count == 0
    assert res0.has_teammates is False

    for expected_n in (1, 2, 3, 4):
        fn = base.copy()
        for i in range(expected_n):
            cx = 640.0 + (i - (expected_n - 1) * 0.5) * 66.5
            _draw_covenant_badge(fn, cx, activated=(i % 2 == 0))
        rn = detector.detect(fn)
        assert rn.scene == GameScene.BATTLE_PREP
        assert rn.covenant_count == expected_n


# =====================================================================
# 4. 连续帧防抖测试
# =====================================================================


def test_debounce_suppresses_transient_flicker(tmp_path: Path) -> None:
    """连续 debounce_frames 帧一致才更新稳定场景，单帧闪烁或黑屏过场被滤除。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=2)

    prep_frame = _make_battle_prep_frame(shop_open=True)
    modal_frame = _make_modal_choice_frame(choice_count=3)
    black_frame = np.zeros((STANDARD_HEIGHT, STANDARD_WIDTH, 3), dtype=np.uint8)

    # 第 1 帧休整期：尚未满足连续 2 帧，维持初始 UNKNOWN_FREE_CURSOR
    assert detector.detect(prep_frame).scene == GameScene.UNKNOWN_FREE_CURSOR
    # 第 2 帧连续休整期：正式切换至 BATTLE_PREP
    assert detector.detect(prep_frame).scene == GameScene.BATTLE_PREP

    # 第 3 帧瞬态黑屏过场闪烁：防抖抑制，保持 BATTLE_PREP
    assert detector.detect(black_frame).scene == GameScene.BATTLE_PREP
    # 第 4 帧瞬态弹窗闪烁 1 帧：防抖抑制，保持 BATTLE_PREP
    assert detector.detect(modal_frame).scene == GameScene.BATTLE_PREP
    # 第 5 帧恢复休整期：清零待切换计数，保持 BATTLE_PREP
    assert detector.detect(prep_frame).scene == GameScene.BATTLE_PREP

    # 第 6、7 帧连续出现三选一弹窗：第 6 帧暂保 BATTLE_PREP，第 7 帧稳定切为 MODAL_CHOICE
    assert detector.detect(modal_frame).scene == GameScene.BATTLE_PREP
    res_modal = detector.detect(modal_frame)
    assert res_modal.scene == GameScene.MODAL_CHOICE
    assert res_modal.choice_count == 3

    # 传入 None 立即安全回退为 UNKNOWN_FREE_CURSOR
    assert detector.detect(None).scene == GameScene.UNKNOWN_FREE_CURSOR


# =====================================================================
# 5. ScreenCapture 同步截屏、备选回退与多线程隔离测试
# =====================================================================


def test_screen_capture_sync_resize_and_fallback() -> None:
    """验证 ScreenCapture 使用 mss 缩放输出 1280x720 BGR，以及失败时触发 PrintWindow 回退。"""
    viewport = _DummyViewport(
        hwnd=12345,
        client_left=100,
        client_top=50,
        client_width=1920,
        client_height=1080,
        active_left=100,
        active_top=50,
        active_width=1920,
        active_height=1080,
        is_foreground=True,
    )

    fake_bgra = np.zeros((1080, 1920, 4), dtype=np.uint8)
    fake_bgra[:, :, 0] = 200  # B
    fake_bgra[:, :, 1] = 100  # G
    fake_bgra[:, :, 2] = 50   # R
    fake_bgra[:, :, 3] = 255  # A

    fake_sct = MagicMock()
    fake_sct.grab.return_value = fake_bgra

    with patch("arkcontroller.vision.screen_capture.mss.mss", return_value=fake_sct):
        capture = ScreenCapture()
        frame = capture.grab_frame(viewport)  # type: ignore[arg-type]
        assert frame is not None
        assert frame.shape == (STANDARD_HEIGHT, STANDARD_WIDTH, 3)
        assert frame.dtype == np.uint8
        assert tuple(frame[0, 0].tolist()) == (200, 100, 50)
        fake_sct.grab.assert_called_once_with(
            {"left": 100, "top": 50, "width": 1920, "height": 1080}
        )

        # 模拟 mss 抛异常，确认回退调用 _grab_via_print_window
        fake_sct.grab.side_effect = RuntimeError("GDI 截屏失败")
        fallback_img = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 77, dtype=np.uint8)
        with patch.object(capture, "_grab_via_print_window", return_value=fallback_img) as mock_pw:
            frame_fb = capture.grab_frame(viewport)  # type: ignore[arg-type]
            mock_pw.assert_called_once()
            assert frame_fb is not None
            assert int(frame_fb[0, 0, 0]) == 77

        capture.close()


def test_screen_capture_async_thread_isolation() -> None:
    """验证异步守护线程独占独立 mss 实例、线程安全更新帧与优雅退出释放句柄。"""
    viewport = _DummyViewport(
        hwnd=999,
        client_left=0,
        client_top=0,
        client_width=1280,
        client_height=720,
        active_left=0,
        active_top=0,
        active_width=1280,
        active_height=720,
        is_foreground=True,
    )

    created_thread_ids: list[int] = []
    closed_instances: list[MagicMock] = []
    frame_ready = threading.Event()

    def _fake_mss_factory() -> MagicMock:
        tid = threading.get_ident()
        created_thread_ids.append(tid)
        inst = MagicMock()

        def _grab(_monitor: dict[str, int]) -> np.ndarray:
            arr = np.full((720, 1280, 4), 123, dtype=np.uint8)
            frame_ready.set()
            return arr

        inst.grab.side_effect = _grab
        inst.close.side_effect = lambda: closed_instances.append(inst)
        return inst

    with patch("arkcontroller.vision.screen_capture.mss.mss", side_effect=_fake_mss_factory):
        with ScreenCapture() as capture:
            # 主线程先执行一次同步抓取
            main_tid = threading.get_ident()
            sync_frame = capture.grab_frame(viewport)  # type: ignore[arg-type]
            assert sync_frame is not None

            # 启动后台守护采样线程
            capture.update_viewport(viewport)  # type: ignore[arg-type]
            capture.start_async(fps=30.0)
            assert capture.is_running
            assert frame_ready.wait(timeout=2.0), "后台采样线程未在预期时间内产出帧"

            # 稍等一次循环写入缓存后读取
            deadline = time.monotonic() + 1.0
            latest = None
            while time.monotonic() < deadline:
                latest = capture.get_latest_frame()
                if latest is not None:
                    break
                time.sleep(0.01)

            assert latest is not None
            assert latest.shape == (STANDARD_HEIGHT, STANDARD_WIDTH, 3)

            # 清空 viewport 后应同步清空缓存帧
            capture.update_viewport(None)
            assert capture.get_latest_frame() is None

            capture.stop_async()
            assert not capture.is_running

    # 验证主线程与后台线程分别创建了各自独立的 mss 实例，且全部在退出时关闭
    assert len(created_thread_ids) == 2
    assert created_thread_ids[0] == main_tid
    assert created_thread_ids[1] != main_tid
    assert len(closed_instances) == 2


def test_real_viewport_and_config_loader_integration(tmp_path: Path) -> None:
    """验证与真实 ViewportRect、ConfigLoader 的跨模块集成以及变体模板/坏 JSON 容错。"""
    # 1. 使用真实 ViewportRect.from_client_rect 验证超宽屏 21:9 扣除黑边区域抓取
    vp = ViewportRect.from_client_rect(
        hwnd=4321,
        client_left=0,
        client_top=0,
        client_width=3440,
        client_height=1440,
        is_foreground=False,
        window_title="明日方舟",
    )
    fake_bgra = np.full((vp.active_height, vp.active_width, 4), 160, dtype=np.uint8)
    fake_sct = MagicMock()
    fake_sct.grab.return_value = fake_bgra

    # 模拟非前台窗口 PrintWindow 返回全黑帧，自动回退至 mss 成功获取画面
    black_pw = np.zeros((STANDARD_HEIGHT, STANDARD_WIDTH, 3), dtype=np.uint8)
    with patch("arkcontroller.vision.screen_capture.mss.mss", return_value=fake_sct):
        with ScreenCapture() as capture:
            with patch.object(capture, "_grab_via_print_window", return_value=black_pw):
                frame = capture.grab_frame(vp)
                assert frame is not None
                assert frame.shape == (STANDARD_HEIGHT, STANDARD_WIDTH, 3)
                assert int(frame[0, 0, 0]) == 160
                fake_sct.grab.assert_called_once_with(
                    {
                        "left": vp.active_left,
                        "top": vp.active_top,
                        "width": vp.active_width,
                        "height": vp.active_height,
                    }
                )
            # 无效视口（None 或 0 宽高）安全返回 None
            assert capture.grab_frame(None) is None
            zero_vp = ViewportRect(0, 0, 0, 0, 0, 0, 0, 0, 0, True)
            assert capture.grab_frame(zero_vp) is None

    # 2. 使用真实 ConfigLoader 联动 VisionCalibrator（含保存覆盖、列举、删除与异常 ROI 校验）
    config_dir = tmp_path / "configs"
    templates_dir = tmp_path / "templates"
    loader = ConfigLoader(config_dir=config_dir)
    calibrator = VisionCalibrator(templates_dir=templates_dir, loader=loader)

    override_path = calibrator.save_layout_override({"keymap": {"ready_hold_ms": 420}})
    assert override_path.exists()
    override_json = json.loads(override_path.read_text(encoding="utf-8"))
    assert override_json["keymap"]["ready_hold_ms"] == 420

    # 采集多变体模板并故意写入一个损坏 JSON 验证容错加载
    sample_frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 24, dtype=np.uint8)
    badge = _draw_distinctive_badge(sample_frame, x=120, y=100, w=200, h=120, seed=555)
    calibrator.capture_scene_template(
        sample_frame, GameScene.MODAL_CONFIRM, roi_norm=(120 / 1280, 100 / 720, 200 / 1280, 120 / 720)
    )
    assert calibrator.list_installed_templates() == ["modal_confirm"]

    # 写入变体模板与损坏 JSON，验证不崩溃且仍能命中
    variant_png = templates_dir / "battle_combat__alt.png"
    ok, encoded = cv2.imencode(".png", badge)
    assert ok
    encoded.tofile(str(variant_png))
    (templates_dir / "battle_combat__alt.json").write_text("{corrupt json", encoding="utf-8")

    detector = SceneDetector(templates_dir=templates_dir, debounce_frames=1)
    assert detector.loaded_template_count == 2
    assert detector.detect(sample_frame).scene in (
        GameScene.MODAL_CONFIRM,
        GameScene.BATTLE_COMBAT,
    )

    assert calibrator.delete_scene_template(GameScene.MODAL_CONFIRM) is True
    assert calibrator.delete_scene_template(GameScene.MODAL_CONFIRM) is False
    detector.reload_templates()
    assert detector.loaded_template_count == 1
    assert detector.detect(sample_frame).scene == GameScene.BATTLE_COMBAT

    # 非法 ROI 参数与无效帧均抛出 ValueError
    with pytest.raises(ValueError):
        calibrator.capture_scene_template(sample_frame, GameScene.BATTLE_PREP, roi_norm=(0.2, 0.2, 0.0, 0.1))
    with pytest.raises(ValueError):
        calibrator.capture_scene_template(np.array([], dtype=np.uint8), GameScene.BATTLE_PREP)


# =====================================================================
# 6. 阶段二专项：回合结束确认弹窗、鲁棒 4/5/6 槽商店与 mss 自愈测试
# =====================================================================


def _make_modal_confirm_frame(*, with_top_ready_pink: bool = True) -> np.ndarray:
    """构造回合结束未用完资金/策略确认弹窗合成帧（含右上模糊粉红按钮残留）。"""
    frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 55, dtype=np.uint8)
    if with_top_ready_pink:
        # 模拟弹窗后方模糊残留的右上角粉红「准备就绪」按钮，检验不会被误判为 BATTLE_PREP
        cv2.rectangle(frame, (1120, 100), (1255, 140), (110, 70, 235), -1)

    # 中央全宽黑底确认框 y=193..448
    frame[193:449, :] = (16, 16, 16)
    # 模拟中央白字提示与「☐ 本局不再提示」
    cv2.rectangle(frame, (440, 290), (840, 345), (210, 210, 210), 2)
    cv2.rectangle(frame, (560, 393), (717, 413), (200, 200, 200), 1)

    # 左下灰色「☒ 取消」栏 y=449..520, x=0..639（HSV S≈8, V≈68）
    frame[449:521, 0:640] = (68, 66, 66)
    cv2.rectangle(frame, (275, 470), (365, 502), (235, 235, 235), 2)

    # 右下暗红「☑ 确认」栏 y=449..520, x=640..1280（BGR (55, 55, 155) -> HSV H=0, S≈165, V=155）
    frame[449:521, 640:1280] = (55, 55, 155)
    cv2.rectangle(frame, (915, 470), (1005, 502), (240, 240, 240), 2)
    return frame


def test_heuristic_modal_confirm_detection_and_priority_over_prep(tmp_path: Path) -> None:
    """验证回合结束确认弹窗高置信度命中 MODAL_CONFIRM，且优先级高于右上角粉红准备按钮。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    confirm_frame = _make_modal_confirm_frame(with_top_ready_pink=True)
    res = detector.detect(confirm_frame)
    assert res.scene == GameScene.MODAL_CONFIRM
    assert res.confidence == 0.95
    assert res.shop_open is False

    # 破坏右下暗红确认栏（改为深灰）-> 不再命中 MODAL_CONFIRM，安全回退为休整期
    broken_red = confirm_frame.copy()
    broken_red[449:521, 640:1280] = (68, 66, 66)
    res_no_red = detector.detect(broken_red)
    assert res_no_red.scene == GameScene.BATTLE_PREP

    # 破坏中央黑底确认框（改为高亮灰白）-> 不再命中 MODAL_CONFIRM
    broken_dark = confirm_frame.copy()
    broken_dark[193:449, :] = (165, 165, 165)
    res_no_dark = detector.detect(broken_dark)
    assert res_no_dark.scene != GameScene.MODAL_CONFIRM


def _make_base_shop_frame() -> np.ndarray:
    """构造带实机右上角粉红准备按钮与右侧黄刷新/蓝冻结双按钮的商店展开基础帧。"""
    frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 38, dtype=np.uint8)
    # 模拟左侧空白商店托盘的暗蓝底部渐变背景（V≈68，无垂直卡片边框）
    frame[690:712, 0:1090] = (68, 32, 22)
    cv2.rectangle(frame, (200, 120), (1040, 480), (88, 92, 98), -1)
    cv2.rectangle(frame, (1120, 100), (1255, 140), (110, 70, 235), -1)
    cv2.rectangle(frame, (1125, 430), (1260, 480), (30, 210, 240), -1)
    cv2.rectangle(frame, (970, 430), (1105, 480), (230, 140, 20), -1)
    return frame


def _draw_magenta_upgrade_card(
    frame: np.ndarray, x1: int, x2: int, *, glowing_blue_purple_bottom: bool = True
) -> None:
    """绘制资金充足紫红色高亮「升级 (LEVEL)」卡片（含真实蓝紫底部发光渐层与 [G] 键帽）。"""
    # 卡片整体深紫灰底 + 亮边框
    cv2.rectangle(frame, (x1, 535), (x2, 708), (115, 70, 125), -1)
    # 上部洋红/紫红高亮 LEVEL 屏幕 y=552..628（BGR (215, 120, 235) -> HSV H≈146, V=235）
    cv2.rectangle(frame, (x1 + 6, 552), (x2 - 6, 628), (215, 120, 235), -1)
    if glowing_blue_purple_bottom:
        # 还原 media_0.png 真实紫红高亮 [G] 升级卡底部高饱和蓝紫发光渐层（strict_blue_purple > 0.45）
        cv2.rectangle(frame, (x1 + 2, 688), (x2 - 2, 712), (215, 95, 110), -1)
    else:
        cv2.rectangle(frame, (x1 + 4, 688), (x2 - 4, 712), (82, 76, 86), -1)
    cv2.rectangle(frame, ((x1 + x2) // 2 - 14, 690), ((x1 + x2) // 2 + 14, 710), (225, 225, 225), -1)
    cv2.rectangle(frame, (x1, 535), (x2, 708), (220, 175, 235), 2)


def _draw_gray_upgrade_card(frame: np.ndarray, x1: int, x2: int) -> None:
    """绘制灰态「升级 (LEVEL)」卡片（资金不足时无底部蓝紫发光渐层）。"""
    cv2.rectangle(frame, (x1, 535), (x2, 708), (110, 110, 110), -1)
    cv2.rectangle(frame, (x1 + 6, 552), (x2 - 6, 628), (215, 215, 215), -1)
    cv2.rectangle(frame, (x1 + 4, 688), (x2 - 4, 712), (92, 92, 92), -1)
    cv2.rectangle(frame, (x1, 535), (x2, 708), (200, 200, 200), 2)


def _draw_bright_map_left_clutter(frame: np.ndarray, x1: int, x2: int) -> None:
    """在升级卡左侧叠加亮地图背景与垂直纹理边缘（梯度远超 8），用于验证防误升槽数。"""
    cv2.rectangle(frame, (x1, 535), (x2, 685), (155, 175, 195), -1)
    mid = (x1 + x2) // 2
    cv2.line(frame, (mid, 535), (mid, 695), (235, 240, 245), 3)
    cv2.line(frame, (mid + 4, 535), (mid + 4, 695), (45, 50, 55), 3)


def _draw_shop_item_card(frame: np.ndarray, x1: int, x2: int, *, sold_out: bool = False) -> None:
    """绘制在售商品卡（含底部高饱和蓝紫装饰条）或已售空 no info 暗蓝空槽。"""
    if sold_out:
        # 均匀暗蓝底（BGR (58, 36, 20) -> HSV H≈108, S≈167, V=58）+ 细亮边框
        cv2.rectangle(frame, (x1, 535), (x2, 708), (58, 36, 20), -1)
        cv2.rectangle(frame, (x1, 535), (x2, 708), (125, 95, 70), 2)
        # 底部极暗无蓝紫装饰条
        frame[690:712, x1 + 2 : x2 - 2] = (20, 18, 16)
    else:
        # 亮面商品卡 + 底部高饱和蓝紫装饰横条 y=690..712（BGR (210, 75, 95) -> HSV H≈115, S≈164, V=210）
        cv2.rectangle(frame, (x1, 535), (x2, 708), (188, 192, 202), -1)
        cv2.rectangle(frame, (x1 + 2, 690), (x2 - 2, 712), (210, 75, 95), -1)
        cv2.rectangle(frame, (x1, 535), (x2, 708), (235, 235, 240), 2)


def test_heuristic_shop_slot_count_shifted_magenta_upgrade_and_empty_slots(
    tmp_path: Path,
) -> None:
    """验证商店 4/5/6 槽对 ±10px 偏移、紫红高亮升级卡、已售空暗蓝槽及商品卡防误判的鲁棒性。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    # 1. 1 本 4 槽：紫红资金充足高亮升级卡 [332..441] + 4 张带蓝紫底条商品卡 -> 判 4 槽
    f4 = _make_base_shop_frame()
    _draw_magenta_upgrade_card(f4, 332, 441)
    for sx in (450, 612, 774, 936):
        _draw_shop_item_card(f4, sx, sx + 154, sold_out=False)
    res4 = detector.detect(f4)
    assert res4.scene == GameScene.BATTLE_PREP
    assert res4.shop_open is True
    assert res4.shop_slot_count == 4

    # 2. 2~3 本 5 槽：整体向右偏移 +8px（旧 ±2px 逻辑必败），紫红高亮升级卡 + 第 5 槽在售商品卡 -> 判 5 槽
    f5_shifted = _make_base_shop_frame()
    _draw_magenta_upgrade_card(f5_shifted, 170 + 8, 279 + 8)
    _draw_shop_item_card(f5_shifted, 288 + 8, 442 + 8, sold_out=False)
    for sx in (450, 612, 774, 936):
        _draw_shop_item_card(f5_shifted, sx, sx + 154, sold_out=False)
    res5_shifted = detector.detect(f5_shifted)
    assert res5_shifted.shop_slot_count == 5

    # 3. 5 槽无升级卡场景：仅凭第 5 槽已售空 no info 暗蓝空槽（带 -7px 偏移）-> 依然稳判 5 槽
    f5_empty = _make_base_shop_frame()
    _draw_shop_item_card(f5_empty, 288 - 7, 442 - 7, sold_out=True)
    for sx in (450, 612, 774, 936):
        _draw_shop_item_card(f5_empty, sx, sx + 154, sold_out=False)
    res5_empty = detector.detect(f5_empty)
    assert res5_empty.shop_slot_count == 5

    # 4. 6 本满级无升级卡且第 6 槽已售空 no info 暗蓝空槽（带 +6px 偏移）-> 依然稳判 6 槽
    f6_empty = _make_base_shop_frame()
    _draw_shop_item_card(f6_empty, 126 + 6, 280 + 6, sold_out=True)
    for sx in (288, 450, 612, 774, 936):
        _draw_shop_item_card(f6_empty, sx, sx + 154, sold_out=False)
    res6_empty = detector.detect(f6_empty)
    assert res6_empty.shop_slot_count == 6

    # 5. 验证 _has_shop_upgrade_card 排除第 6 槽完整商品卡右 71% 重叠区，但放行自带蓝紫发光底部的紫红 [G] 升级卡
    tmp_item6 = _make_base_shop_frame()
    _draw_shop_item_card(tmp_item6, 126, 280, sold_out=False)
    hsv_item6 = cv2.cvtColor(tmp_item6, cv2.COLOR_BGR2HSV)
    assert SceneDetector._has_shop_upgrade_card(hsv_item6, 170, 279) is False

    tmp_upg = _make_base_shop_frame()
    _draw_magenta_upgrade_card(tmp_upg, 170, 279, glowing_blue_purple_bottom=True)
    hsv_upg = cv2.cvtColor(tmp_upg, cv2.COLOR_BGR2HSV)
    assert SceneDetector._has_shop_upgrade_card(hsv_upg, 170, 279) is True

    # 6. 紫红高亮带蓝紫底部发光 [G] 升级卡在 4/5/6 槽任意位置（含紧邻 NO INFO 空槽）均准确识别
    f5_mag_no_info = _make_base_shop_frame()
    _draw_magenta_upgrade_card(f5_mag_no_info, 170, 279, glowing_blue_purple_bottom=True)
    _draw_shop_item_card(f5_mag_no_info, 288, 442, sold_out=True)
    for sx in (450, 612, 774, 936):
        _draw_shop_item_card(f5_mag_no_info, sx, sx + 154, sold_out=False)
    assert detector.detect(f5_mag_no_info).shop_slot_count == 5

    f6_mag_no_info = _make_base_shop_frame()
    _draw_magenta_upgrade_card(f6_mag_no_info, 8, 117, glowing_blue_purple_bottom=True)
    _draw_shop_item_card(f6_mag_no_info, 126, 280, sold_out=True)
    for sx in (288, 450, 612, 774, 936):
        _draw_shop_item_card(f6_mag_no_info, sx, sx + 154, sold_out=False)
    assert detector.detect(f6_mag_no_info).shop_slot_count == 6

    # 7. 防误升槽数回归：4 槽与 5 槽升级卡左侧叠加亮地图背景与强垂直边缘（灰态 & 紫红高亮态均绝不误升）
    for draw_upg in (_draw_gray_upgrade_card, _draw_magenta_upgrade_card):
        f4_clutter = _make_base_shop_frame()
        draw_upg(f4_clutter, 332, 441)
        _draw_bright_map_left_clutter(f4_clutter, 260, 326)
        for sx in (450, 612, 774, 936):
            _draw_shop_item_card(f4_clutter, sx, sx + 154, sold_out=False)
        hsv_4c = cv2.cvtColor(f4_clutter, cv2.COLOR_BGR2HSV)
        assert SceneDetector._has_shop_upgrade_card(hsv_4c, 332, 441) is True
        assert SceneDetector._has_shop_item_card(hsv_4c, 288, 442) is False
        assert detector.detect(f4_clutter).shop_slot_count == 4

        f5_clutter = _make_base_shop_frame()
        draw_upg(f5_clutter, 170, 279)
        _draw_bright_map_left_clutter(f5_clutter, 98, 164)
        _draw_shop_item_card(f5_clutter, 288, 442, sold_out=True)
        for sx in (450, 612, 774, 936):
            _draw_shop_item_card(f5_clutter, sx, sx + 154, sold_out=False)
        hsv_5c = cv2.cvtColor(f5_clutter, cv2.COLOR_BGR2HSV)
        assert SceneDetector._has_shop_upgrade_card(hsv_5c, 170, 279) is True
        assert SceneDetector._has_shop_item_card(hsv_5c, 126, 280) is False
        assert detector.detect(f5_clutter).shop_slot_count == 5


def test_screen_capture_mss_self_healing_and_black_pw_rejection() -> None:
    """验证 UAC 初始化失败次帧自愈、运行中句柄失效重建以及拒绝全黑 PrintWindow 伪帧。"""
    viewport = _DummyViewport(
        hwnd=5555,
        client_left=0,
        client_top=0,
        client_width=1280,
        client_height=720,
        active_left=0,
        active_top=0,
        active_width=1280,
        active_height=720,
        is_foreground=True,
    )

    # (A) 首帧 mss.mss() 抛异常（模拟 UAC 安全桌面切换），次帧自动恢复并成功产出画面
    call_count = 0
    recovered_event = threading.Event()
    closed_instances: list[MagicMock] = []

    def _uac_recovery_factory() -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("UAC secure desktop GDI unavailable")
        inst = MagicMock()

        def _grab(_monitor: dict[str, int]) -> np.ndarray:
            recovered_event.set()
            return np.full((720, 1280, 4), 145, dtype=np.uint8)

        inst.grab.side_effect = _grab
        inst.close.side_effect = lambda: closed_instances.append(inst)
        return inst

    with patch("arkcontroller.vision.screen_capture.mss.mss", side_effect=_uac_recovery_factory):
        with ScreenCapture(enable_print_window_fallback=False) as cap:
            cap.update_viewport(viewport)  # type: ignore[arg-type]
            cap.start_async(fps=40.0)
            assert recovered_event.wait(timeout=2.0), "后台线程未能在首帧 mss 失败后自动重建恢复"

            deadline = time.monotonic() + 1.0
            latest = None
            while time.monotonic() < deadline:
                latest = cap.get_latest_frame()
                if latest is not None:
                    break
                time.sleep(0.01)

            assert latest is not None
            assert latest.shape == (STANDARD_HEIGHT, STANDARD_WIDTH, 3)
            assert int(latest[0, 0, 0]) == 145
            cap.stop_async()

    assert call_count >= 2
    assert len(closed_instances) == 1

    # (B) 运行中 sct.grab() 句柄失效 -> 立即销毁旧实例，下一周期重建新实例恢复采样
    runtime_instances: list[MagicMock] = []
    runtime_closed: list[int] = []
    rebuilt_event = threading.Event()

    def _runtime_rebuild_factory() -> MagicMock:
        idx = len(runtime_instances)
        inst = MagicMock()
        runtime_instances.append(inst)
        inst.close.side_effect = lambda i=idx: runtime_closed.append(i)
        if idx == 0:
            inst.grab.side_effect = RuntimeError("GDI DC invalidated")
        else:
            def _ok_grab(_monitor: dict[str, int]) -> np.ndarray:
                rebuilt_event.set()
                return np.full((720, 1280, 4), 202, dtype=np.uint8)

            inst.grab.side_effect = _ok_grab
        return inst

    with patch("arkcontroller.vision.screen_capture.mss.mss", side_effect=_runtime_rebuild_factory):
        with ScreenCapture(enable_print_window_fallback=False) as cap:
            cap.update_viewport(viewport)  # type: ignore[arg-type]
            cap.start_async(fps=40.0)
            assert rebuilt_event.wait(timeout=2.0), "后台线程未能在句柄异常后重建新 mss 实例"

            deadline = time.monotonic() + 1.0
            latest = None
            while time.monotonic() < deadline:
                latest = cap.get_latest_frame()
                if latest is not None:
                    break
                time.sleep(0.01)

            assert latest is not None
            assert int(latest[0, 0, 0]) == 202
            cap.stop_async()

    assert len(runtime_instances) >= 2
    assert 0 in runtime_closed and 1 in runtime_closed

    # (C) 当 mss 失败且 PrintWindow 返回全黑伪帧（max == 0）时，必须返回 None 而非全黑帧
    black_pw = np.zeros((STANDARD_HEIGHT, STANDARD_WIDTH, 3), dtype=np.uint8)
    broken_sct = MagicMock()
    broken_sct.grab.side_effect = RuntimeError("mss grab failed")

    with patch("arkcontroller.vision.screen_capture.mss.mss", return_value=broken_sct):
        with ScreenCapture(enable_print_window_fallback=True) as cap:
            with patch.object(cap, "_grab_via_print_window", return_value=black_pw):
                # 前台窗口与非前台窗口均不得返回全黑伪帧
                assert cap.grab_frame(viewport) is None  # type: ignore[arg-type]
                bg_vp = _DummyViewport(
                    hwnd=5555,
                    client_left=0,
                    client_top=0,
                    client_width=1280,
                    client_height=720,
                    active_left=0,
                    active_top=0,
                    active_width=1280,
                    active_height=720,
                    is_foreground=False,
                )
                assert cap.grab_frame(bg_vp) is None  # type: ignore[arg-type]


def test_optional_real_screenshots_regression(tmp_path: Path) -> None:
    """无外部硬依赖的可选实机截图回归校验：若本机存在标定截图则验证 0 误判。"""
    a7_up = Path(
        r"C:\Users\MUXI\.gemini\antigravity\brain\a7e72e90-d260-48de-94e8-debc493ed124\.user_uploaded"
    )
    a7_dir = Path(
        r"C:\Users\MUXI\.gemini\antigravity\brain\a7e72e90-d260-48de-94e8-debc493ed124"
    )
    old_sc = Path(
        r"C:\Users\MUXI\.gemini\antigravity\brain\26ef9845-6d52-4d6e-925c-cbef8d7a9ccc\scratch"
    )

    cases: list[tuple[Path, GameScene, bool | None, int | None, int | None]] = [
        (a7_up / "media_1795255173277_9cab756a.jpg", GameScene.MODAL_CONFIRM, False, None, None),
        (a7_up / "media_1795255107638_886a641d.jpg", GameScene.UNKNOWN_FREE_CURSOR, None, None, None),
        (a7_up / "media_1795249195269_5c024fc5.jpg", GameScene.MODAL_CHOICE, False, None, 3),
        (a7_up / "media_1795250588564_ad5eab9e.jpg", GameScene.MODAL_CHOICE, False, None, 6),
        (old_sc / "state1_normal.png", GameScene.BATTLE_PREP, False, None, None),
        (old_sc / "state2_shop_open.png", GameScene.BATTLE_PREP, True, 4, None),
        (a7_dir / "shop_6slot_verify.png", GameScene.BATTLE_PREP, True, 6, None),
    ]

    available = [c for c in cases if c[0].exists()]
    if not available:
        pytest.skip("本机无可选实机标定截图，跳过实机回归校验")

    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)
    for img_path, exp_scene, exp_shop_open, exp_slots, exp_choices in available:
        raw = np.fromfile(str(img_path), dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        assert frame is not None, f"无法读取实机截图: {img_path}"
        res = detector.detect(frame)
        assert res.scene == exp_scene, f"{img_path.name}: expected {exp_scene}, got {res}"
        if exp_shop_open is not None:
            assert res.shop_open is exp_shop_open, f"{img_path.name}: shop_open mismatch {res}"
        if exp_slots is not None:
            assert res.shop_slot_count == exp_slots, f"{img_path.name}: shop_slot_count mismatch {res}"
        if exp_choices is not None:
            assert res.choice_count == exp_choices, f"{img_path.name}: choice_count mismatch {res}"

    # 额外可选校验：桌面实机截图 media_0.png 中带蓝紫底部发光的真实紫红 [G] 升级卡
    media0_path = a7_dir / ".system_generated" / "steps" / "291" / "media_0.png"
    if media0_path.exists():
        m0_full = cv2.imdecode(np.fromfile(str(media0_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if m0_full is not None and m0_full.shape[0] >= 956 and m0_full.shape[1] >= 1496:
            m0_1280 = cv2.resize(
                m0_full[147:956, 58:1496],
                (STANDARD_WIDTH, STANDARD_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            hsv_m0 = cv2.cvtColor(m0_1280, cv2.COLOR_BGR2HSV)
            assert SceneDetector._has_shop_upgrade_card(hsv_m0, 332, 441) is True
            assert SceneDetector._has_shop_item_card(hsv_m0, 288, 442) is False
            assert SceneDetector._has_shop_item_card(hsv_m0, 450, 604) is True


