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
    """校验 GameScene 枚举值与 SceneDetectionResult 冻结属性契约。"""
    expected_values = {
        "activity_hub",
        "pre_match_setup",
        "squad_edit",
        "battle_prep",
        "modal_choice",
        "battle_combat",
        "settlement",
        "unknown_free_cursor",
    }
    assert {scene.value for scene in GameScene} == expected_values

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

    png_path = tmp_path / f"{GameScene.ACTIVITY_HUB.value}.png"
    ok, buf = cv2.imencode(".png", badge)
    assert ok
    buf.tofile(str(png_path))

    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)
    assert detector.loaded_template_count == 1

    # 标准 1280x720 帧命中
    res = detector.detect(frame)
    assert res.scene == GameScene.ACTIVITY_HUB
    assert res.confidence >= 0.95

    # 非 1280x720（如 1920x1080）帧自动缩放后依然高置信度命中
    frame_1080p = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    res_1080p = detector.detect(frame_1080p)
    assert res_1080p.scene == GameScene.ACTIVITY_HUB
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
        source_frame, GameScene.SQUAD_EDIT, roi_norm=roi_norm
    )
    saved_json = saved_png.with_suffix(".json")
    assert saved_png.exists()
    assert saved_json.exists()

    meta = json.loads(saved_json.read_text(encoding="utf-8"))
    assert meta["scene"] == GameScene.SQUAD_EDIT.value
    assert meta["frame_size"] == [STANDARD_WIDTH, STANDARD_HEIGHT]
    assert len(meta["roi_norm"]) == 4
    assert pytest.approx(meta["roi_norm"], abs=0.005) == list(roi_norm)

    # 先初始化为空检测器，随后 reload_templates 即刻生效
    detector = SceneDetector(templates_dir=templates_dir, debounce_frames=1)
    assert detector.loaded_template_count == 1
    res_hit = detector.detect(source_frame)
    assert res_hit.scene == GameScene.SQUAD_EDIT
    assert res_hit.confidence >= 0.98

    # 将相同图案平移到遥远的右下角（超出原 ROI 搜索窗），验证不会被局部 ROI 误匹配
    relocated_frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 20, dtype=np.uint8)
    relocated_frame[520 : 520 + h, 920 : 920 + w] = badge
    res_miss = detector.detect(relocated_frame)
    assert res_miss.scene == GameScene.UNKNOWN_FREE_CURSOR

    # 测试使用默认 ROI（roi_norm=None）也能正常裁剪并识别
    default_roi = DEFAULT_SCENE_ROIS[GameScene.SETTLEMENT]
    dx = int(round(default_roi[0] * STANDARD_WIDTH))
    dy = int(round(default_roi[1] * STANDARD_HEIGHT))
    dw = int(round(default_roi[2] * STANDARD_WIDTH))
    dh = int(round(default_roi[3] * STANDARD_HEIGHT))
    settlement_frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 18, dtype=np.uint8)
    _draw_distinctive_badge(settlement_frame, x=dx, y=dy, w=dw, h=dh, seed=707)

    calibrator.capture_scene_template(settlement_frame, GameScene.SETTLEMENT)
    detector.reload_templates()
    assert detector.loaded_template_count == 2
    assert detector.detect(settlement_frame).scene == GameScene.SETTLEMENT

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
        sample_frame, GameScene.PRE_MATCH_SETUP, roi_norm=(120 / 1280, 100 / 720, 200 / 1280, 120 / 720)
    )
    assert calibrator.list_installed_templates() == ["pre_match_setup"]

    # 写入变体模板与损坏 JSON，验证不崩溃且仍能命中
    variant_png = templates_dir / "activity_hub__alt.png"
    ok, encoded = cv2.imencode(".png", badge)
    assert ok
    encoded.tofile(str(variant_png))
    (templates_dir / "activity_hub__alt.json").write_text("{corrupt json", encoding="utf-8")

    detector = SceneDetector(templates_dir=templates_dir, debounce_frames=1)
    assert detector.loaded_template_count == 2
    assert detector.detect(sample_frame).scene in (
        GameScene.PRE_MATCH_SETUP,
        GameScene.ACTIVITY_HUB,
    )

    assert calibrator.delete_scene_template(GameScene.PRE_MATCH_SETUP) is True
    assert calibrator.delete_scene_template(GameScene.PRE_MATCH_SETUP) is False
    detector.reload_templates()
    assert detector.loaded_template_count == 1
    assert detector.detect(sample_frame).scene == GameScene.ACTIVITY_HUB

    # 非法 ROI 参数与无效帧均抛出 ValueError
    with pytest.raises(ValueError):
        calibrator.capture_scene_template(sample_frame, GameScene.BATTLE_PREP, roi_norm=(0.2, 0.2, 0.0, 0.1))
    with pytest.raises(ValueError):
        calibrator.capture_scene_template(np.array([], dtype=np.uint8), GameScene.BATTLE_PREP)

