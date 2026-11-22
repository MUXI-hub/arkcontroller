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


def _draw_stronghold_in_game_chrome(frame: np.ndarray) -> None:
    """绘制卫戍协议局内双身份锚点：左上角深红退出门按钮 + 顶部中央深黑状态栏。"""
    # 左上角深红退出门（BGR (28, 30, 145) -> HSV H≈1, S≈205, V=145）
    cv2.rectangle(frame, (18, 14), (94, 74), (28, 30, 145), -1)
    # 顶部中央深黑回合/状态信息栏（V=28 <= 70）
    cv2.rectangle(frame, (410, 6), (870, 66), (22, 24, 28), -1)


def _draw_modal_cyan_chrome(frame: np.ndarray) -> None:
    """绘制卫戍协议决策弹窗左上角青蓝色标题带（BGR (220, 215, 25) -> HSV H≈91）。"""
    cv2.rectangle(frame, (160, 26), (380, 76), (220, 215, 25), -1)


def _make_modal_choice_frame(choice_count: int = 3) -> np.ndarray:
    """构造带卫戍青色标题、四周半透明暗化遮罩与中央横向等距高亮卡片的合成弹窗帧。"""
    frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 24, dtype=np.uint8)
    # 模拟暗化后的微弱背景噪声
    frame[100:600, 140:1140] = 32
    _draw_modal_cyan_chrome(frame)

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
    """构造具备局内双锚点、底部 10 格整备槽条纹与右下角高亮准备按钮的局内休整期合成帧。"""
    frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 48, dtype=np.uint8)
    _draw_stronghold_in_game_chrome(frame)
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
    assert res.covenant_overflow is False
    assert res.emote_panel_open is False
    assert res.confirm_has_dont_ask is True
    assert res.board_mirrored is False
    assert res.covenant_centers_u == ()
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
    assert res_none.shop_open is False

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

    # 1. 默认休整期（局内双锚点 + 右上角粉红准备按钮，无商店双按钮、无角色青蓝血条）
    base = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 45, dtype=np.uint8)
    _draw_stronghold_in_game_chrome(base)
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
    _draw_stronghold_in_game_chrome(base_shop)
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
    elif choice_count == 2:
        cols = [(348, 627), (689, 969)]
        boxes = [(xl, 241, xr, 479) for xl, xr in cols]
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
    frame: np.ndarray,
    cx: float,
    *,
    activated: bool = True,
    theme: str = "t1",
    dim: bool = False,
    clip_right_at: int | None = None,
    clip_left_at: int | None = None,
) -> None:
    """在 1280x720 帧顶部栏下方 (cx, 87) 绘制跨主题圆形盟约徽章与底部文字。

    支持：
    - ``theme="t1"``：电玩紫粉风亮青/蓝环；
    - ``theme="t2"``：乌萨斯复古风银灰环（``S=0`` 无彩度）；
    - ``theme="t3"``：罗德岛战术风青绿环（支持 ``dim=True`` 模拟低亮度半环）；
    - ``clip_right_at`` / ``clip_left_at``：模拟溢出滚动态在视口边缘被裁切的半露徽章。
    """
    layer = frame.copy()
    ix, iy = int(round(cx)), 87
    # 外圈暗晕 + 深色圆心（保证环外/环内均有清晰法向梯度）
    cv2.circle(layer, (ix, iy), 26, (20, 22, 24), -1)
    cv2.circle(layer, (ix, iy), 19, (36, 38, 42), -1)

    if theme == "t2":
        ring_color = (165, 165, 165) if dim else (218, 218, 218)
        cv2.circle(layer, (ix, iy), 22, ring_color, 3)
        cv2.circle(layer, (ix, iy), 10, (92, 92, 92), -1)
    elif theme == "t3":
        if dim:
            cv2.circle(layer, (ix, iy), 22, (132, 146, 138), 2)
            cv2.ellipse(layer, (ix, iy), (22, 22), 0, -60, 140, (165, 200, 120), 3)
        else:
            cv2.circle(layer, (ix, iy), 22, (175, 232, 130), 3)
            cv2.circle(layer, (ix, iy), 11, (70, 95, 78), -1)
    else:
        if activated:
            cv2.circle(layer, (ix, iy), 22, (200, 245, 95), 3)
            cv2.circle(layer, (ix, iy), 12, (82, 115, 42), -1)
        else:
            cv2.circle(layer, (ix, iy), 22, (190, 170, 140), 2)
            cv2.ellipse(layer, (ix, iy), (22, 22), 0, 90, 270, (240, 130, 55), 3)

    # 底部层数与盟约名高对比暗影白字模拟（y=102..126）
    cv2.rectangle(layer, (ix - 18, 102), (ix + 18, 126), (16, 18, 20), -1)
    cv2.line(layer, (ix - 12, 109), (ix + 12, 109), (228, 230, 232), 2)
    cv2.line(layer, (ix - 14, 119), (ix + 14, 119), (210, 215, 218), 2)

    x_min = max(0, clip_left_at if clip_left_at is not None else ix - 30)
    x_max = min(STANDARD_WIDTH, clip_right_at if clip_right_at is not None else ix + 31)
    if x_min < x_max:
        frame[55:132, x_min:x_max] = layer[55:132, x_min:x_max]


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
    _draw_stronghold_in_game_chrome(base)
    cv2.rectangle(base, (200, 150), (1040, 500), (88, 92, 98), -1)
    cv2.rectangle(base, (1120, 100), (1255, 140), (110, 70, 235), -1)

    res0 = detector.detect(base)
    assert res0.scene == GameScene.BATTLE_PREP
    assert res0.covenant_count == 0
    assert res0.has_teammates is False

    for expected_n in (1, 2, 3, 4):
        fn = base.copy()
        for i in range(expected_n):
            cx = 640.0 + (i - (expected_n - 1) * 0.5) * 80.0
            _draw_covenant_badge(fn, cx, activated=(i % 2 == 0))
        rn = detector.detect(fn)
        assert rn.scene == GameScene.BATTLE_PREP
        assert rn.covenant_count == expected_n
        assert rn.covenant_overflow is False


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
    """构造回合结束未用完资金/策略确认弹窗合成帧（含右上模糊粉红按钮残留与局内双锚点）。"""
    frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 55, dtype=np.uint8)
    _draw_stronghold_in_game_chrome(frame)
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
    """构造带局内双锚点、实机右上角粉红准备按钮与右侧黄刷新/蓝冻结双按钮的商店展开基础帧。"""
    frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 38, dtype=np.uint8)
    _draw_stronghold_in_game_chrome(frame)
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

    cases: list[
        tuple[
            Path,
            GameScene,
            bool | None,
            int | None,
            int | None,
            int | None,
            bool | None,
            bool | None,
            int | None,
            bool | None,
        ]
    ] = [
        # v0.3.1 / v0.3.2 / v0.3.3 新增多主题、双灰商店、Boss 战镜像、多盟约与作战期实机截图
        (a7_up / "media_1795325671726_73b3d7cb.jpg", GameScene.BATTLE_PREP, True, 4, None, None, False, None, 3, False),
        (a7_up / "media_1795325671732_1bc865a7.jpg", GameScene.BATTLE_PREP, True, 5, None, None, False, None, 4, False),
        (a7_up / "media_1795325671739_1e44ed7e.jpg", GameScene.BATTLE_PREP, True, 5, None, None, False, None, 7, False),
        (a7_up / "media_1795325671744_308f0863.jpg", GameScene.BATTLE_PREP, True, 5, None, None, False, None, 7, True),
        (a7_up / "media_1795328987598_9f30b6b8.jpg", GameScene.BATTLE_COMBAT, False, None, None, None, False, None, 0, False),
        (a7_up / "media_1795332461252_fe50bbf2.jpg", GameScene.BATTLE_PREP, True, 5, None, None, False, None, 6, False),
        (a7_up / "media_1795334505233_979719ff.jpg", GameScene.BATTLE_PREP, True, 6, None, None, False, None, 7, True),
        # 历史核心实机截图回归
        (a7_up / "media_1795266584081_fc922d8f.jpg", GameScene.MODAL_CHOICE, False, None, 6, None, False, None, None, False),
        (a7_up / "media_1795266595030_fc375d1b.jpg", GameScene.MODAL_CONFIRM, False, None, None, None, False, False, None, False),
        (a7_up / "media_1795267445949_6c2d7322.jpg", GameScene.BATTLE_PREP, False, None, None, None, True, None, 0, False),
        (a7_up / "media_1795267505954_2ea0d702.jpg", GameScene.BATTLE_PREP, True, 4, None, None, False, None, 0, False),
        (a7_up / "media_1795255173277_9cab756a.jpg", GameScene.MODAL_CONFIRM, False, None, None, None, False, True, None, False),
        (a7_up / "media_1795255107638_886a641d.jpg", GameScene.UNKNOWN_FREE_CURSOR, False, None, None, None, False, None, None, False),
        (a7_up / "media_1795249195269_5c024fc5.jpg", GameScene.MODAL_CHOICE, False, None, 3, 0, False, None, None, False),
        (a7_up / "media_1795250588564_ad5eab9e.jpg", GameScene.MODAL_CHOICE, False, None, 6, 4, False, None, None, False),
        (a7_up / "media_1795263414030_b12d9638.jpg", GameScene.MODAL_CHOICE, False, None, 2, 1, False, None, None, False),
        (old_sc / "state1_normal.png", GameScene.BATTLE_PREP, False, None, None, None, False, None, 0, False),
        (old_sc / "state2_shop_open.png", GameScene.BATTLE_PREP, True, 4, None, None, False, None, 0, False),
        (old_sc / "state10_aim_done.png", GameScene.BATTLE_PREP, False, None, None, None, False, None, 1, False),
        (old_sc / "state11_board_unit_selected.png", GameScene.BATTLE_PREP, False, None, None, None, False, None, 1, False),
        (old_sc / "state12_after_q_retreat.png", GameScene.BATTLE_PREP, False, None, None, None, False, None, 1, False),
        (old_sc / "state14_key_w.png", GameScene.BATTLE_PREP, False, None, None, None, False, None, 0, False),
        (a7_dir / "shop_6slot_verify.png", GameScene.BATTLE_PREP, True, 6, None, None, False, None, 0, False),
    ]

    available = [c for c in cases if c[0].exists()]
    if not available:
        pytest.skip("本机无可选实机标定截图，跳过实机回归校验")

    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)
    for (
        img_path,
        exp_scene,
        exp_shop_open,
        exp_slots,
        exp_choices,
        exp_sel_choice,
        exp_emote_open,
        exp_dont_ask,
        exp_covenants,
        exp_covenant_overflow,
    ) in available:
        detector.reset()
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
        if exp_sel_choice is not None:
            assert res.selected_choice_index == exp_sel_choice, (
                f"{img_path.name}: selected_choice_index mismatch {res}"
            )
        if exp_emote_open is not None:
            assert res.emote_panel_open is exp_emote_open, (
                f"{img_path.name}: emote_panel_open mismatch {res}"
            )
        if exp_dont_ask is not None:
            assert res.confirm_has_dont_ask is exp_dont_ask, (
                f"{img_path.name}: confirm_has_dont_ask mismatch {res}"
            )
        if exp_covenants is not None:
            assert res.covenant_count == exp_covenants, (
                f"{img_path.name}: covenant_count mismatch {res}"
            )
            assert len(res.covenant_centers_u) == exp_covenants, (
                f"{img_path.name}: covenant_centers_u len mismatch {res}"
            )
        if exp_covenant_overflow is not None:
            assert res.covenant_overflow is exp_covenant_overflow, (
                f"{img_path.name}: covenant_overflow mismatch {res}"
            )
        exp_mirrored = img_path.name == "media_1795334505233_979719ff.jpg"
        assert res.board_mirrored is exp_mirrored, (
            f"{img_path.name}: board_mirrored mismatch {res}"
        )
        if img_path.name == "media_1795328987598_9f30b6b8.jpg":
            assert res.has_teammates is True, f"{img_path.name}: expected has_teammates=True, got {res}"

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


# =====================================================================
# 7. v0.2.1 专项：实机 1x2 二选一弹窗识别 & 局外界面 100% 防误判负例测试
# =====================================================================


def test_real_style_2choice_modal_and_white_bar(tmp_path: Path) -> None:
    """验证实机暗底亮框 1x2 二选一弹窗识别、左右白条索引判定与实机截图直测。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    # 1. 合成 1x2 二选一：未选中
    f2_idle = _make_real_style_modal_frame(choice_count=2, selected_index=None)
    r2_idle = detector.detect(f2_idle)
    assert r2_idle.scene == GameScene.MODAL_CHOICE
    assert r2_idle.choice_count == 2
    assert r2_idle.selected_choice_index is None
    assert r2_idle.shop_open is False

    # 2. 合成 1x2 二选一：选中左卡(0) 与 选中右卡(1) 弹出底部白条
    f2_sel0 = _make_real_style_modal_frame(choice_count=2, selected_index=0)
    r2_sel0 = detector.detect(f2_sel0)
    assert r2_sel0.scene == GameScene.MODAL_CHOICE
    assert r2_sel0.choice_count == 2
    assert r2_sel0.selected_choice_index == 0

    f2_sel1 = _make_real_style_modal_frame(choice_count=2, selected_index=1)
    r2_sel1 = detector.detect(f2_sel1)
    assert r2_sel1.scene == GameScene.MODAL_CHOICE
    assert r2_sel1.choice_count == 2
    assert r2_sel1.selected_choice_index == 1

    # 2b. 二选一两翼外侧叠加高对比压暗地图斜边与交错砖块纹理（模拟雪山冰面背景），绝不误杀
    f2_textured_wings = _make_real_style_modal_frame(choice_count=2, selected_index=1)
    for row in range(260, 460, 20):
        offset = (row // 20 % 2) * 18
        cv2.line(f2_textured_wings, (150 + offset, row), (305, row + 18), (165, 175, 185), 2)
        cv2.line(f2_textured_wings, (995 + offset, row), (1135, row + 18), (165, 175, 185), 2)
        cv2.line(f2_textured_wings, (305 - offset, row), (155, row + 18), (22, 26, 30), 2)
        cv2.line(f2_textured_wings, (1135 - offset, row), (995, row + 18), (22, 26, 30), 2)
    r2_wings = detector.detect(f2_textured_wings)
    assert r2_wings.scene == GameScene.MODAL_CHOICE
    assert r2_wings.choice_count == 2
    assert r2_wings.selected_choice_index == 1

    # 3. 若本机存在实机二选一截图（更换指令），直接实机验证
    real_2choice = Path(
        r"C:\Users\MUXI\.gemini\antigravity\brain\a7e72e90-d260-48de-94e8-debc493ed124\.user_uploaded\media_1795263414030_b12d9638.jpg"
    )
    if real_2choice.exists():
        raw = np.fromfile(str(real_2choice), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        assert img is not None
        r_real = detector.detect(img)
        assert r_real.scene == GameScene.MODAL_CHOICE
        assert r_real.choice_count == 2
        assert r_real.selected_choice_index == 1
        assert r_real.shop_open is False


def test_out_of_game_screens_never_misclassified(tmp_path: Path) -> None:
    """验证 4 组典型局外界面（编队/干员列表基建/主界面橙红横幅/无青色标题双三卡页）严守 UNKNOWN_FREE_CURSOR。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    # 负例 1：战前编队界面（底部 10 格干员槽条纹 + 右下角亮色「开始行动」大按钮，但无左上深红退出门）
    squad_edit = _make_battle_prep_frame(shop_open=False)
    # 清除卫戍局内双锚点，替换为明日方舟局外常见白色返回箭头与亮色顶部导航栏
    squad_edit[0:85, 0:1280] = (195, 198, 202)
    cv2.rectangle(squad_edit, (20, 18), (95, 62), (240, 240, 240), -1)
    assert detector.detect(squad_edit).scene == GameScene.UNKNOWN_FREE_CURSOR

    # 负例 2：干员列表 / 基建界面（底部规则卡栅 + 顶部明亮理智/龙门币资源栏 + 右下角暗色，无左上深红退出门）
    operator_list = _make_battle_combat_frame()
    operator_list[0:85, 0:140] = (70, 75, 82)
    cv2.rectangle(operator_list, (18, 16), (92, 64), (235, 235, 235), -1)
    assert detector.detect(operator_list).scene == GameScene.UNKNOWN_FREE_CURSOR

    # 负例 3：主界面 / 活动大厅右上角橙红活动横幅（H≈7，即使带顶部暗栏也因色相与缺少深红门免疫误判）
    home_orange_banner = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 52, dtype=np.uint8)
    cv2.rectangle(home_orange_banner, (180, 160), (960, 560), (96, 102, 110), -1)
    # 橙红活动横幅（BGR (20, 75, 240) -> HSV H≈7, S≈234, V=240）
    cv2.rectangle(home_orange_banner, (1115, 95), (1260, 145), (20, 75, 240), -1)
    # 即使故意补上局内双锚点，H=7 橙红横幅也应被收窄后的粉红准备按钮阈值拦截
    home_orange_with_chrome = home_orange_banner.copy()
    _draw_stronghold_in_game_chrome(home_orange_with_chrome)
    assert detector.detect(home_orange_banner).scene == GameScene.UNKNOWN_FREE_CURSOR
    assert detector.detect(home_orange_with_chrome).scene == GameScene.UNKNOWN_FREE_CURSOR

    # 负例 4：无卫戍青色标题与右侧方形收起按钮的局外双卡 / 三卡选择页（亮卡回退 & 暗底亮框均拦截）
    for count in (2, 3):
        ext_bright_cards = _make_modal_choice_frame(choice_count=count)
        # 移除左上角青色标题带，模拟局外分队/难度选择弹窗
        ext_bright_cards[18:95, 140:580] = (24, 24, 24)
        assert detector.detect(ext_bright_cards).scene == GameScene.UNKNOWN_FREE_CURSOR

        ext_dark_cards = _make_real_style_modal_frame(choice_count=count, selected_index=0)
        # 移除左上青色标题带与右侧方形「收起」按钮
        ext_dark_cards[18:95, 140:580] = (58, 58, 58)
        ext_dark_cards[310:410, 1135:1280] = (58, 58, 58)
        assert detector.detect(ext_dark_cards).scene == GameScene.UNKNOWN_FREE_CURSOR


# =====================================================================
# 8. v0.2.2 专项：六选一灰卡、0 资金灰刷新商店、表情面板、联机休整与 ESC 退出确认窗
# =====================================================================


def _make_pc_6choice_frame(
    *,
    picked_indices: tuple[int, ...] = (),
    selected_index: int | None = None,
) -> np.ndarray:
    """构造官方 PC 16:9 六选一（2x3）合成帧，支持指定已被队友选走的均匀浅灰卡与白条选中态。"""
    frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 52, dtype=np.uint8)
    _draw_modal_cyan_chrome(frame)
    # 右侧方形「收起」按钮
    cv2.rectangle(frame, (1203, 325), (1273, 395), (120, 120, 120), -1)
    cv2.rectangle(frame, (1203, 325), (1273, 395), (235, 235, 235), 2)

    cols = [(177, 456), (519, 798), (861, 1140)]
    boxes = [(xl, 92, xr, 330) for xl, xr in cols] + [
        (xl, 393, xr, 627) for xl, xr in cols
    ]
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        if idx in picked_indices:
            # 已被其他玩家选走：均匀低饱和浅灰卡块（无高亮白框，实机浅灰 med≈145..165）
            cv2.rectangle(frame, (x1, y1), (x2, y2), (156, 155, 152), -1)
            cv2.rectangle(frame, (x1 + 24, y1 + 40), (x2 - 24, y2 - 40), (144, 142, 140), -1)
        elif selected_index == idx:
            bot_y = y2 + 28
            cv2.rectangle(frame, (x1, y1), (x2, bot_y), (32, 36, 30), -1)
            cv2.rectangle(frame, (x1, bot_y - 50), (x2, bot_y), (242, 244, 242), -1)
            cv2.rectangle(frame, (x1, y1), (x2, bot_y), (245, 248, 245), 3)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (24, 26, 28), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (218, 220, 220), 3)
    return frame


def test_v022_6choice_pc_layout_with_picked_grey_cards(tmp_path: Path) -> None:
    """验证官方 PC 16:9 六选一含 1~3 张已被队友选走浅灰卡及底部白条均精准识别，全灰 6 卡不误判。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    for picked, sel in (
        ((2,), None),
        ((0, 2), 4),
        ((0, 2, 5), 1),
    ):
        frame = _make_pc_6choice_frame(picked_indices=picked, selected_index=sel)
        res = detector.detect(frame)
        assert res.scene == GameScene.MODAL_CHOICE
        assert res.choice_count == 6
        assert res.selected_choice_index == sel
        assert res.shop_open is False

    # 6 张全部为灰卡（0 张可选暗卡）不命中实机六选一
    all_grey = _make_pc_6choice_frame(picked_indices=(0, 1, 2, 3, 4, 5), selected_index=None)
    all_grey_g = cv2.cvtColor(all_grey, cv2.COLOR_BGR2GRAY)
    all_grey_h = cv2.cvtColor(all_grey, cv2.COLOR_BGR2HSV)
    assert SceneDetector._detect_real_modal_choice(all_grey_g, all_grey_h) is None
    assert detector.detect(all_grey).choice_count != 6


def test_v022_zero_gold_gray_refresh_shop_open_and_slots(tmp_path: Path) -> None:
    """验证买完干员剩 0 资金导致刷新按钮变灰时，商店展开态与 4/5/6 槽位仍 100% 精准识别。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    def _make_zero_gold_base(*, frozen: bool = False) -> np.ndarray:
        frame = _make_base_shop_frame()
        # 将右侧黄色刷新按钮覆写为 0 资金暗灰色（S≈0，refresh_yellow == 0.0）
        cv2.rectangle(frame, (1120, 425), (1265, 485), (92, 92, 92), -1)
        if frozen:
            # 冻结后「解冻」青绿色按钮（BGR (170, 180, 40) -> H≈88, S≈198, V≈180）
            cv2.rectangle(frame, (965, 425), (1110, 485), (170, 180, 40), -1)
        return frame

    # 1. 0 资金灰刷新 + 常规深蓝冻结按钮：4 / 5 / 6 槽均精准识别
    f4 = _make_zero_gold_base(frozen=False)
    _draw_gray_upgrade_card(f4, 332, 441)
    for sx in (450, 612, 774, 936):
        _draw_shop_item_card(f4, sx, sx + 154, sold_out=False)
    r4 = detector.detect(f4)
    assert r4.scene == GameScene.BATTLE_PREP
    assert r4.shop_open is True
    assert r4.shop_slot_count == 4

    f5 = _make_zero_gold_base(frozen=True)
    _draw_gray_upgrade_card(f5, 170, 279)
    for sx in (288, 450, 612, 774, 936):
        _draw_shop_item_card(f5, sx, sx + 154, sold_out=False)
    r5 = detector.detect(f5)
    assert r5.scene == GameScene.BATTLE_PREP
    assert r5.shop_open is True
    assert r5.shop_slot_count == 5

    f6 = _make_zero_gold_base(frozen=False)
    _draw_gray_upgrade_card(f6, 8, 117)
    for sx in (126, 288, 450, 612, 774, 936):
        _draw_shop_item_card(f6, sx, sx + 154, sold_out=False)
    r6 = detector.detect(f6)
    assert r6.scene == GameScene.BATTLE_PREP
    assert r6.shop_open is True
    assert r6.shop_slot_count == 6


def test_v022_emote_panel_and_coop_loading_prep_banner(tmp_path: Path) -> None:
    """验证联机 LOADING 无粉红按钮时通过顶部紫框+放大镜稳判休整期，并精准识别左下 2x3 表情面板。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 48, dtype=np.uint8)
    _draw_stronghold_in_game_chrome(frame)
    # 模拟棋盘区域正常灰阶纹理，避免触发纯色帧守卫
    cv2.rectangle(frame, (200, 120), (1040, 500), (88, 92, 98), -1)
    # 模拟联机等待 LOADING...：右上无粉红准备按钮，但顶部中央有「休息一下」紫框 + 左青/右黄双放大镜
    # 紫框 BGR (190, 65, 175) -> HSV H≈147, S≈168, V=190
    cv2.rectangle(frame, (575, 12), (705, 58), (190, 65, 175), -1)
    # 保持顶部中央暗背景占比仍达标（仅紫框覆盖一部分）
    cv2.rectangle(frame, (595, 22), (685, 48), (22, 24, 28), -1)
    # 左青色放大镜 (BGR (220, 210, 30) -> H≈91) + 右黄色放大镜 (BGR (30, 210, 240) -> H≈25)
    cv2.circle(frame, (338, 34), 18, (220, 210, 30), 6)
    cv2.circle(frame, (944, 34), 18, (30, 210, 240), 6)

    # 未开表情面板时：识别为 BATTLE_PREP, shop_open=False, emote_panel_open=False
    r_closed = detector.detect(frame)
    assert r_closed.scene == GameScene.BATTLE_PREP
    assert r_closed.shop_open is False
    assert r_closed.emote_panel_open is False

    # 在左下 ROI (x=18..302, y=438..628) 绘制深黑圆角面板 + 2x3 青绿霓虹表情图标
    frame_emote = frame.copy()
    cv2.rectangle(frame_emote, (12, 435), (306, 632), (24, 26, 28), -1)
    for row_y in (488, 576):
        for col_x in (74, 165, 255):
            # 青绿霓虹图标（BGR (175, 225, 45) -> HSV H≈82, S≈204, V=225）
            cv2.circle(frame_emote, (col_x, row_y), 26, (175, 225, 45), -1)

    r_open = detector.detect(frame_emote)
    assert r_open.scene == GameScene.BATTLE_PREP
    assert r_open.shop_open is False
    assert r_open.emote_panel_open is True

    # 商店展开时，即便左下残留青绿元素也应压制 emote_panel_open=False
    frame_shop = frame_emote.copy()
    cv2.rectangle(frame_shop, (965, 425), (1110, 485), (230, 140, 20), -1)
    r_shop = detector.detect(frame_shop)
    assert r_shop.scene == GameScene.BATTLE_PREP
    assert r_shop.shop_open is True
    assert r_shop.emote_panel_open is False


def test_v022_esc_confirm_without_dont_ask_silhouette_and_cursor_robustness(
    tmp_path: Path,
) -> None:
    """验证 ESC 退出确认窗（无不再提示、右端深色机械剪影、光标/HUD 压边）稳判 MODAL_CONFIRM 且精准区分 dont_ask。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    # 1. 回合确认弹窗（含「本局不再提示」文字行）-> confirm_has_dont_ask=True
    round_confirm = _make_modal_confirm_frame(with_top_ready_pink=True)
    r_round = detector.detect(round_confirm)
    assert r_round.scene == GameScene.MODAL_CONFIRM
    assert r_round.confirm_has_dont_ask is True

    # 2. ESC 暂时离开模拟确认窗：清空「本局不再提示」行，并在右端叠加深暗机械剪影 + 黑/灰带局部压边光标
    esc_confirm = _make_modal_confirm_frame(with_top_ready_pink=True)
    esc_confirm[385:422, 540:740] = (16, 16, 16)
    # 模拟右下确认栏最右端 x=1060..1260 的深暗机械剪影（V=28，极低亮度）
    esc_confirm[449:521, 1060:1280] = (12, 12, 28)
    # 模拟纯红区压暗至 V=48（实机深红渐变暗部）
    esc_confirm[449:521, 980:1060] = (16, 16, 48)
    # 模拟游戏白色鼠标或青色 HUD 选框局部压过中央黑底边缘与灰取消栏（曾导致脆弱 std 超标）
    cv2.rectangle(esc_confirm, (40, 210), (95, 250), (245, 245, 245), 3)
    cv2.circle(esc_confirm, (120, 485), 14, (255, 230, 0), 3)

    r_esc = detector.detect(esc_confirm)
    assert r_esc.scene == GameScene.MODAL_CONFIRM
    assert r_esc.confidence == 0.95
    assert r_esc.shop_open is False
    assert r_esc.confirm_has_dont_ask is False


# =====================================================================
# 9. v0.3.2 专项：三套卫戍主题适配、HSV-V 多盟约与 >7 溢出、去色相表情面板
# =====================================================================


def _make_theme2_ursus_shop_frame(*, shop_open: bool = True) -> np.ndarray:
    """构造主题二（乌萨斯复古辉光管风）休整期合成帧（浅灰绿准备胶囊 + 右金黄放大镜 + 深青绿冻结）。"""
    frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 46, dtype=np.uint8)
    _draw_stronghold_in_game_chrome(frame)
    cv2.rectangle(frame, (190, 150), (1050, 500), (88, 92, 95), -1)
    # 右侧金黄放大镜图标（BGR (35, 130, 195) -> H≈15, S≈209, V=195）
    cv2.circle(frame, (944, 34), 18, (35, 130, 195), 6)
    # 主题二右上角浅灰绿内胶囊准备按钮 + 上方深色暗边
    cv2.rectangle(frame, (1120, 84), (1256, 99), (36, 42, 38), -1)
    cv2.rectangle(frame, (1124, 99), (1256, 138), (180, 195, 165), -1)
    if shop_open:
        # 主题二深青绿「冻结 [S]」+ 金黄「刷新 [D]」
        cv2.rectangle(frame, (965, 425), (1110, 485), (105, 110, 52), -1)
        cv2.rectangle(frame, (1120, 425), (1265, 485), (30, 195, 225), -1)
    return frame


def _make_theme3_rhodes_shop_frame(
    *, shop_open: bool = True, use_top_teal_only: bool = False
) -> np.ndarray:
    """构造主题三（罗德岛战术青绿风）休整期合成帧（青绿准备或顶青绿框 + 天蓝冻结）。"""
    frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 46, dtype=np.uint8)
    _draw_stronghold_in_game_chrome(frame)
    cv2.rectangle(frame, (190, 150), (1050, 500), (84, 90, 96), -1)
    cv2.circle(frame, (944, 34), 18, (35, 135, 200), 6)
    if use_top_teal_only:
        # 仅靠顶部中央青绿八角框 + 右金黄放大镜识别（如联机等待 LOADING）
        cv2.rectangle(frame, (570, 10), (710, 58), (145, 175, 65), -1)
        cv2.rectangle(frame, (596, 22), (684, 46), (22, 26, 28), -1)
    else:
        # 主题三右上青绿「准备就绪」按钮（BGR (175, 205, 55) -> H≈84, S≈187, V=205）
        cv2.rectangle(frame, (1118, 98), (1258, 142), (175, 205, 55), -1)
    if shop_open:
        # 主题三天蓝「冻结 [S]」+ 暖黄「刷新 [D]」
        cv2.rectangle(frame, (965, 425), (1110, 485), (225, 195, 100), -1)
        cv2.rectangle(frame, (1120, 425), (1265, 485), (35, 210, 235), -1)
    return frame


def _draw_themed_shop_item_card(
    frame: np.ndarray,
    x1: int,
    x2: int,
    *,
    theme: str,
    sold_out_with_cursor: bool = False,
) -> None:
    """绘制主题二红棕底条 / 主题三青绿底条商品卡，或带白色鼠标压边的深灰 NO INFO 空槽。"""
    if sold_out_with_cursor:
        cv2.rectangle(frame, (x1, 535), (x2, 708), (46, 48, 52), -1)
        cv2.rectangle(frame, (x1, 535), (x2, 708), (160, 165, 170), 2)
        frame[690:712, x1 + 2 : x2 - 2] = (24, 26, 28)
        # 在空槽右半边叠加纯白游戏大光标，拉高局部 std（验证中位数+p75 抗压边鲁棒分支）
        cv2.rectangle(frame, (x2 - 34, 595), (x2 - 8, 623), (250, 250, 250), -1)
        return

    cv2.rectangle(frame, (x1, 535), (x2, 708), (186, 190, 195), -1)
    if theme == "t2":
        # 主题二红棕底条（BGR (45, 58, 118) -> H≈5, S≈158, V=118）
        cv2.rectangle(frame, (x1 + 2, 690), (x2 - 2, 712), (45, 58, 118), -1)
    else:
        # 主题三青绿底条（BGR (95, 108, 62) -> H≈81, S≈109, V=108）
        cv2.rectangle(frame, (x1 + 2, 690), (x2 - 2, 712), (95, 108, 62), -1)
    cv2.rectangle(frame, (x1, 535), (x2, 708), (228, 232, 235), 2)


def _draw_dark_screen_upgrade_card(
    frame: np.ndarray, x1: int, x2: int, *, style: str = "t2_crt"
) -> None:
    """绘制暗屏升级卡：主题二绿黑 CRT 屏（``t2_crt``）或主题三灰态 ``LEVEL III`` 屏（``t3_grey``）。"""
    if style == "t2_crt":
        cv2.rectangle(frame, (x1, 535), (x2, 708), (58, 72, 52), -1)
        # 绿黑 CRT 屏幕（BGR (55, 85, 48) -> H≈66, S≈111, V=85）
        cv2.rectangle(frame, (x1 + 6, 552), (x2 - 6, 628), (55, 85, 48), -1)
        cv2.rectangle(frame, (x1 + 24, 578), (x2 - 24, 594), (165, 205, 145), -1)
        cv2.rectangle(frame, (x1 + 4, 688), (x2 - 4, 712), (52, 62, 48), -1)
        cv2.rectangle(frame, (x1, 535), (x2, 708), (185, 195, 180), 2)
    else:
        cv2.rectangle(frame, (x1, 535), (x2, 708), (68, 70, 72), -1)
        # 主题三暗灰 LEVEL 屏幕（V≈60）+ 白色 LEVEL III 字样（约 10% 面积）
        cv2.rectangle(frame, (x1 + 6, 552), (x2 - 6, 628), (56, 58, 60), -1)
        cv2.rectangle(frame, (x1 + 20, 576), (x2 - 20, 592), (215, 218, 220), -1)
        cv2.rectangle(frame, (x1 + 4, 688), (x2 - 4, 712), (62, 64, 66), -1)
        cv2.rectangle(frame, (x1, 535), (x2, 708), (205, 210, 215), 2)


def test_v032_three_themes_prep_shop_and_slots(tmp_path: Path) -> None:
    """验证主题二（乌萨斯）、主题三（罗德岛）休整期、商店展开、4/5/6 槽与 Toast 遮挡兜底。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    # 1. 主题二乌萨斯 4 槽（绿黑 CRT 暗屏升级卡 + 红棕底条商品卡 + 深青绿冻结按钮）
    f_t2_4 = _make_theme2_ursus_shop_frame(shop_open=True)
    _draw_dark_screen_upgrade_card(f_t2_4, 332, 441, style="t2_crt")
    for sx in (450, 612, 774, 936):
        _draw_themed_shop_item_card(f_t2_4, sx, sx + 154, theme="t2")
    r_t2_4 = detector.detect(f_t2_4)
    assert r_t2_4.scene == GameScene.BATTLE_PREP
    assert r_t2_4.shop_open is True
    assert r_t2_4.shop_slot_count == 4

    # 1b. 主题二缺少上方暗边时不得误把亮背景认成乌萨斯准备胶囊
    f_t2_no_dark = _make_theme2_ursus_shop_frame(shop_open=False)
    f_t2_no_dark[80:99, 1115:1265] = (200, 205, 200)
    assert detector.detect(f_t2_no_dark).scene == GameScene.UNKNOWN_FREE_CURSOR

    # 2. 主题三罗德岛 5 槽（灰态 LEVEL III 暗屏升级卡 + 青绿底条商品卡 + 天蓝冻结按钮）
    f_t3_5 = _make_theme3_rhodes_shop_frame(shop_open=True)
    _draw_dark_screen_upgrade_card(f_t3_5, 170, 279, style="t3_grey")
    for sx in (288, 450, 612, 774, 936):
        _draw_themed_shop_item_card(f_t3_5, sx, sx + 154, theme="t3")
    r_t3_5 = detector.detect(f_t3_5)
    assert r_t3_5.scene == GameScene.BATTLE_PREP
    assert r_t3_5.shop_open is True
    assert r_t3_5.shop_slot_count == 5

    # 3. 主题三满级无升级卡 + 第 6 槽深灰 NO INFO 空槽被白色大光标压边 -> 仍精准判 6 槽
    f_t3_6 = _make_theme3_rhodes_shop_frame(shop_open=True, use_top_teal_only=True)
    _draw_themed_shop_item_card(f_t3_6, 126, 280, theme="t3", sold_out_with_cursor=True)
    for sx in (288, 450, 612, 774, 936):
        _draw_themed_shop_item_card(f_t3_6, sx, sx + 154, theme="t3")
    r_t3_6 = detector.detect(f_t3_6)
    assert r_t3_6.scene == GameScene.BATTLE_PREP
    assert r_t3_6.shop_open is True
    assert r_t3_6.shop_slot_count == 6

    # 4. 主题三亮场屏蔽：叠加青绿准备按钮后，绝不误触发实机暗底 MODAL_CHOICE
    f_modal_with_t3 = _make_real_style_modal_frame(choice_count=3)
    cv2.rectangle(f_modal_with_t3, (1118, 98), (1258, 142), (175, 205, 55), -1)
    assert (
        SceneDetector._detect_real_modal_choice(
            cv2.cvtColor(f_modal_with_t3, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(f_modal_with_t3, cv2.COLOR_BGR2HSV),
        )
        is None
    )

    # 5. 防关闭防截屏后纯白 Toast 瞬态遮挡顶部紫框：仅剩左右双放大镜共现仍稳守 BATTLE_PREP
    f_toast = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 48, dtype=np.uint8)
    _draw_stronghold_in_game_chrome(f_toast)
    cv2.rectangle(f_toast, (190, 150), (1050, 500), (88, 92, 96), -1)
    cv2.circle(f_toast, (338, 34), 18, (220, 210, 30), 6)
    cv2.circle(f_toast, (944, 34), 18, (35, 185, 225), 6)
    # 模拟纯白 Toast 横幅完全覆盖顶部中央紫框区域
    cv2.rectangle(f_toast, (555, 42), (725, 64), (245, 245, 245), -1)
    r_toast = detector.detect(f_toast)
    assert r_toast.scene == GameScene.BATTLE_PREP
    assert r_toast.confidence == 0.82


def test_v032_vchannel_covenants_cross_theme_and_overflow(tmp_path: Path) -> None:
    """验证 HSV-V 通道跨主题盟约识别（银灰 3 / 青绿 4 & 7 抗单枚弱分 / >7 左右缘溢出态与作战期同步）。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    # 1. 主题二银灰环（S=0）3 枚居中对称 -> count=3, overflow=False
    f_t2_3 = _make_theme2_ursus_shop_frame(shop_open=False)
    for i in range(3):
        cx = 640.0 + (i - 1.0) * 80.0
        _draw_covenant_badge(f_t2_3, cx, theme="t2")
    r_t2_3 = detector.detect(f_t2_3)
    assert r_t2_3.scene == GameScene.BATTLE_PREP
    assert r_t2_3.covenant_count == 3
    assert r_t2_3.covenant_overflow is False

    # 2. 主题三青绿环 4 枚 & 7 枚（第 3 枚模拟低亮度弱环，验证多数门限不回退 n=1）
    for expected_n in (4, 7):
        f_t3 = _make_theme3_rhodes_shop_frame(shop_open=False)
        for i in range(expected_n):
            cx = 640.0 + (i - (expected_n - 1) * 0.5) * 80.0
            _draw_covenant_badge(f_t3, cx, theme="t3", dim=(i == 2))
        r_t3 = detector.detect(f_t3)
        assert r_t3.scene == GameScene.BATTLE_PREP
        assert r_t3.covenant_count == expected_n
        assert r_t3.covenant_overflow is False

    # 3. 溢出态 A（初始未滑动）：左对齐 7 枚 [379 + 80*i] + 右缘 cx=938 半露左半圈 -> (7, True)
    f_ov_right = _make_theme3_rhodes_shop_frame(shop_open=False)
    for i in range(7):
        _draw_covenant_badge(f_ov_right, 379.0 + i * 80.0, theme="t3")
    _draw_covenant_badge(f_ov_right, 938.0, theme="t3", clip_right_at=940)
    r_ov_right = detector.detect(f_ov_right)
    assert r_ov_right.scene == GameScene.BATTLE_PREP
    assert r_ov_right.covenant_count == 7
    assert r_ov_right.covenant_overflow is True

    # 4. 溢出态 B（向右滚动 1 槽后）：7 枚 [379 + 80*i] + 左缘 cx=299 半露右半圈 -> (7, True)
    f_ov_left = _make_theme3_rhodes_shop_frame(shop_open=False)
    for i in range(7):
        _draw_covenant_badge(f_ov_left, 379.0 + i * 80.0, theme="t3")
    _draw_covenant_badge(f_ov_left, 299.0, theme="t3", clip_left_at=297)
    r_ov_left = detector.detect(f_ov_left)
    assert r_ov_left.scene == GameScene.BATTLE_PREP
    assert r_ov_left.covenant_count == 7
    assert r_ov_left.covenant_overflow is True

    # 5. 作战期（BATTLE_COMBAT）同步透出 covenant_count 与 covenant_overflow
    f_combat_ov = _make_battle_combat_frame()
    for i in range(7):
        _draw_covenant_badge(f_combat_ov, 379.0 + i * 80.0, theme="t2")
    _draw_covenant_badge(f_combat_ov, 938.0, theme="t2", clip_right_at=940)
    r_combat = detector.detect(f_combat_ov)
    assert r_combat.scene == GameScene.BATTLE_COMBAT
    assert r_combat.covenant_count == 7
    assert r_combat.covenant_overflow is True


def test_v032_hue_agnostic_emote_panel_sparse_pages(tmp_path: Path) -> None:
    """验证去色相结构化表情面板检测：第 3 页仅 1 枚暖黄表情稳报 True，无外框/无分页点负例报 False。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    base = _make_theme3_rhodes_shop_frame(shop_open=False)
    # 铺亮背景衬托深黑面板的上/右/下三边外框（确保外框梯度远高于 18/14/12）
    cv2.rectangle(base, (0, 415), (340, 660), (145, 148, 152), -1)

    # 1. 第 3 页仅左上角 1 枚暖黄非青绿表情（B=20 彻底不含 H∈[65..102]），其余 5 格纯黑 + 底部居中分页圆点
    f_page3 = base.copy()
    cv2.rectangle(f_page3, (16, 436), (304, 631), (22, 24, 26), -1)
    # 仅第 1 格暖黄高亮表情（BGR (20, 190, 235) -> H≈24）
    cv2.circle(f_page3, (74, 488), 24, (20, 190, 235), -1)
    # 底部居中分页圆点 y=616..630, x=125..195（■ · · 高白对比 + 周边纯黑底）
    cv2.rectangle(f_page3, (144, 621), (156, 626), (235, 235, 235), -1)
    cv2.circle(f_page3, (166, 623), 3, (215, 215, 215), -1)
    cv2.circle(f_page3, (178, 623), 3, (215, 215, 215), -1)

    r_page3 = detector.detect(f_page3)
    assert r_page3.scene == GameScene.BATTLE_PREP
    assert r_page3.emote_panel_open is True

    # 2. 负例 A：擦除底部居中分页圆点（且无青绿表情）-> 必须报 False
    f_no_dots = f_page3.copy()
    f_no_dots[616:630, 125:195] = (22, 24, 26)
    r_no_dots = detector.detect(f_no_dots)
    assert r_no_dots.scene == GameScene.BATTLE_PREP
    assert r_no_dots.emote_panel_open is False

    # 3. 负例 B：破坏右侧外框（延伸黑底覆盖 x=298..320 消除右边框边缘）-> 必须报 False
    f_no_right_edge = f_page3.copy()
    cv2.rectangle(f_no_right_edge, (295, 436), (335, 631), (22, 24, 26), -1)
    r_no_edge = detector.detect(f_no_right_edge)
    assert r_no_edge.scene == GameScene.BATTLE_PREP
    assert r_no_edge.emote_panel_open is False


def _draw_combat_cost_bar(frame: np.ndarray, *, with_progress_bar: bool = True) -> None:
    """在 1280x720 帧右下角 (x=1158..1278, y=592..652) 绘制实机作战期「◆ C 23」费用栏与底部白色进度条。"""
    # 深色费用栏底板（V=32 <= 75）
    cv2.rectangle(frame, (1158, 592), (1278, 652), (28, 30, 32), -1)
    # 纯白低饱和「◆ C 23」图标与数字色块（占比约 22% >= 0.10，S=0, V=242 >= 205）
    cv2.rectangle(frame, (1168, 602), (1188, 624), (242, 242, 242), -1)
    cv2.rectangle(frame, (1198, 598), (1266, 628), (240, 240, 240), -1)
    # 底部贯穿白色费用回复进度条（y=642..648，行均值约 235 >= 140）
    if with_progress_bar:
        cv2.rectangle(frame, (1160, 642), (1276, 648), (236, 236, 236), -1)


def test_v032_combat_cost_bar_detection_and_teammate_drawer(tmp_path: Path) -> None:
    """验证实机作战期右下费用栏「◆ C」直判 BATTLE_COMBAT，根除右下亮地形与队友抽屉误判为休整期+商店展开。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    # 1. 构造作战期 + 底部 10 格条纹 + 右下高饱和土黄地面（故意使 ready_vivid_ratio≈0.55 落入旧 2b 休整误判区间）
    #    + 左侧 4 名联机队友头像框 + 右下费用栏「◆ C 23」+ 顶部 3 枚盟约
    frame = _make_battle_prep_frame(shop_open=True)
    # 右下角改为高饱和土黄沙地背景（覆盖原准备按钮），再在其上绘制费用栏
    cv2.rectangle(frame, (1035, 555), (1279, 710), (45, 165, 220), -1)
    _draw_combat_cost_bar(frame, with_progress_bar=True)
    # 左侧 4 名联机队友头像框（匹配 _detect_teammates_heuristic）
    for mx, my, mw, mh in (
        (19, 160, 67, 66),
        (19, 235, 67, 67),
        (19, 311, 67, 66),
        (19, 387, 67, 66),
    ):
        cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (75, 78, 84), -1)
        cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (238, 240, 242), 2)
    # 顶部 3 枚银灰盟约
    for i in range(3):
        _draw_covenant_badge(frame, 640.0 + (i - 1.0) * 80.0, theme="t2")

    res = detector.detect(frame)
    assert res.scene == GameScene.BATTLE_COMBAT
    assert res.confidence == 0.86
    assert res.shop_open is False
    assert res.shop_slot_count is None
    assert res.has_teammates is True
    assert res.covenant_count == 3
    assert res.covenant_overflow is False
    assert res.emote_panel_open is False

    # 2. 抹除底部白色回复进度条 -> _has_real_pc_combat_cost_bar 必须返回 False
    no_bar = frame.copy()
    _draw_combat_cost_bar(no_bar, with_progress_bar=False)
    assert (
        SceneDetector._has_real_pc_combat_cost_bar(
            cv2.cvtColor(no_bar, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(no_bar, cv2.COLOR_BGR2HSV),
        )
        is False
    )

    # 3. 缺少局内双锚点（局外画面）即使右下恰好存在高对比白条，也必须严守 UNKNOWN_FREE_CURSOR
    out_of_game = frame.copy()
    out_of_game[0:85, 0:1280] = (195, 198, 202)
    assert detector.detect(out_of_game).scene == GameScene.UNKNOWN_FREE_CURSOR

    # 4. 休整期强特征（2a）优先级高于 2a.5：存在右上粉红准备按钮时仍稳定判为 BATTLE_PREP
    prep_priority = frame.copy()
    cv2.rectangle(prep_priority, (1120, 100), (1255, 140), (110, 70, 235), -1)
    assert detector.detect(prep_priority).scene == GameScene.BATTLE_PREP


# =====================================================================
# 10. v0.3.3 专项：双灰按钮商店结构识别、三主题置灰、Boss 镜像锁存与任意相位盟约
# =====================================================================


def _desat_regions_for_test(
    img: np.ndarray, boxes: list[tuple[int, int, int, int]], v_scale: float
) -> np.ndarray:
    """将指定矩形区域去饱和（S=0）并按 v_scale 压缩亮度，仿真按钮置灰状态。"""
    out = img.copy()
    for x1, y1, x2, y2 in boxes:
        patch = out[y1:y2, x1:x2]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = 0
        hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) * v_scale, 0, 255).astype(
            np.uint8
        )
        out[y1:y2, x1:x2] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return out


def _make_dual_gray_shop_frame(*, slot_count: int = 5, dim_upgrade: bool = False) -> np.ndarray:
    """构造冻结 [S] 与刷新 [D] 同时 0 饱和度置灰的实机结构商店展开合成帧。"""
    frame = np.full((STANDARD_HEIGHT, STANDARD_WIDTH, 3), 52, dtype=np.uint8)
    _draw_stronghold_in_game_chrome(frame)
    cv2.rectangle(frame, (190, 140), (1050, 415), (92, 96, 102), -1)
    # 右上粉红准备就绪按钮
    cv2.rectangle(frame, (1120, 100), (1255, 140), (110, 70, 235), -1)
    # 右侧「剩余可放置角色：X」上移黑底白字横幅 y=375..405, x=1105..1270
    cv2.rectangle(frame, (1105, 375), (1270, 405), (22, 24, 26), -1)
    cv2.rectangle(frame, (1125, 384), (1248, 396), (228, 228, 228), -1)
    # 冻结 [S] 与刷新 [D] 同时 0 饱和度置灰（下方衬托暗底形成清晰底缘梯度）
    cv2.rectangle(frame, (960, 424), (1270, 496), (30, 30, 30), -1)
    cv2.rectangle(frame, (970, 428), (1105, 480), (108, 108, 108), -1)
    cv2.rectangle(frame, (1125, 428), (1260, 480), (116, 116, 116), -1)
    # 底部商店暗底托盘（提供 3 条 5px 卡间暗缝 x∈{608,770,932}）
    cv2.rectangle(frame, (0, 528), (1100, 718), (22, 24, 26), -1)

    if slot_count == 6:
        item_xs = (126, 288, 450, 612, 774, 936)
        upg_span = (8, 117)
    elif slot_count == 5:
        item_xs = (288, 450, 612, 774, 936)
        upg_span = (170, 279)
    else:
        item_xs = (450, 612, 774, 936)
        upg_span = (332, 441)

    for sx in item_xs:
        _draw_shop_item_card(frame, sx, sx + 154, sold_out=False)

    ux1, ux2 = upg_span
    if dim_upgrade:
        # 仿真 V*0.55 大幅压暗置灰的升级卡（触发低亮置灰兜底分支）
        _draw_dark_screen_upgrade_card(frame, ux1, ux2, style="t3_grey")
        frame = _desat_regions_for_test(frame, [(ux1, 530, ux2, 714)], 0.55)
    else:
        _draw_gray_upgrade_card(frame, ux1, ux2)
    return frame


def test_v033_dual_gray_shop_and_three_theme_desat_robustness(tmp_path: Path) -> None:
    """验证双灰按钮零色相结构通道、低亮置灰升级卡、Theme 2 准备置灰及三主题全量置灰仿真。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    # 1. 双灰按钮商店展开（4/5/6 槽 + 低亮 V*0.55 置灰升级卡）
    for slots in (4, 5, 6):
        detector.reset()
        frame = _make_dual_gray_shop_frame(slot_count=slots, dim_upgrade=True)
        res = detector.detect(frame)
        assert res.scene == GameScene.BATTLE_PREP
        assert res.shop_open is True
        assert res.shop_slot_count == slots

    # 1b. 破坏上移黑底白字横幅或卡间暗缝 -> 零色相结构通道必须严守 shop_open=False
    f_no_banner = _make_dual_gray_shop_frame(slot_count=5)
    cv2.rectangle(f_no_banner, (1105, 375), (1270, 405), (120, 122, 124), -1)
    assert detector.detect(f_no_banner).shop_open is False

    f_no_gaps = _make_dual_gray_shop_frame(slot_count=5)
    for gx in (608, 770, 932):
        cv2.rectangle(f_no_gaps, (gx - 4, 540), (gx + 4, 700), (150, 150, 150), -1)
    assert detector.detect(f_no_gaps).shop_open is False

    # 2. 乌萨斯 Theme 2 准备按钮置灰兜底（暖橙辉光管 + 上方暗边 + 灰阶胶囊 + 右金黄放大镜）
    f_t2_gray = _make_theme2_ursus_shop_frame(shop_open=False)
    # 顶部暖橙辉光管区域 y=28..84, x=1132..1246（BGR (28, 120, 225) -> H≈14, S≈223, V=225）
    cv2.rectangle(f_t2_gray, (1142, 34), (1236, 78), (28, 120, 225), -1)
    # 将右上准备胶囊去饱和并压暗至 V*0.60（原 t2_sage 必然失效）
    f_t2_gray = _desat_regions_for_test(f_t2_gray, [(1110, 99, 1265, 148)], 0.60)
    detector.reset()
    r_t2_gray = detector.detect(f_t2_gray)
    assert r_t2_gray.scene == GameScene.BATTLE_PREP
    assert r_t2_gray.confidence == 0.84

    # 去掉辉光管暖橙色 -> 不应误触发 Theme 2 置灰兜底
    f_t2_no_nixie = f_t2_gray.copy()
    cv2.rectangle(f_t2_no_nixie, (1130, 26), (1250, 84), (52, 54, 56), -1)
    detector.reset()
    assert detector.detect(f_t2_no_nixie).scene == GameScene.UNKNOWN_FREE_CURSOR

    # 3. 可选实机三主题全按钮置灰仿真（V*0.55 / 0.75 / 0.95，覆盖 T1/T2/T3 共 21 组）
    a7_up = Path(
        r"C:\Users\MUXI\.gemini\antigravity\brain\a7e72e90-d260-48de-94e8-debc493ed124\.user_uploaded"
    )
    real_theme_samples = [
        (a7_up / "media_1795332461252_fe50bbf2.jpg", 5),
        (a7_up / "media_1795334505233_979719ff.jpg", 6),
        (a7_up / "media_1795267505954_2ea0d702.jpg", 4),
        (a7_up / "media_1795325671726_73b3d7cb.jpg", 4),
        (a7_up / "media_1795325671732_1bc865a7.jpg", 5),
        (a7_up / "media_1795325671739_1e44ed7e.jpg", 5),
        (a7_up / "media_1795325671744_308f0863.jpg", 5),
    ]
    desat_boxes = [(955, 418, 1275, 492), (1110, 82, 1272, 152), (5, 515, 445, 718)]
    for path, exp_slots in real_theme_samples:
        if not path.exists():
            continue
        raw = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert raw is not None
        img_std = (
            cv2.resize(raw, (STANDARD_WIDTH, STANDARD_HEIGHT), interpolation=cv2.INTER_AREA)
            if raw.shape[:2] != (STANDARD_HEIGHT, STANDARD_WIDTH)
            else raw
        )
        for v_scale in (0.55, 0.75, 0.95):
            detector.reset()
            sim = _desat_regions_for_test(img_std, desat_boxes, v_scale)
            r_sim = detector.detect(sim)
            assert r_sim.scene == GameScene.BATTLE_PREP, f"{path.name} V*{v_scale}: {r_sim}"
            assert r_sim.shop_open is True, f"{path.name} V*{v_scale}: {r_sim}"
            assert r_sim.shop_slot_count == exp_slots, f"{path.name} V*{v_scale}: {r_sim}"


def _draw_stage_landmarks(
    frame: np.ndarray, *, mirrored: bool, shop_open: bool, mode: str = "both"
) -> None:
    """在 3D 舞台左右 ROI 绘制我方发光青蓝 3D 防御箱（>=50x50 大连通框）与蓝紫小电视终端。"""
    # 蓝紫小电视（BGR (215, 85, 40) -> H≈112, S≈207, V=215）
    tv_bgr = (215, 85, 40)
    # 发光青蓝 3D 防御箱（BGR (225, 200, 70) -> H≈95, S≈175, V=225）
    box_bgr = (225, 200, 70)
    if shop_open:
        tv_rect = (775, 180, 855, 245) if mirrored else (275, 180, 355, 245)
        box_rect = (735, 300, 825, 385) if mirrored else (310, 300, 400, 385)
    else:
        tv_rect = (830, 235, 930, 325) if mirrored else (220, 235, 320, 325)
        box_rect = (790, 385, 905, 495) if mirrored else (255, 385, 370, 495)

    if mode in ("tv", "both"):
        cv2.rectangle(frame, tv_rect[:2], tv_rect[2:], tv_bgr, -1)
    if mode in ("box", "both"):
        cv2.rectangle(frame, box_rect[:2], box_rect[2:], box_bgr, 8)


def test_v033_boss_mirrored_detection_and_latch_lifecycle(tmp_path: Path) -> None:
    """验证 Boss 战右侧镜像双通道检测、小碎点滤噪、防抖与遮挡帧锁存及出局复位完整生命周期。"""
    # 1. 双通道单帧检测（tv 通道 & 纯 box 通道，覆盖 shop_open=True/False；小碎点不误判）
    for shop_open in (True, False):
        for mode in ("tv", "box"):
            f_mirr = _make_base_shop_frame() if shop_open else _make_theme3_rhodes_shop_frame(shop_open=False)
            _draw_stage_landmarks(f_mirr, mirrored=True, shop_open=shop_open, mode=mode)
            hsv_m = cv2.cvtColor(f_mirr, cv2.COLOR_BGR2HSV)
            assert SceneDetector._detect_board_mirrored_heuristic(
                hsv_m, shop_open=shop_open, unit_selected=False
            ) is True
            assert SceneDetector._observe_board_mirrored_heuristic(
                hsv_m, shop_open=shop_open, unit_selected=True
            ) is None

            f_norm = _make_base_shop_frame() if shop_open else _make_theme3_rhodes_shop_frame(shop_open=False)
            _draw_stage_landmarks(f_norm, mirrored=False, shop_open=shop_open, mode=mode)
            hsv_n = cv2.cvtColor(f_norm, cv2.COLOR_BGR2HSV)
            assert SceneDetector._observe_board_mirrored_heuristic(
                hsv_n, shop_open=shop_open, unit_selected=False
            ) is False

    # 1b. 右侧仅存在 <50x50 的小青蓝碎点图标 -> 7x7 膨胀大连通域过滤必须拦掉
    f_small = _make_base_shop_frame()
    cv2.rectangle(f_small, (760, 320), (788, 348), (225, 200, 70), -1)
    assert SceneDetector._detect_board_mirrored_heuristic(
        cv2.cvtColor(f_small, cv2.COLOR_BGR2HSV), shop_open=True, unit_selected=False
    ) is False

    # 2. 完整锁存状态机生命周期（debounce_frames=2）
    det = SceneDetector(templates_dir=tmp_path, debounce_frames=2)
    f_boss_prep = _make_base_shop_frame()
    _draw_stage_landmarks(f_boss_prep, mirrored=True, shop_open=True, mode="both")

    # 连续 2 帧进入 Boss 战镜像休整期 -> 锁存 True
    det.detect(f_boss_prep)
    r_enter = det.detect(f_boss_prep)
    assert r_enter.scene == GameScene.BATTLE_PREP
    assert r_enter.board_mirrored is True

    # 瞬态 1 帧纯黑过场 -> 防抖抑制，且不得清零镜像锁存
    black = np.zeros((STANDARD_HEIGHT, STANDARD_WIDTH, 3), dtype=np.uint8)
    r_blink = det.detect(black)
    assert r_blink.scene == GameScene.BATTLE_PREP
    assert r_blink.board_mirrored is True

    # 打开角色特写详情（unit_selected=True，无地标可测）-> 保持镜像锁存 True
    f_unit_sel = _make_theme3_rhodes_shop_frame(shop_open=False)
    cv2.rectangle(f_unit_sel, (10, 327), (420, 343), (235, 195, 35), -1)
    r_usel = det.detect(f_unit_sel)
    assert r_usel.scene == GameScene.BATTLE_PREP
    assert r_usel.unit_selected is True
    assert r_usel.board_mirrored is True

    # 局内弹出三选一决策窗（连续 2 帧切至 MODAL_CHOICE）-> 维持并透出锁存 True
    f_modal = _make_real_style_modal_frame(choice_count=3)
    det.detect(f_modal)
    r_modal = det.detect(f_modal)
    assert r_modal.scene == GameScene.MODAL_CHOICE
    assert r_modal.board_mirrored is True

    # 切入无地标作战期（如展开 4 人队友抽屉）-> 维持锁存 True
    f_combat_occ = _make_battle_prep_frame(shop_open=False)
    cv2.rectangle(f_combat_occ, (1035, 555), (1279, 710), (45, 165, 220), -1)
    _draw_combat_cost_bar(f_combat_occ, with_progress_bar=True)
    det.detect(f_combat_occ)
    r_combat = det.detect(f_combat_occ)
    assert r_combat.scene == GameScene.BATTLE_COMBAT
    assert r_combat.board_mirrored is True

    # 切回左侧正向地标帧 -> 立即翻回 board_mirrored=False
    f_normal_prep = _make_base_shop_frame()
    _draw_stage_landmarks(f_normal_prep, mirrored=False, shop_open=True, mode="both")
    det.detect(f_normal_prep)
    r_back_normal = det.detect(f_normal_prep)
    assert r_back_normal.scene == GameScene.BATTLE_PREP
    assert r_back_normal.board_mirrored is False

    # 再次锁存 True 后连续 2 帧出局（UNKNOWN_FREE_CURSOR）-> 锁存彻底清零
    det.detect(f_boss_prep)
    assert det.detect(f_boss_prep).board_mirrored is True
    det.detect(black)
    r_exit = det.detect(black)
    assert r_exit.scene == GameScene.UNKNOWN_FREE_CURSOR
    assert r_exit.board_mirrored is False
    # 退出后重新进入无地标作战期 -> 仍保持 False（证明锁存已清零）
    det.detect(f_combat_occ)
    assert det.detect(f_combat_occ).board_mirrored is False

    # 3. 可选实机串行锁存验证：9797(镜像) -> 9f30(队友抽屉遮挡作战期保持 True) -> fe50(正向复位 False)
    a7_up = Path(
        r"C:\Users\MUXI\.gemini\antigravity\brain\a7e72e90-d260-48de-94e8-debc493ed124\.user_uploaded"
    )
    p9797 = a7_up / "media_1795334505233_979719ff.jpg"
    p9f30 = a7_up / "media_1795328987598_9f30b6b8.jpg"
    pfe50 = a7_up / "media_1795332461252_fe50bbf2.jpg"
    if p9797.exists() and p9f30.exists() and pfe50.exists():
        det_real = SceneDetector(templates_dir=tmp_path, debounce_frames=1)
        im_9797 = cv2.imdecode(np.fromfile(str(p9797), dtype=np.uint8), cv2.IMREAD_COLOR)
        im_9f30 = cv2.imdecode(np.fromfile(str(p9f30), dtype=np.uint8), cv2.IMREAD_COLOR)
        im_fe50 = cv2.imdecode(np.fromfile(str(pfe50), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert im_9797 is not None and im_9f30 is not None and im_fe50 is not None
        assert det_real.detect(im_9797).board_mirrored is True
        r_occ_real = det_real.detect(im_9f30)
        assert r_occ_real.scene == GameScene.BATTLE_COMBAT
        assert r_occ_real.board_mirrored is True
        assert det_real.detect(im_fe50).board_mirrored is False


def _draw_glowing_covenant_badge(
    frame: np.ndarray,
    cx: float,
    *,
    clip_left: int = 336,
    clip_right: int = 944,
    dark_core: bool = False,
) -> None:
    """绘制外缘精确位于 r=22.5px 的盟约圆环（支持满层高光亮芯与暗芯两种实机形态）。"""
    layer = frame.copy()
    ix, iy = int(round(cx)), 87
    # 深暗外背景晕，令外缘 r=22.5px 处形成清晰单峰径向梯度
    cv2.circle(layer, (ix, iy), 27, (18, 22, 30), -1, cv2.LINE_AA)
    if dark_core:
        # 暗芯青绿亮环（外缘 r=22.5px，内芯 r=18px 深暗）
        cv2.circle(layer, (ix, iy), 22, (175, 232, 130), -1, cv2.LINE_AA)
        cv2.circle(layer, (ix, iy), 18, (34, 42, 38), -1, cv2.LINE_AA)
        cv2.circle(layer, (ix, iy), 10, (72, 96, 80), -1, cv2.LINE_AA)
    else:
        # 满层高饱和高亮蓝青圆盘（圆心 V≈215, S≈190，外缘精确位于 r=22.5px）
        cv2.circle(layer, (ix, iy), 22, (248, 236, 98), -1, cv2.LINE_AA)
        cv2.circle(layer, (ix, iy), 17, (215, 188, 52), -1, cv2.LINE_AA)
    # 底部层数与盟约名高对比暗影白字
    cv2.rectangle(layer, (ix - 18, 102), (ix + 18, 126), (14, 16, 20), -1)
    cv2.line(layer, (ix - 12, 109), (ix + 12, 109), (235, 238, 240), 2)
    cv2.line(layer, (ix - 14, 119), (ix + 14, 119), (218, 222, 225), 2)

    x_min = max(clip_left, ix - 30)
    x_max = min(clip_right, ix + 31)
    if x_min < x_max:
        frame[55:132, x_min:x_max] = layer[55:132, x_min:x_max]


def test_v033_covenant_arbitrary_scroll_phase_and_glow_badges(tmp_path: Path) -> None:
    """验证满层发光圆环识别与 >7 盟约任意水平滚动相位动态坐标输出（误差 < 3px）。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    # 1. 满层高光亮芯 6 枚居中对称 -> count=6, overflow=False, centers_u 精确对齐
    f_glow6 = _make_theme3_rhodes_shop_frame(shop_open=False)
    gt6 = [640.0 + (i - 2.5) * 80.0 for i in range(6)]
    for cx in gt6:
        _draw_glowing_covenant_badge(f_glow6, cx)
    r_glow6 = detector.detect(f_glow6)
    assert r_glow6.scene == GameScene.BATTLE_PREP
    assert r_glow6.covenant_count == 6
    assert r_glow6.covenant_overflow is False
    assert len(r_glow6.covenant_centers_u) == 6
    for u_pred, x_gt in zip(r_glow6.covenant_centers_u, gt6):
        assert abs(u_pred * STANDARD_WIDTH - x_gt) < 3.0

    # 2. 合成 9 枚混合高光/暗芯盟约在 ScrollRect [336..944] 内连续平移滚动（覆盖非对称/居中微弧/右对齐）
    base = _make_theme3_rhodes_shop_frame(shop_open=False)
    for shift in (0, 20, 38, 44, 60, 100, 160):
        detector.reset()
        sim = base.copy()
        gt_all = [379.0 + k * 80.0 - shift for k in range(9)]
        for idx, cx in enumerate(gt_all):
            _draw_glowing_covenant_badge(
                sim, cx, clip_left=336, clip_right=944, dark_core=(idx % 2 == 1)
            )
        res = detector.detect(sim)
        assert res.scene == GameScene.BATTLE_PREP
        assert res.covenant_overflow is True, f"shift={shift} lost overflow: {res}"
        pred_xs = [u * STANDARD_WIDTH for u in res.covenant_centers_u]
        assert len(pred_xs) == res.covenant_count and len(pred_xs) >= 7
        core_gt = [x for x in gt_all if 362.0 <= x <= 918.0]
        for gx in core_gt:
            err = abs(min(pred_xs, key=lambda px: abs(px - gx)) - gx)
            assert err < 3.0, f"shift={shift} gt={gx} pred={pred_xs} err={err}"

    # 3. 可选实机连续滚动仿真：在 308f0863.jpg 与 979719ff.jpg 上验证多相位 100% 溢出保持与 <3px 精度
    a7_up = Path(
        r"C:\Users\MUXI\.gemini\antigravity\brain\a7e72e90-d260-48de-94e8-debc493ed124\.user_uploaded"
    )
    for fname, base_x0 in (
        ("media_1795325671744_308f0863.jpg", 380.0),
        ("media_1795334505233_979719ff.jpg", 375.5),
    ):
        p = a7_up / fname
        if not p.exists():
            continue
        raw = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert raw is not None
        im_std = (
            cv2.resize(raw, (STANDARD_WIDTH, STANDARD_HEIGHT), interpolation=cv2.INTER_AREA)
            if raw.shape[:2] != (STANDARD_HEIGHT, STANDARD_WIDTH)
            else raw
        )
        x_tile0 = int(round(base_x0 - 40.0))
        tiles = [im_std[58:132, x_tile0 + i * 80 : x_tile0 + (i + 1) * 80].copy() for i in range(7)]
        strip = np.hstack(tiles + [tiles[1].copy(), tiles[3].copy()])
        for shift in (0, 20, 44, 60, 100, 160):
            detector.reset()
            sim = im_std.copy()
            x_strip_start = x_tile0 - shift
            for dst_x in range(336, 944):
                src_x = dst_x - x_strip_start
                if 0 <= src_x < strip.shape[1]:
                    sim[58:132, dst_x] = strip[:, src_x]
            sim[58:87, 922:956] = im_std[58:87, 922:956]
            r_sc = detector.detect(sim)
            assert r_sc.covenant_overflow is True, f"{fname} shift={shift}: {r_sc}"
            pred_xs = [u * STANDARD_WIDTH for u in r_sc.covenant_centers_u]
            gt_all = [base_x0 + k * 80.0 - shift for k in range(9)]
            for gx in [x for x in gt_all if 362.0 <= x <= 918.0]:
                err = abs(min(pred_xs, key=lambda px: abs(px - gx)) - gx)
                assert err < 3.0, f"{fname} shift={shift} gt={gx} pred={pred_xs} err={err}"


def test_v033_in_game_chrome_resists_wda_none_white_hud_and_toast_occlusion(
    tmp_path: Path,
) -> None:
    """验证 WDA_NONE 顶部宽纯白 HUD 胶囊（含镜像战场文案）+ 第 2 行纯白 Toast 同时遮挡时仍稳判局内并保持镜像锁存。"""
    detector = SceneDetector(templates_dir=tmp_path, debounce_frames=1)

    # 1. 合成 Boss 战镜像休整帧，并在顶部中央叠画两行宽纯白 HUD 胶囊+Toast，使原 top_dk 跌穿 0.42（仅保留两翼黑条）
    f_synth = _make_base_shop_frame()
    _draw_stage_landmarks(f_synth, mirrored=True, shop_open=True, mode="both")
    # 扩展顶部暗条两翼覆盖 [385..895]（与实机顶部黑色状态栏同宽）
    cv2.rectangle(f_synth, (385, 6), (895, 66), (22, 24, 28), -1)
    # 叠画第 1 行超宽纯白 HUD 顶签 (x=475..805, y=8..36) + 第 2 行纯白 Toast (x=495..785, y=38..64)
    cv2.rectangle(f_synth, (475, 8), (805, 36), (248, 248, 248), -1)
    cv2.rectangle(f_synth, (495, 38), (785, 64), (246, 246, 246), -1)

    hsv_s = cv2.cvtColor(f_synth, cv2.COLOR_BGR2HSV)
    assert float((hsv_s[6:66, 410:870, 2] <= 70).mean()) < 0.42
    assert SceneDetector._has_stronghold_in_game_chrome(hsv_s) is True
    r_synth = detector.detect(f_synth)
    assert r_synth.scene == GameScene.BATTLE_PREP
    assert r_synth.shop_open is True
    assert r_synth.board_mirrored is True

    # 1b. 若左上深红退出门缺失（局外画面），即便两翼全黑也必须严守 UNKNOWN_FREE_CURSOR
    f_no_esc = f_synth.copy()
    cv2.rectangle(f_no_esc, (12, 10), (100, 80), (180, 180, 180), -1)
    detector.reset()
    assert SceneDetector._has_stronghold_in_game_chrome(cv2.cvtColor(f_no_esc, cv2.COLOR_BGR2HSV)) is False
    assert detector.detect(f_no_esc).scene == GameScene.UNKNOWN_FREE_CURSOR

    # 2. 可选实机回归：在 979719ff.jpg 上直接叠加两行纯白宽 HUD 胶囊+Toast，确认不跌落 UNKNOWN 且稳保 6 槽与镜像
    p9797 = Path(
        r"C:\Users\MUXI\.gemini\antigravity\brain\a7e72e90-d260-48de-94e8-debc493ed124\.user_uploaded\media_1795334505233_979719ff.jpg"
    )
    if p9797.exists():
        raw = cv2.imdecode(np.fromfile(str(p9797), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert raw is not None
        im_9797 = (
            cv2.resize(raw, (STANDARD_WIDTH, STANDARD_HEIGHT), interpolation=cv2.INTER_AREA)
            if raw.shape[:2] != (STANDARD_HEIGHT, STANDARD_WIDTH)
            else raw
        )
        # 复现 WDA_NONE 顶部「PRTS // 休整运营 · 商店 [3/6] · 镜像战场 · 商店展开」宽白胶囊 + 第 2 行白 Toast
        occluded = im_9797.copy()
        cv2.rectangle(occluded, (462, 6), (818, 36), (248, 248, 248), -1)
        cv2.rectangle(occluded, (505, 38), (775, 65), (245, 245, 245), -1)
        hsv_occ = cv2.cvtColor(occluded, cv2.COLOR_BGR2HSV)
        assert float((hsv_occ[6:66, 410:870, 2] <= 70).mean()) < 0.42
        detector.reset()
        r_occ = detector.detect(occluded)
        assert r_occ.scene == GameScene.BATTLE_PREP
        assert r_occ.shop_open is True
        assert r_occ.shop_slot_count == 6
        assert r_occ.board_mirrored is True
        assert r_occ.covenant_count == 7
        assert r_occ.covenant_overflow is True








