"""《明日方舟》卫戍协议界面阶段识别器。

采用双层判别架构与连续帧防抖机制：
1. 模板匹配层：优先加载 ``assets/templates/`` 下的本地校准模板（支持
   ``<scene_value>.png`` 与可选 ``<scene_value>.json`` 归一化 ROI 元数据），
   使用 ``cv2.matchTemplate(..., cv2.TM_CCOEFF_NORMED)`` 在对应局部 ROI 匹配。
2. 启发式特征回退层：当本地模板尚未采集或未命中时，基于 1280x720 帧的亮度分布、
   边缘周期结构与关键区域色块特征进行保守判别；若置信度不足则安全回退至
   ``GameScene.UNKNOWN_FREE_CURSOR``，绝不抛出未捕获异常或错误锁定模式。
3. 连续帧防抖层：连续 ``debounce_frames`` 帧判别一致后才切换稳定场景输出，
   有效抑制过场黑屏与 UI 动画闪烁。
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

STANDARD_WIDTH: int = 1280
STANDARD_HEIGHT: int = 720
DEFAULT_MATCH_THRESHOLD: float = 0.82
DEFAULT_SEARCH_PADDING: float = 0.02


class GameScene(str, Enum):
    """卫戍协议全流程界面阶段枚举。"""

    ACTIVITY_HUB = "activity_hub"
    PRE_MATCH_SETUP = "pre_match_setup"
    SQUAD_EDIT = "squad_edit"
    BATTLE_PREP = "battle_prep"
    MODAL_CHOICE = "modal_choice"
    BATTLE_COMBAT = "battle_combat"
    SETTLEMENT = "settlement"
    UNKNOWN_FREE_CURSOR = "unknown_free_cursor"


# 各场景在 16:9 标准视口下默认的标志性归一化 ROI: (u, v, w, h)
DEFAULT_SCENE_ROIS: dict[GameScene, tuple[float, float, float, float]] = {
    GameScene.ACTIVITY_HUB: (0.03, 0.04, 0.28, 0.14),
    GameScene.PRE_MATCH_SETUP: (0.68, 0.78, 0.28, 0.16),
    GameScene.SQUAD_EDIT: (0.04, 0.04, 0.30, 0.12),
    GameScene.BATTLE_PREP: (0.86, 0.12, 0.13, 0.10),
    GameScene.MODAL_CHOICE: (0.20, 0.22, 0.60, 0.56),
    GameScene.BATTLE_COMBAT: (0.38, 0.02, 0.24, 0.10),
    GameScene.SETTLEMENT: (0.06, 0.14, 0.34, 0.20),
}

# 休整期商店展开状态辅助模板默认 ROI: (u, v, w, h) —— 右侧冻结+刷新按钮带
DEFAULT_SHOP_OPEN_ROI: tuple[float, float, float, float] = (0.75, 0.58, 0.24, 0.12)

# 局内选中角色详情侧栏辅助模板默认 ROI: (u, v, w, h) —— 左侧生命条与特质/技能栏
DEFAULT_UNIT_SELECTED_ROI: tuple[float, float, float, float] = (0.0, 0.44, 0.35, 0.12)

_VALID_SCENE_VALUES: dict[str, GameScene] = {
    scene.value: scene
    for scene in GameScene
    if scene is not GameScene.UNKNOWN_FREE_CURSOR
}


@dataclass(frozen=True)
class SceneDetectionResult:
    """单次场景识别输出结果。"""

    scene: GameScene
    confidence: float
    shop_open: bool = True
    choice_count: int = 3
    unit_selected: bool = False
    shop_slot_count: int | None = None


@dataclass(frozen=True)
class _SceneTemplate:
    """内部使用的已加载模板记录。"""

    scene: GameScene | None
    name: str
    image: np.ndarray
    roi_norm: tuple[float, float, float, float] | None
    threshold: float = DEFAULT_MATCH_THRESHOLD
    search_padding: float = DEFAULT_SEARCH_PADDING


def _imread_bgr(path: Path) -> np.ndarray | None:
    """兼容任意 Windows 路径的 BGR 图像读取。"""
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        if raw.size == 0:
            return None
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return None
        return image
    except OSError:
        logger.warning("读取模板文件失败: %s", path, exc_info=True)
        return None


def _normalize_frame(frame: np.ndarray | None) -> np.ndarray | None:
    """将输入帧规范化为 1280x720 uint8 BGR 连续数组。"""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None

    arr = frame
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    elif arr.ndim != 3 or arr.shape[2] != 3:
        return None

    height, width = arr.shape[:2]
    if width <= 0 or height <= 0:
        return None

    if width != STANDARD_WIDTH or height != STANDARD_HEIGHT:
        interp = (
            cv2.INTER_AREA
            if width >= STANDARD_WIDTH and height >= STANDARD_HEIGHT
            else cv2.INTER_LINEAR
        )
        arr = cv2.resize(arr, (STANDARD_WIDTH, STANDARD_HEIGHT), interpolation=interp)

    return np.ascontiguousarray(arr)


def _parse_roi_norm(raw_roi: object) -> tuple[float, float, float, float] | None:
    """校验并解析归一化 ROI 元数据 ``[u, v, w, h]``。"""
    if not isinstance(raw_roi, (list, tuple)) or len(raw_roi) != 4:
        return None
    try:
        u, v, w, h = (float(raw_roi[0]), float(raw_roi[1]), float(raw_roi[2]), float(raw_roi[3]))
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite([u, v, w, h])):
        return None
    if u < 0.0 or v < 0.0 or u >= 1.0 or v >= 1.0 or w <= 0.0 or h <= 0.0:
        return None
    w = min(w, 1.0 - u)
    h = min(h, 1.0 - v)
    if w <= 0.0 or h <= 0.0:
        return None
    return (u, v, w, h)


def _compute_search_window(
    frame_w: int,
    frame_h: int,
    roi_norm: tuple[float, float, float, float] | None,
    tpl_w: int,
    tpl_h: int,
    pad_ratio: float,
) -> tuple[int, int, int, int] | None:
    """根据模板归一化 ROI 与模板尺寸计算像素级搜索窗口 ``(x1, y1, x2, y2)``。"""
    if tpl_w <= 0 or tpl_h <= 0 or tpl_w > frame_w or tpl_h > frame_h:
        return None

    if roi_norm is None:
        return (0, 0, frame_w, frame_h)

    u, v, w, h = roi_norm
    x1 = int(round(u * frame_w))
    y1 = int(round(v * frame_h))
    x2 = int(round((u + w) * frame_w))
    y2 = int(round((v + h) * frame_h))

    pad_x = int(round(max(0.0, pad_ratio) * frame_w))
    pad_y = int(round(max(0.0, pad_ratio) * frame_h))

    sx1 = max(0, x1 - pad_x)
    sy1 = max(0, y1 - pad_y)
    sx2 = min(frame_w, x2 + pad_x)
    sy2 = min(frame_h, y2 + pad_y)

    # 若扩边后的 ROI 仍略小于模板尺寸，则以原 ROI 为中心扩展到恰好容纳模板
    if sx2 - sx1 < tpl_w:
        cx = (x1 + x2) // 2
        sx1 = max(0, min(frame_w - tpl_w, cx - tpl_w // 2))
        sx2 = sx1 + tpl_w
    if sy2 - sy1 < tpl_h:
        cy = (y1 + y2) // 2
        sy1 = max(0, min(frame_h - tpl_h, cy - tpl_h // 2))
        sy2 = sy1 + tpl_h

    if sx2 - sx1 < tpl_w or sy2 - sy1 < tpl_h:
        return None
    return (sx1, sy1, sx2, sy2)


def _default_templates_dir() -> Path:
    """推断项目根目录下的默认模板文件夹路径。"""
    return Path(__file__).resolve().parents[3] / "assets" / "templates"


class SceneDetector:
    """卫戍协议界面阶段双层识别器（模板匹配 + 启发式回退 + 连续帧防抖）。"""

    def __init__(
        self,
        templates_dir: Path | None = None,
        debounce_frames: int = 2,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> None:
        self.templates_dir: Path = (
            Path(templates_dir) if templates_dir is not None else _default_templates_dir()
        )
        self.debounce_frames: int = max(1, int(debounce_frames))
        self.match_threshold: float = float(match_threshold)

        self._lock = threading.RLock()
        self._scene_templates: list[_SceneTemplate] = []
        self._shop_open_template: _SceneTemplate | None = None
        self._unit_selected_template: _SceneTemplate | None = None

        self._stable_result: SceneDetectionResult = SceneDetectionResult(
            scene=GameScene.UNKNOWN_FREE_CURSOR,
            confidence=0.0,
            shop_open=True,
            choice_count=3,
            unit_selected=False,
        )
        self._pending_scene: GameScene | None = None
        self._pending_count: int = 0

        self.reload_templates()

    @property
    def loaded_template_count(self) -> int:
        """返回当前已加载的场景模板总数（不含 ``shop_open`` 辅助模板）。"""
        with self._lock:
            return len(self._scene_templates)

    def reset(self) -> None:
        """重置防抖状态至初始未知自由光标状态。"""
        with self._lock:
            self._stable_result = SceneDetectionResult(
                scene=GameScene.UNKNOWN_FREE_CURSOR,
                confidence=0.0,
                shop_open=True,
                choice_count=3,
            )
            self._pending_scene = None
            self._pending_count = 0

    def reload_templates(self) -> None:
        """扫描并重新加载 ``templates_dir`` 下的所有本地模板与 ROI 元数据。"""
        scene_templates: list[_SceneTemplate] = []
        shop_template: _SceneTemplate | None = None
        unit_template: _SceneTemplate | None = None

        if self.templates_dir.exists() and self.templates_dir.is_dir():
            for png_path in sorted(self.templates_dir.glob("*.png")):
                tpl = self._load_single_template(png_path)
                if tpl is None:
                    continue
                if tpl.name == "shop_open":
                    shop_template = tpl
                elif tpl.name == "unit_selected":
                    unit_template = tpl
                elif tpl.scene is not None:
                    scene_templates.append(tpl)

        with self._lock:
            self._scene_templates = scene_templates
            self._shop_open_template = shop_template
            self._unit_selected_template = unit_template

        logger.debug(
            "模板重载完成: 目录=%s, 场景模板数=%d, 商店模板=%s, 角色详情模板=%s",
            self.templates_dir,
            len(scene_templates),
            shop_template is not None,
            unit_template is not None,
        )

    def _load_single_template(self, png_path: Path) -> _SceneTemplate | None:
        """加载单个 ``.png`` 模板及其可选同名 ``.json`` 元数据。"""
        stem = png_path.stem
        meta_path = png_path.with_suffix(".json")
        meta: dict[str, object] = {}

        if meta_path.exists() and meta_path.is_file():
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    meta = loaded
            except (OSError, json.JSONDecodeError):
                logger.warning("解析模板元数据失败，将按默认规则加载: %s", meta_path)

        is_aux_template = stem in ("shop_open", "unit_selected")
        scene: GameScene | None = None
        if not is_aux_template:
            scene = self._resolve_scene_from_stem_and_meta(stem, meta)
            if scene is None:
                logger.debug("忽略未映射到已知 GameScene 的模板文件: %s", png_path.name)
                return None

        image = _imread_bgr(png_path)
        if image is None:
            return None

        tpl_h, tpl_w = image.shape[:2]
        if tpl_w > STANDARD_WIDTH or tpl_h > STANDARD_HEIGHT:
            logger.warning(
                "模板尺寸超出 1280x720，已跳过: %s (%dx%d)",
                png_path.name,
                tpl_w,
                tpl_h,
            )
            return None

        roi_norm = _parse_roi_norm(meta.get("roi_norm"))
        raw_thresh = meta.get("threshold")
        threshold = self.match_threshold
        if isinstance(raw_thresh, (int, float)) and np.isfinite(raw_thresh):
            threshold = float(np.clip(raw_thresh, 0.1, 0.999))

        raw_pad = meta.get("search_padding")
        search_padding = DEFAULT_SEARCH_PADDING
        if isinstance(raw_pad, (int, float)) and np.isfinite(raw_pad):
            search_padding = float(np.clip(raw_pad, 0.0, 0.25))

        return _SceneTemplate(
            scene=scene,
            name=stem,
            image=np.ascontiguousarray(image),
            roi_norm=roi_norm,
            threshold=threshold,
            search_padding=search_padding,
        )

    @staticmethod
    def _resolve_scene_from_stem_and_meta(
        stem: str, meta: dict[str, object]
    ) -> GameScene | None:
        """从元数据或文件名解析对应的 ``GameScene``。"""
        meta_scene = meta.get("scene")
        if isinstance(meta_scene, str) and meta_scene in _VALID_SCENE_VALUES:
            return _VALID_SCENE_VALUES[meta_scene]

        if stem in _VALID_SCENE_VALUES:
            return _VALID_SCENE_VALUES[stem]

        # 支持多变体模板命名，例如 battle_prep__variant1.png
        for scene_val, scene_enum in _VALID_SCENE_VALUES.items():
            if stem.startswith(f"{scene_val}__"):
                return scene_enum
        return None

    def detect(self, frame: np.ndarray | None) -> SceneDetectionResult:
        """识别 1280x720 BGR 帧当前所属界面阶段。

        当 ``frame is None`` 或无效时安全返回 ``UNKNOWN_FREE_CURSOR``。
        """
        if frame is None:
            self.reset()
            return SceneDetectionResult(
                scene=GameScene.UNKNOWN_FREE_CURSOR,
                confidence=0.0,
                shop_open=True,
                choice_count=3,
            )

        try:
            normalized = _normalize_frame(frame)
            if normalized is None:
                self.reset()
                return SceneDetectionResult(
                    scene=GameScene.UNKNOWN_FREE_CURSOR,
                    confidence=0.0,
                    shop_open=True,
                    choice_count=3,
                )

            raw_result = self._classify_single_frame(normalized)
            return self._apply_debounce(raw_result)
        except Exception:
            logger.exception("场景识别发生未预期异常，安全回退为 UNKNOWN_FREE_CURSOR")
            return SceneDetectionResult(
                scene=GameScene.UNKNOWN_FREE_CURSOR,
                confidence=0.0,
                shop_open=True,
                choice_count=3,
            )

    def _classify_single_frame(self, frame: np.ndarray) -> SceneDetectionResult:
        """对单帧执行“模板优先 + 启发式回退”分类。"""
        with self._lock:
            templates = list(self._scene_templates)
            shop_template = self._shop_open_template
            unit_template = self._unit_selected_template

        # (a) 第一层：局部 ROI 模板匹配
        best_scene: GameScene | None = None
        best_score: float = 0.0

        for tpl in templates:
            score = self._match_template(frame, tpl)
            if score >= tpl.threshold and score > best_score:
                best_score = score
                best_scene = tpl.scene

        if best_scene is not None:
            shop_open = True
            choice_count = 3
            unit_selected = False
            shop_slot_count: int | None = None
            if best_scene == GameScene.BATTLE_PREP:
                shop_open = self._detect_shop_open(frame, shop_template)
                unit_selected = self._detect_unit_selected(frame, unit_template)
                if shop_open:
                    shop_slot_count = self._detect_shop_slot_count_heuristic(frame)
            elif best_scene == GameScene.MODAL_CHOICE:
                shop_open = False
                choice_count = self._estimate_choice_count(frame)
            elif best_scene == GameScene.BATTLE_COMBAT:
                shop_open = False
                unit_selected = self._detect_unit_selected(frame, unit_template)
            return SceneDetectionResult(
                scene=best_scene,
                confidence=round(best_score, 4),
                shop_open=shop_open,
                choice_count=choice_count,
                unit_selected=unit_selected,
                shop_slot_count=shop_slot_count,
            )

        # (b) 第二层：启发式视觉特征回退
        return self._detect_heuristic(frame, shop_template, unit_template)

    @staticmethod
    def _match_template(frame: np.ndarray, tpl: _SceneTemplate) -> float:
        """在模板指定 ROI（或全帧）内执行归一化互相关匹配。"""
        frame_h, frame_w = frame.shape[:2]
        tpl_h, tpl_w = tpl.image.shape[:2]

        window = _compute_search_window(
            frame_w=frame_w,
            frame_h=frame_h,
            roi_norm=tpl.roi_norm,
            tpl_w=tpl_w,
            tpl_h=tpl_h,
            pad_ratio=tpl.search_padding,
        )
        if window is None:
            return 0.0

        # 过滤纯常数模板与纯常数目标区域，杜绝零方差导致的 TM_CCOEFF_NORMED 伪满分
        if float(tpl.image.std()) < 1.0:
            return 0.0

        sx1, sy1, sx2, sy2 = window
        region = frame[sy1:sy2, sx1:sx2]
        if float(region.std()) < 1.0:
            return 0.0

        result_map = cv2.matchTemplate(region, tpl.image, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result_map)
        if not np.isfinite(max_val):
            return 0.0
        return float(np.clip(max_val, 0.0, 1.0))

    def _detect_shop_open(
        self, frame: np.ndarray, shop_template: _SceneTemplate | None
    ) -> bool:
        """判断休整期底部商店面板是否处于展开状态。"""
        if shop_template is not None:
            score = self._match_template(frame, shop_template)
            return score >= shop_template.threshold
        return self._detect_shop_open_heuristic(frame)

    @staticmethod
    def _measure_top_right_ready_pink_ratio(hsv: np.ndarray) -> float:
        """测量官方 PC 端卫戍协议右上角粉红色「准备就绪」按钮像素占比（1280x720 实机标定）。"""
        roi = hsv[95:145, 1115:1260]
        pink_mask = (
            ((roi[:, :, 0] >= 155) | (roi[:, :, 0] <= 10))
            & (roi[:, :, 1] >= 110)
            & (roi[:, :, 2] >= 130)
        )
        return float(pink_mask.mean())

    @staticmethod
    def _measure_shop_action_buttons(hsv: np.ndarray) -> tuple[float, float]:
        """测量商店展开时右侧「黄色刷新 + 蓝色冻结」双按钮像素占比（实机展开 ~0.61，收起 0.0）。"""
        y_roi = hsv[425:485, 1120:1265]
        refresh_yellow = float(
            (
                (y_roi[:, :, 0] >= 15)
                & (y_roi[:, :, 0] <= 35)
                & (y_roi[:, :, 1] >= 110)
                & (y_roi[:, :, 2] >= 140)
            ).mean()
        )
        b_roi = hsv[425:485, 965:1110]
        # 同时覆盖常规「冻结」（深蓝 H≈105..120）与已冻结「解冻」（青绿 H≈85..95）双态按钮
        freeze_blue = float(
            (
                (b_roi[:, :, 0] >= 80)
                & (b_roi[:, :, 0] <= 125)
                & (b_roi[:, :, 1] >= 90)
                & (b_roi[:, :, 2] >= 90)
            ).mean()
        )
        return (refresh_yellow, freeze_blue)

    @staticmethod
    def _detect_unit_selected_heuristic(frame: np.ndarray) -> bool:
        """基于左侧角色青蓝生命条（排除拖拽售出态）判断角色详情特写面板是否展开。"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ready_pink = SceneDetector._measure_top_right_ready_pink_ratio(hsv)
        if ready_pink < 0.28:
            return False
        hp_roi = hsv[325:345, 5:430]
        hp_blue = float(
            (
                (hp_roi[:, :, 0] >= 92)
                & (hp_roi[:, :, 0] <= 115)
                & (hp_roi[:, :, 1] >= 150)
                & (hp_roi[:, :, 2] >= 150)
            ).mean()
        )
        sell_roi = hsv[520:700, 1105:1260]
        sell_gold = float(
            (
                (sell_roi[:, :, 0] >= 16)
                & (sell_roi[:, :, 0] <= 32)
                & (sell_roi[:, :, 1] >= 130)
                & (sell_roi[:, :, 2] >= 140)
            ).mean()
        )
        return hp_blue >= 0.26 and sell_gold < 0.04

    @staticmethod
    def _detect_shop_open_heuristic(frame: np.ndarray) -> bool:
        """判断休整期商店是否处于展开状态（兼容实机右侧刷新/冻结色块与通用卡牌边缘特征）。"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ready_pink = SceneDetector._measure_top_right_ready_pink_ratio(hsv)
        refresh_yellow, freeze_blue = SceneDetector._measure_shop_action_buttons(hsv)

        # 实机休整期画面（右上角存在粉红准备就绪按钮）：直接采用零误差的刷新/冻结双按钮色块判别
        if ready_pink >= 0.28:
            return refresh_yellow >= 0.25 and freeze_blue >= 0.25

        # 兼容非实机布局/自定义主题及单元测试合成帧
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shop_roi = gray[410:555, 180:1030]
        mean_val = float(shop_roi.mean())
        std_val = float(shop_roi.std())

        if mean_val < 38.0 and std_val < 16.0:
            return False

        edges = cv2.Canny(shop_roi, 45, 130)
        edge_density = float((edges > 0).mean())
        if mean_val < 48.0 and edge_density < 0.015:
            return False

        return True

    @staticmethod
    def _measure_vertical_edge_strength(v_channel: np.ndarray, x: int) -> float:
        """测量 1280x720 商店卡片高度带 (y=540..695) 在指定 x 处的垂直边界梯度强度。"""
        width = v_channel.shape[1]
        x_start = max(0, x - 2)
        x_end = min(width, x + 3)
        if x_end - x_start < 2:
            return 0.0
        strip = v_channel[540:695, x_start:x_end].astype(np.float32)
        return float(np.abs(np.diff(strip, axis=1)).max(axis=1).mean())

    @classmethod
    def _has_shop_upgrade_card(cls, v_channel: np.ndarray, x1: int, x2: int) -> bool:
        """判断给定横向区间是否存在高亮「升级商店 (LEVEL)」卡片（兼容灰/亮可点双色）。"""
        v_top = v_channel[535:635, x1:x2]
        v_all = v_channel[535:695, x1:x2]
        if v_top.size == 0 or v_all.size == 0:
            return False
        top_bright = float((v_top > 120).mean())
        all_bright = float((v_all > 110).mean())
        if top_bright < 0.25 or all_bright < 0.20:
            return False
        edge_l = cls._measure_vertical_edge_strength(v_channel, x1)
        edge_r = cls._measure_vertical_edge_strength(v_channel, x2)
        return edge_l >= 10.0 and edge_r >= 10.0

    @classmethod
    def _detect_shop_slot_count_heuristic(cls, frame: np.ndarray) -> int | None:
        """检测展开后的卫戍协议商店栏位数（1 本 4 槽 / 2~3 本 5 槽 / 4~6 本 6 槽）。

        依据实机精确布局：最右卡槽固定于 x=936..1090，卡片步距 162px，升级卡宽 109px
        紧贴最左商品卡左侧。升级时向左侧延伸：
        - 4 槽：升级卡 x=332..441，卡槽 450..1090
        - 5 槽：升级卡 x=170..279，卡槽 288..1090
        - 6 槽：升级卡 x=8..117（或满级时第 6 张卡槽 x=126..280 存在左右边框），卡槽 126..1090
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ready_pink = cls._measure_top_right_ready_pink_ratio(hsv)
        if ready_pink < 0.28:
            return None
        refresh_yellow, freeze_blue = cls._measure_shop_action_buttons(hsv)
        if refresh_yellow < 0.25 or freeze_blue < 0.25:
            return None

        v_ch = hsv[:, :, 2]

        # 1. 优先判定 6 槽（4~6 本）：最左升级卡 [8..117] 出现，或满级无升级卡但第 6 槽边框 [126..280] 出现
        if cls._has_shop_upgrade_card(v_ch, 8, 117):
            return 6
        if (
            cls._measure_vertical_edge_strength(v_ch, 126) >= 8.0
            and cls._measure_vertical_edge_strength(v_ch, 280) >= 8.0
        ):
            return 6

        # 2. 次优判定 5 槽（2~3 本）：升级卡左移至 [170..279]
        if cls._has_shop_upgrade_card(v_ch, 170, 279):
            return 5

        # 3. 默认 1 本 4 槽
        return 4

    def _detect_unit_selected(
        self, frame: np.ndarray, unit_template: _SceneTemplate | None
    ) -> bool:
        """判断当前是否处于选取角色详情面板（触发界面缩放）状态。"""
        if unit_template is not None:
            score = self._match_template(frame, unit_template)
            return score >= unit_template.threshold
        return self._detect_unit_selected_heuristic(frame)

    def _detect_heuristic(
        self,
        frame: np.ndarray,
        shop_template: _SceneTemplate | None,
        unit_template: _SceneTemplate | None = None,
    ) -> SceneDetectionResult:
        """基于 1280x720 画面亮度/边缘/色块特征的保守启发式回退判别器。"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if float(gray.std()) < 8.0:
            return SceneDetectionResult(
                scene=GameScene.UNKNOWN_FREE_CURSOR,
                confidence=0.0,
                shop_open=True,
                choice_count=3,
            )

        # 1. 检测三选一/策略事件弹窗（四周暗化遮罩 + 中央横向等距高亮卡片）
        modal_result = self._detect_modal_choice_heuristic(gray)
        if modal_result is not None:
            return modal_result

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 2a. 实机官方 PC 端卫戍协议休整期强特征：右上角粉红「准备就绪」按钮（~0.52）
        top_ready_pink = self._measure_top_right_ready_pink_ratio(hsv)
        if top_ready_pink >= 0.28:
            shop_open = self._detect_shop_open(frame, shop_template)
            unit_selected = self._detect_unit_selected(frame, unit_template)
            shop_slot_count = (
                self._detect_shop_slot_count_heuristic(frame) if shop_open else None
            )
            confidence = min(0.90, 0.72 + 0.30 * min(top_ready_pink, 0.60))
            return SceneDetectionResult(
                scene=GameScene.BATTLE_PREP,
                confidence=round(confidence, 4),
                shop_open=shop_open,
                choice_count=3,
                unit_selected=unit_selected,
                shop_slot_count=shop_slot_count,
            )

        # 2b. 通用底部整备槽位条纹特征（兼容旧主题与合成测试帧）
        bench_score = self._measure_bench_slots_score(gray)
        if bench_score >= 0.60:
            # 右下角准备就绪按钮区饱和高亮色块特征
            ready_vivid_ratio = self._measure_ready_button_ratio(hsv)
            if 0.10 <= ready_vivid_ratio <= 0.88:
                confidence = min(0.80, 0.64 + 0.12 * bench_score + 0.08 * min(ready_vivid_ratio / 0.35, 1.0))
                shop_open = self._detect_shop_open(frame, shop_template)
                unit_selected = self._detect_unit_selected(frame, unit_template)
                return SceneDetectionResult(
                    scene=GameScene.BATTLE_PREP,
                    confidence=round(confidence, 4),
                    shop_open=shop_open,
                    choice_count=3,
                    unit_selected=unit_selected,
                )

            # 右下角无准备按钮高亮块，且顶部中央存在作战波次 HUD 面板 -> 作战期
            combat_hud_score = self._measure_combat_top_hud_score(gray, hsv)
            if ready_vivid_ratio < 0.08 and combat_hud_score >= 0.60:
                confidence = min(0.76, 0.62 + 0.08 * bench_score + 0.08 * combat_hud_score)
                unit_selected = self._detect_unit_selected(frame, unit_template)
                return SceneDetectionResult(
                    scene=GameScene.BATTLE_COMBAT,
                    confidence=round(confidence, 4),
                    shop_open=False,
                    choice_count=3,
                    unit_selected=unit_selected,
                )

        # 置信度不足时一律安全返回未知自由光标模式
        return SceneDetectionResult(
            scene=GameScene.UNKNOWN_FREE_CURSOR,
            confidence=0.0,
            shop_open=True,
            choice_count=3,
        )

    def _detect_modal_choice_heuristic(
        self, gray: np.ndarray
    ) -> SceneDetectionResult | None:
        """检测中央横向等距高亮卡片与四周半透明暗化遮罩的弹窗特征。"""
        border_top = gray[14:94, 60:1220]
        border_bottom = gray[615:706, 60:1220]
        border_left = gray[100:610, 16:130]
        border_right = gray[100:610, 1150:1264]

        border_pixels = np.concatenate(
            [
                border_top.ravel(),
                border_bottom.ravel(),
                border_left.ravel(),
                border_right.ravel(),
            ]
        )
        border_mean = float(border_pixels.mean())
        border_p75 = float(np.percentile(border_pixels, 75))

        # 四周必须存在较明显的暗化遮罩
        if border_mean >= 68.0 or border_p75 >= 95.0:
            return None

        cards = self._find_modal_card_spans(gray, border_mean)
        if len(cards) not in (2, 3):
            return None

        widths = [e - s for s, e in cards]
        if min(widths) / max(widths) < 0.70:
            return None

        # 验证卡片间距与整体居中对称性
        centers = [(s + e) * 0.5 for s, e in cards]
        band_width = 1000.0  # x in [140, 1140)
        group_center = (centers[0] + centers[-1]) * 0.5
        if abs(group_center - band_width * 0.5) > 140.0:
            return None

        if len(cards) == 3:
            gap_left = centers[1] - centers[0]
            gap_right = centers[2] - centers[1]
            if min(gap_left, gap_right) <= 0 or min(gap_left, gap_right) / max(gap_left, gap_right) < 0.72:
                return None

        # 验证相邻卡片之间确实存在暗色间隔缝隙
        band = gray[185:515, 140:1140]
        thresh = max(border_mean + 35.0, 95.0)
        for idx in range(len(cards) - 1):
            gap_start = cards[idx][1]
            gap_end = cards[idx + 1][0]
            if gap_end - gap_start < 18:
                return None
            gap_mean = float(band[:, gap_start:gap_end].mean())
            if gap_mean >= thresh - 12.0:
                return None

        card_means = [float(band[:, s:e].mean()) for s, e in cards]
        contrast = float(np.mean(card_means)) - border_mean
        confidence = min(0.88, 0.68 + 0.002 * min(max(contrast, 0.0), 100.0))

        return SceneDetectionResult(
            scene=GameScene.MODAL_CHOICE,
            confidence=round(confidence, 4),
            shop_open=False,
            choice_count=len(cards),
        )

    @staticmethod
    def _find_modal_card_spans(
        gray: np.ndarray, border_mean: float
    ) -> list[tuple[int, int]]:
        """在中央横带提取符合三选一/二选一尺寸比例的高亮卡片列区间。"""
        band = gray[185:515, 140:1140]  # 高 330px, 宽 1000px
        col_mean = band.mean(axis=0)
        thresh = max(border_mean + 35.0, 95.0)
        mask = col_mean >= thresh

        # 合并卡片内部极窄暗纹（< 12px）
        raw_spans: list[tuple[int, int]] = []
        in_span = False
        start = 0
        for idx, val in enumerate(mask):
            if val and not in_span:
                start = idx
                in_span = True
            elif not val and in_span:
                raw_spans.append((start, idx))
                in_span = False
        if in_span:
            raw_spans.append((start, len(mask)))

        merged: list[tuple[int, int]] = []
        for s, e in raw_spans:
            if merged and s - merged[-1][1] < 12:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        valid_cards: list[tuple[int, int]] = []
        for s, e in merged:
            width = e - s
            if not (110 <= width <= 360):
                continue
            patch = band[:, s:e]
            patch_mean = float(patch.mean())
            bright_ratio = float((patch >= thresh - 10.0).mean())
            if patch_mean >= border_mean + 40.0 and bright_ratio >= 0.45:
                valid_cards.append((s, e))
        return valid_cards

    def _estimate_choice_count(self, frame: np.ndarray) -> int:
        """在已判定为 ``MODAL_CHOICE`` 时估算可选卡片数量（默认 3）。"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        border_mean = float(gray[14:94, 60:1220].mean())
        cards = self._find_modal_card_spans(gray, border_mean)
        if len(cards) in (2, 3, 4):
            return len(cards)
        return 3

    @staticmethod
    def _measure_bench_slots_score(gray: np.ndarray) -> float:
        """评估底部整备槽位周期性分隔条纹的特征得分（0.0 ~ 1.0）。"""
        bench_roi = gray[560:695, 160:1030]  # 135 x 870
        gx = np.abs(cv2.Sobel(bench_roi, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(bench_roi, cv2.CV_32F, 0, 1, ksize=3))

        row_grad = gy.mean(axis=1)
        if float(row_grad.max()) < 22.0:
            return 0.0

        col_grad = gx.mean(axis=0)
        mean_col_grad = float(col_grad.mean())
        peak_thresh = max(20.0, mean_col_grad * 1.9)

        # 非极大值抑制提取竖直槽位边框峰（最小间距 42px）
        candidate_indices = np.where(col_grad >= peak_thresh)[0]
        if candidate_indices.size < 5:
            return 0.0

        peaks: list[int] = []
        order = candidate_indices[np.argsort(col_grad[candidate_indices])[::-1]]
        for idx in order:
            idx_int = int(idx)
            if all(abs(idx_int - p) >= 42 for p in peaks):
                peaks.append(idx_int)
        peaks.sort()

        if len(peaks) < 5:
            return 0.0

        spacings = np.diff(peaks).astype(np.float64)
        median_spacing = float(np.median(spacings))
        if not (48.0 <= median_spacing <= 135.0):
            return 0.0

        # 过滤掉离群大空隙后检验槽位间距规律性
        regular_gaps = spacings[np.abs(spacings - median_spacing) <= median_spacing * 0.35]
        regular_ratio = float(regular_gaps.size / spacings.size)
        if regular_ratio < 0.65:
            return 0.0

        count_factor = min(1.0, len(peaks) / 9.0)
        return float(np.clip(0.55 * regular_ratio + 0.45 * count_factor, 0.0, 1.0))

    @staticmethod
    def _measure_ready_button_ratio(hsv: np.ndarray) -> float:
        """测量右下角「准备就绪」高亮高饱和行动区像素占比。"""
        ready_hsv = hsv[555:695, 1035:1255]
        vivid_mask = (ready_hsv[:, :, 1] >= 110) & (ready_hsv[:, :, 2] >= 130)
        return float(vivid_mask.mean())

    @staticmethod
    def _measure_combat_top_hud_score(gray: np.ndarray, hsv: np.ndarray) -> float:
        """评估顶部中央作战波次与剩余敌数 HUD 面板特征得分（0.0 ~ 1.0）。"""
        top_gray = gray[14:86, 460:820]
        top_hsv = hsv[14:86, 460:820]

        std_val = float(top_gray.std())
        if std_val < 18.0:
            return 0.0

        edges = cv2.Canny(top_gray, 50, 140)
        edge_density = float((edges > 0).mean())
        if not (0.025 <= edge_density <= 0.35):
            return 0.0

        vivid_mask = (top_hsv[:, :, 1] >= 100) & (top_hsv[:, :, 2] >= 120)
        vivid_ratio = float(vivid_mask.mean())
        if not (0.03 <= vivid_ratio <= 0.55):
            return 0.0

        return float(np.clip(0.5 + min(std_val / 80.0, 0.3) + min(vivid_ratio, 0.2), 0.0, 1.0))

    def _apply_debounce(self, raw_result: SceneDetectionResult) -> SceneDetectionResult:
        """应用连续帧防抖滤波，抑制瞬态过场画面闪烁。"""
        with self._lock:
            if raw_result.scene == self._stable_result.scene:
                self._stable_result = raw_result
                self._pending_scene = None
                self._pending_count = 0
                return self._stable_result

            if raw_result.scene == self._pending_scene:
                self._pending_count += 1
            else:
                self._pending_scene = raw_result.scene
                self._pending_count = 1

            if self._pending_count >= self.debounce_frames:
                self._stable_result = raw_result
                self._pending_scene = None
                self._pending_count = 0
                return self._stable_result

            return self._stable_result
