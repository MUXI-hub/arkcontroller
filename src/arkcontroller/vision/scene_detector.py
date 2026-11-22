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

# 盟约圆形徽章 36 点圆环采样表（r=22.5px 法向梯度用，预计算避免逐帧重复三角运算）
_COVENANT_RING_ANGLES: np.ndarray = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
_COVENANT_RING_COS: np.ndarray = np.cos(_COVENANT_RING_ANGLES)
_COVENANT_RING_SIN: np.ndarray = np.sin(_COVENANT_RING_ANGLES)


class GameScene(str, Enum):
    """卫戍协议局内界面阶段枚举（局外界面一律回退至 UNKNOWN_FREE_CURSOR）。"""

    BATTLE_PREP = "battle_prep"
    MODAL_CHOICE = "modal_choice"
    MODAL_CONFIRM = "modal_confirm"
    BATTLE_COMBAT = "battle_combat"
    UNKNOWN_FREE_CURSOR = "unknown_free_cursor"


# 各场景在 16:9 标准视口下默认的标志性归一化 ROI: (u, v, w, h)
DEFAULT_SCENE_ROIS: dict[GameScene, tuple[float, float, float, float]] = {
    GameScene.BATTLE_PREP: (0.86, 0.12, 0.13, 0.10),
    GameScene.MODAL_CHOICE: (0.20, 0.22, 0.60, 0.56),
    GameScene.MODAL_CONFIRM: (0.15, 0.28, 0.70, 0.44),
    GameScene.BATTLE_COMBAT: (0.38, 0.02, 0.24, 0.10),
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
    selected_choice_index: int | None = None
    covenant_count: int | None = None
    covenant_overflow: bool = False
    has_teammates: bool = False
    emote_panel_open: bool = False
    confirm_has_dont_ask: bool = True


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
            shop_open=False,
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
                shop_open=False,
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
                shop_open=False,
                choice_count=3,
            )

        try:
            normalized = _normalize_frame(frame)
            if normalized is None:
                self.reset()
                return SceneDetectionResult(
                    scene=GameScene.UNKNOWN_FREE_CURSOR,
                    confidence=0.0,
                    shop_open=False,
                    choice_count=3,
                )

            raw_result = self._classify_single_frame(normalized)
            return self._apply_debounce(raw_result)
        except Exception:
            logger.exception("场景识别发生未预期异常，安全回退为 UNKNOWN_FREE_CURSOR")
            return SceneDetectionResult(
                scene=GameScene.UNKNOWN_FREE_CURSOR,
                confidence=0.0,
                shop_open=False,
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
            selected_choice_index: int | None = None
            covenant_count: int | None = None
            covenant_overflow = False
            has_teammates = False
            emote_panel_open = False
            confirm_has_dont_ask = True
            gray_lazy: np.ndarray | None = None
            hsv_lazy: np.ndarray | None = None

            if best_scene == GameScene.BATTLE_PREP:
                shop_open = self._detect_shop_open(frame, shop_template)
                unit_selected = self._detect_unit_selected(frame, unit_template)
                if shop_open:
                    shop_slot_count = self._detect_shop_slot_count_heuristic(frame)
                gray_lazy = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hsv_lazy = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                covenant_count, covenant_overflow = self._detect_covenants_heuristic(
                    hsv_lazy, gray_lazy
                )
                has_teammates = self._detect_teammates_heuristic(
                    gray_lazy, unit_selected=unit_selected
                )
                if not shop_open and not unit_selected:
                    emote_panel_open = self._detect_emote_panel_open_heuristic(
                        gray_lazy, hsv_lazy
                    )
            elif best_scene == GameScene.MODAL_CHOICE:
                shop_open = False
                gray_lazy = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hsv_lazy = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                choice_count, selected_choice_index = self._analyze_modal_choice_details(
                    gray_lazy, hsv_lazy
                )
                has_teammates = self._detect_teammates_heuristic(gray_lazy, unit_selected=False)
            elif best_scene == GameScene.MODAL_CONFIRM:
                shop_open = False
                gray_lazy = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                confirm_has_dont_ask = self._detect_confirm_has_dont_ask_heuristic(gray_lazy)
            elif best_scene == GameScene.BATTLE_COMBAT:
                shop_open = False
                unit_selected = self._detect_unit_selected(frame, unit_template)
                gray_lazy = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hsv_lazy = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                covenant_count, covenant_overflow = self._detect_covenants_heuristic(
                    hsv_lazy, gray_lazy
                )
                has_teammates = self._detect_teammates_heuristic(
                    gray_lazy, unit_selected=unit_selected
                )
                if not unit_selected:
                    emote_panel_open = self._detect_emote_panel_open_heuristic(
                        gray_lazy, hsv_lazy
                    )
            return SceneDetectionResult(
                scene=best_scene,
                confidence=round(best_score, 4),
                shop_open=shop_open,
                choice_count=choice_count,
                unit_selected=unit_selected,
                shop_slot_count=shop_slot_count,
                selected_choice_index=selected_choice_index,
                covenant_count=covenant_count,
                covenant_overflow=covenant_overflow,
                has_teammates=has_teammates,
                emote_panel_open=emote_panel_open,
                confirm_has_dont_ask=confirm_has_dont_ask,
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
    def _has_stronghold_in_game_chrome(hsv: np.ndarray) -> bool:
        """校验卫戍协议局内双身份锚点，100% 拦截主界面/编队/干员列表等局外误判。

        局内休整/作战/出怪预览画面稳定共现两项标志性 HUD 锚点：
        1. 左上角深红色退出门按钮 ``hsv[12:76, 16:96]``（占比 >= 0.22，兼容联机弹幕局部遮挡）；
        2. 顶部中央深黑回合/状态信息栏 ``hsv[6:66, 410:870]``（``V <= 70`` 占比 >= 0.42）。
        """
        esc_roi = hsv[12:76, 16:96]
        esc_red = float(
            (
                ((esc_roi[:, :, 0] <= 8) | (esc_roi[:, :, 0] >= 170))
                & (esc_roi[:, :, 1] >= 100)
                & (esc_roi[:, :, 2] >= 55)
                & (esc_roi[:, :, 2] <= 195)
            ).mean()
        )
        if esc_red < 0.22:
            return False

        top_dk = float((hsv[6:66, 410:870, 2] <= 70).mean())
        return top_dk >= 0.42

    @staticmethod
    def _measure_top_right_ready_pink_ratio(hsv: np.ndarray) -> float:
        """测量主题一（电玩紫粉风）右上角粉红色「准备就绪」按钮像素占比（1280x720 实机标定）。

        色相严格收窄至 ``H >= 158 | H <= 4``，避免主界面/活动页 ``H=5..10`` 橙红横幅误触发。
        """
        roi = hsv[95:145, 1115:1260]
        pink_mask = (
            ((roi[:, :, 0] >= 158) | (roi[:, :, 0] <= 4))
            & (roi[:, :, 1] >= 110)
            & (roi[:, :, 2] >= 130)
        )
        return float(pink_mask.mean())

    @staticmethod
    def _measure_top_right_ready_teal_ratio(hsv: np.ndarray) -> float:
        """测量主题三（罗德岛战术青绿风）右上角青绿色「准备就绪」矩形按钮像素占比。"""
        roi = hsv[95:145, 1115:1260]
        teal_mask = (
            (roi[:, :, 0] >= 74)
            & (roi[:, :, 0] <= 96)
            & (roi[:, :, 1] >= 135)
            & (roi[:, :, 2] >= 130)
        )
        return float(teal_mask.mean())

    @staticmethod
    def _measure_top_right_ready_sage_capsule(hsv: np.ndarray) -> tuple[float, float]:
        """测量主题二（乌萨斯复古辉光管风）右上角浅灰绿内胶囊占比与上方暗边框占比。"""
        r_inner = hsv[100:136, 1128:1252]
        inner_sage = float(
            (
                (r_inner[:, :, 0] >= 60)
                & (r_inner[:, :, 0] <= 112)
                & (r_inner[:, :, 1] >= 12)
                & (r_inner[:, :, 1] <= 85)
                & (r_inner[:, :, 2] >= 155)
                & (r_inner[:, :, 2] <= 240)
            ).mean()
        )
        r_outer_top = hsv[86:98, 1128:1252]
        top_dark = float((r_outer_top[:, :, 2] <= 115).mean())
        return (inner_sage, top_dark)

    @staticmethod
    def _measure_shop_action_buttons(hsv: np.ndarray) -> tuple[float, float]:
        """测量三主题商店展开时右侧「黄色刷新 + 冻结/解冻」双按钮像素占比。

        覆盖主题一深蓝/青绿、主题二深青绿与主题三天蓝冻结按钮（``H∈[78..125], S>=90, V>=85``）。
        """
        y_roi = hsv[425:485, 1120:1265]
        refresh_yellow = float(
            (
                (y_roi[:, :, 0] >= 14)
                & (y_roi[:, :, 0] <= 35)
                & (y_roi[:, :, 1] >= 110)
                & (y_roi[:, :, 2] >= 140)
            ).mean()
        )
        b_roi = hsv[425:485, 965:1110]
        freeze_blue = float(
            (
                (b_roi[:, :, 0] >= 78)
                & (b_roi[:, :, 0] <= 125)
                & (b_roi[:, :, 1] >= 90)
                & (b_roi[:, :, 2] >= 85)
            ).mean()
        )
        return (refresh_yellow, freeze_blue)

    @staticmethod
    def _measure_top_prep_purple_and_magnifiers(hsv: np.ndarray) -> tuple[float, float, float]:
        """测量顶部中央「休息一下」紫框与左右双放大镜像素占比（右放大镜放宽至 H∈[10..38] 兼容三主题）。"""
        top_p = hsv[6:66, 555:725]
        top_purple = float(
            (
                (top_p[:, :, 0] >= 125)
                & (top_p[:, :, 0] <= 162)
                & (top_p[:, :, 1] >= 80)
                & (top_p[:, :, 2] >= 90)
            ).mean()
        )
        l_mag = hsv[12:56, 302:376]
        lm_cyan = float(
            (
                (l_mag[:, :, 0] >= 78)
                & (l_mag[:, :, 0] <= 108)
                & (l_mag[:, :, 1] >= 90)
                & (l_mag[:, :, 2] >= 100)
            ).mean()
        )
        r_mag = hsv[12:56, 908:982]
        rm_amber = float(
            (
                (r_mag[:, :, 0] >= 10)
                & (r_mag[:, :, 0] <= 38)
                & (r_mag[:, :, 1] >= 90)
                & (r_mag[:, :, 2] >= 100)
            ).mean()
        )
        return (top_purple, lm_cyan, rm_amber)

    @staticmethod
    def _measure_top_prep_teal_ratio(hsv: np.ndarray) -> float:
        """测量主题三顶部中央「休息一下」青绿八角框像素占比。"""
        top_mid = hsv[6:66, 555:725]
        return float(
            (
                (top_mid[:, :, 0] >= 72)
                & (top_mid[:, :, 0] <= 98)
                & (top_mid[:, :, 1] >= 80)
                & (top_mid[:, :, 2] >= 85)
            ).mean()
        )

    @classmethod
    def _eval_real_pc_prep_banner(cls, hsv: np.ndarray) -> tuple[bool, float]:
        """评估当前是否处于三套官方主题之一的实机卫戍休整期并返回 ``(hit, confidence)``。

        在 ``_has_stronghold_in_game_chrome(hsv)`` 双锚点门禁下，统一覆盖：
        - 主题一：粉红准备按钮 / 顶紫框+放大镜 / 左右双放大镜共现（防纯白 Toast 瞬态遮挡紫框）；
        - 主题二：乌萨斯浅灰绿内胶囊 + 上方暗边 + 右金黄放大镜；
        - 主题三：青绿准备按钮 / 顶青绿框+右金黄放大镜。
        """
        if not cls._has_stronghold_in_game_chrome(hsv):
            return (False, 0.0)

        pink = cls._measure_top_right_ready_pink_ratio(hsv)
        if pink >= 0.28:
            return (True, round(min(0.90, 0.72 + 0.30 * min(pink, 0.60)), 4))

        t3_teal = cls._measure_top_right_ready_teal_ratio(hsv)
        if t3_teal >= 0.30:
            return (True, round(min(0.90, 0.74 + 0.30 * min(t3_teal, 0.55)), 4))

        top_purple, lm_cyan, rm_amber = cls._measure_top_prep_purple_and_magnifiers(hsv)

        t2_sage, t2_top_dark = cls._measure_top_right_ready_sage_capsule(hsv)
        if t2_sage >= 0.45 and t2_top_dark >= 0.45 and rm_amber >= 0.05:
            return (True, round(min(0.88, 0.74 + 0.22 * min(t2_sage, 0.65)), 4))

        if top_purple >= 0.045 and (lm_cyan >= 0.14 or rm_amber >= 0.10):
            return (True, round(min(0.86, 0.74 + 0.80 * min(top_purple, 0.12)), 4))

        top_teal = cls._measure_top_prep_teal_ratio(hsv)
        if top_teal >= 0.16 and rm_amber >= 0.05:
            return (True, round(min(0.86, 0.74 + 0.40 * min(top_teal, 0.30)), 4))

        # 防关闭隐藏 HUD 后第二行 Toast 瞬态遮挡顶部紫/青框：左右双放大镜同时出现仍稳判休整期
        if lm_cyan >= 0.14 and rm_amber >= 0.10:
            return (True, 0.82)

        return (False, 0.0)

    @classmethod
    def _has_real_pc_prep_banner(cls, hsv: np.ndarray) -> bool:
        """判断当前是否处于三主题实机卫戍休整期。"""
        hit, _ = cls._eval_real_pc_prep_banner(hsv)
        return hit

    @staticmethod
    def _detect_emote_panel_open_heuristic(gray: np.ndarray, hsv: np.ndarray) -> bool:
        """去色相结构化检测左下角 2x3 深黑圆角「交流」表情面板是否处于展开状态。

        校验面板暗底 + 上/右/下三边锐利外框 + （经典青绿表情占比 或 不限色相的内部亮前景与
        底部居中分页圆点 ``y=616..630, x=125..195`` 黑白对比），彻底避免翻至第 2/3 页仅剩
        1 枚非青绿表情时误报关闭。
        """
        roi_g = gray[438:628, 18:302]
        dark_ratio = float((roi_g <= 68).mean())
        if dark_ratio < 0.45:
            return False

        top_strip = gray[430:444, 34:286].astype(np.float32)
        top_edge = float(np.abs(np.diff(top_strip, axis=0)).mean(axis=1).max())
        bot_strip = gray[624:638, 34:286].astype(np.float32)
        bot_edge = float(np.abs(np.diff(bot_strip, axis=0)).mean(axis=1).max())
        right_strip = gray[452:614, 298:312].astype(np.float32)
        right_edge = float(np.abs(np.diff(right_strip, axis=1)).mean(axis=0).max())
        if top_edge < 18.0 or right_edge < 14.0 or bot_edge < 12.0:
            return False

        roi_h = hsv[438:628, 18:302]
        cyan_ratio = float(
            (
                (roi_h[:, :, 0] >= 65)
                & (roi_h[:, :, 0] <= 102)
                & (roi_h[:, :, 1] >= 55)
                & (roi_h[:, :, 2] >= 100)
            ).mean()
        )
        if cyan_ratio >= 0.12:
            return True

        fg_bright = float((roi_g >= 115).mean())
        dots = gray[616:630, 125:195]
        dot_white = float((dots >= 180).mean())
        dot_dark = float((dots <= 55).mean())
        return fg_bright >= 0.03 and dot_white >= 0.015 and dot_dark >= 0.65

    @staticmethod
    def _detect_unit_selected_heuristic(frame: np.ndarray) -> bool:
        """基于左侧角色青蓝生命条（排除拖拽售出态）判断角色详情特写面板是否展开。"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if not SceneDetector._has_real_pc_prep_banner(hsv):
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
        """判断休整期商店是否处于展开状态（脱离资金刷新变灰依赖，兼容实机与合成测试帧）。"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        refresh_yellow, freeze_blue = SceneDetector._measure_shop_action_buttons(hsv)

        # 实机休整期画面（含粉红准备按钮态、0 资金灰刷新态及联机 LOADING 紫框+放大镜态）
        if SceneDetector._has_real_pc_prep_banner(hsv):
            return (freeze_blue >= 0.45) or (refresh_yellow >= 0.25 and freeze_blue >= 0.25)

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
    def _measure_vertical_edge_strength(
        v_channel: np.ndarray, x: int, half_win: int = 12
    ) -> float:
        """测量 1280x720 商店卡片高度带 (y=540..695) 在 ``x ± half_win`` 内的垂直边框局部峰值。

        先沿垂直方向对每列梯度取均值再在 ``±half_win`` 窗口取列峰最大值：
        - 放宽至 ``±12px`` 可彻底消除分辨率缩放/间距偏移导致的边界断崖衰减；
        - 按列求均值再取峰仅响应贯穿卡高的完整竖直边框，空白托盘杂波响应极低（~1~2）。
        """
        width = v_channel.shape[1]
        x_start = max(0, x - half_win)
        x_end = min(width, x + half_win + 1)
        if x_end - x_start < 2:
            return 0.0
        strip = v_channel[540:695, x_start:x_end].astype(np.float32)
        col_grad = np.abs(np.diff(strip, axis=1)).mean(axis=0)
        if col_grad.size == 0:
            return 0.0
        return float(col_grad.max())

    @staticmethod
    def _measure_shop_item_bottom_bar_ratio(
        hsv: np.ndarray,
        x1: int,
        x2: int,
        *,
        strict_blue_purple: bool = False,
    ) -> float:
        """测量商店底部装饰条带 (y=690..712) 属于三套主题商品卡专属底条的像素占比。

        - 主题一：蓝紫/洋红底条（``H∈[100..152], S>=90, V>=115``）；
        - 主题二：乌萨斯红棕底条（``H<=14 | H>=168, S>=110, V>=75``）；
        - 主题三：罗德岛青绿底条（``H∈[76..96], S>=45, V>=70``）。
        """
        bot = hsv[690:712, x1:x2]
        if bot.size == 0:
            return 0.0
        h, s, v = bot[:, :, 0], bot[:, :, 1], bot[:, :, 2]
        if strict_blue_purple:
            mask = (h >= 105) & (h <= 138) & (s >= 110) & (v >= 125)
            return float(mask.mean())
        t1 = (h >= 100) & (h <= 152) & (s >= 90) & (v >= 115)
        t2 = ((h <= 14) | (h >= 168)) & (s >= 110) & (v >= 75)
        t3 = (h >= 76) & (h <= 96) & (s >= 45) & (v >= 70)
        return float((t1 | t2 | t3).mean())

    @classmethod
    def _has_shop_upgrade_card(cls, hsv: np.ndarray, x1: int, x2: int) -> bool:
        """判断给定横向区间是否存在三主题通用的「升级商店 (LEVEL)」卡片。

        兼容主题一灰/紫红 ``[G]``、主题二绿黑 CRT ``LEVEL I [G]`` 及主题三灰/青绿 ``LEVEL III``：
        1. 左右 ``±12px`` 竖边局部峰值达标（``edge_l >= 12.0, edge_r >= 14.0``）；
        2. 非贴边升级卡（``x1 >= 44``）执行左邻双排斥（三主题底条 ``>= 0.25`` 或更左商品卡外框 ``edge(x1-44)>=18 & edge_l<16``）；
        3. 内容双分支：亮框态（T1 灰/紫红、T3 青绿）或暗屏高对比态（T2 绿黑 CRT、T3 灰态）。
        """
        v_ch = hsv[:, :, 2]
        v_level = v_ch[552:628, x1:x2]
        v_all = v_ch[540:695, x1:x2]
        if v_level.size == 0 or v_all.size == 0:
            return False

        edge_l = cls._measure_vertical_edge_strength(v_ch, x1, half_win=12)
        edge_r = cls._measure_vertical_edge_strength(v_ch, x2, half_win=12)
        if edge_l < 12.0 or edge_r < 14.0:
            return False

        # 若 x1 >= 44 且紧邻左侧外部属于更左一槽商品卡（连续底条或更左强外框+弱内纹），排斥为非升级卡
        if x1 >= 44:
            out_left_bar = cls._measure_shop_item_bottom_bar_ratio(
                hsv, max(0, x1 - 40), max(1, x1 - 12), strict_blue_purple=False
            )
            if out_left_bar >= 0.25:
                return False
            out_item_left_edge = cls._measure_vertical_edge_strength(
                v_ch, x1 - 44, half_win=10
            )
            if out_item_left_edge >= 18.0 and edge_l < 16.0:
                return False

        # 分支 (a)：亮态升级卡（T1 灰态/紫红 [G]、T3 青绿 [G]）
        level_bright = float((v_level > 120).mean())
        all_bright = float((v_all > 105).mean())
        if level_bright >= 0.25 and all_bright >= 0.20:
            return True

        # 分支 (b)：暗屏态升级卡（T2 绿黑 CRT LEVEL I [G] 与 T3 灰态 LEVEL III）
        lvl_std = float(v_level.std())
        lvl_p90 = float(np.percentile(v_level, 90))
        btn_v = v_ch[640:682, x1 + 12 : x2 - 12]
        if btn_v.size == 0:
            return False
        btn_p95 = float(np.percentile(btn_v, 95))
        return (
            lvl_std >= 24.0
            and lvl_p90 >= 120.0
            and btn_p95 >= 140.0
            and edge_l >= 14.0
            and edge_r >= 24.0
        )

    @classmethod
    def _has_shop_item_card(cls, hsv: np.ndarray, x1: int, x2: int) -> bool:
        """判断给定横向区间是否存在三主题商品卡槽（兼容在售商品卡、双头鹰空槽与 ``[ NO INFO ]`` 暗槽）。

        对左专属非重叠区 ``[x1+4..x1+40]`` 与右主区 ``[x1+48..x2-8]`` 施加独立分段约束，
        并以中位数/p75 抗大号游戏鼠标指针压边干扰，杜绝亮地图左边缘导致 4 误判 5 / 5 误判 6。
        """
        v_ch = hsv[:, :, 2]
        edge_l = cls._measure_vertical_edge_strength(v_ch, x1, half_win=12)
        edge_r = cls._measure_vertical_edge_strength(v_ch, x2, half_win=12)
        if edge_l < 10.0 or edge_r < 10.0:
            return False

        left_x1 = min(x2 - 1, x1 + 4)
        left_x2 = min(x2, x1 + 40)
        right_x1 = min(x2 - 1, x1 + 48)
        right_x2 = max(right_x1 + 1, x2 - 8)
        if left_x2 <= left_x1 or right_x2 <= right_x1:
            return False

        mid_seam_edge = cls._measure_vertical_edge_strength(v_ch, x1 + 44, half_win=8)

        # 1. 三主题底部商品卡专属装饰底条（左右两段同时命中，且无升级卡左框断层）
        left_bar = cls._measure_shop_item_bottom_bar_ratio(
            hsv, left_x1, left_x2, strict_blue_purple=False
        )
        right_bar = cls._measure_shop_item_bottom_bar_ratio(
            hsv, right_x1, right_x2, strict_blue_purple=False
        )
        if left_bar >= 0.25 and right_bar >= 0.25 and mid_seam_edge < 32.0:
            return True

        left_v = v_ch[555:680, left_x1:left_x2]
        right_v = v_ch[555:680, right_x1:right_x2]
        left_h = hsv[555:680, left_x1:left_x2]
        right_h = hsv[555:680, right_x1:right_x2]
        if left_v.size == 0 or right_v.size == 0:
            return False

        left_mean = float(left_v.mean())
        right_mean = float(right_v.mean())
        left_std = float(left_v.std())
        right_std = float(right_v.std())
        left_bright = float((left_v > 110).mean())
        right_bright = float((right_v > 110).mean())

        # 2. 在售亮卡 / 双头鹰空槽通用结构特征（要求 x1+44 无低一阶升级卡左外框断层）
        if mid_seam_edge < 20.0:
            if (
                left_bright >= 0.30
                and right_bright >= 0.30
                and edge_l >= 12.0
                and edge_r >= 12.0
            ):
                return True
            if (
                left_mean >= 65.0
                and right_mean >= 65.0
                and edge_l >= 16.0
                and edge_r >= 16.0
            ):
                return True

        # 3. 已售空 [ NO INFO ] 暗灰/暗蓝卡槽：兼容 T1 暗蓝、T3 无饱和度深灰及大号鼠标指针压边
        if mid_seam_edge < 16.0:
            if (
                22.0 <= left_mean <= 68.0
                and 22.0 <= right_mean <= 68.0
                and left_bright <= 0.12
                and right_bright <= 0.12
            ):
                if left_std <= 22.0 and right_std <= 22.0:
                    return True
                left_dk_blue = float(
                    (
                        (left_h[:, :, 0] >= 95)
                        & (left_h[:, :, 0] <= 145)
                        & (left_h[:, :, 1] >= 35)
                        & (left_h[:, :, 2] >= 18)
                        & (left_h[:, :, 2] <= 85)
                    ).mean()
                )
                right_dk_blue = float(
                    (
                        (right_h[:, :, 0] >= 95)
                        & (right_h[:, :, 0] <= 145)
                        & (right_h[:, :, 1] >= 35)
                        & (right_h[:, :, 2] >= 18)
                        & (right_h[:, :, 2] <= 85)
                    ).mean()
                )
                if (
                    left_std <= 32.0
                    and right_std <= 32.0
                    and left_dk_blue >= 0.50
                    and right_dk_blue >= 0.50
                ):
                    return True

            # 中位数 + p75 鲁棒分支：当游戏白色大光标压在空槽边缘拉高局部均值/方差时仍精准命中
            left_med = float(np.median(left_v))
            right_med = float(np.median(right_v))
            left_p75 = float(np.percentile(left_v, 75))
            right_p75 = float(np.percentile(right_v, 75))
            if (
                20.0 <= left_med <= 68.0
                and 20.0 <= right_med <= 68.0
                and left_p75 <= 80.0
                and right_p75 <= 80.0
                and left_bright <= 0.20
                and right_bright <= 0.20
            ):
                return True

        return False

    @classmethod
    def _detect_shop_slot_count_heuristic(cls, frame: np.ndarray) -> int | None:
        """检测展开后的卫戍协议商店栏位数（1 本 4 槽 / 2~3 本 5 槽 / 4~6 本 6 槽）。

        依据实机精确布局：最右卡槽固定于 x=936..1090，卡片步距 162px，升级卡宽 109px
        紧贴最左商品卡左侧。升级时向左侧延伸：
        - 6 槽：升级卡 x=8..117，或无 5 槽升级卡时第 6 槽商品卡/已售空槽 x=126..280；
        - 5 槽：升级卡 x=170..279，或无 4 槽升级卡时第 5 槽商品卡/已售空槽 x=288..442；
        - 4 槽：升级卡 x=332..441（其余兜底默认 4 槽）。
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if not cls._has_real_pc_prep_banner(hsv):
            return None
        refresh_yellow, freeze_blue = cls._measure_shop_action_buttons(hsv)
        if not ((freeze_blue >= 0.45) or (refresh_yellow >= 0.25 and freeze_blue >= 0.25)):
            return None

        has_upg6 = cls._has_shop_upgrade_card(hsv, 8, 117)
        has_upg5 = cls._has_shop_upgrade_card(hsv, 170, 279)
        has_upg4 = cls._has_shop_upgrade_card(hsv, 332, 441)

        # 1. 优先判定 6 槽（4~6 本）：最左升级卡 [8..117]，或满级无升级卡时第 6 槽商品卡/空槽 [126..280]
        if has_upg6 or (not has_upg5 and cls._has_shop_item_card(hsv, 126, 280)):
            return 6

        # 2. 次优判定 5 槽（2~3 本）：升级卡 [170..279]，或无 4 槽升级卡时第 5 槽商品卡/空槽 [288..442]
        if has_upg5 or (not has_upg4 and cls._has_shop_item_card(hsv, 288, 442)):
            return 5

        # 3. 1 本 4 槽（升级卡 [332..441] 或兜底回退 4 槽）
        if has_upg4:
            return 4
        return 4

    @staticmethod
    def _detect_confirm_has_dont_ask_heuristic(gray: np.ndarray) -> bool:
        """判断确认弹窗中央偏下区域是否存在「☐ 本局不再提示」勾选行。"""
        return float((gray[388:418, 545:735] >= 100).mean()) >= 0.05

    @classmethod
    def _detect_modal_confirm_heuristic(
        cls, gray: np.ndarray, hsv: np.ndarray
    ) -> SceneDetectionResult | None:
        """检测回合结束资金/策略确认弹窗及局内 ESC 退出确认弹窗（统一命中 ``MODAL_CONFIRM``）。

        须置于 ``_measure_top_right_ready_pink_ratio`` 之前执行，并具备以下抗干扰加固：
        1. 中央全宽黑底使用分位数（80 分位 < 30.0）替代脆弱标准差，避免光标/选框压边拉高 std；
        2. 左下灰取消栏避开中央文字区且放宽至 ``>= 0.74``；
        3. 右下暗红确认栏聚焦 ``x∈[655..885]∪[980..1080]`` 避开右端深色机械剪影并放宽 ``V >= 42``；
        4. 自动检测 ``gray[388:418, 545:735]`` 区分是否包含「本局不再提示」行。
        """
        # 1. 中央全宽黑底确认框左右空白边缘：y=200..440, x∈[20..220]∪[1060..1260]
        dark_band = np.concatenate(
            [gray[200:440, 20:220], gray[200:440, 1060:1260]], axis=1
        )
        dark_ratio = float((dark_band < 36).mean())
        dark_p80 = float(np.percentile(dark_band, 80))
        if dark_ratio < 0.82 or dark_p80 >= 30.0:
            return None

        # 2. 左下灰色「☒ 取消」栏（避开中央图标文字区）：y=455..515, x∈[30..235]∪[410..615]
        cancel_hsv = np.concatenate(
            [hsv[455:515, 30:235], hsv[455:515, 410:615]], axis=1
        )
        cancel_ratio = float(
            (
                (cancel_hsv[:, :, 1] <= 42)
                & (cancel_hsv[:, :, 2] >= 38)
                & (cancel_hsv[:, :, 2] <= 120)
            ).mean()
        )
        if cancel_ratio < 0.74:
            return None

        # 3. 右下暗红「☑ 确认」栏（避开最右端深色机械剪影）：y=455..515, x∈[655..885]∪[980..1080]
        confirm_hsv = np.concatenate(
            [hsv[455:515, 655:885], hsv[455:515, 980:1080]], axis=1
        )
        confirm_red_ratio = float(
            (
                ((confirm_hsv[:, :, 0] <= 14) | (confirm_hsv[:, :, 0] >= 166))
                & (confirm_hsv[:, :, 1] >= 95)
                & (confirm_hsv[:, :, 2] >= 42)
                & (confirm_hsv[:, :, 2] <= 215)
            ).mean()
        )
        if confirm_red_ratio < 0.72:
            return None

        confirm_has_dont_ask = cls._detect_confirm_has_dont_ask_heuristic(gray)
        return SceneDetectionResult(
            scene=GameScene.MODAL_CONFIRM,
            confidence=0.95,
            shop_open=False,
            confirm_has_dont_ask=confirm_has_dont_ask,
        )

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
                shop_open=False,
                choice_count=3,
            )

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 0. 优先检测回合结束/ESC 退出确认弹窗（须位于粉红准备按钮检测之前，避免模糊背景误判）
        confirm_result = self._detect_modal_confirm_heuristic(gray, hsv)
        if confirm_result is not None:
            return confirm_result

        # 1. 检测二选一/三选一/多人六选一策略与悬赏决策弹窗（优先实机暗底亮框检测，再兼容亮底合成卡片）
        modal_result = self._detect_modal_choice_heuristic(gray, hsv)
        if modal_result is not None:
            return modal_result

        # 局内身份门禁：2a 与 2b 均强制共现卫戍协议局内双锚点，局外界面严守 UNKNOWN_FREE_CURSOR
        in_game_chrome = self._has_stronghold_in_game_chrome(hsv)

        # 2a. 三套官方 PC 端卫戍协议主题休整期强特征（粉红 / 乌萨斯浅灰绿 / 青绿 / 顶栏+放大镜）
        prep_hit, prep_conf = self._eval_real_pc_prep_banner(hsv)
        if in_game_chrome and prep_hit:
            shop_open = self._detect_shop_open(frame, shop_template)
            unit_selected = self._detect_unit_selected(frame, unit_template)
            shop_slot_count = (
                self._detect_shop_slot_count_heuristic(frame) if shop_open else None
            )
            covenant_count, covenant_overflow = self._detect_covenants_heuristic(hsv, gray)
            has_teammates = self._detect_teammates_heuristic(
                gray, unit_selected=unit_selected
            )
            emote_panel_open = (
                self._detect_emote_panel_open_heuristic(gray, hsv)
                if (not shop_open and not unit_selected)
                else False
            )
            return SceneDetectionResult(
                scene=GameScene.BATTLE_PREP,
                confidence=prep_conf,
                shop_open=shop_open,
                choice_count=3,
                unit_selected=unit_selected,
                shop_slot_count=shop_slot_count,
                covenant_count=covenant_count,
                covenant_overflow=covenant_overflow,
                has_teammates=has_teammates,
                emote_panel_open=emote_panel_open,
            )

        # 2b. 通用底部整备槽位条纹特征（需先通过局内身份门禁，彻底免疫战前编队/干员列表/基建误判）
        if in_game_chrome:
            bench_score = self._measure_bench_slots_score(gray)
            if bench_score >= 0.60:
                # 右下角准备就绪按钮区饱和高亮色块特征
                ready_vivid_ratio = self._measure_ready_button_ratio(hsv)
                if 0.10 <= ready_vivid_ratio <= 0.88:
                    confidence = min(
                        0.80,
                        0.64 + 0.12 * bench_score + 0.08 * min(ready_vivid_ratio / 0.35, 1.0),
                    )
                    shop_open = self._detect_shop_open(frame, shop_template)
                    unit_selected = self._detect_unit_selected(frame, unit_template)
                    covenant_count, covenant_overflow = self._detect_covenants_heuristic(
                        hsv, gray
                    )
                    has_teammates = self._detect_teammates_heuristic(
                        gray, unit_selected=unit_selected
                    )
                    emote_panel_open = (
                        self._detect_emote_panel_open_heuristic(gray, hsv)
                        if (not shop_open and not unit_selected)
                        else False
                    )
                    return SceneDetectionResult(
                        scene=GameScene.BATTLE_PREP,
                        confidence=round(confidence, 4),
                        shop_open=shop_open,
                        choice_count=3,
                        unit_selected=unit_selected,
                        covenant_count=covenant_count,
                        covenant_overflow=covenant_overflow,
                        has_teammates=has_teammates,
                        emote_panel_open=emote_panel_open,
                    )

                # 右下角无准备按钮高亮块，且顶部中央存在作战波次 HUD 面板 -> 作战期
                combat_hud_score = self._measure_combat_top_hud_score(gray, hsv)
                if ready_vivid_ratio < 0.08 and combat_hud_score >= 0.60:
                    confidence = min(0.76, 0.62 + 0.08 * bench_score + 0.08 * combat_hud_score)
                    unit_selected = self._detect_unit_selected(frame, unit_template)
                    covenant_count, covenant_overflow = self._detect_covenants_heuristic(
                        hsv, gray
                    )
                    has_teammates = self._detect_teammates_heuristic(
                        gray, unit_selected=unit_selected
                    )
                    emote_panel_open = (
                        self._detect_emote_panel_open_heuristic(gray, hsv)
                        if not unit_selected
                        else False
                    )
                    return SceneDetectionResult(
                        scene=GameScene.BATTLE_COMBAT,
                        confidence=round(confidence, 4),
                        shop_open=False,
                        choice_count=3,
                        unit_selected=unit_selected,
                        covenant_count=covenant_count,
                        covenant_overflow=covenant_overflow,
                        has_teammates=has_teammates,
                        emote_panel_open=emote_panel_open,
                    )

        # 置信度不足时一律安全返回未知自由光标模式
        return SceneDetectionResult(
            scene=GameScene.UNKNOWN_FREE_CURSOR,
            confidence=0.0,
            shop_open=False,
            choice_count=3,
        )

    @staticmethod
    def _v_border_strength(
        gx: np.ndarray,
        x_positions: list[int] | tuple[int, ...],
        y1: int,
        y2: int,
        half_win: int = 6,
    ) -> list[float]:
        """测量 1280x720 图像在一组 x 坐标带内的垂直边框 Sobel 峰值均值。"""
        width = gx.shape[1]
        scores: list[float] = []
        for x in x_positions:
            xs = max(0, x - half_win)
            xe = min(width, x + half_win + 1)
            if xe <= xs or y2 <= y1:
                scores.append(0.0)
            else:
                scores.append(float(gx[y1:y2, xs:xe].max(axis=1).mean()))
        return scores

    @staticmethod
    def _has_modal_chrome(gray: np.ndarray, hsv: np.ndarray) -> bool:
        """检查弹窗标志性外围特征：左上角青蓝色决策标题带或右侧方形「收起」按钮边框。"""
        roi_tl = hsv[18:95, 140:580]
        cyan_tl = float(
            (
                (roi_tl[:, :, 0] >= 82)
                & (roi_tl[:, :, 0] <= 105)
                & (roi_tl[:, :, 1] >= 115)
                & (roi_tl[:, :, 2] >= 125)
            ).mean()
        )
        if cyan_tl >= 0.045:
            return True

        sub_btn = gray[310:410, 1135:1280]
        edges_btn = cv2.Canny(sub_btn, 60, 150)
        cnts, _ = cv2.findContours(
            edges_btn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in cnts:
            _, _, w, h = cv2.boundingRect(cnt)
            if 52 <= w <= 88 and 52 <= h <= 88 and 0.78 <= (w / h) <= 1.28:
                return True
        return False

    @staticmethod
    def _detect_white_confirm_bar_index(
        hsv: np.ndarray,
        card_boxes: list[tuple[int, int, int, int]],
    ) -> int | None:
        """检测哪张决策卡底部弹出了高亮度低饱和度的白色「确认选择」横条。"""
        best_idx: int | None = None
        best_ratio: float = 0.0
        for idx, (x1, y1, x2, y2) in enumerate(card_boxes):
            roi = hsv[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            white_ratio = float(((roi[:, :, 2] >= 198) & (roi[:, :, 1] <= 48)).mean())
            if white_ratio >= 0.42 and white_ratio > best_ratio:
                best_ratio = white_ratio
                best_idx = idx
        return best_idx

    @classmethod
    def _eval_6choice_layout(
        cls,
        gray: np.ndarray,
        hsv: np.ndarray,
        gx: np.ndarray,
        cols: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
        row_specs: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
        confirm_boxes: list[tuple[int, int, int, int]],
        excl_bands: tuple[tuple[int, int], tuple[int, int]] | None = None,
    ) -> SceneDetectionResult | None:
        """评估指定 2x3 六选一列布局是否命中（允许 1~5 张已被队友选走的均匀浅灰卡）。"""
        xs = [x for pair in cols for x in pair]
        if excl_bands is not None:
            for ey1, ey2 in excl_bands:
                ex_edges = cls._v_border_strength(gx, xs, ey1, ey2)
                if float(np.mean(ex_edges)) < 165.0 or min(ex_edges) < 85.0:
                    return None

        all_edges: list[float] = []
        dark_count = 0
        for ey1, ey2, iy1, iy2 in row_specs:
            for xl, xr in cols:
                el, er = cls._v_border_strength(gx, (xl, xr), ey1, ey2)
                min_e = min(el, er)
                all_edges.extend((el, er))

                sub_g = gray[iy1:iy2, xl + 16 : xr - 16]
                sub_s = hsv[iy1:iy2, xl + 16 : xr - 16, 1]
                if sub_g.size == 0:
                    return None
                g_mean = float(sub_g.mean())
                s_mean = float(sub_s.mean())
                grey_ratio = float(((sub_g >= 115) & (sub_g <= 215) & (sub_s <= 35)).mean())

                is_dark_card = min_e >= 135.0 and g_mean < 115.0
                is_picked_grey = (
                    min_e >= 105.0
                    and 110.0 <= g_mean <= 208.0
                    and s_mean <= 35.0
                    and grey_ratio >= 0.45
                )
                if is_dark_card:
                    dark_count += 1
                elif not is_picked_grey:
                    return None

        if dark_count < 1:
            return None

        edge_mean = float(np.mean(all_edges))
        if edge_mean < 180.0:
            return None

        selected_idx = cls._detect_white_confirm_bar_index(hsv, confirm_boxes)
        has_mates = cls._detect_teammates_heuristic(gray, unit_selected=False)
        conf = min(0.94, 0.78 + 0.16 * min(edge_mean / 550.0, 1.0))
        return SceneDetectionResult(
            scene=GameScene.MODAL_CHOICE,
            confidence=round(conf, 4),
            shop_open=False,
            choice_count=6,
            selected_choice_index=selected_idx,
            has_teammates=has_mates,
        )

    @classmethod
    def _detect_real_modal_choice(
        cls, gray: np.ndarray, hsv: np.ndarray
    ) -> SceneDetectionResult | None:
        """识别《明日方舟》卫戍协议实机暗底亮框的六选一（2x3）、三选一（1x3）与二选一（1x2）决策界面。"""
        # 休整期正常亮场右上角粉红/青绿准备按钮高亮；弹窗遮罩激活后会压暗至 <0.05
        if (
            cls._measure_top_right_ready_pink_ratio(hsv) >= 0.22
            or cls._measure_top_right_ready_teal_ratio(hsv) >= 0.25
        ):
            return None

        # 2/3/6 选一统一强制要求卫戍协议弹窗外围特征（左上青色标题带或右侧方形收起按钮），免疫局外弹窗
        if not cls._has_modal_chrome(gray, hsv):
            return None

        gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))

        # (A1) 官方 PC 16:9 实机 2x3 六选一网格（列边界 x=[177,456], [519,798], [861,1140]，兼容已被队友选走浅灰卡）
        cols_6_pc = ((177, 456), (519, 798), (861, 1140))
        confirm_boxes_6_pc = [
            (xl + 14, 305, xr - 14, 358) for xl, xr in cols_6_pc
        ] + [
            (xl + 14, 604, xr - 14, 656) for xl, xr in cols_6_pc
        ]
        hit_6_pc = cls._eval_6choice_layout(
            gray,
            hsv,
            gx,
            cols=cols_6_pc,
            row_specs=((100, 320, 108, 305), (400, 618, 410, 600)),
            confirm_boxes=confirm_boxes_6_pc,
            excl_bands=((108, 225), (520, 615)),
        )
        if hit_6_pc is not None:
            return hit_6_pc

        # (A2) 兼容旧宽屏录屏 2x3 六选一网格（列边界 x=[220,478], [528,785], [835,1093]）
        cols_6_wide = ((220, 478), (528, 785), (835, 1093))
        confirm_boxes_6_wide = [
            (xl + 14, 310, xr - 14, 362) for xl, xr in cols_6_wide
        ] + [
            (xl + 14, 582, xr - 14, 636) for xl, xr in cols_6_wide
        ]
        hit_6_wide = cls._eval_6choice_layout(
            gray,
            hsv,
            gx,
            cols=cols_6_wide,
            row_specs=((130, 325, 135, 295), (395, 595, 405, 565)),
            confirm_boxes=confirm_boxes_6_wide,
            excl_bands=None,
        )
        if hit_6_wide is not None:
            return hit_6_wide

        # (B) 单人/常规 1x3 三选一横排检测（列边界 x=[171,460], [515,801], [857,1144]）
        cols_3 = ((171, 460), (515, 801), (857, 1144))
        xs_3 = [x for pair in cols_3 for x in pair]
        mid_edges = cls._v_border_strength(gx, xs_3, 255, 465)
        if float(np.mean(mid_edges)) >= 230.0 and min(mid_edges) >= 135.0:
            interiors_ok = all(
                float(gray[260:445, xl + 16 : xr - 16].mean()) < 115.0
                for xl, xr in cols_3
            )
            if interiors_ok:
                confirm_boxes_3 = [
                    (xl + 14, 458, xr - 14, 514) for xl, xr in cols_3
                ]
                selected_idx = cls._detect_white_confirm_bar_index(hsv, confirm_boxes_3)
                has_mates = cls._detect_teammates_heuristic(gray, unit_selected=False)
                edge_mean = float(np.mean(mid_edges))
                conf = min(0.93, 0.76 + 0.17 * min(edge_mean / 520.0, 1.0))
                return SceneDetectionResult(
                    scene=GameScene.MODAL_CHOICE,
                    confidence=round(conf, 4),
                    shop_open=False,
                    choice_count=3,
                    selected_choice_index=selected_idx,
                    has_teammates=has_mates,
                )

        # (C) 更换指令等 1x2 二选一横排检测（列边界 x=[348,627], [689,969]，左右外围无第 3 卡）
        cols_2 = ((348, 627), (689, 969))
        xs_2 = [x for pair in cols_2 for x in pair]
        edges_2 = cls._v_border_strength(gx, xs_2, 255, 465)
        if float(np.mean(edges_2)) >= 230.0 and min(edges_2) >= 135.0:
            interiors_ok = all(
                float(gray[260:445, xl + 16 : xr - 16].mean()) < 115.0
                for xl, xr in cols_2
            )
            # 按列求垂直均值再取列峰值：仅响应贯穿卡高的竖直第 3 卡边框，对两翼压暗雪山/砖块斜边免疫
            outer_left = float(gx[265:455, 150:310].mean(axis=0).max())
            outer_right = float(gx[265:455, 990:1140].mean(axis=0).max())
            if interiors_ok and outer_left < 180.0 and outer_right < 180.0:
                confirm_boxes_2 = [
                    (xl + 14, 458, xr - 14, 514) for xl, xr in cols_2
                ]
                selected_idx = cls._detect_white_confirm_bar_index(hsv, confirm_boxes_2)
                has_mates = cls._detect_teammates_heuristic(gray, unit_selected=False)
                edge_mean = float(np.mean(edges_2))
                conf = min(0.93, 0.76 + 0.17 * min(edge_mean / 520.0, 1.0))
                return SceneDetectionResult(
                    scene=GameScene.MODAL_CHOICE,
                    confidence=round(conf, 4),
                    shop_open=False,
                    choice_count=2,
                    selected_choice_index=selected_idx,
                    has_teammates=has_mates,
                )

        return None

    def _detect_modal_choice_heuristic(
        self, gray: np.ndarray, hsv: np.ndarray | None = None
    ) -> SceneDetectionResult | None:
        """检测二选一/三选一/多人六选一弹窗（兼容实机暗底亮框卡片与带弹窗标识的合成亮面卡片）。"""
        if hsv is not None:
            real_modal = self._detect_real_modal_choice(gray, hsv)
            if real_modal is not None:
                return real_modal

        # 亮卡回退同样要求具备卫戍协议决策弹窗外围标识，彻底免疫局外分队/活动双卡三卡页误判
        if hsv is None or not self._has_modal_chrome(gray, hsv):
            return None

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

    def _analyze_modal_choice_details(
        self, gray: np.ndarray, hsv: np.ndarray
    ) -> tuple[int, int | None]:
        """在已判定为 ``MODAL_CHOICE`` 时分析可选卡片数（3/6/2）与已选中待确认卡片索引。"""
        real_modal = self._detect_real_modal_choice(gray, hsv)
        if real_modal is not None:
            return (real_modal.choice_count, real_modal.selected_choice_index)
        border_mean = float(gray[14:94, 60:1220].mean())
        cards = self._find_modal_card_spans(gray, border_mean)
        if len(cards) in (2, 3, 4):
            return (len(cards), None)
        return (3, None)

    def _estimate_choice_count(self, frame: np.ndarray) -> int:
        """在已判定为 ``MODAL_CHOICE`` 时估算可选卡片数量（默认 3，多人 6）。"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        count, _ = self._analyze_modal_choice_details(gray, hsv)
        return count

    @staticmethod
    def _score_covenant_badge_vchannel(
        v_ch: np.ndarray,
        gx: np.ndarray,
        gy: np.ndarray,
        cx: float,
        cy: float = 87.0,
        *,
        partial_side: str | None = None,
    ) -> float:
        """脱离单一色相依赖的 HSV-V 通道圆形盟约徽章打分（兼容 T1 蓝弧 / T2 银灰 / T3 青绿半环）。

        参数:
            partial_side: ``None`` 表示完整圆环；``"right"`` 表示右缘被裁切仅评左半圈（``cx=938``）；
                ``"left"`` 表示左缘被裁切仅评右半圈（``cx=299``，用于溢出态滑至右端场景）。
        """
        ix, iy = int(round(cx)), int(round(cy))
        min_x = 280 if partial_side == "left" else 340
        if ix < min_x or ix > 955 or iy < 65 or iy > 105:
            return 0.0

        # 校验圆心正下方层数与中文盟约名高对比文字带 (y=100..129)
        if partial_side == "right":
            x_l, x_r = max(0, ix - 18), min(STANDARD_WIDTH, ix + 8)
        elif partial_side == "left":
            x_l, x_r = max(0, ix - 8), min(STANDARD_WIDTH, ix + 19)
        else:
            x_l, x_r = max(0, ix - 18), min(STANDARD_WIDTH, ix + 19)
        lbl = v_ch[iy + 13 : iy + 42, x_l:x_r]
        if lbl.size == 0:
            return 0.0
        lbl_p95 = float(np.percentile(lbl, 95))
        lbl_c = lbl_p95 - float(np.percentile(lbl, 5))
        if lbl_p95 < 125.0 or lbl_c < 85.0:
            return 0.0

        best_score = 0.0
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                tx, ty = cx + dx, cy + dy
                rx = np.round(tx + 22.5 * _COVENANT_RING_COS).astype(int)
                ry = np.round(ty + 22.5 * _COVENANT_RING_SIN).astype(int)
                in_x = np.round(tx + 12.0 * _COVENANT_RING_COS).astype(int)
                in_y = np.round(ty + 12.0 * _COVENANT_RING_SIN).astype(int)
                valid = (
                    (rx >= 0)
                    & (rx < STANDARD_WIDTH)
                    & (ry >= 0)
                    & (ry < STANDARD_HEIGHT)
                    & (in_x >= 0)
                    & (in_x < STANDARD_WIDTH)
                    & (in_y >= 0)
                    & (in_y < STANDARD_HEIGHT)
                )
                if partial_side == "right":
                    valid &= _COVENANT_RING_COS <= 0.05
                elif partial_side == "left":
                    valid &= _COVENANT_RING_COS >= -0.05
                if int(valid.sum()) < 12:
                    continue

                cos_v = _COVENANT_RING_COS[valid]
                sin_v = _COVENANT_RING_SIN[valid]
                rad_g = np.abs(
                    gx[ry[valid], rx[valid]] * cos_v + gy[ry[valid], rx[valid]] * sin_v
                )
                rad_mean = float(rad_g.mean())

                if partial_side is None:
                    q_l = float(rad_g[cos_v < -0.25].mean())
                    q_r = float(rad_g[cos_v > 0.25].mean())
                    q_u = float(rad_g[sin_v < -0.25].mean())
                    q_d = float(rad_g[sin_v > 0.25].mean())
                    if min(q_l, q_r, q_u, q_d) < 45.0:
                        continue
                elif partial_side == "right":
                    q_l = float(rad_g[cos_v < -0.25].mean())
                    q_u = float(rad_g[sin_v < -0.25].mean())
                    q_d = float(rad_g[sin_v > 0.25].mean())
                    if q_l < 110.0 or min(q_u, q_d) < 70.0:
                        continue
                else:  # partial_side == "left"
                    q_r = float(rad_g[cos_v > 0.25].mean())
                    q_u = float(rad_g[sin_v < -0.25].mean())
                    q_d = float(rad_g[sin_v > 0.25].mean())
                    if q_r < 110.0 or min(q_u, q_d) < 70.0:
                        continue

                ring_v = v_ch[ry[valid], rx[valid]]
                in_v = v_ch[in_y[valid], in_x[valid]]
                ring_p80 = float(np.percentile(ring_v, 80))
                in_mean = float(in_v.mean())
                if ring_p80 < 120.0:
                    continue
                if in_mean > 140.0 and (ring_p80 - in_mean) < 25.0:
                    continue
                if in_mean <= 140.0 and (ring_p80 - in_mean) < 30.0:
                    continue

                score = rad_mean * (1.0 + 0.004 * max(0.0, ring_p80 - in_mean))
                if score > best_score:
                    best_score = score
        return best_score

    @classmethod
    def _detect_covenants_heuristic(
        cls, hsv: np.ndarray, gray: np.ndarray | None = None
    ) -> tuple[int, bool]:
        """基于 HSV-V 通道检测顶部栏下方多盟约数量（0~7）及 ``>7`` 溢出滚动态。

        返回 ``(visible_count, is_overflow)``：
        1. 步距固定为实测精确值 ``80.0px``（``cy = 87.0px``）；
        2. 优先检测溢出态（左对齐 7 枚 ``379 + 80*i`` + 右缘 ``cx=938`` 左半圈或滑至右端后
           左缘 ``cx=299`` 右半圈半露徽章），命中返回 ``(7, True)``；
        3. 否则降序评估 ``n = 7..1`` 居中对称态，采用两端必达 + 鲁棒多数门限 + 间隙非圆排斥。
        """
        _ = gray  # 保留参数签名兼容
        vf = hsv[:, :, 2].astype(np.float32)
        gx = cv2.Sobel(vf, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(vf, cv2.CV_32F, 0, 1, ksize=3)

        # 1. 检测 >7 溢出态（左对齐 7 枚 + 右缘 938 或左缘 299 半露徽章）
        ov_centers = [379.0 + i * 80.0 for i in range(7)]
        ov_scores = [cls._score_covenant_badge_vchannel(vf, gx, gy, cx, 87.0) for cx in ov_centers]
        ov_right_half = cls._score_covenant_badge_vchannel(
            vf, gx, gy, 938.0, 87.0, partial_side="right"
        )
        ov_left_half = cls._score_covenant_badge_vchannel(
            vf, gx, gy, 299.0, 87.0, partial_side="left"
        )
        c7_centers = [640.0 + (i - 3.0) * 80.0 for i in range(7)]
        c7_scores = [cls._score_covenant_badge_vchannel(vf, gx, gy, cx, 87.0) for cx in c7_centers]
        if (
            sum(1 for s in ov_scores if s >= 150.0) >= 6
            and float(np.mean(ov_scores)) >= 190.0
            and (ov_right_half >= 150.0 or ov_left_half >= 150.0)
            and sum(ov_scores) >= sum(c7_scores) * 0.92
        ):
            return (7, True)

        # 2. 降序检测 7..1 居中对称态（精确步距 80.0px，鲁棒多数门限）
        for n in range(7, 0, -1):
            centers = [640.0 + (i - (n - 1) * 0.5) * 80.0 for i in range(n)]
            scores = [cls._score_covenant_badge_vchannel(vf, gx, gy, cx, 87.0) for cx in centers]
            if scores[0] < 130.0 or scores[-1] < 130.0:
                continue
            strong = sum(1 for s in scores if s >= 145.0)
            min_req = n if n <= 2 else (n - 1)
            if strong < min_req or min(scores) < 90.0 or float(np.mean(scores)) < 155.0:
                continue
            if n >= 2:
                gap_scores = [
                    cls._score_covenant_badge_vchannel(
                        vf, gx, gy, (centers[i] + centers[i + 1]) * 0.5, 87.0
                    )
                    for i in range(n - 1)
                ]
                if max(gap_scores) >= 145.0:
                    continue
            return (n, False)

        return (0, False)

    @classmethod
    def _detect_covenant_count_heuristic(
        cls, hsv: np.ndarray, gray: np.ndarray
    ) -> int:
        """向后兼容封装：返回检测到的顶部可见羁绊徽章数量（0~7）。"""
        count, _ = cls._detect_covenants_heuristic(hsv, gray)
        return count

    @staticmethod
    def _detect_teammates_heuristic(
        gray: np.ndarray, *, unit_selected: bool = False
    ) -> bool:
        """检测左侧边缘是否存在多人联机队友纵向头像框（支持 2~4 人联机）。"""
        if unit_selected:
            return False
        avatar_boxes = (
            (19, 160, 67, 66),
            (19, 235, 67, 67),
            (19, 311, 67, 66),
            (19, 387, 67, 66),
        )
        hits: list[bool] = []
        for mx, my, mw, mh in avatar_boxes:
            top_e = float(
                np.abs(
                    np.diff(gray[my - 3 : my + 4, mx : mx + mw].astype(np.float32), axis=0)
                )
                .max(axis=0)
                .mean()
            )
            bot_e = float(
                np.abs(
                    np.diff(
                        gray[my + mh - 3 : my + mh + 4, mx : mx + mw].astype(np.float32),
                        axis=0,
                    )
                )
                .max(axis=0)
                .mean()
            )
            left_e = float(
                np.abs(
                    np.diff(gray[my : my + mh, mx - 3 : mx + 4].astype(np.float32), axis=1)
                )
                .max(axis=1)
                .mean()
            )
            right_e = float(
                np.abs(
                    np.diff(
                        gray[my : my + mh, mx + mw - 3 : mx + mw + 4].astype(np.float32),
                        axis=1,
                    )
                )
                .max(axis=1)
                .mean()
            )
            hits.append(min(top_e, bot_e, left_e, right_e) >= 18.0)
        return (hits[0] and hits[1]) or sum(hits) >= 3

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
