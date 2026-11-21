"""卫戍协议实时视觉模板采集与校准器。

负责将游戏 ``1280x720`` 采样帧中的标志性区域（默认 ROI 或用户框选 ROI）
裁剪保存为 ``assets/templates/<scene_value>.png`` 与配套归一化元数据
``<scene_value>.json``，供 ``SceneDetector.reload_templates()`` 即刻加载生效。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from arkcontroller.vision.scene_detector import (
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_SCENE_ROIS,
    DEFAULT_SHOP_OPEN_ROI,
    DEFAULT_UNIT_SELECTED_ROI,
    STANDARD_HEIGHT,
    STANDARD_WIDTH,
    GameScene,
    _normalize_frame,
)

if TYPE_CHECKING:
    from arkcontroller.config.loader import ConfigLoader

logger = logging.getLogger(__name__)

_FALLBACK_ROI: tuple[float, float, float, float] = (0.30, 0.30, 0.40, 0.40)


def _imwrite_png(path: Path, image: np.ndarray) -> None:
    """兼容任意 Windows 路径的 PNG 图像写入。"""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"无法将裁剪后的模板编码为 PNG: {path}")
    encoded.tofile(str(path))


def _crop_by_roi_norm(
    frame: np.ndarray, roi_norm: tuple[float, float, float, float]
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """按归一化 ROI ``(u, v, w, h)`` 裁剪 1280x720 帧并返回裁剪图及像素对齐后的 ROI。"""
    if len(roi_norm) != 4:
        raise ValueError(f"roi_norm 必须包含 4 个浮点数 (u, v, w, h)，当前为: {roi_norm!r}")

    u, v, w, h = (float(roi_norm[0]), float(roi_norm[1]), float(roi_norm[2]), float(roi_norm[3]))
    if not all(np.isfinite([u, v, w, h])):
        raise ValueError(f"roi_norm 包含非法非有限数值: {roi_norm!r}")
    if w <= 0.0 or h <= 0.0:
        raise ValueError(f"roi_norm 的宽高必须大于 0，当前为: {roi_norm!r}")

    u_clamped = max(0.0, min(1.0, u))
    v_clamped = max(0.0, min(1.0, v))
    u_end = max(u_clamped, min(1.0, u + w))
    v_end = max(v_clamped, min(1.0, v + h))

    frame_h, frame_w = frame.shape[:2]
    x1 = int(round(u_clamped * frame_w))
    y1 = int(round(v_clamped * frame_h))
    x2 = int(round(u_end * frame_w))
    y2 = int(round(v_end * frame_h))

    x1 = max(0, min(frame_w - 1, x1))
    y1 = max(0, min(frame_h - 1, y1))
    x2 = max(x1, min(frame_w, x2))
    y2 = max(y1, min(frame_h, y2))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"归一化 ROI 裁剪区域像素面积为零: roi_norm={roi_norm!r} -> ({x1}, {y1}, {x2}, {y2})"
        )

    cropped = np.ascontiguousarray(frame[y1:y2, x1:x2])
    aligned_roi = (
        round(x1 / frame_w, 6),
        round(y1 / frame_h, 6),
        round((x2 - x1) / frame_w, 6),
        round((y2 - y1) / frame_h, 6),
    )
    return cropped, aligned_roi


class VisionCalibrator:
    """卫戍协议界面模板采样器与用户校准覆盖保存协调器。"""

    def __init__(
        self,
        templates_dir: Path,
        loader: ConfigLoader,
        default_threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> None:
        self.templates_dir: Path = Path(templates_dir)
        self.loader: ConfigLoader = loader
        self.default_threshold: float = float(default_threshold)

    def capture_scene_template(
        self,
        frame: np.ndarray,
        scene: GameScene,
        roi_norm: tuple[float, float, float, float] | None = None,
        threshold: float | None = None,
    ) -> Path:
        """从 ``1280x720`` 画面裁剪指定场景的标志性 ROI，保存 ``.png`` 与 ``.json`` 元数据。

        参数:
            frame: 当前游戏 16:9 视口采样帧（若分辨率不同将自动缩放到 1280x720）。
            scene: 目标界面阶段枚举。
            roi_norm: 可选归一化裁剪区域 ``(u, v, w, h)``，缺省使用该场景默认标志性 ROI。
            threshold: 可选匹配阈值，缺省为 ``0.82``。

        返回:
            已保存的模板 ``.png`` 文件路径。
        """
        if scene == GameScene.UNKNOWN_FREE_CURSOR:
            raise ValueError("不能为 UNKNOWN_FREE_CURSOR 采集场景识别模板")

        normalized = _normalize_frame(frame)
        if normalized is None:
            raise ValueError("输入画面帧无效，无法裁剪并保存场景模板")

        effective_roi = roi_norm if roi_norm is not None else DEFAULT_SCENE_ROIS.get(scene, _FALLBACK_ROI)
        cropped, aligned_roi = _crop_by_roi_norm(normalized, effective_roi)

        self.templates_dir.mkdir(parents=True, exist_ok=True)
        png_path = self.templates_dir / f"{scene.value}.png"
        json_path = self.templates_dir / f"{scene.value}.json"

        _imwrite_png(png_path, cropped)

        effective_threshold = (
            float(np.clip(threshold, 0.1, 0.999))
            if threshold is not None
            else self.default_threshold
        )
        metadata = {
            "scene": scene.value,
            "template_file": png_path.name,
            "frame_size": [STANDARD_WIDTH, STANDARD_HEIGHT],
            "roi_norm": list(aligned_roi),
            "threshold": effective_threshold,
        }
        json_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        logger.info(
            "已采集场景模板: scene=%s, path=%s, roi_norm=%s",
            scene.value,
            png_path,
            aligned_roi,
        )
        return png_path

    def capture_shop_open_template(
        self,
        frame: np.ndarray,
        roi_norm: tuple[float, float, float, float] | None = None,
        threshold: float | None = None,
    ) -> Path:
        """采集休整期商店展开状态辅助模板 ``shop_open.png`` 与 ``shop_open.json``。"""
        normalized = _normalize_frame(frame)
        if normalized is None:
            raise ValueError("输入画面帧无效，无法保存商店展开状态模板")

        effective_roi = roi_norm if roi_norm is not None else DEFAULT_SHOP_OPEN_ROI
        cropped, aligned_roi = _crop_by_roi_norm(normalized, effective_roi)

        self.templates_dir.mkdir(parents=True, exist_ok=True)
        png_path = self.templates_dir / "shop_open.png"
        json_path = self.templates_dir / "shop_open.json"

        _imwrite_png(png_path, cropped)

        effective_threshold = (
            float(np.clip(threshold, 0.1, 0.999))
            if threshold is not None
            else self.default_threshold
        )
        metadata = {
            "scene": "shop_open",
            "template_file": png_path.name,
            "frame_size": [STANDARD_WIDTH, STANDARD_HEIGHT],
            "roi_norm": list(aligned_roi),
            "threshold": effective_threshold,
        }
        json_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return png_path

    def capture_unit_selected_template(
        self,
        frame: np.ndarray,
        roi_norm: tuple[float, float, float, float] | None = None,
        threshold: float | None = None,
    ) -> Path:
        """采集选取角色详情缩放状态辅助模板 ``unit_selected.png`` 与 ``unit_selected.json``。"""
        normalized = _normalize_frame(frame)
        if normalized is None:
            raise ValueError("输入画面帧无效，无法保存选中角色状态模板")

        effective_roi = roi_norm if roi_norm is not None else DEFAULT_UNIT_SELECTED_ROI
        cropped, aligned_roi = _crop_by_roi_norm(normalized, effective_roi)

        self.templates_dir.mkdir(parents=True, exist_ok=True)
        png_path = self.templates_dir / "unit_selected.png"
        json_path = self.templates_dir / "unit_selected.json"

        _imwrite_png(png_path, cropped)

        effective_threshold = (
            float(np.clip(threshold, 0.1, 0.999))
            if threshold is not None
            else self.default_threshold
        )
        metadata = {
            "scene": "unit_selected",
            "template_file": png_path.name,
            "frame_size": [STANDARD_WIDTH, STANDARD_HEIGHT],
            "roi_norm": list(aligned_roi),
            "threshold": effective_threshold,
        }
        json_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return png_path

    def save_layout_override(self, override_data: dict[str, Any]) -> Path:
        """通过关联的 ``ConfigLoader`` 保存校准后的布局/键位覆盖配置。"""
        return self.loader.save_user_override(override_data)

    def list_installed_templates(self) -> list[str]:
        """返回模板目录中当前已采集的模板名称列表（不含扩展名）。"""
        if not self.templates_dir.exists() or not self.templates_dir.is_dir():
            return []
        return sorted(p.stem for p in self.templates_dir.glob("*.png"))

    def delete_scene_template(self, scene: GameScene) -> bool:
        """删除指定场景已采集的 ``.png`` 模板与 ``.json`` 元数据。"""
        removed = False
        for suffix in (".png", ".json"):
            target = self.templates_dir / f"{scene.value}{suffix}"
            if target.exists():
                target.unlink()
                removed = True
        return removed
