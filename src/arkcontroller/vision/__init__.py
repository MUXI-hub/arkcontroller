"""非侵入式屏幕采样、卫戍协议界面阶段识别与实时模板校准模块。"""

from arkcontroller.vision.calibrator import VisionCalibrator
from arkcontroller.vision.scene_detector import (
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_SCENE_ROIS,
    DEFAULT_SHOP_OPEN_ROI,
    DEFAULT_UNIT_SELECTED_ROI,
    STANDARD_HEIGHT,
    STANDARD_WIDTH,
    GameScene,
    SceneDetectionResult,
    SceneDetector,
)
from arkcontroller.vision.screen_capture import ScreenCapture

__all__ = [
    "DEFAULT_MATCH_THRESHOLD",
    "DEFAULT_SCENE_ROIS",
    "DEFAULT_SHOP_OPEN_ROI",
    "DEFAULT_UNIT_SELECTED_ROI",
    "GameScene",
    "STANDARD_HEIGHT",
    "STANDARD_WIDTH",
    "SceneDetectionResult",
    "SceneDetector",
    "ScreenCapture",
    "VisionCalibrator",
]
