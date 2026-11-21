"""配置文件加载、用户覆盖合并与热重载管理模块。"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from arkcontroller.config.models import KeymapConfig, LayoutProfile

DEFAULT_KEYMAP_FILENAME = "default_bindings.json"
LEGACY_KEYMAP_FILENAME = "default_keymap.json"
DEFAULT_LAYOUT_FILENAME = "stronghold_layout_16_9.json"
USER_OVERRIDE_FILENAME = "user_override.json"

_LAYOUT_TOP_LEVEL_KEYS = frozenset(
    {
        "activity_hub_nodes",
        "pre_match_nodes",
        "squad_edit_nodes",
        "settlement_nodes",
        "shop_zone",
        "bench_zone",
        "equip_zone",
        "choice_zone",
        "board_quad",
        "buttons",
    }
)

_KEYMAP_TOP_LEVEL_KEYS = frozenset(
    {
        "left_deadzone",
        "right_deadzone",
        "cursor_max_speed_px",
        "cursor_gamma",
        "ready_hold_ms",
        "repeat_initial_ms",
        "repeat_interval_ms",
        "use_native_keyboard_shortcuts",
        "flick_distance_ratio",
        "vibration_enabled",
        "trigger_threshold",
        "scroll_speed",
        "allow_retreat_chord",
        "keyboard_shortcuts",
        "bindings",
    }
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """深度合并两个字典并返回新副本，不修改原始入参。"""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_override_structure(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """将覆盖字典规整为标准的 {'keymap': {...}, 'layout': {...}} 分组结构。"""
    normalized: dict[str, dict[str, Any]] = {"keymap": {}, "layout": {}}
    if isinstance(raw.get("keymap"), dict):
        normalized["keymap"] = _deep_merge(normalized["keymap"], raw["keymap"])
    if isinstance(raw.get("layout"), dict):
        normalized["layout"] = _deep_merge(normalized["layout"], raw["layout"])

    # 兼容调用方直接传入扁平的 layout 或 keymap 顶层字段
    for key, value in raw.items():
        if key in ("keymap", "layout"):
            continue
        if key in _LAYOUT_TOP_LEVEL_KEYS:
            normalized["layout"] = _deep_merge(normalized["layout"], {key: value})
        elif key in _KEYMAP_TOP_LEVEL_KEYS:
            normalized["keymap"] = _deep_merge(normalized["keymap"], {key: value})
    return normalized


class ConfigLoader:
    """加载默认配置、合并用户校准覆盖文件并支持热重载。"""

    def __init__(self, config_dir: Path | str | None = None) -> None:
        if config_dir is None:
            repo_configs = Path(__file__).resolve().parents[3] / "configs"
            cwd_configs = Path.cwd() / "configs"
            self.config_dir = repo_configs if repo_configs.exists() else cwd_configs
        else:
            self.config_dir = Path(config_dir)

        self._keymap: KeymapConfig | None = None
        self._layout: LayoutProfile | None = None

    @property
    def keymap_path(self) -> Path:
        """默认按键映射配置文件路径（优先 default_bindings.json，兼容旧名）。"""
        primary = self.config_dir / DEFAULT_KEYMAP_FILENAME
        if primary.exists():
            return primary
        legacy = self.config_dir / LEGACY_KEYMAP_FILENAME
        if legacy.exists():
            return legacy
        return primary

    @property
    def layout_path(self) -> Path:
        """默认 16:9 卫戍协议布局配置文件路径。"""
        return self.config_dir / DEFAULT_LAYOUT_FILENAME

    @property
    def override_path(self) -> Path:
        """用户本地校准与自定义覆盖文件路径。"""
        return self.config_dir / USER_OVERRIDE_FILENAME

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        """读取 UTF-8 编码的 JSON 对象文件。"""
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            raise ValueError(f"配置文件顶层必须为 JSON 对象: {path}")
        return data

    def _read_user_override(self) -> dict[str, dict[str, Any]]:
        """读取并规整用户覆盖配置；文件不存在时返回空结构。"""
        if not self.override_path.exists():
            return {"keymap": {}, "layout": {}}
        raw = self._read_json(self.override_path)
        return _normalize_override_structure(raw)

    def load(self) -> tuple[KeymapConfig, LayoutProfile]:
        """加载并返回 (KeymapConfig, LayoutProfile)；若已加载则返回缓存实例。"""
        if self._keymap is not None and self._layout is not None:
            return self._keymap, self._layout
        return self.reload()

    def reload(self) -> tuple[KeymapConfig, LayoutProfile]:
        """强制从磁盘重新读取默认配置与用户覆盖配置并刷新缓存。"""
        keymap_raw = self._read_json(self.keymap_path)
        layout_raw = self._read_json(self.layout_path)
        override = self._read_user_override()

        merged_keymap = _deep_merge(keymap_raw, override["keymap"])
        merged_layout = _deep_merge(layout_raw, override["layout"])

        self._keymap = KeymapConfig.from_dict(merged_keymap)
        self._layout = LayoutProfile.from_dict(merged_layout)
        return self._keymap, self._layout

    def save_user_override(self, override_data: dict[str, Any]) -> Path:
        """将新覆盖项深度合并保存至 configs/user_override.json 并刷新内部缓存。"""
        if not isinstance(override_data, dict):
            raise TypeError("override_data 必须为字典类型")

        existing = self._read_user_override()
        incoming = _normalize_override_structure(override_data)
        merged = {
            "keymap": _deep_merge(existing["keymap"], incoming["keymap"]),
            "layout": _deep_merge(existing["layout"], incoming["layout"]),
        }

        self.config_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.override_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as fp:
            json.dump(merged, fp, ensure_ascii=False, indent=2)
            fp.write("\n")
        os.replace(tmp_path, self.override_path)

        # 若默认配置文件存在则立即校验并刷新缓存
        if self.keymap_path.exists() and self.layout_path.exists():
            self.reload()
        else:
            self._keymap = None
            self._layout = None
        return self.override_path
