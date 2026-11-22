"""配置文件加载、用户覆盖合并与热重载管理模块。"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
from typing import Any

from arkcontroller.config.models import KeymapConfig, LayoutProfile

DEFAULT_KEYMAP_FILENAME = "default_bindings.json"
LEGACY_KEYMAP_FILENAME = "default_keymap.json"
DEFAULT_LAYOUT_FILENAME = "stronghold_layout_16_9.json"
USER_OVERRIDE_FILENAME = "user_override.json"

_LAYOUT_TOP_LEVEL_KEYS = frozenset(
    {
        "shop_zone",
        "bench_zone",
        "equip_zone",
        "choice_zone",
        "choice_2_zone",
        "choice_6_top_zone",
        "choice_6_bottom_zone",
        "emote_top_zone",
        "emote_bottom_zone",
        "covenant_zone",
        "board_quad",
        "mirrored_board_quad",
        "mirrored_equip_zone",
        "buttons",
        "view_overrides",
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
        "left_stick_mode",
        "left_stick_dpad_threshold",
        "left_stick_dpad_release_threshold",
        "auto_toggle_shop",
        "hide_hud_from_capture",
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

    def __init__(
        self,
        config_dir: Path | str | None = None,
        *,
        use_user_override: bool = True,
    ) -> None:
        self._fallback_config_dir: Path | None = None
        if config_dir is None:
            if getattr(sys, "frozen", False):
                exe_dir = Path(sys.executable).resolve().parent
                bundle_dir = Path(getattr(sys, "_MEIPASS", exe_dir))
                self.config_dir = exe_dir / "configs"
                self._fallback_config_dir = bundle_dir / "configs"
            else:
                repo_configs = Path(__file__).resolve().parents[3] / "configs"
                cwd_configs = Path.cwd() / "configs"
                self.config_dir = repo_configs if repo_configs.exists() else cwd_configs
        else:
            self.config_dir = Path(config_dir)

        self.use_user_override: bool = bool(use_user_override)
        self._keymap: KeymapConfig | None = None
        self._layout: LayoutProfile | None = None

    @property
    def keymap_path(self) -> Path:
        """默认按键映射配置文件路径（优先外置 configs/，冻结态缺省时回退内置 _MEIPASS/configs/）。"""
        for base_dir in (self.config_dir, self._fallback_config_dir):
            if base_dir is None:
                continue
            primary = base_dir / DEFAULT_KEYMAP_FILENAME
            if primary.exists():
                return primary
            legacy = base_dir / LEGACY_KEYMAP_FILENAME
            if legacy.exists():
                return legacy
        return self.config_dir / DEFAULT_KEYMAP_FILENAME

    @property
    def layout_path(self) -> Path:
        """默认 16:9 卫戍协议布局配置文件路径（冻结态缺省时回退内置 _MEIPASS/configs/）。"""
        primary = self.config_dir / DEFAULT_LAYOUT_FILENAME
        if primary.exists():
            return primary
        if self._fallback_config_dir is not None:
            fallback = self._fallback_config_dir / DEFAULT_LAYOUT_FILENAME
            if fallback.exists():
                return fallback
        return primary

    @property
    def override_path(self) -> Path:
        """用户本地校准与自定义覆盖文件路径（冻结态始终指向可写的 exe_dir/configs/user_override.json）。"""
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
        override = (
            self._read_user_override()
            if self.use_user_override
            else {"keymap": {}, "layout": {}}
        )

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
