"""PyInstaller 冻结运行环境（sys.frozen）、无控制台流保护、图标加载与双层配置解析单元测试。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import struct
import sys
from unittest import mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from arkcontroller import __version__ as APP_VERSION
from arkcontroller.config.loader import (
    _KEYMAP_TOP_LEVEL_KEYS,
    _LAYOUT_TOP_LEVEL_KEYS,
    ConfigLoader,
)
import arkcontroller.main as app_main
from arkcontroller.platform import uac

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_CONFIGS = REPO_ROOT / "configs"
REPO_ICON = REPO_ROOT / "assets" / "arkcontroller.ico"


def _copy_default_configs(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("default_bindings.json", "stronghold_layout_16_9.json"):
        shutil.copy2(REPO_CONFIGS / name, target_dir / name)


class TestFrozenConfigAndKeys:
    """验证补全的扁平覆盖键集合及 PyInstaller 冻结态双层配置解析。"""

    def test_top_level_override_key_sets_include_all_v100_fields(self) -> None:
        for layout_key in (
            "choice_2_zone",
            "emote_top_zone",
            "emote_bottom_zone",
            "mirrored_board_quad",
            "mirrored_equip_zone",
        ):
            assert layout_key in _LAYOUT_TOP_LEVEL_KEYS
        assert "hide_hud_from_capture" in _KEYMAP_TOP_LEVEL_KEYS

    def test_flat_override_saves_new_keymap_and_layout_keys(self, tmp_path: Path) -> None:
        _copy_default_configs(tmp_path)
        loader = ConfigLoader(config_dir=tmp_path)
        override_file = loader.save_user_override(
            {
                "hide_hud_from_capture": False,
                "choice_2_zone": {"start_v": 0.51, "end_v": 0.51},
                "mirrored_equip_zone": {"start_u": 0.23},
            }
        )
        raw = json.loads(override_file.read_text(encoding="utf-8"))
        assert raw["keymap"]["hide_hud_from_capture"] is False
        assert raw["layout"]["choice_2_zone"]["start_v"] == pytest.approx(0.51)
        assert raw["layout"]["mirrored_equip_zone"]["start_u"] == pytest.approx(0.23)

        keymap, layout = loader.reload()
        assert keymap.hide_hud_from_capture is False
        assert layout.choice_2_zone.start_v == pytest.approx(0.51)
        assert layout.mirrored_equip_zone.start_u == pytest.approx(0.23)

    def test_frozen_standalone_exe_falls_back_to_meipass_and_writes_next_to_exe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """纯单 .exe 拷入空目录时，默认配置回退读取 _MEIPASS，覆盖文件写入 exe 旁 configs/。"""
        exe_dir = tmp_path / "portable_empty"
        exe_dir.mkdir()
        bundle_dir = tmp_path / "_meipass_bundle"
        _copy_default_configs(bundle_dir / "configs")

        fake_exe = exe_dir / "ArkController.exe"
        fake_exe.write_bytes(b"")

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(bundle_dir), raising=False)
        monkeypatch.setattr(sys, "executable", str(fake_exe))

        loader = ConfigLoader()
        assert loader.config_dir == exe_dir / "configs"
        assert not loader.config_dir.exists()
        assert loader.keymap_path == bundle_dir / "configs" / "default_bindings.json"
        assert loader.layout_path == bundle_dir / "configs" / "stronghold_layout_16_9.json"
        assert loader.override_path == exe_dir / "configs" / "user_override.json"

        keymap, layout = loader.load()
        assert keymap.hide_hud_from_capture is True
        assert layout.board_quad.rows == 4

        # 保存覆盖项应自动创建 exe_dir/configs/user_override.json 并无缝合并内置默认配置重载
        saved = loader.save_user_override({"hide_hud_from_capture": False, "ready_hold_ms": 420})
        assert saved == exe_dir / "configs" / "user_override.json"
        assert saved.is_file()
        reloaded_keymap, _ = loader.reload()
        assert reloaded_keymap.hide_hud_from_capture is False
        assert reloaded_keymap.ready_hold_ms == 420

    def test_frozen_external_configs_take_precedence_over_meipass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """当 exe 旁存在外置 configs/ 时，优先读取外置默认配置而非内置 _MEIPASS。"""
        exe_dir = tmp_path / "dist_dir"
        bundle_dir = tmp_path / "_meipass_bundle"
        _copy_default_configs(exe_dir / "configs")
        _copy_default_configs(bundle_dir / "configs")

        # 修改外置 default_bindings.json 以验证优先级
        ext_bindings = exe_dir / "configs" / "default_bindings.json"
        data = json.loads(ext_bindings.read_text(encoding="utf-8"))
        data["ready_hold_ms"] = 610
        ext_bindings.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        fake_exe = exe_dir / "ArkController.exe"
        fake_exe.write_bytes(b"")

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(bundle_dir), raising=False)
        monkeypatch.setattr(sys, "executable", str(fake_exe))

        loader = ConfigLoader()
        assert loader.keymap_path == ext_bindings
        assert loader.layout_path == exe_dir / "configs" / "stronghold_layout_16_9.json"
        keymap, _ = loader.load()
        assert keymap.ready_hold_ms == 610


class TestFrozenUACAndMainEntry:
    """验证冻结态 UAC 工作目录锁定、无控制台 stdout/stderr=None 保护、模板目录与 dry-run 冒烟。"""

    def test_relaunch_as_admin_sets_exe_parent_as_workdir_when_frozen(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exe_dir = tmp_path / "app_bin"
        exe_dir.mkdir()
        fake_exe = exe_dir / "ArkController.exe"
        fake_exe.write_bytes(b"")

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "executable", str(fake_exe))
        monkeypatch.setattr(sys, "argv", [str(fake_exe), "--minimized"])

        fake_shell32 = mock.MagicMock()
        fake_shell32.ShellExecuteW.return_value = 42

        with (
            mock.patch.object(uac.sys, "platform", "win32"),
            mock.patch.object(uac, "is_admin", return_value=False),
            mock.patch.object(uac.ctypes, "WinDLL", return_value=fake_shell32),
        ):
            assert uac.relaunch_as_admin(["--calibrate"]) is True

        call_args = fake_shell32.ShellExecuteW.call_args.args
        assert call_args[1] == "runas"
        assert call_args[2] == str(fake_exe)
        assert call_args[3].split() == ["--minimized", "--calibrate"]
        assert "arkcontroller.main" not in call_args[3]
        assert call_args[4] == str(exe_dir.resolve())

    def test_frozen_templates_and_icon_resolution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exe_dir = tmp_path / "portable"
        exe_dir.mkdir()
        bundle_dir = tmp_path / "meipass"
        _copy_default_configs(bundle_dir / "configs")
        (bundle_dir / "assets" / "templates").mkdir(parents=True)
        shutil.copy2(REPO_ICON, bundle_dir / "assets" / "arkcontroller.ico")

        fake_exe = exe_dir / "ArkController.exe"
        fake_exe.write_bytes(b"")

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(bundle_dir), raising=False)
        monkeypatch.setattr(sys, "executable", str(fake_exe))

        loader = ConfigLoader()
        # 外置 templates 不存在时：读取回退 _MEIPASS，校准写入始终指向 exe_dir/assets/templates
        assert app_main._resolve_templates_dir(loader) == bundle_dir / "assets" / "templates"
        assert app_main._resolve_writable_templates_dir(loader) == exe_dir / "assets" / "templates"
        assert app_main._resolve_app_icon_path() == bundle_dir / "assets" / "arkcontroller.ico"

        # 创建外置 templates 与图标后立即切换为外置优先
        (exe_dir / "assets" / "templates").mkdir(parents=True)
        shutil.copy2(REPO_ICON, exe_dir / "assets" / "arkcontroller.ico")
        assert app_main._resolve_templates_dir(loader) == exe_dir / "assets" / "templates"
        assert app_main._resolve_app_icon_path() == exe_dir / "assets" / "arkcontroller.ico"

    def test_frozen_windowed_main_dry_run_with_none_stdio(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """模拟 PyInstaller console=False（stdout/stderr=None）+ 单 exe 空目录完整运行 dry-run。"""
        exe_dir = tmp_path / "solo_exe"
        exe_dir.mkdir()
        bundle_dir = tmp_path / "meipass"
        _copy_default_configs(bundle_dir / "configs")
        (bundle_dir / "assets").mkdir(parents=True)
        shutil.copy2(REPO_ICON, bundle_dir / "assets" / "arkcontroller.ico")

        fake_exe = exe_dir / "ArkController.exe"
        fake_exe.write_bytes(b"")

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(bundle_dir), raising=False)
        monkeypatch.setattr(sys, "executable", str(fake_exe))
        monkeypatch.setattr(sys, "stdout", None)
        monkeypatch.setattr(sys, "stderr", None)

        exit_code = app_main.main(["--no-elevate", "--dry-run"])
        assert exit_code == 0
        devnull_out = sys.stdout
        devnull_err = sys.stderr
        assert devnull_out is not None
        assert devnull_err is not None
        if getattr(devnull_out, "name", "") == os.devnull:
            devnull_out.close()
        if getattr(devnull_err, "name", "") == os.devnull:
            devnull_err.close()

        qt_app = QApplication.instance()
        assert isinstance(qt_app, QApplication)
        assert not qt_app.windowIcon().isNull()

    def test_packaged_ico_contains_standard_multi_sizes_and_version_is_100(self) -> None:
        assert APP_VERSION == "1.0.0"
        assert REPO_ICON.is_file()
        raw = REPO_ICON.read_bytes()
        reserved, icon_type, count = struct.unpack("<HHH", raw[:6])
        assert reserved == 0
        assert icon_type == 1
        assert count == 6
        sizes = [(raw[6 + i * 16] or 256, raw[7 + i * 16] or 256) for i in range(count)]
        assert sizes == [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

        app = QApplication.instance() or QApplication(["arkcontroller-ico-test"])
        assert app is not None
        icon = QIcon(str(REPO_ICON))
        assert not icon.isNull()
