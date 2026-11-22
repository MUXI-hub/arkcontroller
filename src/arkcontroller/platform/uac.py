"""Windows DPI 感知初始化与 UAC 管理员权限检测/提权封装。"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
import sys

DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
PROCESS_PER_MONITOR_DPI_AWARE = 2
ERROR_ACCESS_DENIED = 5
E_ACCESSDENIED_HRESULT = -2147024891  # 0x80070005
SW_SHOWNORMAL = 1


def enable_per_monitor_dpi_awareness() -> bool:
    """启用 Windows Per-Monitor DPI V2 感知，保证物理像素 1:1 映射。

    优先调用 SetProcessDpiAwarenessContext(-4)，在旧版本 Windows 或受限环境下
    依次回退到 Shcore.SetProcessDpiAwareness(2) 与 User32.SetProcessDPIAware()。
    """
    if sys.platform != "win32":
        return False

    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        set_ctx = getattr(user32, "SetProcessDpiAwarenessContext", None)
        if set_ctx is not None:
            set_ctx.argtypes = [ctypes.c_void_p]
            set_ctx.restype = ctypes.c_int
            if set_ctx(ctypes.c_void_p(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)):
                return True
            # 若进程先前已设置过 DPI 感知（如 Qt 提前初始化），视为开启成功
            if ctypes.get_last_error() == ERROR_ACCESS_DENIED:
                return True
    except (OSError, AttributeError, ValueError):
        pass

    try:
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        set_aware = getattr(shcore, "SetProcessDpiAwareness", None)
        if set_aware is not None:
            set_aware.argtypes = [ctypes.c_int]
            set_aware.restype = ctypes.c_long
            hr = int(set_aware(PROCESS_PER_MONITOR_DPI_AWARE))
            if hr == 0 or hr == E_ACCESSDENIED_HRESULT:
                return True
    except (OSError, AttributeError, ValueError):
        pass

    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        legacy = getattr(user32, "SetProcessDPIAware", None)
        if legacy is not None:
            legacy.argtypes = []
            legacy.restype = ctypes.c_int
            return bool(legacy())
    except (OSError, AttributeError, ValueError):
        pass

    return False


def is_admin() -> bool:
    """检测当前进程是否具备 Windows 管理员权限（高完整性级别）。"""
    if sys.platform != "win32":
        return False
    try:
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        is_user_admin = getattr(shell32, "IsUserAnAdmin", None)
        if is_user_admin is None:
            return False
        is_user_admin.argtypes = []
        is_user_admin.restype = ctypes.c_int
        return bool(is_user_admin())
    except (OSError, AttributeError):
        return False


def _build_relaunch_args(extra_args: list[str] | None = None) -> list[str]:
    """构建提权重启时的命令行参数列表。"""
    args: list[str] = []
    if getattr(sys, "frozen", False):
        args.extend(sys.argv[1:])
    else:
        entry = sys.argv[0] if sys.argv else ""
        if entry:
            abs_entry = os.path.abspath(entry)
            if os.path.isfile(abs_entry):
                args.append(abs_entry)
            else:
                args.extend(["-m", "arkcontroller.main"])
        else:
            args.extend(["-m", "arkcontroller.main"])
        args.extend(sys.argv[1:])

    if extra_args:
        for item in extra_args:
            if item not in args:
                args.append(item)
    return args


def relaunch_as_admin(extra_args: list[str] | None = None) -> bool:
    """通过 Windows UAC (ShellExecuteW runas) 请求管理员权限重启当前程序。

    返回值:
        True: 当前已是管理员，或已成功拉起提权后的新进程。
        False: 非 Windows 环境、用户拒绝 UAC 授权或提权启动失败。
    """
    if sys.platform != "win32":
        return False
    if is_admin():
        return True

    args = _build_relaunch_args(extra_args)
    params = subprocess.list2cmdline(args)
    executable = sys.executable
    work_dir = (
        str(Path(sys.executable).resolve().parent)
        if getattr(sys, "frozen", False)
        else os.getcwd()
    )

    try:
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        shell_exec = shell32.ShellExecuteW
        shell_exec.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_int,
        ]
        shell_exec.restype = ctypes.c_void_p
        result = shell_exec(
            None,
            "runas",
            executable,
            params,
            work_dir,
            SW_SHOWNORMAL,
        )
        # Win32 ShellExecuteW 返回值 > 32 表示调用成功
        return int(result or 0) > 32
    except (OSError, AttributeError, ValueError):
        return False
