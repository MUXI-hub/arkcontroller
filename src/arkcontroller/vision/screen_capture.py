"""非侵入式游戏视口屏幕采样器。

基于 ``mss`` 高速截取 ``ViewportRect`` 对应的 16:9 有效区域，当窗口被遮挡或
``mss`` 截取失败时自动回退至 Win32 ``PrintWindow``，最终统一缩放为
``1280x720`` BGR ``np.ndarray`` 供 ``SceneDetector`` 识别。

线程安全约定：
- 后台异步采样线程内部独占 ``mss.mss()`` 实例并在退出时立即释放；
- 同步 ``grab_frame()`` 通过 ``threading.local()`` 为每个调用线程分配独立
  ``mss.mss()`` 实例，避免 Windows 跨线程访问 GDI 设备上下文报错。
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import logging
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

import cv2
import mss
import numpy as np

if TYPE_CHECKING:
    from arkcontroller.platform.window_tracker import ViewportRect

logger = logging.getLogger(__name__)

TARGET_WIDTH: int = 1280
TARGET_HEIGHT: int = 720

_PW_CLIENTONLY = 0x00000001
_PW_RENDERFULLCONTENT = 0x00000002
_BI_RGB = 0
_DIB_RGB_COLORS = 0


class _BITMAPINFOHEADER(ctypes.Structure):
    """Win32 BITMAPINFOHEADER 结构体。"""

    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _BITMAPINFO(ctypes.Structure):
    """Win32 BITMAPINFO 结构体（无调色板真彩色 32bpp）。"""

    _fields_ = [
        ("bmiHeader", _BITMAPINFOHEADER),
        ("bmiColors", wintypes.DWORD * 3),
    ]


def _setup_win32_gdi() -> tuple[Any, Any] | None:
    """初始化带 64 位安全指针签名的 User32/Gdi32 函数句柄。"""
    if sys.platform != "win32":
        return None
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

        user32.GetDC.argtypes = [ctypes.c_void_p]
        user32.GetDC.restype = ctypes.c_void_p

        user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        user32.ReleaseDC.restype = ctypes.c_int

        user32.PrintWindow.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]
        user32.PrintWindow.restype = wintypes.BOOL

        gdi32.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
        gdi32.CreateCompatibleDC.restype = ctypes.c_void_p

        gdi32.DeleteDC.argtypes = [ctypes.c_void_p]
        gdi32.DeleteDC.restype = wintypes.BOOL

        gdi32.CreateCompatibleBitmap.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        gdi32.CreateCompatibleBitmap.restype = ctypes.c_void_p

        gdi32.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        gdi32.SelectObject.restype = ctypes.c_void_p

        gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
        gdi32.DeleteObject.restype = wintypes.BOOL

        gdi32.GetDIBits.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.UINT,
            wintypes.UINT,
            ctypes.c_void_p,
            ctypes.POINTER(_BITMAPINFO),
            wintypes.UINT,
        ]
        gdi32.GetDIBits.restype = ctypes.c_int

        return user32, gdi32
    except OSError:
        logger.warning("加载 Win32 User32/Gdi32 失败，PrintWindow 备选不可用", exc_info=True)
        return None


def _is_non_black_frame(frame: np.ndarray | None) -> bool:
    """判断帧是否有效且非全黑伪帧（排除 Unity DX11 下 PrintWindow 返回全零画面的情况）。"""
    return frame is not None and frame.size > 0 and int(frame.max()) > 0


def _resize_to_standard_bgr(
    bgra_or_bgr: np.ndarray,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
) -> np.ndarray | None:
    """将截取的 BGRA/BGR 图像转换为标准尺寸连续内存 BGR ``uint8`` 矩阵。"""
    if bgra_or_bgr is None or bgra_or_bgr.size == 0 or bgra_or_bgr.ndim != 3:
        return None

    channels = bgra_or_bgr.shape[2]
    if channels == 4:
        bgr = cv2.cvtColor(bgra_or_bgr, cv2.COLOR_BGRA2BGR)
    elif channels == 3:
        bgr = bgra_or_bgr
    else:
        return None

    height, width = bgr.shape[:2]
    if width <= 0 or height <= 0:
        return None

    if width != target_width or height != target_height:
        interp = (
            cv2.INTER_AREA
            if width >= target_width and height >= target_height
            else cv2.INTER_LINEAR
        )
        bgr = cv2.resize(bgr, (target_width, target_height), interpolation=interp)

    return np.ascontiguousarray(bgr, dtype=np.uint8)


class ScreenCapture:
    """非侵入式 16:9 游戏视口屏幕采样器（支持同步抓取与后台守护线程异步采样）。"""

    def __init__(
        self,
        target_size: tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT),
        enable_print_window_fallback: bool = True,
    ) -> None:
        self._target_width: int = max(1, int(target_size[0]))
        self._target_height: int = max(1, int(target_size[1]))
        self._enable_print_window_fallback: bool = bool(enable_print_window_fallback)

        # 线程局部 mss 容器，避免同步跨线程调用共享同一 GDI 句柄
        self._thread_local = threading.local()
        self._win32_gdi: tuple[Any, Any] | None = (
            _setup_win32_gdi() if self._enable_print_window_fallback else None
        )

        # 后台异步采集共享状态（全部受 _state_lock 保护）
        self._state_lock = threading.Lock()
        self._viewport: ViewportRect | None = None
        self._latest_frame: np.ndarray | None = None

        self._thread_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._async_fps: float = 5.0

    @property
    def is_running(self) -> bool:
        """后台异步采样线程是否正在运行。"""
        with self._thread_lock:
            return self._worker_thread is not None and self._worker_thread.is_alive()

    def _get_thread_mss(self) -> Any:
        """获取当前线程独占的 ``mss.mss()`` 实例。"""
        sct = getattr(self._thread_local, "sct", None)
        if sct is None:
            sct = mss.mss()
            self._thread_local.sct = sct
        return sct

    def _close_thread_mss(self) -> None:
        """关闭并释放当前线程持有的 ``mss.mss()`` 实例。"""
        sct = getattr(self._thread_local, "sct", None)
        if sct is not None:
            try:
                sct.close()
            except Exception:
                logger.debug("关闭线程局部 mss 实例时忽略异常", exc_info=True)
            finally:
                self._thread_local.sct = None

    def grab_frame(self, viewport: ViewportRect | None) -> np.ndarray | None:
        """同步截取指定 ``ViewportRect`` 的 16:9 有效区域并缩放为 ``1280x720`` BGR 帧。

        优先使用 ``mss`` 高速截屏；当窗口不在前台被遮挡或 ``mss`` 失败时，
        回退至 Win32 ``PrintWindow``。
        """
        if viewport is None:
            return None
        if viewport.active_width <= 0 or viewport.active_height <= 0:
            return None

        return self._grab_with_sct(viewport, sct=None)

    def _grab_with_sct(
        self, viewport: ViewportRect, sct: Any | None
    ) -> np.ndarray | None:
        """使用指定或线程局部的 ``mss`` 实例执行截图并在必要时回退 ``PrintWindow``。"""
        frame, _ = self._grab_with_sct_checked(viewport, sct=sct, allow_mss=True)
        return frame

    def _grab_with_sct_checked(
        self,
        viewport: ViewportRect,
        sct: Any | None,
        *,
        allow_mss: bool = True,
    ) -> tuple[np.ndarray | None, bool]:
        """执行截图并额外返回 ``mss`` 句柄是否因异常失效需要销毁重建。

        若 ``PrintWindow`` 返回全黑伪帧（``max == 0``）且 ``mss`` 不可用或失败，
        返回 ``None`` 而非全黑伪帧，避免污染下游场景识别器。
        """
        tried_pw = False
        # 当窗口被其他窗口遮挡（非前台）且开启了 PrintWindow 备选时，优先尝试 PrintWindow；
        # 若 Unity/DirectX 窗口 PrintWindow 返回全黑帧，直接舍弃并尝试 mss 通道。
        if not viewport.is_foreground and self._enable_print_window_fallback and viewport.hwnd:
            tried_pw = True
            pw_frame = self._grab_via_print_window(viewport)
            if _is_non_black_frame(pw_frame):
                return (pw_frame, False)

        sct_broken = False
        if allow_mss:
            mss_frame, sct_broken = self._try_grab_via_mss(viewport, sct=sct)
            if mss_frame is not None:
                return (mss_frame, False)

        if not tried_pw and self._enable_print_window_fallback and viewport.hwnd:
            pw_frame = self._grab_via_print_window(viewport)
            if _is_non_black_frame(pw_frame):
                return (pw_frame, sct_broken)

        return (None, sct_broken)

    def _grab_via_mss(
        self, viewport: ViewportRect, sct: Any | None = None
    ) -> np.ndarray | None:
        """使用 ``mss`` 截取桌面指定区域。"""
        frame, _ = self._try_grab_via_mss(viewport, sct=sct)
        return frame

    def _try_grab_via_mss(
        self, viewport: ViewportRect, sct: Any | None = None
    ) -> tuple[np.ndarray | None, bool]:
        """尝试通过 ``mss`` 截屏，返回 ``(frame, sct_broken)``。"""
        monitor = {
            "left": int(viewport.active_left),
            "top": int(viewport.active_top),
            "width": int(viewport.active_width),
            "height": int(viewport.active_height),
        }
        if monitor["width"] <= 0 or monitor["height"] <= 0:
            return (None, False)

        try:
            active_sct = sct if sct is not None else self._get_thread_mss()
            raw = active_sct.grab(monitor)
            arr = np.asarray(raw, dtype=np.uint8)
            frame = _resize_to_standard_bgr(
                arr,
                target_width=self._target_width,
                target_height=self._target_height,
            )
            return (frame, False)
        except Exception:
            logger.debug("mss 截取游戏视口失败，准备销毁失效句柄并尝试备选通道", exc_info=True)
            if sct is None:
                # 同步线程 mss 上下文若损坏则清理以便下次重建
                self._close_thread_mss()
            return (None, True)

    def _grab_via_print_window(self, viewport: ViewportRect) -> np.ndarray | None:
        """使用 Win32 ``PrintWindow`` 非侵入式捕获被遮挡窗口客户区并裁剪 16:9 区域。"""
        if self._win32_gdi is None or viewport.hwnd <= 0:
            return None

        client_w = int(viewport.client_width)
        client_h = int(viewport.client_height)
        if client_w <= 0 or client_h <= 0:
            return None

        user32, gdi32 = self._win32_gdi
        hwnd = ctypes.c_void_p(int(viewport.hwnd))
        src_dc = None
        mem_dc = None
        hbmp = None
        old_obj = None

        try:
            src_dc = user32.GetDC(hwnd)
            if not src_dc:
                return None

            mem_dc = gdi32.CreateCompatibleDC(src_dc)
            if not mem_dc:
                return None

            hbmp = gdi32.CreateCompatibleBitmap(src_dc, client_w, client_h)
            if not hbmp:
                return None

            old_obj = gdi32.SelectObject(mem_dc, hbmp)

            ok = bool(
                user32.PrintWindow(
                    hwnd, mem_dc, _PW_CLIENTONLY | _PW_RENDERFULLCONTENT
                )
            )
            if not ok:
                ok = bool(user32.PrintWindow(hwnd, mem_dc, _PW_CLIENTONLY))
            if not ok:
                return None

            bmi = _BITMAPINFO()
            bmi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
            bmi.bmiHeader.biWidth = client_w
            # 负高度表示自顶向下的 DIB，避免图像上下翻转
            bmi.bmiHeader.biHeight = -client_h
            bmi.bmiHeader.biPlanes = 1
            bmi.bmiHeader.biBitCount = 32
            bmi.bmiHeader.biCompression = _BI_RGB

            buffer = np.empty((client_h, client_w, 4), dtype=np.uint8)
            scanlines = gdi32.GetDIBits(
                mem_dc,
                hbmp,
                0,
                client_h,
                buffer.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(bmi),
                _DIB_RGB_COLORS,
            )
            if scanlines == 0:
                return None

            # 按 active 区域相对 client 区域的偏移裁剪扣除黑边后的 16:9 画面
            off_x = max(0, int(viewport.active_left - viewport.client_left))
            off_y = max(0, int(viewport.active_top - viewport.client_top))
            end_x = min(client_w, off_x + int(viewport.active_width))
            end_y = min(client_h, off_y + int(viewport.active_height))
            if end_x <= off_x or end_y <= off_y:
                return None

            cropped = buffer[off_y:end_y, off_x:end_x, :]
            return _resize_to_standard_bgr(
                cropped,
                target_width=self._target_width,
                target_height=self._target_height,
            )
        except Exception:
            logger.debug("PrintWindow 备选截取失败", exc_info=True)
            return None
        finally:
            if mem_dc and old_obj:
                gdi32.SelectObject(mem_dc, old_obj)
            if hbmp:
                gdi32.DeleteObject(hbmp)
            if mem_dc:
                gdi32.DeleteDC(mem_dc)
            if src_dc:
                user32.ReleaseDC(hwnd, src_dc)

    def start_async(self, fps: float = 5.0) -> None:
        """启动独立后台守护线程以指定帧率持续采样最新画面。"""
        safe_fps = max(0.5, min(60.0, float(fps)))
        with self._thread_lock:
            self._async_fps = safe_fps
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._capture_loop,
                name="ArkController-ScreenCapture",
                daemon=True,
            )
            self._worker_thread.start()

    def stop_async(self, timeout: float = 2.0) -> None:
        """安全停止后台守护采样线程并等待其退出释放资源。"""
        with self._thread_lock:
            thread = self._worker_thread
            if thread is None:
                return
            self._stop_event.set()

        if thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(0.1, float(timeout)))

        with self._thread_lock:
            if self._worker_thread is thread and not thread.is_alive():
                self._worker_thread = None

    def update_viewport(self, viewport: ViewportRect | None) -> None:
        """线程安全地更新当前追踪的游戏视口矩形。"""
        with self._state_lock:
            self._viewport = viewport
            if viewport is None:
                self._latest_frame = None

    def get_latest_frame(self) -> np.ndarray | None:
        """线程安全地获取最新采样到的 ``1280x720`` BGR 帧副本。"""
        with self._state_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def close(self) -> None:
        """停止后台线程并释放当前线程的 ``mss`` 句柄。"""
        self.stop_async()
        self._close_thread_mss()

    def __enter__(self) -> ScreenCapture:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def _capture_loop(self) -> None:
        """后台守护线程主循环，线程内独立持有、异常自愈并释放 ``mss.mss()`` 实例。"""
        sct: Any | None = None
        mss_init_warned = False

        try:
            while not self._stop_event.is_set():
                t0 = time.perf_counter()

                # 惰性初始化与 UAC/桌面切换后自动重试自愈，绝不让 sct 永久停留在 None
                if sct is None:
                    try:
                        sct = mss.mss()
                        if mss_init_warned:
                            logger.info("后台线程 mss 上下文已自动重建恢复")
                            mss_init_warned = False
                    except Exception:
                        sct = None
                        if not mss_init_warned:
                            logger.warning(
                                "后台线程初始化 mss 失败，将在后续周期自动重试恢复",
                                exc_info=True,
                            )
                            mss_init_warned = True
                        else:
                            logger.debug("后台线程重试初始化 mss 仍未就绪", exc_info=True)

                with self._state_lock:
                    viewport = self._viewport

                if viewport is not None and viewport.active_width > 0 and viewport.active_height > 0:
                    frame, sct_broken = self._grab_with_sct_checked(
                        viewport,
                        sct=sct,
                        allow_mss=(sct is not None),
                    )
                    if sct_broken and sct is not None:
                        try:
                            sct.close()
                        except Exception:
                            logger.debug("关闭失效的后台 mss 实例时忽略异常", exc_info=True)
                        finally:
                            sct = None
                        logger.warning("后台线程 mss 抓取异常，已销毁旧句柄并将在下一周期重建")

                    with self._state_lock:
                        # 仅当外部尚未清空 viewport 时才写回最新帧
                        if self._viewport is not None:
                            self._latest_frame = frame

                interval = 1.0 / max(0.5, self._async_fps)
                elapsed = time.perf_counter() - t0
                wait_sec = max(0.005, interval - elapsed)
                self._stop_event.wait(timeout=wait_sec)
        finally:
            if sct is not None:
                try:
                    sct.close()
                except Exception:
                    logger.debug("后台线程释放 mss 实例时忽略异常", exc_info=True)
            self._close_thread_mss()
