"""ArkController Windows 独立单文件可执行程序（ArkController.exe）构建脚本。

功能概述：
1. 使用 PySide6 离屏高分辨率矢量绘制罗德岛 PRTS 纯白极简 + 源石琥珀几何标识，
   生成标准多尺寸（16/32/48/64/128/256）Windows 图标 assets/arkcontroller.ico；
2. 自动读取 arkcontroller.__version__ 生成 PyInstaller Windows 版本资源文件；
3. 调用 PyInstaller 构建无黑框单文件 GUI 程序 dist/ArkController.exe，剔除冗余
   PySide6 大型扩展模块（WebEngine/Qml/Quick/3D/Multimedia/Pdf/Sql/Network 等）以精简体积；
4. 构建完成后自动同步一份干净的默认配置至 dist/configs/（严禁复制本机 user_override.json）。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
CONFIGS_DIR = REPO_ROOT / "configs"
ASSETS_DIR = REPO_ROOT / "assets"
BUILD_DIR = REPO_ROOT / "build"
DIST_DIR = REPO_ROOT / "dist"
ICON_PATH = ASSETS_DIR / "arkcontroller.ico"

# 需要随 exe 同步发布的默认配置文件（严格排除本机 user_override.json）
CLEAN_DEFAULT_CONFIGS: tuple[str, ...] = (
    "default_bindings.json",
    "stronghold_layout_16_9.json",
)

# 瘦身剔除清单：移除未使用的 PySide6 扩展模块与测试框架以大幅缩小 exe 体积并加快解压启动
EXCLUDED_MODULES: tuple[str, ...] = (
    "PySide6.QtWebEngine",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebChannel",
    "PySide6.QtQml",
    "PySide6.QtQmlCore",
    "PySide6.QtQmlModels",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickControls2",
    "PySide6.QtQuickWidgets",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DExtras",
    "PySide6.Qt3DAnimation",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtNetwork",
    "PySide6.QtNetworkAuth",
    "PySide6.QtWebSockets",
    "PySide6.QtSql",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtTest",
    "PySide6.QtDesigner",
    "PySide6.QtUiTools",
    "PySide6.QtHelp",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtGraphs",
    "PySide6.QtGraphsWidgets",
    "PySide6.QtPositioning",
    "PySide6.QtLocation",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtStateMachine",
    "PySide6.QtSpatialAudio",
    "PySide6.QtTextToSpeech",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    "PySide6.QtPrintSupport",
    "PySide6.QtXml",
    "PySide6.QtDBus",
    "PySide6.QtHttpServer",
    "PySide6.QtConcurrent",
    "tkinter",
    "_tkinter",
    "pytest",
    "_pytest",
    "coverage",
    "pydoc",
    "doctest",
    "matplotlib",
    "scipy",
    "PIL",
)


def read_app_version() -> str:
    """从 src/arkcontroller/__init__.py 读取语义化版本号。"""
    init_file = SRC_DIR / "arkcontroller" / "__init__.py"
    text = init_file.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise RuntimeError(f"无法在 {init_file} 中解析 __version__")
    return match.group(1)


def parse_version_quad(version_str: str) -> tuple[int, int, int, int]:
    """将语义化版本号转为 Windows 四段式整数元组 (major, minor, patch, build)。"""
    nums = [int(part) for part in re.findall(r"\d+", version_str)[:4]]
    while len(nums) < 4:
        nums.append(0)
    return (nums[0], nums[1], nums[2], nums[3])


def _render_icon_png(size: int) -> bytes:
    """使用 PySide6 QPainter 离屏绘制单帧罗德岛 PRTS 纯白极简 + 源石琥珀几何图标 PNG。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QBuffer, QIODevice, QPointF, Qt
    from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPainterPath, QPen, QPolygonF

    if QGuiApplication.instance() is None:
        QGuiApplication(["arkcontroller-icon-gen"])

    img = QImage(size, size, QImage.Format.Format_RGBA8888)
    img.fill(Qt.GlobalColor.transparent)

    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    s = float(size)
    margin = max(1.0, s * 0.055)
    cut = s * 0.16
    x0, y0 = margin, margin
    x1, y1 = s - margin, s - margin

    # 1. 瓷白切角底板 + 炭黑细边框（右下角切面体现工业终端质感）
    plate = QPainterPath()
    plate.moveTo(x0 + s * 0.08, y0)
    plate.lineTo(x1 - s * 0.08, y0)
    plate.quadTo(x1, y0, x1, y0 + s * 0.08)
    plate.lineTo(x1, y1 - cut)
    plate.lineTo(x1 - cut, y1)
    plate.lineTo(x0 + s * 0.08, y1)
    plate.quadTo(x0, y1, x0, y1 - s * 0.08)
    plate.lineTo(x0, y0 + s * 0.08)
    plate.quadTo(x0, y0, x0 + s * 0.08, y0)
    plate.closeSubpath()

    painter.setBrush(QColor("#FAFAFA"))
    border_w = max(1.1, s * 0.042)
    painter.setPen(QPen(QColor("#18181B"), border_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap, Qt.PenJoinStyle.MiterJoin))
    painter.drawPath(plate)

    # 2. 三角 PRTS 取景折角（使用连续路径避免直角缺口，仅 >= 32px 渲染）
    if size >= 32:
        inset = s * 0.165
        arm = s * 0.13
        pen_c = QPen(
            QColor("#27272A"),
            max(1.5, s * 0.042),
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.FlatCap,
            Qt.PenJoinStyle.MiterJoin,
        )
        painter.setPen(pen_c)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for pts in (
            # 左上
            (QPointF(inset, inset + arm), QPointF(inset, inset), QPointF(inset + arm, inset)),
            # 右上
            (
                QPointF(s - inset - arm, inset),
                QPointF(s - inset, inset),
                QPointF(s - inset, inset + arm),
            ),
            # 左下
            (
                QPointF(inset, s - inset - arm),
                QPointF(inset, s - inset),
                QPointF(inset + arm, s - inset),
            ),
        ):
            corner = QPainterPath()
            corner.moveTo(pts[0])
            corner.lineTo(pts[1])
            corner.lineTo(pts[2])
            painter.drawPath(corner)

    # 3. 中央炭黑战术菱形外框 + 内部源石琥珀实心菱核（16px~256px 全尺寸高辨识）
    cx, cy = s * 0.50, s * 0.50
    outer_r = s * 0.285
    outer_diamond = QPolygonF(
        [
            QPointF(cx, cy - outer_r),
            QPointF(cx + outer_r, cy),
            QPointF(cx, cy + outer_r),
            QPointF(cx - outer_r, cy),
        ]
    )
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(
        QPen(
            QColor("#18181B"),
            max(1.4, s * 0.054),
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.SquareCap,
            Qt.PenJoinStyle.MiterJoin,
        )
    )
    painter.drawPolygon(outer_diamond)

    # 四向极简刻度线（呼应定朝向准星）
    if size >= 32:
        tick_inner = outer_r + s * 0.036
        tick_outer = outer_r + s * 0.095
        painter.setPen(
            QPen(
                QColor("#18181B"),
                max(1.2, s * 0.034),
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.FlatCap,
            )
        )
        painter.drawLine(QPointF(cx, cy - tick_inner), QPointF(cx, cy - tick_outer))
        painter.drawLine(QPointF(cx, cy + tick_inner), QPointF(cx, cy + tick_outer))
        painter.drawLine(QPointF(cx - tick_inner, cy), QPointF(cx - tick_outer, cy))
        painter.drawLine(QPointF(cx + tick_inner, cy), QPointF(cx + tick_outer, cy))

    # 源石琥珀实心菱核
    inner_r = s * 0.15
    inner_diamond = QPolygonF(
        [
            QPointF(cx, cy - inner_r),
            QPointF(cx + inner_r, cy),
            QPointF(cx, cy + inner_r),
            QPointF(cx - inner_r, cy),
        ]
    )
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#F59E0B"))
    painter.drawPolygon(inner_diamond)

    # 右下切角内侧源石琥珀标识角块（紧贴内切斜边，绝不溢出外框）
    if size >= 32:
        pad_in = border_w * 0.95 + s * 0.025
        tri_sz = s * 0.105
        # 以右下切面中点偏内为斜边绘制精致小三角
        ax = x1 - pad_in
        ay = y1 - cut - pad_in * 0.35
        bx = x1 - cut - pad_in * 0.35
        by = y1 - pad_in
        tri = QPolygonF(
            [
                QPointF(ax, ay - tri_sz),
                QPointF(ax, ay),
                QPointF(bx, by),
                QPointF(bx - tri_sz, by),
            ]
        )
        # 改为与切面平行的高级琥珀斜条
        painter.setBrush(QColor("#F59E0B"))
        painter.drawPolygon(tri)

    painter.end()

    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    ok = img.save(buf, "PNG")
    if not ok:
        raise RuntimeError(f"渲染 {size}x{size} 图标 PNG 失败")
    return bytes(buf.data())


def generate_app_icon(output_path: Path = ICON_PATH) -> Path:
    """生成标准多尺寸（16/32/48/64/128/256）Windows ICO 图标文件。"""
    sizes = (16, 32, 48, 64, 128, 256)
    frames = [(sz, _render_icon_png(sz)) for sz in sizes]

    # ICONDIR Header: reserved(2B)=0, type(2B)=1(ICO), count(2B)
    header = struct.pack("<HHH", 0, 1, len(frames))
    entries: list[bytes] = []
    offset = 6 + 16 * len(frames)
    for sz, png_bytes in frames:
        dim = 0 if sz >= 256 else sz
        entry = struct.pack(
            "<BBBBHHII",
            dim,            # bWidth
            dim,            # bHeight
            0,              # bColorCount
            0,              # bReserved
            1,              # wPlanes
            32,             # wBitCount
            len(png_bytes), # dwBytesInRes
            offset,         # dwImageOffset
        )
        entries.append(entry)
        offset += len(png_bytes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fp:
        fp.write(header)
        for entry in entries:
            fp.write(entry)
        for _, png_bytes in frames:
            fp.write(png_bytes)
    return output_path


def write_version_info_file(version_str: str, target_path: Path) -> Path:
    """生成 PyInstaller Windows 版本信息描述文件（VSVersionInfo）。"""
    quad = parse_version_quad(version_str)
    ver_dot = ".".join(str(n) for n in quad)
    content = f"""# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={quad},
    prodvers={quad},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          '080404B0',
          [
            StringStruct('CompanyName', 'ArkController Project'),
            StringStruct('FileDescription', 'ArkController - 明日方舟卫戍协议手柄操控系统'),
            StringStruct('FileVersion', '{ver_dot}'),
            StringStruct('InternalName', 'ArkController'),
            StringStruct('LegalCopyright', 'ArkController Contributors'),
            StringStruct('OriginalFilename', 'ArkController.exe'),
            StringStruct('ProductName', 'ArkController'),
            StringStruct('ProductVersion', '{version_str}'),
          ]
        )
      ]
    ),
    VarFileInfo([VarStruct('Translation', [2052, 1200])])
  ]
)
"""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")
    return target_path


def write_entry_wrapper(target_path: Path) -> Path:
    """在 build/ 目录生成轻量启动入口，确保 arkcontroller 作为标准包导入。"""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        '"""PyInstaller 单文件构建启动包装入口。"""\n\n'
        "from arkcontroller.main import main\n\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n",
        encoding="utf-8",
    )
    return target_path


def sync_clean_dist_configs() -> Path:
    """将干净的默认 JSON 配置文件同步至 dist/configs/（严禁复制 user_override.json）。"""
    dist_configs = DIST_DIR / "configs"
    dist_configs.mkdir(parents=True, exist_ok=True)

    # 若历史构建残留 user_override.json 则移除，保证分发目录纯净
    stale_override = dist_configs / "user_override.json"
    if stale_override.exists():
        stale_override.unlink()

    for filename in CLEAN_DEFAULT_CONFIGS:
        src = CONFIGS_DIR / filename
        if not src.is_file():
            raise FileNotFoundError(f"缺少必需的默认配置文件: {src}")
        shutil.copy2(src, dist_configs / filename)
    return dist_configs


def build_executable(*, regen_icon: bool = False) -> Path:
    """执行 PyInstaller 单文件无控制台 GUI 打包并同步干净默认配置目录。"""
    version_str = read_app_version()
    if regen_icon or not ICON_PATH.is_file():
        print(f"[build] 正在生成多尺寸应用图标: {ICON_PATH}")
        generate_app_icon(ICON_PATH)

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    version_file = write_version_info_file(version_str, BUILD_DIR / "version_info.txt")
    entry_script = write_entry_wrapper(BUILD_DIR / "entry_main.py")
    work_dir = BUILD_DIR / "pyinstaller_work"

    sep = ";" if os.name == "nt" else ":"
    cmd: list[str] = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name",
        "ArkController",
        "--icon",
        str(ICON_PATH),
        "--version-file",
        str(version_file),
        "--specpath",
        str(BUILD_DIR),
        "--workpath",
        str(work_dir),
        "--distpath",
        str(DIST_DIR),
        "--paths",
        str(SRC_DIR),
        "--add-data",
        f"{CONFIGS_DIR / 'default_bindings.json'}{sep}configs",
        "--add-data",
        f"{CONFIGS_DIR / 'stronghold_layout_16_9.json'}{sep}configs",
        "--add-data",
        f"{ICON_PATH}{sep}assets",
        "--hidden-import",
        "arkcontroller",
        "--hidden-import",
        "arkcontroller.main",
        "--hidden-import",
        "PySide6.QtCore",
        "--hidden-import",
        "PySide6.QtGui",
        "--hidden-import",
        "PySide6.QtWidgets",
        "--hidden-import",
        "mss",
        "--hidden-import",
        "cv2",
        "--hidden-import",
        "numpy",
    ]
    for mod in EXCLUDED_MODULES:
        cmd.extend(["--exclude-module", mod])
    cmd.append(str(entry_script))

    print(f"[build] 开始构建 ArkController v{version_str} -> {DIST_DIR / 'ArkController.exe'}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

    exe_path = DIST_DIR / "ArkController.exe"
    if not exe_path.is_file():
        raise FileNotFoundError(f"PyInstaller 构建完成但未找到目标产物: {exe_path}")

    dist_configs = sync_clean_dist_configs()
    size_mb = exe_path.stat().st_size / (1024 * 1024)
    print(f"[build] 构建成功: {exe_path} ({size_mb:.2f} MB)")
    print(f"[build] 已同步干净默认配置至: {dist_configs}")
    return exe_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 ArkController Windows 独立可执行程序 (.exe)")
    parser.add_argument(
        "--icon-only",
        action="store_true",
        help="仅重新生成多尺寸 assets/arkcontroller.ico，不运行 PyInstaller",
    )
    parser.add_argument(
        "--regen-icon",
        action="store_true",
        help="打包前强制重新生成 assets/arkcontroller.ico",
    )
    args = parser.parse_args(argv)

    if args.icon_only:
        icon_file = generate_app_icon(ICON_PATH)
        print(f"[build] 多尺寸图标已生成: {icon_file} ({icon_file.stat().st_size} bytes)")
        return 0

    build_executable(regen_icon=args.regen_icon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
