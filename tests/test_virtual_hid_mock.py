"""Win32 SendInput 虚拟键鼠合成层 Mock 单元测试。"""

from __future__ import annotations

import ctypes
from unittest.mock import patch

import pytest

from arkcontroller.input.virtual_hid import (
    HARDWAREINPUT,
    INPUT,
    INPUT_KEYBOARD,
    INPUT_MOUSE,
    KEYBDINPUT,
    KEYEVENTF_EXTENDEDKEY,
    KEYEVENTF_KEYUP,
    KEYEVENTF_SCANCODE,
    MAPVK_VK_TO_VSC,
    MOUSEEVENTF_ABSOLUTE,
    MOUSEEVENTF_HWHEEL,
    MOUSEEVENTF_LEFTDOWN,
    MOUSEEVENTF_LEFTUP,
    MOUSEEVENTF_MIDDLEDOWN,
    MOUSEEVENTF_MIDDLEUP,
    MOUSEEVENTF_MOVE,
    MOUSEEVENTF_RIGHTDOWN,
    MOUSEEVENTF_RIGHTUP,
    MOUSEEVENTF_VIRTUALDESK,
    MOUSEEVENTF_WHEEL,
    MOUSEINPUT,
    SM_CXVIRTUALSCREEN,
    SM_CYVIRTUALSCREEN,
    SM_XVIRTUALSCREEN,
    SM_YVIRTUALSCREEN,
    VirtualHID,
)


class _FakeUser32:
    """模拟 Win32 user32 接口，完整记录所有 SendInput 数据包。"""

    def __init__(
        self,
        vx: int = 0,
        vy: int = 0,
        vw: int = 1920,
        vh: int = 1080,
        scancode_offset: int = 0x10,
    ) -> None:
        self.metrics: dict[int, int] = {
            SM_XVIRTUALSCREEN: vx,
            SM_YVIRTUALSCREEN: vy,
            SM_CXVIRTUALSCREEN: vw,
            SM_CYVIRTUALSCREEN: vh,
        }
        self.scancode_offset = scancode_offset
        self.return_zero_scancode = False
        self.sent_packets: list[INPUT] = []
        self.cb_sizes: list[int] = []
        self.mapped_vks: list[tuple[int, int]] = []

    def GetSystemMetrics(self, index: int) -> int:
        return self.metrics[index]

    def MapVirtualKeyW(self, vk: int, map_type: int) -> int:
        self.mapped_vks.append((vk, map_type))
        if self.return_zero_scancode:
            return 0
        # 返回非零、可区分的确定性扫描码
        return (vk & 0x7F) + self.scancode_offset

    def SendInput(self, count: int, inputs_array: object, cb_size: int) -> int:
        self.cb_sizes.append(cb_size)
        for idx in range(count):
            # 拷贝保存结构体快照，防止外部数组释放
            cloned = INPUT()
            ctypes.memmove(ctypes.byref(cloned), ctypes.byref(inputs_array[idx]), cb_size)  # type: ignore[index]
            self.sent_packets.append(cloned)
        return count


def _make_hid(
    vx: int = 0,
    vy: int = 0,
    vw: int = 1920,
    vh: int = 1080,
) -> tuple[VirtualHID, _FakeUser32, list[float]]:
    fake = _FakeUser32(vx=vx, vy=vy, vw=vw, vh=vh)
    sleeps: list[float] = []
    hid = VirtualHID(user32=fake, sleep_fn=sleeps.append)
    return hid, fake, sleeps


class TestWin32InputStructAlignment:
    """严格校验 64 位 Windows 下 SendInput 结构体内存对齐与字节大小。"""

    def test_input_struct_size_matches_win64_abi(self) -> None:
        assert ctypes.sizeof(HARDWAREINPUT) == 8
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            assert ctypes.sizeof(MOUSEINPUT) == 32
            assert ctypes.sizeof(KEYBDINPUT) == 24
            assert ctypes.sizeof(INPUT) == 40
        else:
            assert ctypes.sizeof(INPUT) == 28


class TestVirtualDesktopCoordinateNormalization:
    """校验单屏与多显示器（含负坐标左副屏）虚拟桌面 [0, 65535] 映射。"""

    def test_single_monitor_1080p_endpoints_and_center(self) -> None:
        hid, _, _ = _make_hid(vx=0, vy=0, vw=1920, vh=1080)
        assert hid.screen_to_absolute(0, 0) == (0, 0)
        assert hid.screen_to_absolute(1919, 1079) == (65535, 65535)
        abs_x, abs_y = hid.screen_to_absolute(960, 540)
        assert abs(abs_x - 32785) <= 1
        assert abs(abs_y - 32798) <= 1

    def test_multi_monitor_negative_origin_and_clamping(self) -> None:
        # 左侧扩展屏 [-1920, -1]，主屏 [0, 1919]，总宽 3840
        hid, _, _ = _make_hid(vx=-1920, vy=-200, vw=3840, vh=1280)
        assert hid.screen_to_absolute(-1920, -200) == (0, 0)
        assert hid.screen_to_absolute(1919, 1079) == (65535, 65535)
        # 超出虚拟桌面边界时应安全钳制在 [0, 65535]
        assert hid.screen_to_absolute(-5000, -5000) == (0, 0)
        assert hid.screen_to_absolute(99999, 99999) == (65535, 65535)


class TestMouseOperations:
    """校验移动、点击、双击、平滑拖拽、四向滑动与滚轮事件包构造。"""

    def test_move_cursor_and_click_sequence(self) -> None:
        hid, fake, sleeps = _make_hid(0, 0, 1920, 1080)
        hid.click(1919, 1079, button="left", hold_ms=35)

        assert len(fake.sent_packets) == 3
        assert all(sz == ctypes.sizeof(INPUT) for sz in fake.cb_sizes)

        move_pkt, down_pkt, up_pkt = fake.sent_packets
        assert move_pkt.type == INPUT_MOUSE
        assert move_pkt.mi.dx == 65535
        assert move_pkt.mi.dy == 65535
        assert move_pkt.mi.dwFlags == (
            MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        )
        assert down_pkt.mi.dwFlags == MOUSEEVENTF_LEFTDOWN
        assert up_pkt.mi.dwFlags == MOUSEEVENTF_LEFTUP
        assert sleeps == [pytest.approx(0.035)]

    def test_right_and_middle_click_and_invalid_button(self) -> None:
        hid, fake, _ = _make_hid()
        hid.click(100, 100, button="right", hold_ms=0)
        hid.click(200, 200, button="middle", hold_ms=0)
        flags = [pkt.mi.dwFlags for pkt in fake.sent_packets]
        assert MOUSEEVENTF_RIGHTDOWN in flags
        assert MOUSEEVENTF_RIGHTUP in flags
        assert MOUSEEVENTF_MIDDLEDOWN in flags
        assert MOUSEEVENTF_MIDDLEUP in flags

        with pytest.raises(ValueError):
            hid.mouse_down("side_extra")
        with pytest.raises(ValueError):
            hid.mouse_up("side_extra")

    def test_double_click_interval_and_event_count(self) -> None:
        hid, fake, sleeps = _make_hid()
        hid.double_click(500, 400, interval_ms=55)

        # 每次 click 产生 move + down + up 共 3 包，双击共 6 包
        assert len(fake.sent_packets) == 6
        assert sleeps == [
            pytest.approx(0.025),
            pytest.approx(0.055),
            pytest.approx(0.025),
        ]

    def test_smooth_drag_steps_and_optional_release(self) -> None:
        hid, fake, sleeps = _make_hid(0, 0, 1001, 1001)
        hid.smooth_drag(
            start=(0, 0),
            end=(1000, 500),
            duration_ms=100,
            steps=5,
            release_at_end=True,
        )

        # 序列：move(start) + left_down + 5 次插值 move + left_up = 8 包
        assert len(fake.sent_packets) == 8
        assert fake.sent_packets[0].mi.dx == 0
        assert fake.sent_packets[0].mi.dy == 0
        assert fake.sent_packets[1].mi.dwFlags == MOUSEEVENTF_LEFTDOWN

        step_xs = [pkt.mi.dx for pkt in fake.sent_packets[2:7]]
        step_ys = [pkt.mi.dy for pkt in fake.sent_packets[2:7]]
        # 插值点单调递增且最后一步精确落到 (1000, 500) -> (65535, 32768)
        assert step_xs == sorted(step_xs)
        assert step_ys == sorted(step_ys)
        assert step_xs[-1] == 65535
        assert step_ys[-1] == round(500 * 65535 / 1000)
        assert fake.sent_packets[-1].mi.dwFlags == MOUSEEVENTF_LEFTUP
        assert len(sleeps) == 5
        assert sum(sleeps) == pytest.approx(0.100)

        # 测试 release_at_end=False 不自动抬起，后续可手动 mouse_up
        fake.sent_packets.clear()
        hid.smooth_drag((10, 10), (100, 100), duration_ms=40, steps=4, release_at_end=False)
        assert fake.sent_packets[-1].mi.dwFlags == (
            MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        )
        hid.mouse_up("left")
        assert fake.sent_packets[-1].mi.dwFlags == MOUSEEVENTF_LEFTUP

    @pytest.mark.parametrize(
        ("direction", "expected_end"),
        [
            ("up", (500, 400)),
            ("down", (500, 600)),
            ("left", (400, 500)),
            ("right", (600, 500)),
        ],
    )
    def test_directional_flick_all_four_directions(
        self,
        direction: str,
        expected_end: tuple[int, int],
    ) -> None:
        hid, fake, _ = _make_hid(0, 0, 1001, 1001)
        hid.directional_flick((500, 500), direction, distance_px=100)  # type: ignore[arg-type]

        # 最后一个包为抬起，倒数第二个包为终点移动
        assert fake.sent_packets[-1].mi.dwFlags == MOUSEEVENTF_LEFTUP
        end_pkt = fake.sent_packets[-2]
        exp_abs_x, exp_abs_y = hid.screen_to_absolute(*expected_end)
        assert end_pkt.mi.dx == exp_abs_x
        assert end_pkt.mi.dy == exp_abs_y

    def test_directional_flick_invalid_direction_raises(self) -> None:
        hid, _, _ = _make_hid()
        with pytest.raises(ValueError):
            hid.directional_flick((100, 100), "diagonal", 50)  # type: ignore[arg-type]

    def test_scroll_vertical_horizontal_and_zero_noop(self) -> None:
        hid, fake, _ = _make_hid()

        hid.scroll(0)
        assert fake.sent_packets == []

        hid.scroll(120, horizontal=False)
        hid.scroll(-120, horizontal=False)
        hid.scroll(-240, horizontal=True)

        assert len(fake.sent_packets) == 3
        p_up, p_down, p_horiz = fake.sent_packets
        assert p_up.mi.dwFlags == MOUSEEVENTF_WHEEL
        assert p_up.mi.mouseData == 120
        assert p_down.mi.dwFlags == MOUSEEVENTF_WHEEL
        assert p_down.mi.mouseData == ((-120) & 0xFFFFFFFF)
        assert p_horiz.mi.dwFlags == MOUSEEVENTF_HWHEEL
        assert p_horiz.mi.mouseData == ((-240) & 0xFFFFFFFF)


class TestKeyboardOperationsAndWindllPatch:
    """校验全部官方快捷键扫描码发送、回退机制及 ctypes.windll.user32 直接 Patch。"""

    @pytest.mark.parametrize(
        "key_name",
        [
            "space",
            "d",
            "f",
            "r",
            "e",
            "q",
            "a",
            "z",
            "tab",
            "grave",
            "esc",
            "1",
            "2",
            "3",
            "4",
        ],
    )
    def test_tap_key_covers_all_required_shortcuts_with_scancode(
        self,
        key_name: str,
    ) -> None:
        hid, fake, sleeps = _make_hid()
        hid.tap_key(key_name, hold_ms=25)

        assert len(fake.sent_packets) == 2
        down_pkt, up_pkt = fake.sent_packets

        assert down_pkt.type == INPUT_KEYBOARD
        assert down_pkt.ki.wVk == 0
        assert down_pkt.ki.wScan > 0
        assert down_pkt.ki.dwFlags == KEYEVENTF_SCANCODE

        assert up_pkt.type == INPUT_KEYBOARD
        assert up_pkt.ki.wVk == 0
        assert up_pkt.ki.wScan == down_pkt.ki.wScan
        assert up_pkt.ki.dwFlags == (KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP)

        assert all(map_type == MAPVK_VK_TO_VSC for _, map_type in fake.mapped_vks)
        assert sleeps == [pytest.approx(0.025)]

    def test_extended_key_and_zero_scancode_fallback_and_unknown_key(self) -> None:
        hid, fake, _ = _make_hid()
        hid.key_down("right")
        assert fake.sent_packets[0].ki.dwFlags == (
            KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY
        )

        # 若 MapVirtualKeyW 返回 0，应安全回退为 wVk 模式而非发空按键
        fake.sent_packets.clear()
        fake.return_zero_scancode = True
        hid.key_down("space")
        assert fake.sent_packets[0].ki.wVk == 0x20
        assert fake.sent_packets[0].ki.wScan == 0
        assert fake.sent_packets[0].ki.dwFlags == 0

        with pytest.raises(ValueError):
            hid.tap_key("nonexistent_key")

    def test_patch_ctypes_windll_user32_sendinput_directly(self) -> None:
        """验证未显式注入 user32 时，直接 Mock ctypes.windll.user32.SendInput 可正常拦截。"""
        if not hasattr(ctypes, "windll"):
            pytest.skip("仅在 Windows 平台运行 windll Mock 测试")

        recorded: list[tuple[int, int]] = []

        def fake_send_input(count: int, arr: object, cb_size: int) -> int:
            recorded.append((count, cb_size))
            return count

        with (
            patch.object(ctypes.windll.user32, "SendInput", side_effect=fake_send_input),
            patch.object(
                ctypes.windll.user32,
                "GetSystemMetrics",
                side_effect=lambda idx: {
                    SM_XVIRTUALSCREEN: 0,
                    SM_YVIRTUALSCREEN: 0,
                    SM_CXVIRTUALSCREEN: 1920,
                    SM_CYVIRTUALSCREEN: 1080,
                }[idx],
            ),
        ):
            hid = VirtualHID(sleep_fn=lambda _: None)
            hid.move_cursor(960, 540)

        assert recorded == [(1, ctypes.sizeof(INPUT))]
