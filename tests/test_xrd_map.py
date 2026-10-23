"""The map plan against a mocked Perkin Elmer (conftest ``pe``) and fast shutter."""

from typing import Any

import pytest
from bluesky import plan_stubs as bps
from bluesky.run_engine import RunEngine
from ophyd_async.core import (
    callback_on_mock_put,
    get_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.adcore import ContAcqDetector
from ophyd_async.sim import SimMotor

from pdftools.shutter import FastShutter, OnOff
from pdftools.xrd import xrd_map


@pytest.fixture
def fs(RE: RunEngine) -> FastShutter:
    with init_devices(mock=True):
        fs = FastShutter("FS:", name="fs", settle_time=0)
    set_mock_value(fs.status, OnOff.ON)  # closed
    callback_on_mock_put(fs.cmd, lambda value, **_: set_mock_value(fs.status, value))
    return fs


@pytest.fixture
def stages(RE: RunEngine) -> tuple[SimMotor, SimMotor]:
    with init_devices():
        grid_x = SimMotor(name="grid_x", instant=True)
        grid_y = SimMotor(name="grid_y", instant=True)
    RE(bps.mv(grid_x.velocity, 2.5))
    return grid_x, grid_y


def _run(RE: RunEngine, plan) -> tuple[dict[str, list[Any]], list]:
    docs: dict[str, list[Any]] = {}
    msgs: list = []
    RE.msg_hook = msgs.append  # pyright: ignore[reportAttributeAccessIssue]
    try:
        RE(plan, lambda name, doc: docs.setdefault(name, []).append(doc))
    finally:
        RE.msg_hook = None
    return docs, msgs


def _sets(msgs: list, *objs) -> list[tuple[str, Any]]:
    return [
        (m.obj.name, m.args[0]) for m in msgs if m.command == "set" and m.obj in objs
    ]


def _map(pe, fs, grid_x, grid_y, **kwargs):
    return xrd_map(
        [pe],
        grid_x,
        0.0,
        2.0,
        3,
        grid_y,
        0.0,
        1.0,
        2,
        0.1,
        sample_name="Ni",
        shutter=fs,
        **kwargs,
    )


async def test_xrd_map_snakes_one_run_with_an_event_per_pixel(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter, stages
):
    grid_x, grid_y = stages
    docs, msgs = _run(RE, _map(pe, fs, grid_x, grid_y, backoff=0.5))
    (start,) = docs["start"]
    assert start["plan_name"] == "xrd_map"
    assert start["sample_name"] == "Ni"
    assert start["map_size"] == [3, 2]
    assert start["hints"]["dimensions"] == [
        [["start_grid_x"], "primary"],
        [["grid_y"], "primary"],
    ]
    assert len(docs["event"]) == 6
    assert {"start_grid_x", "stop_grid_x", "grid_y"} <= set(docs["event"][0]["data"])
    # back off before the row, fly past the end; the next row flies back
    assert _sets(msgs, grid_x, grid_y) == [
        ("grid_y", 0.0),
        ("grid_x", -0.5),
        ("grid_x", 2.5),
        ("grid_y", 1.0),
        ("grid_x", 2.5),
        ("grid_x", -0.5),
    ]
    # 2.0 over 3 pixels of 0.1 s each (one 0.1 s frame); original speed otherwise
    speed = pytest.approx(2.0 / 0.3)
    assert [v for _, v in _sets(msgs, grid_x.velocity)] == [
        2.5,
        speed,
        2.5,
        speed,
        2.5,
    ]
    assert await grid_x.velocity.get_value() == 2.5


async def test_xrd_map_typewriter_flies_one_way(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter, stages
):
    grid_x, grid_y = stages
    _, msgs = _run(RE, _map(pe, fs, grid_x, grid_y, snake=False))
    assert [v for _, v in _sets(msgs, grid_x)] == [0.0, 2.0, 0.0, 2.0]


async def test_xrd_map_shutter_closed_except_while_imaging(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter, stages
):
    grid_x, grid_y = stages
    set_mock_value(fs.status, OnOff.OFF)  # left open by an earlier plan
    _, msgs = _run(RE, _map(pe, fs, grid_x, grid_y))
    steps = []
    for m in msgs:
        if m.command == "set" and m.obj in (fs, grid_y):
            steps.append((m.obj.name, m.args[0]))
        elif m.command == "save" and steps[-1] != ("save",):
            steps.append(("save",))
        elif m.command == "wait" and str(m.kwargs.get("group")).startswith("fly"):
            steps.append(("fly done",))
    # closed to reposition, open only while imaging, closed before the overshoot
    row = [("fs", "Open"), ("save",), ("fs", "Close"), ("fly done",)]
    assert steps == [
        ("fs", "Close"),
        ("grid_y", 0.0),
        *row,
        ("fs", "Close"),
        ("grid_y", 1.0),
        *row,
        ("fs", "Close"),
    ]
    assert await fs.status.get_value() == OnOff.ON


async def test_xrd_map_backs_off_away_from_a_descending_row(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter, stages
):
    grid_x, grid_y = stages
    _, msgs = _run(
        RE,
        xrd_map(
            [pe],
            grid_x,
            2.0,
            0.0,
            3,
            grid_y,
            0.0,
            1.0,
            2,
            0.1,
            backoff=0.5,
            shutter=fs,
        ),
    )
    assert [v for _, v in _sets(msgs, grid_x)] == [2.5, -0.5, -0.5, 2.5]


async def test_xrd_map_first_image_starts_at_the_row_start(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter
):
    with init_devices():
        x = SimMotor(name="x")  # moves in real time, 0.5 s to reach speed
        y = SimMotor(name="y", instant=True)
    RE(bps.mv(x.velocity, 10.0))
    docs, _ = _run(
        RE,
        xrd_map([pe], x, 0.0, 1.0, 2, y, 0.0, 0.0, 1, 0.1, backoff=0.5, shutter=fs),
    )
    # without waiting for the row start, the first image starts near -0.5
    assert docs["event"][0]["data"]["start_x"] >= -0.02


async def test_xrd_map_restores_velocity_when_the_shutter_fails_to_close(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter, stages
):
    grid_x, grid_y = stages
    opened: list[bool] = []

    def _stuck_open(value, **_):
        if value == OnOff.ON and opened:
            raise RuntimeError("shutter stuck")
        opened.append(value == OnOff.OFF)
        set_mock_value(fs.status, value)

    callback_on_mock_put(fs.cmd, _stuck_open)
    with pytest.raises(Exception, match="shutter stuck"):
        RE(_map(pe, fs, grid_x, grid_y))
    assert await grid_x.velocity.get_value() == 2.5


async def test_xrd_map_restores_velocity_and_closes_the_shutter_on_failure(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter, stages
):
    grid_x, grid_y = stages

    def _fault(value, **_):
        raise RuntimeError("detector fault")

    callback_on_mock_put(pe.cb.trigger_, _fault)  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(Exception, match="detector fault"):
        RE(_map(pe, fs, grid_x, grid_y))
    assert await grid_x.velocity.get_value() == 2.5
    assert await fs.status.get_value() == OnOff.ON
    assert get_mock_put(fs.cmd).call_count >= 2


async def test_xrd_map_sums_frames_to_the_dwell_time(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter, stages
):
    grid_x, grid_y = stages
    docs, msgs = _run(
        RE,
        xrd_map([pe], grid_x, 0.0, 2.0, 3, grid_y, 0.0, 1.0, 2, 0.45, shutter=fs),
    )
    (start,) = docs["start"]
    # the PE's frame time is 0.1 s: 0.45 s rounds up to 5 frames, 0.5 s per pixel
    assert start["sp_time_per_frame"] == 0.1
    assert start["sp_num_frames"] == 5
    assert start["sp_requested_exposure"] == 0.45
    assert start["sp_computed_exposure"] == pytest.approx(0.5)
    assert await pe.proc.num_filter.get_value() == 5  # pyright: ignore[reportAttributeAccessIssue]
    assert _sets(msgs, grid_x.velocity)[1][1] == pytest.approx(2.0 / (3 * 0.5))


async def test_xrd_map_sets_the_frame_time_when_given(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter, stages
):
    grid_x, grid_y = stages
    docs, _ = _run(
        RE,
        xrd_map(
            [pe],
            grid_x,
            0.0,
            2.0,
            3,
            grid_y,
            0.0,
            1.0,
            2,
            1.0,
            frame_time=0.25,
            shutter=fs,
        ),
    )
    assert await pe.driver.acquire_time.get_value() == 0.25
    assert docs["start"][0]["sp_num_frames"] == 4


async def test_xrd_map_refuses_a_dwell_shorter_than_one_frame(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter, stages
):
    grid_x, grid_y = stages
    msgs: list = []
    RE.msg_hook = msgs.append  # pyright: ignore[reportAttributeAccessIssue]
    try:
        with pytest.raises(Exception, match="shorter than"):
            RE(
                xrd_map(
                    [pe], grid_x, 0.0, 2.0, 3, grid_y, 0.0, 1.0, 2, 0.05, shutter=fs
                )
            )
    finally:
        RE.msg_hook = None
    assert _sets(msgs, grid_x, grid_y) == []
