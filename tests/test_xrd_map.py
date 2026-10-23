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
    # 2.0 over 3 pixels of 0.1 s each; original speed to reposition and after
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


async def test_xrd_map_closes_the_shutter_whenever_motors_reposition(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter, stages
):
    grid_x, grid_y = stages
    set_mock_value(fs.status, OnOff.OFF)  # left open by an earlier plan
    _, msgs = _run(RE, _map(pe, fs, grid_x, grid_y))
    assert _sets(msgs, fs, grid_y) == [
        ("fs", "Close"),
        ("grid_y", 0.0),
        ("fs", "Open"),
        ("fs", "Close"),
        ("grid_y", 1.0),
        ("fs", "Open"),
        ("fs", "Close"),
    ]
    assert await fs.status.get_value() == OnOff.ON


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
