"""The jog plan against a mocked Perkin Elmer (conftest ``pe``) and fast shutter."""

import asyncio
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
from pdftools.xrd import xrd_jog


@pytest.fixture
def fs(RE: RunEngine) -> FastShutter:
    with init_devices(mock=True):
        fs = FastShutter("FS:", name="fs", settle_time=0)
    set_mock_value(fs.status, OnOff.ON)  # closed
    callback_on_mock_put(fs.cmd, lambda value, **_: set_mock_value(fs.status, value))
    return fs


@pytest.fixture
def grid_y(RE: RunEngine) -> SimMotor:
    with init_devices():
        grid_y = SimMotor(name="grid_y", instant=True)
    RE(bps.mv(grid_y.velocity, 2.5))
    return grid_y


def _run(RE: RunEngine, plan) -> tuple[dict[str, list[Any]], list]:
    docs: dict[str, list[Any]] = {}
    msgs: list = []
    RE.msg_hook = msgs.append  # pyright: ignore[reportAttributeAccessIssue]
    try:
        RE(plan, lambda name, doc: docs.setdefault(name, []).append(doc))
    finally:
        RE.msg_hook = None
    return docs, msgs


def _moves(msgs: list, obj) -> list[float]:
    return [m.args[0] for m in msgs if m.command == "set" and m.obj is obj]


async def test_xrd_jog_sweeps_back_and_forth_over_each_image(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter, grid_y: SimMotor
):
    docs, msgs = _run(
        RE,
        xrd_jog(
            [pe],
            grid_y,
            0.0,
            2.0,
            0.1,
            images_to_average=5,
            num_images=3,
            sample_name="Ni",
            shutter=fs,
        ),
    )
    (start,) = docs["start"]
    assert start["plan_name"] == "xrd_jog"
    assert start["sample_name"] == "Ni"
    assert start["jog"] == {"motor": "grid_y", "start": 0.0, "stop": 2.0}
    assert len(docs["event"]) == 3
    # to start, then across; the next image sweeps back the other way
    assert _moves(msgs, grid_y) == [0.0, 2.0, 2.0, 0.0, 0.0, 2.0]
    # each sweep lasts one averaged image: 2.0 mm over 5 x 0.1 s
    assert _moves(msgs, grid_y.velocity) == [pytest.approx(4.0)] * 3 + [2.5]
    assert await grid_y.velocity.get_value() == 2.5
    assert await pe.proc.num_filter.get_value() == 5  # pyright: ignore[reportAttributeAccessIssue]
    assert [c.args[0] for c in get_mock_put(fs.cmd).call_args_list] == [
        OnOff.OFF,
        OnOff.ON,
    ]


async def test_xrd_jog_without_travel_does_not_move(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter, grid_y: SimMotor
):
    docs, msgs = _run(
        RE, xrd_jog([pe], grid_y, 1.0, 1.0, 0.1, num_images=2, shutter=fs)
    )
    assert len(docs["event"]) == 2
    assert _moves(msgs, grid_y) == []
    assert await grid_y.velocity.get_value() == 2.5


async def test_xrd_jog_restores_velocity_and_closes_the_shutter_on_failure(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter, grid_y: SimMotor
):
    tiff, cb = pe.tiff, pe.cb  # pyright: ignore[reportAttributeAccessIssue]
    triggers: list[bool] = []

    async def _fault_on_second_image(value):
        if not value:
            return
        triggers.append(value)
        if len(triggers) == 2:
            raise RuntimeError("detector fault")
        set_mock_value(tiff.num_captured, await tiff.num_captured.get_value() + 1)
        asyncio.get_running_loop().call_later(0.01, set_mock_value, cb.trigger_, False)

    callback_on_mock_put(cb.trigger_, _fault_on_second_image)
    with pytest.raises(Exception, match="detector fault"):
        RE(xrd_jog([pe], grid_y, 0.0, 2.0, 0.1, num_images=3, shutter=fs))
    assert await grid_y.velocity.get_value() == 2.5
    assert await fs.status.get_value() == OnOff.ON


async def test_xrd_jog_restores_velocity_when_the_shutter_fails_to_close(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter, grid_y: SimMotor
):
    def _stuck_open(value, **_):
        if value == OnOff.ON:
            raise RuntimeError("shutter stuck")
        set_mock_value(fs.status, value)

    callback_on_mock_put(fs.cmd, _stuck_open)
    with pytest.raises(Exception, match="shutter stuck"):
        RE(xrd_jog([pe], grid_y, 0.0, 2.0, 0.1, num_images=2, shutter=fs))
    assert await grid_y.velocity.get_value() == 2.5
