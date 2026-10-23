"""The temperature-series plan against a mocked Eurotherm, Perkin Elmer and shutter."""

from typing import Any

import pytest
from bluesky.run_engine import RunEngine
from ophyd_async.core import (
    callback_on_mock_put,
    get_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.adcore import ContAcqDetector

from pdftools.eurotherm3k import Eurotherm3k
from pdftools.shutter import FastShutter, OnOff
from pdftools.xrd import xrd_temperature_series


@pytest.fixture
def fs(RE: RunEngine) -> FastShutter:
    with init_devices(mock=True):
        fs = FastShutter("FS:", name="fs", settle_time=0)
    set_mock_value(fs.status, OnOff.ON)  # closed
    callback_on_mock_put(fs.cmd, lambda value, **_: set_mock_value(fs.status, value))
    return fs


@pytest.fixture
def eurotherm(RE: RunEngine) -> Eurotherm3k:
    with init_devices(mock=True):
        eurotherm = Eurotherm3k("ET:", name="eurotherm")
    set_mock_value(eurotherm.loop1.readback, 25.0)
    return eurotherm


def _run(RE: RunEngine, plan) -> tuple[dict[str, list[Any]], list]:
    docs: dict[str, list[Any]] = {}
    msgs: list = []
    RE.msg_hook = msgs.append  # pyright: ignore[reportAttributeAccessIssue]
    try:
        RE(plan, lambda name, doc: docs.setdefault(name, []).append(doc))
    finally:
        RE.msg_hook = None
    return docs, msgs


async def test_xrd_temperature_series_counts_at_each_settled_temperature(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter, eurotherm: Eurotherm3k
):
    loop = eurotherm.loop1
    callback_on_mock_put(
        loop.setpoint, lambda value, **_: set_mock_value(loop.readback, value + 0.2)
    )
    docs, msgs = _run(
        RE,
        xrd_temperature_series(
            [pe],
            loop,
            [100.0, 200.0, 300.0],
            0.1,
            num_iterations=2,
            sample_name="Ni",
            shutter=fs,
        ),
    )
    assert [c.args[0] for c in get_mock_put(loop.setpoint).call_args_list] == [
        100.0,
        200.0,
        300.0,
    ]
    starts = docs["start"]
    assert len(starts) == 6  # two runs at each temperature
    assert {s["plan_name"] for s in starts} == {"xrd_temperature_series"}
    assert {s["sample_name"] for s in starts} == {"Ni"}
    assert [s["temperature"] for s in starts[::2]] == [
        {"controller": "eurotherm-loop1", "setpoint": t, "readback": t + 0.2}
        for t in (100.0, 200.0, 300.0)
    ]
    # closed while the temperature moves, open only to count
    moves = [
        (m.obj.name, m.args[0])
        for m in msgs
        if m.command == "set" and m.obj in (fs, loop)
    ]
    assert moves == [
        step
        for t in (100.0, 200.0, 300.0)
        for step in (("fs", "Close"), ("eurotherm-loop1", t), ("fs", "Open"))
    ] + [("fs", "Close")]
    assert await fs.status.get_value() == OnOff.ON
    assert get_mock_put(loop.ramp_rate).call_count == 0


async def test_xrd_temperature_series_sets_the_ramp_rate_once(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter, eurotherm: Eurotherm3k
):
    loop = eurotherm.loop1
    callback_on_mock_put(
        loop.setpoint, lambda value, **_: set_mock_value(loop.readback, value)
    )
    _run(
        RE,
        xrd_temperature_series(
            [pe], loop, [50.0, 60.0], 0.1, ramp_rate=5.0, shutter=fs
        ),
    )
    assert [c.args[0] for c in get_mock_put(loop.ramp_rate).call_args_list] == [5.0]


async def test_xrd_temperature_series_closes_the_shutter_if_never_settled(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter, eurotherm: Eurotherm3k
):
    loop = eurotherm.loop1
    # mock=True gives a StandardMovable InstantMovableMock: replace it so the
    # readback stays at 25 instead of jumping to the setpoint
    callback_on_mock_put(loop.setpoint, lambda *_, **__: None)
    loop.move_timeout = 0.2
    docs: dict[str, list[Any]] = {}
    with pytest.raises(Exception):  # noqa: B017  # bluesky wraps the timeout
        RE(
            xrd_temperature_series([pe], loop, [300.0], 0.1, shutter=fs),
            lambda name, doc: docs.setdefault(name, []).append(doc),
        )
    assert "start" not in docs  # nothing counted off temperature
    assert await fs.status.get_value() == OnOff.ON


async def test_xrd_temperature_series_closes_an_open_shutter_before_moving(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter, eurotherm: Eurotherm3k
):
    loop = eurotherm.loop1
    set_mock_value(fs.status, OnOff.OFF)  # left open by an earlier plan
    shutter_at_move: list[OnOff] = []

    async def _reach(value, **_):
        shutter_at_move.append(await fs.status.get_value())
        set_mock_value(loop.readback, value)

    callback_on_mock_put(loop.setpoint, _reach)
    _run(RE, xrd_temperature_series([pe], loop, [100.0], 0.1, shutter=fs))
    assert shutter_at_move == [OnOff.ON]


async def test_xrd_temperature_series_reads_the_temperature_before_each_run(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter, eurotherm: Eurotherm3k
):
    loop = eurotherm.loop1
    callback_on_mock_put(
        loop.setpoint, lambda value, **_: set_mock_value(loop.readback, value)
    )
    drift = iter([100.4, 100.7])

    def _drift(name, doc):
        if name == "stop":  # the temperature wanders between runs
            set_mock_value(loop.readback, next(drift, 100.9))

    RE.subscribe(_drift)
    docs, _ = _run(
        RE,
        xrd_temperature_series([pe], loop, [100.0], 0.1, num_iterations=3, shutter=fs),
    )
    assert [s["temperature"]["readback"] for s in docs["start"]] == [
        100.0,
        100.4,
        100.7,
    ]
