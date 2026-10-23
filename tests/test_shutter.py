import asyncio

import bluesky.plan_stubs as bps
import pytest
from bluesky.run_engine import RunEngine
from ophyd_async.core import (
    callback_on_mock_put,
    get_mock_put,
    init_devices,
    set_mock_value,
)

from pdftools.shutter import FastShutter, OnOff


@pytest.fixture
def fs(RE: RunEngine) -> FastShutter:
    with init_devices(mock=True):
        fs = FastShutter("TEST:FS:", name="fs", settle_time=0)
    return fs


def _status_follows_command(fs: FastShutter) -> None:
    callback_on_mock_put(fs.cmd, lambda value, **_: set_mock_value(fs.status, value))


@pytest.mark.parametrize(
    ("value", "command", "state"),
    [
        (1, OnOff.OFF, "Open"),
        ("Open", OnOff.OFF, "Open"),
        (0, OnOff.ON, "Close"),
        ("Close", OnOff.ON, "Close"),
    ],
)
def test_mv_sends_inverted_command_and_reads_state(
    RE: RunEngine, fs: FastShutter, value, command: OnOff, state: str
):
    # The hardware is inverted: OFF opens the shutter, ON closes it.
    set_mock_value(fs.status, OnOff.ON if command == OnOff.OFF else OnOff.OFF)
    _status_follows_command(fs)

    RE(bps.mv(fs, value))

    assert get_mock_put(fs.cmd).call_args.args[0] == command
    reading = asyncio.run_coroutine_threadsafe(fs.read(), RE.loop).result()
    assert reading["fs-state"]["value"] == state


async def test_set_waits_for_the_status_readback(fs: FastShutter):
    set_mock_value(fs.status, OnOff.ON)
    with pytest.raises(asyncio.TimeoutError):
        await fs.set("Open", timeout=0.1)


async def test_set_rejects_unknown_values(fs: FastShutter):
    with pytest.raises(ValueError, match="Open"):
        await fs.set(2)
