import asyncio

import bluesky.plan_stubs as bps
import pytest
from bluesky.run_engine import RunEngine
from ophyd_async.core import (
    callback_on_mock_execute,
    get_mock_execute,
    init_devices,
    set_mock_value,
)

from pdftools.shutter import FilterBank, TwoButtonStatus

PREFIX = "XF:28ID1B-OP{Fltr:"


@pytest.fixture
def fb(RE: RunEngine) -> FilterBank:
    with init_devices(mock=True):
        fb = FilterBank(PREFIX, name="fb")
    return fb


def test_pv_names_match_the_old_profile_and_screens(fb: FilterBank):
    # pdf-profile-collection 11-motors and cs-studio-xf ut/_valve2.bob (OCM screen)
    for i, flt in enumerate((fb.flt1, fb.flt2, fb.flt3, fb.flt4), start=1):
        dev = f"{PREFIX}{i}}}"
        assert flt.status.source == f"mock+ca://{dev}Pos-Sts"
        assert flt.open_cmd.source == f"mock+ca://{dev}Cmd:Opn-Cmd"
        assert flt.close_cmd.source == f"mock+ca://{dev}Cmd:Cls-Cmd"


@pytest.mark.parametrize(
    ("value", "start", "state"),
    [
        (1, TwoButtonStatus.CLOSED, "Open"),
        ("Open", TwoButtonStatus.CLOSED, "Open"),
        (0, TwoButtonStatus.OPEN, "Close"),
        ("Close", TwoButtonStatus.OPEN, "Close"),
    ],
)
def test_mv_fires_one_command_and_reads_state(
    RE: RunEngine, fb: FilterBank, value, start: TwoButtonStatus, state: str
):
    flt = fb.flt4
    set_mock_value(flt.status, start)

    RE(bps.mv(flt, value))

    fired, idle = (
        (flt.open_cmd, flt.close_cmd)
        if state == "Open"
        else (flt.close_cmd, flt.open_cmd)
    )
    assert get_mock_execute(fired).call_count == 1
    assert get_mock_execute(idle).call_count == 0
    reading = asyncio.run_coroutine_threadsafe(fb.read(), RE.loop).result()
    assert reading["fb-flt4-state"]["value"] == state


async def test_set_skips_the_command_when_already_there(fb: FilterBank):
    set_mock_value(fb.flt1.status, TwoButtonStatus.CLOSED)
    await fb.flt1.set("Close")
    assert get_mock_execute(fb.flt1.close_cmd).call_count == 0


async def test_set_waits_for_the_status_readback(fb: FilterBank):
    set_mock_value(fb.flt1.status, TwoButtonStatus.CLOSED)
    # The filter does not move: the status never follows the command.
    callback_on_mock_execute(fb.flt1.open_cmd, lambda: None)
    with pytest.raises(asyncio.TimeoutError):
        await fb.flt1.set("Open", timeout=0.1)


async def test_set_rejects_unknown_values(fb: FilterBank):
    with pytest.raises(ValueError, match="Open"):
        await fb.flt1.set(2)
