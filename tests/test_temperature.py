"""Cheng-Hung's Lake Shore helper plans against a mocked cryostat."""

import pytest
from bluesky.run_engine import RunEngine
from ophyd_async.core import (
    callback_on_mock_put,
    get_mock_put,
    set_mock_value,
)

from pdftools.lakeshore336 import (
    LakeShore336,
    Lakeshore336LoopInput,
    LakeShore336LoopLogic,
    Lakeshore336LoopMode,
    Lakeshore336RangeSelect,
    Lakeshore336Resistance,
    Lakeshore336Switch,
)
from pdftools.temperature import default_loop, move_and_continue, t_list


def _stick_at(lake: LakeShore336, stuck_at: float) -> None:
    def _reach(value, **_):
        if value != stuck_at:
            set_mock_value(lake.input_c.temp, value)

    callback_on_mock_put(lake.out1.sp, _reach)


def test_default_loop_holds_the_temperature_before_turning_the_heater_on(
    RE: RunEngine, fast_lake: LakeShore336
):
    loop = fast_lake.out1
    sets: list[tuple[str, object]] = []

    def _hook(msg):
        if msg.command == "set":
            sets.append((msg.obj.name, msg.args[0]))

    RE.msg_hook = _hook  # pyright: ignore[reportAttributeAccessIssue]
    try:
        RE(default_loop(loop))
    finally:
        RE.msg_hook = None
    assert sets[0] == (loop.sp.name, 80.0)
    assert sets[-1] == (loop.range.name, Lakeshore336RangeSelect.RANGE3)
    assert set(sets[1:-1]) == {
        (loop.loop_in_sel.name, Lakeshore336LoopInput.C),
        (loop.ramp_enbl.name, Lakeshore336Switch.ON),
        (loop.ramp.name, 6),
        (loop.enbl.name, Lakeshore336Switch.ON),
        (loop.loop_mode.name, Lakeshore336LoopMode.PID),
        (loop.maxi.name, 2),
        (loop.resistance.name, Lakeshore336Resistance.LO25),
    }


def test_t_list_visits_each_temperature(RE: RunEngine, fast_lake: LakeShore336):
    RE(t_list([90.0, 100.0], fast_lake.out1))
    assert [c.args[0] for c in get_mock_put(fast_lake.out1.sp).call_args_list] == [
        90.0,
        100.0,
    ]


def test_t_list_goes_on_past_a_temperature_that_did_not_settle(
    RE: RunEngine, fast_lake: LakeShore336, monkeypatch: pytest.MonkeyPatch, capsys
):
    async def _short(self, old_position, new_position):
        return 0.3

    monkeypatch.setattr(LakeShore336LoopLogic, "calculate_timeout", _short)
    _stick_at(fast_lake, 150.0)
    with pytest.warns(RuntimeWarning, match="did not settle at 150.0"):
        RE(t_list([150.0, 100.0], fast_lake.out1))
    assert [c.args[0] for c in get_mock_put(fast_lake.out1.sp).call_args_list] == [
        150.0,
        100.0,
    ]
    assert "1/2 setpoints did not settle" in capsys.readouterr().out


def test_move_and_continue_stops_on_other_failures(
    RE: RunEngine, fast_lake: LakeShore336
):
    with pytest.raises(Exception, match="outside"):
        RE(move_and_continue(fast_lake.out1, 600.0))
    assert get_mock_put(fast_lake.out1.sp).call_count == 0
