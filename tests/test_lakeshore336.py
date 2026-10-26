"""Lake Shore 336 (PDF cryostat) against mocked PVs, pinned to the live IOC."""

import time

import pytest
from bluesky import plan_stubs as bps
from bluesky.run_engine import RunEngine
from ophyd_async.core import (
    Device,
    SignalR,
    StandardReadable,
    callback_on_mock_put,
    get_mock_put,
    init_devices,
    set_mock_put_proceeds,
    set_mock_value,
)

from pdftools.lakeshore336 import (
    LakeShore336,
    LakeShore336Input,
    LakeShore336Loop,
    Lakeshore336LoopInput,
    Lakeshore336LoopMode,
    Lakeshore336RangeSelect,
    Lakeshore336Resistance,
    Lakeshore336Switch,
)

# Suffixes served per input and per loop by the lakeshore336 IOC on
# xf28id1-inst-ioc1 (its records.dbl, 2026-10-26).
IOC_INPUT_SUFFIXES = {
    "Alrm-Calc_", "Alrm:High-Sts", "Alrm:Low-Sts", "Cmd:Fltr-Calc_", "Cmd:Fltr-Sel",
    "Cmd:Fltr-Sts", "FOut", "T:C-I", "T-I", "T:Lim-RB", "T:Lim-SP", "T:Name-RB",
    "T:Name-SP", "T-Sts", "Val:FltrP-RB", "Val:FltrP-SP", "Val:FltrW-RB",
    "Val:FltrW-SP", "Val:Sens-I",
}  # fmt: skip
IOC_LOOP_SUFFIXES = {
    "Alrm-Calc_", "Enbl:Ramp-Calc_", "Enbl:Ramp-Sel", "Enbl:Ramp-Sts", "Enbl-Sel",
    "Enbl-Sts", "Err:Htr-Sts", "FOut", "Gain-Calc_", "Gain:D-RB", "Gain:D-SP",
    "Gain:I-RB", "Gain:I-SP", "Gain:P-RB", "Gain:P-SP", "Mode:ATune-Sel",
    "Mode:ATune-Sts", "Mode-Calc_", "Mode-Sel", "Mode-Sts", "Out:Disp-RB",
    "Out:Disp-SP", "Out-I", "Out:Man-RB", "Out:Man-SP", "Out:MaxI-RB", "Out:MaxI-SP",
    "Out:Max-RB", "Out:Max-SP", "Out:R-Calc_", "Out:R-RB", "Out:R-SP", "Out-Sel",
    "Out-Sts", "Sts:Ramp-Sts", "T-RB", "T-SP", "Val:Ramp-RB", "Val:Ramp-SP",
    "Val:Range-Sel", "Val:Range-Sts",
}  # fmt: skip
# Enum states of the same IOC's lakeshore336_loop.db records
IOC_STATES = {
    Lakeshore336Switch: ["OFF", "ON"],  # Enbl-Sel, Enbl:Ramp-Sel
    Lakeshore336LoopMode: ["OFF", "PID", "ZONE", "OPEN LOOP", "MONITOR", "WARMUP"],
    Lakeshore336RangeSelect: [
        "OFF", "RANGE 1", "RANGE 2", "RANGE 3", "RANGE 4", "RANGE 5",
    ],
    Lakeshore336Resistance: ["25 OHM", "50 OHM"],
    Lakeshore336LoopInput: ["NONE", "A", "B", "C", "D", "D2", "D3", "D4", "D5"],
}  # fmt: skip


class _Cryostat(StandardReadable):
    """One input and one loop, settling fast enough for a test."""

    def __init__(self, prefix: str, name: str = "") -> None:
        with self.add_children_as_readables():
            self.input_c = LakeShore336Input(
                f"{prefix}-Chan:C}}", Lakeshore336LoopInput.C
            )
            self.out1 = LakeShore336Loop(
                f"{prefix}-Out:1}}", self.input_c, tolerance=1, settle_time=0.05
            )
        super().__init__(name=name)


@pytest.fixture
def lake(RE: RunEngine) -> LakeShore336:
    with init_devices(mock=True):
        lake = LakeShore336("XF:28ID1-ES{LS336:1", name="Lakeshore")
    return lake


@pytest.fixture
def cryostat(RE: RunEngine) -> _Cryostat:
    with init_devices(mock=True):
        cryostat = _Cryostat("XF:28ID1-ES{LS336:1", name="cryostat")
    set_mock_value(cryostat.input_c.temp, 80.0)
    set_mock_value(cryostat.out1.ramp, 6.0)
    # the loop drives the input it reads to the new setpoint
    callback_on_mock_put(
        cryostat.out1.sp,
        lambda value, **_: set_mock_value(cryostat.input_c.temp, value),
    )
    return cryostat


def _epics_signals(device: Device) -> list[SignalR]:
    found = []
    for _, child in device.children():
        if isinstance(child, SignalR) and child.source.startswith("mock+ca://"):
            found.append(child)
        elif not isinstance(child, SignalR):
            found.extend(_epics_signals(child))
    return found


def test_every_pv_is_served_by_the_pdf_ioc(lake: LakeShore336):
    inputs = [lake.input_a, lake.input_b, lake.input_c, lake.input_d]
    inputs += [lake.input_d2, lake.input_d3, lake.input_d4, lake.input_d5]
    loops = [lake.out1, lake.out2, lake.out3, lake.out4]
    for devices, suffixes, kind in (
        (inputs, IOC_INPUT_SUFFIXES, "Chan"),
        (loops, IOC_LOOP_SUFFIXES, "Out"),
    ):
        for device in devices:
            signals = _epics_signals(device)
            assert len(signals) >= 4, device.name
            for signal in signals:
                pv = signal.source.removeprefix("mock+ca://")
                prefix, suffix = pv.split("}", 1)
                assert prefix.startswith(f"XF:28ID1-ES{{LS336:1-{kind}:"), pv
                assert suffix in suffixes, pv


def test_loops_read_the_inputs_of_the_old_profile(lake: LakeShore336):
    assert lake.out1.loop_input is lake.input_c
    assert {lake.out2.loop_input, lake.out3.loop_input, lake.out4.loop_input} == {
        lake.input_a
    }


@pytest.mark.parametrize("enum", list(IOC_STATES))
def test_enum_states_match_the_ioc(enum):
    assert [member.value for member in enum] == IOC_STATES[enum]


async def test_select_input_points_the_loop_at_its_input(lake: LakeShore336):
    await lake.out1.select_input()
    assert await lake.out1.loop_in_sel.get_value() == Lakeshore336LoopInput.C


async def test_readback_mirrors_the_input_and_is_named_after_the_loop(
    lake: LakeShore336,
):
    set_mock_value(lake.input_c.temp, 77.0)
    location = await lake.out1.locate()
    assert location["readback"] == 77.0
    assert lake.out1.movable_logic.readback.name == "Lakeshore-out1"
    assert lake.input_c.temp.name == "Lakeshore-input_c-temp"


@pytest.mark.parametrize(
    ("start", "target", "pid"),
    [
        (80.0, 100.0, (50, 10, 1)),  # heating, below 140 K
        (250.0, 280.0, (26, 5, 3)),  # heating, 270-300 K
        (250.0, 150.0, (25, 6, 3)),  # cooling, 140-200 K
        (250.0, 100.0, (25, 4, 3)),  # cooling, below 140 K
    ],
)
async def test_a_move_sets_the_pids_for_its_direction_and_settles(
    RE: RunEngine, cryostat: _Cryostat, start, target, pid
):
    set_mock_value(cryostat.input_c.temp, start)
    RE(bps.mv(cryostat.out1, target))
    out1 = cryostat.out1
    assert await out1.sp.get_value() == target
    assert (
        await out1.p.get_value(),
        await out1.i.get_value(),
        await out1.d.get_value(),
    ) == pid


async def test_a_small_move_keeps_the_pids(RE: RunEngine, cryostat: _Cryostat):
    set_mock_value(cryostat.out1.p, 1.0)
    RE(bps.mv(cryostat.out1, 80.3))
    assert await cryostat.out1.p.get_value() == 1.0


async def test_the_timeout_allows_for_the_ramp_and_the_settle(cryostat: _Cryostat):
    logic = cryostat.out1.movable_logic
    # 60 K at 6 K/min, plus the settle time, plus 30 s
    assert await logic.calculate_timeout(80.0, 140.0) == pytest.approx(630.05)


@pytest.mark.parametrize("target", [4.0, 551.0])
def test_moves_outside_5_to_550_k_are_refused(
    RE: RunEngine, cryostat: _Cryostat, target
):
    with pytest.raises(Exception, match="outside 5-550 K"):
        RE(bps.mv(cryostat.out1, target))
    get_mock_put(cryostat.out1.sp).assert_not_called()


async def test_a_move_that_never_settles_times_out(cryostat: _Cryostat):
    callback_on_mock_put(cryostat.out1.sp, lambda *_, **__: None)
    with pytest.raises(TimeoutError):
        await cryostat.out1.set(100.0, timeout=0.2)


async def test_with_the_ramp_off_the_timeout_assumes_0_015_k_per_second(
    cryostat: _Cryostat,
):
    set_mock_value(cryostat.out1.ramp, 0.0)
    logic = cryostat.out1.movable_logic
    # 60 K at 0.015 K/s, plus the settle time, plus 30 s
    assert await logic.calculate_timeout(80.0, 140.0) == pytest.approx(4030.05)


async def test_a_move_with_its_own_timeout_still_sets_the_pids(cryostat: _Cryostat):
    await cryostat.out1.set(100.0, timeout=5.0)
    out1 = cryostat.out1
    gains = (await out1.p.get_value(), await out1.i.get_value())
    assert (*gains, await out1.d.get_value()) == (50, 10, 1)


async def test_the_timeout_also_bounds_the_setpoint_write(cryostat: _Cryostat):
    set_mock_put_proceeds(cryostat.out1.sp, False)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        await cryostat.out1.set(100.0, timeout=0.2)
    assert time.monotonic() - start < 2.0
    set_mock_put_proceeds(cryostat.out1.sp, True)


async def test_the_loop_reads_its_measured_temperature(cryostat: _Cryostat):
    set_mock_value(cryostat.input_c.temp, 77.0)
    reading = await cryostat.out1.read()
    assert reading["cryostat-out1"]["value"] == 77.0
    assert "cryostat-out1" in await cryostat.out1.describe()
    assert "cryostat-out1" in cryostat.out1.hints.get("fields", [])
