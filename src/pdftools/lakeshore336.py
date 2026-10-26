# ruff: noqa: D101, D102
"""Lake Shore 336 temperature controller (the PDF cryostat, ``XF:28ID1-ES{LS336:1}``).

Ported from pdf-profile-collection ``startup/101-Lakeshore336Cryostat.py``
(Cheng-Hung Lin, 129831d), keeping its names, PID tables and limits. The PV
suffixes and enum states match the live IOC's records (``lakeshore336`` on
xf28id1-inst-ioc1, read 2026-10-26).
"""

import asyncio
from bisect import bisect
from dataclasses import dataclass, field
from functools import cached_property
from math import isclose
from time import perf_counter
from typing import Annotated as A

from ophyd_async.core import (
    MovableLogic,
    Reference,
    SignalR,
    SignalRW,
    StandardMovable,
    StandardReadable,
    StrictEnum,
    TimeoutCalculator,
    derived_signal_r,
)
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.core import EpicsDevice, PvSuffix


class Lakeshore336Switch(StrictEnum):
    OFF = "OFF"
    ON = "ON"


class Lakeshore336LoopMode(StrictEnum):
    OFF = "OFF"
    PID = "PID"
    ZONE = "ZONE"
    OLOOP = "OPEN LOOP"
    MON = "MONITOR"
    WARM = "WARMUP"


class Lakeshore336RangeSelect(StrictEnum):
    OFF = "OFF"
    RANGE1 = "RANGE 1"
    RANGE2 = "RANGE 2"
    RANGE3 = "RANGE 3"
    RANGE4 = "RANGE 4"
    RANGE5 = "RANGE 5"


class Lakeshore336Resistance(StrictEnum):
    LO25 = "25 OHM"
    HI50 = "50 OHM"


class Lakeshore336LoopInput(StrictEnum):
    NONE = "NONE"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    D2 = "D2"
    D3 = "D3"
    D4 = "D4"
    D5 = "D5"


class LakeShore336Input(EpicsDevice, StandardReadable):
    """One Lake Shore 336 temperature input."""

    temp: A[SignalR[float], PvSuffix("T-I"), Format.HINTED_SIGNAL]
    tempc: A[SignalR[float], PvSuffix("T:C-I")]
    raw: A[SignalR[float], PvSuffix("Val:Sens-I")]
    status: A[SignalR[str], PvSuffix("T-Sts")]

    def __init__(
        self, prefix: str, channel: Lakeshore336LoopInput, *, name: str = ""
    ) -> None:
        # The input's own identity, whichever loop (if any) reads it.
        self.channel = channel
        super().__init__(prefix, name=name)


# PID gains (P, I, D) chosen by target temperature (K) and move direction,
# as tuned on the PDF cryostat.
HEATING_THRESHOLDS = [140, 270, 300]
HEATING_PIDS = [(50, 10, 1), (25, 6, 3), (26, 5, 3), (26, 5, 3)]
COOLING_THRESHOLDS = [140, 200]
COOLING_PIDS = [(25, 4, 3), (25, 6, 3), (35, 8, 3)]
MIN_TEMPERATURE = 5
MAX_TEMPERATURE = 550


@dataclass
class LakeShore336LoopLogic(MovableLogic[float]):
    """What a move means for one Lake Shore control loop."""

    tolerance: float
    settle_time: float
    move_timeout: float
    p: SignalRW[float]
    i: SignalRW[float]
    d: SignalRW[float]
    ramp: SignalRW[float]
    delta: float = field(default=0.0, init=False)

    async def check_move(self, new_position: float) -> None:
        if not MIN_TEMPERATURE <= new_position <= MAX_TEMPERATURE:
            raise ValueError(
                f"{new_position} K is outside {MIN_TEMPERATURE}-{MAX_TEMPERATURE} K"
            )

    async def delta_t(self, old_position: float, new_position: float) -> None:
        self.delta = new_position - old_position

    async def set_pids(self, new_position: float) -> None:
        if self.delta > 0.5:
            temp_thresholds, pids = HEATING_THRESHOLDS, HEATING_PIDS
        elif self.delta < -0.5:
            temp_thresholds, pids = COOLING_THRESHOLDS, COOLING_PIDS
        else:
            return
        _p, _i, _d = pids[bisect(temp_thresholds, new_position)]
        await self.p.set(_p)
        await self.i.set(_i)
        await self.d.set(_d)

    async def calculate_timeout(
        self, old_position: float, new_position: float
    ) -> float:
        delta = new_position - old_position
        print(f"Delta = {delta:.2f}")

        ramp_rate = await self.ramp.get_value()  # K/min
        if ramp_rate == 0:
            ramp_rate = 0.015 * 60  # 0.015 K/s
        travel_time = abs(delta) / ramp_rate * 60
        timeout = travel_time + self.settle_time + 30
        print(f"{timeout:.2f} Seconds to complete the temp change")
        return timeout

    async def move(self, new_position: float, timeout: TimeoutCalculator) -> None:
        """Write the setpoint and wait for the readback to settle around it."""
        start_time = perf_counter()
        original_temp = await self.readback.get_value()
        print(f"Changing temp from {original_temp:.2f} to {new_position:.2f}")
        await self.delta_t(original_temp, new_position)

        loop = asyncio.get_running_loop()
        settled = asyncio.Event()
        settle_timer: asyncio.TimerHandle | None = None

        def update_settled_state(reading) -> None:
            nonlocal settle_timer
            value = reading[self.readback.name]["value"]
            in_tolerance = isclose(
                value, new_position, rel_tol=0.0, abs_tol=self.tolerance
            )
            # The timer starts on entering tolerance and is cancelled on leaving it.
            if in_tolerance and settle_timer is None:
                settle_timer = loop.call_later(self.settle_time, settled.set)
            elif not in_tolerance and settle_timer is not None:
                settle_timer.cancel()
                settle_timer = None
                settled.clear()

        # Subscribe before writing so that a fast readback update cannot be missed.
        self.readback.subscribe_reading(update_settled_state)
        try:
            async with asyncio.timeout(timeout()):
                await self.set_pids(new_position)
                await self.setpoint.set(new_position)
                await settled.wait()
            minutes = (perf_counter() - start_time) / 60
            print(f"Temp change completed in {minutes:.2f} mins")
        finally:
            if settle_timer is not None:
                settle_timer.cancel()
            self.readback.clear_sub(update_settled_state)


class LakeShore336Loop(EpicsDevice, StandardReadable, StandardMovable[float]):
    """One heater/control loop: setpoint, PID, heater range, ramp settings."""

    enbl: A[SignalRW[str], PvSuffix("Enbl-Sel")]  # the loop's general on/off
    sp: A[SignalRW[float], PvSuffix("T-SP"), Format.CONFIG_SIGNAL]
    sprdk: A[SignalR[float], PvSuffix("T-RB"), Format.HINTED_SIGNAL]
    p: A[
        SignalRW[float],
        PvSuffix(write_suffix="Gain:P-SP", read_suffix="Gain:P-RB"),
        Format.CONFIG_SIGNAL,
    ]
    i: A[
        SignalRW[float],
        PvSuffix(write_suffix="Gain:I-SP", read_suffix="Gain:I-RB"),
        Format.CONFIG_SIGNAL,
    ]
    d: A[
        SignalRW[float],
        PvSuffix(write_suffix="Gain:D-SP", read_suffix="Gain:D-RB"),
        Format.CONFIG_SIGNAL,
    ]
    ramp: A[
        SignalRW[float],
        PvSuffix(write_suffix="Val:Ramp-SP", read_suffix="Val:Ramp-RB"),
        Format.CONFIG_SIGNAL,
    ]
    range: A[
        SignalRW[str],
        PvSuffix(write_suffix="Val:Range-Sel", read_suffix="Val:Range-Sts"),
        Format.CONFIG_SIGNAL,
    ]
    maxi: A[SignalRW[float], PvSuffix("Out:MaxI-SP")]
    loop_mode: A[SignalRW[str], PvSuffix("Mode-Sel"), Format.CONFIG_SIGNAL]
    resistance: A[SignalRW[str], PvSuffix("Out:R-SP"), Format.CONFIG_SIGNAL]
    ramp_enbl: A[SignalRW[str], PvSuffix("Enbl:Ramp-Sel"), Format.CONFIG_SIGNAL]
    autotune: A[
        SignalRW[str],
        PvSuffix(write_suffix="Mode:ATune-Sel", read_suffix="Mode:ATune-Sts"),
    ]
    loop_in_sel: A[
        SignalRW[Lakeshore336LoopInput], PvSuffix("Out-Sel"), Format.CONFIG_SIGNAL
    ]
    loop_in_rb: A[
        SignalR[Lakeshore336LoopInput], PvSuffix("Out-Sts"), Format.CONFIG_SIGNAL
    ]

    def __init__(
        self,
        prefix: str,
        loop_input: LakeShore336Input,
        *,
        tolerance: float = 0.1,
        settle_time: float = 5.0,
        move_timeout: float = 600.0,
        name: str = "",
    ) -> None:
        # A Reference, so the input (a child of LakeShore336) is not re-parented.
        self._loop_input_ref = Reference(loop_input)
        self._tolerance = tolerance
        self._settle_time = settle_time
        self._move_timeout = move_timeout
        super().__init__(prefix, name=name)
        self.add_readables([self.movable_logic.readback], Format.HINTED_SIGNAL)

    @property
    def loop_input(self) -> LakeShore336Input:
        return self._loop_input_ref()

    async def select_input(self) -> None:
        """Point the loop's PID input at the channel ``loop_input`` was built with."""
        await self.loop_in_sel.set(self.loop_input.channel)

    async def get_loop_temp_k(self) -> float:
        return await self.loop_input.temp.get_value()

    @cached_property
    def movable_logic(self) -> MovableLogic[float]:
        # A private mirror of the input's temperature: StandardMovable renames its
        # readback after this loop, which would rename the shared input signal.
        def temp(temp: float) -> float:
            return temp

        return LakeShore336LoopLogic(
            setpoint=self.sp,
            readback=derived_signal_r(temp, temp=self.loop_input.temp),
            tolerance=self._tolerance,
            settle_time=self._settle_time,
            move_timeout=self._move_timeout,
            p=self.p,
            i=self.i,
            d=self.d,
            ramp=self.ramp,
        )


class LakeShore336(StandardReadable):
    """Lake Shore 336: eight inputs (A-D, D2-D5) and four control loops."""

    def __init__(self, prefix: str, *, name: str = "") -> None:
        with self.add_children_as_readables():
            self.input_a = LakeShore336Input(
                f"{prefix}-Chan:A}}", Lakeshore336LoopInput.A
            )
            self.input_b = LakeShore336Input(
                f"{prefix}-Chan:B}}", Lakeshore336LoopInput.B
            )
            self.input_c = LakeShore336Input(
                f"{prefix}-Chan:C}}", Lakeshore336LoopInput.C
            )
            self.input_d = LakeShore336Input(
                f"{prefix}-Chan:D}}", Lakeshore336LoopInput.D
            )
            self.input_d2 = LakeShore336Input(
                f"{prefix}-Chan:D2}}", Lakeshore336LoopInput.D2
            )
            self.input_d3 = LakeShore336Input(
                f"{prefix}-Chan:D3}}", Lakeshore336LoopInput.D3
            )
            self.input_d4 = LakeShore336Input(
                f"{prefix}-Chan:D4}}", Lakeshore336LoopInput.D4
            )
            self.input_d5 = LakeShore336Input(
                f"{prefix}-Chan:D5}}", Lakeshore336LoopInput.D5
            )
            self.out1 = LakeShore336Loop(
                f"{prefix}-Out:1}}", self.input_c, tolerance=1, settle_time=120
            )
            self.out2 = LakeShore336Loop(f"{prefix}-Out:2}}", self.input_a)
            self.out3 = LakeShore336Loop(f"{prefix}-Out:3}}", self.input_a)
            self.out4 = LakeShore336Loop(f"{prefix}-Out:4}}", self.input_a)
        super().__init__(name=name)
