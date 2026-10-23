"""Sample environment devices for the PDF beamline."""

import asyncio

from bluesky.protocols import Locatable, Location, Stoppable
from ophyd_async.core import (
    CALCULATE_TIMEOUT,
    DEFAULT_TIMEOUT,
    AsyncStatus,
    CalculatableTimeout,
    StandardReadable,
    StrictEnum,
    WatchableAsyncStatus,
    WatcherUpdate,
    observe_value,
)
from ophyd_async.core import (
    StandardReadableFormat as Format,
)
from ophyd_async.epics.core import (
    EpicsDevice,
    epics_signal_r,
    epics_signal_rw,
)


class LNPMode(StrictEnum):
    """Enum for LNP mode settings."""

    MANUAL = "Manual"
    AUTO = "Auto"


class LinkamT96(EpicsDevice, StandardReadable, Locatable[float], Stoppable):
    """Linkam T96 temperature controller: ramp, hold, read back and stop."""

    def __init__(self, prefix: str, name: str):
        self.heat = epics_signal_rw(bool, prefix + "STARTHEAT", name="heat")

        with self.add_children_as_readables(Format.HINTED_SIGNAL):
            self.temperature = epics_signal_r(
                float, prefix + "TEMP", name="temperature"
            )

        with self.add_children_as_readables(Format.CONFIG_SIGNAL):
            # Device information signals
            self.model = epics_signal_r(str, prefix + "MODEL", name="model")
            self.serial = epics_signal_r(str, prefix + "SERIAL", name="serial")
            self.fw_vers = epics_signal_r(str, prefix + "FIRM:VER", name="fw_vers")
            self.hw_vers = epics_signal_r(str, prefix + "HARD:VER", name="hw_vers")
            self.stage_serial = epics_signal_r(
                str, prefix + "STAGE:SERIAL", name="stage_serial"
            )

            self.setpoint = epics_signal_rw(
                float,
                prefix + "SETPOINT",
                write_pv=prefix + "SETPOINT:SET",
                name="setpoint",
            )
            self.ramp_rate = epics_signal_rw(
                float,
                prefix + "RAMPRATE",
                write_pv=prefix + "RAMPRATE:SET",
                name="ramp_rate",
            )
            self.hold_time = epics_signal_rw(
                float,
                prefix + "HOLDTIME",
                write_pv=prefix + "HOLDTIME:SET",
                name="hold_time",
            )

            self.lnp_mode = epics_signal_rw(
                LNPMode,
                prefix + "LNP_MODE:SET",
                name="lnp_mode",
            )
            self.lnp_speed = epics_signal_rw(
                float,
                prefix + "LNP_SPEED",
                write_pv=prefix + "LNP_SPEED:SET",
                name="lnp_speed",
            )

        self.ramping = epics_signal_r(bool, prefix + "STATUS", name="ramping")
        self.power = epics_signal_r(float, prefix + "POWER", name="power")

        super().__init__(prefix, name=name)

    @WatchableAsyncStatus.wrap
    async def set(self, value: float, timeout: CalculatableTimeout = CALCULATE_TIMEOUT):
        inital_temp, ramp_rate = await asyncio.gather(
            self.temperature.get_value(),
            self.ramp_rate.get_value(),
        )

        if ramp_rate == 0 and timeout == CALCULATE_TIMEOUT:
            raise ValueError("Ramp rate cannot be zero when calculating timeout.")

        await self.setpoint.set(value)
        if timeout == CALCULATE_TIMEOUT:
            timeout = abs(value - inital_temp) / ramp_rate + DEFAULT_TIMEOUT

        async for temp in observe_value(self.temperature, done_timeout=timeout):
            yield WatcherUpdate(
                current=temp,
                initial=inital_temp,
                target=value,
                name=self.name,
                unit="C",
                precision=3,
            )
            if not await self.ramping.get_value():
                break

    async def locate(self) -> Location[float]:
        setpoint, temperature = await asyncio.gather(
            self.setpoint.get_value(),
            self.temperature.get_value(),
        )
        return Location(setpoint=setpoint, readback=temperature)

    @AsyncStatus.wrap
    async def stop(self, success: bool = True):
        await self.heat.set(False)
