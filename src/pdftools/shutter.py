"""Fast shutter for the PDF beamline."""

import asyncio

from bluesky.protocols import Movable
from ophyd_async.core import (
    AsyncStatus,
    StandardReadable,
    StrictEnum,
    derived_signal_r,
    wait_for_value,
)
from ophyd_async.epics.core import epics_signal_r, epics_signal_rw


class OnOff(StrictEnum):
    """States of the fast shutter's Cmd and Sts records."""

    OFF = "OFF"
    ON = "ON"


# The PDF fast shutter is inverted: OFF opens it and ON closes it.
_COMMANDS = {"Open": OnOff.OFF, 1: OnOff.OFF, "Close": OnOff.ON, 0: OnOff.ON}


def _state(status: OnOff) -> str:
    return "Open" if status == OnOff.OFF else "Close"


class FastShutter(StandardReadable, Movable[str | int]):
    """Fast shutter: 1 or ``"Open"`` opens it, 0 or ``"Close"`` closes it.

    A move finishes when the ``Sts`` readback matches the command, then waits
    ``settle_time`` seconds.
    """

    def __init__(self, prefix: str, name: str = "", settle_time: float = 0.1):
        self.cmd = epics_signal_rw(OnOff, prefix + "Cmd")
        self.status = epics_signal_r(OnOff, prefix + "Sts")
        with self.add_children_as_readables():
            self.state = derived_signal_r(_state, status=self.status)
        self._settle_time = settle_time
        super().__init__(name=name)

    @AsyncStatus.wrap
    async def set(self, value: str | int, timeout: float = 10.0) -> None:
        try:
            command = _COMMANDS[value]
        except KeyError:
            raise ValueError(
                f"Fast shutter takes 'Open'/1 or 'Close'/0, not {value!r}"
            ) from None
        await self.cmd.set(command)
        await wait_for_value(self.status, command, timeout=timeout)
        await asyncio.sleep(self._settle_time)
