"""Fast shutter and filter bank for the PDF beamline."""

import asyncio

from bluesky.protocols import Movable
from ophyd_async.core import (
    AsyncStatus,
    DeviceMock,
    StandardReadable,
    StrictEnum,
    callback_on_mock_execute,
    default_mock_class,
    derived_signal_r,
    set_mock_value,
    wait_for_value,
)
from ophyd_async.epics.core import (
    epics_signal_r,
    epics_signal_rw,
    epics_triggerable_command,
)


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


class TwoButtonStatus(StrictEnum):
    """States of a two-button actuator's Pos-Sts record."""

    OPEN = "Open"
    CLOSED = "Not Open"


_TWO_BUTTON_TARGETS = {
    "Open": TwoButtonStatus.OPEN,
    1: TwoButtonStatus.OPEN,
    "Close": TwoButtonStatus.CLOSED,
    0: TwoButtonStatus.CLOSED,
}


def _two_button_state(status: TwoButtonStatus) -> str:
    return "Open" if status == TwoButtonStatus.OPEN else "Close"


class MockTwoButtonShutter(DeviceMock["TwoButtonShutter"]):
    """In mock mode the status follows each command at once."""

    async def connect(self, device: "TwoButtonShutter") -> None:
        set_mock_value(device.status, TwoButtonStatus.CLOSED)
        callback_on_mock_execute(
            device.open_cmd, lambda: set_mock_value(device.status, TwoButtonStatus.OPEN)
        )
        callback_on_mock_execute(
            device.close_cmd,
            lambda: set_mock_value(device.status, TwoButtonStatus.CLOSED),
        )


@default_mock_class(MockTwoButtonShutter)
class TwoButtonShutter(StandardReadable, Movable[str | int]):
    """Actuator with separate open and close commands, such as one filter.

    1 or ``"Open"`` fires ``Cmd:Opn-Cmd``, 0 or ``"Close"`` fires ``Cmd:Cls-Cmd``;
    a move finishes when ``Pos-Sts`` reports the new position.
    """

    def __init__(self, prefix: str, name: str = ""):
        self.open_cmd = epics_triggerable_command(prefix + "Cmd:Opn-Cmd")
        self.close_cmd = epics_triggerable_command(prefix + "Cmd:Cls-Cmd")
        self.status = epics_signal_r(TwoButtonStatus, prefix + "Pos-Sts")
        with self.add_children_as_readables():
            self.state = derived_signal_r(_two_button_state, status=self.status)
        super().__init__(name=name)

    @AsyncStatus.wrap
    async def set(self, value: str | int, timeout: float = 10.0) -> None:
        try:
            target = _TWO_BUTTON_TARGETS[value]
        except KeyError:
            raise ValueError(
                f"{self.name} takes 'Open'/1 or 'Close'/0, not {value!r}"
            ) from None
        if await self.status.get_value() == target:
            return
        command = self.open_cmd if target == TwoButtonStatus.OPEN else self.close_cmd
        await command.trigger()
        await wait_for_value(self.status, target, timeout=timeout)


class FilterBank(StandardReadable):
    """The four pneumatic filters in the PDF optics hutch.

    "Open" takes a filter out of the beam and "Close" puts it in, as on the OCM
    screen (cs-studio-xf ``mobile-ocm.bob``): filters 1 and 2 are copper, filter 4
    is the attenuator old plans closed to take darks.
    """

    def __init__(self, prefix: str, name: str = ""):
        with self.add_children_as_readables():
            self.flt1 = TwoButtonShutter(prefix + "1}")
            self.flt2 = TwoButtonShutter(prefix + "2}")
            self.flt3 = TwoButtonShutter(prefix + "3}")
            self.flt4 = TwoButtonShutter(prefix + "4}")
        super().__init__(name=name)
