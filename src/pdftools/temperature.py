"""Temperature-controller plans for the PDF cryostat and furnaces.

``default_loop``, ``move_and_continue`` and ``t_list`` are ported from
pdf-profile-collection ``startup/101-Lakeshore336Cryostat.py`` (Cheng-Hung Lin,
129831d), keeping their names.
"""

import warnings
from collections.abc import Sequence

from bluesky import plan_stubs as bps
from bluesky.utils import FailedStatus

from pdftools.eurotherm3k import Eurotherm3kLoop
from pdftools.lakeshore336 import (
    LakeShore336Loop,
    Lakeshore336LoopMode,
    Lakeshore336RangeSelect,
    Lakeshore336Resistance,
    Lakeshore336Switch,
)
from pdftools.utils import ensure_available

TemperatureController = Eurotherm3kLoop | LakeShore336Loop


def move_and_continue(device: TemperatureController, target: float):
    """Move ``device`` to ``target``; if it does not settle in time, warn and go on.

    The RunEngine re-raises a failed move into the plan as ``FailedStatus``, with
    the device's own exception as ``__cause__``. Only a ``TimeoutError`` is let
    through, so an out-of-range setpoint or a lost connection still stops the plan.

    Returns
    -------
    FailedStatus or None
        the failure if the move timed out, else ``None``
    """
    try:
        yield from bps.mv(device, target)
    except FailedStatus as e:
        if not isinstance(e.__cause__, TimeoutError):
            raise
        warnings.warn(
            f"{device.name} did not settle at {target} ({e.__cause__!r}); continuing",
            RuntimeWarning,
            stacklevel=2,
        )
        return e
    return None


def t_list(tlist: Sequence[float], controller: LakeShore336Loop | None = None):
    """Visit each temperature in turn, going on past any that does not settle.

    Parameters
    ----------
    tlist : list of float
        the setpoints, in K, in the order to visit them
    controller : LakeShore336Loop, optional
        the loop to move; the session's ``cryostat`` if not given

    Returns
    -------
    list of (float, FailedStatus)
        the setpoints that did not settle, with why
    """
    controller = ensure_available(LakeShore336Loop, cryostat=controller)
    failures = []
    for temp in tlist:
        exc = yield from move_and_continue(controller, temp)
        if exc is not None:
            failures.append((temp, exc))
    if failures:
        print(f"{len(failures)}/{len(tlist)} setpoints did not settle:")
        for temp, exc in failures:
            print(f"  {temp}: {exc.__cause__!r}")
    return failures


def default_loop(loop: LakeShore336Loop):
    """Put a Lake Shore loop in its standard PID state and turn its heater on.

    The setpoint is first set to the current temperature, so nothing moves
    when the heater comes on. Then the loop reads its own input, ramps at
    6 K/min in PID mode with a 25 ohm heater and maximum-current setting 2,
    and finally the heater range is set to 3.
    """
    current = yield from bps.rd(loop.loop_input.temp)
    yield from bps.mv(loop.sp, current)
    yield from bps.mv(
        loop.loop_in_sel, loop.loop_input.channel,
        loop.ramp_enbl, Lakeshore336Switch.ON,
        loop.ramp, 6,
        loop.enbl, Lakeshore336Switch.ON,
        loop.loop_mode, Lakeshore336LoopMode.PID,
        loop.maxi, 2,
        loop.resistance, Lakeshore336Resistance.LO25,
    )  # fmt: skip
    yield from bps.mv(loop.range, Lakeshore336RangeSelect.RANGE3)
