"""XRD acquisition plans for the PDF beamline.

Ported from hextools.xrd (hextools#110, 986fd54): the same count, dark and scan
plans, with PDF's fast shutter ``fs`` in place of HEX's front-end and photon
shutters, internal triggering only, and the detectors passed explicitly since
PDF has several (``pe1c``, ``pe2c``, ``pilatus1``, ``lambda1``).
"""

from collections.abc import Mapping, Sequence
from typing import Any

from bluesky import plan_stubs as bps
from bluesky import plans as bp
from bluesky import preprocessors as bpp
from bluesky.protocols import Movable, Readable
from bluesky.utils import short_uid
from ophyd_async.core import DetectorTrigger, StandardDetector, TriggerInfo
from ophyd_async.epics.adcore import ADImageMode, ContAcqDetector
from ophyd_async.epics.motor import Motor
from ophyd_async.sim import SimMotor

from pdftools.eurotherm3k import Eurotherm3kLoop
from pdftools.shutter import FastShutter
from pdftools.utils import ensure_available

Steppable = Motor | SimMotor
# Moving one waits until its reading has settled at the setpoint.
TemperatureController = Eurotherm3kLoop


def ensure_cont_acq_exposure(detector: StandardDetector, exposure_time: float):
    """Have a continuous-acquisition detector acquiring at ``exposure_time``.

    A ``ContAcqDetector`` only checks the exposure time it is prepared with, so
    the driver is stopped, set and restarted here. Other detectors set their
    exposure when prepared, so this does nothing for them.
    """
    if not isinstance(detector, ContAcqDetector):
        return
    acquiring = yield from bps.rd(detector.driver.acquire)
    acquire_time = yield from bps.rd(detector.driver.acquire_time)
    image_mode = yield from bps.rd(detector.driver.image_mode)
    if (
        acquiring
        and acquire_time == exposure_time
        and image_mode == ADImageMode.CONTINUOUS
    ):
        return
    if acquiring:
        yield from bps.abs_set(detector.driver.acquire, False, wait=True)
    yield from bps.mv(
        detector.driver.acquire_time,
        exposure_time,
        detector.driver.image_mode,
        ADImageMode.CONTINUOUS,
    )
    # A busy record: waiting on it would block until acquisition stops.
    yield from bps.abs_set(detector.driver.acquire, True, wait=False)


def _md(
    plan_name: str, sample_name: str | None, description: str | None, md: dict | None
) -> dict[str, Any]:
    _md: dict[str, Any] = {**(md or {}), "plan_name": plan_name}
    if sample_name is not None:
        _md["sample_name"] = sample_name
    if description is not None:
        _md["description"] = description
    return _md


def _count_runs(
    detectors: Sequence[StandardDetector],
    exposure_time: float,
    num_images: int,
    images_to_average: int,
    num_iterations: int,
    time_gap: float,
    md: dict[str, Any],
):
    trigger_info = TriggerInfo(
        trigger=DetectorTrigger.INTERNAL,
        livetime=exposure_time,
        exposures_per_collection=images_to_average,
        collections_per_event=num_images,
        number_of_events=1,
    )

    def _per_shot(detectors, take_reading=None):
        # stage() at the start of each run clears what prepare() set
        for det in detectors:
            yield from bps.prepare(det, trigger_info, group="prepare")
        yield from bps.wait(group="prepare")
        for det in detectors:
            yield from bps.trigger(det, group="trigger")
        yield from bps.wait(group="trigger")
        yield from bps.create("primary")
        for det in detectors:
            yield from bps.read(det)
        yield from bps.save()

    # Each count is its own run, so each writes its own file.
    for i in range(num_iterations):
        if i > 0:
            yield from bps.sleep(time_gap)
        yield from bp.count(detectors, num=1, per_shot=_per_shot, md=md)


def xrd_count(
    detectors: Sequence[StandardDetector],
    exposure_time: float,
    num_images: int = 1,
    images_to_average: int = 1,
    num_iterations: int = 1,
    time_gap: float = 0.0,
    sample_name: str | None = None,
    description: str | None = None,
    use_shutter: bool = True,
    shutter: FastShutter | None = None,
    md: dict | None = None,
):
    """Take XRD images without moving anything.

    Parameters
    ----------
    detectors : list of detectors
        the detectors to read, e.g. ``[pe1c]``
    exposure_time : float
        exposure time per frame, in seconds
    num_images : int
        number of images saved per run
    images_to_average : int
        number of frames combined into each saved image (Proc1 on the Perkin Elmers)
    num_iterations : int
        number of runs to take, each its own file
    time_gap : float
        idle time between runs, in seconds
    sample_name : str, optional
        name of the sample
    description : str, optional
        description of the acquisition
    use_shutter : bool
        whether to open the fast shutter for the acquisition and close it after
    shutter : FastShutter, optional
        the fast shutter; the ``fs`` in the session if not given
    md : dict, optional
        extra metadata for each start document
    """
    if use_shutter:
        shutter = ensure_available(FastShutter, fs=shutter)

    def _body():
        for det in detectors:
            yield from ensure_cont_acq_exposure(det, exposure_time)
        if use_shutter:
            yield from bps.mv(shutter, "Open")
        yield from _count_runs(
            detectors,
            exposure_time,
            num_images,
            images_to_average,
            num_iterations,
            time_gap,
            _md("xrd_count", sample_name, description, md),
        )

    def _cleanup():
        if use_shutter:
            yield from bps.mv(shutter, "Close")

    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))


def xrd_dark(
    detectors: Sequence[StandardDetector],
    exposure_time: float,
    num_images: int = 1,
    images_to_average: int = 1,
    sample_name: str | None = None,
    description: str | None = None,
    shutter: FastShutter | None = None,
    md: dict | None = None,
):
    """Take XRD dark images with the fast shutter closed.

    Parameters
    ----------
    detectors : list of detectors
        the detectors to read, e.g. ``[pe1c]``
    exposure_time : float
        exposure time per frame, in seconds; use the one the data will use
    num_images : int
        number of dark images saved
    images_to_average : int
        number of frames combined into each saved image
    sample_name : str, optional
        name of the sample
    description : str, optional
        description of the acquisition
    shutter : FastShutter, optional
        the fast shutter, closed before the darks; the session's ``fs`` if not given
    md : dict, optional
        extra metadata for the start document
    """
    shutter = ensure_available(FastShutter, fs=shutter)
    yield from bps.mv(shutter, "Close")
    for det in detectors:
        yield from ensure_cont_acq_exposure(det, exposure_time)
    return (
        yield from _count_runs(
            detectors,
            exposure_time,
            num_images,
            images_to_average,
            1,
            0.0,
            _md("xrd_dark", sample_name, description, md),
        )
    )


def xrd_scan(
    detectors: Sequence[StandardDetector],
    motor: Steppable,
    start: float,
    stop: float,
    num_points: int,
    exposure_time: float,
    images_to_average: int = 1,
    num_iterations: int = 1,
    time_between_iterations: float = 0.0,
    sample_name: str | None = None,
    description: str | None = None,
    reset_position: bool = True,
    use_shutter: bool = True,
    shutter: FastShutter | None = None,
    md: dict | None = None,
):
    """Step one motor and take an XRD image at each point.

    Parameters
    ----------
    detectors : list of detectors
        the detectors to read, e.g. ``[pe1c]``
    motor : Motor
        the motor to step
    start : float
        first motor position
    stop : float
        last motor position
    num_points : int
        number of points, including start and stop
    exposure_time : float
        exposure time per frame, in seconds
    images_to_average : int
        number of frames combined into the image saved at each point
    num_iterations : int
        number of times to repeat the scan, each its own run
    time_between_iterations : float
        idle time between repeats, in seconds
    sample_name : str, optional
        name of the sample
    description : str, optional
        description of the scan
    reset_position : bool
        whether to move the motor back to ``start`` afterwards
    use_shutter : bool
        whether to open the fast shutter for the scan and close it after
    shutter : FastShutter, optional
        the fast shutter; the ``fs`` in the session if not given
    md : dict, optional
        extra metadata for each start document
    """
    if use_shutter:
        shutter = ensure_available(FastShutter, fs=shutter)
    trigger_info = TriggerInfo(
        trigger=DetectorTrigger.INTERNAL,
        livetime=exposure_time,
        exposures_per_collection=images_to_average,
    )

    to_prepare = list(detectors)

    # bluesky's scan_nd checks these parameter names, so they must stay as they are.
    def _per_step(
        detectors: Sequence[Readable],
        step: Mapping[Movable, Any],
        pos_cache: dict[Movable, Any],
        take_reading: bps.TakeReading | None = None,
    ):
        # stage() at the start of each run clears what prepare() set
        for det in to_prepare:
            yield from bps.prepare(det, trigger_info, wait=True)
        return (
            yield from bps.one_nd_step(
                detectors,
                step,
                pos_cache,
                take_reading=take_reading or bps.trigger_and_read,
            )
        )

    scan_md = _md("xrd_scan", sample_name, description, md)

    def _body():
        for det in detectors:
            yield from ensure_cont_acq_exposure(det, exposure_time)
        if use_shutter:
            yield from bps.mv(shutter, "Open")
        yield from bp.scan(
            list(detectors),
            motor,
            start,
            stop,
            num_points,
            per_step=_per_step,
            md=scan_md,
        )

    def _cleanup():
        if use_shutter:
            yield from bps.mv(shutter, "Close")
        if reset_position:
            yield from bps.mv(motor, start)

    return (
        yield from bps.repeat(
            lambda: bpp.finalize_wrapper(_body(), _cleanup()),
            num=num_iterations,
            delay=time_between_iterations,
        )
    )


def xrd_jog(
    detectors: Sequence[StandardDetector],
    motor: Steppable,
    start: float,
    stop: float,
    exposure_time: float,
    images_to_average: int = 1,
    num_images: int = 1,
    sample_name: str | None = None,
    description: str | None = None,
    use_shutter: bool = True,
    shutter: FastShutter | None = None,
    md: dict | None = None,
):
    """Take XRD images while sweeping one motor across the sample.

    Each image is taken while the motor travels from one end to the other, at
    the speed that makes the sweep last the whole image (``exposure_time`` x
    ``images_to_average``); the next image sweeps back. As pdf-profile-collection
    ``jog``/``rocking_ct`` did; the motor's velocity is restored afterwards.

    Parameters
    ----------
    detectors : list of detectors
        the detectors to read, e.g. ``[pe1c]``
    motor : Motor
        the motor to sweep, e.g. a sample stage axis
    start : float
        one end of the sweep
    stop : float
        the other end; equal to ``start`` takes images without moving
    exposure_time : float
        exposure time per frame, in seconds
    images_to_average : int
        number of frames combined into each saved image
    num_images : int
        number of images, one sweep each
    sample_name : str, optional
        name of the sample
    description : str, optional
        description of the acquisition
    use_shutter : bool
        whether to open the fast shutter for the acquisition and close it after
    shutter : FastShutter, optional
        the fast shutter; the session's ``fs`` if not given
    md : dict, optional
        extra metadata for the start document
    """
    if use_shutter:
        shutter = ensure_available(FastShutter, fs=shutter)
    trigger_info = TriggerInfo(
        trigger=DetectorTrigger.INTERNAL,
        livetime=exposure_time,
        exposures_per_collection=images_to_average,
    )
    sweep_time = exposure_time * images_to_average
    ends = [start, stop]
    to_prepare = list(detectors)
    jog_md = _md("xrd_jog", sample_name, description, md)
    jog_md["jog"] = {"motor": motor.name, "start": start, "stop": stop}

    # bluesky's count passes the detectors as the first argument, by this name.
    def _per_shot(detectors, take_reading=None):
        # stage() at the start of the run clears what prepare() set
        for det in to_prepare:
            yield from bps.prepare(det, trigger_info, wait=True)
        if start == stop:
            return (yield from bps.trigger_and_read(list(detectors)))
        here, there = ends
        yield from bps.mv(motor, here)
        yield from bps.mv(motor.velocity, abs(there - here) / sweep_time)
        group = short_uid("jog")
        yield from bps.abs_set(motor, there, group=group)
        yield from bps.trigger_and_read(list(detectors))
        yield from bps.wait(group=group)
        ends.reverse()

    def _body():
        for det in detectors:
            yield from ensure_cont_acq_exposure(det, exposure_time)
        if use_shutter:
            yield from bps.mv(shutter, "Open")
        yield from bp.count(detectors, num=num_images, per_shot=_per_shot, md=jog_md)

    def _cleanup():
        if use_shutter:
            yield from bps.mv(shutter, "Close")
        yield from bps.mv(motor.velocity, velocity)

    velocity = yield from bps.rd(motor.velocity)
    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))


def xrd_temperature_series(
    detectors: Sequence[StandardDetector],
    controller: TemperatureController,
    temperatures: Sequence[float],
    exposure_time: float,
    num_images: int = 1,
    images_to_average: int = 1,
    num_iterations: int = 1,
    time_gap: float = 0.0,
    ramp_rate: float | None = None,
    sample_name: str | None = None,
    description: str | None = None,
    use_shutter: bool = True,
    shutter: FastShutter | None = None,
    md: dict | None = None,
):
    """Take XRD images at each of a list of temperatures.

    For each temperature the controller is moved, which waits until its reading
    has settled there (set ``tolerance`` and ``settle_time`` on the controller,
    e.g. ``eurotherm.loop1.tolerance = 0.5``), then the images are taken as in
    ``xrd_count``. The shutter is closed while the temperature changes. Replaces
    pdf-profile-collection ``pbar_set_temp`` followed by a count.

    Parameters
    ----------
    detectors : list of detectors
        the detectors to read, e.g. ``[pe1c]``
    controller : temperature controller
        the loop to set, e.g. ``eurotherm.loop1``
    temperatures : list of float
        the setpoints, in the order to visit them
    exposure_time : float
        exposure time per frame, in seconds
    num_images : int
        number of images saved per run
    images_to_average : int
        number of frames combined into each saved image
    num_iterations : int
        number of runs at each temperature, each its own file
    time_gap : float
        idle time between runs at one temperature, in seconds
    ramp_rate : float, optional
        ramp rate to set once before the first temperature, in the controller's
        units; left as it is if not given
    sample_name : str, optional
        name of the sample
    description : str, optional
        description of the acquisition
    use_shutter : bool
        whether to open the fast shutter for each count and close it after
    shutter : FastShutter, optional
        the fast shutter; the session's ``fs`` if not given
    md : dict, optional
        extra metadata for each start document
    """
    if use_shutter:
        shutter = ensure_available(FastShutter, fs=shutter)

    def _body():
        for det in detectors:
            yield from ensure_cont_acq_exposure(det, exposure_time)
        if ramp_rate is not None:
            yield from bps.mv(controller.ramp_rate, ramp_rate)
        for setpoint in temperatures:
            yield from bps.mv(controller, setpoint)
            readback = yield from bps.rd(controller.readback)
            run_md = _md("xrd_temperature_series", sample_name, description, md)
            run_md["temperature"] = {
                "controller": controller.name,
                "setpoint": setpoint,
                "readback": readback,
            }
            if use_shutter:
                yield from bps.mv(shutter, "Open")
            yield from _count_runs(
                detectors,
                exposure_time,
                num_images,
                images_to_average,
                num_iterations,
                time_gap,
                run_md,
            )
            if use_shutter:
                yield from bps.mv(shutter, "Close")

    def _cleanup():
        if use_shutter:
            yield from bps.mv(shutter, "Close")

    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))
