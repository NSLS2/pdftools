"""XRD plans against a mocked Perkin Elmer (conftest ``pe``) and fast shutter."""

import asyncio
from typing import Any

import pytest
from bluesky import plan_stubs as bps
from bluesky.run_engine import RunEngine
from ophyd_async.core import (
    callback_on_mock_put,
    get_mock_put,
    init_devices,
    set_mock_put_proceeds,
    set_mock_value,
)
from ophyd_async.epics.adcore import ADImageMode, ContAcqDetector
from ophyd_async.sim import SimMotor

from pdftools.shutter import FastShutter, OnOff
from pdftools.xrd import ensure_cont_acq_exposure, xrd_count, xrd_dark, xrd_scan


@pytest.fixture
def fs(RE: RunEngine) -> FastShutter:
    with init_devices(mock=True):
        fs = FastShutter("FS:", name="fs", settle_time=0)
    set_mock_value(fs.status, OnOff.ON)  # closed
    callback_on_mock_put(fs.cmd, lambda value, **_: set_mock_value(fs.status, value))
    return fs


def _commands(fs: FastShutter) -> list[OnOff]:
    return [c.args[0] for c in get_mock_put(fs.cmd).call_args_list]


def _collect(RE: RunEngine, plan) -> dict[str, list[Any]]:
    docs: dict[str, list[Any]] = {}
    RE(plan, lambda name, doc: docs.setdefault(name, []).append(doc))
    return docs


def test_ensure_cont_acq_exposure_leaves_a_matching_detector_alone(
    RE: RunEngine, pe: ContAcqDetector
):
    RE(ensure_cont_acq_exposure(pe, 0.1))
    get_mock_put(pe.driver.acquire).assert_not_called()
    get_mock_put(pe.driver.acquire_time).assert_not_called()


async def test_ensure_cont_acq_exposure_restarts_at_a_new_exposure(
    RE: RunEngine, pe: ContAcqDetector
):
    RE(ensure_cont_acq_exposure(pe, 0.5))
    assert [c.args[0] for c in get_mock_put(pe.driver.acquire).call_args_list] == [
        False,
        True,
    ]
    assert await pe.driver.acquire_time.get_value() == 0.5
    assert await pe.driver.image_mode.get_value() == ADImageMode.CONTINUOUS


async def test_xrd_count_opens_the_shutter_and_takes_each_run(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter
):
    docs = _collect(
        RE,
        xrd_count(
            [pe],
            0.2,
            num_images=3,
            images_to_average=5,
            num_iterations=2,
            sample_name="Ni",
            shutter=fs,
        ),
    )
    assert [d["plan_name"] for d in docs["start"]] == ["xrd_count", "xrd_count"]
    assert all(d["sample_name"] == "Ni" for d in docs["start"])
    assert len(docs["event"]) == 2
    assert docs["descriptor"][0]["data_keys"]["pe1"]["shape"][0] == 3
    assert await pe.driver.acquire_time.get_value() == 0.2
    assert await pe.proc.num_filter.get_value() == 5  # pyright: ignore[reportAttributeAccessIssue]
    # opened for the acquisition, closed after it (OFF opens the PDF fast shutter)
    assert _commands(fs) == [OnOff.OFF, OnOff.ON]


async def test_xrd_count_without_shutter_leaves_it_alone(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter
):
    RE(xrd_count([pe], 0.1, use_shutter=False, shutter=fs))
    get_mock_put(fs.cmd).assert_not_called()


async def test_xrd_dark_closes_the_shutter_and_never_opens_it(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter
):
    set_mock_value(fs.status, OnOff.OFF)  # left open
    docs = _collect(RE, xrd_dark([pe], 0.1, num_images=4, shutter=fs))
    assert [d["plan_name"] for d in docs["start"]] == ["xrd_dark"]
    assert _commands(fs) == [OnOff.ON]
    assert await fs.status.get_value() == OnOff.ON


async def test_xrd_scan_takes_one_averaged_image_per_point(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter
):
    with init_devices():
        det_z = SimMotor(name="det_z", instant=True)
    docs = _collect(
        RE,
        xrd_scan(
            [pe],
            det_z,
            0.0,
            4.0,
            5,
            0.1,
            images_to_average=3,
            num_iterations=2,
            shutter=fs,
        ),
    )
    assert [d["plan_name"] for d in docs["start"]] == ["xrd_scan", "xrd_scan"]
    positions = [e["data"]["det_z"] for e in docs["event"]]
    assert positions == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0] * 2)
    assert await pe.proc.num_filter.get_value() == 3  # pyright: ignore[reportAttributeAccessIssue]
    assert await det_z.user_readback.get_value() == 0.0
    assert await fs.status.get_value() == OnOff.ON


async def test_xrd_scan_closes_the_shutter_and_returns_the_motor_on_failure(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter
):
    with init_devices():
        det_z = SimMotor(name="det_z", instant=True)
    tiff, cb = pe.tiff, pe.cb  # pyright: ignore[reportAttributeAccessIssue]
    triggers: list[bool] = []

    async def _fault_on_second_point(value):
        if not value:
            return
        triggers.append(value)
        if len(triggers) == 2:
            raise RuntimeError("detector fault")
        set_mock_value(tiff.num_captured, await tiff.num_captured.get_value() + 1)
        asyncio.get_running_loop().call_later(0.01, set_mock_value, cb.trigger_, False)

    callback_on_mock_put(cb.trigger_, _fault_on_second_point)
    with pytest.raises(Exception, match="detector fault"):
        RE(xrd_scan([pe], det_z, 1.0, 3.0, 3, 0.1, shutter=fs))
    assert await det_z.user_readback.get_value() == 1.0
    assert await fs.status.get_value() == OnOff.ON


def _busy_acquire(pe: ContAcqDetector, timed_out: list[bool]) -> None:
    """Acquire as the IOC runs it: the readback follows after a delay, and the put
    completes only when acquisition stops."""
    acquire = pe.driver.acquire

    def _release_after_timeout():
        timed_out.append(True)
        set_mock_put_proceeds(acquire, True)

    def _on_put(value, **_):
        if not value:
            set_mock_put_proceeds(acquire, True)
            return None
        set_mock_put_proceeds(acquire, False)
        loop = asyncio.get_running_loop()
        loop.call_later(0.2, set_mock_value, acquire, True)
        # bound the test if the plan waits on the put rather than the readback
        loop.call_later(15.0, _release_after_timeout)
        return False  # the readback has not caught up yet

    callback_on_mock_put(acquire, _on_put)


async def test_ensure_cont_acq_exposure_returns_once_the_detector_is_acquiring(
    RE: RunEngine, pe: ContAcqDetector
):
    timed_out: list[bool] = []
    acquiring: list[bool] = []
    _busy_acquire(pe, timed_out)

    def _plan():
        yield from ensure_cont_acq_exposure(pe, 0.3)
        acquiring.append((yield from bps.rd(pe.driver.acquire)))

    RE(_plan())
    RE.loop.call_soon_threadsafe(set_mock_put_proceeds, pe.driver.acquire, True)
    assert acquiring == [True]
    assert timed_out == []


async def test_xrd_scan_after_a_restart_waits_for_the_acquire_readback_not_its_put(
    RE: RunEngine, pe: ContAcqDetector, fs: FastShutter
):
    timed_out: list[bool] = []
    _busy_acquire(pe, timed_out)
    with init_devices():
        det_z = SimMotor(name="det_z", instant=True)
    docs = _collect(RE, xrd_scan([pe], det_z, 0.0, 1.0, 2, 0.3, shutter=fs))
    RE.loop.call_soon_threadsafe(set_mock_put_proceeds, pe.driver.acquire, True)
    assert len(docs["event"]) == 2
    assert timed_out == []
    assert await pe.driver.acquire_time.get_value() == 0.3


def test_the_shutter_comes_from_the_session_when_not_given(
    RE: RunEngine, pe: ContAcqDetector, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("pdftools.utils.get_obj_from_ipython_ns", lambda *_: None)
    with pytest.raises(ValueError, match="fs"):
        RE(xrd_count([pe], 0.1))
