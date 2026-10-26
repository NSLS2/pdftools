import asyncio
import pprint
import warnings
from pathlib import Path

# Pre-import websockets modules that trigger DeprecationWarnings so they are
# cached in sys.modules before uvicorn's server thread imports them.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        import uvicorn.protocols.websockets.auto  # noqa: F401
    except (ImportError, Exception):
        pass

from collections.abc import Generator

import pytest
from bluesky._vendor.super_state_machine.errors import TransitionError  # noqa: PLC2701
from bluesky.run_engine import RunEngine
from ophyd_async.core import (
    StaticPathProvider,
    UUIDFilenameProvider,
    callback_on_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.adcore import ADImageMode, ADWriterFactory, ContAcqDetector
from ophyd_async.sim import SimBlobDetector, SimPointDetector
from pytest import FixtureRequest
from tiled.client import from_uri
from tiled.client.container import Container
from tiled.server import SimpleTiledServer

from pdftools.lakeshore336 import LakeShore336, LakeShore336LoopLogic

_ALLOWED_PYTEST_TASKS = {"async_finalizer", "async_setup", "async_teardown"}


# ==================================================================================
# Copied from ophyd-async conftest.py
# ==================================================================================
def _error_and_kill_pending_tasks(
    loop: asyncio.AbstractEventLoop, test_name: str, test_passed: bool
) -> set[asyncio.Task]:
    """Cancels pending tasks in the event loop for a test. Raises an exception if
    the test hasn't already.

    Args:
        loop: The event loop to check for pending tasks.
        test_name: The name of the test.
        test_passed: Indicates whether the test passed.

    Returns:
        set[asyncio.Task]: The set of unfinished tasks that were cancelled.

    Raises:
        RuntimeError: If there are unfinished tasks and the test didn't fail.
    """
    unfinished_tasks = {
        task
        for task in asyncio.all_tasks(loop)
        if (coro := task.get_coro()) is not None
        and hasattr(coro, "__name__")
        and coro.__name__ not in _ALLOWED_PYTEST_TASKS
        and not task.done()
    }
    for task in unfinished_tasks:
        task.cancel()

    # We only raise an exception here if the test didn't fail anyway.
    # If it did then it makes sense that there's some tasks we need to cancel,
    # but an exception will already have been raised.
    if unfinished_tasks and test_passed:
        raise RuntimeError(
            f"Not all tasks closed during test {test_name}:\n"
            f"{pprint.pformat(unfinished_tasks, width=88)}"
        )

    return unfinished_tasks


@pytest.fixture(autouse=True, scope="function")
async def fail_test_on_unclosed_tasks(request: FixtureRequest):
    """Used on every test to ensure failure if there are pending tasks
    by the end of the test.
    """
    try:
        fail_count = request.session.testsfailed
        loop = asyncio.get_running_loop()

        loop.set_debug(True)

        request.addfinalizer(
            lambda: _error_and_kill_pending_tasks(
                loop, request.node.name, request.session.testsfailed == fail_count
            )
        )
    # Once https://github.com/bluesky/ophyd-async/issues/683
    # is finished we can remove this try, except.
    except RuntimeError as error:
        if str(error) != "no running event loop":
            raise error


@pytest.fixture(autouse=True)
def _set_ci_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PDFTOOLS_RUNNING_IN_CI", "YES")


@pytest.fixture(scope="function")
def RE(request: FixtureRequest) -> RunEngine:
    loop = asyncio.new_event_loop()
    loop.set_debug(True)
    RE = RunEngine(
        {"cycle": "2026-1", "data_session": "pass-123456"},
        call_returns_result=True,
        loop=loop,
    )
    fail_count = request.session.testsfailed

    async def _settle_in_flight_tasks():
        # A plan may end with a deliberate wait=False put (e.g. enable_preview_mode
        # starting Acquire); give it a moment to finish before checking for leaks.
        # On Python 3.11 asyncio.wait_for wraps it in an extra task, so it often
        # outlives the RE call. Anything still pending afterwards is a real leak.
        pending = asyncio.all_tasks() - {asyncio.current_task()}
        if pending:
            await asyncio.wait(pending, timeout=1.0)

    def clean_event_loop():
        if RE.state not in ("idle", "panicked"):
            try:
                RE.halt()
            except TransitionError:
                pass

        if loop.is_running():
            asyncio.run_coroutine_threadsafe(_settle_in_flight_tasks(), loop).result()
        loop.call_soon_threadsafe(loop.stop)
        RE._th.join()

        try:
            _error_and_kill_pending_tasks(
                loop, request.node.name, request.session.testsfailed == fail_count
            )
        finally:
            loop.close()

    request.addfinalizer(clean_event_loop)
    return RE


# ==================================================================================


@pytest.fixture
def tiled_client(tmp_path: Path) -> Generator[tuple[Path, Container], None, None]:
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    server = SimpleTiledServer(
        tmp_path / "tiled_data", readable_storage=[tmp_path / "data"]
    )
    client = from_uri(server.uri)
    yield tmp_path / "data", client
    client.context.close()
    server.close()


@pytest.fixture
def output_tiled_client(tmp_path: Path) -> Generator[Container, None, None]:
    server = SimpleTiledServer(tmp_path / "output_tiled_data")
    client = from_uri(server.uri)
    yield client
    client.context.close()
    server.close()


@pytest.fixture
def static_path_provider(tmp_path: Path) -> StaticPathProvider:
    return StaticPathProvider(UUIDFilenameProvider(), tmp_path)


@pytest.fixture
def sample_detectors_factory(RE: RunEngine):
    def _factory(write_path: Path) -> tuple[SimBlobDetector, SimPointDetector]:
        with init_devices():
            sim_point = SimPointDetector(None)
            sim_blob = SimBlobDetector(
                StaticPathProvider(UUIDFilenameProvider(), write_path)
            )
        return sim_blob, sim_point

    return _factory


@pytest.fixture
def pe(RE: RunEngine, static_path_provider: StaticPathProvider) -> ContAcqDetector:
    """A Perkin Elmer acquiring continuously at 0.1 s, as the profile builds it.

    The mock passes frames as the IOC does: one CB trigger releases ``PostCount``
    frames, Proc1 combines every ``NumFilter`` of them, and the TIFF writer
    captures what is left. A mock adding one frame per trigger only fits one
    image per event, and hid that until xrd_count asked for three.
    """
    with init_devices(mock=True):
        pe = ContAcqDetector(
            "PE1:",
            ADWriterFactory.tiff(static_path_provider),
            name="pe1",
            proc_suffix="Proc1:",
        )
    set_mock_value(pe.driver.image_mode, ADImageMode.CONTINUOUS)
    set_mock_value(pe.driver.acquire, True)
    set_mock_value(pe.driver.acquire_time, 0.1)
    set_mock_value(pe.driver.array_size_x, 4)
    set_mock_value(pe.driver.array_size_y, 3)
    tiff, cb, proc = pe.tiff, pe.cb, pe.proc  # pyright: ignore[reportAttributeAccessIssue]
    set_mock_value(tiff.file_path_exists, True)

    async def _buffer_frames(value):
        if not value:
            return
        frames = await cb.post_count.get_value()
        frames //= max(1, await proc.num_filter.get_value())
        set_mock_value(tiff.num_captured, await tiff.num_captured.get_value() + frames)
        asyncio.get_running_loop().call_later(0.01, set_mock_value, cb.trigger_, False)

    callback_on_mock_put(cb.trigger_, _buffer_frames)
    # each run opens a new file, which starts the count again
    callback_on_mock_put(tiff.capture, lambda _: set_mock_value(tiff.num_captured, 0))
    return pe


@pytest.fixture
def fast_lake(RE: RunEngine) -> LakeShore336:
    """The PDF Lake Shore, mocked; out1 settles in 0.05 s and follows its setpoint."""
    with init_devices(mock=True):
        lake = LakeShore336("XF:28ID1-ES{LS336:1", name="Lakeshore")
    logic = lake.out1.movable_logic
    assert isinstance(logic, LakeShore336LoopLogic)
    logic.settle_time = 0.05
    set_mock_value(lake.input_c.temp, 80.0)
    set_mock_value(lake.out1.ramp, 6.0)
    callback_on_mock_put(
        lake.out1.sp, lambda value, **_: set_mock_value(lake.input_c.temp, value)
    )
    return lake
