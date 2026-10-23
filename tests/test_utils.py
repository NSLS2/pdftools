import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from bluesky.run_engine import RunEngine
from ophyd_async.core import Device, NotConnectedError, soft_signal_rw

import pdftools.utils as utils
from pdftools.utils import (
    auto_init_devices,
    initialize_run_engine,
    is_running_in_ci,
    print_proposal_info,
    remove_md_keys,
)

PROFILE = Path(__file__).parents[1] / "src" / "pdftools" / "profiles" / "collection.py"


@pytest.mark.parametrize(
    "value, expected",
    [("YES", True), ("true", True), ("1", True), ("false", False), ("no", False)],
)
def test_is_running_in_ci(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool):
    monkeypatch.setenv(utils.CI_ENV_VAR, value)
    assert is_running_in_ci() is expected


def test_initialize_run_engine_mock_proposal():
    RE = initialize_run_engine()
    try:
        assert RE.md["data_session"] == "pass-123456"
        assert set(RE.md["proposal"]) == {"title", "type", "pi_name"}
    finally:
        RE._loop.call_soon_threadsafe(RE._loop.stop)
        RE._th.join()
        RE._loop.close()


def test_initialize_run_engine_uses_redis_outside_ci(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(utils.CI_ENV_VAR, "false")
    calls = []

    def fake_open_redis_client(**kwargs):
        calls.append(kwargs)
        return "client"

    monkeypatch.setattr(utils, "open_redis_client", fake_open_redis_client)
    monkeypatch.setattr(
        utils, "RedisJSONDict", lambda client, prefix: {"client": client}
    )
    RE = initialize_run_engine()
    try:
        assert calls == [{"redis_ssl": True}]
        assert RE.md["client"] == "client"
    finally:
        RE._loop.call_soon_threadsafe(RE._loop.stop)
        RE._th.join()
        RE._loop.close()


def test_print_proposal_info(capsys: pytest.CaptureFixture[str]):
    print_proposal_info(
        {
            "proposal": {
                "title": "Powder study",
                "type": "General User",
                "pi_name": "A. B",
            }
        }
    )
    out = capsys.readouterr().out
    assert "Powder study" in out
    assert "General User" in out
    assert "A. B" in out


def test_print_proposal_info_keeps_markup_like_text(capsys: pytest.CaptureFixture[str]):
    print_proposal_info(
        {
            "proposal": {
                "title": "Study [in situ]",
                "type": "General [/bold] User",
                "pi_name": "A. B",
            }
        }
    )
    out = capsys.readouterr().out
    assert "Study [in situ]" in out
    assert "General [/bold] User" in out


def test_print_proposal_info_without_proposal(capsys: pytest.CaptureFixture[str]):
    print_proposal_info({})
    assert capsys.readouterr().out == ""


def test_remove_md_keys_removes_only_matching_keys():
    md = {"PDF:bt_safN": 1, "x_PDF:y": 2, "scan_id": 3, "data_session": "pass-1"}
    assert sorted(remove_md_keys(md, prefix="PDF:")) == ["PDF:bt_safN", "x_PDF:y"]
    assert md == {"scan_id": 3, "data_session": "pass-1"}


class _Box(Device):
    def __init__(self, name: str = ""):
        self.value = soft_signal_rw(float)
        super().__init__(name=name)


class _Unreachable(Device):
    def __init__(self, name: str = ""):
        self.value = soft_signal_rw(float)
        super().__init__(name=name)

    async def connect(self, mock=False, timeout=10.0, force_reconnect=False):
        raise NotConnectedError({"value": NotConnectedError("no IOC")})


def test_auto_init_devices_names_and_reports(
    RE: RunEngine, capsys: pytest.CaptureFixture[str]
):
    with auto_init_devices(timeout=0.1):
        box = _Box()
        gone = _Unreachable()

    assert box.name == "box"
    assert gone.name == "gone"
    out = capsys.readouterr().out
    assert "box" in out and "OK" in out
    assert "gone" in out and "FAILED" in out


def test_profile_boots_in_mock_mode(tmp_path: Path):
    env = {k: v for k, v in os.environ.items() if k != utils.CI_ENV_VAR}
    result = subprocess.run(
        [sys.executable, str(PROFILE), "--mock"],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "Mock mode proposal" in result.stdout
    assert "Connecting to beamline devices" in result.stdout
    for name in ("lambda1", "pilatus1", "pe1c", "pe2c"):
        assert re.search(rf"{re.escape(name)} \.+ \[\s*OK\s*\]", result.stdout), name
