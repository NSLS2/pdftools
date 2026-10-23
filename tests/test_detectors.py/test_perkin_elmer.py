import datetime
from pathlib import PurePosixPath, PureWindowsPath

import bluesky.plans as bp
import pytest
from bluesky.run_engine import RunEngine
from nslsii.ophyd_async.providers import NSLS2PathProvider
from ophyd_async.core import (
    callback_on_mock_put,
    get_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.adcore import (
    ADBaseDataType,
    ADImageMode,
    ADWriterFactory,
    ContAcqDetector,
)


def test_perkin_elmer_count_sums_in_proc_and_writes_tiff_to_windows_share(
    RE: RunEngine, monkeypatch: pytest.MonkeyPatch
):
    # As the profile sets it: a count keeps the frame count summed in the IOC's Proc1.
    monkeypatch.setenv("OPHYD_ASYNC_PRESERVE_DETECTOR_STATE", "YES")
    RE.md.update({"cycle": "2026-3", "data_session": "pass-123456"})
    path_provider = NSLS2PathProvider(
        RE.md,
        base_data_dir=PurePosixPath("/nsls2/data/pdf/proposals"),
        base_write_dir=PureWindowsPath("J:\\proposals"),
        separator="/",
    )
    with init_devices(mock=True):
        pe = ContAcqDetector(
            "TEST:PE2:",
            ADWriterFactory.tiff(path_provider),
            name="pe2",
            proc_suffix="Proc1:",
        )
    set_mock_value(pe.driver.image_mode, ADImageMode.CONTINUOUS)
    set_mock_value(pe.driver.acquire, True)
    set_mock_value(pe.driver.array_size_x, 4)
    set_mock_value(pe.driver.array_size_y, 3)
    set_mock_value(pe.driver.data_type, ADBaseDataType.UINT16)
    set_mock_value(pe.tiff.file_path_exists, True)  # pyright: ignore[reportAttributeAccessIssue]
    set_mock_value(pe.proc.enable_filter, True)  # pyright: ignore[reportAttributeAccessIssue]
    set_mock_value(pe.proc.num_filter, 5)  # pyright: ignore[reportAttributeAccessIssue]

    def _on_cb_trigger(value, **kwargs):
        if value:
            set_mock_value(pe.tiff.num_captured, 1)  # pyright: ignore[reportAttributeAccessIssue]
        return False

    callback_on_mock_put(pe.cb.trigger_, _on_cb_trigger)  # pyright: ignore[reportAttributeAccessIssue]

    docs: dict[str, list] = {}
    RE(bp.count([pe]), lambda name, doc: docs.setdefault(name, []).append(doc))

    written_to = get_mock_put(pe.tiff.file_path).call_args.args[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert written_to.startswith("J:\\proposals\\2026-3\\pass-123456\\")
    sres = docs["stream_resource"][0]
    assert sres["mimetype"] == "multipart/related;type=image/tiff"
    # The read side is POSIX: no Windows separators may leak in from the write root.
    today = datetime.date.today()
    assert sres["uri"] == (
        "file://localhost/nsls2/data/pdf/proposals/2026-3/pass-123456/"
        f"assets/pe2/{today:%Y/%m/%d}/"
    )
    get_mock_put(pe.proc.num_filter).assert_called_with(5)  # pyright: ignore[reportAttributeAccessIssue]
    desc = docs["descriptor"][0]["data_keys"]["pe2"]
    assert desc["shape"] == [1, 3, 4]
    assert desc["dtype_numpy"] == "<u2"
