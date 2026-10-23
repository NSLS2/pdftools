from pathlib import Path

import bluesky.plans as bp
from bluesky.run_engine import RunEngine
from ophyd_async.core import (
    StaticFilenameProvider,
    StaticPathProvider,
    callback_on_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.adcore import ADBaseDataType, ADWriterFactory
from ophyd_async.epics.adpilatus import PilatusDetector


def test_pilatus_count_writes_tiff(RE: RunEngine, tmp_path: Path):
    with init_devices(mock=True):
        pilatus = PilatusDetector(
            "TEST:PILATUS:",
            ADWriterFactory.tiff(
                StaticPathProvider(StaticFilenameProvider("scan"), tmp_path)
            ),
            name="pilatus-1",
        )
    set_mock_value(pilatus.tiff.file_path_exists, True)  # pyright: ignore[reportAttributeAccessIssue]
    set_mock_value(pilatus.driver.array_size_x, 4)
    set_mock_value(pilatus.driver.array_size_y, 3)
    set_mock_value(pilatus.driver.data_type, ADBaseDataType.INT32)

    def _on_acquire(value, **kwargs):
        set_mock_value(pilatus.driver.armed, bool(value))
        if value:
            set_mock_value(pilatus.tiff.num_captured, 1)  # pyright: ignore[reportAttributeAccessIssue]

    callback_on_mock_put(pilatus.driver.acquire, _on_acquire)

    docs: dict[str, list] = {}
    RE(bp.count([pilatus]), lambda name, doc: docs.setdefault(name, []).append(doc))

    desc = docs["descriptor"][0]["data_keys"]["pilatus-1"]
    assert desc["shape"] == [1, 3, 4]
    assert desc["dtype_numpy"] == "<i4"
    sres = docs["stream_resource"][0]
    assert sres["mimetype"] == "multipart/related;type=image/tiff"
    assert sres["uri"].startswith(f"file://localhost{tmp_path}")
    assert docs["stream_datum"][0]["indices"] == {"start": 0, "stop": 1}
