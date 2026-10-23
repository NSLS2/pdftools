from bluesky.run_engine import RunEngine
from ophyd_async.core import init_devices

from pdftools.eurotherm3k import Eurotherm3k

# Records listed in pdf-profile-collection 16-eurotherm_HAB.py for XF:28ID1-ES{ET:05}
OLD_PROFILE_LOOP1 = {
    "setpoint": "LOOP1:SP",
    "readback": "LOOP1:PV:RBV",
    "working_setpoint": "LOOP1:WSP:RBV",
    "ramp_rate": "LOOP1:RR",
    "output": "LOOP1:O",
    "autotune": "LOOP1:AUTOTUNE",
}


def test_pdf_eurotherm_pvs_match_the_old_profile(RE: RunEngine):
    with init_devices(mock=True):
        eurotherm = Eurotherm3k("XF:28ID1-ES{ET:05}", name="eurotherm")
    for attr, suffix in OLD_PROFILE_LOOP1.items():
        source = getattr(eurotherm.loop1, attr).source
        assert source.startswith(f"mock+ca://XF:28ID1-ES{{ET:05}}{suffix}"), source
    assert eurotherm.disable.source == "mock+ca://XF:28ID1-ES{ET:05}DISABLE"
