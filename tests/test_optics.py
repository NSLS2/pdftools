"""PV names of the optics classes must match the beamline records.

Expected suffixes are pdf-profile-collection 12-optics.py; the slit blades (T/B/I/O)
were also checked with cainfo on 28ID1A and 28ID1B and against cs-studio-xf
common/op/slits_4ax.opi (2026-10-23).
"""

import pytest
from ophyd_async.epics.motor import Motor

from pdftools.motors import Mirror, OCMTable, OpticsTableADC, SideBounceMono, Slits

PREFIX = "XF:TEST{Dev-Ax:"

CASES = {
    Slits: {"top": "T", "bottom": "B", "inboard": "I", "outboard": "O"},
    SideBounceMono: {
        "x_wedgemount": "X",
        "y_wedgemount": "Y",
        "yaw": "Yaw",
        "pitch": "Pitch",
        "roll": "Roll",
        "bend_inboard_u": "IU",
        "bend_inboard_l": "IL",
        "bend_outboard_u": "OU",
        "bend_outboard_l": "OL",
    },
    Mirror: {
        "y_upstream": "YU",
        "y_downstream_inboard": "YDI",
        "y_downstream_outboard": "YDO",
        "bend_upstream": "BndU",
        "bend_downstream": "BndD",
    },
    OpticsTableADC: {
        "y_upstream_inboard": "YUI",
        "y_upstream_outboard": "YUO",
        "y_downstream_outboard": "YD",
        "x_upstream": "XU",
        "x_downstream": "XD",
        "z": "Z",
    },
    OCMTable: {"x": "X", "upstream_y": "YU", "downstream_y": "YD"},
}


@pytest.mark.parametrize(
    ("cls", "axes"), CASES.items(), ids=[c.__name__ for c in CASES]
)
def test_motor_pvs_match_the_beamline_records(cls, axes):
    device = cls(PREFIX, name="device")
    motors = {n: c for n, c in device.children() if isinstance(c, Motor)}
    assert {n: m.user_setpoint.source for n, m in motors.items()} == {
        n: f"ca://{PREFIX}{suffix}}}Mtr.VAL" for n, suffix in axes.items()
    }


def test_mirror_encoder_pvs():
    mirror = Mirror(PREFIX, name="mirror")
    assert mirror.bend_encoder.source == f"ca://{PREFIX}BndU}}Pos:Enc-I"
    assert mirror.twist_encoder.source == f"ca://{PREFIX}BndD}}Pos:Enc-I"
