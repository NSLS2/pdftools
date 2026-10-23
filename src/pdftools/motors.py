"""Motion device classes for the PDF beamline."""

from ophyd_async.epics.core import EpicsDevice, epics_signal_r
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor


class XYStage(EpicsDevice):
    """Two-axis (X, Y) motor stage."""

    def __init__(self, prefix: str, name: str):
        self.x = AsyncEpicsMotor(prefix + "X}Mtr", name="x")
        self.y = AsyncEpicsMotor(prefix + "Y}Mtr", name="y")
        super().__init__(prefix, name=name)


class XYZStage(XYStage):
    """Three-axis (X, Y, Z) motor stage."""

    def __init__(self, prefix: str, name: str):
        self.z = AsyncEpicsMotor(prefix + "Z}Mtr", name="z")
        super().__init__(prefix, name=name)


class XYZSpinner(XYZStage):
    """X, Y, Z stage with a rotation (Ry) axis."""

    def __init__(self, prefix: str, name: str):
        self.ry = AsyncEpicsMotor(prefix + "Ry}Mtr", name="ry")
        super().__init__(prefix, name=name)


class OCMTable(EpicsDevice):
    """OCM table: X plus upstream and downstream Y jacks."""

    def __init__(self, prefix: str, name: str):
        self.x = AsyncEpicsMotor(prefix + "X}Mtr", name="x")
        self.upstream_y = AsyncEpicsMotor(prefix + "YU}Mtr", name="upstream_y")
        self.downstream_y = AsyncEpicsMotor(prefix + "YD}Mtr", name="downstream_y")
        super().__init__(prefix, name=name)


class Slits(EpicsDevice):
    """Four-blade slits (top, bottom, inboard, outboard)."""

    def __init__(self, prefix: str, name: str):
        self.top = AsyncEpicsMotor(prefix + "T}Mtr", name="top")
        self.bottom = AsyncEpicsMotor(prefix + "B}Mtr", name="bottom")
        self.inboard = AsyncEpicsMotor(prefix + "L}Mtr", name="inboard")
        self.outboard = AsyncEpicsMotor(prefix + "R}Mtr", name="outboard")
        super().__init__(prefix, name=name)


class SideBounceMono(EpicsDevice):
    """Side-bounce monochromator: wedge-mount translations, angles and benders."""

    def __init__(self, prefix: str, name: str):
        self.x_wedgemount = AsyncEpicsMotor(prefix + "X}Mtr", name="x_wedgemount")
        self.y_wedgemount = AsyncEpicsMotor(prefix + "Y}Mtr", name="y_wedgemount")
        self.yaw = AsyncEpicsMotor(prefix + "Yaw}Mtr", name="yaw")
        self.pitch = AsyncEpicsMotor(prefix + "Pitch}Mtr", name="pitch")
        self.roll = AsyncEpicsMotor(prefix + "Roll}Mtr", name="roll")
        self.bend_inboard_u = AsyncEpicsMotor(prefix + "IU}Mtr", name="bend_inboard_u")
        self.bend_inboard_l = AsyncEpicsMotor(prefix + "IL}Mtr", name="bend_inboard_l")
        self.bend_outboard_u = AsyncEpicsMotor(
            prefix + "OU}Mtr", name="bend_outboard_u"
        )
        self.bend_outboard_l = AsyncEpicsMotor(
            prefix + "OL}Mtr", name="bend_outboard_l"
        )
        super().__init__(prefix, name=name)


class Mirror(EpicsDevice):
    """Mirror: Y jacks and benders, with bend and twist encoder readbacks."""

    def __init__(self, prefix: str, name: str):
        self.y_upstream = AsyncEpicsMotor(prefix + "YU}Mtr", name="y_upstream")
        self.y_downstream_inboard = AsyncEpicsMotor(
            prefix + "YDI}Mtr", name="y_downstream_inboard"
        )
        self.y_downstream_outboard = AsyncEpicsMotor(
            prefix + "YDO}Mtr", name="y_downstream_outboard"
        )
        self.bend_upstream = AsyncEpicsMotor(prefix + "BndU}Mtr", name="bend_upstream")
        self.bend_downstream = AsyncEpicsMotor(
            prefix + "BndD}Mtr", name="bend_downstream"
        )

        # Encoder readouts for bend and twist
        self.bend_encoder = epics_signal_r(
            float, prefix + "BndU}Pos:Enc-I", name="bend_encoder"
        )
        self.twist_encoder = epics_signal_r(
            float, prefix + "BndD}Pos:Enc-I", name="twist_encoder"
        )
        super().__init__(prefix, name=name)


class OpticsTableADC(EpicsDevice):
    """Optics table: Y jacks, X translations and Z."""

    def __init__(self, prefix: str, name: str):
        self.y_upstream_inboard = AsyncEpicsMotor(
            prefix + "YUI}Mtr", name="y_upstream_inboard"
        )
        self.y_upstream_outboard = AsyncEpicsMotor(
            prefix + "YUO}Mtr", name="upstream_jack_outboard"
        )
        self.y_downstream_outboard = AsyncEpicsMotor(
            prefix + "YD}Mtr", name="y_downstream_outboard"
        )
        self.x_upstream = AsyncEpicsMotor(prefix + "XU}Mtr", name="x_upstream")
        self.x_downstream = AsyncEpicsMotor(prefix + "XD}Mtr", name="x_downstream")
        self.z = AsyncEpicsMotor(prefix + "Z}Mtr", name="z")
        super().__init__(prefix, name=name)
