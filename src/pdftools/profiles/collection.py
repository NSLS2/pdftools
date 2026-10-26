"""Bluesky profile for the 28-ID-1 PDF beamline at NSLS-II."""

import argparse
import os
from pathlib import PureWindowsPath

import numpy as np
from bluesky import plan_stubs as bps
from bluesky import plans as bp
from bluesky import preprocessors as bpp
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.magics import BlueskyMagics
from bluesky.run_engine import RunEngine, autoawait_in_bluesky_event_loop
from bluesky.utils import ProgressBarManager
from bluesky_tiled_plugins import TiledWriter
from IPython.core.getipython import get_ipython
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from nslsii.ophyd_async.providers import NSLS2PathProvider
from ophyd_async.epics.adcore import ADWriterFactory, ContAcqDetector
from ophyd_async.epics.adpilatus import PilatusDetector
from tiled.client import from_uri, simple

import pdftools
from pdftools.detectors import XSPDetector
from pdftools.eurotherm3k import Eurotherm3k
from pdftools.lakeshore336 import LakeShore336
from pdftools.motors import Mirror, OCMTable, OpticsTableADC, SideBounceMono, Slits
from pdftools.shutter import FastShutter, FilterBank
from pdftools.utils import (
    CI_ENV_VAR,
    ProposalIDPrompt,
    auto_init_devices,
    initialize_run_engine,
    is_running_in_ci,
    print_proposal_info,
    print_version_info,
    remove_md_keys,
)
from pdftools.xrd import (  # noqa: F401  for the session
    xrd_count,
    xrd_dark,
    xrd_jog,
    xrd_map,
    xrd_scan,
    xrd_temperature_series,
)

# Mechanism for passing arguments to profile.
_parser = argparse.ArgumentParser("Collection profile for pdftools.")
_parser.add_argument(
    "--mock",
    action="store_true",
    help="Run the profile in mock mode with simulated devices and metadata.",
)
if vars(_parser.parse_args())["mock"]:
    os.environ[CI_ENV_VAR] = "YES"
MOCK = is_running_in_ci()

# Beamline Redis for RunEngine metadata, as pdf-profile-collection's 00-base uses it
# (port 6380 with SSL is open_redis_client's default when redis_ssl=True).
os.environ.setdefault("REDIS_HOST", "xf28id1-pdf-redis1.nsls2.bnl.gov")
# NSLS2PathProvider derives /nsls2/data/<acronym>/proposals from this.
os.environ.setdefault("BEAMLINE_ACRONYM", "PDF")
# Have ophyd-async keep detector state constant when not explicitly overridden.
os.environ["OPHYD_ASYNC_PRESERVE_DETECTOR_STATE"] = "YES"

# Tiled catalog that pdf-profile-collection's 00-base writes to today.
TILED_URI = "https://tiled.nsls2.bnl.gov"
TILED_CATALOG = ("pdf", "migration")

print_version_info()

# Set up the RunEngine and its metadata.
RE: RunEngine = initialize_run_engine()
if not MOCK:
    # Trim the PDF: entries 00-base removes from Redis on every start.
    remove_md_keys(RE.md, prefix="PDF:")
RE.md["versions"]["pdftools"] = pdftools.__version__
RE.md["facility"] = "NSLS-II"
RE.md["group"] = "PDF"
RE.md["beamline_id"] = "28-ID-1"

# Display active proposal information in the console.
print_proposal_info(RE.md)

# Get the IPython shell and set up the custom prompt to show the current proposal ID.
ipython = get_ipython()
if ipython is not None and isinstance(ipython, TerminalInteractiveShell):
    ipython.prompts = ProposalIDPrompt(RE, ipython)
    BlueskyMagics.RE = RE
    ipython.register_magics(BlueskyMagics)
    autoawait_in_bluesky_event_loop()

    import matplotlib.pyplot as plt

    plt.ion()  # Enable interactive mode for matplotlib

    # Make tracebacks minimal by default to reduce clutter.
    # Use %tb verbose to view full traceback.
    ipython.run_line_magic("xmode", "minimal")

# Construct our tiled clients for writing and (in an interactive session) reading.
# In CI/mock mode, an in-process server stands in for both.
if not MOCK:
    tiled_writing_client = from_uri(
        TILED_URI,
        api_key=os.environ.get("TILED_BLUESKY_WRITING_API_KEY_PDF", ""),
    )[TILED_CATALOG[0]][TILED_CATALOG[1]]
    tiled_writing_client.context.http_client.headers["tiled-qos"] = "acquisition"
    if ipython is not None and isinstance(ipython, TerminalInteractiveShell):
        tiled_reading_client = c = from_uri(TILED_URI)[TILED_CATALOG[0]][
            TILED_CATALOG[1]
        ]
else:
    tiled_writing_client = tiled_reading_client = c = simple()

# Subscribe the tiled writer; documents it cannot send are kept on local disk.
RE.subscribe(TiledWriter(tiled_writing_client, backup_directory="/tmp/bluesky-docs"))

# Subscribe the best effort callback
bec = BestEffortCallback()
RE.subscribe(bec)

# Disable bec printout of baseline readings since it is kind of noisy
bec.disable_baseline()

# Configure the default terminal progress bar manager
RE.waiting_hook = ProgressBarManager()  # type: ignore

# Define our global default path provider for the beamline
path_provider = NSLS2PathProvider(RE.md)

# The Perkin Elmer IOCs run on Windows hosts and write to the proposals share as J:.
pe_path_provider = NSLS2PathProvider(
    RE.md,
    base_write_dir=PureWindowsPath("J:\\proposals"),
    # Else the Windows root sets "\\" for the read-side date dirs too.
    separator="/",
)

# Devices are added here, step by step (detectors, motors, sample environment).
with auto_init_devices(timeout=2.0):
    # Area detectors writing TIFF, as in pdf-profile-collection (81-pilatus, 84-lambda)
    lambda1 = XSPDetector(
        "XF:28ID1-ES{Lambda-Det:1}",
        ADWriterFactory.tiff(path_provider),
        name="lambda1",
    )
    # Assumed PILATUS3 (default readout_time); the model is not yet confirmed.
    pilatus1 = PilatusDetector(
        "XF:28ID1-ES{Det:Pilatus}",
        ADWriterFactory.tiff(path_provider),
        name="pilatus-1",
    )
    # Free-running flat panels; Proc1 sums frames per point, as pe1c/pe2c do today.
    pe1c = ContAcqDetector(
        "XF:28ID1-ES{Det:PE1}",
        ADWriterFactory.tiff(pe_path_provider),
        name="pe1",
        proc_suffix="Proc1:",
    )
    pe2c = ContAcqDetector(
        "XF:28ID1-ES{Det:PE2}",
        ADWriterFactory.tiff(pe_path_provider),
        name="pe2",
        proc_suffix="Proc1:",
    )
    # Fast shutter, as in pdf-profile-collection 12-optics (inverted Cmd/Sts records)
    fs = FastShutter("XF:28ID1B-OP{PSh:1-Det:2}", name="fs")
    # Filter bank: the old profile's fb_two_button_shutters (fb2), now just fb
    fb = FilterBank("XF:28ID1B-OP{Fltr:", name="fb")

    # Eurotherm 3504, as in pdf-profile-collection 16-eurotherm_HAB;
    # the old eurotherm3504 is loop1
    eurotherm = Eurotherm3k("XF:28ID1-ES{ET:05}", name="eurotherm")
    # Lake Shore 336 cryostat, as in pdf-profile-collection 101-Lakeshore336Cryostat
    lake = LakeShore336("XF:28ID1-ES{LS336:1", name="Lakeshore")

    # Optics, as in pdf-profile-collection 12-optics
    OCM_slits = Slits("XF:28ID1B-OP{Slt:2-Ax:", name="OCM_slits")
    BDM_slits = Slits("XF:28ID1A-OP{Slt:1-Ax:", name="BDM_slits")
    WB_slits = Slits("XF:28ID1A-OP{Slt:0-Ax:", name="WB_slits")
    cleanup_slits = Slits("XF:28ID1B-OP{Slt:AS-Ax:", name="cleanup_slits")
    sbm = SideBounceMono("XF:28ID1A-OP{Mono:SBM-Ax:", name="sbm")
    Mirror_VFM = Mirror("XF:28ID1A-OP{Mir:VFM-Ax:", name="Mirror_VFM")
    optics_table_adc = OpticsTableADC("XF:28ID1B-ES{Tbl:1-Ax:", name="optics_table_adc")
    OCM_table = OCMTable("XF:28ID1B-ES{OCM-Ax:", name="OCM_table")

# Outside init_devices, which would rename lake.out1 to "cryostat"
cryostat = lake.out1

# Add baseline=[<device1>, <device2>, ...] to SupplementalData as devices arrive.
sd = bpp.SupplementalData()
RE.preprocessors.append(sd)
