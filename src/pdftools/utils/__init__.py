"""Profile utilities for pdftools, following hextools.utils."""

import os
from collections.abc import MutableMapping
from datetime import datetime
from typing import Any

from bluesky.run_engine import RunEngine
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.terminal.prompts import Prompts
from nslsii.utils import open_redis_client
from ophyd_async.core import (
    Device,
    DeviceProcessor,
    NotConnectedError,
    Signal,
    wait_for_connection,
    walk_devices,
)
from pygments.token import Token
from redis_json_dict.redis_json_dict import RedisJSONDict
from rich import print as rprint
from rich.console import Console

CI_ENV_VAR = "PDFTOOLS_RUNNING_IN_CI"


def is_running_in_ci() -> bool:
    """Check whether pdftools runs in CI or mock mode.

    Returns
    -------
    bool
        True if ``PDFTOOLS_RUNNING_IN_CI`` is set to a true value.
    """
    return os.environ.get(CI_ENV_VAR, "false").lower() in ["true", "1", "yes"]


def print_version_info():
    """Print version information for bluesky, ophyd_async, tiled, and pdftools."""
    from bluesky import __version__ as bluesky_version
    from ophyd_async import __version__ as ophyd_async_version
    from tiled import __version__ as tiled_version

    from pdftools import __version__ as pdftools_version

    rprint("\n[bold]Version Information[/bold]")
    rprint(f"  [bold]bluesky[/bold]: [blue]{bluesky_version}[/blue]")
    rprint(f"  [bold]ophyd_async[/bold]: [blue]{ophyd_async_version}[/blue]")
    rprint(f"  [bold]tiled[/bold]: [blue]{tiled_version}[/blue]")
    rprint(f"  [bold]pdftools[/bold]: [blue]{pdftools_version}[/blue]\n")


class ProposalIDPrompt(Prompts):
    """Custom IPython prompt that shows the current proposal ID."""

    def __init__(self, RE: RunEngine, shell: TerminalInteractiveShell):
        super().__init__(shell)
        self._RE = RE

    def in_prompt_tokens(self, cli=None):
        """Return the prompt tokens: ``<data_session> [<n>]: ``."""
        return [
            (Token.Prompt, f"{self._RE.md.get('data_session', 'N/A')} ["),
            (Token.PromptNum, str(self.shell.execution_count)),
            (Token.Prompt, "]: "),
        ]


def initialize_run_engine() -> RunEngine:
    """Create the RunEngine, with Redis-backed metadata outside CI/mock mode.

    Returns
    -------
    RunEngine
        A RunEngine whose ``md`` is a ``RedisJSONDict`` on the beamline Redis
        (``REDIS_HOST``), or an in-memory mock proposal in CI/mock mode.
    """
    if is_running_in_ci():
        today = datetime.today()
        return RunEngine(
            {
                "data_session": "pass-123456",
                "cycle": f"{today.year}-{(today.month - 1) // 4 + 1}",
                "proposal": {
                    "title": "Mock mode proposal",
                    "type": "Mock Commissioning",
                    "pi_name": os.getenv("USER", "Unknown"),
                },
            }
        )
    return RunEngine(RedisJSONDict(open_redis_client(redis_ssl=True), ""))  # type: ignore[arg-type]


def print_proposal_info(md: MutableMapping[str, Any]):
    """Print the active proposal from the RunEngine metadata.

    Parameters
    ----------
    md : MutableMapping[str, Any]
        The RunEngine metadata.
    """
    proposal_md = md.get("proposal", {})
    if proposal_md:
        rprint(
            "Active proposal:\n"
            f"  Proposal title: [italic]{proposal_md['title']}[/italic]\n"
            f"  Proposal type: [italic]{proposal_md['type']}[/italic]\n"
            f"  Proposal PI: [italic]{proposal_md['pi_name']}[/italic]\n"
        )


def remove_md_keys(md: MutableMapping[str, Any], prefix: str = "PDF:") -> list[str]:
    """Delete every metadata key that contains ``prefix``.

    Ported from ``remove_keys_redis`` in pdf-profile-collection's ``00-base.py``
    (C.-H. Lin, 2025-11-18), which trims unneeded ``PDF:`` entries from Redis.

    Parameters
    ----------
    md : MutableMapping[str, Any]
        The RunEngine metadata, usually Redis-backed.
    prefix : str, optional
        The text a key must contain to be removed, by default ``"PDF:"``.

    Returns
    -------
    list[str]
        The keys that were removed.
    """
    removed = [key for key in list(md) if prefix in key]
    for key in removed:
        del md[key]
    return removed


def _failed_paths(error: Exception, prefix: str = "") -> list[str]:
    """Dotted paths of the leaves of a (nested) NotConnectedError."""
    sub = error.sub_errors if isinstance(error, NotConnectedError) else {}
    if not sub:
        return [prefix.rstrip(".")]
    return [
        p for name, err in sub.items() for p in _failed_paths(err, f"{prefix}{name}.")
    ]


def _count_connected_signals(device: Device, error: Exception) -> tuple[int, int]:
    """Count (connected, total) signals of ``device``, given its connection error."""
    signals = [
        path for path, d in walk_devices(device).items() if isinstance(d, Signal)
    ]
    if not signals:
        return 0, 1
    failed = [p for p in _failed_paths(error) if p]
    if not failed:
        return 0, len(signals)
    down = [s for s in signals if any(s == f or s.startswith(f + ".") for f in failed)]
    return len(signals) - len(down), len(signals)


def auto_init_devices(
    timeout: float = 1.0, ok_message="OK", fail_message="FAILED", verbose: bool = False
) -> DeviceProcessor:
    """Create a DeviceProcessor that connects devices, printing status for each.

    Devices connect in mock mode when :func:`is_running_in_ci` is true. A device
    that fails to connect is reported and does not stop the others.

    Parameters
    ----------
    timeout : float
        The timeout in seconds for each device connection attempt.
    ok_message : str, optional
        The label printed for a connected device.
    fail_message : str, optional
        The label printed for a device that failed to connect.
    verbose : bool, optional
        Whether to print each failure's reason, by default False.

    Returns
    -------
    DeviceProcessor
        A DeviceProcessor that connects devices and prints their connection status.
    """
    mock = is_running_in_ci()
    # highlight=False stops rich from coloring dot-runs as ellipses.
    console = Console(highlight=False)
    console.print("\n[bold]Connecting to beamline devices...[/bold]\n")

    async def _process_devices(devices: dict[str, Device]):
        for name, device in devices.items():
            if not device.name:
                device.set_name(name)
        coros = {
            name: device.connect(mock, timeout) for name, device in devices.items()
        }
        failed: set[str] = set()
        reasons: dict[str, str] = {}
        e = NotConnectedError({})
        try:
            await wait_for_connection(**coros)
        except NotConnectedError as err:
            e = err
            failed = set(e.sub_errors.keys())
            reasons = {name: str(err) for name, err in e.sub_errors.items()}

        width = max(len(ok_message), len(fail_message)) + 2
        for name, device in devices.items():
            dots = "." * (40 - len(name))
            # Escaped brackets stay uncolored; only the label text is styled.
            if name not in failed:
                console.print(
                    f"  {name} {dots} \\[[bold green]"
                    f"{ok_message.center(width)}[/bold green]]"
                )
                continue
            line = (
                f"  {name} {dots} \\[[bold red]{fail_message.center(width)}[/bold red]]"
            )
            connected, total = _count_connected_signals(device, e.sub_errors[name])
            if 0 < connected < total:
                line += f" [dim]({connected}/{total} signals connected)[/dim]"
            console.print(line)
        if verbose and reasons:
            console.print(
                "\n".join(f"{name}: {reason}" for name, reason in reasons.items())
            )

    return DeviceProcessor(_process_devices)
