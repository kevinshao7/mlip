#!/usr/bin/env python3
"""Build or run scp commands for pulling generated outputs from Dais.

Default use prints a PowerShell-ready command and does not transfer anything:

    python copy_outputs.py 7_20_repex

Run the printed command by adding:

    python copy_outputs.py 7_20_repex --execute
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import PurePosixPath


DEFAULT_SCP = r"C:\Program Files\Git\usr\bin\scp.exe"
DEFAULT_PROXY_JUMP = "kshao@gate1.mpcdf.mpg.de"
DEFAULT_REMOTE_HOST = "kshao@dais11"
DEFAULT_REMOTE_ROOT = "/dais/fs/scratch/kshao/jaxmd-cli/pt_output"
DEFAULT_LOCAL_ROOT = "/c/Users/shaoq/Documents/Mainz/mlip/outputsfull"


def shell_quote(value: str) -> str:
    if os.name == "nt":
        return f'"{value}"'
    return shlex.quote(value)


def powershell_quote(value: str) -> str:
    escaped = value.replace('"', '`"')
    return f'"{escaped}"'


def format_command(argv: list[str], powershell: bool) -> str:
    if powershell:
        exe, *rest = argv
        return "& " + " ".join([powershell_quote(exe), *[powershell_quote(arg) for arg in rest]])
    return " ".join(shell_quote(arg) for arg in argv)


def remote_spec(host: str, path: PurePosixPath, copy_contents: bool) -> str:
    suffix = "/." if copy_contents else "/"
    return f"{host}:{path}{suffix}"


def local_path(root: str, run: str, subpath: str) -> str:
    parts = [root.rstrip("/"), run]
    if subpath:
        parts.append(subpath.strip("/"))
    return "/".join(parts) + "/"


def build_paths(args: argparse.Namespace) -> tuple[str, str]:
    remote_root = PurePosixPath(args.remote_root)

    if args.remote_subpath is not None:
        remote_path = remote_root / args.remote_subpath.strip("/")
        local = args.local if args.local else local_path(args.local_root, args.run, args.local_subpath)
        return remote_spec(args.remote_host, remote_path, args.contents), local

    if args.mode == "analysis":
        remote_path = remote_root / args.run / "analysis_plots"
        local = local_path(args.local_root, args.run, "analysis_plots")
        return remote_spec(args.remote_host, remote_path, True), local

    if args.mode == "run":
        remote_path = remote_root / args.run
        local = local_path(args.local_root, args.run, "")
        return remote_spec(args.remote_host, remote_path, True), local

    if args.mode == "pt-output":
        remote_path = remote_root
        local = local_path(args.local_root, args.run, "")
        return remote_spec(args.remote_host, remote_path, False), local

    raise ValueError(f"Unknown mode: {args.mode}")


def build_command(args: argparse.Namespace) -> list[str]:
    source, destination = build_paths(args)
    return [
        args.scp,
        "-r",
        "-o",
        f"ProxyJump={args.proxy_jump}",
        source,
        destination,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print or run scp commands for copying remote output folders."
    )
    parser.add_argument(
        "run",
        nargs="?",
        default="7_20_repex",
        help="Run/output folder name under the local outputsfull directory.",
    )
    parser.add_argument(
        "--mode",
        choices=("analysis", "run", "pt-output"),
        default="analysis",
        help=(
            "analysis: copy pt_output/RUN/analysis_plots/. into local analysis_plots; "
            "run: copy pt_output/RUN/. into local RUN; "
            "pt-output: copy pt_output/ into local RUN."
        ),
    )
    parser.add_argument(
        "--remote-subpath",
        help=(
            "Custom path below remote root, for example "
            "7_20_repex/analysis_plots or analysis_plots."
        ),
    )
    parser.add_argument(
        "--local-subpath",
        default="",
        help="Custom path below local outputsfull/RUN for --remote-subpath transfers.",
    )
    parser.add_argument(
        "--local",
        help="Full destination path override. Useful for one-off copies.",
    )
    parser.add_argument(
        "--contents",
        action="store_true",
        help="With --remote-subpath, copy directory contents by appending '/.'.",
    )
    parser.add_argument("--execute", action="store_true", help="Run scp instead of only printing it.")
    parser.add_argument("--bash", action="store_true", help="Print bash-style command instead of PowerShell.")
    parser.add_argument("--scp", default=DEFAULT_SCP, help="Path to scp executable.")
    parser.add_argument("--proxy-jump", default=DEFAULT_PROXY_JUMP)
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--local-root", default=DEFAULT_LOCAL_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = build_command(args)
    print(format_command(command, powershell=not args.bash))

    if not args.execute:
        return 0

    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
