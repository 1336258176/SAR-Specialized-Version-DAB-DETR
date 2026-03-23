"""
record_env_info.py  ——  Record experiment environment information to a text file.

Collects:
  - System info  (OS, hostname, CPU, RAM)
  - GPU info     (device name, VRAM, driver / CUDA runtime version)
  - Python info  (version, executable path)
  - PyTorch info (version, CUDA build version, cuDNN version)
  - Key package versions (torchvision, numpy, opencv, pillow, …)
  - Environment variables relevant to deep-learning experiments

Usage:
    python tools/record_env_info.py
    python tools/record_env_info.py --output output/my_exp/env.txt
    python tools/record_env_info.py --output output/my_exp/env.txt --extra-packages timm einops

Extend this script by adding new collector functions and registering them in
COLLECTORS at the bottom of the file.  Each function must return a plain str.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple


# ── helpers ────────────────────────────────────────────────────────────────


def _section(title: str, body: str) -> str:
    """Wrap *body* with a titled section header/footer."""
    width = 72
    bar_top = "=" * width
    bar_bot = "-" * width
    return f"\n{bar_top}\n  {title}\n{bar_bot}\n{body.rstrip()}\n"


def _run(cmd: List[str], default: str = "N/A") -> str:
    """Run a subprocess command and return its stdout, or *default* on error."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or default
    except Exception:
        return default


def _pkg_version(package: str) -> str:
    """Return the installed version of *package*, or 'not installed'."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


# ── collector functions ────────────────────────────────────────────────────
# Each collector returns a single str that will be appended to the report.
# Add new collectors here and register them in COLLECTORS below.


def collect_datetime() -> str:
    now = datetime.now()
    lines = [
        f"Timestamp : {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Timezone  : {datetime.now().astimezone().tzname()}",
    ]
    return _section("Date & Time", "\n".join(lines))


def collect_system() -> str:
    uname = platform.uname()
    mem_total = "N/A"
    try:
        import psutil

        mem_total = f"{psutil.virtual_memory().total / (1024**3):.1f} GB"
    except ImportError:
        # fallback for Linux
        if Path("/proc/meminfo").exists():
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    mem_total = f"{kb / (1024**2):.1f} GB"
                    break

    cpu_model = platform.processor() or "N/A"
    # Linux: try /proc/cpuinfo for a richer model name
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if "model name" in line:
                cpu_model = line.split(":", 1)[-1].strip()
                break

    cpu_count_logical: Optional[int] = None
    cpu_count_physical: Optional[int] = None
    try:
        import psutil

        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)
    except ImportError:
        cpu_count_logical = os.cpu_count()

    cpu_count_str = str(cpu_count_logical)
    if cpu_count_physical is not None:
        cpu_count_str += f" logical / {cpu_count_physical} physical"

    lines = [
        f"Hostname  : {socket.gethostname()}",
        f"OS        : {uname.system} {uname.release} ({uname.version})",
        f"Machine   : {uname.machine}",
        f"CPU model : {cpu_model}",
        f"CPU cores : {cpu_count_str}",
        f"RAM total : {mem_total}",
    ]
    return _section("System Information", "\n".join(lines))


def collect_python() -> str:
    lines = [
        f"Version    : {sys.version}",
        f"Executable : {sys.executable}",
        f"Prefix     : {sys.prefix}",
    ]
    return _section("Python", "\n".join(lines))


def collect_pytorch() -> str:
    try:
        import torch
    except ImportError:
        return _section("PyTorch", "PyTorch is not installed.")

    cuda_available = torch.cuda.is_available()
    lines = [
        f"torch version      : {torch.__version__}",
        f"CUDA available     : {cuda_available}",
        f"CUDA build version : {torch.version.cuda or 'N/A'}",
        f"cuDNN version      : {torch.backends.cudnn.version() if cuda_available else 'N/A'}",
        f"cuDNN enabled      : {torch.backends.cudnn.enabled}",
        f"cuDNN benchmark    : {torch.backends.cudnn.benchmark}",
    ]

    # torchvision / torchaudio are optional companions
    for pkg in ("torchvision", "torchaudio"):
        lines.append(f"{pkg:<20} : {_pkg_version(pkg)}")

    return _section("PyTorch", "\n".join(lines))


def collect_gpu() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return _section("GPU", "CUDA not available — no GPU information.")
    except ImportError:
        return _section("GPU", "PyTorch not installed — skipping GPU info.")

    import torch

    n = torch.cuda.device_count()
    lines = [f"Device count : {n}"]

    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / (1024**3)
        lines += [
            f"\n  [GPU {i}]",
            f"    Name              : {props.name}",
            f"    Total memory      : {total_mem:.1f} GB",
            f"    Compute capability: {props.major}.{props.minor}",
            f"    Multi-processors  : {props.multi_processor_count}",
        ]

    # CUDA runtime version via nvidia-smi
    nvidia_smi = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if nvidia_smi != "N/A":
        lines.append(f"\n  NVIDIA driver version : {nvidia_smi.splitlines()[0]}")

    cuda_runtime = _run(["nvcc", "--version"])
    if cuda_runtime != "N/A":
        # extract "release X.Y" from nvcc banner
        for part in cuda_runtime.split(","):
            if "release" in part.lower():
                lines.append(f"  CUDA runtime (nvcc)   : {part.strip()}")
                break

    return _section("GPU / CUDA", "\n".join(lines))


def collect_key_packages() -> str:
    """Versions of packages commonly used in detection / vision research."""
    packages = [
        "numpy",
        "scipy",
        "opencv-python",
        "opencv-python-headless",
        "Pillow",
        "matplotlib",
        "pycocotools",
        "timm",
        "einops",
        "transformers",
        "accelerate",
        "mmdet",
        "mmcv",
        "scikit-learn",
        "pandas",
    ]
    lines = []
    for pkg in packages:
        ver = _pkg_version(pkg)
        if ver != "not installed":
            lines.append(f"{pkg:<30} : {ver}")

    body = "\n".join(lines) if lines else "(none of the listed packages found)"
    return _section("Key Packages", body)


def collect_extra_packages(names: List[str]) -> str:
    """Versions of user-specified extra packages."""
    if not names:
        return ""
    lines = [f"{pkg:<30} : {_pkg_version(pkg)}" for pkg in names]
    return _section("Extra Packages (user-specified)", "\n".join(lines))


def collect_env_vars() -> str:
    """Snapshot of deep-learning-relevant environment variables."""
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "CUDA_PATH",
        "LD_LIBRARY_PATH",
        "PATH",
        "PYTHONPATH",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NCCL_DEBUG",
        "MASTER_ADDR",
        "MASTER_PORT",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
    ]
    lines = []
    for key in keys:
        val = os.environ.get(key, "(not set)")
        # truncate very long values (e.g. PATH) for readability
        if len(val) > 200:
            val = val[:197] + "..."
        lines.append(f"{key:<25} : {val}")
    return _section("Environment Variables", "\n".join(lines))


def collect_pip_freeze() -> str:
    """Full pip freeze output — comprehensive but verbose."""
    output = _run([sys.executable, "-m", "pip", "freeze"])
    return _section("pip freeze (full)", output)


# ── COLLECTORS registry ────────────────────────────────────────────────────
# Each entry is (label, callable).  Disable a section by commenting it out.
# The callable receives no arguments (extra_packages is handled separately).

COLLECTORS: List[Tuple[str, Callable[[], str]]] = [
    ("datetime", collect_datetime),
    ("system", collect_system),
    ("python", collect_python),
    ("pytorch", collect_pytorch),
    ("gpu", collect_gpu),
    ("key_packages", collect_key_packages),
    ("env_vars", collect_env_vars),
    # ("pip_freeze", collect_pip_freeze),   # uncomment for full package list
]


# ── main ───────────────────────────────────────────────────────────────────


def build_report(extra_packages: Optional[List[str]] = None) -> str:
    """Run all collectors and concatenate their output into a single report."""
    parts: List[str] = []

    for _label, fn in COLLECTORS:
        try:
            parts.append(fn())
        except Exception as exc:
            parts.append(_section(_label, f"[ERROR] {exc}"))

    if extra_packages:
        try:
            parts.append(collect_extra_packages(extra_packages))
        except Exception as exc:
            parts.append(_section("extra_packages", f"[ERROR] {exc}"))

    header = "=" * 100 + "\n  Experiment Environment Report\n" + "=" * 100
    return header + "".join(parts) + "\n"


def save_report(report: str, output_path: str) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record experiment environment information to a text file."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output/env.txt",
        help="Path of the output text file (default: output/env.txt).",
    )
    parser.add_argument(
        "--extra-packages",
        "-e",
        nargs="+",
        default=[],
        metavar="PKG",
        help="Additional package names whose versions should be recorded.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Also print the report to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(extra_packages=args.extra_packages)

    out_path = save_report(report, args.output)
    print(f"Environment report saved to: {out_path}")

    if args.stdout:
        print(report)


if __name__ == "__main__":
    main()
