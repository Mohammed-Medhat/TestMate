"""
Create version-matched venv clusters for the local ablation.

Each eval file was harvested from a specific (repo, version). The host packages
are newer, so the source modules reference removed symbols and fail to import.
This script builds one lightweight venv per major/LTS cluster, each containing
only `pytest pytest-cov coverage` + the pinned framework. The ablation harness
(_ablation_common.py:_select_cluster_python) routes each file's import gate +
pytest steps to the matching venv when it exists, and falls back to host python
otherwise.

The model/generation runs on the HOST python (with torch+CUDA); these venvs are
used only to *execute* the generated tests under the right package version, so
they do NOT need torch/transformers — keeping them ~50–150 MB each.

Usage:
    python setup_version_venvs.py                 # create ALL clusters
    python setup_version_venvs.py --only django42 # create just one
    python setup_version_venvs.py --list          # show clusters + status

Venv root defaults to D:\\TestMate\\venvs (override with TESTMATE_VENV_ROOT).
Cluster names here must match _CLUSTER_MAP in _ablation_common.py.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

VENV_ROOT = Path(os.environ.get("TESTMATE_VENV_ROOT", r"D:\TestMate\venvs"))

# Base test runners every cluster needs, plus the version pins.
_BASE = ["pytest", "pytest-cov", "coverage"]

CLUSTERS = {
    "django32":   ["django==3.2"],
    "django42":   ["django==4.2"],
    "django50":   ["django==5.0"],
    "sympy11":    ["sympy==1.1", "mpmath"],
    "sympy112":   ["sympy==1.12"],
    "sklearn022": ["scikit-learn==0.22", "numpy<1.24", "scipy<1.8", "threadpoolctl", "joblib"],
}


def _venv_python(name: str) -> Path:
    """Path to the venv's python (Windows Scripts/, POSIX bin/)."""
    if os.name == "nt":
        return VENV_ROOT / name / "Scripts" / "python.exe"
    return VENV_ROOT / name / "bin" / "python"


def _exists(name: str) -> bool:
    return _venv_python(name).exists()


def create_cluster(name: str, pins: list[str]) -> bool:
    """Create one venv and install its pinned packages. Idempotent."""
    py = _venv_python(name)
    if py.exists():
        print(f"  ✓ {name}: already exists ({py})")
        return True

    venv_dir = VENV_ROOT / name
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"  → creating venv {name} at {venv_dir} ...")
    r = subprocess.run([sys.executable, "-m", "venv", str(venv_dir)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ✗ venv creation failed: {r.stderr[-400:]}")
        return False

    pkgs = _BASE + pins
    print(f"  → pip install {' '.join(pkgs)} ...")
    r = subprocess.run([str(py), "-m", "pip", "install", "--quiet",
                        "--disable-pip-version-check", *pkgs],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ✗ pip install failed: {r.stderr[-600:]}")
        return False
    print(f"  ✓ {name}: ready")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Build version-matched test venvs.")
    ap.add_argument("--only", help="create just this cluster (e.g. django42)")
    ap.add_argument("--list", action="store_true", help="list clusters + status")
    args = ap.parse_args()

    if args.list:
        print(f"Venv root: {VENV_ROOT}")
        for name, pins in CLUSTERS.items():
            status = "present" if _exists(name) else "missing"
            print(f"  {name:12s} [{status:7s}]  {', '.join(pins)}")
        return 0

    targets = {args.only: CLUSTERS[args.only]} if args.only else CLUSTERS
    if args.only and args.only not in CLUSTERS:
        print(f"Unknown cluster '{args.only}'. Known: {', '.join(CLUSTERS)}")
        return 2

    print(f"Building {len(targets)} cluster venv(s) under {VENV_ROOT}\n")
    ok = all(create_cluster(n, p) for n, p in targets.items())
    print("\nDone." if ok else "\nDone with errors.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
