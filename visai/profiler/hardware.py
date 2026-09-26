"""Best-effort hardware facts. Never fails (ported from kernel-forge lib/hardware.py + Apple Silicon)."""

from __future__ import annotations

import platform
import shutil
import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def probe() -> dict:
    info: dict = {"os": platform.platform(), "machine": platform.machine()}
    if platform.system() == "Darwin":
        try:
            chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5).strip()
            mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=5).strip())
            info.update({"chip": chip.replace("Apple ", "").replace(" ", ""), "chip_name": chip, "memory_gb": round(mem / 2**30)})
        except Exception as exc:  # noqa: BLE001
            info["chip"] = "apple-silicon"
            info["probe_error"] = str(exc)
        try:
            import mlx.core as mx

            info["backend"] = "mlx"
            info["metal"] = bool(mx.metal.is_available())
            info["mlx_version"] = mx.__version__
        except Exception:  # noqa: BLE001
            pass
        info.setdefault("note", "Apple Silicon unified memory: decode is bandwidth-bound; fusion and quantization matter.")
        return info
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.check_output(
                [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True, timeout=10
            ).strip()
            parts = [p.strip() for p in out.split(",")]
            info.update({"gpu": parts[0], "driver": parts[1] if len(parts) > 1 else "", "backend": "cuda"})
            info["chip"] = parts[0].replace(" ", "")
        except Exception as exc:  # noqa: BLE001
            info["probe_error"] = str(exc)
    info.setdefault("chip", "unknown")
    return info
