# utils/hardware_info.py
"""
Detects available hardware resources at runtime.
Used to inform the LLM about memory, CPU and GPU constraints
so generated code stays within actual machine limits.
"""
from __future__ import annotations

import os
import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def get_hardware_context() -> str:
    """
    Returns a formatted hardware constraints block ready to inject into prompts.
    Result is cached after the first call.
    """
    ram_str  = _detect_ram()
    cpu_str  = _detect_cpu()
    gpu_str  = _detect_gpu()

    lines = [
        "## Hardware Constraints (machine where the code will run)",
        f"- RAM available : {ram_str}",
        f"- CPU cores     : {cpu_str}",
        f"- GPU           : {gpu_str}",
        "",
        "Design your solution to fit comfortably within these limits.",
        "Prefer memory-efficient data structures and algorithms.",
        "Set n_jobs, batch_size, and chunk sizes accordingly.",
    ]
    # ASCII-safe output (avoids encoding issues on Windows cp1252 terminals)
    return "\n".join(lines).encode("ascii", errors="replace").decode("ascii")


# ── RAM ──────────────────────────────────────────────────────────────────

def _detect_ram() -> str:
    # psutil (most accurate)
    try:
        import psutil
        total = psutil.virtual_memory().total
        available = psutil.virtual_memory().available
        return f"{_fmt_bytes(total)} total, {_fmt_bytes(available)} available"
    except ImportError:
        pass

    # Windows: PowerShell CIM (wmic is deprecated/removed in newer Windows)
    try:
        total_kb = int(subprocess.check_output(
            ["powershell", "-Command",
             "(Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize"],
            timeout=8, stderr=subprocess.DEVNULL,
        ).decode().strip())
        free_kb = int(subprocess.check_output(
            ["powershell", "-Command",
             "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
            timeout=8, stderr=subprocess.DEVNULL,
        ).decode().strip())
        return f"{_fmt_bytes(total_kb * 1024)} total, {_fmt_bytes(free_kb * 1024)} available"
    except Exception:
        pass

    # Linux: /proc/meminfo
    try:
        info = _parse_proc_meminfo()
        if "MemTotal" in info:
            total = info["MemTotal"] * 1024
            avail = info.get("MemAvailable", info.get("MemFree", 0)) * 1024
            return f"{_fmt_bytes(total)} total, {_fmt_bytes(avail)} available"
    except Exception:
        pass

    return "unknown"


# ── CPU ──────────────────────────────────────────────────────────────────

def _detect_cpu() -> str:
    logical  = os.cpu_count() or 1
    physical = None

    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
    except ImportError:
        pass

    if physical and physical != logical:
        return f"{physical} physical cores, {logical} logical (n_jobs up to {logical})"
    return f"{logical} cores (n_jobs up to {logical})"


# ── GPU ──────────────────────────────────────────────────────────────────

def _detect_gpu() -> str:
    # Try nvidia-smi first (works without torch)
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5, stderr=subprocess.DEVNULL,
        ).decode().strip()
        if out:
            gpus = []
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    name, vram_mb = parts
                    vram_gb = round(int(vram_mb) / 1024, 1)
                    gpus.append(f"{name} ({vram_gb} GB VRAM)")
            if gpus:
                return ", ".join(gpus) + " -> deep learning is viable"
    except Exception:
        pass

    # Try torch
    try:
        import torch
        if torch.cuda.is_available():
            names = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_gb = round(props.total_memory / 1024 ** 3, 1)
                names.append(f"{props.name} ({vram_gb} GB VRAM)")
            return ", ".join(names) + " -> deep learning is viable"
    except ImportError:
        pass

    return "None detected -> use CPU-only algorithms; avoid heavy deep learning"


# ── Helpers ──────────────────────────────────────────────────────────────

def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _wmic_value(text: str, key: str) -> str | None:
    for line in text.splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip()
    return None


def _parse_proc_meminfo() -> dict:
    result = {}
    with open("/proc/meminfo", "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                result[key] = int(parts[1])
    return result
