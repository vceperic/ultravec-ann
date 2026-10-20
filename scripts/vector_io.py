"""Small, domain-neutral helpers shared by ANN experiment drivers."""
from __future__ import annotations

import os
from pathlib import Path
import struct
import subprocess

import numpy as np


def wr(path: str | os.PathLike[str], values: np.ndarray) -> None:
    """Write a dense 2-D array in little-endian Texmex fvecs format."""
    array = np.ascontiguousarray(values, dtype="<f4")
    if array.ndim != 2:
        raise ValueError(f"fvecs output must be 2-D, got shape {array.shape}")
    with Path(path).open("wb") as handle:
        header = struct.pack("<i", array.shape[1])
        for row in array:
            handle.write(header)
            handle.write(row.tobytes())


def rd(path: str | os.PathLike[str]) -> np.ndarray:
    """Read a non-ragged little-endian Texmex fvecs file, failing on truncation."""
    rows: list[np.ndarray] = []
    dimension: int | None = None
    with Path(path).open("rb") as handle:
        while True:
            header = handle.read(4)
            if not header:
                break
            if len(header) != 4:
                raise ValueError(f"truncated fvecs header in {path}")
            current = struct.unpack("<i", header)[0]
            if current <= 0 or (dimension is not None and current != dimension):
                raise ValueError(f"invalid or ragged fvecs dimension {current} in {path}")
            dimension = current
            payload = handle.read(4 * current)
            if len(payload) != 4 * current:
                raise ValueError(f"truncated fvecs row in {path}")
            rows.append(np.frombuffer(payload, dtype="<f4").copy())
    if not rows:
        raise ValueError(f"empty fvecs file: {path}")
    return np.ascontiguousarray(rows, dtype=np.float32)


def norm(values: np.ndarray) -> np.ndarray:
    """Row-normalize without producing NaNs for zero rows."""
    array = np.asarray(values, dtype=np.float32)
    return array / (np.linalg.norm(array, axis=1, keepdims=True) + 1e-9)


def q8(coefficients: np.ndarray) -> np.ndarray:
    """Per-column affine uint8-grid reconstruction used by PCA-head controls."""
    values = np.asarray(coefficients, dtype=np.float32)
    lo = values.min(axis=0, keepdims=True)
    hi = values.max(axis=0, keepdims=True)
    scale = (hi - lo) / 255.0
    scale[scale == 0] = 1.0
    return np.round((values - lo) / scale) * scale + lo


def recon(src: str, backend: str, out: str, dim: int, bits: int = 2) -> np.ndarray:
    """Invoke the artifact CLI and return the unpadded reconstruction."""
    binary = os.environ.get("ULTRAVEC_BIN", "./target/release/ultravec")
    env = dict(os.environ)
    env.setdefault("ULTRAVEC_TRELLIS_MEM", "12")
    subprocess.run(
        [binary, "recon", "--dataset", src, "--backend", backend, "--bits", str(bits), "--out", out],
        check=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return rd(out)[:, :dim]
