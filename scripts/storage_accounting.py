"""Serialization-size helpers shared by experiment drivers."""
from __future__ import annotations


def trellis_serialized_bytes(source_dim: int, bits: int, memory: int) -> int:
    """Return scalar-trellis bytes including padding, start state, and rescale."""
    if source_dim <= 0 or bits <= 0 or memory < 0:
        raise ValueError("source_dim and bits must be positive; memory must be non-negative")
    transformed_dim = 1 << (source_dim - 1).bit_length()
    payload = (transformed_dim * bits + 7) // 8
    start_state = (memory + 7) // 8
    return payload + start_state + 4
