"""Shared helpers for signed 16-bit little-endian PCM sample handling."""

from __future__ import annotations

PCM16_MIN = -32768
PCM16_MAX = 32767


def decode_pcm16(frames: bytes) -> list[int]:
    """Decode raw little-endian signed 16-bit PCM frames into integer samples."""
    return [int.from_bytes(frames[index : index + 2], "little", signed=True) for index in range(0, len(frames), 2)]


def clamp_pcm16(value: int) -> int:
    """Clamp an integer sample to the representable signed 16-bit range."""
    return max(PCM16_MIN, min(PCM16_MAX, value))


def encode_pcm16(value: int) -> bytes:
    """Clamp ``value`` and encode it as a little-endian signed 16-bit sample."""
    return clamp_pcm16(value).to_bytes(2, "little", signed=True)
