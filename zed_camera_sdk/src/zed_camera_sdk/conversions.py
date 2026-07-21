"""Pure helpers: parameter parsing, ZED SDK enum-name resolution, and frame
byte conversions. Nothing here imports the ZED SDK, so it stays importable and
unit-testable without CUDA or a camera. The functions return canonical SDK enum
member names as strings; the thin sdk layer resolves them against ``pyzed.sl``.
"""

from __future__ import annotations

import numpy as np

# resolution parameter -> sl.RESOLUTION member name, with the legal frame rates
# the SDK accepts for that mode.
_RESOLUTIONS: dict[str, tuple[str, frozenset[int]]] = {
    "vga": ("VGA", frozenset({15, 30, 60, 100})),
    "hd720": ("HD720", frozenset({15, 30, 60})),
    "hd1080": ("HD1080", frozenset({15, 30})),
    "hd2k": ("HD2K", frozenset({15})),
}

# depth_mode parameter -> sl.DEPTH_MODE member name. Only the GPU neural modes
# are offered; the SDK-free sibling covers CPU stereo matching.
_DEPTH_MODES: frozenset[str] = frozenset({"NEURAL", "NEURAL_LIGHT", "NEURAL_PLUS"})

_COLOR_MODES: frozenset[str] = frozenset({"auto", "manual"})

# z16 depth is unsigned 16-bit millimetres; 0 marks an invalid measurement.
_Z16_MAX_MM = 65535


def resolution_member(resolution: str) -> str:
    """The sl.RESOLUTION member name for a resolution parameter."""
    entry = _RESOLUTIONS.get(resolution.lower())
    if entry is None:
        raise ValueError(
            f"resolution must be one of {sorted(_RESOLUTIONS)}, got {resolution!r}"
        )
    return entry[0]


def validate_frame_rate(resolution: str, frame_rate: int) -> int:
    """The frame rate, checked against the rates the resolution allows."""
    entry = _RESOLUTIONS.get(resolution.lower())
    if entry is None:
        raise ValueError(
            f"resolution must be one of {sorted(_RESOLUTIONS)}, got {resolution!r}"
        )
    legal = entry[1]
    if frame_rate not in legal:
        raise ValueError(
            f"frame_rate {frame_rate} is not legal for {resolution!r}; "
            f"allowed: {sorted(legal)}"
        )
    return frame_rate


def depth_mode_member(depth_mode: str) -> str:
    """The sl.DEPTH_MODE member name for a depth_mode parameter."""
    member = depth_mode.upper()
    if member not in _DEPTH_MODES:
        raise ValueError(
            f"depth_mode must be one of {sorted(_DEPTH_MODES)}, got {depth_mode!r}"
        )
    return member


def color_mode(mode: str) -> str:
    """A validated set_color_* exposure/white-balance mode."""
    if mode not in _COLOR_MODES:
        raise ValueError(f"mode must be one of {sorted(_COLOR_MODES)}, got {mode!r}")
    return mode


def min_depth_mm(min_depth_m: float) -> float:
    """The SDK's depth_minimum_distance in millimetres (coordinate unit)."""
    if not min_depth_m > 0.0:
        raise ValueError(f"min_depth_m must be positive, got {min_depth_m}")
    return min_depth_m * 1000.0


def bgra_to_bgr_bytes(image: np.ndarray) -> bytes:
    """Packed BGR bytes from the SDK's HxWx4 BGRA left image, alpha dropped."""
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"expected an HxWx4 BGRA image, got shape {image.shape}")
    return np.ascontiguousarray(image[:, :, :3]).tobytes()


def depth_mm_to_z16_bytes(depth_mm: np.ndarray) -> bytes:
    """Little-endian z16 bytes from an HxW float32 millimetre depth map.

    Non-finite samples (the SDK's markers for occluded, too-near, or too-far
    pixels) and values outside the u16 range collapse to 0, the z16 invalid
    marker; finite in-range samples round to the nearest millimetre.
    """
    if depth_mm.ndim != 2:
        raise ValueError(f"expected an HxW depth map, got shape {depth_mm.shape}")
    valid = np.isfinite(depth_mm) & (depth_mm >= 0.0) & (depth_mm <= _Z16_MAX_MM)
    rounded = np.rint(np.where(valid, depth_mm, 0.0))
    return rounded.astype("<u2").tobytes()
