"""
QTAM – Q‑Transform Amplitude Modulation

A PyTorch implementation of the Q‑transform with amplitude modulation,
supporting single‑Q and multi‑Q transforms, multiple window families,
and efficient down/upsampling.
"""

__version__ = "0.2.1"   

# Core functionality (single‑Q, single‑config)
from .QTAM import (
    QTile,
    SingleQTransform,
    # Window functions (scalar versions)
    planck_taper_window_range,
    kaiser_window_range,
    tukey_window,
    bisquare_window,
    hann_window,
)

# Multi‑configuration scanning and batched windows
from .QTAM_Scan import (
    QTileMulti,
    SingleQMultiTransform,
    QScanMulti,
    # Batched window functions
    planck_taper_window_range_batch,
    kaiser_window_range_batch,
    tukey_window_batch,
    bisquare_window_batched,
    hann_window_batched,
)

# Public API
__all__ = [
    # Core
    "QTile",
    "SingleQTransform",
    "planck_taper_window_range",
    "kaiser_window_range",
    "tukey_window",
    "bisquare_window",
    "hann_window",
    # Scan
    "QTileMulti",
    "SingleQMultiTransform",
    "QScanMulti",
    "planck_taper_window_range_batch",
    "kaiser_window_range_batch",
    "tukey_window_batch",
    "bisquare_window_batched",
    "hann_window_batched",
]
