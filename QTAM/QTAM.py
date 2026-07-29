import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from torch_spline_interpolation_1_0_0 import *
import numpy as np
import gc
from collections import OrderedDict

import warnings

"""
All based on https://github.com/gwpy/gwpy/blob/v3.0.8/gwpy/signal/qtransform.py
"""

#-------------------------------------------------------------------------------------------------------------
def _centered_pad_or_crop(X: torch.Tensor, M: int) -> torch.Tensor:
    """Helper for ideal band-pass filtering via crop/pad in FFT domain."""
    N = X.shape[-1]
    if M == N:
        return X
    # Use F.fftshift, assuming `import torch.nn.functional as F`
    Xs = torch.fft.fftshift(X, dim=-1)
    if M > N:
        pad_left = (M - N) // 2
        pad_right = M - N - pad_left
        Y = F.pad(Xs, (pad_left, pad_right))
    else:
        start = (N - M) // 2
        end = start + M
        Y = Xs[..., start:end]
    return torch.fft.ifftshift(Y, dim=-1)
#-------------------------------------------------------------------------------------------------------


def _phasor_from_integer_shift(shift: int, T: int, device, dtype, sign: int = +1):
    """
    Return rot[n] = exp(sign * i * 2π * shift * n / T)
    computed via integer modulo so phases stay in [0, 2π) for float32 stability.
    """
    n = torch.arange(T, device=device, dtype=torch.int64)          # [T]
    shift_mod = int(shift) % T
    m = (shift_mod * n) % T                                       # [T] integers in [0, T)
    phase = (sign * (2.0 * math.pi) / T) * m.to(dtype)            # [T] in [0, 2π)
    return torch.polar(torch.ones(T, device=device, dtype=dtype), phase)  # complex

# ============================
# Principled geometric grid
# ============================

def principled_geometric_freqs(
    q: float,
    frange: List[float],
    duration: float,
    qprime: Optional[float] = None,
    return_meta: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """
    Build a *principled* geometric grid of center frequencies that is
    guaranteed (by construction) to cover the analysis band
    `[f_min, f_max]` with the natural CQT window bandwidth at every
    center frequency.

    The natural CQT window at frequency `f` has bandwidth
    ``Δf_window(f) ≈ f / q`` (in Hz).  On a geometric grid with `B`
    bins per octave, the spacing between adjacent centers is
    ``Δf_grid(f) ≈ f · ln(2) / B`` (in Hz).  For the window to cover
    the gap to the next center, we need ``Δf_window ≥ Δf_grid``, i.e.
    ``B ≥ q · ln(2) ≈ 0.693 · q``.

    The *minimum* number of bins per octave is therefore

        B_min = ⌈ q · ln(2) ⌉

    and the *minimum* number of frequency bins to span
    `[f_min, f_max]` is

        nfreq = ⌈ B_min · log2(f_max / f_min) ⌉.

    The grid is built in Hz (no FFT-bin snapping) as

        f_k = f_min · 2^(k / B_min),   k = 0, ..., nfreq.

    The first and last entries are snapped to ``f_min`` and
    ``f_max`` exactly so the grid spans the full requested band.

    Parameters
    ----------
    q : float
        The quality factor of the CQT (must be > 0).
    frange : list of two floats
        ``[f_min, f_max]`` in Hz, with ``f_min > 0`` and
        ``f_max > f_min``.
    duration : float
        Length of the time series in seconds (used only to define
        ``qprime`` if not given; not used in the grid construction).
    qprime : float, optional
        The GWpy-style "prime" quality factor ``q / sqrt(11)``.
        If not given, it is computed from ``q``.
    return_meta : bool
        If True, also return a dict with ``B_min``, ``nfreq``, and
        the grid construction parameters.

    Returns
    -------
    freqs : torch.Tensor
        1D tensor of shape ``[nfreq + 1]`` with the grid in Hz,
        strictly increasing, spanning ``[f_min, f_max]``.
    meta : dict (only if ``return_meta=True``)
        ``B_min`` (int), ``nfreq`` (int), ``fstep`` (float, in
        octaves), ``freq_base`` (float, multiplicative step).

    Notes
    -----
    This function does *not* snap the grid to the FFT bin grid
    (``1/duration``), which is the source of the inter-window
    coverage gaps in the legacy GWpy-style mismatch method
    (see ``get_freqs`` with ``spacing='mismatch'``).
    """
    if qprime is None:
        qprime = q / math.sqrt(11)

    f_min, f_max = float(frange[0]), float(frange[1])
    if f_min <= 0:
        raise ValueError(f"f_min must be > 0, got {f_min}")
    if f_max <= f_min:
        raise ValueError(f"f_max must be > f_min, got {f_min}, {f_max}")

    # Minimum number of bins per octave for the windows to cover
    # the gap to the next center frequency.  This is the standard
    # constant-Q relation: Δf_window(f) ≈ f / q and
    # Δf_grid(f) ≈ f · ln(2) / B; setting Δf_window ≥ Δf_grid
    # gives B ≥ q · ln(2).
    B_min = max(1, int(math.ceil(q * math.log(2))))

    # Minimum number of bins to span [f_min, f_max].
    nfreq = max(1, int(math.ceil(B_min * math.log2(f_max / f_min))))

    # Build the grid in Hz (no FFT-bin snapping).
    # f_k = f_min * 2^(k / B_min) for k = 0, ..., nfreq
    # We use nfreq + 1 points so that the first and last can be
    # snapped to f_min and f_max exactly.
    freqs = torch.tensor(
        [f_min * 2.0 ** (k / B_min) for k in range(nfreq + 1)],
        dtype=torch.float64,
    )
    # Snap the first and last entries to f_min and f_max exactly.
    freqs[0] = f_min
    freqs[-1] = f_max
    # Keep only strictly-increasing entries (defensive).
    freqs = torch.unique(freqs)

    if return_meta:
        meta = {
            "B_min": B_min,
            "nfreq": nfreq,
            "fstep_octaves": 1.0 / B_min,
            "freq_base": 2.0 ** (1.0 / B_min),
        }
        return freqs, meta
    return freqs


def check_window_coverage(
    freqs: torch.Tensor,
    q: float,
    duration: float,
    qprime: Optional[float] = None,
    frange: Optional[List[float]] = None,
    raise_on_failure: bool = False,
    spacing: str = "geometric",
    nyquist: Optional[float] = None,
    window_left_edges: Optional[List[float]] = None,
    window_right_edges: Optional[List[float]] = None,
) -> dict:
    """
    Check whether the supports of the CQT windows centered at
    ``freqs`` collectively cover the analysis band ``[f_min, f_max]``.

    The natural CQT window at frequency ``f`` has bandwidth
    ``Δf_window(f) ≈ f / q`` (in Hz).  The window is centered at
    ``f`` and has half-bandwidth ``Δf_window(f) / 2`` on each side,
    so its support is ``[f - f/(2q), f + f/(2q)]``.

    Two adjacent windows at ``f_k`` and ``f_{k+1}`` overlap iff
    ``f_{k+1} - f_k ≤ (f_k + f_{k+1}) / (2 q)``, i.e. the gap
    between them is at most the *average* of their half-bandwidths.

    Parameters
    ----------
    freqs : torch.Tensor
        1D tensor of center frequencies in Hz, strictly increasing.
    q : float
        The quality factor of the CQT.
    duration : float
        Length of the time series in seconds (used only to define
        ``qprime`` if not given).
    qprime : float, optional
        The GWpy-style "prime" quality factor ``q / sqrt(11)``.
        If not given, it is computed from ``q``.
    window_left_edges : list of float, optional
        If given, a list of length ``len(freqs)`` with the actual
        LEFT edge of each window in Hz.  When provided, the coverage
        check uses these (and ``window_right_edges``) instead of the
        theoretical CQT support ``[fk - fk/(2q), fk + fk/(2q)]``.
        This is what `SingleQTransform.diagnose()` passes to reflect
        the actual `QTile.full_window` support (which is shrunk by
        any `max_window_size` cap, and may be extended to 0 by the
        DC patch on the first tile, and truncated at Nyquist on the
        last tile in 'nyquist' mode).
    window_right_edges : list of float, optional
        If given, a list of length ``len(freqs)`` with the actual
        RIGHT edge of each window in Hz.  Must be the same length
        as ``window_left_edges`` and ``freqs`` (otherwise the
        theoretical CQT support is used instead).
    frange : list of two floats, optional
        ``[f_min, f_max]`` in Hz.  If given, the function also
        checks that the first window's support extends down to
        ``f_min`` and the last window's support extends up to
        ``f_max``.  If not given, only the inter-window coverage
        is checked.
    raise_on_failure : bool
        If True, raise a ``RuntimeError`` when the coverage check
        fails.  If False (default), return a dict with the
        diagnostics.
    spacing : str, optional
        The grid-spacing mode of the calling ``SingleQTransform``.
        If ``'nyquist'``, the last window's right edge is treated
        as truncated at Nyquist (i.e. the "last window does not
        reach f_max" check is bypassed, replaced by the check
        ``f_max <= Nyquist``).  Default: ``'geometric'``.
    nyquist : float, optional
        The Nyquist frequency (sample_rate / 2).  Required when
        ``spacing='nyquist'`` to validate that ``f_max`` is within
        the allowed range.  Default: ``None``.

    Returns
    -------
    report : dict
        Dictionary with the following keys:

        - ``covered`` (bool): True iff the windows collectively
          cover the analysis band.
        - ``bad_pairs`` (list of int): indices ``k`` such that the
          windows at ``freqs[k]`` and ``freqs[k+1]`` do *not*
          overlap.  Empty if coverage is good.
        - ``bad_pairs_gaps`` (list of float): the size of the gap
          (in Hz) for each bad pair.
        - ``bad_segments`` (list of (float, float)): contiguous
          frequency ranges (in Hz) that are not covered by any
          window.  Empty if coverage is good.
        - ``first_window_does_not_reach_fmin`` (bool): True iff
          the first window's support does not extend down to
          ``f_min``.  Only present if ``frange`` is given.
        - ``last_window_does_not_reach_fmax`` (bool): True iff
          the last window's support does not extend up to
          ``f_max``.  Only present if ``frange`` is given.

    Notes
    -----
    The "average half-bandwidth" criterion
    ``f_{k+1} - f_k ≤ (f_k + f_{k+1}) / (2q)`` is exactly the
    condition ``Δf_window(f) ≥ Δf_grid(f)`` for the geometric
    grid, where ``Δf_grid(f) ≈ f · ln(2) / B`` and
    ``B ≥ q · ln(2)``.  It is the *necessary* condition for the
    admissibility condition ``P[f] > 0`` to hold on the gap
    between two adjacent windows (it is not *sufficient* if the
    window shapes are not flat-topped, but it is conservative for
    the bisquare windows used by default).
    """
    if qprime is None:
        qprime = q / math.sqrt(11)

    freqs = torch.as_tensor(freqs, dtype=torch.float64)
    if freqs.numel() < 2:
        return {
            "covered": True,
            "bad_pairs": [],
            "bad_pairs_gaps": [],
            "bad_segments": [],
        }

    # Compute the gap between adjacent centers and the average
    # half-bandwidth of the two windows.
    #
    # If the caller passed actual window edges (which reflect
    # the max_window_size cap, the DC patch on the first tile,
    # and the Nyquist clamp on the last tile in 'nyquist' mode),
    # use those for the coverage check.  Otherwise fall back to
    # the theoretical CQT support.
    use_actual = (
        window_left_edges is not None
        and window_right_edges is not None
        and len(window_left_edges) == len(freqs)
        and len(window_right_edges) == len(freqs)
    )
    if use_actual:
        # Two adjacent windows at f_k and f_{k+1} overlap iff
        # the right edge of window k is >= the left edge of
        # window k+1.  The "gap" between them is the difference
        # (positive iff they don't overlap).
        right_k = torch.tensor(window_right_edges[:-1], dtype=torch.float64)
        left_kp1 = torch.tensor(window_left_edges[1:], dtype=torch.float64)
        gap = left_kp1 - right_k                                # [K-1] (positive = uncovered)
        is_bad = gap > 0                                        # [K-1] bool
        bad_pairs = torch.nonzero(is_bad, as_tuple=False).flatten().tolist()
        bad_pairs_gaps = gap[is_bad].tolist()                   # size of the gap (Hz)
    else:
        gaps = torch.diff(freqs)                              # [K-1]
        avg_half_bw = (freqs[:-1] + freqs[1:]) / (2.0 * q)    # [K-1]
        is_bad = gaps > avg_half_bw                           # [K-1] bool
        bad_pairs = torch.nonzero(is_bad, as_tuple=False).flatten().tolist()
        bad_pairs_gaps = (gaps[is_bad] - avg_half_bw[is_bad]).tolist()

    # Build the list of uncovered segments.
    bad_segments = []
    for k in bad_pairs:
        # The gap is between freqs[k] + Δf/2 and freqs[k+1] - Δf/2,
        # but the uncovered segment is the full gap [freqs[k], freqs[k+1]]
        # minus the supports.  Conservatively, report the full gap.
        bad_segments.append((float(freqs[k]), float(freqs[k + 1])))

    report = {
        "covered": len(bad_pairs) == 0,
        "bad_pairs": bad_pairs,
        "bad_pairs_gaps": bad_pairs_gaps,
        "bad_segments": bad_segments,
    }

    # Check the first/last window reach the band edges.
    if frange is not None:
        f_min, f_max = float(frange[0]), float(frange[1])
        if use_actual:
            # Use the actual left/right edges of the first/last
            # window (which already reflect the DC patch and the
            # Nyquist clamp).
            first_window_left = float(window_left_edges[0])
            last_window_right = float(window_right_edges[-1])
        else:
            # Use qprime (not q) for the natural support: the actual
            # bisquare window has bandwidth f/qprime = f*sqrt(11)/q
            # in the frequency domain (this matches the windowsize
            # formula in production QTAM: windowsize =
            # 2*int(f/qprime*duration)+1 samples).  Using q instead
            # of qprime would under-estimate the actual window
            # bandwidth by a factor of sqrt(11).
            first_half_bw = float(freqs[0]) / (2.0 * qprime)
            last_half_bw = float(freqs[-1]) / (2.0 * qprime)
            first_window_left = freqs[0] - first_half_bw
            last_window_right = freqs[-1] + last_half_bw
        report["first_window_does_not_reach_fmin"] = first_window_left > f_min
        # Last window reach check: in `spacing='nyquist'` mode
        # the last window's right edge is truncated at Nyquist
        # by get_full_window, so the last window always reaches
        # Nyquist by construction.  We detect nyquist mode via
        # the `spacing` argument.  For all other modes, the
        # last window's natural right edge is `last_window_right`
        # (which is either the actual right edge from the qtile
        # or the theoretical CQT right edge), and we compare
        # against the user-specified f_max.
        if spacing == "nyquist":
            # The last window is truncated at Nyquist by
            # get_full_window, so it always reaches Nyquist.
            # We report it as reaching f_max iff f_max <= Nyquist.
            if nyquist is None:
                nyquist_eff = float("inf")
            else:
                nyquist_eff = float(nyquist)
            report["last_window_does_not_reach_fmax"] = f_max > nyquist_eff
        else:
            report["last_window_does_not_reach_fmax"] = (
                last_window_right < f_max
            )
        report["covered"] = (
            report["covered"]
            and not report["first_window_does_not_reach_fmin"]
            and not report["last_window_does_not_reach_fmax"]
        )

    if raise_on_failure and not report["covered"]:
        msg = (
            f"QTAM: the CQT window bank does not cover the analysis "
            f"band. {len(bad_pairs)} inter-window gap(s) detected, "
            f"plus {len(bad_segments)} uncovered segment(s). "
            f"Either increase the number of frequency bins "
            f"(`num_freq`) or use `spacing='geometric'` (the default) "
            f"to let the algorithm choose the minimum number of bins "
            f"automatically."
        )
        raise RuntimeError(msg)

    return report


# ============================
# External Window Functions
# ============================

def planck_taper_window_range(N: int, epsilon: float, x_min: float = -1, x_max: float = 1, device: str = 'cpu',data_type=torch.float32) -> torch.Tensor:
    """
    Constructs a Planck-taper window defined over an arbitrary range [x_min, x_max].
    Internally, it maps the coordinate linearly to the canonical range [-1,1] and then applies
    your provided Planck-taper formula.
    
    Args:
        N (int): Window length (number of samples).
        epsilon (float): Taper fraction (0 < epsilon < 0.5).
        x_min (float): Minimum value of the input coordinate.
        x_max (float): Maximum value of the input coordinate.
        device (str): Device.
        
    Returns:
        Tensor: A 1D tensor of shape [N] representing the Planck-taper window.
    """
    # Create coordinate x in [x_min, x_max]
    x = torch.linspace(x_min, x_max, steps=N, device=device, dtype=data_type)
    # Map x linearly to the canonical domain [-1,1]
    x_canonical = 2 * (x - x_min) / (x_max - x_min) - 1
    # Map to y in [0,1]
    y = (x_canonical + 1) / 2
    w = torch.ones(N, device=device, dtype=data_type)
    # Rising edge: 0 < y < epsilon
    mask_rise = (y > 0) & (y < epsilon)
    if mask_rise.any():
        Z_plus = 2 * epsilon * (1 / (1 + 2 * y[mask_rise] - 1) + 1 / (1 - 2 * epsilon + 2 * y[mask_rise] - 1))
        w[mask_rise] = 1.0 / (torch.exp(Z_plus) + 1.0)
    # Flat region: epsilon <= y <= 1 - epsilon
    mask_flat = (y >= epsilon) & (y <= 1 - epsilon)
    w[mask_flat] = 1.0
    # Falling edge: 1 - epsilon < y < 1
    mask_fall = (y > (1 - epsilon)) & (y < 1)
    if mask_fall.any():
        Z_minus = 2 * epsilon * (1 / (1 - 2 * y[mask_fall] + 1) + 1 / (1 - 2 * epsilon - 2 * y[mask_fall] + 1))
        w[mask_fall] = 1.0 / (torch.exp(Z_minus) + 1.0)
    # Endpoints set to 0
    w[0] = 0.0
    w[-1] = 0.0
    return w

def kaiser_window_range(L: int, beta: float = 8.6, x_min: float = -1, x_max: float = 1, device: str = 'cpu') -> torch.Tensor:
    """
    Returns a Kaiser window of length L defined over an arbitrary range [x_min, x_max].
    The window values are generated by torch.kaiser_window (which is independent of the coordinate),
    so the coordinate mapping is handled externally.
    
    Args:
        L (int): Window length.
        beta (float): Kaiser beta parameter.
        x_min (float): Minimum coordinate value.
        x_max (float): Maximum coordinate value.
        device (str): Device.
    
    Returns:
        Tensor: A 1D tensor of shape [L] representing the Kaiser window.
    """
    return torch.kaiser_window(L, beta=beta, periodic=False, device=device)

def tukey_window(window_length, alpha=0.05):
    """Generates a Tukey window."""
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    window = torch.ones(window_length)
    if alpha == 0:
        return window
    ramp = int(alpha * window_length / 2)
    if ramp == 0:
        return window
    w = torch.linspace(0, 1, ramp)
    cosine = 0.5 * (1 + torch.cos(math.pi * (w - 1)))
    window[:ramp] = cosine
    window[-ramp:] = cosine.flip(0)
    return window

def bisquare_window(L: int, device: str = 'cpu',data_type=torch.float32) -> torch.Tensor:
    """
    Compute the bisquare window defined as:
      w(x) = (1 - x^2)^2, with x linearly spaced from -1 to 1.
    """
    x = torch.linspace(-1, 1, steps=L, device=device, dtype=data_type)
    return (1 - x**2)**2


def hann_window(L: int, device: str = "cpu",data_type=torch.float32) -> torch.Tensor:
    """
    Hann window.
    """
    n = torch.arange(L, device=device, dtype=data_type)
    w = 0.5 * (1 - torch.cos(2 * math.pi * n / (L - 1)))
    return w

                  
# ============================
# QTile Class
# ============================
class QTile(torch.nn.Module):
    """
    Compute the row of Q-tiles for a single Q value and a single
    frequency for a batch of multi-channel frequency series data.
    Invertible version with windows defined over full frequency series.
    """

    def __init__(
        self,
        q: float,
        frequency: float,
        duration: float,
        sample_rate: float,
        mismatch: float,
        logf: bool = False, # log spaced frequecies
        #energy_mode: bool = True, #return energy (True) or amplitude (False) of the spectrogram
        #phase_mode: bool = False, #return also the phase of the spectrogram
        window_param: Optional[Union[str, torch.Tensor]] = None, # window type ('kaiser', 'hann', 'bisqaure','tukey','planck-taper')
        tau: float = 1/2,   # parameter for planck-taper and tukey window function 
        beta: float = 8.6,  # parameter for kaiser window function 
        eps: float = 1e-5,  # small epsilon for padded values
        max_window_size = None, # maximum width of window function
        frange: Optional[list] = None,
        from_0: bool = False,
        is_first: bool = False,
        is_last: bool = False,
        spacing: str = 'geometric',
        synthesis_to_nyquist: bool = False,

    ):
        super().__init__()
        self.q = q
        self.frequency = frequency
        self.duration = duration
        self.sample_rate = sample_rate
        self.mismatch = mismatch
        self.logf = logf
        #self.energy_mode = energy_mode
        #self.phase_mode = phase_mode
        self.window_param = window_param
        self.tau = tau
        self.beta = beta
        self.eps = eps
        self.max_window_size=max_window_size
        self.frange =frange
        self.is_first=is_first
        self.is_last=is_last
        self.spacing=spacing
        self.from_0=from_0
        # `frange[1]` is normally the last *centre* frequency, not
        # necessarily the last FFT bin required for synthesis.  When
        # the parent chose that centre automatically, its natural
        # window is designed to reach Nyquist; retain that right tail.
        self.synthesis_to_nyquist = synthesis_to_nyquist

        self.qprime = self.q / (11 ** 0.5)
        self.deltam = torch.tensor(2 * (self.mismatch / 3.0) ** 0.5)

        self.windowsize = 2 * int(self.frequency / self.qprime * self.duration) + 1
        #print('-----------------------------')
        if self.max_window_size:
            #print(f'{self.windowsize=} ; {self.max_window_size=}')
            self.windowsize = min(self.windowsize, self.max_window_size)
            
        self.pad_len = (self.duration*self.sample_rate)//2 +1 - self.windowsize
        self.pad_left = int((self.pad_len) // 2)
        self.pad_right = int((self.pad_len + 1) // 2)  

        self.register_buffer("window", self.get_window())
        self.register_buffer("full_window", self.get_full_window())
        
        #print(self.frequency,self.shift,self.windowsize,self.window.shape,self.full_window.shape)
    
    def compute_window_energy(self, window):
        #Normalize by imposing Parseval condition: sum |w[t]|^2 dt = (1/N) * sum |W[f]|^2 = 1
        return torch.sum(window**2).item()/self.duration

    
    def get_window(self):

        dtype = torch.get_default_dtype()
    
        if self.window_param is None:
            window = bisquare_window(self.windowsize, data_type=dtype)
    
        elif isinstance(self.window_param, torch.Tensor):
            w = self.window_param.flatten().to(dtype)
            if w.shape[0] != self.windowsize:
                w = F.interpolate(
                    w[None, None],
                    size=self.windowsize,
                    mode="linear",
                    align_corners=False
                ).squeeze()
            window = w
    
        elif self.window_param.lower() == "hann":
            window = hann_window(self.windowsize, data_type=dtype)
    
        elif self.window_param.lower() == "tukey":
            window = tukey_window(self.windowsize, alpha=self.tau).to(dtype)
    
        elif self.window_param.lower() == "planck-taper":
            window = planck_taper_window_range(
                self.windowsize,
                epsilon=self.tau,
                x_min=-1,
                x_max=1,
                data_type=dtype
            )
    
        elif self.window_param.lower() == "kaiser":
            window = kaiser_window_range(
                self.windowsize,
                beta=self.beta
            ).to(dtype)
    
        else:
            raise ValueError(f"Unsupported window_param: {self.window_param}")
    
        return window
    
    def get_full_window(self):

        # Integer center only (required for exact multirate inversion)
        self.shift = int(self.frequency * self.duration)

        total_len = self.pad_left + self.windowsize + self.pad_right

        full_w = torch.full(
            (1, 1, total_len),
            0.0,
            device=self.window.device,
            dtype=self.window.dtype
        )

        if self.frange is None:
            # Work on a local copy; never mutate the parent
            # SingleQTransform.frange (which is a shared list).
            frange_eff = [0.0, self.sample_rate / 2]
        else:
            # Copy so we can adjust the band edges locally
            # (DC patch on first tile, Nyquist clamp on last
            # tile in 'nyquist' mode) without mutating the
            # parent's frange.  Without this copy, the auto-set
            # f_min (qprime/T) gets wiped out by the DC patch
            # and `self.frange[0]` ends up as 0.0 again.
            frange_eff = [float(self.frange[0]), float(self.frange[1])]

        if self.from_0:
            # DC patch: extend the first tile down to 0.
            frange_eff[0] = 0.0

        # ✅ Nyquist clamping: in `spacing='nyquist'` mode the last
        # window is allowed to have its center at f_max = Nyquist
        # (or any value up to Nyquist).  Its natural right edge
        # f_max * (1 + 1/(2*qprime)) may extend ABOVE Nyquist,
        # which would alias back below Nyquist.  To prevent this,
        # we clamp the right edge of the LAST window to Nyquist.
        # The natural left edge is preserved, so the window is
        # *truncated* (its right tail is cut, but it remains a
        # well-defined positive kernel).
        if self.is_last and (self.spacing == 'nyquist' or self.synthesis_to_nyquist):
            frange_eff[1] = float(self.sample_rate) / 2.0

        center_idx = self.shift

        # ✅ Even/odd safe symmetric placement
        half_left = self.windowsize // 2
        half_right = self.windowsize - half_left - 1

        win_start_idx = center_idx - half_left
        win_end_idx = center_idx + half_right

        dst_start = max(int(frange_eff[0] * self.duration), win_start_idx)
        dst_end = min(int(frange_eff[1] * self.duration), win_end_idx)

        if dst_start <= dst_end:
            src_start = dst_start - win_start_idx
            src_end = src_start + (dst_end - dst_start) + 1
            full_w[0, 0, dst_start:dst_end + 1] = self.window[src_start:src_end]

        # ✅ DC patch (robust): extend the first nonzero placed value down to DC
        if self.is_first and self.from_0:
            nz = (full_w[0, 0] != 0).nonzero(as_tuple=False).flatten()
            if nz.numel() > 0:
                first_nz = int(nz[0].item())
                if first_nz > 0:
                    full_w[0, 0, :first_nz] = full_w[0, 0, first_nz]

        # Normalize
        wen = self.compute_window_energy(full_w)
        if wen > 0:
            full_w *= wen ** -0.5

        return full_w
        

    def forward(
        self, 
        fseries: torch.Tensor, 
        polar_mode: bool = True, 
        energy_mode: bool = True, 
        phase_mode: bool = True, 
        complex_mode: bool = False,
        num_time: Optional[int] = None,
        am_mode: bool = True
    ):
        while len(fseries.shape) < 3:
            fseries = fseries[None]
    
        # ✅ No dtype casting here anymore
        wenergy = fseries * self.full_window
        T_in = wenergy.shape[-1]
    
        if num_time is None:
    
            tdenergy = torch.fft.ifft(wenergy, norm='ortho')
            tdenergy *= (self.sample_rate) ** 0.5
    
        else:
    
            T_out = num_time
    
            # ✅ Exact integer demodulation
            wenergy_baseband = torch.roll(
                wenergy, shifts=-self.shift, dims=-1
            )
    
            wenergy_baseband_cropped = _centered_pad_or_crop(
                wenergy_baseband, T_out
            )
    
            tdenergy_baseband = torch.fft.ifft(
                wenergy_baseband_cropped, norm='ortho'
            )
    
            tdenergy_baseband *= math.sqrt(T_out / T_in)
    
            if am_mode:
                tdenergy = tdenergy_baseband * (self.sample_rate) ** 0.5
            else:
                # ✅ Float32-stable remodulation using integer-bin phasor
                rot = _phasor_from_integer_shift(
                    shift=self.shift,
                    T=T_out,
                    device=fseries.device,
                    dtype=fseries.real.dtype,
                    sign=+1
                ).view(1, 1, -1)

                tdenergy = tdenergy_baseband * rot * (self.sample_rate) ** 0.5
    
        if polar_mode:
    
            if energy_mode:
                energy = tdenergy.real**2 + tdenergy.imag**2
            else:
                energy = torch.sqrt(tdenergy.real**2 + tdenergy.imag**2)
    
            if phase_mode:
                phase = torch.atan2(tdenergy.imag, tdenergy.real)
                return torch.stack([energy, phase], dim=2)
    
            return energy.unsqueeze(2)
    
        elif complex_mode:
    
            return tdenergy
    
        else:
    
            return torch.stack(
                [tdenergy.real, tdenergy.imag], dim=2
            )
        

    def invert(self, tile, polar_mode: bool = True, energy_mode: bool = True, phase_mode: bool = True, complex_mode: bool = False):
        # Extract amplitude and phase from the tile
        
        if polar_mode:
            amplitude = torch.sqrt(tile[:, :, 0]) if energy_mode else tile[:, :, 0]
            
            if not phase_mode:
                print(f"\033[91m[Warning]\033[0m Qtile.invert: phase_mode is False, assuming all 0 phase.")

            amplitude /= (self.sample_rate)**0.5
            
            if phase_mode:
                phase = tile[:, :, 1]
            else:
                phase = torch.zeros_like(amplitude)
                
            tdenergy = amplitude * torch.exp(1j * phase)   
            
        elif complex_mode:
            tdenergy= tile/(self.sample_rate)**0.5
        else:
            tdenergy= torch.complex(tile[:,:,0,:],tile[:,:,1,:])/ (self.sample_rate)**0.5
        
        # FFT back to frequency domain
        wenergy = torch.fft.fft(tdenergy, norm= 'ortho')

        # Divide by full_window to recover original fseries
        fseries = wenergy / self.full_window
        
        return fseries
        
##########################################################################
# Single Q Qtransform Class
##########################################################################

class SingleQTransform(torch.nn.Module):
    """
    Compute the Q-transform for a single Q value for a batch of
    multi-channel time series data. Input data should have
    three dimensions or fewer.

    Args:
        duration:
            Length of the time series data in seconds
        sample_rate:
            Sample rate of the data in Hz
        spectrogram_shape:
            The shape of the interpolated spectrogram, specified as
            `(num_f_bins, num_t_bins)`. Because the
            frequency spacing of the Q-tiles is in log-space, the frequency
            interpolation is log-spaced as well.
        q:
            The Q value to use for the Q transform
        frange:
            The lower and upper frequency limit to consider for
            the transform. If unspecified, default values will
            be chosen based on q, sample_rate, and duration
        mismatch:
            The maximum fractional mismatch between neighboring tiles.
            Only used by the legacy ``spacing='mismatch'`` mode.
        spacing:
            Grid-spacing mode.  Four options:

              - ``'geometric'`` (default): a *principled* geometric
                grid whose number of bins is chosen so that the
                natural CQT window bandwidth at every center
                frequency covers the gap to the next center.  The
                auto-set f_max is
                ``f_max = Nyquist / (1 + 1/(2 qprime))`` (the
                largest center frequency such that the last
                window's right edge is exactly at Nyquist).  This
                is the *only* mode that guarantees the
                admissibility condition by construction without
                truncating any window.

              - ``'mismatch'``: the legacy GWpy-style mismatch
                formula, with FFT-bin snapping.  This mode does
                *not* strictly guarantee the admissibility
                condition and emits a ``RuntimeWarning`` on use.

              - ``'linear'``: linear spacing (only meaningful when
                ``num_freq`` is explicitly given).

              - ``'nyquist'``: same geometric grid as
                ``'geometric'``, but the user is allowed to set
                ``f_max = Nyquist`` (or any value up to
                Nyquist).  The last window's right edge is then
                truncated at Nyquist by ``get_full_window`` to
                prevent aliasing.  This mode *does* guarantee the
                inter-window coverage (geometric spacing is the
                same as in ``'geometric'``); only the LAST
                window is *truncated* (its right tail is cut at
                Nyquist).  Use this when you want to use the
                full spectrum up to Nyquist, accepting that the
                last window is incomplete.  Requires an explicit
                ``f_max`` (raises ``ValueError`` if
                ``frange[1] = +inf``).
        num_freq:
            If given, an explicit number of frequency bins.  When
            combined with ``spacing='geometric'``, the explicit
            number is used (with a coverage check); when combined
            with ``spacing='mismatch'`` or ``'linear'``, the
            explicit number is used as-is.
    """

    def __init__(
        self,
        duration: float,
        sample_rate: float,
        q: float = 12,
        frange: Optional[List[float]] = None,
        mismatch: float = 0.2,
        num_freq: int = 0,
        logf: bool = False,
        spacing: str = 'geometric',   # NEW: 'geometric' | 'mismatch' | 'linear'
        window_param: Optional[Union[str, torch.Tensor]] = None,
        tau: float = 1/2,
        beta: float = 8.6,
        max_window_size = False,
        eps=1e-5,
        from_0: bool = True,           # DC patch on first window (centered at fmin)
        warn_on_bad_coverage: bool = True,   # NEW: emit RuntimeWarning on coverage failure
        raise_on_bad_coverage: bool = False, # NEW: raise RuntimeError on coverage failure
        
    ):
        super().__init__()
        self.q = q
        self.sample_rate = sample_rate
        # Never use or mutate a shared mutable default list.  In particular,
        # automatic f_max handling below replaces +inf with the final centre
        # frequency; if that modified a default list, later constructions
        # without `frange` would incorrectly inherit that finite f_max and
        # clip synthesis before Nyquist.
        if frange is None:
            frange = [0.0, float("inf")]
        if len(frange) != 2:
            raise ValueError("frange must be None or a two-element [f_min, f_max] sequence.")
        # `frange` is the ANALYSIS / SYNTHESIS band, not the range of
        # tile-centre frequencies.  Keep those concepts separate: when the
        # upper analysis edge is +inf it means Nyquist, while the highest CQT
        # centre is chosen lower so that its window reaches Nyquist.
        self._frange_input = [float(frange[0]), float(frange[1])]
        analysis_fmin = self._frange_input[0]
        analysis_fmax = (sample_rate / 2.0 if math.isinf(self._frange_input[1])
                         else self._frange_input[1])
        self.frange = [analysis_fmin, analysis_fmax]
        self.duration = duration
        self.mismatch = mismatch
        self.logf = logf
        self.spacing = spacing
        self.window_param = window_param
        self.tau = tau
        self.beta = beta
        self.num_freq = num_freq
        self.eps = eps
        self.warn_on_bad_coverage = warn_on_bad_coverage
        self.raise_on_bad_coverage = raise_on_bad_coverage
        
        qprime = self.q / 11 ** (1 / 2.0)
        self.qprime = qprime
        
        # Apply DC patch to the first window (centered at fmin) so that it is extended
        # to the left to reach 0Hz, eliminating any spectral gap from 0Hz to fmin.
        self.from_0 = from_0 or (self.frange[0] <= 0)
        # Keep a separate centre-frequency range.  For an unbounded analysis
        # request, the final centre is below Nyquist while its window reaches
        # the Nyquist analysis edge.
        auto_last_centre = (sample_rate / 2) / (1 + 1 / (2 * qprime))
        requested_fmax = float(self._frange_input[1])
        if math.isinf(requested_fmax) and self.spacing == 'nyquist':
            raise ValueError(
                "QTAM: spacing='nyquist' requires an explicit f_max; "
                f"got +inf (Nyquist is {sample_rate / 2} Hz)."
            )
        grid_min = (float(self.frange[0]) if float(self.frange[0]) > 0
                    else qprime / duration)
        grid_max = auto_last_centre if math.isinf(requested_fmax) else requested_fmax
        self.center_frange = [grid_min, grid_max]
        # `frange` already ends at Nyquist for +inf input, so this flag is
        # mainly retained for compatibility with the explicit nyquist mode.
        self._synthesis_to_nyquist = (
            self.spacing != 'nyquist' and math.isinf(requested_fmax)
        )
        self.freqs = self.get_freqs()

        # Report the grid-density requirement before discussing the window
        # cap: a cap cannot repair gaps caused by too few frequency rows.
        self._num_freq_info = self._get_num_freq_info()
        self._report_num_freq_requirement()
        
        if max_window_size:
            #print(f'{max_window_size=}')
            self.max_window_size = self.get_max_window_size(max_window_size)
        else:
            self.max_window_size = None
        
        self.qtile_transforms = torch.nn.ModuleList(
            [
                QTile(
                    self.q, freq, self.duration, sample_rate, self.mismatch,
                    self.logf, window_param=self.window_param, tau=self.tau, beta=self.beta,
                    max_window_size=self.max_window_size, eps=self.eps, frange=self.frange,
                    from_0=(self.from_0 and (i == 0)),
                    is_first=(i == 0),
                    is_last=(i == len(self.freqs) - 1),
                    spacing=self.spacing,
                    synthesis_to_nyquist=(
                        self._synthesis_to_nyquist and (i == len(self.freqs) - 1)
                    ),
                )
                for i, freq in enumerate(self.freqs)
            ]
        )
        self.qtiles = None
        self.phase_qtiles = None  

        # ---- Coverage check -----------------------------------------
        # After the grid and tiles are built, check window coverage.
        self._coverage_report = self._check_coverage()
        self._report_coverage()

    def get_freqs(self):
        """
        Calculate the frequencies that will be used in this transform.
        For each frequency, a `QTile` is created.

        Four grid-spacing modes are supported
        (see the ``spacing`` argument of ``__init__``):

          - ``'geometric'`` (default): a principled geometric grid
            computed by ``principled_geometric_freqs``.  This is
            the *only* mode that guarantees the admissibility
            condition by construction AND keeps every window's
            full support inside the spectrum (no truncation).

          - ``'mismatch'``: the legacy GWpy-style mismatch
            formula, with FFT-bin snapping.  This mode does
            *not* strictly guarantee the admissibility
            condition and emits a ``RuntimeWarning`` on use.

          - ``'linear'``: linear spacing, only if ``num_freq``
            is explicitly given.

          - ``'nyquist'``: like ``'geometric'`` but allows
            ``f_max = Nyquist`` by truncating the last
            window's right edge at Nyquist (in
            ``get_full_window``).  Use this when you want
            to use the *full* spectrum up to Nyquist,
            accepting that the last window is effectively
            cut.  This mode *does* guarantee the
            inter-window coverage (the geometric spacing
            is the same as in ``'geometric'`` mode); only
            the *last* window is truncated.  Requires an
            explicit ``f_max`` (raises ``ValueError`` if
            ``frange[1] = +inf``).
        """
        # Frequency-centre range is distinct from the analysis/synthesis band.
        minf, maxf = self.center_frange

        # When the user passed f_min <= 0 (typically 0), the
        # geometric grid builder needs a strictly-positive lower
        # bound.  We use the principled minimum qprime/T for the
        # grid construction (so the grid is well-defined), but
        # we DO NOT change `frange[0]` itself -- the user's
        # f_min=0 is preserved for invertibility (the first QTile
        # uses the DC patch to cover [0, f_center]).  This is
        # purely a numerical workaround for np.geomspace.
        grid_minf = minf if minf > 0 else self.qprime / self.duration

        # Path 1: user explicitly specified a number of frequency bins
        if self.num_freq:
            if self.spacing == 'geometric':
                # Build a geometric grid of exactly num_freq points
                # spanning [grid_minf, maxf].  This does NOT guarantee
                # coverage; we warn if num_freq is too small.
                freqs = torch.tensor(
                    np.geomspace(grid_minf, maxf, self.num_freq)
                )
                return torch.unique(freqs)
            elif self.spacing == 'nyquist':
                # Same as 'geometric' with explicit num_freq, but
                # the LAST window's right edge will be truncated
                # at Nyquist by get_full_window.  This is a
                # legitimate use case (e.g. match the grid of an
                # external pipeline).
                freqs = torch.tensor(
                    np.geomspace(grid_minf, maxf, self.num_freq)
                )
                return torch.unique(freqs)
            elif self.spacing == 'mismatch' or self.logf:
                if self.logf:
                    freqs = torch.tensor(
                        np.geomspace(grid_minf, maxf, self.num_freq)
                    )
                else:
                    freqs = torch.linspace(minf, maxf, self.num_freq)
                return torch.unique(freqs)
            else:
                freqs = torch.linspace(minf, maxf, self.num_freq)
                return torch.unique(freqs)

        # Path 2: 'mismatch' mode (legacy GWpy-style)
        if self.spacing == 'mismatch':
            warnings.warn(
                "QTAM: spacing='mismatch' does NOT guarantee the "
                "admissibility condition. The FFT-bin snapping step "
                "in the legacy GWpy-style formula can leave "
                "inter-window coverage gaps, especially in the "
                "high-frequency tail. Use spacing='geometric' (the "
                "default) for invertibility by construction.",
                RuntimeWarning
            )
            fcum_mismatch = (
                math.log(maxf / minf) * (2 + self.q**2) ** (1 / 2.0) / 2.0
            )
            deltam = 2 * (self.mismatch / 3.0) ** (1 / 2.0)
            nfreq = int(max(1, math.ceil(fcum_mismatch / deltam)))
            fstep = fcum_mismatch / nfreq
            fstepmin = 1 / self.duration

            freq_base = math.exp(2 / ((2 + self.q**2) ** (1 / 2.0)) * fstep)
            freqs = torch.Tensor([freq_base ** (i + 0.5) for i in range(nfreq)])
            freqs = (minf * freqs // fstepmin) * fstepmin
            return torch.unique(freqs)

        # Path 3: 'nyquist' mode (geometric grid up to a
        # user-specified f_max <= Nyquist; the last window's
        # right edge is truncated at Nyquist by get_full_window)
        if self.spacing == 'nyquist':
            # Same geometric grid as 'geometric' mode, but the
            # user is allowed to set f_max = Nyquist (or any
            # value up to Nyquist).  The geometric spacing is
            # the same as in 'geometric' mode, so the
            # inter-window coverage is guaranteed by
            # construction.  Only the LAST window is
            # truncated (its right edge is clamped to Nyquist
            # in get_full_window, when is_last and
            # spacing='nyquist').
            #
            # The user must have set f_max explicitly;
            # the __init__ raises ValueError if frange[1]
            # is +inf.  We still defensively check it here.
            if not math.isfinite(maxf) or maxf > self.sample_rate / 2:
                raise ValueError(
                    f"QTAM: spacing='nyquist' requires "
                    f"0 < f_max <= Nyquist = {self.sample_rate/2} Hz, "
                    f"got f_max = {maxf}."
                )
            freqs, meta = principled_geometric_freqs(
                q=self.q,
                frange=[grid_minf, maxf],
                duration=self.duration,
                qprime=self.qprime,
                return_meta=True,
            )
            return torch.unique(freqs)

        # Path 4 (default): 'geometric' mode (principled)
        freqs, meta = principled_geometric_freqs(
            q=self.q,
            frange=[grid_minf, maxf],
            duration=self.duration,
            qprime=self.qprime,
            return_meta=True,
        )
        return torch.unique(freqs)

    def _check_coverage(self, frange: Optional[List[float]] = None):
        """Check actual windows over ``frange`` (default: requested band).

        Passing ``[0, Nyquist]`` answers the separate question of whether the
        original full-band real signal is invertible; a finite user frange can
        be perfectly covered while intentionally discarding frequencies outside
        that band.
        """
        if frange is None:
            frange = list(self.frange)
        else:
            frange = [float(frange[0]), float(frange[1])]
        if not torch.is_tensor(self.freqs):
            self.freqs = torch.as_tensor(self.freqs)

        window_left_edges = None
        window_right_edges = None
        if hasattr(self, 'qtile_transforms') and len(self.qtile_transforms) == len(self.freqs):
            window_left_edges = []
            window_right_edges = []
            n_actual = len(self.freqs)
            for k, qt in enumerate(self.qtile_transforms):
                fk = float(self.freqs[k].item())
                L = int(qt.windowsize)
                T = float(qt.duration)
                half_bw_hz = L / (2.0 * T)
                left = fk - half_bw_hz
                right = fk + half_bw_hz
                if k == 0 and getattr(qt, 'from_0', False):
                    left = 0.0
                if k == n_actual - 1 and (
                    self.spacing == "nyquist" or self._synthesis_to_nyquist
                ):
                    right = min(right, float(self.sample_rate) / 2.0)
                window_left_edges.append(float(left))
                window_right_edges.append(float(right))

        report = check_window_coverage(
            freqs=self.freqs,
            q=self.q,
            duration=self.duration,
            qprime=self.qprime,
            frange=frange,
            window_left_edges=window_left_edges,
            window_right_edges=window_right_edges,
            spacing=self.spacing,
            nyquist=self.sample_rate / 2.0 if self.spacing == "nyquist" else None,
        )

        # The geometric edge test above is a useful continuous-frequency
        # design test, but it cannot see a one-FFT-bin hole caused by integer
        # centre rounding or by a taper that is exactly zero at its endpoints.
        # Synthesis divides by sum_i |W_i[k]|^2, so this discrete test is the
        # authoritative invertibility condition for the window bank.
        if hasattr(self, 'qtile_transforms') and len(self.qtile_transforms):
            all_windows = torch.stack([
                qt.full_window.squeeze() for qt in self.qtile_transforms
            ])
            denominator = torch.sum(all_windows.square(), dim=0)
            missing = torch.nonzero(denominator <= 0, as_tuple=False).flatten()
            # Restrict the discrete test to the requested band.  A finite
            # frange is allowed to discard bins outside it; those must not
            # make the *band-limited* coverage verdict fail.
            first_bin = max(0, int(math.ceil(float(frange[0]) * self.duration)))
            last_bin = min(
                denominator.numel() - 1,
                int(math.floor(float(frange[1]) * self.duration)),
            )
            missing = missing[(missing >= first_bin) & (missing <= last_bin)]
            if missing.numel() > 0:
                missing_bins = missing.detach().cpu().tolist()
                runs = []
                start = previous = missing_bins[0]
                for b in missing_bins[1:]:
                    if b != previous + 1:
                        runs.append((start, previous))
                        start = b
                    previous = b
                runs.append((start, previous))

                # Express a missing bin run as a half-open Hz interval.
                # This makes the reported width agree with the FFT bin width.
                segments = [
                    (a / self.duration, (b + 1) / self.duration)
                    for a, b in runs
                ]
                report["covered"] = False
                report["bad_pairs"] = list(range(len(segments)))
                report["bad_pairs_gaps"] = [right - left for left, right in segments]
                report["bad_segments"] = segments
                report["first_window_does_not_reach_fmin"] = (first_bin in missing_bins)
                report["last_window_does_not_reach_fmax"] = (last_bin in missing_bins)
                report["missing_fft_bins"] = missing_bins

        return report

    def _report_coverage(self):
        """Emit a warning (or raise) if the coverage check failed."""
        report = self._coverage_report
        if report["covered"]:
            return

        n_gaps = len(report.get("bad_segments", []))
        first = report.get("bad_segments", [None])[0]
        where = (f" First gap: [{first[0]:.3f}, {first[1]:.3f}] Hz."
                 if first is not None else "")
        msg = (
            f"QTAM: window-bank coverage failed ({n_gaps} gap(s)).{where} "
            "Increase max_window_size and/or num_freq, then run diagnose() "
            "for the full report."
        )

        if self.raise_on_bad_coverage:
            raise RuntimeError(msg)
        if self.warn_on_bad_coverage:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    def _get_num_freq_info(self) -> dict:
        """Return the theoretical geometric-grid density requirement.

        The requirement is expressed as a number of *points*, whereas the
        constant-Q formula first gives a number of intervals.  It applies to
        logarithmic/geometric grids; a user-selected linear grid has no single
        bins-per-octave criterion and is checked from its actual windows.
        """
        grid_min = float(self.center_frange[0])
        grid_max = float(self.center_frange[1])
        applicable = (
            (self.spacing in ("geometric", "nyquist")) or bool(self.logf)
        ) and math.isfinite(grid_min) and math.isfinite(grid_max) and grid_max > grid_min

        if not applicable:
            return {
                "applicable": False,
                "minimum_points": None,
                "minimum_intervals": None,
                "bins_per_octave_min": None,
                "grid_min": grid_min,
                "grid_max": grid_max,
                "requested_points": int(self.num_freq),
                "actual_points": int(len(self.freqs)),
                "sufficient": None,
            }

        bins_per_octave_min = max(1, int(math.ceil(self.q * math.log(2))))
        minimum_intervals = max(
            1, int(math.ceil(bins_per_octave_min * math.log2(grid_max / grid_min)))
        )
        minimum_points = minimum_intervals + 1
        # num_freq == 0 means automatic selection, not zero rows.
        # Coverage is governed by the points that survived grid construction
        # (``torch.unique`` can theoretically remove duplicate requested ones).
        requested_or_actual = int(len(self.freqs))
        return {
            "applicable": True,
            "minimum_points": minimum_points,
            "minimum_intervals": minimum_intervals,
            "bins_per_octave_min": bins_per_octave_min,
            "grid_min": grid_min,
            "grid_max": grid_max,
            "requested_points": int(self.num_freq),
            "actual_points": int(len(self.freqs)),
            "sufficient": requested_or_actual >= minimum_points,
        }

    def _report_num_freq_requirement(self) -> None:
        """Keep construction quiet; diagnose() reports the density reference.

        The q*ln(2) value is conservative rather than an exact failure
        criterion for the discrete qprime-based bank.  Emitting it during
        construction is noisy and can be misleading; diagnose() presents it
        next to the authoritative coverage verdict instead.
        """
        return

    def get_max_window_size(self, max_w_s: Optional[Union[int, str]]):
        """
        Determines the final max_window_size based on user input.
        """
        # Case 1: No cap provided (pure CQT)
        if max_w_s is None:
            return None

        # Case 2: User provides a specific integer value
        if isinstance(max_w_s, (int, float)):
            max_size = int(max_w_s)

            # Compare the requested cap with the endpoint-safe discrete
            # requirement.  The concise warning below is the only
            # construction-time cap message; diagnose() gives the details.
            # Calculate the principled minimum size for comparison
            if len(self.freqs) > 1:
                # Windows such as the default bisquare/Hann taper are zero
                # at both endpoints.  Thus an odd L-bin window has only
                # L-2 nonzero bins.  Work on the *actual integer FFT
                # centres*, rather than continuous Hz gaps, and require an
                # endpoint-safe overlap.  The smallest odd L satisfying this
                # is the smallest odd integer >= max_centre_gap + 2.
                centre_bins = [int(float(f) * self.duration)
                               for f in self.freqs]
                max_centre_gap = max(
                    b - a for a, b in zip(centre_bins[:-1], centre_bins[1:])
                )
                principled_min_size = max_centre_gap + 2
                if principled_min_size % 2 == 0:
                    principled_min_size += 1
                if max_size < principled_min_size:
                    warnings.warn(
                        "\033[91m[Warning]\033[0m "
                        f"User-defined cap ({max_size}) is smaller than the "
                        f"principled minimum ({principled_min_size}) required "
                        "to guarantee no frequency gaps. This may affect "
                        "invertibility.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            return max_size
            
        # Case 3: User requests principled calculation
        if isinstance(max_w_s, str) and max_w_s.lower() == 'auto':
            if len(self.freqs) > 1:
                freq_spacings = torch.diff(self.freqs)
                max_spacing_hz = torch.max(freq_spacings)
                max_size = (math.ceil(self.duration * max_spacing_hz.item())//2)*2+1
                return max_size
            else:
                return None
        
        raise ValueError(f"Invalid input for max_window_size. Must be None, an integer, or 'auto'. Got: {max_w_s}")

    def get_max_energy(
        self, fsearch_range: List[float] = None, dimension: str = "both"
    ):
        """
        Gets the maximum energy value among the QTiles. The maximum can
        be computed across all batches and channels, across all channels,
        across all batches, or individually for each channel/batch
        combination. This could be useful for allowing the use of different
        Q values for different channels and batches, but the slicing would
        be slow, so this isn't used yet.

        Optionally, a pair of frequency values can be specified for
        `fsearch_range` to restrict the frequencies in which the maximum
        energy value is sought.
        """
        allowed_dimensions = ["both", "neither", "channel", "batch"]
        if dimension not in allowed_dimensions:
            raise ValueError(f"Dimension must be one of {allowed_dimensions}")

        if self.qtiles is None:
            raise RuntimeError(
                "Q-tiles must first be computed with .compute_qtiles()"
            )

        if fsearch_range is not None:
            start = min(torch.argwhere(self.freqs > fsearch_range[0]))
            stop = min(torch.argwhere(self.freqs > fsearch_range[1]))
            qtiles = self.qtiles[start:stop]
        else:
            qtiles = self.qtiles

        if dimension == "both":
            return max([torch.max(qtile) for qtile in qtiles])

        max_across_t = [torch.max(qtile, dim=-1).values for qtile in qtiles]
        max_across_t = torch.stack(max_across_t, dim=-1)
        max_across_ft = torch.max(max_across_t, dim=-1).values

        if dimension == "neither":
            return max_across_ft
        if dimension == "channel":
            return torch.max(max_across_ft, dim=-2).values
        if dimension == "batch":
            return torch.max(max_across_ft, dim=-1).values


    def diagnose(
        self,
        fig_file: Optional[str] = None,
        show: bool = True,
    ) -> dict:
        """
        Diagnose the window bank of *this* SingleQTransform.

        This is a configuration-aware diagnostic: it works on the
        *specific* parameters the user passed to ``__init__`` --
        ``q``, ``duration``, ``sample_rate``, ``frange``,
        ``num_freq``, ``logf``, ``spacing``, ``mismatch`` and
        ``max_window_size``.  It reports:

        1. **The effective config** the transform is using (after
           auto-set of ``f_min`` / ``f_max``), and how it relates
           to the user's input.
        2. **Whether the actual grid covers the analysis band**:
           inter-window overlaps, first/last window reach.
        3. **What ``max_window_size`` is doing** (whether it is
           clamping any window, and to what size).

        If ``max_window_size`` is active, the report flags which
        windows are clamped and by how much.

        If matplotlib is installed, the method also saves a
        single-panel figure showing the user's actual grid
        supports and the analysis band (with the coverage gaps
        highlighted in red if any).  No "what-if" or comparison
        scenarios are run -- the diagnostic is strictly about
        the user's current configuration.

        For the 5-mode comparison (geometric, mismatch, linear,
        etc.), run ``coverage_test.py`` directly.

        Parameters
        ----------
        fig_file : str, optional
            Path to save the figure.  Default: ``None`` (saves
            to ``qtam_diagnose.png`` in the current working
            directory).
        show : bool, optional
            If True (default), call ``plt.show()`` to display
            the figure.  Set to False for non-interactive
            environments (headless servers, CI).  When False,
            matplotlib is forced to the ``Agg`` backend so the
            figure can be saved without a display.  When True,
            the user's currently-selected matplotlib backend is
            preserved (so ``%matplotlib inline`` etc. work as
            expected).

        Returns
        -------
        report : dict
            Nested dict with the user's config, the diagnostic
            for the actual grid, the ``max_window_size`` info,
            and the figure object.  See source for full
            structure.
        """
        # ---- Config from the actual transform (post-auto-set) ----
        # Note: the user's `frange[0]` is kept as-is (we do NOT
        # auto-set it to a "principled minimum").  Invertibility
        # requires the transform to be defined all the way down
        # to DC if the user asks for f_min=0; the first QTile
        # activates the DC patch (mirroring the first non-zero
        # value of its window down to DC) to cover the [0, f_center]
        # range.  The effective f_min for the diagnostic is therefore
        # either 0.0 (if from_0) or the user's value.
        f_min_eff = float(self.frange[0])
        f_max_eff = float(self.frange[1])
        if not math.isfinite(f_min_eff):
            raise ValueError(
                f"diagnose(): effective f_min must be finite, "
                f"got {f_min_eff}"
            )
        if not math.isfinite(f_max_eff):
            raise ValueError(
                f"diagnose(): effective f_max must be finite, "
                f"got {f_max_eff}"
            )
        if f_max_eff <= f_min_eff:
            raise ValueError(
                f"diagnose(): effective f_max must be > f_min, "
                f"got [{f_min_eff}, {f_max_eff}]"
            )
        nyquist = float(self.sample_rate) / 2.0

        # Detect what was auto-set.  We use the saved _frange_input
        # to know what the user actually passed: if the user
        # passed <= 0 (or a non-finite number) for f_min, the
        # auto-set branch in __init__ triggered; similarly for
        # f_max = +inf.
        frange_input = list(getattr(self, "_frange_input", self.frange))
        f_min_user = frange_input[0]
        f_max_user = frange_input[1]
        # We do NOT auto-set f_min to a "principled minimum"
        # anymore (the user explicitly said this breaks
        # invertibility -- if they want f_min=0, we keep f_min=0
        # and let the DC patch handle the [0, f_center] range).
        # The principled value is still reported for reference.
        f_min_auto_value = self.qprime / self.duration
        f_min_was_auto_set = False
        # The auto-set f_max formula: Nyquist / (1 + 1/(2*qprime))
        f_max_auto_value = nyquist / (1 + 1 / (2 * self.qprime))
        f_max_was_auto_set = (
            self.spacing != "nyquist"
            and (not math.isfinite(f_max_user))
        )

        config = {
            "q": float(self.q),
            "duration": float(self.duration),
            "sample_rate": float(self.sample_rate),
            "frange_input": frange_input,
            "frange_effective": [f_min_eff, f_max_eff],
            "center_frange": [float(self.center_frange[0]), float(self.center_frange[1])],
            "num_freq": int(self.num_freq),
            "logf": bool(self.logf),
            "spacing": str(self.spacing),
            "mismatch": float(self.mismatch),
            "qprime": float(self.qprime),
            "max_window_size": (
                int(self.max_window_size)
                if self.max_window_size is not None else None
            ),
            "nyquist": nyquist,
            "f_min_was_auto_set": f_min_was_auto_set,
            "f_min_auto_value": f_min_auto_value,
            "f_max_was_auto_set": f_max_was_auto_set,
            "f_max_auto_value": f_max_auto_value,
        }

        # ---- Section 1: Print the actual config ----
        def _banner(title):
            line = "=" * 72
            print()
            print(line)
            print(title)
            print(line)

        def _kv(label, value):
            print(f"  {label:<35} : {value}")

        _banner("YOUR CONFIG (as passed to SingleQTransform.__init__)")
        _kv("q", config["q"])
        _kv("duration", f"{config['duration']} s")
        _kv("sample_rate (f_s)", f"{config['sample_rate']} Hz")
        _kv("Nyquist", f"{config['nyquist']} Hz")
        _kv("frange (user input)", frange_input)
        if self.from_0:
            if f_min_user <= 0 or not math.isfinite(f_min_user):
                _kv("  f_min (DC patch active on first tile, reaching 0 Hz)",
                    f"{f_min_eff} Hz  (principled reference: q'/T = "
                    f"{f_min_auto_value:.3f} Hz)")
            else:
                _kv("  f_min (user-specified; DC patch active on first tile, reaching 0 Hz)",
                    f"{f_min_eff} Hz")
        else:
            _kv("  f_min (user-specified)", f"{f_min_eff} Hz")
        if f_max_was_auto_set:
            _kv("  analysis f_max (auto-set from +inf)",
                f"{f_max_eff:.3f} Hz (= Nyquist)")
            _kv("  maximum tile-centre frequency (auto)",
                f"{self.center_frange[1]:.3f} Hz "
                f"(Nyquist/(1+1/(2*q')))" )
        else:
            _kv("  analysis f_max (user-specified)", f"{f_max_eff} Hz")
            _kv("  maximum tile-centre frequency", f"{self.center_frange[1]:.3f} Hz")
        _kv("num_freq", 
            config["num_freq"] if config["num_freq"] > 0 else "auto")
        _kv("logf", config["logf"])
        _kv("spacing", config["spacing"])
        _kv("mismatch", config["mismatch"])
        _kv("max_window_size",
            config["max_window_size"] if config["max_window_size"] is not None
            else "None (pure CQT)")
        _kv("n_windows (actual)", len(self.freqs))

        num_freq_info = self._get_num_freq_info()
        if num_freq_info["applicable"]:
            requested = (num_freq_info["requested_points"]
                         if num_freq_info["requested_points"] > 0 else "auto")
            reference = num_freq_info["minimum_points"]
            if num_freq_info["sufficient"]:
                print(f"  [OK]  Grid density: {requested} centres (reference: {reference}).")
            else:
                print(f"  [NOTE] Grid density: {requested} centres (conservative reference: {reference}).")
                print("         Exact coverage below is the decisive invertibility test.")
        if self.spacing == "nyquist":
            # In nyquist mode the f_max was explicitly given.
            # The natural right edge of the last window is
            # f_max * (1 + 1/(2*q')) and may exceed Nyquist; the
            # production code clamps it at Nyquist in get_full_window.
            last_right_natural = f_max_eff * (1 + 1 / (2 * self.qprime))
            last_right_clamped = min(last_right_natural, nyquist)
            _kv("Last window: center",
                f"{f_max_eff} Hz")
            _kv("Last window: natural right edge",
                f"{last_right_natural:.3f} Hz")
            _kv("Last window: clamped right edge",
                f"{last_right_clamped:.3f} Hz (= Nyquist)")
            _kv("Last window: truncation gap (natural - clamped)",
                f"{last_right_natural - last_right_clamped:.3f} Hz")

        # ---- Section 2: Diagnostic on the actual grid ----
        # Note: we deliberately do NOT print "First gap" / "Last gap"
        # values, because they are the inter-center spacings (which
        # are non-zero by construction on a geometric grid) and
        # NOT coverage gaps.  The coverage verdict below is the
        # only number the user needs to read.
        def _print_band(label, freqs, frange_):
            fmin, fmax = float(frange_[0]), float(frange_[1])
            print(f"  {label}: {len(freqs)} centre frequencies")
            print(f"    centres : {float(freqs[0]):.3f} -- {float(freqs[-1]):.3f} Hz")
            print(f"    band    : [{fmin:.3f}, {fmax:.3f}] Hz")

        def _print_coverage(label, report):
            if report["covered"]:
                print(f"  [OK]  Coverage: {label}")
                return
            n = len(report.get("bad_segments", []))
            seg = report.get("bad_segments", [None])[0]
            detail = ""
            if seg is not None:
                detail = f"; first gap [{seg[0]:.3f}, {seg[1]:.3f}] Hz"
            print(f"  \033[91m[Warning]\033[0m Coverage: {label} -- {n} gap(s){detail}")

        def _coverage_kwargs(spacing):
            kw = {"spacing": spacing}
            if spacing == "nyquist":
                kw["nyquist"] = nyquist
            return kw

        # The user's actual grid
        actual_freqs = self.freqs
        if not torch.is_tensor(actual_freqs):
            actual_freqs = torch.as_tensor(actual_freqs)

        # Compute the EFFECTIVE left/right edges of every qtile's
        # window.  We use the *theoretical* (centered) support
        # `[f - L/(2T), f + L/(2T)]` rather than the integer-shift-
        # based support, because:
        #
        # 1. The integer-shift rounding in production QTAM
        #    (`shift = int(f * T)`) introduces an off-by-up-to-
        #    1-sample artifact, which makes very small windows
        #    (e.g. 3 samples at low f) look off-center on a log
        #    plot.  The conceptual support is `[f - L/(2T),
        #    f + L/(2T)]`, exactly centered on f.
        #
        # 2. The integer-shift rounding is bounded by 1 sample
        #    = 1/T = (sample_rate/2) / N Nyquist, which is small
        #    compared to the band -- so using the centered
        #    support is a good approximation for both the
        #    coverage check and the plot.
        #
        # We then apply three corrections:
        #  - DC patch on the first tile (left edge -> 0)
        #  - Nyquist clamp on the last tile in 'nyquist' mode
        #  - max_window_size cap reflected in the qtile.windowsize
        n_actual = int(actual_freqs.shape[0])
        actual_left_edges = []
        actual_right_edges = []
        for k, qt in enumerate(self.qtile_transforms):
            fk = float(actual_freqs[k].item())
            L = int(qt.windowsize)
            T = float(qt.duration)
            half_bw_hz = L / (2.0 * T)
            left = fk - half_bw_hz
            right = fk + half_bw_hz
            # Apply DC patch for the first tile
            if k == 0 and getattr(qt, 'from_0', False):
                left = 0.0
            # Apply Nyquist clamp for the last tile in nyquist mode
            if k == n_actual - 1 and (
                self.spacing == "nyquist" or self._synthesis_to_nyquist
            ):
                right = min(right, float(self.sample_rate) / 2.0)
            actual_left_edges.append(float(left))
            actual_right_edges.append(float(right))

        # Use the same coverage report as construction.  In addition to
        # continuous support edges, it checks the exact FFT-bin synthesis
        # denominator, which catches endpoint-zero / integer-rounding holes.
        requested_band_coverage = self._check_coverage([f_min_eff, f_max_eff])
        full_spectrum_coverage = self._check_coverage([0.0, nyquist])
        # Backward-compatible name used by plotting/returned diagnostics:
        # `actual_coverage` always means the user-requested analysis band.
        actual_coverage = requested_band_coverage

        _banner("YOUR ACTUAL GRID (the grid the transform is using)")
        _print_band("actual grid", actual_freqs, [f_min_eff, f_max_eff])
        same_as_full_band = (abs(f_min_eff) < 1e-12 and abs(f_max_eff - nyquist) < 1e-12)
        if same_as_full_band:
            _print_coverage(f"full spectrum [0, {nyquist:.3f}] Hz", full_spectrum_coverage)
        else:
            _print_coverage(
                f"requested band [{f_min_eff:.3f}, {f_max_eff:.3f}] Hz",
                requested_band_coverage,
            )
            _print_coverage(
                f"full spectrum [0, {nyquist:.3f}] Hz",
                full_spectrum_coverage,
            )
            if requested_band_coverage["covered"] and not full_spectrum_coverage["covered"]:
                print("  [NOTE] The requested band is invertible; the original")
                print("         full-band signal is not, because the remaining")
                print("         frequencies are intentionally discarded.")

        # ---- Section 3: max_window_size info ----
        max_ws_info = {
            "active": config["max_window_size"] is not None,
            "user_value": config["max_window_size"],
            "principled_min": None,
            "clamped_windows": [],
        }
        if config["max_window_size"] is not None and len(actual_freqs) > 1:
            # Endpoint-safe discrete-bin bound.  The standard windows used
            # here vanish at their two endpoints, so L nominal samples give
            # only L-2 nonzero FFT bins.  Compute from the integer centres
            # actually used by QTile (`int(f*T)`), then choose the smallest
            # odd L >= largest centre separation + 2.
            centre_bins = [int(float(f) * self.duration)
                           for f in actual_freqs.tolist()]
            max_centre_gap = max(
                b - a for a, b in zip(centre_bins[:-1], centre_bins[1:])
            )
            principled_min_size = max_centre_gap + 2
            if principled_min_size % 2 == 0:
                principled_min_size += 1
            max_ws_info["principled_min"] = principled_min_size
            # Find which windows are actually clamped.  A window
            # at frequency f has natural size
            # L = 2 * int(f * T / qprime) + 1; it's clamped iff
            # L > max_window_size.
            clamped = []
            for fk in actual_freqs.tolist():
                L_natural = 2 * int(fk * self.duration / self.qprime) + 1
                if L_natural > config["max_window_size"]:
                    clamped.append((float(fk), L_natural, config["max_window_size"]))
            max_ws_info["clamped_windows"] = clamped

            _banner("WINDOW CAP")
            cap = max_ws_info["user_value"]
            if cap < principled_min_size:
                print(f"  \033[91m[Warning]\033[0m cap={cap}; endpoint-safe recommendation={principled_min_size}.")
                print("            This cap can create spectral gaps.")
            else:
                print(f"  [OK]  cap={cap}; endpoint-safe recommendation={principled_min_size}.")
            print(f"  clamped tiles: {len(clamped)}/{len(actual_freqs)}")

        # ---- Section 4: Plotting (single panel, user's actual grid) ----
        fig_obj = {"fig": None, "saved_to": None}
        try:
            import matplotlib
            # ✅ Backend selection: when show=False (headless / CI),
            # force Agg so fig.savefig() works.  When show=True, do
            # NOT call matplotlib.use() at all -- the user has
            # already chosen a backend via %matplotlib inline /
            # %matplotlib notebook / matplotlib.use("Qt5Agg") etc.,
            # and overwriting it here with `None` would crash newer
            # matplotlib versions (see matplotlib >= 3.7 where
            # `matplotlib.use(None)` raises AttributeError).
            if not show:
                matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import numpy as np

            LOG_FLOOR_HZ = 1.0
            fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=110)

            freqs = actual_freqs
            coverage_ = actual_coverage
            spacing_ = self.spacing
            title = "Your actual config"
            fmin_p, fmax_p = float(f_min_eff), float(f_max_eff)
            n = len(freqs)
            qp = self.qprime
            dc_patch = (fmin_p <= 0) or (fmin_p < qp / self.duration)
            # ✅ Use the ACTUAL QTile.full_window support (which
            # reflects the max_window_size cap, the DC patch on the
            # first tile, and the Nyquist clamp on the last tile
            # in 'nyquist' mode), not the theoretical CQT support
            # `[fk ± fk/(2q')]`.  The theoretical support can be
            # very different from the actual support when a
            # max_window_size cap is in effect (the cap shrinks
            # the high-frequency windows, which is invisible on
            # the plot if we use the theoretical edges).
            #
            # We use the *centered* support `[f - L/(2T), f + L/(2T)]`
            # (rather than the integer-shift-based support) so
            # the windows appear centered on their nominal
            # frequency on a log plot.
            left_edges = []
            right_edges = []
            for k, qt in enumerate(self.qtile_transforms):
                # The qtile's window has `windowsize` samples,
                # placed symmetrically around the center frequency
                # `f`.  The effective support in Hz is
                # `[f - L/(2T), f + L/(2T)]`.
                fk = float(freqs[k].item()) if hasattr(freqs[k], 'item') else float(freqs[k])
                L = int(qt.windowsize)
                T = float(qt.duration)
                half_bw_hz = L / (2.0 * T)
                left = fk - half_bw_hz
                right = fk + half_bw_hz
                # Apply DC patch for the first tile
                if k == 0 and getattr(qt, 'from_0', False):
                    left = 0.0
                # Apply Nyquist clamp for the last tile in nyquist mode
                if k == n - 1 and (
                    spacing_ == "nyquist" or self._synthesis_to_nyquist
                ):
                    right = min(right, float(self.sample_rate) / 2.0)
                left_edges.append(left)
                right_edges.append(right)

            full_y_lo, full_y_hi = -0.5, float(n) - 0.5

            def _shade(x_lo, x_hi, y_lo, y_hi, color, alpha, zorder):
                if x_lo >= x_hi:
                    return
                verts_x = np.linspace(
                    max(x_lo, LOG_FLOOR_HZ), max(x_hi, LOG_FLOOR_HZ), 50
                )
                verts = (
                    [(x, y_lo) for x in verts_x]
                    + [(x, y_hi) for x in verts_x[::-1]]
                )
                ax.add_patch(
                    mpatches.Polygon(
                        verts, closed=True, facecolor=color,
                        alpha=alpha, edgecolor="none", zorder=zorder,
                    )
                )

            if left_edges[0] > fmin_p:
                _shade(max(fmin_p, LOG_FLOOR_HZ), left_edges[0],
                       full_y_lo, full_y_hi, "red", 0.35, 2)
            for k in range(n - 1):
                if right_edges[k] < left_edges[k + 1]:
                    _shade(right_edges[k], left_edges[k + 1],
                           full_y_lo, full_y_hi, "red", 0.35, 2)
            if right_edges[-1] < fmax_p:
                _shade(right_edges[-1], fmax_p,
                       full_y_lo, full_y_hi, "red", 0.35, 2)

            if left_edges[0] > fmin_p:
                for x_edge in (fmin_p, left_edges[0]):
                    ax.plot([x_edge, x_edge], [full_y_lo, full_y_hi],
                            color="red", linestyle="--",
                            linewidth=0.7, zorder=4, alpha=0.7)
            for k in range(n - 1):
                if right_edges[k] < left_edges[k + 1]:
                    for x_edge in (right_edges[k], left_edges[k + 1]):
                        ax.plot([x_edge, x_edge],
                                [full_y_lo, full_y_hi],
                                color="red", linestyle="--",
                                linewidth=0.7, zorder=4, alpha=0.7)
            if right_edges[-1] < fmax_p:
                for x_edge in (right_edges[-1], fmax_p):
                    ax.plot([x_edge, x_edge], [full_y_lo, full_y_hi],
                            color="red", linestyle="--",
                            linewidth=0.7, zorder=4, alpha=0.7)

            # A logarithmic axis cannot represent DC.  Do not pass x=0
            # to Matplotlib: depending on backend/version it can make the
            # saved figure appear blank.  Draw the DC band edge at the
            # visible floor and label it explicitly as DC.
            fmin_draw = max(fmin_p, LOG_FLOOR_HZ)
            fmin_label = ("f_min=0 Hz (DC)" if fmin_p <= 0
                          else f"f_min={fmin_p:g} Hz")
            ax.plot([fmin_draw, fmin_draw], [-0.5, n - 0.5],
                    color="black", linestyle="-", linewidth=1.2,
                    zorder=2)
            ax.plot([fmax_p, fmax_p], [-0.5, n - 0.5],
                    color="black", linestyle="-", linewidth=1.2,
                    zorder=2)
            ax.text(fmin_draw, n - 0.2, fmin_label,
                    ha="left", va="bottom", fontsize=8,
                    color="black", zorder=6)
            ax.text(fmax_p, n - 0.2, f"f_max={fmax_p:g} Hz",
                    ha="right", va="bottom", fontsize=8,
                    color="black", zorder=6)
            if spacing_ == "nyquist" and nyquist is not None and nyquist < fmax_p:
                ax.axvline(nyquist, color="purple", linestyle=":",
                           linewidth=0.9, alpha=0.7, zorder=2)
                ax.text(nyquist, n - 0.2, f"  Nyquist={nyquist:g} Hz",
                        ha="left", va="bottom", fontsize=8,
                        color="purple", zorder=6)

            colors = plt.cm.viridis(np.linspace(0, 1, max(n, 2)))
            for k, fk in enumerate(freqs.tolist()):
                left = max(left_edges[k], LOG_FLOOR_HZ)
                right = right_edges[k]
                ax.plot([left, right], [k, k],
                        color=colors[k], linewidth=3.5,
                        solid_capstyle="butt", zorder=3)
                ax.plot([fk, fk], [k - 0.35, k + 0.35],
                        color="black", linewidth=0.6, zorder=4)
                ax.text(fk, k - 0.4, f"{fk:.1f}",
                        ha="center", va="top", fontsize=6,
                        color=colors[k], zorder=4)

            ax.set_xscale("log")
            # The automatic geometric f_max is a centre frequency; its
            # last retained window extends to Nyquist for synthesis.
            f_plot_max = (nyquist if self._synthesis_to_nyquist else fmax_p)
            ax.set_xlim(LOG_FLOOR_HZ, f_plot_max * 1.1)
            ax.set_ylim(-0.7, n + 0.3)
            ax.set_xlabel("Frequency [Hz]  (log scale)", fontsize=10)
            ax.set_ylabel("Window index k", fontsize=10)
            ax.set_title(title, fontsize=11)
            ax.grid(True, which="both", alpha=0.3)

            if coverage_["covered"]:
                status = f"COVERED ({n} windows, all gaps are 0 Hz)"
                color = "green"
            else:
                n_bad = len(coverage_.get("bad_pairs", []))
                worst = (max(coverage_["bad_pairs_gaps"])
                         if coverage_.get("bad_pairs_gaps") else 0.0)
                status = (f"NOT COVERED ({n_bad} inter-window gaps, "
                          f"worst overshoot = {worst:.2f} Hz)")
                color = "red"
            ax.text(0.98, 0.98, status, transform=ax.transAxes,
                    fontsize=10, verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round",
                              facecolor=color, alpha=0.2))

            legend_handles = [
                mpatches.Patch(facecolor="lightgreen", alpha=0.5,
                               label="Covered region (spectrum tile)"),
                mpatches.Patch(facecolor="salmon", alpha=0.5,
                               label="Uncovered region (gap)"),
                plt.Line2D([0], [0], color="black", linewidth=4,
                           label="CQT window support (actual, incl. cap)"),
                plt.Line2D([0], [0], color="red", linestyle="--",
                           label="Edge of coverage gap"),
                plt.Line2D([0], [0], color="black", linestyle="-",
                           linewidth=0.8,
                           label="f_min / f_max (band edges)"),
            ]
            ax.legend(handles=legend_handles, loc="upper left",
                      ncol=1, fontsize=7, frameon=True,
                      bbox_to_anchor=(0.02, 0.95))

            # Use the actual lowest frequency in the grid for
            # the octave count (when f_min=0 the grid starts
            # at qprime/T, not 0).
            f_min_for_octaves = f_min_eff
            if f_min_for_octaves <= 0 and self.freqs is not None and len(self.freqs) > 0:
                f_min_for_octaves = float(self.freqs[0].item())
            if f_min_for_octaves > 0 and f_max_eff > f_min_for_octaves:
                octaves_str = f"{math.log2(f_max_eff / f_min_for_octaves):.2f} octaves"
            else:
                octaves_str = "(degenerate band)"
            band = (f"band=[{f_min_eff:.3f}, {f_max_eff:.3f}] Hz, "
                    f"{octaves_str}")
            fig.suptitle(
                f"QTAM diagnose  (q={self.q}, {band})",
                fontsize=12, y=1.02,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])

            out_path = fig_file if fig_file else "qtam_diagnose.png"
            try:
                fig.savefig(out_path, dpi=110, bbox_inches="tight")
                fig_obj["saved_to"] = out_path
                print()
                print(f"  Figure saved to: {out_path}")
            except Exception as exc:
                print(f"  (could not save figure: {exc})")
            fig_obj["fig"] = fig

            if show:
                try:
                    plt.show()
                except Exception as exc:
                    print(f"  (could not display figure: {exc})")
            else:
                plt.close(fig)

        except ImportError:
            warnings.warn(
                "QTAM.diagnose(): matplotlib is not installed; "
                "skipping the figure. Install matplotlib "
                "(`pip install matplotlib`) to enable the figure.",
                RuntimeWarning,
            )

        return {
            "config": config,
            "actual": {
                "freqs": actual_freqs.detach().clone(),
                "coverage": actual_coverage,
                "requested_band_coverage": requested_band_coverage,
                "full_spectrum_coverage": full_spectrum_coverage,
                "n_windows": int(actual_freqs.shape[0]),
            },
            "num_freq_info": num_freq_info,
            "max_window_size_info": max_ws_info,
            "figure": fig_obj,
        }

    def compute_qtiles(
        self,
        X: torch.Tensor,
        polar_mode: bool = True, 
        energy_mode: bool = True, 
        phase_mode: bool = True, 
        complex_mode: bool = False,
        num_time: Optional[int] = None, 
        am_mode: bool = True           
    ):
        """
        Take the FFT of the input timeseries and calculate the transform
        for each `QTile`
        """
        X_fft = torch.fft.rfft(X, norm='ortho')
        
        # Pass the new arguments to each QTile's forward method
        self.qtiles = [
            qtile(
                X_fft, polar_mode, energy_mode, phase_mode, complex_mode,
                num_time=num_time, am_mode=am_mode
            ) 
            for qtile in self.qtile_transforms
        ]
        
#------------------------------------------------------------------------------------------------------------------------------------
    
    ### SPLINE INTERPOLATION ###
    def interpolate(self, Z, num_f_bins, num_t_bins, polar_mode: bool = True, phase_mode: bool = True, complex_mode: bool = False):
        
        '''
            WARNING: 
            - Spline inteprolation is not invertible, for invertible down/upsampling use XXX
            - Interpolation of phase is not really meaningfull because of Aliasing. It is particularly bad for polar_mode=False. 
              For a more sound phase interpolation use: XXX
                    
        '''
        
        device= Z.device

        #Devide input into Re/IM or Amp/Phase depending on modes
        if (not polar_mode): 
            if complex_mode:
                A = Z.real
                P = Z.imag
            else:
                A = Z[:,:,0,:,:]
                P = Z[:,:,1,:,:]
        else:
            A = Z[:,:,0,:,:]
            if phase_mode:
                P = Z[:,:,1,:,:]
                
        # Build grids for natural bicubic spline
        xin = torch.linspace(0.0,self.duration, steps=Z.shape[-1])
        xout= torch.linspace(0.0,self.duration, steps=num_t_bins)
        
        if self.logf:
            yout = torch.tensor(np.geomspace(
                self.frange[0],
                self.frange[1],
                num=num_f_bins,))
        else:
            yout=torch.linspace(self.frange[0], self.frange[1],num_f_bins)
        
        #define NN for 2d interpolation
        spline_interpolate_2d=SplineInterpolate2D(num_t_bins=num_t_bins, num_f_bins=num_f_bins,logf=self.logf,frange=self.frange).to(device)
        
        #interpolate Qtransform
        resampled=spline_interpolate_2d(A.transpose(-1,-2),xin=xin,xout=xout,yin=self.freqs,yout=yout)    

        # Interpolate phase/imag if required
        if phase_mode or (not polar_mode):  
            
            phase_interp = spline_interpolate_2d(
                P.transpose(-1,-2), xin=xin, xout=xout,
                yin=self.freqs.to(device), yout=yout
            )

            if (not polar_mode) and complex_mode:
                return torch.complex(resampled,phase_interp)
                
            return torch.stack([resampled, phase_interp], dim=2) #.detach().cpu()  

        return resampled #.detach().cpu()
#-------------------------------------------------------------------------------------------------------------
    ### FOURIER INTERPOLATION ###
    def _centered_pad_or_crop(self, X: torch.Tensor, M: int) -> torch.Tensor:
        """Helper for ideal band-pass filtering via crop/pad in FFT domain."""
        N = X.shape[-1]
        if M == N:
            return X
        # Use F.fftshift, assuming `import torch.nn.functional as F`
        Xs = torch.fft.fftshift(X, dim=-1)
        if M > N:
            pad_left = (M - N) // 2
            pad_right = M - N - pad_left
            Y = F.pad(Xs, (pad_left, pad_right))
        else:
            start = (N - M) // 2
            end = start + M
            Y = Xs[..., start:end]
        return torch.fft.ifftshift(Y, dim=-1)

    def _row_mod(self, Zc: torch.Tensor, sign: int) -> torch.Tensor:
        """
        Apply exp(sign * i * 2π * shift_k * n / T) per frequency row k,
        computed with integer modulo for float32 stability.
        """
        B, C, n_freqs, T = Zc.shape
        device = Zc.device
        dtype = Zc.real.dtype
    
        n = torch.arange(T, device=device, dtype=torch.int64)  # [T]
    
        # collect integer shifts for each row
        shifts = torch.tensor([qt.shift for qt in self.qtile_transforms],
                              device=device, dtype=torch.int64)  # [n_freqs]
        shifts = shifts % T
    
        # m[k,n] = (shift_k * n) mod T
        m = (shifts.view(1, 1, n_freqs, 1) * n.view(1, 1, 1, T)) % T
    
        phase = (sign * (2.0 * math.pi) / T) * m.to(dtype)
        rot = torch.polar(torch.ones_like(phase), phase)
    
        return Zc * rot

    def downsample(self, Z_in, T_out: int,
               polar_mode: bool,
               energy_mode: bool,
               phase_mode: bool,
               complex_mode: bool,
               preserve_amplitude=True,
               remod: bool = False):

        T_in = Z_in.shape[-1]
        if T_in == T_out:
            return Z_in
    
        # Convert to complex
        if polar_mode:
            amplitude = torch.sqrt(Z_in[:, :, 0]) if energy_mode else Z_in[:, :, 0]
            phase = Z_in[:, :, 1] if phase_mode else torch.zeros_like(amplitude)
            Zc_in = torch.polar(amplitude, phase)
        elif complex_mode:
            Zc_in = Z_in
        else:
            Zc_in = torch.complex(Z_in[:, :, 0], Z_in[:, :, 1])
    
        Zf_in = torch.fft.fft(Zc_in, dim=-1, norm='ortho')
        C_k_ds_list = []
    
        for k, qt in enumerate(self.qtile_transforms):
    
            Zf_k = Zf_in[:, :, k, :]
    
            # ✅ Exact integer demodulation
            Zf_k_bb = torch.roll(Zf_k, shifts=-qt.shift, dims=-1)
    
            Zf_k_bb_cropped = self._centered_pad_or_crop(Zf_k_bb, T_out)
    
            C_k_ds = torch.fft.ifft(Zf_k_bb_cropped, dim=-1, norm='ortho')
            C_k_ds_list.append(C_k_ds)
    
        C_ds = torch.stack(C_k_ds_list, dim=2)
    
        if preserve_amplitude:
            C_ds *= math.sqrt(T_out / T_in)
    
        if remod:
            Zc_out = self._row_mod(C_ds, sign=+1)
        else:
            Zc_out = C_ds
    
        # Convert back
        if polar_mode:
            energy = Zc_out.abs()**2 if energy_mode else Zc_out.abs()
            if phase_mode:
                return torch.stack([energy, Zc_out.angle()], dim=2)
            return energy.unsqueeze(2)
    
        elif complex_mode:
            return Zc_out
    
        else:
            return torch.stack([Zc_out.real, Zc_out.imag], dim=2)

    def upsample(self, Z_coarse, T_in: int,
             polar_mode: bool,
             energy_mode: bool,
             phase_mode: bool,
             complex_mode: bool,
             preserve_amplitude=True,
             demod: bool = False):

        T_out = Z_coarse.shape[-1]
        if T_in == T_out:
            return Z_coarse
    
        if polar_mode:
            amplitude = torch.sqrt(Z_coarse[:, :, 0]) if energy_mode else Z_coarse[:, :, 0]
            phase = Z_coarse[:, :, 1] if phase_mode else torch.zeros_like(amplitude)
            Zc_coarse = torch.polar(amplitude, phase)
        elif complex_mode:
            Zc_coarse = Z_coarse
        else:
            Zc_coarse = torch.complex(Z_coarse[:, :, 0], Z_coarse[:, :, 1])
    
        if demod:
            C_ds = self._row_mod(Zc_coarse, sign=-1)
        else:
            C_ds = Zc_coarse
    
        Zf_k_recon_list = []
    
        for k, qt in enumerate(self.qtile_transforms):
    
            C_k_ds = C_ds[:, :, k, :]
    
            Zf_k_bb_cropped = torch.fft.fft(C_k_ds, dim=-1, norm='ortho')
    
            Zf_k_bb = self._centered_pad_or_crop(Zf_k_bb_cropped, T_in)
    
            # ✅ Exact integer remodulation
            Zf_k_recon = torch.roll(Zf_k_bb, shifts=qt.shift, dims=-1)
            Zf_k_recon_list.append(Zf_k_recon)
    
        Zf_recon = torch.stack(Zf_k_recon_list, dim=2)
    
        Zc_out = torch.fft.ifft(Zf_recon, dim=-1, norm='ortho')
    
        if preserve_amplitude:
            Zc_out *= math.sqrt(T_in / T_out)
    
        if polar_mode:
            energy = Zc_out.abs()**2 if energy_mode else Zc_out.abs()
            if phase_mode:
                return torch.stack([energy, Zc_out.angle()], dim=2)
            return energy.unsqueeze(2)
    
        elif complex_mode:
            return Zc_out
    
        else:
            return torch.stack([Zc_out.real, Zc_out.imag], dim=2)

    def check_aliasing_and_report(self, T_out: int):
        """
        Checks for potential information loss (aliasing) and reports detailed diagnostics,
        taking into account the VQT `max_window_size` if it is set.
        """
        print(f"\n--- Downsampling check (T_out={T_out}) ---")
        qprime = self.q / (11**0.5)
        duration = self.duration
        cqt_sizes = 2 * torch.floor(self.freqs.cpu() / qprime * duration) + 1
        
        max_size = getattr(self, 'max_window_size', None)
        
        if max_size is not None:
            final_sizes = torch.minimum(cqt_sizes, torch.tensor(float(max_size)))
        else:
            final_sizes = cqt_sizes
            
        n_bins_required = final_sizes.numpy()
        aliasing_mask = T_out < n_bins_required
        n_bad = int(aliasing_mask.sum())
    
        min_T_out_for_lossless = int(np.max(n_bins_required))
        
        supported_mask = ~aliasing_mask
        max_f_supported = float(self.freqs[supported_mask].max().item()) if supported_mask.any() else 0.0
        
        # Temporal resampling and frequency-bank invertibility are separate
        # requirements.  A large enough T_out cannot repair an FFT bin where
        # every analysis window is zero.
        requested_coverage = self._check_coverage()
        bank_coverage = self._check_coverage([0.0, float(self.sample_rate) / 2.0])
        bank_has_holes = not bank_coverage["covered"]
        if bank_has_holes:
            missing = bank_coverage.get("missing_fft_bins", [])
            if requested_coverage["covered"]:
                print("[NOTE] Requested-band reconstruction is possible; full-band reconstruction is not.")
            else:
                print("\033[91m[Warning]\033[0m Spectral coverage: one or more FFT bins are uncovered.")
            if missing:
                print(f"       Missing bins: {len(missing)} (first: {missing[0]}).")
            print("       Increase max_window_size and/or num_freq; T_out cannot repair this.")

        if n_bad > 0:
            affected_freqs = self.freqs.cpu().numpy()[aliasing_mask]
            print(f"\033[91m[Warning]\033[0m Temporal resampling: T_out={T_out}; need at least {min_T_out_for_lossless}.")
            print(f"       {n_bad}/{len(self.freqs)} rows are truncated (from {affected_freqs.min():.2f} Hz).")
        elif not bank_has_holes:
            print(f"[OK]  Temporal resampling: T_out={T_out} "
                  f"(minimum: {min_T_out_for_lossless}).")
        
        print("---------------------------\n")
        return torch.from_numpy(aliasing_mask)
#-------------------------------------------------------------------------------------------------------------
    
    def forward(
        self,
        X: torch.Tensor,
        
        #spectroram parameters
        normalize: bool = False,
        polar_mode: bool = True,
        energy_mode: bool = True,
        phase_mode: bool = True,
        complex_mode: bool = False,
        
        #interpolation parameters
        interp_mode : str = None, #other modes is 'spline'
        num_time : int = None,
        spectrogram_shape: Optional[Tuple[int, int]] = None,
        am_mode: bool = True,
        

    ):
        """
        Compute the Q-tiles and interpolate

        Args:
            X:
                Time series of data. Should have the duration and sample rate
                used to initialize this object. Expected input shape is
                `(B, C, T)`, where T is the number of samples, C is the number
                of channels, and B is the number of batches. If less than
                three-dimensional, axes will be added during Q-tile
                computation.
                
            normalize:
                normalize input with repsect to max for nicer visualization if data is not already normalized or whitened. 
                Note the THIS BREAKS INVERTIBILITY, for VISUALIZATION ONLY!
                
            spectrogram_shape:
                The shape of the interpolated spectrogram, specified as
                `(num_f_bins, num_t_bins)`. Because the
                frequency spacing of the Q-tiles is in log-space, the frequency
                interpolation is log-spaced as well. If not given, the shape
                used to initialize the transform will be used.

        Returns:
            The interpolated Q-transform for the batch of data. Output will
            have one more dimension than the input
        """
        if normalize:
            X_norm=X.clone()
            X_norm/=torch.max(X_norm,dim=-1).values.unsqueeze(-1)
            X=X_norm
            
        # Path 1: Efficiently compute downsampled transform directly
        if num_time is not None:
            
            # Check for aliasing before computation
            T_in = (int(self.sample_rate * self.duration) // 2) + 1
            if num_time < T_in:
                 self.check_aliasing_and_report(num_time)

            self.compute_qtiles(
                X, polar_mode, energy_mode, phase_mode, complex_mode,
                num_time=num_time, am_mode=am_mode
            )

        # Path 2: Compute full-res tiles for spline interpolation or raw output
        else:
            self.compute_qtiles(
                X, polar_mode, energy_mode, phase_mode, complex_mode,
                num_time=None, am_mode=am_mode # Ensures full resolution
            )
        
        # --- Stack the computed tiles ---
        if not polar_mode and complex_mode:
            stacking_dim = 2 # B, C, F, T
        else:
            stacking_dim = 3 # B, C, P, F, T
        qtiles_stacked = torch.stack(self.qtiles, dim=stacking_dim)

        # --- Handle post-processing (spline interpolation) ---
        if interp_mode and interp_mode.lower() == 'spline':
            if num_time is not None:
                 print("[Info] `interp_mode='spline'` is active, but a downsampled transform was already computed. "
                       "Spline interpolation will not be performed.")
                 return qtiles_stacked

            if spectrogram_shape is None:
                num_f_bins = qtiles_stacked.shape[-2]
                num_t_bins = qtiles_stacked.shape[-1]
                warnings.warn(f"`spectrogram_shape` not provided for spline mode. Returning full-resolution tiles.")
            else:
                num_f_bins, num_t_bins = spectrogram_shape
                return self.interpolate(
                    Z=qtiles_stacked, 
                    num_f_bins=num_f_bins, 
                    num_t_bins=num_t_bins,
                    polar_mode=polar_mode,
                    phase_mode=phase_mode,
                    complex_mode=complex_mode
                )
        
        return qtiles_stacked

    def invert_qtransform(self, qtransform, idx: Optional[int] = None, polar_mode: bool = True, energy_mode: bool =True, phase_mode: bool =True, complex_mode: bool = False, am_mode: bool = True):
        """
        Invert the Q-transform to recover the time-domain signal.

        Args:
            qtransform (torch.Tensor): 
                The spectrogram to invert. Expected shape is 
                [Batch, Channel, 2 (E/P), Frequency, Time].
            idx (Optional[int]): 
                - If an integer is provided, inverts using ONLY the specified
                  frequency tile `idx`. This is useful for analyzing a single
                  band but does not reconstruct the full signal.
                - If `None` (default), performs a full signal reconstruction by 
                  combining information from ALL frequency tiles.
        """
        # Check and format input dimensions
        
        if len(qtransform.shape) < 2:
            raise ValueError('Input must have at least 2 dimensions F,T. If not provided B,C will be added for consistency and phase will be assumed to be 0')
            
        if (not polar_mode) and complex_mode:
            
            if len(qtransform.shape) > 4:
                raise ValueError('with polar_mode= False and complex_mode=true, Input must have at most 4 dimensions B,C,F,T')
                
            while len(qtransform.shape) < 4:
                qtransform = qtransform.unsqueeze(0)
       
        else:
        
            if len(qtransform.shape) > 5:
                raise ValueError('Input must have at most 5 dimensions B,C,P,F,T')
                
            while len(qtransform.shape) < 5:
                qtransform = qtransform.unsqueeze(0)
                
            if qtransform.shape[2] > 2:
                 raise ValueError(f'Phase dimension [2] expected to be 1 or 2, found {qtransform.shape[2]} instead')


        # ---  Resample to original time dimension if necessary ---
        n_samples_original = int(self.sample_rate * self.duration)//2+1
        n_times_input = qtransform.shape[-1]
        
        if n_times_input < n_samples_original:
            print(f"Input spectrogram is downsampled (T={n_times_input}). "
                  f"Upsampling to original T={n_samples_original} before inversion...")
            
            qtransform = self.upsample(
                qtransform,
                n_samples_original,
                polar_mode=polar_mode,
                energy_mode=energy_mode,
                phase_mode=phase_mode,
                complex_mode=complex_mode,
                preserve_amplitude=True,
                demod = not am_mode
            )
        elif n_times_input > n_samples_original: 
            print(f"Input spectrogram is upsampled (T={n_times_input}). "
                  f"Downsampling to original T={n_samples_original} before inversion...")
            
            qtransform = self.downsample(
                qtransform,
                n_samples_original,
                polar_mode=polar_mode,
                energy_mode=energy_mode,
                phase_mode=phase_mode,
                complex_mode=complex_mode,
                preserve_amplitude=True,
                remod = not am_mode
            )

        # --- CASE 1: Invert using a single specified tile ---
        if idx is not None:
            print(f"Inverting using single tile index: {idx}")
            if (not polar_mode) and complex_mode:
                tile = qtransform[:, :, idx, :]
            else:
                tile = qtransform[:, :, :, idx, :]
            # Invert single tile (maximum redundancy of the Qtransform)
            fseries_scaled = self.qtile_transforms[idx].invert(tile, polar_mode, energy_mode, phase_mode, complex_mode)
            x_rec = torch.fft.irfft(fseries_scaled, norm='ortho', n=int(self.sample_rate * self.duration))
            return x_rec

        # --- CASE 2: Invert using ALL tiles for full reconstruction (VECTORIZED) ---
        else:
            
            # --- Reverse the forward process for ALL tiles at once ---
            
            # 1. Convert entire spectrogram from Energy/Phase to complex tdenerg

            if polar_mode:
                amplitude = torch.sqrt(qtransform[:, :, 0]) if energy_mode else qtransform[:, :, 0]
                        
                amplitude /= (self.sample_rate)**0.5
                
                if phase_mode:
                    phase = qtransform[:, :, 1]
                else:
                    phase = torch.zeros_like(amplitude)
                    
                tdenergy = amplitude * torch.exp(1j * phase)   
                
            elif complex_mode:
                tdenergy= qtransform/(self.sample_rate)**0.5
            else:
                tdenergy= torch.complex(qtransform[:,:,0,:,:],qtransform[:,:,1,:,:])/ (self.sample_rate)**0.5
            
            # 2. FFT all tdenergy tiles to get their windowed spectra (wenergy)
            n_rfft = (int(self.sample_rate * self.duration) // 2) + 1
            wenergy = torch.fft.fft(tdenergy, norm= 'ortho')
            #print(f'{wenergy.shape=}')
            #print(f'{n_rfft=}')

            # 3. Stack all windows to create a [F, T_freq] tensor
            all_windows = torch.stack([qt.full_window.squeeze() for qt in self.qtile_transforms])
            
            # --- Implement the Wiener Filter formula ---
    
            # 4. Numerator: Sum(wenergy * window)
            numerator = torch.sum(wenergy * all_windows[None, None, :, :], dim=2)
            
            # 5. Denominator: Sum(window^2)
            denominator = torch.sum(all_windows**2, dim=0)

            # 6. Reconstruct the spectrum
            mask = denominator > 0
            #print(f'{mask.shape=}')
            #print(f'{torch.where(mask==False)=}')
            fseries_reconstructed = torch.zeros_like(numerator)
            fseries_reconstructed[:, :, mask] = numerator[:, :, mask] / denominator[mask].view(1, 1, -1)  
            
            # 7. Recover the final time-domain signal
            x_rec = torch.fft.irfft(fseries_reconstructed, norm='ortho', n=int(self.sample_rate*self.duration))
            return x_rec
