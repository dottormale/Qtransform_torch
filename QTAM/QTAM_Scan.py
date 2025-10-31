import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from torch_spline_interpolation_1_0_0 import *
import numpy as np
import gc
from collections import OrderedDict



# ============================
# VECTORIZED BATCHED WINDOWS
# ============================

import torch
import torch.nn.functional as F
import math

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


def planck_taper_window_range_batch(N: int, epsilon: torch.Tensor, x_min: float = -1, x_max: float = 1, device: str = 'cpu') -> torch.Tensor:
    """
    Constructs Planck-taper windows for a batch of epsilon values over an 
    arbitrary range [x_min, x_max], using the specific algebraic formula 
    from the user's original scalar function for Z.
    """
    # 1. Coordinate Generation and Mapping (N samples)
    x = torch.linspace(x_min, x_max, steps=N, device=device, dtype=torch.float32)
    x_canonical = 2 * (x - x_min) / (x_max - x_min) - 1
    y = (x_canonical + 1) / 2
    
    # E is the number of epsilon values (batch size)
    E = epsilon.numel()
    
    # 2. Reshaping for Broadcasting
    epsilon_rs = epsilon.to(device).float().view(E, 1)
    y_rs = y.view(1, N)
    y_expanded = y_rs.expand(E, N) # Expanded y coordinate for correct indexing
    
    # 3. Initialization of the Window Tensor
    w = torch.ones(E, N, device=device, dtype=torch.float32)
    
    # 4. Rising Edge: 0 < y < epsilon
    mask_rise = (y_rs > 0) & (y_rs < epsilon_rs)
    if mask_rise.any():
        y_masked = y_expanded[mask_rise]
        epsilon_masked = epsilon_rs.expand(E, N)[mask_rise]
        
        # *** CORRECTION APPLIED HERE (Matching original scalar function) ***
        Z_plus = 2 * epsilon_masked * (
            1 / (1 + 2 * y_masked - 1) + 
            1 / (1 - 2 * epsilon_masked + 2 * y_masked - 1)
        )
        
        w[mask_rise] = 1.0 / (torch.exp(Z_plus) + 1.0)
        
    # 5. Flat Region: epsilon <= y <= 1 - epsilon (Handled by initialization)
    
    # 6. Falling Edge: 1 - epsilon < y < 1
    mask_fall = (y_rs > (1 - epsilon_rs)) & (y_rs < 1)
    if mask_fall.any():
        y_masked = y_expanded[mask_fall]
        epsilon_masked = epsilon_rs.expand(E, N)[mask_fall]
        
        # *** CORRECTION APPLIED HERE (Matching original scalar function) ***
        Z_minus = 2 * epsilon_masked * (
            1 / (1 - 2 * y_masked + 1) + 
            1 / (1 - 2 * epsilon_masked - 2 * y_masked + 1)
        )

        w[mask_fall] = 1.0 / (torch.exp(Z_minus) + 1.0)

    # 7. Endpoints set to 0
    w[:, 0] = 0.0
    w[:, -1] = 0.0
    
    return w

def kaiser_window_range_batch(L: int, beta: torch.Tensor, x_min: float = -1, x_max: float = 1, device: str = 'cpu') -> torch.Tensor:
    """
    Constructs Kaiser windows for a batch of beta values.
    The window values are independent of the coordinate range [x_min, x_max].
    
    Args:
        L (int): Window length.
        beta (torch.Tensor): 1D tensor of Kaiser beta parameters.
        x_min (float): Minimum coordinate value (ignored for window calculation).
        x_max (float): Maximum coordinate value (ignored for window calculation).
        device (str): Device.
        
    Returns:
        Tensor: A 2D tensor of shape [B, L] representing the Kaiser windows,
                where B is the number of beta values.
    """
    # B is the number of beta values (batch size)
    B = beta.numel()
    
    # 1. Coordinate (Index) Setup
    # The Kaiser window formula uses an index m from 0 to L-1.
    # We define m_norm in the range [-1, 1] for the formula.
    m = torch.arange(L, dtype=torch.float32, device=device)
    m_norm = (2 * m / (L - 1)) - 1
    
    # Reshape m_norm to [1, L] for broadcasting
    m_norm_rs = m_norm.view(1, L)
    
    # 2. Reshape beta to [B, 1] for broadcasting
    beta_rs = beta.to(device).float().view(B, 1)

    # 3. Calculate the argument for the Bessel function: sqrt(1 - m_norm^2)
    arg_sqrt = torch.sqrt(1 - m_norm_rs.pow(2)) # Shape [1, L]
    
    # 4. Calculate the numerator: I_0(beta * sqrt(1 - m_norm^2))
    numerator_arg = beta_rs * arg_sqrt # Shape [B, L]
    numerator = torch.i0(numerator_arg) # Shape [B, L]
    
    # 5. Calculate the denominator: I_0(beta)
    denominator = torch.i0(beta_rs) # Shape [B, 1]
    
    # 6. Calculate the window W = N / D
    window = numerator / denominator # Shape [B, L]
    
    return window

import torch

def tukey_window_batch(window_length: int, alpha: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
    """
    Generates Tukey windows for a batch of alpha values, matching the 
    integer-ramp-based logic of the user's original scalar function.
    """
    L = window_length
    A = alpha.numel()
    
    # 1. Reshape alpha to [A, 1]
    alpha_rs = alpha.to(device).float().view(A, 1)

    # 2. Initialize the result window W as [A, L] tensor of ones
    W = torch.ones(A, L, device=device, dtype=torch.float32)
    
    # 3. Calculate the integer ramp length R for each alpha
    # R = floor(alpha * L / 2). This must be computed for each alpha.
    R_float = (alpha_rs * L / 2.0).floor().long() # Shape [A, 1]
    
    for i in range(A):
        alpha_val = alpha[i].item()
        R = R_float[i].item()
        
        # NOTE: We skip the alpha=0 check, as requested, but we handle the 
        # R=0 case which is equivalent for the logic flow.
        if R == 0 or alpha_val == 0.0:
            # If R=0, the window remains all ones (W[i] = 1.0)
            continue
            
        # 4. Create the linear space 'w' from 0 to 1 over the integer ramp length R
        w = torch.linspace(0, 1, R, device=device, dtype=torch.float32)
        
        # 5. Calculate the cosine taper matching the user's original formula
        # cosine = 0.5 * (1 + torch.cos(torch.pi * (w - 1)))
        cosine = 0.5 * (1 + torch.cos(torch.pi * (w - 1)))
        
        # 6. Apply to the rising and falling edges of the i-th window
        W[i, :R] = cosine
        W[i, L - R:] = cosine.flip(0)
        
    return W

def bisquare_window_batched(L: int, batch_size: int = 1, device: str = "cpu", epsilon: float = None) -> torch.Tensor:
    """
    Batched Bisquare windows.
    """
    x = torch.linspace(-1, 1, L, device=device)
    w = (1 - x ** 2)** 2
 
    
    if epsilon:
        w[0]+=epsilon
        w[-1]+=epsilon
    return w.unsqueeze(0).expand(batch_size, -1)

def hann_window_batched(L: int, batch_size: int = 1, device: str = "cpu", epsilon: float = None) -> torch.Tensor:
    """
    Batched Hann windows.
    """
    n = torch.arange(L, device=device, dtype=torch.float32)
    w = 0.5 * (1 - torch.cos(2 * math.pi * n / (L - 1)))

    if epsilon:
        w[0]+=epsilon
        w[-1]+=epsilon
    return w.unsqueeze(0).expand(batch_size, -1)



# ============================
# QTileMulti Class
# ============================

class QTileMulti(torch.nn.Module):
    """
    Multi-configuration QTile: supports multiple window types and
    parameter scans (taus, betas) in a fully vectorized way.
    """

    def __init__(
        self,
        q: float,
        frequency: float,
        duration: float,
        sample_rate: float,
        mismatch: float,
        window_types: Optional[List[str]] = None,
        taus: Optional[Union[List[float], torch.Tensor]] = None,
        betas: Optional[Union[List[float], torch.Tensor]] = None,
        logf: bool = False,
        eps: float = 1e-5,
        device: str = "cpu",
        max_window_size = None # maximum width of window function

    ):
        super().__init__()
        self.q = q
        self.frequency = frequency
        self.duration = duration
        self.sample_rate = sample_rate
        self.mismatch = mismatch
        self.logf = logf
        self.max_window_size=max_window_size
        self.eps = eps
        self.device = device

        self.window_types = window_types if window_types is not None else [None]
        self.taus = torch.as_tensor(taus, device=device, dtype=torch.float32) if taus is not None else None
        self.betas = torch.as_tensor(betas, device=device, dtype=torch.float32) if betas is not None else None

        # precompute constants
        self.qprime = self.q / (11.0 ** 0.5)
        self.deltam = torch.tensor(2 * (self.mismatch / 3.0) ** 0.5, device=device)
        
        self.windowsize = 2 * int(self.frequency / self.qprime * self.duration) + 1
        #print('-----------------------------')
        if self.max_window_size:
            #print(f'{self.windowsize=} ; {self.max_window_size=}')
            self.windowsize = min(self.windowsize, self.max_window_size)
            
        self.pad_len = (self.duration*self.sample_rate)//2 +1 - self.windowsize
        self.pad_left = int((self.pad_len) // 2)
        self.pad_right = int((self.pad_len + 1) // 2)

        config, window =  self.get_window()
        
        self.configs = config
        self.register_buffer("window", window)
        self.register_buffer("full_window", self.get_full_window())

    def get_window(self):
        # Build all windows
        windows, configs = [], []
        for wtype in self.window_types:
            
            if wtype is None or wtype.lower() == "bisquare":
                win = bisquare_window_batched(self.windowsize, device=self.device, epsilon=self.eps)
                cfgs = [(self.q,"bisquare", None, None)]
            elif wtype.lower() == "hann":
                win = hann_window_batched(self.windowsize, device=self.device, epsilon=self.eps)
                cfgs = [(self.q,"hann", None, None)]
            elif wtype.lower() == "tukey":
                assert self.taus is not None, "Tukey requires taus"
                #print(f'{self.taus=}')
                win = tukey_window_batch(self.windowsize, self.taus.to(self.device), device=self.device)
                cfgs = [(self.q,"tukey", tau.item(), None) for tau in self.taus]
            elif wtype.lower() == "planck-taper":
                assert self.taus is not None, "Planck-taper requires taus"
                #print(f'{self.taus=}')
                win = planck_taper_window_range_batch(self.windowsize, self.taus.to(self.device), -1, 1, device=self.device)
                cfgs = [(self.q,"planck-taper", tau.item(), None) for tau in self.taus]
            elif wtype.lower() == "kaiser":
                assert self.betas is not None, "Kaiser requires betas"
                #print(f'{self.betas=}')
                win = kaiser_window_range_batch(self.windowsize, self.betas.to(self.device), device=self.device)
                cfgs = [(self.q,"kaiser", None, beta.item()) for beta in self.betas]
            else:
                raise ValueError(f"Unsupported window type {wtype}")
            
            windows.append(win)
            configs.extend(cfgs)

        final_windows = torch.cat(windows, dim=0)  # [n_configs, win_len]
        return configs, final_windows

    def compute_window_energy(self, window):
        #Normalize by imposing Parseval condition: sum |w[t]|^2 dt = (1/N) * sum |W[f]|^2 = 1
        return window.square().sum(dim=[0, 1, 3])/self.duration 

    def get_full_window(self):
        
        # Pad window to full length with small epsilon and shift to center frequency
        full = F.pad(self.window.unsqueeze(0).unsqueeze(0), (self.pad_left, self.pad_right), value=self.eps)

        #Center window on central frequency
        self.shift = int(self.frequency * self.duration)
        #full_w= torch.roll(full, shifts=self.shift- (self.windowsize // 2 +self.pad_right))
        full_w = torch.roll(full, shifts=self.shift - (self.pad_left + self.windowsize // 2))

        #Normalize window    
        wen=self.compute_window_energy(full_w)
        norm = (wen) ** -0.5 
        full_w *= norm[None,None,:,None]
        return full_w

    def forward(
        self, 
        fseries: torch.Tensor, 
        polar_mode: bool = True, 
        energy_mode: bool = True, 
        phase_mode: bool = True, 
        complex_mode: bool = False,
        num_time: Optional[int] = None, # Target number of time samples
        am_mode: bool = True             # True for baseband (amplitude), False for remodulated AM signal
    ):
        while len(fseries.shape) < 3:
            fseries = fseries[None]
        
        # Step 1: Bandpass filtering in frequency domain
        
        wenergy = fseries[:,:,None,:] * self.full_window.to(fseries.device)
        T_in = wenergy.shape[-1]

        
        # If num_time is not specified, compute full resolution tile (original behavior)
        if num_time is None:
            tdenergy = torch.fft.ifft(wenergy, norm='ortho')
            tdenergy *= (self.sample_rate)**0.5 
        
        # If num_time is specified, perform efficient downsampling
        else:
            T_out = num_time
            # Step 2: Demodulate by shifting to baseband in frequency domain
            wenergy_baseband = torch.roll(wenergy, shifts=-self.shift, dims=-1)
            
            # Step 3: Low-pass filter by cropping the spectrum
            wenergy_baseband_cropped = _centered_pad_or_crop(wenergy_baseband, T_out)
            
            # Step 4: Perform small IFFT to get downsampled baseband signal
            tdenergy_baseband = torch.fft.ifft(wenergy_baseband_cropped, norm='ortho')
            
            # Apply amplitude correction for energy conservation
            tdenergy_baseband *= math.sqrt(T_out / T_in)
            
            if am_mode:
                # We want the baseband signal (the envelope)
                tdenergy = tdenergy_baseband * (self.sample_rate)**0.5
            else:
                # Step 5 (Optional): Remodulate to get AM signal
                # Create time vector for the downsampled signal
                t = torch.linspace(0, self.duration, T_out, device=fseries.device, dtype=torch.float32)
                
                # Create rotation phasor
                phase = 2 * math.pi * self.frequency * t.view(1, 1, -1)
                rot = torch.polar(torch.ones_like(phase), phase)
                
                # Apply remodulation
                tdenergy = tdenergy_baseband * rot * (self.sample_rate)**0.5

        # The rest of the function remains the same, converting the complex `tdenergy`
        # to the desired output format (polar, complex, or real/imag)
        if polar_mode:
            if energy_mode:
                energy = tdenergy.real**2 + tdenergy.imag**2
            else:
                energy = torch.sqrt(tdenergy.real**2 + tdenergy.imag**2)
    
            phase = None
            if phase_mode:
                phase = torch.atan2(tdenergy.imag, tdenergy.real)
                return torch.stack([energy, phase],dim=2)
                
            return energy.unsqueeze(2)
        
        elif complex_mode:
            return tdenergy
        else:
            return torch.stack([tdenergy.real,tdenergy.imag], dim=2)
        

    def invert(self, tile, polar_mode: bool = True, energy_mode: bool = True, phase_mode: bool = True, complex_mode: bool = False):
        # Extract amplitude and phase from the tile
        
        if polar_mode:
            amplitude = torch.sqrt(tile[:, :, 0]) if energy_mode else tile[:, :, 0]
            
            if not phase_mode:
                print(f"\033[93m[Warning]\033[0m Qtile.invert: phase_mode is False, assuming all 0 phase.")

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

class SingleQMultiTransform(torch.nn.Module):
    """
    Compute the Q-transform for a single Q value and multiple window configurations.
    Uses QTileMulti to handle multiple parameter combinations (taus, betas, window types).

    For now, it computes all configurations without selecting the best one.
    """

    def __init__(
        self,
        duration: float,
        sample_rate: float,
        spectrogram_shape: Tuple[int, int],
        q: float = 12,
        eps: float = 1e-5,
        frange: List[float] = [0, torch.inf],
        mismatch: float = 0.2,
        num_freq: int = 0,
        logf: bool = False,
        max_window_size = False,

        window_types: Optional[List[str]] = None,
        taus: Optional[Union[List[float], torch.Tensor]] = None,
        betas: Optional[Union[List[float], torch.Tensor]] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.q = q
        self.spectrogram_shape = spectrogram_shape
        self.frange = frange
        self.duration = duration
        self.mismatch = mismatch
        self.logf = logf
        self.device = device
        self.eps = eps

        self.sample_rate = sample_rate
        self.num_freq = num_freq
        self.window_types = window_types
        self.taus = taus
        self.betas = betas

        qprime = self.q / 11 ** 0.5
        if self.frange[0] <= 0:
            self.frange[0] = 50 * self.q / (2 * torch.pi * duration)
        if math.isinf(self.frange[1]):
            self.frange[1] = sample_rate / 2 / (1 + 1 / qprime)

        # Frequency grid
        self.freqs = self.get_freqs()

        if max_window_size:
            #print(f'{max_window_size=}')
            self.max_window_size =self.get_max_window_size(max_window_size)
        else:
            self.max_window_size = None

        # Create QTileMulti for each frequency
        self.qtile_multi_transforms = torch.nn.ModuleList([
            QTileMulti(
                q=self.q,
                frequency=freq,
                duration=self.duration,
                sample_rate=self.sample_rate,
                mismatch=self.mismatch,
                window_types=self.window_types,
                taus=self.taus,
                betas=self.betas,
                logf=self.logf,
                device=self.device,
                max_window_size=self.max_window_size,
                eps=self.eps,
            )
            for freq in self.freqs
        ])

        self.qtiles = None

        self.configs=np.array(self.qtile_multi_transforms[0].configs, dtype=object)

    
    def get_freqs(self):
        """
        Calculate the frequencies that will be used in this transform.
        For each frequency, a `QTile` is created.
        """
        minf, maxf = self.frange

        # manually set log spaced frequencies given the desired number of tiles
        if self.num_freq:
            if self.logf:
                freqs = torch.tensor(np.geomspace(minf, maxf, self.num_freq))
            else:
                freqs=torch.linspace(minf,maxf,self.num_freq)

        # use gwpy mismatch method instead
        else:
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

    def get_max_energy(
        self,
        fsearch_range: Optional[List[float]] = None,
        criterion: str = "total",   # "pixel", "total", "median"
        per_element: bool = True,   # True -> per (B,C); False -> global
        energy_mode: bool = True,
        phase_mode: bool = True,
        polar_mode: bool = True,
        complex_mode: bool = False,
    ):
        """
        Compute maximum energy among configurations of this SingleQMultiTransform.
    
        Returns:
            best_val:  [B, C] or scalar
            best_idx:  [B, C] or scalar (index into self.configs)
            best_spec: [B, C, n_freq, T] or [n_freq, T]
        """

        if complex_mode:
            raise ValueError(f"Scan works only for real tensors (complex_mode = False=")
        if not polar_mode:
            raise ValueError(f"Scan works only for amplitude/energy (polar_mode = True)")
            
        
        
        if phase_mode:
            qtiles = self.qtiles[:,:,:,0,:,:]
        else:
            qtiles = self.qtiles  # [B, C, n_cfg, n_freq, T]
    
        # restrict frequency range if requested
        if fsearch_range is not None and self.freqs is not None:
            fmask = (self.freqs >= fsearch_range[0]) & (self.freqs <= fsearch_range[1])
            qtiles = qtiles[..., fmask, :]
        
        if energy_mode:
            energy=qtiles
        else:
            energy = qtiles.pow(2)  # [B, C, n_cfg, n_freq, T]
    
        # ---- compute energy metric per configuration ----
        if criterion == "pixel":
            energy_metric = energy.amax(dim=(-1, -2))  # [B, C, n_cfg]
        elif criterion == "total":
            energy_metric = energy.sum(dim=(-1, -2))   # [B, C, n_cfg]
        elif criterion == "median":
            energy_metric = energy.median(dim=-1).values.median(dim=-1).values
        else:
            raise ValueError(f"Unknown criterion '{criterion}'")
    
        # ---- choose best configuration ----
        if per_element:
            best_val, best_idx = torch.max(energy_metric, dim=-1)  # [B, C]
        else:
            global_mean = energy_metric.mean(dim=(0, 1))  # [n_cfg]
            best_idx = torch.argmax(global_mean)
            best_val = global_mean[best_idx]
    
        # ---- extract best spectrogram ----
        if per_element:
            B, C, n_cfg, F, T = qtiles.shape
            best_spec = torch.zeros((B, C, F, T), dtype=qtiles.dtype, device=qtiles.device)
            for b in range(B):
                for c in range(C):
                    best_spec[b, c] = qtiles[b, c, best_idx[b, c]]
        else:
            best_spec = qtiles[..., best_idx, :, :].mean(dim=(0, 1))
        
        best_idx_numpy = best_idx.detach().cpu().numpy()
        best_configs = self.configs[best_idx_numpy]

        return best_val, best_configs, best_spec

    
    def compute_qtiles(
        self, 
        X: torch.Tensor, 
        polar_mode: bool = True, 
        energy_mode: bool = True, 
        phase_mode: bool = True, 
        complex_mode: bool = False,
        num_time: Optional[int] = None, 
        am_mode: bool = True,         
                             ):
        """
        Compute the Q-tiles for all frequencies and all configurations.
        """
        X = torch.fft.rfft(X, norm="ortho")
        
        all_qtiles = [qmulti(
                X, polar_mode, energy_mode, phase_mode, complex_mode,
                num_time=num_time, am_mode=am_mode
            ) for qmulti in self.qtile_multi_transforms ]
            
        # Stack across frequencies
        self.qtiles = torch.stack(all_qtiles, dim=-2)  # [B, C,(2) ,n_cfg, n_freqs, T]
##########################################################################
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
        """Helper to apply phase rotation for remodulation (sign=+1) or demodulation (sign=-1)."""
        B, C, n_freqs, T = Zc.shape
        device = Zc.device
        dtype = Zc.real.dtype
        
        step = self.duration / T
        end_point = self.duration #- step
        t = torch.linspace(0, end_point, T, device=device, dtype=dtype)
        
        f = self.freqs.to(device, dtype).view(1, 1, n_freqs, 1)
        phase = 2 * math.pi * f * t.view(1, 1, 1, T)
        rot = torch.polar(torch.ones_like(phase), sign * phase)
        return Zc * rot

    def downsample(self, Z_in, T_out: int, polar_mode: bool, energy_mode: bool, phase_mode: bool, complex_mode: bool, preserve_amplitude=True, remod: bool = False):
        """
        Downsamples the spectrogram using the stable frequency-domain method.
        Handles various input formats by passing mode flags.
        """
        T_in = Z_in.shape[-1]
        if T_in == T_out:
            return Z_in

        # --- 1. Convert any input format to complex tdenergy ---
        if polar_mode:
            amplitude = torch.sqrt(Z_in[:, :, 0]) if energy_mode else Z_in[:, :, 0]
            phase = Z_in[:, :, 1] if phase_mode else torch.zeros_like(amplitude)
            if not phase_mode:
                print(f"\033[93m[Warning]\033[0m phase_mode is False, assuming all 0 phase.")
            Zc_in = torch.polar(amplitude, phase)
        elif complex_mode:
            Zc_in = Z_in
        else: # Real/Imag channels
            Zc_in = torch.complex(Z_in[:, :, 0], Z_in[:, :, 1])
            
        # --- 2. Core resampling logic (operates on complex values) ---
        Zf_in = torch.fft.fft(Zc_in, dim=-1, norm='ortho')
        C_k_ds_list = []
        for k, qt in enumerate(self.qtile_multi_transforms):
            center_freq_bin = qt.shift
            print(f'frequency:{self.freqs[k]}, idx:{k}, shift: {qt.shift}')
            Zf_k = Zf_in[:, :, k, :]
            Zf_k_bb = torch.roll(Zf_k, shifts=-center_freq_bin, dims=-1)
            Zf_k_bb_cropped = self._centered_pad_or_crop(Zf_k_bb, T_out)
            C_k_ds = torch.fft.ifft(Zf_k_bb_cropped, dim=-1, norm='ortho')
            C_k_ds_list.append(C_k_ds)
        C_ds = torch.stack(C_k_ds_list, dim=2)
        
        if preserve_amplitude:
            C_ds *= math.sqrt(T_out / T_in)
            
        if remod==True:
            Zc_out = self._row_mod(C_ds, sign=+1)
        else:
            Zc_out= C_ds 
        
        # --- 3. Convert complex output back to the original format ---
        if polar_mode:
            energy = Zc_out.abs()**2 if energy_mode else Zc_out.abs()
            if phase_mode:
                phase = Zc_out.angle()
                return torch.stack([energy, phase], dim=2)
            return energy.unsqueeze(2)
        elif complex_mode:
            return Zc_out
        else: # Real/Imag channels
            return torch.stack([Zc_out.real, Zc_out.imag], dim=2)

    def upsample(self, Z_coarse, T_in: int, polar_mode: bool, energy_mode: bool, phase_mode: bool, complex_mode: bool, preserve_amplitude=True, demod:bool = False):
        """
        Upsamples the spectrogram by reversing the frequency-domain process.
        """
        T_out = Z_coarse.shape[-1]
        if T_in == T_out:
            return Z_coarse

        if polar_mode:
            amplitude = torch.sqrt(Z_coarse[:, :, 0]) if energy_mode else Z_coarse[:, :, 0]
            phase = Z_coarse[:, :, 1] if phase_mode else torch.zeros_like(amplitude)
            if not phase_mode:
                print(f"\033[93m[Warning]\033[0m phase_mode is False, assuming all 0 phase.")
            Zc_coarse = torch.polar(amplitude, phase)
        elif complex_mode:
            Zc_coarse = Z_coarse
        else: # Real/Imag channels
            Zc_coarse = torch.complex(Z_coarse[:, :, 0], Z_coarse[:, :, 1])
            
        if demod == True:  
            C_ds = self._row_mod(Zc_coarse, sign=-1)
        else:
            C_ds = Zc_coarse
        
        Zf_k_recon_list = []
        for k, qt in enumerate(self.qtile_multi_transforms):
            C_k_ds = C_ds[:, :, k, :]
            Zf_k_bb_cropped = torch.fft.fft(C_k_ds, dim=-1, norm='ortho')
            Zf_k_bb = self._centered_pad_or_crop(Zf_k_bb_cropped, T_in)
            center_freq_bin = qt.shift
            Zf_k_recon = torch.roll(Zf_k_bb, shifts=center_freq_bin, dims=-1)
            Zf_k_recon_list.append(Zf_k_recon)
        
        Zf_recon = torch.stack(Zf_k_recon_list, dim=2)
        Zc_out = torch.fft.ifft(Zf_recon, dim=-1, norm='ortho')
        
        if preserve_amplitude:
            Zc_out *= math.sqrt(T_in / T_out)
    
        if polar_mode:
            energy = Zc_out.abs()**2 if energy_mode else Zc_out.abs()
            if phase_mode:
                phase = Zc_out.angle()
                return torch.stack([energy, phase], dim=2)
            return energy.unsqueeze(2)
        elif complex_mode:
            return Zc_out
        else: # Real/Imag channels
            return torch.stack([Zc_out.real, Zc_out.imag], dim=2)

    def check_aliasing_and_report(self, T_out: int):
        """
        Checks for potential information loss (aliasing) and reports detailed diagnostics,
        taking into account the VQT `max_window_size` if it is set.
        """
        print("\n--- Downsampling Analysis ---")
        qprime = self.q / (11**0.5)
        duration = self.duration
        cqt_sizes = 2 * torch.floor(self.freqs.cpu() / qprime * duration) + 1
        
        max_size = getattr(self, 'max_window_size', None)
        
        if max_size is not None:
            print(f"[Info] VQT mode detected with max_window_size = {max_size}")
            final_sizes = torch.minimum(cqt_sizes, torch.tensor(float(max_size)))
        else:
            print("[Info] Pure CQT mode detected (no window size cap).")
            final_sizes = cqt_sizes
            
        n_bins_required = final_sizes.numpy()
        aliasing_mask = T_out < n_bins_required
        n_bad = int(aliasing_mask.sum())
    
        min_T_out_for_lossless = int(np.max(n_bins_required))
        
        supported_mask = ~aliasing_mask
        max_f_supported = float(self.freqs[supported_mask].max().item()) if supported_mask.any() else 0.0
        
        if n_bad > 0:
            affected_freqs = self.freqs.cpu().numpy()[aliasing_mask]
            print(f"\033[93m[Warning]\033[0m {n_bad}/{len(self.freqs)} frequency rows will be truncated (information loss).")
            print(f"          Your chosen T_out = {T_out} is too small for the windows being used.")
            print(f"          Affected frequencies start from ~{affected_freqs.min():.2f} Hz upwards.")
            print("-" * 20)
            print(f"To be lossless for all frequencies, you MUST use at least T_out = {min_T_out_for_lossless}.")
            print(f"With T_out = {T_out}, the maximum fully supported frequency is ~{max_f_supported:.2f} Hz.")
        else:
            print("[Info] No information loss detected.")
            print(f"       Your chosen T_out = {T_out} is sufficient for lossless resampling.")
            print(f"       (Minimum required T_out for this spectrogram is {min_T_out_for_lossless}).")
        
        print("---------------------------\n")
        return torch.from_numpy(aliasing_mask)
##########################################################################
    
        
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
        interp_mode : str = None, #other modes is 'spline' but not implemented yet
        num_time : int = None,
        am_mode: bool = True,

    ):
        """
        Compute multi-configuration Q-transform.
        Returns results for all configurations (no selection).
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
                X, polar_mode=polar_mode, energy_mode=energy_mode, phase_mode=phase_mode, complex_mode=complex_mode,
                num_time=None, am_mode=am_mode # Ensures full resolution
            )
        '''
        # --- Stack the computed tiles ---
        if not polar_mode and complex_mode:
            stacking_dim = 2 # B, C, F, T
        else:
            stacking_dim = 3 # B, C, P, F, T
        qtiles_stacked = torch.stack(self.qtiles, dim=stacking_dim)
        '''

        return self.qtiles

    

##########################################################################
# Multi Q Qtransform Class
##########################################################################

class QScanMulti(nn.Module):
    """
    Multi-Q scanner based on SingleQMultiTransform.
    For each Q, multiple window configurations are computed.
    Then, the best configuration per Q is selected based on energy.
    Finally, the best Q (and config) overall is chosen.
    """

    def __init__(
        self,
        duration: float,
        sample_rate: float,
        spectrogram_shape: Tuple[int, int],
        qrange: List[float] = [4, 64],
        qlist: List[float] = None,
        frange: List[float] = [0, torch.inf],
        mismatch: float = 0.2,
        window_types: Optional[List[str]] = None,
        taus: Optional[List[float]] = None,
        betas: Optional[List[float]] = None,
        device: str = "cpu",
        max_window_size = False,
        logf: bool = True,
    ):
        super().__init__()
        self.qrange = qrange
        self.mismatch = mismatch
        self.qlist = qlist
        self.qs = self._get_qs()
        self.logf = logf
        self.frange = frange
        self.device = device
        self.max_window_size = max_window_size

        self.transforms = nn.ModuleList([
            SingleQMultiTransform(
                q=q,
                frange=self.frange.copy(),
                duration=duration,
                sample_rate=sample_rate,
                mismatch=self.mismatch,
                spectrogram_shape=spectrogram_shape,
                window_types=window_types,
                taus=taus,
                betas=betas,
                device=self.device,
                max_window_size = self.max_window_size,
                logf= self.logf,
            )
            for q in self.qs
        ])

    # -------------------------------------------------------------------------
    def _get_qs(self):
        if self.qlist is None:
            """Calculate Q values to scan."""
            deltam = 2 * (self.mismatch / 3.0) ** 0.5
            cumum = math.log(self.qrange[1] / self.qrange[0]) / 2 ** 0.5
            nplanes = int(max(math.ceil(cumum / deltam), 1))
            dq = cumum / nplanes
            return [
                self.qrange[0] * math.exp(2 ** 0.5 * dq * (i + 0.5))
                for i in range(nplanes)
            ]
        else:
            return self.qlist
    # -------------------------------------------------------------------------
    def forward(
        self,
        X: torch.Tensor, #input data
        fsearch_range: Optional[List[float]] = None, # restrict frequency range if requested
        criterion: str = "total", # 'total' : total energy of the spectrogram, 'pixel': highest pixel value in the spectrogram, 'median': median of spectrogram values
        per_element: bool = True, #compute max per each element in batch and channel dim separately (True) or average over such dimension (False)
        normalize: bool = False, # 
        scan_mode ='True', # choose best configuration across Q and parameter values (True) or return MultiQMultiTransform (False)
        energy_mode: bool = True,
        phase_mode: bool = False,
        polar_mode: bool = True,
        complex_mode: bool = False,
        
        #interpolation parameters
        interp_mode : str = None, #other modes is 'spline' but not implemented yet
        num_time : int = None,
        am_mode: bool = True,
    ):
        """
        Compute all Q transforms and select the best (Q, config)
        combination per (B,C) or globally.
    
        Returns:
             If scan_mode == True:
                If per_element == True:
                    best_spectrograms : list of lists [[spec_b0c0, spec_b0c1,...], [...]] where spec_b,c is a torch.Tensor [F_q, T_q]
                    best_qs            : torch.Tensor of shape [B, C] containing the Q value chosen for each (b,c)
                    best_configs       : numpy.ndarray (dtype=object) shape [B, C] with the winning config objects
                If per_element == False:
                    best_spec          : torch.Tensor [F_q, T_q] for the single global winner
                    best_q             : scalar torch.Tensor (or float) of the winning Q
                    best_config        : single config object (python object)
             else:
                results    : List of Torch tensors containing SingleQMultiTransforms, one for each Q value, i.e. len(results) = len(sef.qs)       
            
        """
        results = []
        if scan_mode:
            for transform in self.transforms:
                transform.compute_qtiles(
                X, polar_mode=polar_mode, energy_mode=energy_mode, phase_mode=phase_mode, complex_mode=complex_mode,
                num_time=num_time, am_mode=am_mode
            )
                best_val, best_configs, best_spec = transform.get_max_energy(
                    fsearch_range=fsearch_range,
                    criterion=criterion,
                    per_element=per_element,
                )
                # keep tuple per Q: (best_val, best_configs, best_spec, q_value)
                results.append((best_val, best_configs, best_spec, transform.q))
        
            # Build tensor of best values across Qs
            # results[i][0] is either [B,C] (per_element=True) or scalar (per_element=False)
            best_vals_list = [r[0] for r in results]
        
            if per_element:
                # stack => [n_Q, B, C]
                best_vals = torch.stack(best_vals_list, dim=0)  # dtype and device from elements
                # best_q_idx: [B, C] indices into results
                best_q_idx = best_vals.argmax(dim=0)            # torch.LongTensor [B, C]
        
                # Prepare outputs
                B = X.shape[0]
                C = X.shape[1]
        
                # best_qs_out: Q value per (B,C)
                q_tensor = torch.tensor(self.qs, device=best_q_idx.device, dtype=best_q_idx.dtype)
                best_qs_out = q_tensor[best_q_idx]  # [B, C]
        
                # Build nested lists for spectrograms and configs (because F_q may vary with q)
                best_spectrograms = [[None for _ in range(C)] for _ in range(B)]
                best_configs_out = np.empty((B, C), dtype=object)
        
                for b in range(B):
                    for c in range(C):
                        q_idx = int(best_q_idx[b, c].item())  # index into results
                        # results[q_idx][2] is best_spec for that Q
                        # when per_element=True it's a tensor [B, C, F_q, T_q]
                        spec_q = results[q_idx][2]             # tensor [B, C, F_q, T_q]
                        cfgs_q = results[q_idx][1]             # numpy array (B, C) dtype=object
        
                        best_spectrograms[b][c] = spec_q[b, c]           # tensor [F_q, T_q]
                        best_configs_out[b, c] = cfgs_q[b, c]            # python object
        
                return best_spectrograms, best_qs_out, best_configs_out
    
            else:
                # per_element == False: find single winning Q (global)
                # best_vals_list are scalars -> stack to [n_Q]
                best_vals = torch.stack([torch.tensor(v, device='cpu') if not torch.is_tensor(v) else v for v in best_vals_list], dim=0)
                # if some best_val were tensors on CPU/GPU, ensure we're consistent:
                best_vals = best_vals.to(self.device)
                best_q_idx = int(best_vals.argmax().item())  # scalar index
        
                # take the results corresponding to the winning Q
                best_val_q, best_configs_q, best_spec_q, best_q_value = results[best_q_idx]
        
                # best_spec_q is expected to be [F_q, T_q] (per_element=False case in SingleQMultiTransform)
                # best_configs_q is the single winning config (python object)
                # best_q_value is a float
                # return (spec, qvalue, config)
                return best_spec_q, torch.tensor(best_q_value), best_configs_q
        
        else:
            for transform in self.transforms:
                qt=transform(X, normalize=normalize, polar_mode=polar_mode, energy_mode=energy_mode, phase_mode=phase_mode, complex_mode=complex_mode,
                num_time=num_time, am_mode=am_mode)
                results.append(qt)
            return results

