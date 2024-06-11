import math
from typing import List

import torch
import torch.nn.functional as F

from torch_spline_interpolation import *

class QTile(torch.nn.Module):
    def __init__(
        self,
        q: float,
        frequency: float,
        duration: float,
        sample_rate: float,
        mismatch: float,
    ):
        super().__init__()
        self.mismatch = mismatch
        self.q = float(q)
        self.deltam = 2 * (self.mismatch / 3.0) ** (1 / 2.0)
        self.qprime = self.q / 11 ** (1 / 2.0)
        self.frequency = frequency
        self.duration = duration
        self.sample_rate = sample_rate

        self.windowsize = (
            2 * int(self.frequency / self.qprime * self.duration) + 1
        )
        pad = self.ntiles() - self.windowsize
        padding = torch.Tensor((int((pad - 1) / 2.0), int((pad + 1) / 2.0)))
        self.register_buffer("padding", padding)
        self.register_buffer("indices", self.get_data_indices())
        self.register_buffer("window", self.get_window())

    def ntiles(self):
        tcum_mismatch = self.duration * 2 * torch.pi * self.frequency / self.q
        return int(2 ** torch.ceil(torch.log2(tcum_mismatch / self.deltam)))

    def _get_indices(self):
        half = int((self.windowsize - 1) / 2)
        return torch.arange(-half, half + 1)

    def get_window(self):
        wfrequencies = self._get_indices() / self.duration
        xfrequencies = wfrequencies * self.qprime / self.frequency
        norm = (
            self.ntiles()
            / (self.duration * self.sample_rate)
            * (315 * self.qprime / (128 * self.frequency)) ** (1 / 2.0)
        )
        return torch.Tensor((1 - xfrequencies**2) ** 2 * norm)

    def get_data_indices(self):
        return torch.round(
            self._get_indices() + 1 + self.frequency * self.duration,
        ).type(torch.long)

    def forward(self, fseries: torch.Tensor, norm: str = "median"):
        while len(fseries.shape) < 3:
            fseries = fseries[None]
        windowed = fseries[..., self.indices] * self.window
        left, right = self.padding
        padded = F.pad(windowed, (int(left), int(right)), mode="constant")
        wenergy = torch.fft.ifftshift(padded, dim=-1)

        tdenergy = torch.fft.ifft(wenergy)
        energy = tdenergy.real**2.0 + tdenergy.imag**2.0
        if norm:
            norm = norm.lower() if isinstance(norm, str) else norm
            if norm == "median":
                medians = torch.quantile(energy, q=0.5, dim=-1)
                medians = medians.repeat(energy.shape[-1], 1, 1)
                medians = medians.transpose(0, -1).transpose(0, 1)
                energy /= medians
            elif norm == "mean":
                means = torch.mean(energy, dim=-1)
                means = means.repeat(energy.shape[-1], 1, 1)
                means = means.transpose(0, -1).transpose(0, 1)
                energy /= means
            else:
                raise ValueError("Invalid normalisation %r" % norm)
            return energy.type(torch.float32)
        return energy


class SingleQTransformLinear(torch.nn.Module):
    def __init__(
        self,
        duration: float,
        sample_rate: float,
        q: float = 12,
        frange: List[float] = [0, torch.inf],
        mismatch: float = 0.2,
    ):
        super().__init__()
        self.q = q
        self.frange = frange
        self.duration = duration
        self.mismatch = mismatch

        qprime = self.q / 11 ** (1 / 2.0)
        if self.frange[0] == 0:
            self.frange[0] = 50 * self.q / (2 * torch.pi * duration)
        if math.isinf(self.frange[1]):
            self.frange[1] = sample_rate / 2 / (1 + 1 / qprime)
        self.freqs = self.get_freqs()
        self.qtile_transforms = torch.nn.ModuleList(
            [
                QTile(self.q, freq, self.duration, sample_rate, self.mismatch)
                for freq in self.freqs
            ]
        )
        self.qtiles = None

    def get_freqs(self):
        minf, maxf = self.frange
        fcum_mismatch = maxf - minf
        deltam = 2 * (self.mismatch / 3.0) ** (1 / 2.0)
        nfreq = int(max(1, math.ceil(fcum_mismatch / deltam)))
        fstep = fcum_mismatch / nfreq
        return torch.linspace(minf, maxf, nfreq)

    def get_max_energy(
        self, fsearch_range: List[float] = None, dimension: str = "both"
    ):
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
            return torch.max(max_across_ft, dim=0).values
        if dimension == "batch":
            return torch.max(max_across_ft, dim=-1).values

    def compute_qtiles(self, X: torch.Tensor, norm: str = "median"):
        X = torch.fft.rfft(X, norm="forward")
        X[..., 1:] *= 2
        self.qtiles = [qtile(X, norm) for qtile in self.qtile_transforms]

    def interpolate_original(self, num_f_bins: int, num_t_bins: int):
        if self.qtiles is None:
            raise RuntimeError(
                "Q-tiles must first be computed with .compute_qtiles()"
            )
        resampled = [
            F.interpolate(qtile, num_t_bins, mode="bicubic")
            for qtile in self.qtiles
        ]
        resampled = torch.stack(resampled, dim=-2)
        resampled = F.interpolate(
            resampled, (num_f_bins, num_t_bins), mode="bicubic"
        )

        
        return torch.squeeze(resampled)
    
    def interpolate(self, num_f_bins, num_t_bins, device):
        if self.qtiles is None:
            raise RuntimeError(
                "Q-tiles must first be computed with .compute_qtiles()"
            )

         
        #Uncomment to return only qtiles. Needed to test 1D interpolation
        print('Returning Qtiles!!')
        return self.qtiles
        
        

        # Extract time values
        t = torch.linspace(0, 1, num_t_bins).to(device)
        
        print("Number of time bins:", num_t_bins)
        
        #interpolate tiles to
        x_bins=self.qtiles[-1].squeeze(0).squeeze(0).shape[0]
        print(f'{x_bins=}')
        # Interpolate along the time dimension using natural cubic spline
        resampled=[]

        #ToDO: Tensorize the following for loop. Pad each qtile to the size of the largest qtile and convert list to tensor using torch.stack. Then implement batch dimension in torch_spline_interpolation.spline_interpolate to pass the whole tensor at once.
        
        for qtile in self.qtiles:
            NCS=spline_interpolate(qtile.squeeze(0).squeeze(0),num_t_bins,s=0.001)
            resampled.append(NCS)
            
        resampled = torch.stack(resampled, dim=-2)
        resampled = resampled.squeeze(-1).squeeze()
        
        resampled=spline_interpolate_2d(resampled.T,num_t_bins,num_f_bins,False,3,3,0.001,0.001)      

        return resampled.to(device)





    
    def forward(
        self,
        X: torch.Tensor,
        num_f_bins: int,
        num_t_bins: int,
        norm: str = "median",
    ):
        self.compute_qtiles(X, norm)
        return self.interpolate(num_f_bins, num_t_bins, X.device)


class QScan(torch.nn.Module):
    def __init__(
        self,
        duration: float,
        sample_rate: float,
        qrange: List[float] = [4, 64],
        frange: List[float] = [0, torch.inf],
        mismatch: float = 0.2,
    ):
        super().__init__()
        self.qrange = qrange
        self.mismatch = mismatch
        self.qs = self.get_qs()
        self.frange = frange

        self.q_transforms = torch.nn.ModuleList(
            [
                SingleQTransformLinear(
                    duration=duration,
                    sample_rate=sample_rate,
                    q=q,
                    frange=self.frange.copy(),
                    mismatch=self.mismatch,
                )
                for q in self.qs
            ]
        )

    def get_qs(self):
        deltam = 2 * (self.mismatch / 3.0) ** (1 / 2.0)
        cumum = math.log(self.qrange[1] / self.qrange[0]) / 2 ** (1 / 2.0)
        nplanes = int(max(math.ceil(cumum / deltam), 1))
        dq = cumum / nplanes
        qs = [
            self.qrange[0] * math.exp(2 ** (1 / 2.0) * dq * (i + 0.5))
            for i in range(nplanes)
        ]
        return qs

    def forward(
        self,
        X: torch.Tensor,
        num_f_bins: int,
        num_t_bins: int,
        fsearch_range: List[float] = None,
        norm: str = "median",
    ):
        for transform in self.q_transforms:
            transform.compute_qtiles(X, norm)
        idx = torch.argmax(
            torch.Tensor(
                [
                    transform.get_max_energy(fsearch_range=fsearch_range)
                    for transform in self.q_transforms
                ]
            )
        )
        return (
            self.q_transforms[idx].interpolate(num_f_bins, num_t_bins),
            self.qs[idx],
        )
