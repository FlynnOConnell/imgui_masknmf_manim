"""
A movie split the way masknmf's compression rebuilds it (CompressionArray.getitem_tensor with rescale and the trend):

    registered = background + signal + residual
    background = mean image + spatial trend @ temporal trend        (the detrender's slow drift)
    signal     = noise SD * (U @ V)                                 (the block-wise low-rank fluctuations)
    residual   = whatever is left: masknmf's CompressionResidualArray

The movie, registration and compression come from imgui_masknmf's own steps, which call masknmf, so the blocks keep
every candidate component and the roughness test behind each block's rank.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from imgui_masknmf.pipeline import Run, Settings, step_compress, step_movie, step_register

STEPS = (("movie", step_movie), ("register", step_register), ("compress", step_compress))


@dataclass
class Decomposition:
    """One movie and its three pieces, all (frames, height, width) float32 in raw units."""

    name: str
    fs: float
    recorded: np.ndarray  # as loaded (flipped for a negative indicator)
    registered: np.ndarray
    shifts: np.ndarray  # (frames, 2), the (row, col) shift registration applied
    mean_image: np.ndarray  # (height, width), the trend's mean over time folded in
    trend_spatial: np.ndarray  # (height, width, trend rank); rank 0 without the detrender
    trend_temporal: np.ndarray  # (trend rank, frames), zero mean over time
    noise_sd: np.ndarray  # (height, width), what each pixel is divided by before the SVD
    background: np.ndarray
    signal: np.ndarray
    residual: np.ndarray
    blocks: list  # imgui_masknmf Blocks: position, every candidate, which were kept
    block_sizes: tuple
    rank: int
    rank_image: np.ndarray  # (height, width), components covering each pixel
    spatial_threshold: float
    temporal_threshold: float
    nonzero_u: int
    settings: Settings
    range_movie: tuple  # (low, high) to show the registered movie and the background with
    limit_signed: float  # +/- this shows the signal, the residual and the fluctuations

    @property
    def n_frames(self) -> int:
        return self.registered.shape[0]

    @property
    def shape(self) -> tuple:
        return self.registered.shape[1:]

    @property
    def stored_fraction(self) -> float:
        """What U (its nonzeros) and V hold, as a fraction of the movie's values."""
        pixels = self.shape[0] * self.shape[1]
        return (self.nonzero_u + self.rank * self.n_frames) / (pixels * self.n_frames)

    @property
    def fluctuations(self) -> np.ndarray:
        """The registered movie minus its background: what the SVD sees, before dividing by the noise SD."""
        return self.registered - self.background


def variance_shares(d: Decomposition) -> dict:
    """Each piece's variance over time, summed over pixels, as a share of the registered movie's."""
    total = float(d.registered.var(axis=0).sum())
    return {name: float(getattr(d, name).var(axis=0).sum()) / total for name in ("background", "signal", "residual")}


def lag1_correlation(frames: np.ndarray) -> np.ndarray:
    """(height, width) correlation of each pixel's trace with itself one frame later."""
    centered = frames - frames.mean(axis=0)
    numerator = (centered[1:] * centered[:-1]).sum(axis=0)
    return numerator / np.maximum((centered**2).sum(axis=0), 1e-12)


def busiest_pixel(d: Decomposition) -> tuple:
    """(row, col) where the signal varies most: a good first pixel to follow."""
    return tuple(int(v) for v in np.unravel_index(np.argmax(d.signal.std(axis=0)), d.shape))


def run_steps(settings: Settings, log: Optional[Callable[[str], None]] = None) -> Run:
    """
    Load, register and compress a movie with imgui_masknmf's steps.

    Parameters
    ----------
    settings : Settings
        imgui_masknmf's settings; only the movie, registration and compression ones are read.
    log : callable, optional
        Called with each step's name before it runs.

    Returns
    -------
    Run
    """
    run = Run(settings=settings.copy())
    for name, step in STEPS:
        if log is not None:
            log(name)
        step(run)
    return run


def split(run: Run) -> Decomposition:
    """Split a run's registered movie into background, signal and residual, as masknmf's compression rebuilds it."""
    pmd = run.compression.pmd
    n_frames, height, width = run.registration.frames.shape
    noise_sd = pmd.noise_variance_image.float()
    u, v = pmd.spatial_compressed.float(), pmd.temporal_compressed.float()
    signal = torch.sparse.mm(u, v) * noise_sd.flatten()[:, None]  ## (pixels, frames)
    mean_image = pmd.mean_image.float()
    if pmd.spatial_trend_basis is not None:
        trend_spatial, trend_temporal = pmd.spatial_trend_basis.float(), pmd.temporal_trend_basis.float()
    else:
        trend_spatial, trend_temporal = torch.zeros((height * width, 0)), torch.zeros((0, n_frames))
    background = mean_image.flatten()[:, None] + trend_spatial @ trend_temporal  ## (pixels, frames)
    registered = np.ascontiguousarray(run.registration.frames, dtype=np.float32)
    sample = slice(None, None, max(1, n_frames // 300))
    signal = np.ascontiguousarray(signal.T.reshape(n_frames, height, width).cpu().numpy())
    background = np.ascontiguousarray(background.T.reshape(n_frames, height, width).cpu().numpy())
    compression = run.compression
    return Decomposition(
        name=run.movie.name, fs=float(run.movie.fs),
        recorded=np.asarray(run.movie.frames, dtype=np.float32), registered=registered,
        shifts=np.asarray(run.registration.shifts, dtype=np.float32),
        mean_image=mean_image.reshape(height, width).cpu().numpy(),
        trend_spatial=trend_spatial.reshape(height, width, -1).cpu().numpy(),
        trend_temporal=trend_temporal.cpu().numpy(),
        noise_sd=noise_sd.reshape(height, width).cpu().numpy(),
        background=background, signal=signal, residual=registered - background - signal,
        blocks=compression.blocks, block_sizes=tuple(compression.block_sizes), rank=int(u.shape[1]),
        rank_image=np.asarray(compression.rank_image), spatial_threshold=float(compression.spatial_threshold),
        temporal_threshold=float(compression.temporal_threshold), nonzero_u=int(u._nnz()), settings=run.settings,
        range_movie=tuple(float(v) for v in np.percentile(registered[sample], [0.5, 99.8])),
        limit_signed=float(np.percentile(np.abs(signal[sample]), 99.7)),
    )


def decompose(settings: Settings, log: Optional[Callable[[str], None]] = None) -> Decomposition:
    """run_steps, then split; log also hears "split" before the split."""
    run = run_steps(settings, log)
    if log is not None:
        log("split")
    return split(run)
