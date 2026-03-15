"""
spectrogram.py
──────────────
STFT-based spectrogram generation and visualisation for radar RF signals.

Key responsibilities
  • Compute magnitude spectrograms via Short-Time Fourier Transform (STFT)
  • Log-scale compression and min-max normalisation → [0, 1] → (1, F, T) tensor
  • Save individual and grid spectrogram images
  • Plot per-class sample spectrograms for exploratory analysis
"""

import numpy as np
from scipy.signal import stft, get_window
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for scripts
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Default STFT hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_NPERSEG  = 128   # FFT window length (samples)
DEFAULT_NOVERLAP = 112   # overlap (87.5 % → time resolution)
DEFAULT_FS       = 20_000.0


# ──────────────────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────────────────
def compute_spectrogram(
    signal:   np.ndarray,
    fs:       float = DEFAULT_FS,
    nperseg:  int   = DEFAULT_NPERSEG,
    noverlap: int   = DEFAULT_NOVERLAP,
    window:   str   = "hann",
    db_scale: bool  = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the magnitude spectrogram of a 1-D real signal via STFT.

    Parameters
    ----------
    signal   : 1-D array  – preprocessed waveform
    fs       : float      – sampling frequency (Hz)
    nperseg  : int        – STFT window length
    noverlap : int        – number of overlapping samples
    window   : str        – window type ('hann', 'hamming', …)
    db_scale : bool       – apply 20·log10 compression

    Returns
    -------
    f     : frequencies array (Hz)
    t     : time array (s)
    S_mag : 2-D magnitude spectrogram (F × T)
    """
    win = get_window(window, nperseg)
    f, t, Zxx = stft(signal, fs=fs, window=win, nperseg=nperseg, noverlap=noverlap)
    S_mag = np.abs(Zxx)
    if db_scale:
        S_mag = 20 * np.log10(S_mag + 1e-10)
    return f, t, S_mag


def normalize_spectrogram(S: np.ndarray) -> np.ndarray:
    """Min-max normalise spectrogram to [0, 1]."""
    S_min, S_max = S.min(), S.max()
    return (S - S_min) / (S_max - S_min + 1e-8)


def signal_to_spectrogram_tensor(
    signal:   np.ndarray,
    fs:       float = DEFAULT_FS,
    nperseg:  int   = DEFAULT_NPERSEG,
    noverlap: int   = DEFAULT_NOVERLAP,
) -> np.ndarray:
    """
    Full pipeline: raw signal → normalised spectrogram → (1, F, T) ndarray
    ready to feed into a CNN.
    """
    _, _, S = compute_spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    S_norm = normalize_spectrogram(S)
    return S_norm[np.newaxis, ...]          # add channel dim


def batch_to_spectrograms(
    signals:  np.ndarray,
    fs:       float = DEFAULT_FS,
    nperseg:  int   = DEFAULT_NPERSEG,
    noverlap: int   = DEFAULT_NOVERLAP,
) -> np.ndarray:
    """
    Convert a batch of raw signals to a batch of spectrogram tensors.

    Parameters
    ----------
    signals : (N, L) ndarray

    Returns
    -------
    specs : (N, 1, F, T) ndarray
    """
    sample_tensor = signal_to_spectrogram_tensor(signals[0], fs, nperseg, noverlap)
    _, F, T = sample_tensor.shape
    out = np.empty((len(signals), 1, F, T), dtype=np.float32)
    for i, sig in enumerate(signals):
        out[i] = signal_to_spectrogram_tensor(sig, fs, nperseg, noverlap)
    logger.info("Converted %d signals → spectrograms, shape %s", len(signals), out.shape)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────
_CMAP = "magma"

# Colour palette for each class
_CLASS_COLORS = [
    "#4a9eff", "#ff6b6b", "#51cf66",
    "#ffd43b", "#cc5de8", "#ff922b",
]


def plot_single_spectrogram(
    signal:     np.ndarray,
    label:      str,
    class_idx:  int,
    save_path:  Optional[Path] = None,
    fs:         float = DEFAULT_FS,
    nperseg:    int   = DEFAULT_NPERSEG,
    noverlap:   int   = DEFAULT_NOVERLAP,
    show:       bool  = False,
) -> None:
    """Plot waveform + spectrogram side-by-side for one signal."""
    f, t, S = compute_spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

    fig = plt.figure(figsize=(14, 5), facecolor="#0d0d1a")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    color = _CLASS_COLORS[class_idx % len(_CLASS_COLORS)]

    # ── Waveform ────────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.set_facecolor("#12122a")
    time_axis = np.linspace(0, len(signal) / fs, len(signal))
    ax0.plot(time_axis * 1e3, signal, color=color, linewidth=0.7, alpha=0.9)
    ax0.set_xlabel("Time (ms)", color="#c0c0d0", fontsize=9)
    ax0.set_ylabel("Amplitude",  color="#c0c0d0", fontsize=9)
    ax0.set_title("RF Waveform",  color="#e0e0ff", fontsize=11, fontweight="bold")
    ax0.tick_params(colors="#888899")
    for spine in ax0.spines.values():
        spine.set_edgecolor("#2a2a4a")

    # ── Spectrogram ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    ax1.set_facecolor("#12122a")
    im = ax1.pcolormesh(t * 1e3, f / 1e3, normalize_spectrogram(S),
                        cmap=_CMAP, shading="gouraud", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax1, pad=0.02)
    cbar.set_label("Normalised Power", color="#c0c0d0", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#888899")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#888899", fontsize=7)
    ax1.set_xlabel("Time (ms)",      color="#c0c0d0", fontsize=9)
    ax1.set_ylabel("Frequency (kHz)", color="#c0c0d0", fontsize=9)
    ax1.set_title("STFT Spectrogram", color="#e0e0ff", fontsize=11, fontweight="bold")
    ax1.tick_params(colors="#888899")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#2a2a4a")

    # ── Title ───────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Signal Class: {label}",
        color=color, fontsize=14, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info("Saved spectrogram → %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_class_grid(
    signals:      np.ndarray,
    labels:       np.ndarray,
    class_names:  dict,
    n_per_class:  int  = 3,
    save_path:    Optional[Path] = None,
    fs:           float = DEFAULT_FS,
    nperseg:      int   = DEFAULT_NPERSEG,
    noverlap:     int   = DEFAULT_NOVERLAP,
    show:         bool  = False,
) -> None:
    """
    Plot a grid of spectrogram samples: rows = classes, cols = samples.
    """
    n_classes = len(class_names)
    fig, axes = plt.subplots(
        n_classes, n_per_class,
        figsize=(n_per_class * 4, n_classes * 2.8),
        facecolor="#0a0a18",
    )
    fig.suptitle(
        "Radar RF Signal Spectrograms – Class Overview",
        color="#e0e0ff", fontsize=15, fontweight="bold", y=1.01,
    )

    for cls_idx, cls_name in class_names.items():
        cls_signals = signals[labels == cls_idx]
        color = _CLASS_COLORS[cls_idx % len(_CLASS_COLORS)]
        for col in range(n_per_class):
            ax = axes[cls_idx][col]
            ax.set_facecolor("#0f0f22")
            if col < len(cls_signals):
                _, t, S = compute_spectrogram(
                    cls_signals[col], fs=fs, nperseg=nperseg, noverlap=noverlap,
                )
                ax.pcolormesh(
                    t * 1e3, np.arange(S.shape[0]),
                    normalize_spectrogram(S),
                    cmap=_CMAP, shading="gouraud",
                )
            if col == 0:
                ax.set_ylabel(cls_name, color=color, fontsize=8, fontweight="bold",
                              rotation=90, labelpad=5)
            if cls_idx == 0:
                ax.set_title(f"Sample {col + 1}", color="#aaaacc", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.2)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info("Saved class-grid → %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_stft_components(
    signal:    np.ndarray,
    label:     str,
    save_path: Optional[Path] = None,
    fs:        float = DEFAULT_FS,
    nperseg:   int   = DEFAULT_NPERSEG,
    noverlap:  int   = DEFAULT_NOVERLAP,
    show:      bool  = False,
) -> None:
    """
    Four-panel diagnostic plot:
      1. Raw waveform
      2. Power spectrum (FFT)
      3. STFT magnitude spectrogram
      4. Phase spectrogram
    """
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap,
                     window=get_window("hann", nperseg))
    S_mag   = np.abs(Zxx)
    S_phase = np.angle(Zxx)
    fft_mag = np.abs(np.fft.rfft(signal))
    fft_f   = np.fft.rfftfreq(len(signal), 1.0 / fs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#0d0d1a")
    fig.suptitle(f"STFT Analysis – {label}",
                 color="#e0e0ff", fontsize=14, fontweight="bold")

    panel_bg = "#12122a"
    ax_kw    = dict(color="#c0c0d0", fontsize=9)
    tick_kw  = dict(colors="#888899")

    # 1. Waveform
    ax = axes[0, 0]
    ax.set_facecolor(panel_bg)
    tx = np.linspace(0, len(signal) / fs, len(signal))
    ax.plot(tx * 1e3, signal, color="#4a9eff", linewidth=0.6)
    ax.set_title("Raw Waveform",    color="#e0e0ff", fontsize=10)
    ax.set_xlabel("Time (ms)",      **ax_kw)
    ax.set_ylabel("Amplitude",      **ax_kw)
    ax.tick_params(**tick_kw)

    # 2. Power spectrum
    ax = axes[0, 1]
    ax.set_facecolor(panel_bg)
    ax.semilogy(fft_f / 1e3, fft_mag + 1e-10, color="#ff6b6b", linewidth=0.8)
    ax.set_title("Power Spectrum",  color="#e0e0ff", fontsize=10)
    ax.set_xlabel("Frequency (kHz)", **ax_kw)
    ax.set_ylabel("Magnitude (log)", **ax_kw)
    ax.tick_params(**tick_kw)

    # 3. Magnitude spectrogram
    ax = axes[1, 0]
    ax.set_facecolor(panel_bg)
    im = ax.pcolormesh(t * 1e3, f / 1e3, 20 * np.log10(S_mag + 1e-10),
                       cmap=_CMAP, shading="gouraud")
    fig.colorbar(im, ax=ax).set_label("dB", color="#c0c0d0", fontsize=8)
    ax.set_title("Magnitude Spectrogram", color="#e0e0ff", fontsize=10)
    ax.set_xlabel("Time (ms)",            **ax_kw)
    ax.set_ylabel("Frequency (kHz)",      **ax_kw)
    ax.tick_params(**tick_kw)

    # 4. Phase spectrogram
    ax = axes[1, 1]
    ax.set_facecolor(panel_bg)
    im2 = ax.pcolormesh(t * 1e3, f / 1e3, S_phase,
                        cmap="twilight", shading="gouraud",
                        vmin=-np.pi, vmax=np.pi)
    fig.colorbar(im2, ax=ax).set_label("Phase (rad)", color="#c0c0d0", fontsize=8)
    ax.set_title("Phase Spectrogram", color="#e0e0ff", fontsize=10)
    ax.set_xlabel("Time (ms)",         **ax_kw)
    ax.set_ylabel("Frequency (kHz)",   **ax_kw)
    ax.tick_params(**tick_kw)

    for a in axes.flat:
        for spine in a.spines.values():
            spine.set_edgecolor("#2a2a4a")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info("Saved STFT component plot → %s", save_path)
    if show:
        plt.show()
    plt.close(fig)