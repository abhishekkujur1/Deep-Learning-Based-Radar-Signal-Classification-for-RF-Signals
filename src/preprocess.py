"""
preprocess.py
─────────────
Signal preprocessing pipeline for radar RF waveform data.

Responsibilities:
  • Synthetic dataset generation (6 radar signal classes)
  • Per-signal normalisation (zero-mean, unit-variance)
  • Optional Gaussian-noise removal via Wiener filter
  • Train / validation / test splitting with stratification
  • PyTorch Dataset wrapper that returns (spectrogram_tensor, label) pairs
"""

import numpy as np
from scipy.signal import wiener
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Signal class definitions
# ──────────────────────────────────────────────────────────────────────────────
SIGNAL_CLASSES = {
    0: "Background Noise",
    1: "Aerial Target – Fighter",
    2: "Aerial Target – UAV",
    3: "Ground Clutter",
    4: "Weather Clutter",
    5: "Jamming Signal",
}
NUM_CLASSES = len(SIGNAL_CLASSES)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic waveform generators
# ──────────────────────────────────────────────────────────────────────────────
def _generate_background_noise(n_samples: int, fs: float) -> np.ndarray:
    """White Gaussian noise – baseline environment."""
    return np.random.randn(n_samples) * 0.1


def _generate_fighter_signal(n_samples: int, fs: float) -> np.ndarray:
    """
    High-speed aerial target: LFM (chirp) with Doppler shift and
    micro-Doppler blade-flash modulation.
    """
    t = np.linspace(0, n_samples / fs, n_samples)
    f0, f1 = 1e3, 8e3                              # chirp band
    chirp = np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / t[-1] * t ** 2))
    doppler = np.sin(2 * np.pi * 2500 * t)         # Doppler carrier
    blade_flash = 0.3 * np.sin(2 * np.pi * 50 * t) # rotor modulation
    noise = np.random.randn(n_samples) * 0.05
    return chirp + 0.4 * doppler + blade_flash + noise


def _generate_uav_signal(n_samples: int, fs: float) -> np.ndarray:
    """
    Small UAV: narrow-band Doppler with high-frequency propeller harmonics.
    """
    t = np.linspace(0, n_samples / fs, n_samples)
    carrier = np.sin(2 * np.pi * 3000 * t)
    prop1   = 0.5 * np.sin(2 * np.pi * 150 * t)   # fundamental rotor
    prop2   = 0.3 * np.sin(2 * np.pi * 300 * t)   # 2nd harmonic
    prop3   = 0.2 * np.sin(2 * np.pi * 450 * t)   # 3rd harmonic
    noise   = np.random.randn(n_samples) * 0.08
    return carrier + prop1 + prop2 + prop3 + noise


def _generate_ground_clutter(n_samples: int, fs: float) -> np.ndarray:
    """Slow-moving / stationary ground clutter – low-frequency burst."""
    t = np.linspace(0, n_samples / fs, n_samples)
    burst = np.sin(2 * np.pi * 200 * t) * np.exp(-2 * t)
    noise = np.random.randn(n_samples) * 0.15
    return burst + noise


def _generate_weather_clutter(n_samples: int, fs: float) -> np.ndarray:
    """Distributed weather return: band-limited, slowly drifting."""
    t = np.linspace(0, n_samples / fs, n_samples)
    freqs = np.random.uniform(100, 2000, 10)
    signal = sum(np.sin(2 * np.pi * f * t + np.random.rand()) for f in freqs)
    envelope = 1 + 0.5 * np.sin(2 * np.pi * 0.5 * t)   # slow amplitude drift
    noise = np.random.randn(n_samples) * 0.2
    return signal * envelope / 10 + noise


def _generate_jamming_signal(n_samples: int, fs: float) -> np.ndarray:
    """Swept-frequency jammer: aggressive frequency-hopping pattern."""
    t = np.linspace(0, n_samples / fs, n_samples)
    hop_rate = 20                                       # hops per second
    hop_idx  = (t * hop_rate).astype(int) % 8
    freqs    = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    signal   = np.array([np.sin(2 * np.pi * freqs[h] * t[i])
                         for i, h in enumerate(hop_idx)])
    noise = np.random.randn(n_samples) * 0.1
    return signal + noise


_GENERATORS = [
    _generate_background_noise,
    _generate_fighter_signal,
    _generate_uav_signal,
    _generate_ground_clutter,
    _generate_weather_clutter,
    _generate_jamming_signal,
]


# ──────────────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────────────
def generate_synthetic_dataset(
    n_samples_per_class: int = 200,
    signal_length: int = 2048,
    fs: float = 20_000.0,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a balanced synthetic radar dataset.

    Returns
    -------
    signals : ndarray, shape (N, signal_length)   – raw waveforms
    labels  : ndarray, shape (N,)                 – integer class labels
    """
    np.random.seed(random_seed)
    signals, labels = [], []

    for cls_idx, generator in enumerate(_GENERATORS):
        logger.info(
            "Generating %d samples for class %d: %s",
            n_samples_per_class, cls_idx, SIGNAL_CLASSES[cls_idx],
        )
        for _ in range(n_samples_per_class):
            sig = generator(signal_length, fs)
            signals.append(sig)
            labels.append(cls_idx)

    signals = np.array(signals, dtype=np.float32)
    labels  = np.array(labels,  dtype=np.int64)

    # Shuffle
    idx = np.random.permutation(len(signals))
    return signals[idx], labels[idx]


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation (per signal)."""
    mu  = signal.mean()
    std = signal.std()
    return (signal - mu) / (std + 1e-8)


def denoise_signal(signal: np.ndarray, mysize: int = 5) -> np.ndarray:
    """Wiener adaptive noise filter."""
    return wiener(signal, mysize=mysize)


def preprocess_signals(
    signals: np.ndarray,
    apply_denoising: bool = False,
) -> np.ndarray:
    """
    Apply normalisation (and optional Wiener denoising) to every signal.

    Parameters
    ----------
    signals        : ndarray (N, L) – raw waveforms
    apply_denoising: bool           – run Wiener filter before normalisation

    Returns
    -------
    ndarray (N, L) – preprocessed waveforms
    """
    processed = np.empty_like(signals)
    for i, sig in enumerate(signals):
        if apply_denoising:
            sig = denoise_signal(sig)
        processed[i] = normalize_signal(sig)
    logger.info("Preprocessed %d signals (denoising=%s)", len(signals), apply_denoising)
    return processed


def split_dataset(
    signals: np.ndarray,
    labels:  np.ndarray,
    test_size:  float = 0.2,
    val_size:   float = 0.1,
    random_seed: int  = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train / validation / test split.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        signals, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_seed,
    )
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=relative_val,
        stratify=y_tmp,
        random_state=random_seed,
    )
    logger.info(
        "Split sizes – train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────
class RadarSignalDataset(Dataset):
    """
    Wraps pre-computed spectrogram tensors for use with DataLoader.

    Parameters
    ----------
    spectrograms : ndarray (N, 1, F, T)  – magnitude spectrograms
    labels       : ndarray (N,)          – integer labels
    augment      : bool                  – apply random time/freq masking
    """

    def __init__(
        self,
        spectrograms: np.ndarray,
        labels:       np.ndarray,
        augment:      bool = False,
    ) -> None:
        self.spectrograms = torch.FloatTensor(spectrograms)
        self.labels       = torch.LongTensor(labels)
        self.augment      = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        spec = self.spectrograms[idx]
        if self.augment:
            spec = self._random_mask(spec)
        return spec, self.labels[idx]

    @staticmethod
    def _random_mask(spec: torch.Tensor, mask_frac: float = 0.1) -> torch.Tensor:
        """SpecAugment-style random frequency and time masking."""
        _, F, T = spec.shape
        # Frequency mask
        f_width = int(F * mask_frac)
        f_start = np.random.randint(0, F - f_width + 1)
        spec[:, f_start:f_start + f_width, :] = 0.0
        # Time mask
        t_width = int(T * mask_frac)
        t_start = np.random.randint(0, T - t_width + 1)
        spec[:, :, t_start:t_start + t_width] = 0.0
        return spec