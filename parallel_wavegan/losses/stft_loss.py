# -*- coding: utf-8 -*-

"""STFT-based Loss modules."""

import torch


def stft(x, fft_size, hop_size, window_length, window):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

    """
    x_stft = torch.stft(x, fft_size, hop_size, window_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    return torch.sqrt(real ** 2 + imag ** 2).transpose(2, 1)


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Spectral convergence loss values.

        """
        numerator = torch.norm(y_mag - x_mag, p='fro', dim=2).view(-1)
        denominator = torch.norm(y_mag, p='fro', dim=2).view(-1)

        return torch.mean(numerator / denominator)


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Log STFT magnitude loss values.

        """
        n_bins = x_mag.size(2)
        x_lmag = torch.log(x_mag)
        y_lmag = torch.log(y_mag)

        return torch.mean(torch.norm(y_lmag - x_lmag, p=1, dim=2) / n_bins)


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, window_length=600, window_type='hann_window'):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.window_length = window_length
        self.window = getattr(torch, window_type)(window_length)
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: STFT loss values.

        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.window_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.window_length, self.window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss + mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 shift_sizes=[120, 240, 50],
                 window_lengths=[600, 1200, 240],
                 window_type='hann_window'):
        """Initialize Multi resolution STFT loss module."""
        super(STFTLoss, self).__init__()
        assert len(fft_sizes) == len(shift_sizes) == len(window_lengths)
        self.fft_sizes = fft_sizes
        self.shift_sizes = shift_sizes
        self.window_lengths = window_lengths
        self.windows = []
        for wl in window_lengths:
            self.windows += [getattr(torch, window_type)(wl)]
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Mutli resolution STFT loss values.

        """
        loss = 0.0
        for fs, ss, wl, w in zip(self.fft_sizes, self.shift_sizes, self.window_lengths, self.windows):
            x_mag = stft(x, fs, ss, wl, w)
            y_mag = stft(y, fs, ss, wl, w)
            loss += self.spectral_convergenge_loss(x_mag, y_mag)
            loss += self.log_stft_magnitude_loss(x_mag, y_mag)

        return loss / len(self.fft_sizes)
