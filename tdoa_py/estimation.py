import numpy as np
from scipy.signal import correlate, correlation_lags
from scipy.fft import fft, ifft

def estimate_tdoa(sig_ref, sig, fs, method='classic'):
    if method == 'classic':
        corr = correlate(sig_ref, sig, mode='full')
        lags = correlation_lags(len(sig), len(sig_ref), mode='full')
        lag = lags[np.argmax(corr)]
        return lag / fs

    elif method == 'phat':
        n = len(sig_ref) + len(sig) - 1
        X1 = fft(sig_ref, n=n)
        X2 = fft(sig, n=n)
        R = X1 * np.conj(X2)
        R /= np.abs(R) + 1e-15  # Evitar división por cero
        corr = np.real(ifft(R))
        lags = np.arange(-n // 2, n // 2 + 1)
        corr = np.roll(corr, n // 2)
        lag = lags[np.argmax(corr)]
        return lag / fs

    else:
        raise ValueError("Método no reconocido. Usar 'classic' o 'phat'.")

def estimate_doa(signals, d, fs, c=343.0, method='classic'):
    ref_idx = 0
    ref_signal = signals[ref_idx]
    n_mics = len(signals)

    tdoas = []
    for i in range(n_mics):
        if i == ref_idx:
            continue
        tdoa = estimate_tdoa(ref_signal, signals[i], fs, method)
        tdoas.append(tdoa)

    distances = [abs(i - ref_idx) * d for i in range(n_mics) if i != ref_idx]

    angles = []
    weighted_sum = 0
    idx = 0
    for i in range(n_mics):
        if i == ref_idx:
            continue
        dist = distances[idx]
        tdoa = tdoas[idx]
        idx += 1

        arg = tdoa * c / dist
        arg = np.clip(arg, -1.0, 1.0)
        phi_rad = np.arccos(arg)
        phi_deg = np.degrees(phi_rad)

        angles.append(phi_deg)
        weighted_sum += np.sign(i - ref_idx) * tdoa

    doa = np.mean(angles)
    return doa, tdoas
