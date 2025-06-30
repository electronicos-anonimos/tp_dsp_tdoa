## IMPLEMENTACION OPTIMIZADA - DEMORA UN CUARTO DEL TIEMPO

import numpy as np
import soundfile as sf
from joblib import Parallel, delayed, parallel_backend
from scipy.signal import correlate, correlation_lags
from scipy.fft import fft, ifft, fftshift, next_fast_len

BACKEND = 'numpy'

def print_backend():
    print(f"⚙️  FFT backend: {BACKEND}")

def load_signals(signals_input, fs=None, return_array=False):
    if isinstance(signals_input, list) and all(isinstance(p, str) for p in signals_input):
        signals = []
        max_len = 0
        for f in signals_input:
            sig, curr_fs = sf.read(f)
            if fs is None:
                fs = curr_fs
            elif fs != curr_fs:
                raise ValueError(f"{f}: fs={curr_fs}, se esperaba fs={fs}")
            signals.append(np.asarray(sig, dtype=np.float32))
            max_len = max(max_len, len(sig))
    elif isinstance(signals_input, np.ndarray):
        signals = [np.asarray(sig, dtype=np.float32) for sig in signals_input]
        max_len = max(len(sig) for sig in signals)
    else:
        raise TypeError("signals_input debe ser lista de paths `.wav` o array NumPy (N, L)")
    padded = [np.pad(sig, (0, max_len - len(sig))) for sig in signals]
    return (padded if not return_array else np.array(padded)), fs

def precompute_ffts(signals, method):
    n = next_fast_len(signals.shape[1] * 2 - 1)
    ffts = fft(signals, n=n, axis=1)
    if method == 'phat':
        denom = np.abs(ffts) + 1e-10
        normed = ffts / denom
    elif method == 'roth':
        denom = np.abs(ffts[:, np.newaxis])**2 + 1e-10
        normed = ffts[:, np.newaxis] * np.conj(ffts) / denom
    elif method == 'scot':
        denom = np.sqrt((np.abs(ffts[:, np.newaxis])**2) * (np.abs(ffts)**2)) + 1e-10
        normed = ffts[:, np.newaxis] * np.conj(ffts) / denom
    else:
        normed = ffts[:, np.newaxis] * np.conj(ffts)
    return normed, n

def estimate_doa(signals_input, d, fs=None, c=343.0, method='classic', n_jobs=-1, verbose=False):
    signals, fs = load_signals(signals_input, fs, return_array=True)
    n_mics = signals.shape[0]
    pairs = [(i, j) for i in range(n_mics - 1) for j in range(i + 1, n_mics)]

    if method != 'classic':
        cross_spectra, n = precompute_ffts(signals, method)

    def compute_pair(i, j):
        if method == 'classic':
            sig1, sig2 = signals[i], signals[j]
            corr = correlate(sig1, sig2, mode='full')
            lags = correlation_lags(len(sig2), len(sig1), mode='full')
            lag = lags[np.argmax(corr)]
            tdoa = lag / fs
        else:
            R = np.array(cross_spectra[i, j], copy=True)
            if R.ndim == 0:
                R = R[np.newaxis]
            elif R.ndim == 1:
                R = R.reshape(1, -1)
            corr = fftshift(np.real(ifft(R, axis=-1))).squeeze()
            lags = np.arange(-n // 2 + 1, n // 2 + 1)
            lag = lags[np.argmax(corr)]
            tdoa = lag / fs
        dist = abs(j - i) * d
        angle = np.degrees(np.arccos(np.clip(tdoa * c / dist, -1, 1)))
        return i, angle, tdoa

    with parallel_backend("threading", n_jobs=n_jobs):
        results = Parallel()(delayed(compute_pair)(i, j) for i, j in pairs)

    angles_per_mic_ref = [[] for _ in range(n_mics - 1)]
    tdoas_per_mic_ref = [[] for _ in range(n_mics - 1)]
    for i, angle, tdoa in results:
        angles_per_mic_ref[i].append(angle)
        tdoas_per_mic_ref[i].append(tdoa)

    avg_angle = np.mean([np.mean(a) for a in angles_per_mic_ref])
    avg_tdoa = np.mean([np.mean(t) for t in tdoas_per_mic_ref])

    if verbose:
        print_backend()

    return avg_angle, avg_tdoa, angles_per_mic_ref, tdoas_per_mic_ref


## CODIGO VIEJO POR LAS DUDAS - DEMORA 4 VECES MAS


# import numpy as np
# import soundfile as sf
# from scipy.signal import correlate, correlation_lags
# from scipy.fft import fft, ifft

# def estimate_tdoa(sig_ref, sig, fs, method='classic'):
#     if method == 'classic':
#         corr = correlate(sig_ref, sig, mode='full')
#         lags = correlation_lags(len(sig), len(sig_ref), mode='full')
#         lag = lags[np.argmax(corr)]
#         return lag / fs

#     elif method == 'roth':
#         n = len(sig_ref) + len(sig) - 1
#         X1 = fft(sig_ref, n=n)
#         X2 = fft(sig, n=n)
#         R = X1 * np.conj(X2) / (np.abs(X2)**2 + 1e-10)
#         corr = np.real(ifft(R))
#         lags = np.arange(-n // 2, n // 2 + 1)
#         corr = np.roll(corr, n // 2)
#         lag = lags[np.argmax(corr)]
#         return lag / fs

#     elif method == 'phat':
#         n = len(sig_ref) + len(sig) - 1
#         X1 = fft(sig_ref, n=n)
#         X2 = fft(sig, n=n)
#         R = X1 * np.conj(X2)
#         R = R / (np.abs(R)+1e-10)
#         corr = np.real(ifft(R))
#         lags = np.arange(-n // 2, n // 2 + 1)
#         corr = np.roll(corr, n // 2)
#         lag = lags[np.argmax(corr)]
#         return lag / fs

#     elif method == 'scot':
#         n = len(sig_ref) + len(sig) - 1
#         X1 = fft(sig_ref, n=n)
#         X2 = fft(sig, n=n)
#         R = (X1 * np.conj(X2)) / np.sqrt(np.abs(X1)**2 * np.abs(X2)**2 + 1e-10) 
#         corr = np.real(ifft(R))
#         lags = np.arange(-n // 2, n // 2 + 1)
#         corr = np.roll(corr, n // 2)
#         lag = lags[np.argmax(corr)]
#         return lag / fs

#     else:
#         raise ValueError("Método no reconocido. Usar 'classic', 'phat', 'roth' o 'scot'.")

# def load_signals(signal_input, fs=None):
#     if isinstance(signal_input, list) and all(isinstance(p, str) for p in signal_input):
#         signals = []
#         max_len = 0
#         for f in signal_input:
#             sig, curr_fs = sf.read(f)
#             if fs is None:
#                 fs = curr_fs
#             elif fs != curr_fs:
#                 raise ValueError(f"{f}: fs={curr_fs}, se esperaba fs={fs}")
#             signals.append(np.asarray(sig, dtype=np.float32))
#             max_len = max(max_len, len(sig))
#     elif isinstance(signal_input, np.ndarray):
#         signals = [np.asarray(sig, dtype=np.float32) for sig in signal_input]
#         max_len = max(len(sig) for sig in signals)
#     else:
#         raise TypeError("signal_input debe ser lista de rutas `.wav` o un array NumPy (N, L).")

#     signals = [np.pad(sig, (0, max_len - len(sig))) for sig in signals]
#     return signals, fs

# def estimate_doa(signals, d, fs, c=343.0, method='classic'):
#     signals, fs = load_signals(signals, fs)
#     n_mics = len(signals)

#     angles_per_mic_ref = []
#     tdoas_per_mic_ref = []


#     for i in range(n_mics - 1):
#         ref_signal = signals[i]
#         angles = []
#         tdoas = []
#         for j in range(i + 1, n_mics):
#             sig = signals[j]
#             tdoa = estimate_tdoa(ref_signal, sig, fs, method)
#             dist = abs(j - i) * d
#             arg = tdoa * c / dist
#             arg = np.clip(arg, -1, 1)
#             phi_deg = np.degrees(np.arccos(arg))
#             angles.append(phi_deg)
#             tdoas.append(tdoa)
#         angles_per_mic_ref.append(angles)
#         tdoas_per_mic_ref.append(tdoas)

#     avg_angle = np.mean([np.mean(angles) for angles in angles_per_mic_ref])
#     avg_tdoa = np.mean([np.mean(tdoas) for tdoas in tdoas_per_mic_ref])

#     return avg_angle, avg_tdoa, angles_per_mic_ref, tdoas_per_mic_ref


# import numpy as np
# import soundfile as sf
# from scipy.signal import correlate, correlation_lags
# try:
#     from pyfftw.interfaces.scipy_fft import fft, ifft
#     import pyfftw
#     pyfftw.config.NUM_THREADS = 4
# except ImportError:
#     from scipy.fft import fft, ifft

# def load_signals(signal_input, fs=None):
#     if isinstance(signal_input, list) and all(isinstance(p, str) for p in signal_input):
#         signals = []
#         max_len = 0
#         for f in signal_input:
#             sig, curr_fs = sf.read(f)
#             if fs is None:
#                 fs = curr_fs
#             elif fs != curr_fs:
#                 raise ValueError(f"{f}: fs={curr_fs}, se esperaba fs={fs}")
#             signals.append(np.asarray(sig, dtype=np.float32))
#             max_len = max(max_len, len(sig))
#     elif isinstance(signal_input, np.ndarray):
#         signals = [np.asarray(sig, dtype=np.float32) for sig in signal_input]
#         max_len = max(len(sig) for sig in signals)
#     else:
#         raise TypeError("signal_input debe ser lista de rutas `.wav` o array NumPy (N, L).")

#     signals = [np.pad(sig, (0, max_len - len(sig))) for sig in signals]
#     return signals, fs

# def estimate_tdoa(sig_ref, sig, fs, method):
#     if method == 'classic':
#         corr = correlate(sig_ref, sig, mode='full')
#         lags = correlation_lags(len(sig), len(sig_ref), mode='full')
#         lag = lags[np.argmax(corr)]
#         return lag / fs
#     else:
#         n = len(sig_ref) + len(sig) - 1
#         X1 = fft(sig_ref, n=n)
#         X2 = fft(sig, n=n)
#         R = X1 * np.conj(X2)

#         if method == 'phat':
#             R /= (np.abs(R) + 1e-10)
#         elif method == 'roth':
#             R /= (np.abs(X2)**2 + 1e-10)
#         elif method == 'scot':
#             R /= (np.sqrt(np.abs(X1)**2 * np.abs(X2)**2) + 1e-10)
#         else:
#             raise ValueError("Método no reconocido: usar 'classic', 'phat', 'roth' o 'scot'.")

#         corr = np.real(ifft(R))
#         corr = np.roll(corr, n // 2)
#         lags = np.arange(-n // 2, n // 2 + 1)
#         lag = lags[np.argmax(corr)]
#         return lag / fs

# def estimate_doa(signals_input, d, fs=None, c=343.0, method='classic'):
#     signals, fs = load_signals(signals_input, fs)
#     n_mics = len(signals)
    
#     angles_per_ref = []
#     tdoas_per_ref = []

#     for i in range(n_mics - 1):
#         ref_signal = signals[i]
#         angles, tdoas = [], []
#         for j in range(i + 1, n_mics):
#             sig = signals[j]
#             tdoa = estimate_tdoa(ref_signal, sig, fs, method)
#             dist = abs(j - i) * d
#             angle = np.degrees(np.arccos(np.clip(tdoa * c / dist, -1, 1)))
#             angles.append(angle)
#             tdoas.append(tdoa)
#         angles_per_ref.append(angles)
#         tdoas_per_ref.append(tdoas)

#     avg_angle = np.mean([np.mean(a) for a in angles_per_ref])
#     avg_tdoa = np.mean([np.mean(t) for t in tdoas_per_ref])

#     return avg_angle, avg_tdoa, angles_per_ref, tdoas_per_ref