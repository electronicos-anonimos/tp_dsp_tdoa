import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags
from scipy.fft import fft, ifft

def estimate_tdoa(sig_ref, sig, fs, method='classic'):
    if method == 'classic':
        corr = correlate(sig_ref, sig, mode='full')
        lags = correlation_lags(len(sig), len(sig_ref), mode='full')
        lag = lags[np.argmax(corr)]
        return lag / fs

    elif method == 'roth':
        n = len(sig_ref) + len(sig) - 1
        X1 = fft(sig_ref, n=n)
        X2 = fft(sig, n=n)
        R = X1 * np.conj(X2) / (np.abs(X2)**2)
        corr = np.real(ifft(R))
        lags = np.arange(-n // 2, n // 2 + 1)
        corr = np.roll(corr, n // 2)
        lag = lags[np.argmax(corr)]
        return lag / fs

    elif method == 'phat':
        n = len(sig_ref) + len(sig) - 1
        X1 = fft(sig_ref, n=n)
        X2 = fft(sig, n=n)
        R = X1 * np.conj(X2)
        R = R / np.abs(R)
        corr = np.real(ifft(R))
        lags = np.arange(-n // 2, n // 2 + 1)
        corr = np.roll(corr, n // 2)
        lag = lags[np.argmax(corr)]
        return lag / fs

    elif method == 'scot':
        n = len(sig_ref) + len(sig) - 1
        X1 = fft(sig_ref, n=n)
        X2 = fft(sig, n=n)
        R = (X1 * np.conj(X2)) / np.sqrt(np.abs(X1)**2 * np.abs(X2)**2)
        corr = np.real(ifft(R))
        lags = np.arange(-n // 2, n // 2 + 1)
        corr = np.roll(corr, n // 2)
        lag = lags[np.argmax(corr)]
        return lag / fs

    else:
        raise ValueError("Método no reconocido. Usar 'classic', 'phat', 'roth' o 'scot'.")

def load_signals(signal_input, fs=None):
    if isinstance(signal_input, list) and all(isinstance(p, str) for p in signal_input):
        signals = []
        max_len = 0
        for f in signal_input:
            sig, curr_fs = sf.read(f)
            if fs is None:
                fs = curr_fs
            elif fs != curr_fs:
                raise ValueError(f"{f}: fs={curr_fs}, se esperaba fs={fs}")
            signals.append(np.asarray(sig, dtype=np.float32))
            max_len = max(max_len, len(sig))
    elif isinstance(signal_input, np.ndarray):
        signals = [np.asarray(sig, dtype=np.float32) for sig in signal_input]
        max_len = max(len(sig) for sig in signals)
    else:
        raise TypeError("signal_input debe ser lista de rutas `.wav` o un array NumPy (N, L).")

    signals = [np.pad(sig, (0, max_len - len(sig))) for sig in signals]
    return signals, fs

def estimate_doa(signals, d, fs, c=343.0, method='classic'):
    signals, fs = load_signals(signals, fs)
    n_mics = len(signals)

    angles_per_mic = []
    tdoas_per_mic = []

    for i in range(n_mics - 1):
        ref_signal = signals[i]
        angles = []
        tdoas = []
        for j in range(i + 1, n_mics):
            sig = signals[j]
            tdoa = estimate_tdoa(ref_signal, sig, fs, method)
            dist = abs(j - i) * d
            arg = tdoa * c / dist
            arg = np.clip(arg, -1, 1)
            phi_deg = np.degrees(np.arccos(arg))
            angles.append(phi_deg)
            tdoas.append(tdoa)
        angles_per_mic.append(np.mean(angles))
        tdoas_per_mic.append(np.mean(tdoas))

    avg_angle = np.mean(angles_per_mic)
    avg_tdoa = np.mean(tdoas_per_mic)

    return avg_angle, avg_tdoa, angles_per_mic, tdoas_per_mic

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
#         R = X1 * np.conj(X2) / (np.abs(X2)**2) # Ponderación Roth
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
#         R = R / np.abs(R)
#         corr = np.real(ifft(R))
#         lags = np.arange(-n // 2, n // 2 + 1)
#         corr = np.roll(corr, n // 2)
#         lag = lags[np.argmax(corr)]
#         return lag / fs

#     elif method == 'scot':
#         n = len(sig_ref) + len(sig) - 1
#         X1 = fft(sig_ref, n=n)
#         X2 = fft(sig, n=n)
#         R = (X1 * np.conj(X2)) / np.sqrt(np.abs(X1)**2 * np.abs(X2)**2)
#         corr = np.real(ifft(R))
#         lags = np.arange(-n // 2, n // 2 + 1)
#         corr = np.roll(corr, n // 2)
#         lag = lags[np.argmax(corr)]
#         return lag / fs

#     else:
#         raise ValueError("Método no reconocido. Usar 'classic', 'phat', 'roth' o 'scot'.")

# def load_signals(signal_input, fs=None):
#     """
#     Carga señales desde archivos `.wav` o desde `numpy arrays`.

#     Parámetros:
#     - signal_input: lista de rutas a archivos `.wav` o `numpy array` de forma (N, L)
#     - fs: frecuencia de muestreo deseada (se validará contra los archivos si son `.wav`)

#     Retorna:
#     - signals: lista de señales en `numpy arrays`
#     - fs: frecuencia de muestreo detectada
#     """
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
#          # Si `signal_input` ya es un array de forma (N, L)
#         signals = [np.asarray(sig, dtype=np.float32) for sig in signal_input]
#         max_len = max(len(sig) for sig in signals)
#     else:
#         raise TypeError("signal_input debe ser lista de rutas `.wav` o un array NumPy (N, L).")

#     # Padding para igualar la longitud de todas las señales
#     signals = [np.pad(sig, (0, max_len - len(sig))) for sig in signals]
#     return signals, fs

# def estimate_doa(signals, d, fs, c=343.0, method='classic'):
    
#     signals, fs = load_signals(signals, fs)
#     n_mics = len(signals)
#     results = []
#     avg_angles_ref = []

#     for ref_idx in range(n_mics):
#         ref_signal = signals[ref_idx]
#         tdoas = []
#         distances = []

#         for i in range(n_mics):
#             if i == ref_idx:
#                 continue
#             tdoa = estimate_tdoa(ref_signal, signals[i], fs, method)
#             tdoas.append(tdoa)
#             dist = abs(i - ref_idx) * d
#             distances.append(dist)

#         args = [tdoa * c / dist for tdoa, dist in zip(tdoas, distances)]
#         max_val = np.max(np.abs(args))
#         normalized_args = args / max_val if max_val >= 1 else args
#         angles = [np.degrees(np.arccos(arg)) for arg in normalized_args]
#         avg_angle = np.mean(angles)
#         avg_tdoa = np.mean(tdoas)
#         avg_angles_ref.append(avg_angle)
#         # results.append({
#         #     'ref_index': ref_idx,
#         #     'avg_angle': avg_angle,
#         #     'avg_tdoa': avg_tdoa,
#         #     'angles': angles,
#         #     'tdoas': tdoas
#         # })
#     angle = np.mean(avg_angles_ref)
    
#     return avg_angles_ref
