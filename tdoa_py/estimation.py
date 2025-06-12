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
        R = X1 * np.conj(X2) / (np.abs(X2)**2)  # Ponderación Roth 
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
        R = R / np.abs(R)  # Evitar división por cero
        corr = np.real(ifft(R))
        lags = np.arange(-n // 2, n // 2 + 1)
        corr = np.roll(corr, n // 2)
        lag = lags[np.argmax(corr)]
        return lag / fs

    elif method == 'scot':
        n = len(sig_ref) + len(sig) - 1
        X1 = fft(sig_ref, n=n)
        X2 = fft(sig, n=n)
        R = (X1 * np.conj(X2)) / np.sqrt( np.abs(X1)**2 * np.abs(X2)**2)
        corr = np.real(ifft(R))
        lags = np.arange(-n // 2, n // 2 + 1)
        corr = np.roll(corr, n // 2)
        lag = lags[np.argmax(corr)]
        return lag / fs

    else:
        raise ValueError("Método no reconocido. Usar 'classic', 'phat' o 'roth'.")

def load_signals(signal_input, fs=None):
    """
    Carga señales desde archivos `.wav` o desde `numpy arrays`.

    Parámetros:
    - signal_input: lista de rutas a archivos `.wav` o `numpy array` de forma (N, L)
    - fs: frecuencia de muestreo deseada (se validará contra los archivos si son `.wav`)

    Retorna:
    - signals: lista de señales en `numpy arrays`
    - fs: frecuencia de muestreo detectada
    """
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
        # Si `signal_input` ya es un array de forma (N, L)
        signals = [np.asarray(sig, dtype=np.float32) for sig in signal_input]
        max_len = max(len(sig) for sig in signals)
    else:
        raise TypeError("signal_input debe ser una lista de rutas `.wav` o un `numpy array` de forma (N, L).")

    # Padding para igualar la longitud de todas las señales
    signals = [np.pad(sig, (0, max_len - len(sig))) for sig in signals]

    return signals, fs

def estimate_doa(signals, d, fs, c=343.0, method='classic'):
    
    signals, fs = load_signals(signals, fs)
    ref_idx = 0
    ref_signal = signals[ref_idx]
    n_mics = len(signals)

    tdoas = []
    for i in range(n_mics):
        if i == ref_idx:
            continue
        tdoa = estimate_tdoa(ref_signal, signals[i], fs, method) # dif temporal de arribo entre el n-ésimo mic y el de referencia
        tdoas.append(tdoa)

    distances = [abs(i - ref_idx) * d for i in range(n_mics) if i != ref_idx]

    angles = []
    weighted_sum = 0
    idx = 0
    args = []
    normalized_args = []
    
    for i in range(n_mics):
        if i == ref_idx:
            continue
        dist = distances[idx]
        tdoa = tdoas[idx]
        idx += 1
    
        args.append(tdoa * c / dist)

    max_val = np.max(np.abs(args))  # valor máximo en valor absoluto
    
    if max_val >= 1:
        normalized_args = args / max_val
    else:
        normalized_args = args        

    for arg in normalized_args:
        phi_rad = np.arccos(arg)
        phi_deg = np.degrees(phi_rad)
        angles.append(phi_deg)
        # weighted_sum += np.sign(i - ref_idx) * tdoa

    avg_angle = np.mean(angles)
    avg_tdoa = np.mean(tdoas)
    
    return avg_angle, avg_tdoa, angles, tdoas
