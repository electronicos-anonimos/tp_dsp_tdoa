from scipy.fft import fft, ifft
import numpy as np
import soundfile as sf

def cross_correlation_fft(x, y):
    """Calcula la correlación cruzada entre dos señales usando únicamente FFT."""
    N = len(x) + len(y) - 1
    X = fft(x, N)
    Y = fft(y, N)
    corr = ifft(X * np.conj(Y)).real
    return np.fft.fftshift(corr)

def gcc_roth(x, y):
    """Aplica GCC con el procesador de Roth."""
    N = len(x) + len(y) - 1
    X = fft(x, N)
    Y = fft(y, N)
    Pyy = np.abs(Y)**2 / N  
    Pyy[Pyy == 0] = 1  
    gcc = ifft((X * np.conj(Y)) / Pyy).real
    return np.fft.fftshift(gcc)

def gcc_phat(x, y):
    """Aplica GCC con el procesador de PHAT."""
    N = len(x) + len(y) - 1
    X = fft(x, N)
    Y = fft(y, N)
    weighting = np.abs(X * np.conj(Y))
    weighting[weighting == 0] = 1  
    gcc = ifft((X * np.conj(Y)) / weighting).real
    return np.fft.fftshift(gcc)

def correlation_lags_fft(len_x, len_y):
    """Calcula los desplazamientos de la correlación cruzada basados en FFT."""
    return np.arange(-len_x + 1, len_y)

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

def estimate_doa(signal_input, mic_d, fs, method="classic"):
    """
    Estima el ángulo de arribo (DOA) y el TDOA promedio desde un array lineal con micrófonos omni.

    Parámetros:
    - signal_input: lista de rutas a archivos `.wav` o `numpy array` de forma (N, L)
    - mic_d: separación entre micrófonos (m)
    - fs: frecuencia de muestreo
    - method: método de correlación ('classic', 'gcc_roth', 'gcc_phat')

    Retorna:
    - avg_angle_deg: ángulo de arribo promedio (°)
    - avg_tdoa: TDOA promedio (s)
    - hemi_avgs: promedio de ángulos por hemisferio
    - tdoas: lista de TDOAs
    - angles: lista de ángulos estimados
    """
    c = 343  
    signals, fs = load_signals(signal_input, fs)
    n_mics = len(signals)

    ref_idx = n_mics // 2
    ref_signal = signals[ref_idx]

    tdoas = []
    angles = []

    for i, sig in enumerate(signals):
        if i == ref_idx:
            continue

        if method == 'classic':
            corr = cross_correlation_fft(sig, ref_signal)
        elif method == "gcc_roth":
            corr = gcc_roth(sig, ref_signal)
        elif method == "gcc_phat":
            corr = gcc_phat(sig, ref_signal)
        else:
            raise ValueError("Método de correlación no válido")

        lags = correlation_lags_fft(len(sig), len(ref_signal))
        lag = lags[np.argmax(corr)]
        tdoa = lag / fs
        tdoas.append(tdoa)

        baseline = mic_d * abs(i - ref_idx)
        if baseline == 0:
            continue

        cos_val = np.clip(tdoa * c / baseline, -1.0, 1.0)
        angle_rad = np.arccos(cos_val)
        angle_deg = np.degrees(angle_rad)

        if tdoa < 0:
            angle_deg = (360 - angle_deg) % 360

        angles.append(angle_deg)

    hemispheres = {
        "H1": [a for a in angles if (0 <= a <= 90) or (270 <= a < 360)],
        "H2": [a for a in angles if 90 <= a < 270],
    }
    
    print(hemispheres)

    hemi_avgs = {h: np.mean(a) for h, a in hemispheres.items() if len(a) > 0}

    dominant_hemi, dominant_angles = max(hemispheres.items(), key=lambda x: len(x[1]))
    if not dominant_angles:
        raise RuntimeError("No se pudo determinar un hemisferio dominante.")

    avg_angle_deg = np.mean(dominant_angles)
    avg_tdoa = np.mean(tdoas)

    return avg_angle_deg, avg_tdoa, hemi_avgs, tdoas, angles
