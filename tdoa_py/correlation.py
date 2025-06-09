from scipy.fft import fft, ifft
import numpy as np
import os
import soundfile as sf

def cross_correlation_fft(x, y):
    """
    Calcula la correlación cruzada entre dos señales usando únicamente FFT.
    """
    N = len(x) + len(y) - 1
    X = fft(x, N)
    Y = fft(y, N)
    corr = ifft(X * np.conj(Y)).real
    return np.fft.fftshift(corr)

def gcc_roth(x, y):
    """
    Aplica la correlación cruzada generalizada (GCC) con el procesador de Roth.
    """
    N = len(x) + len(y) - 1
    X = fft(x, N)
    Y = fft(y, N)

    # Densidad espectral de la referencia
    Pyy = np.abs(Y)**2 / N  # Normalización para obtener densidad espectral
    Pyy[Pyy == 0] = 1  # Evita división por cero
    
    gcc = ifft((X * np.conj(Y)) / Pyy).real
    return np.fft.fftshift(gcc)

def gcc_phat(x, y):
    """
    Aplica la correlación cruzada generalizada (GCC) con el procesador de PHAT.
    """
    N = len(x) + len(y) - 1
    X = fft(x, N)
    Y = fft(y, N)

    # Ponderación de PHAT: Normalización espectral
    weighting = np.abs(X * np.conj(Y))
    weighting[weighting == 0] = 1  # Evita división por cero
    
    gcc = ifft((X * np.conj(Y)) / weighting).real
    return np.fft.fftshift(gcc)

def correlation_lags_fft(len_x, len_y):
    """
    Calcula los desplazamientos de la correlación cruzada basados en FFT.
    """
    N = len_x + len_y - 1
    return np.arange(-len_x + 1, len_y)

def estimate_doa_from_wavs(wav_folder, mic_d, fs=None, method="classic"):
    """
    Estima el ángulo de arribo (DOA) y el TDOA promedio desde un array lineal con micrófonos omni.
    """
    c = 343  # velocidad del sonido (m/s)

    # Archivos .wav
    wav_files = sorted([
        os.path.join(wav_folder, f)
        for f in os.listdir(wav_folder)
        if f.endswith(".wav")
    ])
    n_mics = len(wav_files)
    if n_mics < 2:
        raise ValueError("Se requieren al menos 2 micrófonos.")

    # Cargar señales
    signals = []
    max_len = 0
    for f in wav_files:
        sig, curr_fs = sf.read(f)
        if fs is None:
            fs = curr_fs
        elif fs != curr_fs:
            raise ValueError(f"{f}: fs={curr_fs}, se esperaba fs={fs}")
        signals.append(sig)
        max_len = max(max_len, len(sig))

    # Zero padding
    signals = [np.pad(sig, (0, max_len - len(sig))) for sig in signals]

    # Micrófono de referencia (centro)
    ref_idx = n_mics // 2
    ref_signal = signals[ref_idx]

    tdoas = []
    angles = []

    for i, sig in enumerate(signals):
        if i == ref_idx:
            continue

        # Seleccionar método de correlación
        if method == 'classic':
            corr = cross_correlation_fft(sig, ref_signal)
        elif method == "gcc_roth":
            corr = gcc_roth(sig, ref_signal)
        elif method == "gcc_phat":
            corr = gcc_phat(sig, ref_signal)
        else:
            raise ValueError("Método de correlacion no válido")

        lags = correlation_lags_fft(len(sig), len(ref_signal))
        lag = lags[np.argmax(corr)]
        tdoa = lag / fs
        tdoas.append(tdoa)

        # Distancia efectiva
        baseline = mic_d * abs(i - ref_idx)
        if baseline == 0:
            continue

        # Calcular ángulo
        cos_val = np.clip(tdoa * c / baseline, -1.0, 1.0)
        angle_rad = np.arccos(cos_val)
        angle_deg = np.degrees(angle_rad)

        # Expandir a [0, 360) según signo del TDOA
        if tdoa < 0:
            angle_deg = (360 - angle_deg) % 360

        print(f"Mic {i}: TDOA = {tdoa:.6f} s, Ángulo estimado = {angle_deg:.2f}°")
        angles.append(angle_deg)

    # Agrupar por hemisferios
    hemispheres = {
        "H1": [a for a in angles if (0 <= a < 90) or (270 <= a < 360)],
        "H2": [a for a in angles if 90 <= a < 270],
    }

    # Calcular promedio por hemisferio presente
    hemi_avgs = {h: np.mean(a) for h, a in hemispheres.items() if len(a) > 0}

    # Determinar hemisferio dominante
    dominant_hemi, dominant_angles = max(hemispheres.items(), key=lambda x: len(x[1]))
    if not dominant_angles:
        raise RuntimeError("No se pudo determinar un hemisferio dominante.")

    avg_angle_deg = np.mean(dominant_angles)
    avg_tdoa = np.mean(tdoas)

    print(f"\nHemisferio dominante: {dominant_hemi}")
    for h, val in hemi_avgs.items():
        print(f"{h}: Promedio = {val:.2f}°")

    return avg_angle_deg, avg_tdoa, hemi_avgs

