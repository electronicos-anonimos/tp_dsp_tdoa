from scipy.signal import correlate
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

def estimate_doa(audio_file1, audio_file2, mic_distance=0.1, sound_speed=343.0, fs=None):
    """
    Estima el ángulo de arribo (DOA) de una fuente sonora en campo lejano con dos micrófonos.

    Parámetros:
    - audio_file1, audio_file2: rutas a los archivos de audio (wav) grabados por los micrófonos.
    - mic_distance: distancia D entre micrófonos en metros (default 0.1 m).
    - sound_speed: velocidad del sonido en m/s (default 343 m/s).
    - fs: frecuencia de muestreo. Si None, se toma del archivo.

    Retorna:
    - doa_angle_deg: ángulo estimado en grados (respecto al eje del array).
    - tdoa: diferencia de tiempo estimada entre señales (en segundos).
    """

    # Cargar audios
    x1, fs1 = sf.read(audio_file1)
    x2, fs2 = sf.read(audio_file2)

    if fs is None:
        fs = fs1
    assert fs1 == fs2, "Las frecuencias de muestreo deben coincidir."

    # Asegurar igual longitud
    # min_len = min(len(x1), len(x2))
    # x1 = x1[:min_len]
    # x2 = x2[:min_len]

    # Correlación cruzada vía FFT
    corr = correlate(x1, x2, mode='full', method='fft')
    lags = correlation_lags(len(x1), len(x2), mode='full')
    lag_max = lags[np.argmax(corr)]

    # Calcular retardo temporal
    tdoa = lag_max / fs
    # Calcular ángulo
    arg = sound_speed * tdoa / mic_distance
    print(arg)
    if abs(arg) > 1:
        print("Advertencia: valor fuera del dominio de arccos, truncando.")
        arg = np.clip(arg, -1.0, 1.0)

    angle_rad = np.arccos(arg)
    doa_angle_deg = np.degrees(angle_rad)

    return doa_angle_deg, tdoa


