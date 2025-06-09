import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import os
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, correlate, correlation_lags

class AudioLoader:
    """Carga y gestiona el audio anecoico desde un directorio y archivo especificado."""
    def __init__(self, dir_path, file_name):
        self.dir_path = dir_path
        self.file_name = file_name
        self.file_path = os.path.join(self.dir_path, self.file_name)
        self.audio_data = None
        self.fs = None  # Se determinará al cargar el archivo

    def load_audio(self):
        """Carga el archivo de audio y extrae la frecuencia de muestreo."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"El archivo {self.file_name} no existe en {self.dir_path}")
        
        self.audio_data, self.fs = sf.read(self.file_path)
        if self.audio_data is None or len(self.audio_data) == 0:
            raise ValueError("El archivo de audio está vacío o no pudo cargarse correctamente.")
        
        print(f"Audio cargado: {self.file_path}, Frecuencia de muestreo detectada: {self.fs} Hz")

class MicrophoneArray:
    """Representa un array de micrófonos dentro de una sala, con posición central como referencia."""
    def __init__(self, n_mics, mic_origin, displacement, axis):
        self.n_mics = n_mics
        self.mic_origin = np.array(mic_origin)
        self.displacement = displacement
        self.axis = axis.lower()
        self.positions = self.generate_positions()

    def generate_positions(self):
        """Genera posiciones de los micrófonos de manera simétrica respecto al punto medio."""
        positions = []
        mid_index = self.n_mics // 2

        for i in range(-mid_index, mid_index + 1):
            if self.n_mics % 2 == 0 and i == 0:
                continue  # Evitar micrófono central en arreglos pares

            new_position = self.mic_origin.copy()

            if self.axis == 'x':
                new_position[0] += i * self.displacement
            elif self.axis == 'y':
                new_position[1] += i * self.displacement
            elif self.axis == 'z':
                new_position[2] += i * self.displacement

            positions.append(new_position)

        return positions

class RoomSimulation:
    """Representa una sala acústica con un array de micrófonos."""
    def __init__(self, room_dim, rt60, microphone_array, audio_loader):
        self.room_dim = room_dim
        self.rt60 = rt60
        self.microphone_array = microphone_array
        self.audio_loader = audio_loader
        self.room = None

    def setup_room(self):
        """Configura la sala en pyroomacoustics."""
        abs_coeff, max_order = pra.inverse_sabine(self.rt60, self.room_dim)
        self.room = pra.ShoeBox(self.room_dim, fs=self.audio_loader.fs, materials=pra.Material(abs_coeff), max_order=max_order)
        for pos in self.microphone_array.positions:
            self.room.add_microphone(pos)

    def add_source(self, source_position):
        """Añade una fuente de sonido a la sala y verifica que la señal esté disponible."""
        if self.audio_loader.audio_data is None or len(self.audio_loader.audio_data) == 0:
            raise ValueError("La señal de audio está vacía o no ha sido cargada correctamente.")
        
        self.room.add_source(source_position, signal=self.audio_loader.audio_data)
        self.room.compute_rir()

    def simulate(self):
        """Ejecuta la simulación acústica."""
        if not self.room.rir or not self.room.rir[0]:
            raise ValueError("La respuesta al impulso (RIR) no ha sido calculada correctamente.")
        self.room.simulate()

class DirectionEstimator:
    """Estima la dirección de arribo de una fuente sonora usando el array de micrófonos."""
    @staticmethod
    def estimate_doa(signals, mic_d, fs):
        """Calcula el DOA y TDOA a partir de señales en memoria."""
        if signals is None or len(signals) == 0:
            raise ValueError("Las señales no han sido cargadas correctamente.")
        
        c = 343  # velocidad del sonido (m/s)
        n_mics = len(signals)

        if n_mics < 2:
            raise ValueError("Se requieren al menos 2 micrófonos.")

        ref_idx = n_mics // 2
        ref_signal = signals[ref_idx]

        tdoas = []
        angles = []

        for i, sig in enumerate(signals):
            if i == ref_idx:
                continue

            corr = correlate(sig, ref_signal, mode='full', method='fft')
            lags = correlation_lags(len(sig), len(ref_signal), mode='full')
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

        avg_angle_deg = np.mean(angles)
        avg_tdoa = np.mean(tdoas)

        return avg_angle_deg, avg_tdoa

