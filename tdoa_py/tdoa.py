import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import os
from scipy.signal import correlate, correlation_lags

class SimulationRoom:
    
    def __init__(self, room_dim, rt60, snr_db, simulation_name, fs=44100):
        self.room_dim = room_dim
        self.center_x = self.room_dim[0] / 2 #centrado en x
        self.center_y = self.room_dim[1] / 2 #centrado en y
        self.rt60 = rt60
        self.snr_db = snr_db
        self.fs = fs
        self.room = self._create_room()
        self.signal = None
        self.simulation_name = simulation_name
        # self.source = None
        
    def _create_room(self):
        abs_coeff, max_order = pra.inverse_sabine(self.rt60, self.room_dim)
        room = pra.ShoeBox(
            self.room_dim,
            fs=self.fs,
            materials=pra.Material(abs_coeff),
            max_order=max_order
        )
        return room

    def array_microphone(self, n_mics, mic_d, mic_z, mic_directivity):
          # Validar longitud del array
        array_length = (n_mics - 1) * mic_d
        if array_length > self.room_dim[0]:
            raise ValueError(f"La longitud del array de micrófonos ({array_length:.2f} m) excede el largo de la sala en X ({self.room_dim[0]} m). Reducir n_mics o mic_d.")
        
        # Posiciones de micrófonos (lineal en X)
        mic_x = self.center_x
        mic_y = self.center_y
        x_offsets = np.linspace(-(n_mics - 1) * mic_d / 2, (n_mics - 1) * mic_d / 2, n_mics)
        mic_positions = np.array([
            mic_x + x_offsets,
            [mic_y] * n_mics,
            [mic_z] * n_mics
        ])
        self.room.add_microphone_array(mic_positions)
    
    def load_wav(self, folder, wav_file):
        # Cargar señal anecoica
        self.signal, file_fs = sf.read(f'{folder}/{wav_file}')
        if file_fs != self.fs:
            raise ValueError(f"La señal tiene fs={file_fs}, pero se espera fs={self.fs}")
    
    def source(self, src_dist, src_az_deg, src_z):
        # Limitar src_dist según dimensiones
        max_radius = min(self.room_dim[0], self.room_dim[1]) / 2
        if src_dist > max_radius:
            raise ValueError(f"src_dist = {src_dist} supera el máximo permitido ({max_radius}) para las dimensiones {self.room_dim[:2]}.")
        
        # Validar altura
        if not (0 <= src_z <= self.room_dim[2]):
            raise ValueError(f"La altura de la fuente src_z = {src_z} no está dentro de los límites [0, {self.room_dim[2]}]")

        # Calcular posición de fuente (desde centro de sala)
        az_rad = np.radians(src_az_deg)
        src_x = self.center_x + src_dist * np.cos(az_rad)
        src_y = self.center_y + src_dist * np.sin(az_rad)
        src_pos = [src_x, src_y, src_z]
        print(f"Fuente en: x={src_x:.2f}, y={src_y:.2f}, z={src_z:.2f}")
        
        # Verificar que esté dentro de la sala
        if not (0 <= src_x <= self.room_dim[0]) or not (0 <= src_y <= self.room_dim[1]):
            raise ValueError(f"La posición de la fuente {src_pos} está fuera de los límites de la sala {room_dim}")
        
        # Agregar fuente
        self.room.add_source(position=src_pos, signal=self.signal)
    
    def run(self):
        # Simulación
        self.room.simulate(snr=self.snr_db)
        
    def save_simulation(self, out_dir):
        # Guardar señales simuladas
        out_dir = os.path.join(out_dir, self.simulation_name)
        os.makedirs(out_dir, exist_ok=True)
        
        paths = []
        for i, sig in enumerate(self.room.mic_array.signals):
            path = os.path.join(out_dir, f"mic_{i+1}_{self.simulation_name}.wav")
            sf.write(path, sig, self.fs)
            paths.append(path)

        return paths
    
    
class EstimateDOA:
    def __init__(self, mic_d, fs, wav_files, c=343):
        self.mic_d = mic_d
        self.wav_files = wav_files
        self.c = c
        self.n_mics = self._determinate_n_mics()
        self.fs = fs
        self.tdoa_values = []
        self.angle_values = []
        self.signals = [] 
        self.ref_signal = []
        self.avg_tdoa = []
        self.avg_angle_deg = []
        self.hemi_avgs = []

    def _determinate_n_mics(self):
        if len(self.wav_files) < 2:
            raise ValueError("Se requieren al menos 2 señales")
        
        print(len(self.wav_files))
        return len(self.wav_files)
            
    def load_signals(self):
        max_len = 0
        for f in self.wav_files:
            sig, curr_fs = sf.read(f)
            if self.fs is None:
                self.fs = curr_fs
            elif self.fs != curr_fs:
                raise ValueError(f"{f}: fs={curr_fs}, se esperaba fs={self.fs}")
            self.signals.append(sig)
            max_len = max(max_len, len(sig))
            
        # Zero padding
        self.signals = [np.pad(sig, (0, max_len - len(sig))) for sig in self.signals]
        
        # Micrófono de referencia (centro)
        self.ref_idx = self.n_mics // 2
        self.ref_signal = self.signals[self.ref_idx]

    def cross_correlation_classic(self):
        
        for i, sig in enumerate(self.signals):
                if i == self.ref_idx:
                    continue

                # Correlación cruzada
                corr = correlate(sig, self.ref_signal, mode='full', method='fft')
                lags = correlation_lags(len(sig), len(self.ref_signal), mode='full')
                lag = lags[np.argmax(corr)]
                tdoa = lag / self.fs
                self.tdoa_values.append(tdoa)

                # Distancia efectiva
                baseline = self.mic_d * abs(i - self.ref_idx)
                if baseline == 0:
                    continue

                # Calcular ángulo
                cos_val = np.clip(tdoa * self.c / baseline, -1.0, 1.0)
                angle_rad = np.arccos(cos_val)
                angle_deg = np.degrees(angle_rad)

                # Expandir a [0, 360) según signo del TDOA
                if tdoa < 0:
                    angle_deg = (360 - angle_deg) % 360

                print(f"Mic {i}: TDOA = {tdoa:.6f} s, Ángulo estimado = {angle_deg:.2f}°")
                self.angle_values.append(angle_deg)
        
        # Agrupar por hemisferios
        hemispheres = {
            "H1": [a for a in self.angle_values if (0 <= a < 90) or (270 <= a < 360)],
            "H2": [a for a in self.angle_values if 90 <= a < 270],
        }

        # Calcular promedio por hemisferio presente
        self.hemi_avgs = {h: np.mean(a) for h, a in hemispheres.items() if len(a) > 0}

        # Determinar hemisferio dominante
        dominant_hemi, dominant_angles = max(hemispheres.items(), key=lambda x: len(x[1]))
        if not dominant_angles:
            raise RuntimeError("No se pudo determinar un hemisferio dominante.")

        self.avg_angle_deg = np.mean(dominant_angles)
        self.avg_tdoa = np.mean(self.tdoa_values)

        print(f"\nHemisferio dominante: {dominant_hemi}")
        for h, val in self.hemi_avgs.items():
            print(f"{h}: Promedio = {val:.2f}°")

    
        return self.avg_angle_deg, self.avg_tdoa, self.hemi_avgs