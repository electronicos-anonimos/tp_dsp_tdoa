import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import os

class SimulationRoom():
    
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
    
    
