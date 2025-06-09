import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import os

def sim_room_Nmics(
    wav_path,
    out_dir,
    fs=44100,

    room_dim=(10, 10, 5),
    rt60=0.4,
    snr_db=0,

    n_mics=2,
    mic_d=0.1,
    mic_z=1.2,
    mic_directivity='omni',

    src_dist=5.0,
    src_az_deg=45.0,
    src_z=1.2
):
    """
    Simula una sala reverberante con N micrófonos en array lineal y una fuente en campo lejano.

    El array se alinea a lo largo del eje X y se centra respecto al centro del plano XY.

    Parámetros:
    - wav_path: archivo .wav anecoico de entrada
    - out_dir: carpeta donde se guardarán los .wav simulados
    - fs: frecuencia de muestreo

    - room_dim: (Lx, Ly, Lz) dimensiones de la sala (m)
    - rt60: tiempo de reverberación (s)
    - snr_db: relación señal a ruido (dB)

    - n_mics: cantidad de micrófonos
    - mic_d: separación entre micrófonos (m)
    - mic_z: altura del array de micrófonos (m)
    - mic_directivity: tipo de directividad

    - src_dist: distancia radial de la fuente al centro de la sala (m)
    - src_az_deg: ángulo azimutal (°), CCW desde eje X+
    - src_z: altura absoluta de la fuente (m)

    Retorna:
    - src_angle: ángulo real de arribo (°)
    - mic_paths: lista de paths a los .wav simulados
    """

    os.makedirs(out_dir, exist_ok=True)

    # Centro de la sala
    center_x = room_dim[0] / 2
    center_y = room_dim[1] / 2

    # Validar longitud del array
    array_length = (n_mics - 1) * mic_d
    if array_length > room_dim[0]:
        raise ValueError(f"La longitud del array de micrófonos ({array_length:.2f} m) excede el largo de la sala en X ({room_dim[0]} m). Reducí n_mics o mic_d.")

    # Limitar src_dist según dimensiones
    max_radius = min(room_dim[0], room_dim[1]) / 2
    if src_dist > max_radius:
        raise ValueError(f"src_dist = {src_dist} supera el máximo permitido ({max_radius}) para las dimensiones {room_dim[:2]}.")

    # Validar altura
    if not (0 <= src_z <= room_dim[2]):
        raise ValueError(f"La altura de la fuente src_z = {src_z} no está dentro de los límites [0, {room_dim[2]}]")

    # Crear sala
    abs_coeff, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials=pra.Material(abs_coeff),
        max_order=max_order
    )

    # Posiciones de micrófonos (lineal en X)
    mic_x = center_x
    mic_y = center_y
    x_offsets = np.linspace(-(n_mics - 1) * mic_d / 2, (n_mics - 1) * mic_d / 2, n_mics)
    mic_positions = np.array([
        mic_x + x_offsets,
        [mic_y] * n_mics,
        [mic_z] * n_mics
    ])
    room.add_microphone_array(mic_positions)

    # Cargar señal anecoica
    signal, file_fs = sf.read(wav_path)
    if file_fs != fs:
        raise ValueError(f"La señal tiene fs={file_fs}, pero se espera fs={fs}")

    # Calcular posición de fuente (desde centro de sala)
    az_rad = np.radians(src_az_deg)
    src_x = center_x + src_dist * np.cos(az_rad)
    src_y = center_y + src_dist * np.sin(az_rad)
    src_pos = [src_x, src_y, src_z]
    print(f"Fuente en: x={src_x:.2f}, y={src_y:.2f}, z={src_z:.2f}")

    # Verificar que esté dentro de la sala
    if not (0 <= src_x <= room_dim[0]) or not (0 <= src_y <= room_dim[1]):
        raise ValueError(f"La posición de la fuente {src_pos} está fuera de los límites de la sala {room_dim}")

    # Agregar fuente
    room.add_source(position=src_pos, signal=signal)

    # Simulación
    room.simulate(snr=snr_db)

    # Guardar señales simuladas
    paths = []
    for i, sig in enumerate(room.mic_array.signals):
        path = os.path.join(out_dir, f"mic_{i+1}.wav")
        sf.write(path, sig, fs)
        paths.append(path)

    return src_az_deg, paths




