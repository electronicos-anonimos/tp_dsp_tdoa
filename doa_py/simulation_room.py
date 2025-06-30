## CODIGO OPTIMIZADO PARA SIMULACIÓN DE SALA CON N MICRÓFONOS - DEMORA UN POQUITO MENOS
import numpy as np
import pyroomacoustics as pra
import os
import soundfile as sf

def build_room_Nmics(
    fs,
    room_dim,
    rt60,
    n_mics,
    mic_d,
    mic_z,
    mic_directivity='omni'
):
    room_x, room_y, room_z = room_dim
    cx, cy = room_x / 2, room_y / 2

    if (n_mics - 1) * mic_d > room_x:
        raise ValueError(f"El arreglo ({(n_mics - 1) * mic_d:.2f} m) excede room_x={room_x} m")

    x_offsets = np.linspace(-(n_mics - 1) * mic_d / 2, (n_mics - 1) * mic_d / 2, n_mics)
    mic_positions = np.vstack([
        cx + x_offsets,
        np.full(n_mics, cy),
        np.full(n_mics, mic_z)
    ])

    abs_coeff, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(abs_coeff), max_order=max_order)
    room.add_microphone_array(mic_positions)

    # Guardar info necesaria para reconstrucción
    room._sim_config = {
        "fs": fs,
        "room_dim": room_dim,
        "rt60": rt60,
        "n_mics": n_mics,
        "mic_d": mic_d,
        "mic_z": mic_z,
        "mic_directivity": mic_directivity,
        "mic_positions": mic_positions
    }

    return room, (cx, cy)


def sim_room_Nmics(
    wav_path,
    out_dir,
    audio_name,
    fs=48000,
    room_dim=(10, 10, 5),
    rt60=0.4,
    snr_db=0,
    n_mics=2,
    mic_d=0.1,
    mic_z=1.2,
    mic_directivity='omni',
    src_dist=5.0,
    src_az_deg=45.0,
    src_z=1.2,
    save_audio=True,
    prebuilt_room=None,
    room_center=None
):
    if save_audio:
        os.makedirs(out_dir, exist_ok=True)

    if prebuilt_room is not None and room_center is not None:
        config = prebuilt_room._sim_config
        room = pra.ShoeBox(
            config["room_dim"],
            fs=config["fs"],
            materials=pra.Material(pra.inverse_sabine(config["rt60"], config["room_dim"])[0]),
            max_order=pra.inverse_sabine(config["rt60"], config["room_dim"])[1]
        )
        room.add_microphone_array(config["mic_positions"])
        cx, cy = room_center
    else:
        room, (cx, cy) = build_room_Nmics(fs, room_dim, rt60, n_mics, mic_d, mic_z, mic_directivity)

    az = np.radians(src_az_deg)
    src_pos = [cx + src_dist * np.cos(az), cy + src_dist * np.sin(az), src_z]

    if not all(0 <= p <= d for p, d in zip(src_pos, room_dim)):
        raise ValueError(f"Fuente fuera de la sala: {src_pos}")

    signal, file_fs = sf.read(wav_path, dtype='float32')
    if file_fs != fs:
        raise ValueError(f"Se esperaba fs={fs}, pero se leyó fs={file_fs}")
    signal /= np.max(np.abs(signal)) + 1e-8

    room.add_source(position=src_pos, signal=signal)
    room.simulate(snr=snr_db)

    mic_signals = np.array(room.mic_array.signals)
    mic_signals /= np.maximum(np.abs(mic_signals).max(axis=1, keepdims=True), 1e-9)

    if save_audio:
        paths = []
        for i, sig in enumerate(mic_signals):
            fname = os.path.join(out_dir, f"mic_{i+1}_{audio_name}.wav")
            sf.write(fname, sig, fs)
            paths.append(fname)
        return src_az_deg, paths

    return src_az_deg, mic_signals
