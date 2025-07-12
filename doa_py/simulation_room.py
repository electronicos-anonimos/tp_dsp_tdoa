# import numpy as np
# import pyroomacoustics as pra
# import os
# import soundfile as sf

# def build_room_Nmics(
#     fs,
#     room_dim,
#     rt60,
#     n_mics,
#     mic_d,
#     mic_z,
#     mic_directivity='omni'
# ):
#     room_x, room_y, room_z = room_dim
#     cx, cy = room_x / 2, room_y / 2

#     if (n_mics - 1) * mic_d > room_x:
#         raise ValueError(f"El arreglo ({(n_mics - 1) * mic_d:.2f} m) excede room_x={room_x} m")

#     x_offsets = np.linspace(-(n_mics - 1) * mic_d / 2, (n_mics - 1) * mic_d / 2, n_mics)
#     mic_positions = np.vstack([
#         cx + x_offsets,
#         np.full(n_mics, cy),
#         np.full(n_mics, mic_z)
#     ])

#     abs_coeff, max_order = pra.inverse_sabine(rt60, room_dim)
#     room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(abs_coeff), max_order=max_order)
#     room.add_microphone_array(mic_positions)

#     room._sim_config = {
#         "fs": fs,
#         "room_dim": room_dim,
#         "rt60": rt60,
#         "n_mics": n_mics,
#         "mic_d": mic_d,
#         "mic_z": mic_z,
#         "mic_directivity": mic_directivity,
#         "mic_positions": mic_positions
#     }

#     return room, (cx, cy)

# callback_mix_kwargs = {
#     'snr': 30,
#     'sir': 10,
#     'n_src': 6,
#     'n_tgt': 2,
#     'ref_mic': 0,
# }

# def callback_mix(premix, snr=0, sir=0, ref_mic=0, n_src=None, n_tgt=None):
#     p_mic_ref = np.std(premix[:,ref_mic,:], axis=1)
#     premix /= p_mic_ref[:,None,None]

#     sigma_i = np.sqrt(10 ** (- sir / 10) / (n_src - n_tgt))
#     premix[n_tgt:n_src,:,:] *= sigma_i

#     sigma_n = np.sqrt(10 ** (- snr / 10))
#     mix = np.sum(premix[:n_src,:], axis=0) + sigma_n * np.random.randn(*premix.shape[1:])

#     return mix

# def sim_room_Nmics(
#     wav_path,
#     out_dir,
#     audio_name,
#     fs=48000,
#     room_dim=(10, 10, 5),
#     rt60=0.4,
#     snr_db=0,
#     n_mics=2,
#     mic_d=0.1,
#     mic_z=1.2,
#     mic_directivity='omni',
#     src_dist=5.0,
#     src_az_deg=45.0,
#     src_z=1.2,
#     save_audio=True,
#     prebuilt_room=None,
#     room_center=None
# ):
#     if save_audio:
#         os.makedirs(out_dir, exist_ok=True)

#     if prebuilt_room is not None and room_center is not None:
#         config = prebuilt_room._sim_config
#         room = pra.ShoeBox(
#             config["room_dim"],
#             fs=config["fs"],
#             materials=pra.Material(pra.inverse_sabine(config["rt60"], config["room_dim"])[0]),
#             max_order=pra.inverse_sabine(config["rt60"], config["room_dim"])[1]
#         )
#         room.add_microphone_array(config["mic_positions"])
#         cx, cy = room_center
#     else:
#         room, (cx, cy) = build_room_Nmics(fs, room_dim, rt60, n_mics, mic_d, mic_z, mic_directivity)

#     az = np.radians(src_az_deg)
#     src_pos = [cx + src_dist * np.cos(az), cy + src_dist * np.sin(az), src_z]

#     if not all(0 <= p <= d for p, d in zip(src_pos, room_dim)):
#         raise ValueError(f"Fuente fuera de la sala: {src_pos}")

#     signal, file_fs = sf.read(wav_path, dtype='float32')
#     if file_fs != fs:
#         raise ValueError(f"Se esperaba fs={fs}, pero se leyó fs={file_fs}")
#     signal /= np.max(np.abs(signal)) + 1e-8

#     room.add_source(position=src_pos, signal=signal)
#     room.simulate(snr=snr_db)

#     mic_signals = np.array(room.mic_array.signals)
#     mic_signals /= np.maximum(np.abs(mic_signals).max(axis=1, keepdims=True), 1e-9)

#     if save_audio:
#         paths = []
#         for i, sig in enumerate(mic_signals):
#             fname = os.path.join(out_dir, f"mic_{i+1}_{audio_name}.wav")
#             sf.write(fname, sig, fs)
#             paths.append(fname)
#         return src_az_deg, paths

#     return src_az_deg, mic_signals


# # ✅ NUEVA función para agregar ruido blanco digital a micrófonos
# def add_awgn_noise(signal_array, snr_db):
#     """
#     Agrega ruido blanco gaussiano a señales de micrófonos simuladas.
#     Controla el SNR en dB para análisis de robustez de estimadores DOA.
#     """
#     noisy = []
#     for sig in signal_array:
#         rms_sig = np.sqrt(np.mean(sig ** 2))
#         rms_noise = rms_sig / (10 ** (snr_db / 20))
#         noise = np.random.normal(0, rms_noise, size=sig.shape)
#         noisy.append(sig + noise)
#     return np.array(noisy)



# ## CODIGO OPTIMIZADO PARA SIMULACIÓN DE SALA CON N MICRÓFONOS - DEMORA UN POQUITO MENOS
# import numpy as np
# import pyroomacoustics as pra
# import os
# import soundfile as sf

# def add_awgn_per_channel(mic_signals, snr_db):
#     noisy_signals = []
#     for signal in mic_signals:
#         power_signal = np.mean(signal**2)
#         power_noise = power_signal / (10**(snr_db / 10))
#         noise = np.random.normal(0, np.sqrt(power_noise), size=signal.shape)
#         noisy_signals.append(signal + noise)
#     return np.array(noisy_signals)


# def build_room_Nmics(
#     fs,
#     room_dim,
#     rt60,
#     n_mics,
#     mic_d,
#     mic_z,
#     mic_directivity='omni'
# ):
#     room_x, room_y, room_z = room_dim
#     cx, cy = room_x / 2, room_y / 2

#     if (n_mics - 1) * mic_d > room_x:
#         raise ValueError(f"El arreglo ({(n_mics - 1) * mic_d:.2f} m) excede room_x={room_x} m")

#     x_offsets = np.linspace(-(n_mics - 1) * mic_d / 2, (n_mics - 1) * mic_d / 2, n_mics)
#     mic_positions = np.vstack([
#         cx + x_offsets,
#         np.full(n_mics, cy),
#         np.full(n_mics, mic_z)
#     ])

#     abs_coeff, max_order = pra.inverse_sabine(rt60, room_dim)
#     room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(abs_coeff), max_order=max_order)
#     room.add_microphone_array(mic_positions)

#     # Guardar info necesaria para reconstrucción
#     room._sim_config = {
#         "fs": fs,
#         "room_dim": room_dim,
#         "rt60": rt60,
#         "n_mics": n_mics,
#         "mic_d": mic_d,
#         "mic_z": mic_z,
#         "mic_directivity": mic_directivity,
#         "mic_positions": mic_positions
#     }

#     return room, (cx, cy)

# def sim_room_Nmics(
#     wav_path,
#     out_dir,
#     audio_name,
#     fs=48000,
#     room_dim=(10, 10, 5),
#     rt60=0.4,
#     snr_db=0,
#     snr_src_room=None,
#     n_mics=2,
#     mic_d=0.1,
#     mic_z=1.2,
#     mic_directivity='omni',
#     src_dist=5.0,
#     src_az_deg=45.0,
#     src_z=1.2,
#     save_audio=True,
#     prebuilt_room=None,
#     room_center=None,
# ):
#     if save_audio:
#         os.makedirs(out_dir, exist_ok=True)

#     if prebuilt_room is not None and room_center is not None:
#         config = prebuilt_room._sim_config
#         room = pra.ShoeBox(
#             config["room_dim"],
#             fs=config["fs"],
#             materials=pra.Material(pra.inverse_sabine(config["rt60"], config["room_dim"])[0]),
#             max_order=pra.inverse_sabine(config["rt60"], config["room_dim"])[1]
#         )
#         room.add_microphone_array(config["mic_positions"])
#         cx, cy = room_center
#     else:
#         room, (cx, cy) = build_room_Nmics(fs, room_dim, rt60, n_mics, mic_d, mic_z, mic_directivity)

#     az = np.radians(src_az_deg)
#     src_pos = [cx + src_dist * np.cos(az), cy + src_dist * np.sin(az), src_z]

#     if not all(0 <= p <= d for p, d in zip(src_pos, room_dim)):
#         raise ValueError(f"Fuente fuera de la sala: {src_pos}")

#     signal, file_fs = sf.read(wav_path, dtype='float32')
#     # signal_power = np.mean(signal**2)
#     if file_fs != fs:
#         raise ValueError(f"Se esperaba fs={fs}, pero se leyó fs={file_fs}")
#     signal /= np.max(np.abs(signal)) + 1e-8
    
#     # Fuente de ruido: señal blanca normalizada
#     if snr_src_room is not None:
#         # Señal principal 
#         power_signal = np.mean(signal**2)

#         # Generar ruido blanco
#         noise_signal = np.random.normal(0, 1, len(signal)).astype('float32')

#         # Calcular potencia actual del ruido
#         power_noise_raw = np.mean(noise_signal**2)

#         # Escalar ruido para que tenga la potencia deseada
#         target_power_noise = power_signal / (10**(snr_src_room / 10))
#         scaling_factor = np.sqrt(target_power_noise / power_noise_raw)
#         noise_signal *= scaling_factor

#         # No normalizar después del escalado
#         # Añadir como fuente en la sala
#         noise_pos = [room_dim[0] - src_pos[0], room_dim[1] - src_pos[1], src_z]
#         room.add_source(position=noise_pos, signal=noise_signal)


#     room.add_source(position=src_pos, signal=signal)
#     room.simulate(snr=snr_db)

#     mic_signals = np.array(room.mic_array.signals)
#     mic_signals /= np.maximum(np.abs(mic_signals).max(axis=1, keepdims=True), 1e-9)

#     # Añadir AWGN independiente por canal
#     if snr_db > 0:
#         mic_signals = add_awgn_per_channel(mic_signals, snr_db)


#     if save_audio:
#         paths = []
#         for i, sig in enumerate(mic_signals):
#             fname = os.path.join(out_dir, f"mic_{i+1}_{audio_name}.wav")
#             sf.write(fname, sig, fs)
#             paths.append(fname)
#         return src_az_deg, paths

#     return src_az_deg, mic_signals
import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import os

def build_room_Nmics(fs, room_dim, rt60, n_mics, mic_d, mic_z, mic_directivity='omni'):
    room_x, room_y, room_z = room_dim
    cx, cy = room_x / 2, room_y / 2
    x_offsets = np.linspace(-(n_mics - 1) * mic_d / 2, (n_mics - 1) * mic_d / 2, n_mics)
    mic_positions = np.vstack([
        cx + x_offsets,
        np.full(n_mics, cy),
        np.full(n_mics, mic_z)
    ])
    abs_coeff, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(abs_coeff), max_order=max_order)
    room.add_microphone_array(mic_positions)
    room._sim_config = {
        "fs": fs, "room_dim": room_dim, "rt60": rt60,
        "n_mics": n_mics, "mic_d": mic_d, "mic_z": mic_z,
        "mic_directivity": mic_directivity, "mic_positions": mic_positions
    }
    return room, (cx, cy)

def simulate_signal(room, signal, position):
    room.add_source(position=position, signal=signal)
    room.simulate()
    return np.array(room.mic_array.signals)

def combine_signals_with_snr(clean, noise, snr_db):
    power_clean = np.mean(clean**2, axis=1, keepdims=True)
    power_noise = np.mean(noise**2, axis=1, keepdims=True)
    target_noise_power = power_clean / (10**(snr_db / 10))
    scaling = np.sqrt(target_noise_power / power_noise)
    return clean + noise * scaling

def add_awgn_per_channel(mic_signals, snr_db):
    noisy_signals = []
    for signal in mic_signals:
        power_signal = np.mean(signal**2)
        power_noise = power_signal / (10**(snr_db / 10))
        noise = np.random.normal(0, np.sqrt(power_noise), size=signal.shape)
        noisy_signals.append(signal + noise)
    return np.array(noisy_signals)
