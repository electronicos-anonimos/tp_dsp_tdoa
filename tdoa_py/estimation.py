import numpy as np
from scipy.signal import correlate, correlation_lags

def estimate_tdoa(sig1, sig2, fs):
    corr = correlate(sig1, sig2, mode='full')  # mic1 sobre mic2
    lags = correlation_lags(len(sig1), len(sig2), mode='full')
    lag = lags[np.argmax(corr)]
    return lag / fs

def estimate_doa_from_array_axis(signals, fs, d, c=343.0):
    ref_signal = signals[0]
    tdoas = []

    for i in range(1, len(signals)):
        tdoa = estimate_tdoa(ref_signal, signals[i], fs)
        tdoas.append(tdoa)

    avg_tdoa = np.mean(tdoas)
    argument = avg_tdoa * c / d
    argument = np.clip(argument, -1.0, 1.0)

    phi_rad = np.arccos(argument)
    phi_deg = np.degrees(phi_rad)

    return phi_deg, tdoas
