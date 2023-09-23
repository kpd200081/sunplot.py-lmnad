import numpy as np
from scipy.signal import butter, sosfiltfilt


def get_isocline(x, z, s, s0):
    ret = np.zeros(len(x), dtype=np.float64)
    for i in range(0, len(x)):
        if type(s0) is np.ndarray:
            min_i = np.nanargmin(np.abs(s[:, i] - s0[i]))
            delta = np.abs(s0[i] - s[min_i, i])
        else:
            min_i = np.nanargmin(np.abs(s[:, i] - s0))
            delta = np.abs(s0 - s[min_i, i])
        if delta > np.abs(s[min_i, i] - s[min_i - 1 if min_i > 0 else min_i + 1, i]) * 1.5:
            ret[i] = np.NaN
        else:
            ret[i] = z[min_i]
    return ret


def butter_lowpass_filter(data, normal_cutoff, fs=1, order=5):
    sos = butter(order, normal_cutoff, 'low', fs=fs, output='sos')
    y = sosfiltfilt(sos, data, padtype=None)
    return y


def prepare_fft_sample(data, fs):
    filter = butter_lowpass_filter(data, 0.0002, fs)
    resample = filter[0::1]
    return filter  # np.append(resample, np.arange(len(data)*0.1))


def calculate_spectrum_x(isoclines, dx, fs_t):
    x_points_count = isoclines.shape[1]
    t_points_count = isoclines.shape[0]
    if x_points_count % 2 == 0:
        fft_size = round((x_points_count * 1.0) / 2) + 1
    else:
        fft_size = round((x_points_count * 1.0 + 1) / 2)
    ret = np.zeros((t_points_count, fft_size), dtype=np.float64)
    f = np.zeros(fft_size, dtype=np.float64)
    for i in range(0, t_points_count):
        data = isoclines[i, :]
        data = prepare_fft_sample(data, fs_t)
        ret[i, :] = 2 * np.abs(np.fft.rfft(data)) / fft_size
    f[:] = 1 / dx * np.arange(fft_size) / x_points_count
    return ret, f


def calculate_spectrum_t(isoclines, dt, fs_x):
    x_points_count = isoclines.shape[1]
    t_points_count = isoclines.shape[0]
    if t_points_count % 2 == 0:
        fft_size = round(t_points_count / 2) + 1
    else:
        fft_size = round((t_points_count + 1) / 2)
    ret = np.zeros((fft_size, x_points_count), dtype=np.float64)
    f = np.zeros(fft_size, dtype=np.float64)
    for i in range(0, x_points_count):
        data = isoclines[:, i]
        data = prepare_fft_sample(data, fs_x)
        ret[:, i] = 2 * np.abs(np.fft.rfft(data)) / fft_size
    f[:] = 2 * np.pi / dt * np.arange(fft_size) / t_points_count
    return ret, f


def calculate_diff_by_z(array, z):
    ret = np.empty(array.shape, dtype=np.float64)
    ret[:] = np.NaN
    for i in range(0, array.shape[1]):
        for j in range(0, array.shape[0] - 1):
            if array[j + 1, i] != np.NaN:
                ret[j, i] = array[j, i] - array[j + 1, i]
                ret[j, i] /= z[j] - z[j + 1]
    return ret
