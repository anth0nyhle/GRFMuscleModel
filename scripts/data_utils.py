import opensim as osim
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def ad2float(array_double_col):
    float_col = [float(array_double_col.get(i)) for i in range(array_double_col.getSize())]

    return np.array(float_col)


def interp_segments(segments, n_interp_points):
    segments_resampled = []
    time_resampled = []

    num_segments = len(segments)

    for i in range(num_segments):
        original_time = np.linspace(0, 1, len(segments[i]))
        resampled_time = np.linspace(0, 1, n_interp_points)

        interp_seg = np.zeros(n_interp_points)

        interp_func = interp1d(original_time, segments[i], kind='linear')
        interp_seg = interp_func(resampled_time)

        segments_resampled.append(interp_seg)
        time_resampled.append(resampled_time)

    return segments_resampled, time_resampled


def exclude_segments(segments, min_len, max_len):
    segments = [seg for seg in segments if len(seg) >= min_len and len(seg) <= max_len]

    return segments

