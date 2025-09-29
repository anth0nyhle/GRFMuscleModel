import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def ad2float(array_double_col):
    """
    Convert an OpenSim ArrayDouble into a Numpy array of floats

    Args:
        array_double_col (object): OpenSim object with methods '.getSize()' and '.get(i)' for retrieving values

    Returns:
        np.ndarray: 1D Numpy array of floats containing the values from 'array_double_col'
    """
    # extract each value and cast to float
    float_col = [float(array_double_col.get(i)) for i in range(array_double_col.getSize())]

    return np.array(float_col)


def interp_segments(segments: list, n_interp_points: int):
    """
    Resample segments to a fixed number of interpolation points using linear interpolation

    Args:
        segments (list): list of 1D arrays representing signal segments of varying lengths
        n_interp_points (int): number of interpolation points to resample each segment to

    Returns:
        segments_resampled (list): list of resampled segments, each with length 'n_interp_points'
        time_resampled (list): list of resampled time vectors corresponding to each segment
    """
    segments_resampled = []
    time_resampled = []

    num_segments = len(segments)

    for i in range(num_segments):
        # original segment norimalized to [0, 1] in time
        original_time = np.linspace(0, 1, len(segments[i]))
        resampled_time = np.linspace(0, 1, n_interp_points)

        # linear interpolation of the segment
        interp_func = interp1d(original_time, segments[i], kind='linear')
        interp_seg = interp_func(resampled_time)

        segments_resampled.append(interp_seg)
        time_resampled.append(resampled_time)

    return segments_resampled, time_resampled


def exclude_segments(segments: list, min_len: int, max_len: int):
    """
    Filter out segments whose length falls outside a specified range

    Args:
        segments (list): list of 1D arrays representing signal segments
        min_len (int): minimum allowable length for a segment
        max_len (int): maximum allowable length for a segment

    Returns:
        list: filtered list of segments within the length range
    """
    return [seg for seg in segments if len(seg) >= min_len and len(seg) <= max_len]


def load_data(data_dir: str):
    """
    Load ground reaction force and estimated muscle force data from .npy files and concatenate

    Args:
        data_dir (str): path to the directory containing 'grf.npy' and 'muscle.npy'

    Returns:
        np.ndarray: Combined data array with shape (samples, timesteps, channels), where channels = GRF (3) + muscle forces (n)
    """
    # load the data with memory mapping for efficiency
    grf_data = np.load(data_dir + 'grf.npy', mmap_mode='r')
    muscle_data = np.load(data_dir + 'muscle.npy', mmap_mode='r')

    # concatenate along the last axis (channel dimension)
    data = np.concatenate((grf_data, muscle_data), axis=2)

    return data


def split_data(data: np.ndarray):
    """
    Split data into training, validation, and test sets (80/10/10 split)

    Args:
        data (np.ndarray): full dataset of shape (samples, timesteps, channels)

    Returns:
        tuples: X_train, y_train, X_val, y_val, X_test, y_test arrays, where X contains GRF channels (first 3), y contains muscle force channels (9)
    """
    # ensure reproducibility
    np.random.seed(42)
    np.random.shuffle(data)

    num_samples = data.shape[0]
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)

    print(f"Number of samples: {num_samples}")
    # print(f"Train size: {train_size}")
    # print(f"Validation size: {val_size}")
    print("--------------------")

    # split dataset
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]

    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print("--------------------")

    # separate inputs (GRF: first 3 channels) and outputs (muscles: remaining 9)
    X_train, y_train = train_data[:, :, :3], train_data[:, :, 3:]
    X_val, y_val = val_data[:, :, :3], val_data[:, :, 3:]
    X_test, y_test = test_data[:, :, :3], test_data[:, :, 3:]

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_achilles_force(y_train: np.ndarray, muscle_index: int = 8):
    """
    Plot Achilles tendon force across stance for training samples

    Args:
        y_train (np.ndarray): training output data of shape (samples, timesteps, muscles)
        muscle_index (int): Achilles tendon is asumed to be at index 8 (last in order)

    Returns:
        fig, ax (matplotlib Figure and Axes): figure and axes objects for further customization
    """
    num_samples, num_timesteps, _ = y_train.shape
    perc_stance = np.linspace(0, 100, num_timesteps)  # percent stance (0–100%)

    fig, ax = plt.subplots(figsize=(8, 6))

    # plot all samples with transparency
    for i in range(num_samples):
        ax.plot(perc_stance, y_train[i, :, muscle_index], linewidth=2, color="#A2C7E7", alpha=0.4)

    # overlay the mean curve in bold
    mean_curve = y_train[:, :, muscle_index].mean(axis=0)
    ax.plot(perc_stance, mean_curve, linewidth=3, color='#1A5EB6', label="mean")

    ax.set_ylabel("Muscle Force (N)", fontsize=14)
    ax.set_xlabel("Percent Normalized Stance (%)", fontsize=14)
    ax.set_title("Achilles Tendon Force Across Training Samples", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    return fig, ax
