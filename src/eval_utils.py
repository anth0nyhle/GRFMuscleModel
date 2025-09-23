import numpy as np
import torch
import torch.nn as nn


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    return model


def calc_r2_muscle(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=(0, 1))
    ss_tot = np.sum((y_true - np.mean(y_true, axis=(0, 1), keepdims=True)) ** 2, axis=(0, 1))

    r2_muscle = 1 - (ss_res / ss_tot)

    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl',
                     'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

    for label, r2 in zip(muscle_labels, r2_muscle):
        print(f"{label}: {r2:.4f}")

    return r2_muscle


def calc_r2_overall(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2_overall = 1 - (ss_res / ss_tot)

    return r2_overall


def calc_rmse_muscle(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 1)))

    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl',
                     'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

    for label, rmse_val in zip(muscle_labels, rmse):
        print(f"{label}: {rmse_val:.4f}")

    return rmse


def calc_rrmse_muscle(y_true, y_pred):
    ranges = y_true.max(axis=(0, 1)) - y_true.min(axis=(0, 1))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 1)))
    relative_rmse_range = rmse / ranges

    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl',
                     'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

    for label, rel_range in zip(muscle_labels, relative_rmse_range):
        print(f"{label}: {rel_range:.4f}")

    return relative_rmse_range


def calc_rmse_overall(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    rmse_overall = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return rmse_overall


def calc_rrmse_overall(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    ranges = y_true.max() - y_true.min()
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    relative_rmse_overall = rmse / ranges

    return relative_rmse_overall


def calc_rrmse_weighted(y_true, y_pred):
    ranges = y_true.max(axis=(0, 1)) - y_true.min(axis=(0, 1))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 1)))
    relative_rmse_range = rmse / ranges

    relative_rmse_weighted = np.sum(relative_rmse_range * ranges) / np.sum(ranges)

    return relative_rmse_weighted


def calc_mae_muscle(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred), axis=(0, 1))  # Calculate MAE for each muscle

    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl',
                     'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

    for label, mae_val in zip(muscle_labels, mae):
        print(f"{label}: {mae_val:.4f}")

    return mae


def calc_mae_weighted(y_true, y_pred):
    ranges = y_true.max(axis=(0, 1)) - y_true.min(axis=(0, 1))
    mae = np.mean(np.abs(y_true - y_pred), axis=(0, 1))  # Calculate MAE for each muscle

    relative_mae_range = mae / ranges
    relative_mae_weighted = np.sum(relative_mae_range * ranges) / np.sum(ranges)

    return relative_mae_weighted


def calc_mae_overall(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mae_overall = np.mean(np.abs(y_true - y_pred))  # Calculate overall MAE

    return mae_overall


def eval_model(model, X_test_tensor, y_test_tensor):
    model.eval()

    test_loss = 0

    criterion = nn.MSELoss()

    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)

        test_loss = criterion(y_pred_tensor, y_test_tensor).item()

    return test_loss, y_pred_tensor


def generate_latex_table(results_muscle_dict, results_overall_dict, muscle_labels):
    table = "\\begin{table}\n"
    table += "\\centering\n"
    table += "\\begin{tabular}{lcccc}\n"
    table += "\\toprule\n"
    table += "\\textbf{Muscle} & \\textbf{LSTM} & \\textbf{CNN-LSTM} & \\textbf{LSTM+Attention} & \\textbf{Transformer}\\\\\n"
    table += "\\midrule\n"

    for muscle, metrics in zip(muscle_labels, zip(*results_muscle_dict.values())):
        table += f"{{{muscle}}} & {metrics[0]:.4f} & {metrics[1]:.4f} & {metrics[2]:.4f} & {metrics[3]:.4f} \\\\\n"

    table += "\\midrule\n"

    table += f"Overall & {results_overall_dict['LSTM']:.4f} & {results_overall_dict['CNN-LSTM']:.4f} & {results_overall_dict['LSTM+Attention']:.4f} & {results_overall_dict['Transformer']:.4f} \\\\\n"

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\\caption{Caption}\n"
    table += "\\label{tab:results}\n"
    table += "\\end{table}\n"

    return table


def calc_mae_sample_muscle(y_true, y_pred):
    """
    Calculate MAE for each sample and muscle by averaging over time steps.

    Returns:
        mae_matrix: (n_samples, n_muscles)
    """
    abs_error = np.abs(y_true - y_pred)         # (n_samples, n_timesteps, n_muscles)
    mae_matrix = np.mean(abs_error, axis=1)     # average over time → (n_samples, n_muscles)

    return mae_matrix
