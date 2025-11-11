import numpy as np
import torch
import torch.nn as nn


def load_model(model, model_path):
    """
    Load a trained model's state dictionary from disk.

    Args:
        model (nn.Module): The model architecture to load weights into.
        model_path (str): Path to the saved model state dictionary.

    Returns:
        nn.Module: Model with loaded weights.
    """
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    return model


def calc_r2_muscle(y_true, y_pred):
    """
    Calculate R² score for each muscle across all samples and timesteps.

    Args:
        y_true (np.ndarray): Ground truth values, shape (n_samples, n_timesteps, n_muscles).
        y_pred (np.ndarray): Predicted values, shape (n_samples, n_timesteps, n_muscles).

    Returns:
        np.ndarray: R² scores for each muscle, shape (n_muscles,).
    """
    # Sum of squared residuals for each muscle (average over samples and timesteps)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=(0, 1))

    # Total sum of squares for each muscles (variance from mean)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=(0, 1), keepdims=True)) ** 2, axis=(0, 1))

    # Calculate R-squared
    r2_muscle = 1 - (ss_res / ss_tot)

    # Print results for each muscle
    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl',
                     'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

    for label, r2 in zip(muscle_labels, r2_muscle):
        print(f"{label}: {r2:.4f}")

    return r2_muscle


def calc_r2_overall(y_true, y_pred):
    """
    Calculate overall R² score across all muscles, samples, and timesteps.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Overall R² score.
    """
    # Flatten all dimensions to calculate global R-squared
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Calculate sum of squared residuals and total sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2_overall = 1 - (ss_res / ss_tot)

    return r2_overall


def calc_rmse_muscle(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE) for each muscle.

    Args:
        y_true (np.ndarray): Ground truth values, shape (n_samples, n_timesteps, n_muscles).
        y_pred (np.ndarray): Predicted values, shape (n_samples, n_timesteps, n_muscles).

    Returns:
        np.ndarray: RMSE for each muscle, shape (n_muscles,).
    """
    # Calculate RMSE for each muscle (average over samples and timesteps)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 1)))

    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl',
                     'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

    # Print RMSE for each muscle
    for label, rmse_val in zip(muscle_labels, rmse):
        print(f"{label}: {rmse_val:.4f}")

    return rmse


def calc_rrmse_muscle(y_true, y_pred):
    """
    Calculate Relative RMSE (RMSE normalized by range) for each muscle.

    Args:
        y_true (np.ndarray): Ground truth values, shape (n_samples, n_timesteps, n_muscles).
        y_pred (np.ndarray): Predicted values, shape (n_samples, n_timesteps, n_muscles).

    Returns:
        np.ndarray: Relative RMSE for each muscle, shape (n_muscles,).
    """
    # Calculate range (max - min) for each muscle
    ranges = y_true.max(axis=(0, 1)) - y_true.min(axis=(0, 1))

    # Calculate RMSE for each muscle
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 1)))

    # NOrmalize RMSE by range
    relative_rmse_range = rmse / ranges

    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl',
                     'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

    # Print relative RMSE for each muscle
    for label, rel_range in zip(muscle_labels, relative_rmse_range):
        print(f"{label}: {rel_range:.4f}")

    return relative_rmse_range


def calc_rmse_overall(y_true, y_pred):
    """
    Calculate overall RMSE across all muscles, samples, and timesteps.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Overall RMSE.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    rmse_overall = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return rmse_overall


def calc_rrmse_overall(y_true, y_pred):
    """
    Calculate overall Relative RMSE (normalized by global data range).

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Overall relative RMSE.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Calculate global range
    ranges = y_true.max() - y_true.min()

    # Calculate RMSE and normalize by range
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    relative_rmse_overall = rmse / ranges

    return relative_rmse_overall


def calc_rrmse_weighted(y_true, y_pred):
    """
    Calculate range-weighted average of Relative RMSE across muscles.

    This weights each muscle's relative RMSE by its data range, giving more
    importance to muscles with larger force ranges.

    Args:
        y_true (np.ndarray): Ground truth values, shape (n_samples, n_timesteps, n_muscles).
        y_pred (np.ndarray): Predicted values, shape (n_samples, n_timesteps, n_muscles).

    Returns:
        float: Weighted relative RMSE.
    """
    # Calculate range for each muscle
    ranges = y_true.max(axis=(0, 1)) - y_true.min(axis=(0, 1))

    # Calculate RMSE for each muscle
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 1)))

    # Calculate relative RMSE
    relative_rmse_range = rmse / ranges

    # Weight by muscle ranges (muscles with larger ranges contribute more)
    relative_rmse_weighted = np.sum(relative_rmse_range * ranges) / np.sum(ranges)

    return relative_rmse_weighted


def calc_mae_muscle(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE) for each muscle.

    Args:
        y_true (np.ndarray): Ground truth values, shape (n_samples, n_timesteps, n_muscles).
        y_pred (np.ndarray): Predicted values, shape (n_samples, n_timesteps, n_muscles).

    Returns:
        np.ndarray: MAE for each muscle, shape (n_muscles,).
    """
    # Caluclate MAE for each muscle (average over ramples and timesteps)
    mae = np.mean(np.abs(y_true - y_pred), axis=(0, 1))

    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl',
                     'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

    # Print MAE for each muscle
    for label, mae_val in zip(muscle_labels, mae):
        print(f"{label}: {mae_val:.4f}")

    return mae


def calc_mae_weighted(y_true, y_pred):
    """
    Calculate range-weighted average of MAE across muscles.

    Args:
        y_true (np.ndarray): Ground truth values, shape (n_samples, n_timesteps, n_muscles).
        y_pred (np.ndarray): Predicted values, shape (n_samples, n_timesteps, n_muscles).

    Returns:
        float: Weighted MAE.
    """
    # Calculate range for each muscle
    ranges = y_true.max(axis=(0, 1)) - y_true.min(axis=(0, 1))

    # Calculate MAE for each muscle
    mae = np.mean(np.abs(y_true - y_pred), axis=(0, 1))

    # Normalize MAE by range
    relative_mae_range = mae / ranges

    # Weight by muscle ranges
    relative_mae_weighted = np.sum(relative_mae_range * ranges) / np.sum(ranges)

    return relative_mae_weighted


def calc_mae_overall(y_true, y_pred):
    """
    Calculate overall MAE across all muscles, samples, and timesteps.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Overall MAE.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mae_overall = np.mean(np.abs(y_true - y_pred))  # Calculate overall MAE

    return mae_overall


def eval_model(model, X_test_tensor, y_test_tensor):
    """
    Evaluate a trained model on test data.

    Args:
        model (nn.Module): The trained model to evaluate.
        X_test_tensor (torch.Tensor): Test input features.
        y_test_tensor (torch.Tensor): Test target values.

    Returns:
        tuple: (test_loss, predictions)
            - test_loss (float): MSE loss on test set
            - predictions (torch.Tensor): Model predictions
    """
    # Set model to evaluation mode (disables dropout, batch norm, etc.)
    model.eval()

    # Initialize loss
    test_loss = 0

    # Use Mean Squared Error as loss criterion
    criterion = nn.MSELoss()

    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Forward pass
        y_pred_tensor = model(X_test_tensor)

        # Calculate loss
        test_loss = criterion(y_pred_tensor, y_test_tensor).item()

    return test_loss, y_pred_tensor


def generate_latex_table(results_muscle_dict, results_overall_dict, muscle_labels):
    """
    Generate a LaTeX table comparing model performance across muscles.

    Args:
        results_muscle_dict (dict): Dictionary mapping model names to per-muscle metrics.
                                   Format: {'LSTM': [metric1, metric2, ...], ...}
        results_overall_dict (dict): Dictionary mapping model names to overall metrics.
                                    Format: {'LSTM': overall_metric, ...}
        muscle_labels (list): List of muscle names for table rows.

    Returns:
        str: LaTeX table code ready for inclusion in a document.
    """
    table = "\\begin{table}\n"
    table += "\\centering\n"
    table += "\\begin{tabular}{lcccc}\n"
    table += "\\toprule\n"
    table += "\\textbf{Muscle} & \\textbf{LSTM} & \\textbf{CNN-LSTM} & \\textbf{LSTM+Attention} & \\textbf{Transformer}\\\\\n"
    table += "\\midrule\n"

    for muscle, metrics in zip(muscle_labels, zip(*results_muscle_dict.values())):
        table += f"{{{muscle}}} & {metrics[0]:.4f} & {metrics[1]:.4f} & {metrics[2]:.4f} & {metrics[3]:.4f} \\\\\n"

    table += "\\midrule\n"

    table += f"Overall & {results_overall_dict['LSTM']:.4f} & \
        {results_overall_dict['CNN-LSTM']:.4f} & {results_overall_dict['LSTM+Attention']:.4f} & \
        {results_overall_dict['Transformer']:.4f} \\\\\n"

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\\caption{Caption}\n"
    table += "\\label{tab:results}\n"
    table += "\\end{table}\n"

    return table


def calc_mae_sample_muscle(y_true, y_pred):
    """
    Calculate MAE for each sample and muscle by averaging over time steps.

    This is useful for per-trial or per-subject analysis.

    Args:
        y_true (np.ndarray): Ground truth values, shape (n_samples, n_timesteps, n_muscles).
        y_pred (np.ndarray): Predicted values, shape (n_samples, n_timesteps, n_muscles).

    Returns:
        np.ndarray: MAE matrix of shape (n_samples, n_muscles).
    """
    # Calculate absolute error for all predictions
    abs_error = np.abs(y_true - y_pred)  # (n_samples, n_timesteps, n_muscles)

    # Average over time dimension to get per-sample, per-muscle MAE
    mae_matrix = np.mean(abs_error, axis=1)  # (n_samples, n_muscles)
    return mae_matrix
