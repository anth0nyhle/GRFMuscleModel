import numpy as np
import torch
import torch.nn as nn


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    return model


def calc_r2_muscle(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=(0, 1))
    ss_tot = np.sum((y_true - np.mean(y_true, axis=(0, 1), keepdims=True)) ** 2, axis=(0, 1))
    
    r2_muscle = 1 - ss_res / ss_tot
    
    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl', 'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']
    
    for label, r2 in zip(muscle_labels, r2_muscle):
        print(f"{label}: {r2:.4f}")

    return r2_muscle


def calc_r2_overall(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    r2_overall = 1 - ss_res / ss_tot

    return r2_overall


def calc_rmse_muscle(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 1)))

    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl', 'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

    for label, rmse_val in zip(muscle_labels, rmse):
        print(f"{label}: {rmse_val:.4f}")

    return rmse


def calc_rrmse_muscle(y_true, y_pred):
    ranges = y_true.max(axis=(0, 1)) - y_true.min(axis=(0, 1))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 1)))
    relative_rmse_range = rmse / ranges

    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl', 'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

    for label, rel_range in zip(muscle_labels, relative_rmse_range):
        print(f"{label}: {rel_range:.4f}")

    return relative_rmse_range


def calc_nrmse_muscle(y_true, y_pred):
    means = y_true.mean(axis=(0, 1))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 1)))
    relative_rmse_mean = rmse / means

    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl', 'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

    for label, rel_range, rel_mean in zip(muscle_labels, relative_rmse_mean):
        print(f"{label}: {rel_range:.4f}, {rel_mean:.4f}")

    return relative_rmse_mean


def calc_rmspe_muscle(y_true, y_pred):
    rmspe = np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2, axis=(0, 1)))

    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl', 'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

    for label, rmspe_val in zip(muscle_labels, rmspe):
        print(f"{label}: {rmspe_val:.4f}")

    return rmspe


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


def calc_rmspe_overall(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    rmspe_overall = np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))

    return rmspe_overall


def eval_model(model, X_test_tensor, y_test_tensor):
    model.eval()

    test_loss = 0

    criterion = nn.MSELoss()

    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)

        test_loss = criterion(y_pred_tensor, y_test_tensor).item()

    return test_loss, y_pred_tensor