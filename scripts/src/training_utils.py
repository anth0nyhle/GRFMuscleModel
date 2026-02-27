import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import cast, Sized


# define data labels
GRF_LABELS = ['GRF_x', 'GRF_y', 'GRF_z']
MUSCLE_LABELS = ['tibpost', 'tibant', 'edl', 'ehl', 'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

# define index for data labels
GRF_DICT = {0: 'GRF_x', 1: 'GRF_y', 2: 'GRF_z'}
MUSCLE_DICT = {0: 'tibpost', 1: 'tibant', 2: 'edl', 3: 'ehl', 4: 'fdl', 5: 'fhl', 6: 'perbrev', 7: 'perlong', 8: 'achilles'}


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    patience: int = 10,
    criterion=nn.MSELoss()
):
    """
    Train a PyTorch model with early stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training dataset loader.
        val_loader (DataLoader): Validation dataset loader.
        num_epochs (int): Maximum number of epochs.
        learning_rate (float): Learning rate for Adam optimizer.
        patience (int): Early stopping patience (epochs).
        criterion: Loss function (default: MSELoss).

    Returns:
        model (nn.Module): Best trained model.
        history (dict): Training/validation loss history.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = model.state_dict()
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        # ---- Training phase ----
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(cast(Sized, train_loader.dataset))

        # ---- Validation phase ----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(cast(Sized, val_loader.dataset))

        # record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # progress update
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # ---- Early stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # save best weights
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # restore best model
    model.load_state_dict(best_model_state)

    return model, history


def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    criterion=nn.MSELoss(),
    device: torch.device = torch.device('cpu')
):
    """
    Evaluate a trained PyTorch model on test data.

    Args:
        model (nn.Module): The trained model to evaluate.
        X_test (np.ndarray): Test input features.
        y_test (np.ndarray): Test target values.
        criterion: Loss function (default: MSELoss).
        device (torch.device): Device to run evaluation on.

    Returns:
        float: Test loss value.
    """
    # Convert test data to torch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Set model to evaluation mode
    model.eval()

    # No need to calculate gradients during testing
    with torch.no_grad():
        # Forward pass
        test_outputs = model(X_test_tensor)

        # Calculate loss
        test_loss = criterion(test_outputs, y_test_tensor).item()

    print(f"Test Loss: {test_loss:.4f}")

    return test_loss, test_outputs, y_test_tensor


def plot_preds(
    test_outputs: torch.Tensor,
    y_test_tensor: torch.Tensor,
    sample_idx: int = 0,
    figsize: tuple = (15, 10)
) -> None:
    """
    Visualize predicted vs true muscle forces for a single sample.

    Args:
        test_outputs (torch.Tensor): Model predictions (batch_size, seq_length, num_muscles).
        y_test_tensor (torch.Tensor): Ground truth values (batch_size, seq_length, num_muscles).
        perc_stance (np.ndarray): Percent normalized stance for x-axis.
        muscle_dict (Dict[int, str]): Dictionary mapping muscle index to muscle name.
        sample_idx (int): Index of the sample to visualize (default: 0).
        figsize (tuple): Figure size (default: (15, 10)).

    Returns:
        None: Displays the plot.
    """
    # Move predictions and ground truth to CPU for visualization
    pred = test_outputs[sample_idx].cpu().numpy()  # Shape: (seq_length, num_muscles)
    true = y_test_tensor[sample_idx].cpu().numpy()  # Shape: (seq_length, num_muscles)

    num_muscles = pred.shape[1]
    perc_stance = np.linspace(0, 1, len(true[:, 0]))

    # Calculate grid dimensions (square-ish layout)
    n_rows = int(np.ceil(np.sqrt(num_muscles)))
    n_cols = int(np.ceil(num_muscles / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i in range(num_muscles):
        axes[i].plot(perc_stance, true[:, i], label="True")
        axes[i].plot(perc_stance, pred[:, i], label="Predicted", linestyle='dashed')
        axes[i].set_title(MUSCLE_DICT[i])
        axes[i].set_xlabel("Percent Normalized Stance (%)")
        axes[i].set_ylabel("Muscle Force (N)")
        axes[i].legend()

    # Hide unused subplots
    for i in range(num_muscles, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
