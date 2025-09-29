import torch
import torch.nn as nn


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

        train_loss /= len(train_loader.dataset)

        # ---- Validation phase ----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

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
