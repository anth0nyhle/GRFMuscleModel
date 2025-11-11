import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional


def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoader objects for training and validation datasets.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        batch_size (int): Batch size for both loaders.
        shuffle_train (bool): Whether to shuffle training data.

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def init_model_optimizer(
    model_class,
    model_params: Dict[str, Any],
    learning_rate: float,
    weight_decay: float,
    device: torch.device
):
    """
    Initialize model and optimizer with specified hyperparameters.

    Args:
        model_class: Model class to instantiate.
        model_params (dict): Dictionary of model parameters.
        learning_rate (float): Learning rate for optimizer.
        weight_decay (float): L2 regularization strength.
        device (torch.device): Device to move model to.

    Returns:
        tuple: (model, optimizer, criterion)
    """
    # Initialize model
    model = model_class(**model_params)
    model.to(device)

    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    return model, optimizer, criterion


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Train model for one epoch.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        optimizer (optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to use.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        # Move data to device if needed
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        train_loss += loss.item() * X_batch.size(0)

    # Average loss over all samples
    train_loss /= len(train_loader.dataset)  # type: ignore

    return train_loss


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Validate model for one epoch.

    Args:
        model (nn.Module): Model to validate.
        val_loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to use.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            # Move data to device if needed
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Accumulate loss
            val_loss += loss.item() * X_batch.size(0)

    # Average loss over all samples
    val_loss /= len(val_loader.dataset)  # type: ignore

    return val_loss


def train_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int = 50,
    patience: int = 10,
    verbose: bool = False
) -> Tuple[float, Dict[str, Any]]:
    """
    Train model with early stopping.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to use.
        num_epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        verbose (bool): Whether to print progress.

    Returns:
        tuple: (best_val_loss, best_model_state_dict)
    """
    best_val_loss = float('inf')
    best_model_state_dict = model.state_dict()
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # Print progress if verbose
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Break if patience limit reached
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    return best_val_loss, best_model_state_dict


def train_eval_optuna(
    train_dataset,
    val_dataset,
    model_class,
    model_params: Dict[str, Any],
    learning_rate: float,
    batch_size: int,
    regularization: float,
    device: torch.device,
    num_epochs: int = 50,
    patience: int = 10,
    verbose: bool = False
) -> Tuple[float, nn.Module]:
    """
    Complete training and evaluation pipeline for Optuna trials.

    This is the main function to call from Optuna objective.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        model_class: Model class to instantiate.
        model_params (dict): Dictionary of model hyperparameters.
        learning_rate (float): Learning rate.
        batch_size (int): Batch size.
        regularization (float): L2 regularization weight.
        device (torch.device): Device to use.
        num_epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        verbose (bool): Whether to print progress.

    Returns:
        tuple: (best_val_loss, trained_model)
    """
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, batch_size
    )

    # Initialize model and optimizer
    model, optimizer, criterion = init_model_optimizer(
        model_class, model_params, learning_rate, regularization, device
    )

    # Train with early stopping
    best_val_loss, best_model_state_dict = train_early_stopping(
        model, train_loader, val_loader, optimizer, criterion,
        device, num_epochs, patience, verbose
    )

    # Load best model state
    model.load_state_dict(best_model_state_dict)

    return best_val_loss, model


def train_final_model(
    train_dataset,
    val_dataset,
    test_dataset,
    model_class,
    best_params: Dict[str, Any],
    device: torch.device,
    num_epochs: int = 1000,
    patience: int = 20,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Train the final model with best hyperparameters and evaluate on test set.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        model_class: Model class to instantiate.
        best_params (dict): Best hyperparameters from Optuna.
        device (torch.device): Device to use.
        num_epochs (int): Maximum number of epochs for final training.
        patience (int): Early stopping patience for final training.
        save_path (str, optional): Path to save the final model. If None, model is not saved.
        verbose (bool): Whether to print progress.

    Returns:
        tuple: (trained_model, results_dict)
            - trained_model: The best trained model
            - results_dict: Dictionary with 'train_loss', 'val_loss', 'test_loss'
    """
    if verbose:
        print("=" * 50)
        print("Training final model with best hyperparameters:")
        print("=" * 50)
        for key, value in best_params.items():
            print(f"{key}: {value}")
        print("=" * 50)

    # Extract hyperparameters
    batch_size = best_params['batch_size']
    learning_rate = best_params['learning_rate']
    regularization = best_params.get('regularization', 0.0)

    # Prepare model parameters (exclude training-specific params)
    model_params = {k: v for k, v in best_params.items()
                    if k not in ['batch_size', 'learning_rate', 'regularization']}

    # Add any fixed parameters (e.g., input_size, output_size)
    if 'input_size' not in model_params:
        model_params['input_size'] = 3  # Adjust based on your data
    if 'output_size' not in model_params:
        model_params['output_size'] = 9  # Adjust based on your data

    # Train the model
    final_val_loss, best_model = train_eval_optuna(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_class=model_class,
        model_params=model_params,
        learning_rate=learning_rate,
        batch_size=batch_size,
        regularization=regularization,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        verbose=verbose
    )

    # Evaluate on test set
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()
    test_loss = validate_epoch(best_model, test_loader, criterion, device)

    # Calculate training loss on full training set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    train_loss = validate_epoch(best_model, train_loader, criterion, device)

    # Print results
    if verbose:
        print("=" * 50)
        print("Final Model Results:")
        print("=" * 50)
        print(f"Training Loss:   {train_loss:.6f}")
        print(f"Validation Loss: {final_val_loss:.6f}")
        print(f"Test Loss:       {test_loss:.6f}")
        print("=" * 50)

    # Save model if path is provided
    if save_path:
        torch.save(best_model.state_dict(), save_path)
        if verbose:
            print(f"Model saved to: {save_path}")

    # Return model and results
    results = {
        'train_loss': train_loss,
        'val_loss': final_val_loss,
        'test_loss': test_loss
    }

    return best_model, results
