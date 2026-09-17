"""
Optional shallow MLP regressor using PyTorch.

This is included ONLY because a shallow MLP can sometimes capture
non-linear feature interactions that linear models miss, while being
more interpretable than a deep network.

Design decisions:
- 2-layer MLP (128 → 64 → 1): deep enough for interactions, shallow
  enough to train quickly and avoid overfitting on 20K samples.
- Dropout for regularization instead of weight decay.
- Early stopping based on validation loss.
- If the MLP doesn't beat Random Forest, we don't use it — honesty matters.

Interview talking point:
- "I included an MLP experiment to test whether learnable feature
   interactions improve over Random Forest's implicit ones. In practice,
   RF was [competitive/better] because our dataset is small enough that
   the MLP's extra capacity doesn't help."
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PriceMLPModel(nn.Module):
    """Shallow feedforward network for price regression."""

    def __init__(self, input_dim: int, hidden_sizes: List[int], dropout: float = 0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h

        # Output layer: single regression value
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class MLPTrainer:
    """Trains and evaluates the MLP regressor."""

    def __init__(self, config=None):
        self.config = config or AppConfig.model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[PriceMLPModel] = None
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def _to_dataloader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool = True
    ) -> DataLoader:
        """Convert numpy arrays to a DataLoader."""
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y)
        dataset = TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=self.config.mlp_batch_size, shuffle=shuffle)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "MLPTrainer":
        """
        Train the MLP with early stopping.

        Returns:
            self for chaining.
        """
        input_dim = X_train.shape[1]

        self.model = PriceMLPModel(
            input_dim=input_dim,
            hidden_sizes=self.config.mlp_hidden_sizes,
            dropout=self.config.mlp_dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.mlp_lr)
        criterion = nn.MSELoss()

        train_loader = self._to_dataloader(X_train, y_train, shuffle=True)
        val_loader = self._to_dataloader(X_val, y_val, shuffle=False)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        logger.info(f"Training MLP: {input_dim} → {self.config.mlp_hidden_sizes} → 1")
        logger.info(f"  Device: {self.device}, LR: {self.config.mlp_lr}, "
                     f"Patience: {self.config.mlp_patience}")

        for epoch in range(self.config.mlp_epochs):
            # Training
            self.model.train()
            epoch_train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * len(y_batch)

            epoch_train_loss /= len(y_train)
            self.train_losses.append(epoch_train_loss)

            # Validation
            self.model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    y_pred = self.model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    epoch_val_loss += loss.item() * len(y_batch)

            epoch_val_loss /= len(y_val)
            self.val_losses.append(epoch_val_loss)

            # Early stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{self.config.mlp_epochs} — "
                    f"train_loss: {epoch_train_loss:.6f}, val_loss: {epoch_val_loss:.6f}"
                )

            if patience_counter >= self.config.mlp_patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        logger.info(f"  MLP training complete. Best val loss: {best_val_loss:.6f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call .train() first.")

        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_t).cpu().numpy()

        return predictions

    def save(self, path=None) -> None:
        """Save model weights."""
        path = path or (AppConfig.paths.models / "mlp_model.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved MLP model → {path}")

    def load(self, path=None, input_dim: int = None) -> None:
        """Load model weights."""
        path = path or (AppConfig.paths.models / "mlp_model.pt")
        if input_dim is None:
            raise ValueError("input_dim required to reconstruct model architecture")
        self.model = PriceMLPModel(
            input_dim=input_dim,
            hidden_sizes=self.config.mlp_hidden_sizes,
            dropout=self.config.mlp_dropout,
        )
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.model.to(self.device)
        logger.info(f"Loaded MLP model ← {path}")
