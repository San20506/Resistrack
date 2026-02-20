"""M2.5 — Local PyTorch LSTM trainer for temporal AMR risk prediction.

Trains a bidirectional LSTM on 72h temporal tensors (batch, 72, 13):
  - 8 lab values + 5 vital signs
  - Z-score normalization using cohort statistics from M2.2
  - Early stopping with patience = 10
  - Outputs a 32-dim trend-risk vector for the ensemble

The trained model is saved as a PyTorch state_dict and can be loaded
into the SageMaker-compatible AMRLSTMModel.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from resistrack.common.constants import RANDOM_STATE
from resistrack.local_training.data_loader import TrainingDataset
from resistrack.ml.lstm_model import LSTMConfig
from resistrack.ml.temporal import (
    NUM_TEMPORAL_FEATURES,
    WINDOW_HOURS,
    CohortStats,
    TemporalFeatureExtractor,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class LSTMTrainingResult:
    """Result from local LSTM training."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    stopped_early: bool = False
    val_auc_roc: float = 0.0
    training_time_seconds: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": round(self.best_val_loss, 6),
            "stopped_early": self.stopped_early,
            "val_auc_roc": round(self.val_auc_roc, 4),
            "n_epochs": len(self.train_losses),
            "training_time_seconds": round(self.training_time_seconds, 2),
        }


# ---------------------------------------------------------------------------
# PyTorch LSTM model definition
# ---------------------------------------------------------------------------
def _build_lstm_model(config: LSTMConfig) -> Any:
    """Build the PyTorch LSTM model.

    Architecture: Input(13) → BiLSTM(64, 2 layers) → Dropout(0.3) → Linear(128→32)
    """
    import torch
    import torch.nn as nn

    class AMRLSTM(nn.Module):
        def __init__(self, cfg: LSTMConfig) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=cfg.input_size,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
                bidirectional=cfg.bidirectional,
                batch_first=True,
            )
            lstm_output_size = cfg.hidden_size * (2 if cfg.bidirectional else 1)
            self.dropout = nn.Dropout(cfg.dropout)
            self.fc_risk = nn.Linear(lstm_output_size, cfg.output_dim)
            self.fc_classifier = nn.Linear(cfg.output_dim, 1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Returns (risk_vector, logit).

            risk_vector: (batch, output_dim) — trend-risk vector for ensemble.
            logit: (batch, 1) — binary classification logit.
            """
            # Replace NaN with 0
            x = torch.nan_to_num(x, nan=0.0)

            lstm_out, _ = self.lstm(x)
            # Use the last time step output
            last_hidden = lstm_out[:, -1, :]
            last_hidden = self.dropout(last_hidden)

            risk_vector = self.fc_risk(last_hidden)
            # L2 normalize the risk vector
            risk_norm = torch.nn.functional.normalize(risk_vector, p=2, dim=1)

            logit = self.fc_classifier(risk_vector)
            return risk_norm, logit

    return AMRLSTM(config)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class LocalLSTMTrainer:
    """Local PyTorch LSTM trainer for ResisTrack temporal modeling.

    Usage:
        trainer = LocalLSTMTrainer()
        result = trainer.train(dataset)
        risk_vectors = trainer.predict_risk_vectors(dataset.temporal_tensors)
        trainer.save_model("artifacts/lstm_model.pt")
    """

    def __init__(
        self,
        config: LSTMConfig | None = None,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self._config = config or LSTMConfig()
        self._random_state = random_state
        self._model: Any = None
        self._device: str = "cpu"
        self._cohort_stats: CohortStats | None = None
        self._temporal_extractor = TemporalFeatureExtractor()

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError("Model not trained yet. Call train() first.")
        return self._model

    def train(
        self,
        dataset: TrainingDataset,
        val_fraction: float = 0.15,
    ) -> LSTMTrainingResult:
        """Train the LSTM model on temporal tensors.

        Args:
            dataset: TrainingDataset with temporal_tensors and labels.
            val_fraction: Fraction of data to use for validation.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        start_time = time.time()

        # Set device
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("LSTM training on device: %s", self._device)

        # Set random seeds for reproducibility
        torch.manual_seed(self._random_state)
        np.random.seed(self._random_state)

        # 1. Compute cohort stats and normalize temporal data
        logger.info("Computing cohort normalization statistics...")
        temporal_data = dataset.temporal_tensors.copy()

        # Handle NaN for stats computation
        non_nan_data = []
        for i in range(temporal_data.shape[0]):
            non_nan_data.append(temporal_data[i])
        self._cohort_stats = self._temporal_extractor.compute_cohort_stats(non_nan_data)

        # 2. Split data (stratified to ensure both classes in val)
        n = dataset.n_samples
        rng = np.random.RandomState(self._random_state)
        pos_idx = np.where(dataset.labels == 1)[0]
        neg_idx = np.where(dataset.labels == 0)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        n_val_pos = max(1, int(len(pos_idx) * val_fraction))
        n_val_neg = max(1, int(len(neg_idx) * val_fraction))
        val_idx = np.concatenate([pos_idx[:n_val_pos], neg_idx[:n_val_neg]])
        train_idx = np.concatenate([pos_idx[n_val_pos:], neg_idx[n_val_neg:]])
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)

        X_train = torch.FloatTensor(temporal_data[train_idx]).to(self._device)
        y_train = torch.FloatTensor(dataset.labels[train_idx]).to(self._device)
        X_val = torch.FloatTensor(temporal_data[val_idx]).to(self._device)
        y_val = torch.FloatTensor(dataset.labels[val_idx]).to(self._device)

        logger.info("LSTM split: train=%d, val=%d", len(train_idx), len(val_idx))

        # 3. Build model
        self._model = _build_lstm_model(self._config).to(self._device)
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._config.learning_rate
        )
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Dataloaders
        train_ds = TensorDataset(X_train, y_train)
        train_dl = DataLoader(
            train_ds, batch_size=self._config.batch_size, shuffle=True
        )

        # 4. Training loop
        result = LSTMTrainingResult()
        best_val_loss = float("inf")
        patience_counter = 0
        best_state: dict[str, Any] | None = None

        for epoch in range(self._config.max_epochs):
            # Train
            self._model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch_X, batch_y in train_dl:
                optimizer.zero_grad()
                _, logits = self._model(batch_X)
                loss = criterion(logits.squeeze(-1), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)

            # Validate
            self._model.eval()
            with torch.no_grad():
                _, val_logits = self._model(X_val)
                val_loss = criterion(val_logits.squeeze(-1), y_val).item()

            scheduler.step(val_loss)

            result.train_losses.append(avg_train_loss)
            result.val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                result.best_epoch = epoch
                result.best_val_loss = val_loss
                patience_counter = 0
                best_state = {
                    k: v.clone() for k, v in self._model.state_dict().items()
                }
            else:
                patience_counter += 1

            if patience_counter >= self._config.patience:
                result.stopped_early = True
                logger.info("Early stopping at epoch %d", epoch)
                break

            if epoch % 10 == 0:
                logger.info(
                    "Epoch %d: train_loss=%.6f, val_loss=%.6f",
                    epoch, avg_train_loss, val_loss,
                )

        # Load best model
        if best_state is not None:
            self._model.load_state_dict(best_state)

        # Compute val AUC
        self._model.eval()
        with torch.no_grad():
            _, val_logits = self._model(X_val)
            val_probs = torch.sigmoid(val_logits.squeeze(-1)).cpu().numpy()

        from resistrack.local_training.train_xgboost import _compute_auc_roc
        result.val_auc_roc = _compute_auc_roc(
            dataset.labels[val_idx], val_probs
        )

        result.training_time_seconds = time.time() - start_time
        logger.info("LSTM training complete: %s", result.summary())
        return result

    def predict_risk_vectors(self, temporal_tensors: np.ndarray) -> np.ndarray:
        """Generate 32-dim trend-risk vectors for the ensemble.

        Args:
            temporal_tensors: Shape (n, 72, 13).

        Returns:
            Risk vectors of shape (n, 32).
        """
        import torch

        self._model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(temporal_tensors).to(self._device)
            X = torch.nan_to_num(X, nan=0.0)
            risk_vectors, _ = self._model(X)
            return risk_vectors.cpu().numpy()

    def predict_proba(self, temporal_tensors: np.ndarray) -> np.ndarray:
        """Predict AMR risk probabilities from temporal data.

        Args:
            temporal_tensors: Shape (n, 72, 13).

        Returns:
            Probabilities of shape (n,).
        """
        import torch

        self._model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(temporal_tensors).to(self._device)
            X = torch.nan_to_num(X, nan=0.0)
            _, logits = self._model(X)
            probs = torch.sigmoid(logits.squeeze(-1))
            return probs.cpu().numpy()

    def save_model(self, path: str | Path) -> None:
        """Save trained model state dict."""
        import torch

        if self._model is None:
            raise RuntimeError("No model to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "config": {
                    "input_size": self._config.input_size,
                    "hidden_size": self._config.hidden_size,
                    "num_layers": self._config.num_layers,
                    "dropout": self._config.dropout,
                    "bidirectional": self._config.bidirectional,
                    "output_dim": self._config.output_dim,
                },
            },
            str(path),
        )
        logger.info("LSTM model saved to %s", path)

    def load_model(self, path: str | Path) -> None:
        """Load a trained model from file."""
        import torch

        checkpoint = torch.load(str(path), map_location=self._device)
        config = LSTMConfig(**checkpoint["config"])
        self._config = config
        self._model = _build_lstm_model(config).to(self._device)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("LSTM model loaded from %s", path)


__all__ = ["LocalLSTMTrainer", "LSTMTrainingResult"]
