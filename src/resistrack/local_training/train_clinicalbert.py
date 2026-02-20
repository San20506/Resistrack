"""M2.3 — Local ClinicalBERT fine-tuning trainer.

Fine-tunes Bio_ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT) from HuggingFace
for AMR risk signal extraction from clinical notes:
  - Loads the pre-trained model locally (no external API calls with PHI)
  - Fine-tunes a classification head: 768-dim → 32-dim risk vector
  - Note selection: last 3 notes, max 512 tokens
  - Truncation: first 128 + last 384 tokens
  - Outputs 32-dim embeddings for ensemble integration

The trained model is saved as a HuggingFace-compatible checkpoint.
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
from resistrack.ml.clinicalbert import (
    AMR_EMBEDDING_DIM,
    CLINICALBERT_CHECKPOINT,
    FIRST_TOKENS,
    LAST_TOKENS,
    MAX_NOTES,
    MAX_SEQ_LENGTH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ClinicalBERTTrainingResult:
    """Result from local ClinicalBERT fine-tuning."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    val_auc_roc: float = 0.0
    training_time_seconds: float = 0.0
    checkpoint_name: str = CLINICALBERT_CHECKPOINT

    def summary(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint_name,
            "best_epoch": self.best_epoch,
            "best_val_loss": round(self.best_val_loss, 6),
            "val_auc_roc": round(self.val_auc_roc, 4),
            "n_epochs": len(self.train_losses),
            "training_time_seconds": round(self.training_time_seconds, 2),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class LocalClinicalBERTTrainer:
    """Local ClinicalBERT fine-tuning trainer.

    Usage:
        trainer = LocalClinicalBERTTrainer()
        result = trainer.train(dataset)
        embeddings = trainer.extract_embeddings(notes)
        trainer.save_model("artifacts/clinicalbert/")
    """

    def __init__(
        self,
        checkpoint: str = CLINICALBERT_CHECKPOINT,
        embedding_dim: int = AMR_EMBEDDING_DIM,
        max_seq_length: int = MAX_SEQ_LENGTH,
        random_state: int = RANDOM_STATE,
        n_epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
    ) -> None:
        self._checkpoint = checkpoint
        self._embedding_dim = embedding_dim
        self._max_seq_length = max_seq_length
        self._random_state = random_state
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._lr = learning_rate
        self._model: Any = None
        self._tokenizer: Any = None
        self._head: Any = None
        self._device: str = "cpu"

    def _truncate_tokens(
        self, input_ids: list[int], attention_mask: list[int]
    ) -> tuple[list[int], list[int]]:
        """Apply first-128 + last-384 truncation strategy."""
        if len(input_ids) <= self._max_seq_length:
            return input_ids, attention_mask

        first_ids = input_ids[:FIRST_TOKENS]
        last_ids = input_ids[-LAST_TOKENS:]
        first_mask = attention_mask[:FIRST_TOKENS]
        last_mask = attention_mask[-LAST_TOKENS:]

        return first_ids + last_ids, first_mask + last_mask

    def _prepare_notes(
        self, notes: list[str]
    ) -> tuple[list[int], list[int]]:
        """Tokenize and truncate a list of clinical notes."""
        # Select last MAX_NOTES notes
        selected = notes[-MAX_NOTES:] if len(notes) > MAX_NOTES else notes

        # Concatenate notes with separator
        combined_text = " [SEP] ".join(selected)

        # Tokenize
        encoded = self._tokenizer(
            combined_text,
            max_length=self._max_seq_length * 2,  # Over-tokenize then truncate
            truncation=False,
            padding=False,
            return_tensors=None,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Apply first-128 + last-384 truncation
        input_ids, attention_mask = self._truncate_tokens(input_ids, attention_mask)

        return input_ids, attention_mask

    def train(
        self,
        dataset: TrainingDataset,
        val_fraction: float = 0.15,
    ) -> ClinicalBERTTrainingResult:
        """Fine-tune ClinicalBERT on clinical notes for AMR risk classification.

        Args:
            dataset: TrainingDataset with clinical_notes and labels.
            val_fraction: Fraction for validation.
        """
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer

        start_time = time.time()

        # Set device
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("ClinicalBERT training on device: %s", self._device)

        torch.manual_seed(self._random_state)
        np.random.seed(self._random_state)

        # 1. Load pre-trained model and tokenizer
        logger.info("Loading %s...", self._checkpoint)
        self._tokenizer = AutoTokenizer.from_pretrained(self._checkpoint)
        bert_model = AutoModel.from_pretrained(self._checkpoint)
        bert_model = bert_model.to(self._device)

        # Classification head: 768 → 32 → 1
        class AMRClassificationHead(nn.Module):
            def __init__(self, hidden_size: int, embedding_dim: int) -> None:
                super().__init__()
                self.projection = nn.Linear(hidden_size, embedding_dim)
                self.dropout = nn.Dropout(0.1)
                self.classifier = nn.Linear(embedding_dim, 1)

            def forward(self, cls_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                embedding = self.projection(cls_output)
                embedding = torch.relu(embedding)
                embedding_norm = torch.nn.functional.normalize(embedding, p=2, dim=1)
                logit = self.classifier(self.dropout(embedding))
                return embedding_norm, logit

        hidden_size = bert_model.config.hidden_size
        self._head = AMRClassificationHead(hidden_size, self._embedding_dim).to(self._device)
        self._model = bert_model

        # 2. Tokenize all notes
        logger.info("Tokenizing %d patient note sets...", dataset.n_samples)
        all_input_ids: list[list[int]] = []
        all_attention_masks: list[list[int]] = []

        for notes in dataset.clinical_notes:
            ids, mask = self._prepare_notes(notes)
            all_input_ids.append(ids)
            all_attention_masks.append(mask)

        # Pad to same length
        max_len = min(
            max(len(ids) for ids in all_input_ids),
            self._max_seq_length,
        )
        padded_ids = np.zeros((dataset.n_samples, max_len), dtype=np.int64)
        padded_masks = np.zeros((dataset.n_samples, max_len), dtype=np.int64)

        for i, (ids, mask) in enumerate(zip(all_input_ids, all_attention_masks)):
            length = min(len(ids), max_len)
            padded_ids[i, :length] = ids[:length]
            padded_masks[i, :length] = mask[:length]

        # 3. Split
        rng = np.random.RandomState(self._random_state)
        indices = rng.permutation(dataset.n_samples)
        n_val = max(1, int(dataset.n_samples * val_fraction))
        train_idx = indices[:-n_val]
        val_idx = indices[-n_val:]

        # 4. Training
        optimizer = torch.optim.AdamW(
            list(bert_model.parameters()) + list(self._head.parameters()),
            lr=self._lr,
            weight_decay=0.01,
        )
        criterion = nn.BCEWithLogitsLoss()

        result = ClinicalBERTTrainingResult(checkpoint_name=self._checkpoint)
        best_val_loss = float("inf")
        best_head_state: dict[str, Any] | None = None

        for epoch in range(self._n_epochs):
            # Train
            bert_model.train()
            self._head.train()
            epoch_loss = 0.0
            n_batches = 0

            # Mini-batch training
            rng.shuffle(train_idx)
            for batch_start in range(0, len(train_idx), self._batch_size):
                batch_idx = train_idx[batch_start:batch_start + self._batch_size]

                input_ids = torch.LongTensor(padded_ids[batch_idx]).to(self._device)
                attn_mask = torch.LongTensor(padded_masks[batch_idx]).to(self._device)
                labels = torch.FloatTensor(dataset.labels[batch_idx]).to(self._device)

                optimizer.zero_grad()
                outputs = bert_model(input_ids=input_ids, attention_mask=attn_mask)
                cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token

                _, logits = self._head(cls_output)
                loss = criterion(logits.squeeze(-1), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(bert_model.parameters()) + list(self._head.parameters()),
                    1.0,
                )
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)

            # Validate
            bert_model.eval()
            self._head.eval()
            with torch.no_grad():
                val_input = torch.LongTensor(padded_ids[val_idx]).to(self._device)
                val_mask = torch.LongTensor(padded_masks[val_idx]).to(self._device)
                val_labels = torch.FloatTensor(dataset.labels[val_idx]).to(self._device)

                val_out = bert_model(input_ids=val_input, attention_mask=val_mask)
                val_cls = val_out.last_hidden_state[:, 0, :]
                _, val_logits = self._head(val_cls)
                val_loss = criterion(val_logits.squeeze(-1), val_labels).item()

            result.train_losses.append(avg_train_loss)
            result.val_losses.append(val_loss)

            logger.info(
                "Epoch %d/%d: train_loss=%.6f, val_loss=%.6f",
                epoch + 1, self._n_epochs, avg_train_loss, val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                result.best_epoch = epoch
                result.best_val_loss = val_loss
                best_head_state = {
                    k: v.clone() for k, v in self._head.state_dict().items()
                }

        # Load best head state
        if best_head_state is not None:
            self._head.load_state_dict(best_head_state)

        # Compute val AUC
        bert_model.eval()
        self._head.eval()
        with torch.no_grad():
            val_input = torch.LongTensor(padded_ids[val_idx]).to(self._device)
            val_mask = torch.LongTensor(padded_masks[val_idx]).to(self._device)
            val_out = bert_model(input_ids=val_input, attention_mask=val_mask)
            val_cls = val_out.last_hidden_state[:, 0, :]
            _, val_logits = self._head(val_cls)
            val_probs = torch.sigmoid(val_logits.squeeze(-1)).cpu().numpy()

        from resistrack.local_training.train_xgboost import _compute_auc_roc
        result.val_auc_roc = _compute_auc_roc(dataset.labels[val_idx], val_probs)

        result.training_time_seconds = time.time() - start_time
        logger.info("ClinicalBERT training complete: %s", result.summary())
        return result

    def extract_embeddings(
        self, clinical_notes: list[list[str]]
    ) -> np.ndarray:
        """Extract 32-dim AMR risk embeddings from clinical notes.

        Args:
            clinical_notes: List of note lists per patient.

        Returns:
            Embeddings of shape (n_patients, 32).
        """
        import torch

        if self._model is None or self._head is None:
            raise RuntimeError("Model not trained. Call train() first.")

        self._model.eval()
        self._head.eval()

        embeddings: list[np.ndarray] = []

        with torch.no_grad():
            for notes in clinical_notes:
                ids, mask = self._prepare_notes(notes)

                input_ids = torch.LongTensor([ids[:self._max_seq_length]]).to(self._device)
                attn_mask = torch.LongTensor([mask[:self._max_seq_length]]).to(self._device)

                outputs = self._model(input_ids=input_ids, attention_mask=attn_mask)
                cls_output = outputs.last_hidden_state[:, 0, :]
                embedding, _ = self._head(cls_output)
                embeddings.append(embedding.cpu().numpy()[0])

        return np.stack(embeddings, axis=0)

    def predict_proba(self, clinical_notes: list[list[str]]) -> np.ndarray:
        """Predict AMR risk probabilities from clinical notes.

        Args:
            clinical_notes: List of note lists per patient.

        Returns:
            Probabilities of shape (n_patients,).
        """
        import torch

        if self._model is None or self._head is None:
            raise RuntimeError("Model not trained. Call train() first.")

        self._model.eval()
        self._head.eval()

        probs: list[float] = []

        with torch.no_grad():
            for notes in clinical_notes:
                ids, mask = self._prepare_notes(notes)

                input_ids = torch.LongTensor([ids[:self._max_seq_length]]).to(self._device)
                attn_mask = torch.LongTensor([mask[:self._max_seq_length]]).to(self._device)

                outputs = self._model(input_ids=input_ids, attention_mask=attn_mask)
                cls_output = outputs.last_hidden_state[:, 0, :]
                _, logit = self._head(cls_output)
                prob = torch.sigmoid(logit).item()
                probs.append(prob)

        return np.array(probs, dtype=np.float32)

    def save_model(self, path: str | Path) -> None:
        """Save fine-tuned model and head."""
        import torch

        if self._model is None:
            raise RuntimeError("No model to save.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(str(path / "bert"))
        self._tokenizer.save_pretrained(str(path / "bert"))
        torch.save(self._head.state_dict(), str(path / "amr_head.pt"))
        logger.info("ClinicalBERT model saved to %s", path)

    def load_model(self, path: str | Path) -> None:
        """Load a fine-tuned model from directory."""
        import torch
        from transformers import AutoModel, AutoTokenizer
        import torch.nn as nn

        path = Path(path)
        self._tokenizer = AutoTokenizer.from_pretrained(str(path / "bert"))
        self._model = AutoModel.from_pretrained(str(path / "bert")).to(self._device)

        hidden_size = self._model.config.hidden_size

        class AMRClassificationHead(nn.Module):
            def __init__(self, hs: int, ed: int) -> None:
                super().__init__()
                self.projection = nn.Linear(hs, ed)
                self.dropout = nn.Dropout(0.1)
                self.classifier = nn.Linear(ed, 1)

            def forward(self, cls_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                embedding = self.projection(cls_output)
                embedding = torch.relu(embedding)
                embedding_norm = torch.nn.functional.normalize(embedding, p=2, dim=1)
                logit = self.classifier(self.dropout(embedding))
                return embedding_norm, logit

        self._head = AMRClassificationHead(hidden_size, self._embedding_dim).to(self._device)
        self._head.load_state_dict(torch.load(str(path / "amr_head.pt"), map_location=self._device))
        logger.info("ClinicalBERT model loaded from %s", path)


__all__ = ["LocalClinicalBERTTrainer", "ClinicalBERTTrainingResult"]
