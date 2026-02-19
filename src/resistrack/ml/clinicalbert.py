"""M2.3 ClinicalBERT NLP module for clinical note embeddings.

Uses Bio_ClinicalBERT to extract 32-dimensional AMR risk vectors
from clinical notes within a 72h window. Truncation strategy:
first 128 + last 384 tokens (max 512).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from resistrack.common.constants import RANDOM_STATE

# Bio_ClinicalBERT checkpoint
CLINICALBERT_CHECKPOINT: str = "emilyalsentzer/Bio_ClinicalBERT"
MAX_SEQ_LENGTH: int = 512
FIRST_TOKENS: int = 128
LAST_TOKENS: int = 384
AMR_EMBEDDING_DIM: int = 32
MAX_NOTES: int = 3
NOTE_WINDOW_HOURS: int = 72


@dataclasses.dataclass(frozen=True)
class NoteInput:
    """A single clinical note for processing."""

    text: str
    timestamp_hours: float
    note_type: str = "progress_note"


@dataclasses.dataclass(frozen=True)
class ClinicalBERTOutput:
    """Output from ClinicalBERT processing."""

    amr_embedding: np.ndarray  # shape: (AMR_EMBEDDING_DIM,)
    note_count: int
    avg_note_length: int
    has_notes: bool


class ClinicalNoteProcessor:
    """Processes clinical notes for ClinicalBERT input.

    Applies truncation strategy: keeps first 128 + last 384 tokens
    from each note to preserve both admission context and recent findings.
    """

    def __init__(self, max_seq_length: int = MAX_SEQ_LENGTH) -> None:
        self._max_seq_length = max_seq_length
        self._first_tokens = FIRST_TOKENS
        self._last_tokens = LAST_TOKENS

    def select_notes(
        self,
        notes: list[NoteInput],
        window_hours: float = NOTE_WINDOW_HOURS,
    ) -> list[NoteInput]:
        """Select the most recent notes within the time window.

        Returns up to MAX_NOTES notes sorted by recency.
        """
        in_window = [n for n in notes if 0 <= n.timestamp_hours <= window_hours]
        in_window.sort(key=lambda n: n.timestamp_hours, reverse=True)
        return in_window[:MAX_NOTES]

    def truncate_tokens(self, tokens: list[str]) -> list[str]:
        """Apply first-128 + last-384 truncation strategy."""
        if len(tokens) <= self._max_seq_length:
            return tokens
        first_part = tokens[: self._first_tokens]
        last_part = tokens[-self._last_tokens :]
        return first_part + last_part

    def tokenize_simple(self, text: str) -> list[str]:
        """Simple whitespace tokenizer (placeholder for real BPE tokenizer)."""
        return text.split()

    def prepare_input(self, notes: list[NoteInput]) -> dict[str, Any]:
        """Prepare notes for model input.

        Returns a dict with concatenated token IDs and attention mask.
        """
        selected = self.select_notes(notes)
        if not selected:
            return {
                "input_ids": [],
                "attention_mask": [],
                "note_count": 0,
                "has_notes": False,
            }

        all_tokens: list[str] = []
        for note in selected:
            tokens = self.tokenize_simple(note.text)
            truncated = self.truncate_tokens(tokens)
            all_tokens.extend(truncated)

        # Final truncation to max sequence length
        all_tokens = all_tokens[: self._max_seq_length]

        return {
            "input_ids": list(range(len(all_tokens))),
            "attention_mask": [1] * len(all_tokens),
            "note_count": len(selected),
            "has_notes": True,
        }


class ClinicalBERTEmbedder:
    """Extracts 32-dim AMR risk embeddings from clinical notes.

    In production, loads Bio_ClinicalBERT from SageMaker endpoint.
    This implementation provides the embedding pipeline with a
    deterministic fallback for testing without GPU/model weights.
    """

    def __init__(
        self,
        embedding_dim: int = AMR_EMBEDDING_DIM,
        use_model: bool = False,
    ) -> None:
        self._embedding_dim = embedding_dim
        self._use_model = use_model
        self._processor = ClinicalNoteProcessor()
        self._rng = np.random.RandomState(RANDOM_STATE)

    @property
    def embedding_dim(self) -> int:
        """Dimension of output embedding vector."""
        return self._embedding_dim

    def _generate_deterministic_embedding(
        self,
        prepared: dict[str, Any],
    ) -> np.ndarray:
        """Generate a deterministic embedding based on input characteristics.

        Used when model weights are not available (testing/development).
        """
        if not prepared["has_notes"]:
            return np.zeros(self._embedding_dim, dtype=np.float32)

        seed = len(prepared["input_ids"]) + prepared["note_count"]
        rng = np.random.RandomState(seed % (2**31))
        embedding = rng.randn(self._embedding_dim).astype(np.float32)
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def embed(self, notes: list[NoteInput]) -> ClinicalBERTOutput:
        """Extract AMR risk embedding from clinical notes."""
        prepared = self._processor.prepare_input(notes)

        if self._use_model:
            # Production: call SageMaker endpoint
            embedding = self._call_sagemaker_endpoint(prepared)
        else:
            embedding = self._generate_deterministic_embedding(prepared)

        selected = self._processor.select_notes(notes)
        avg_length = (
            int(np.mean([len(n.text) for n in selected])) if selected else 0
        )

        return ClinicalBERTOutput(
            amr_embedding=embedding,
            note_count=prepared["note_count"],
            avg_note_length=avg_length,
            has_notes=prepared["has_notes"],
        )

    def embed_batch(
        self,
        batch_notes: list[list[NoteInput]],
    ) -> list[ClinicalBERTOutput]:
        """Process a batch of patient note sets."""
        return [self.embed(notes) for notes in batch_notes]

    def _call_sagemaker_endpoint(
        self,
        prepared: dict[str, Any],
    ) -> np.ndarray:
        """Call SageMaker endpoint for inference (production only)."""
        raise NotImplementedError(
            "SageMaker endpoint call requires deployment. "
            "Use use_model=False for local testing."
        )


__all__ = [
    "CLINICALBERT_CHECKPOINT",
    "MAX_SEQ_LENGTH",
    "AMR_EMBEDDING_DIM",
    "NoteInput",
    "ClinicalBERTOutput",
    "ClinicalNoteProcessor",
    "ClinicalBERTEmbedder",
]
