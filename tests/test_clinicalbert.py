"""Tests for M2.3 ClinicalBERT NLP module."""

from __future__ import annotations

import numpy as np
import pytest

from resistrack.ml.clinicalbert import (
    AMR_EMBEDDING_DIM,
    MAX_NOTES,
    MAX_SEQ_LENGTH,
    ClinicalBERTEmbedder,
    ClinicalBERTOutput,
    ClinicalNoteProcessor,
    NoteInput,
)


# ── Fixtures ──


def _make_note(
    text: str = "Patient presents with fever and elevated WBC",
    hours: float = 24.0,
    note_type: str = "progress_note",
) -> NoteInput:
    return NoteInput(text=text, timestamp_hours=hours, note_type=note_type)


def _make_notes(count: int = 3) -> list[NoteInput]:
    return [
        _make_note(f"Note {i} with clinical content about infection", hours=float(i * 12))
        for i in range(count)
    ]


# ── NoteInput tests ──


class TestNoteInput:
    def test_creation(self) -> None:
        note = _make_note()
        assert note.text == "Patient presents with fever and elevated WBC"
        assert note.timestamp_hours == 24.0
        assert note.note_type == "progress_note"

    def test_frozen(self) -> None:
        note = _make_note()
        with pytest.raises(AttributeError):
            note.text = "changed"  # type: ignore[misc]


# ── ClinicalNoteProcessor tests ──


class TestClinicalNoteProcessor:
    def setup_method(self) -> None:
        self.processor = ClinicalNoteProcessor()

    def test_select_notes_within_window(self) -> None:
        notes = [
            _make_note(hours=10.0),
            _make_note(hours=50.0),
            _make_note(hours=100.0),  # outside 72h
        ]
        selected = self.processor.select_notes(notes, window_hours=72.0)
        assert len(selected) == 2

    def test_select_notes_max_limit(self) -> None:
        notes = _make_notes(count=10)
        selected = self.processor.select_notes(notes)
        assert len(selected) == MAX_NOTES

    def test_select_notes_sorted_by_recency(self) -> None:
        notes = [
            _make_note(hours=10.0),
            _make_note(hours=50.0),
            _make_note(hours=30.0),
        ]
        selected = self.processor.select_notes(notes)
        assert selected[0].timestamp_hours == 50.0
        assert selected[1].timestamp_hours == 30.0

    def test_select_notes_empty(self) -> None:
        selected = self.processor.select_notes([])
        assert selected == []

    def test_truncate_tokens_short(self) -> None:
        tokens = ["word"] * 100
        result = self.processor.truncate_tokens(tokens)
        assert len(result) == 100

    def test_truncate_tokens_exact_limit(self) -> None:
        tokens = ["word"] * MAX_SEQ_LENGTH
        result = self.processor.truncate_tokens(tokens)
        assert len(result) == MAX_SEQ_LENGTH

    def test_truncate_tokens_over_limit(self) -> None:
        tokens = [f"word_{i}" for i in range(1000)]
        result = self.processor.truncate_tokens(tokens)
        assert len(result) == MAX_SEQ_LENGTH
        # First 128 tokens preserved
        assert result[0] == "word_0"
        assert result[127] == "word_127"
        # Last 384 tokens preserved
        assert result[-1] == "word_999"
        assert result[128] == "word_616"

    def test_tokenize_simple(self) -> None:
        tokens = self.processor.tokenize_simple("hello world test")
        assert tokens == ["hello", "world", "test"]

    def test_prepare_input_valid(self) -> None:
        notes = _make_notes(2)
        result = self.processor.prepare_input(notes)
        assert result["has_notes"] is True
        assert result["note_count"] == 2
        assert len(result["input_ids"]) > 0
        assert len(result["attention_mask"]) > 0
        assert all(m == 1 for m in result["attention_mask"])

    def test_prepare_input_empty(self) -> None:
        result = self.processor.prepare_input([])
        assert result["has_notes"] is False
        assert result["note_count"] == 0
        assert result["input_ids"] == []

    def test_prepare_input_max_length(self) -> None:
        long_text = " ".join(["word"] * 2000)
        notes = [_make_note(text=long_text, hours=1.0)]
        result = self.processor.prepare_input(notes)
        assert len(result["input_ids"]) <= MAX_SEQ_LENGTH


# ── ClinicalBERTEmbedder tests ──


class TestClinicalBERTEmbedder:
    def setup_method(self) -> None:
        self.embedder = ClinicalBERTEmbedder(use_model=False)

    def test_embedding_dimension(self) -> None:
        assert self.embedder.embedding_dim == AMR_EMBEDDING_DIM

    def test_embed_with_notes(self) -> None:
        notes = _make_notes(2)
        output = self.embedder.embed(notes)
        assert isinstance(output, ClinicalBERTOutput)
        assert output.amr_embedding.shape == (AMR_EMBEDDING_DIM,)
        assert output.has_notes is True
        assert output.note_count == 2
        assert output.avg_note_length > 0

    def test_embed_no_notes(self) -> None:
        output = self.embedder.embed([])
        assert output.has_notes is False
        assert output.note_count == 0
        assert np.all(output.amr_embedding == 0)

    def test_embed_l2_normalized(self) -> None:
        notes = _make_notes(2)
        output = self.embedder.embed(notes)
        norm = float(np.linalg.norm(output.amr_embedding))
        assert abs(norm - 1.0) < 1e-5

    def test_embed_deterministic(self) -> None:
        notes = _make_notes(2)
        out1 = self.embedder.embed(notes)
        out2 = self.embedder.embed(notes)
        np.testing.assert_array_equal(out1.amr_embedding, out2.amr_embedding)

    def test_embed_batch(self) -> None:
        batch = [_make_notes(2), _make_notes(1), []]
        outputs = self.embedder.embed_batch(batch)
        assert len(outputs) == 3
        assert outputs[0].has_notes is True
        assert outputs[2].has_notes is False

    def test_sagemaker_endpoint_raises(self) -> None:
        embedder = ClinicalBERTEmbedder(use_model=True)
        with pytest.raises(NotImplementedError):
            embedder.embed(_make_notes(1))

    def test_custom_embedding_dim(self) -> None:
        embedder = ClinicalBERTEmbedder(embedding_dim=64, use_model=False)
        output = embedder.embed(_make_notes(1))
        assert output.amr_embedding.shape == (64,)

    def test_float32_dtype(self) -> None:
        output = self.embedder.embed(_make_notes(1))
        assert output.amr_embedding.dtype == np.float32
