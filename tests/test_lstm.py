"""Tests for M2.5 LSTM Model."""

import numpy as np
import pytest

from resistrack.ml.lstm_model import AMRLSTMModel, LSTMConfig, LSTMTrainingResult


@pytest.fixture
def config() -> LSTMConfig:
    return LSTMConfig()


@pytest.fixture
def model(config: LSTMConfig) -> AMRLSTMModel:
    return AMRLSTMModel(config)


@pytest.fixture
def sample_tensor() -> np.ndarray:
    """Sample temporal tensor: (batch=4, seq=72, features=13)."""
    rng = np.random.RandomState(42)
    return rng.randn(4, 72, 13)


@pytest.fixture
def single_tensor() -> np.ndarray:
    """Single patient tensor: (seq=72, features=13)."""
    rng = np.random.RandomState(42)
    return rng.randn(72, 13)


# ── LSTMConfig tests ──

class TestLSTMConfig:
    def test_default_values(self, config: LSTMConfig) -> None:
        assert config.input_size == 13
        assert config.hidden_size == 64
        assert config.num_layers == 2
        assert config.dropout == 0.3
        assert config.bidirectional is True
        assert config.output_dim == 32
        assert config.patience == 10
        assert config.random_state == 42

    def test_custom_config(self) -> None:
        cfg = LSTMConfig(hidden_size=128, num_layers=3)
        assert cfg.hidden_size == 128
        assert cfg.num_layers == 3

    def test_frozen(self, config: LSTMConfig) -> None:
        with pytest.raises(AttributeError):
            config.hidden_size = 256  # type: ignore[misc]


# ── Model initialization tests ──

class TestAMRLSTMModelInit:
    def test_default_init(self, model: AMRLSTMModel) -> None:
        assert not model.is_trained
        assert model.config.input_size == 13

    def test_weights_initialized(self, model: AMRLSTMModel) -> None:
        assert "lstm_weight" in model._weights
        assert "output_weight" in model._weights
        assert "output_bias" in model._weights

    def test_custom_config(self) -> None:
        cfg = LSTMConfig(hidden_size=128)
        m = AMRLSTMModel(cfg)
        assert m.config.hidden_size == 128


# ── Prediction tests ──

class TestPrediction:
    def test_batch_predict_shape(self, model: AMRLSTMModel, sample_tensor: np.ndarray) -> None:
        output = model.predict(sample_tensor)
        assert output.shape == (4, 32)

    def test_single_predict_shape(self, model: AMRLSTMModel, single_tensor: np.ndarray) -> None:
        output = model.predict(single_tensor)
        assert output.shape == (32,)

    def test_output_l2_normalized(self, model: AMRLSTMModel, sample_tensor: np.ndarray) -> None:
        output = model.predict(sample_tensor)
        norms = np.linalg.norm(output, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_single_output_l2_normalized(self, model: AMRLSTMModel, single_tensor: np.ndarray) -> None:
        output = model.predict(single_tensor)
        norm = np.linalg.norm(output)
        assert abs(norm - 1.0) < 1e-6

    def test_deterministic_output(self, model: AMRLSTMModel, sample_tensor: np.ndarray) -> None:
        out1 = model.predict(sample_tensor)
        out2 = model.predict(sample_tensor)
        np.testing.assert_array_equal(out1, out2)

    def test_different_inputs_different_outputs(self, model: AMRLSTMModel) -> None:
        rng = np.random.RandomState(42)
        t1 = rng.randn(2, 72, 13)
        t2 = rng.randn(2, 72, 13) * 5
        out1 = model.predict(t1)
        out2 = model.predict(t2)
        assert not np.allclose(out1, out2)


# ── Training tests ──

class TestTraining:
    def test_train_returns_result(self, model: AMRLSTMModel) -> None:
        rng = np.random.RandomState(42)
        train = rng.randn(20, 72, 13)
        labels = rng.randint(0, 2, 20).astype(float)
        val = rng.randn(5, 72, 13)
        val_labels = rng.randint(0, 2, 5).astype(float)

        result = model.train(train, labels, val, val_labels)
        assert isinstance(result, LSTMTrainingResult)
        assert len(result.train_losses) > 0
        assert len(result.val_losses) > 0

    def test_marks_trained(self, model: AMRLSTMModel) -> None:
        rng = np.random.RandomState(42)
        model.train(rng.randn(10, 72, 13), rng.rand(10), rng.randn(5, 72, 13), rng.rand(5))
        assert model.is_trained

    def test_early_stopping(self) -> None:
        cfg = LSTMConfig(patience=5, max_epochs=200)
        model = AMRLSTMModel(cfg)
        rng = np.random.RandomState(42)
        result = model.train(rng.randn(10, 72, 13), rng.rand(10), rng.randn(5, 72, 13), rng.rand(5))
        assert result.stopped_early or len(result.train_losses) <= 200

    def test_losses_are_positive(self, model: AMRLSTMModel) -> None:
        rng = np.random.RandomState(42)
        result = model.train(rng.randn(10, 72, 13), rng.rand(10), rng.randn(5, 72, 13), rng.rand(5))
        assert all(loss > 0 for loss in result.train_losses)
        assert all(loss > 0 for loss in result.val_losses)


# ── Serialization tests ──

class TestSerialization:
    def test_get_state(self, model: AMRLSTMModel) -> None:
        state = model.get_state()
        assert "config" in state
        assert "weights" in state
        assert state["config"]["input_size"] == 13

    def test_from_state_roundtrip(self, model: AMRLSTMModel, sample_tensor: np.ndarray) -> None:
        out_before = model.predict(sample_tensor)
        state = model.get_state()
        restored = AMRLSTMModel.from_state(state)
        out_after = restored.predict(sample_tensor)
        np.testing.assert_allclose(out_before, out_after, atol=1e-10)

    def test_from_state_preserves_trained(self, model: AMRLSTMModel) -> None:
        rng = np.random.RandomState(42)
        model.train(rng.randn(10, 72, 13), rng.rand(10), rng.randn(5, 72, 13), rng.rand(5))
        state = model.get_state()
        restored = AMRLSTMModel.from_state(state)
        assert restored.is_trained
