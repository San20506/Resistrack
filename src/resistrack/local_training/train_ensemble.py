"""M2.6 — Local ensemble trainer (meta-learner + Platt calibration).

Combines the three sub-model predictions (XGBoost + LSTM + ClinicalBERT)
via a learned meta-learner and applies Platt scaling calibration.
Reuses the existing EnsemblePredictor from resistrack.ml.ensemble to
ensure full compatibility with the SageMaker deployment path.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from resistrack.common.constants import RANDOM_STATE
from resistrack.local_training.data_loader import TrainingDataset
from resistrack.local_training.train_xgboost import (
    LocalXGBoostTrainer,
    _compute_auc_roc,
    _compute_auprc,
    _sensitivity_at_specificity,
)
from resistrack.local_training.train_lstm import LocalLSTMTrainer
from resistrack.local_training.train_clinicalbert import LocalClinicalBERTTrainer
from resistrack.ml.ensemble import (
    EnsemblePredictor,
    EnsembleTrainingResult,
    SubModelOutput,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class LocalEnsembleResult:
    """Combined result from ensemble training."""

    xgboost_auc: float
    lstm_auc: float
    clinicalbert_auc: float
    ensemble_val_auc: float
    ensemble_test_auc: float
    ensemble_test_auprc: float
    sensitivity_at_80_spec: float
    false_positive_rate: float
    brier_score: float
    meta_weights: list[float]
    calibration_platt_a: float
    calibration_platt_b: float
    training_time_seconds: float

    def summary(self) -> dict[str, Any]:
        return {
            "Sub-model AUCs": {
                "XGBoost": round(self.xgboost_auc, 4),
                "LSTM": round(self.lstm_auc, 4),
                "ClinicalBERT": round(self.clinicalbert_auc, 4),
            },
            "Ensemble": {
                "val_auc_roc": round(self.ensemble_val_auc, 4),
                "test_auc_roc": round(self.ensemble_test_auc, 4),
                "test_auprc": round(self.ensemble_test_auprc, 4),
                "sensitivity@80%spec": round(self.sensitivity_at_80_spec, 4),
                "FPR": round(self.false_positive_rate, 4),
                "brier_score": round(self.brier_score, 4),
            },
            "Meta-learner weights": [round(w, 4) for w in self.meta_weights],
            "Platt calibration": {
                "a": round(self.calibration_platt_a, 4),
                "b": round(self.calibration_platt_b, 4),
            },
            "training_time_seconds": round(self.training_time_seconds, 2),
        }

    @property
    def meets_acceptance_criteria(self) -> bool:
        """Check Phase 2 acceptance criteria:
        AUC-ROC ≥ 0.82, AUPRC ≥ 0.70, Sensitivity@80%Spec ≥ 0.80,
        FPR ≤ 0.20, Brier ≤ 0.15.
        """
        return (
            self.ensemble_test_auc >= 0.82
            and self.ensemble_test_auprc >= 0.70
            and self.sensitivity_at_80_spec >= 0.80
            and self.false_positive_rate <= 0.20
            and self.brier_score <= 0.15
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class LocalEnsembleTrainer:
    """Trains the full 3-model ensemble locally.

    Usage:
        trainer = LocalEnsembleTrainer()
        result = trainer.train(dataset)
        trainer.save_ensemble("artifacts/ensemble/")
    """

    def __init__(
        self,
        n_hpo_trials: int = 50,
        random_state: int = RANDOM_STATE,
        skip_clinicalbert: bool = False,
    ) -> None:
        self._n_hpo_trials = n_hpo_trials
        self._random_state = random_state
        self._skip_clinicalbert = skip_clinicalbert

        self._xgb_trainer = LocalXGBoostTrainer(
            n_hpo_trials=n_hpo_trials, random_state=random_state
        )
        self._lstm_trainer = LocalLSTMTrainer(random_state=random_state)
        self._bert_trainer: LocalClinicalBERTTrainer | None = None
        if not skip_clinicalbert:
            self._bert_trainer = LocalClinicalBERTTrainer(random_state=random_state)

        self._ensemble: EnsemblePredictor | None = None

    @property
    def ensemble(self) -> EnsemblePredictor:
        if self._ensemble is None:
            raise RuntimeError("Ensemble not trained.")
        return self._ensemble

    def train(
        self,
        dataset: TrainingDataset,
        use_hpo: bool = True,
    ) -> LocalEnsembleResult:
        """Train all three sub-models and the ensemble meta-learner.

        Args:
            dataset: Complete TrainingDataset.
            use_hpo: Whether to use Bayesian HPO for XGBoost.
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("Starting ResisTrack Ensemble Training Pipeline")
        logger.info("=" * 60)

        # ── Step 1: XGBoost ──────────────────────────────────────────
        logger.info("\n[1/4] Training XGBoost model...")
        xgb_result = self._xgb_trainer.train(dataset, use_hpo=use_hpo)
        xgb_proba = self._xgb_trainer.predict_proba(dataset.tabular_features)
        logger.info("XGBoost val AUC: %.4f", xgb_result.val_auc_roc)

        # ── Step 2: LSTM ─────────────────────────────────────────────
        logger.info("\n[2/4] Training LSTM model...")
        lstm_result = self._lstm_trainer.train(dataset)
        lstm_proba = self._lstm_trainer.predict_proba(dataset.temporal_tensors)
        logger.info("LSTM val AUC: %.4f", lstm_result.val_auc_roc)

        # ── Step 3: ClinicalBERT ─────────────────────────────────────
        if self._bert_trainer is not None:
            logger.info("\n[3/4] Fine-tuning ClinicalBERT...")
            bert_result = self._bert_trainer.train(dataset)
            bert_proba = self._bert_trainer.predict_proba(dataset.clinical_notes)
            bert_auc = bert_result.val_auc_roc
            logger.info("ClinicalBERT val AUC: %.4f", bert_auc)
        else:
            logger.info("\n[3/4] Skipping ClinicalBERT (skip_clinicalbert=True)")
            # Generate synthetic NLP scores
            rng = np.random.RandomState(self._random_state)
            bert_proba = rng.beta(2, 5, size=dataset.n_samples).astype(np.float32)
            bert_auc = 0.0

        # ── Step 4: Ensemble meta-learner ────────────────────────────
        logger.info("\n[4/4] Training ensemble meta-learner...")

        sub_outputs = [
            SubModelOutput(model_name="xgboost", scores=xgb_proba),
            SubModelOutput(model_name="lstm", scores=lstm_proba),
            SubModelOutput(model_name="clinicalbert", scores=bert_proba),
        ]

        # Get feature importances for SHAP attribution
        feature_importances = [
            xgb_result.feature_importances,
            np.ones(len(dataset.feature_names), dtype=np.float32) / len(dataset.feature_names),
            np.ones(len(dataset.feature_names), dtype=np.float32) / len(dataset.feature_names),
        ]

        self._ensemble = EnsemblePredictor(model_version="2.0.0-local")
        ensemble_result = self._ensemble.train(
            sub_model_outputs=sub_outputs,
            labels=dataset.labels.astype(np.float64),
            feature_importances=feature_importances,
            val_fraction=0.3,
        )

        logger.info("Ensemble val AUC: %.4f", ensemble_result.val_auc_roc)
        logger.info("Calibration Brier: %.4f", ensemble_result.calibration.brier_score)

        # ── Evaluate on test split (stratified to ensure both classes) ─
        rng = np.random.RandomState(self._random_state)
        pos_idx = np.where(dataset.labels == 1)[0]
        neg_idx = np.where(dataset.labels == 0)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        n_test_pos = max(1, int(len(pos_idx) * 0.15))
        n_test_neg = max(1, int(len(neg_idx) * 0.15))
        test_idx = np.concatenate([pos_idx[:n_test_pos], neg_idx[:n_test_neg]])

        test_scores = np.column_stack([
            xgb_proba[test_idx],
            lstm_proba[test_idx],
            bert_proba[test_idx],
        ])

        raw_test = self._ensemble._meta.predict_raw(test_scores)
        cal_test = self._ensemble._calibrator.calibrate(raw_test)

        test_labels = dataset.labels[test_idx]
        test_auc = _compute_auc_roc(test_labels, cal_test)
        test_auprc = _compute_auprc(test_labels, cal_test)
        sens, fpr = _sensitivity_at_specificity(test_labels, cal_test)
        brier = float(np.mean((cal_test - test_labels) ** 2))

        elapsed = time.time() - start_time

        result = LocalEnsembleResult(
            xgboost_auc=xgb_result.val_auc_roc,
            lstm_auc=lstm_result.val_auc_roc,
            clinicalbert_auc=bert_auc,
            ensemble_val_auc=ensemble_result.val_auc_roc,
            ensemble_test_auc=test_auc,
            ensemble_test_auprc=test_auprc,
            sensitivity_at_80_spec=sens,
            false_positive_rate=fpr,
            brier_score=brier,
            meta_weights=ensemble_result.meta_weights.tolist(),
            calibration_platt_a=ensemble_result.calibration.platt_a,
            calibration_platt_b=ensemble_result.calibration.platt_b,
            training_time_seconds=elapsed,
        )

        logger.info("\n" + "=" * 60)
        logger.info("ENSEMBLE TRAINING COMPLETE")
        logger.info("=" * 60)
        for key, val in result.summary().items():
            logger.info("  %s: %s", key, val)

        if result.meets_acceptance_criteria:
            logger.info("✅ All Phase 2 acceptance criteria MET")
        else:
            logger.warning("⚠️  Some Phase 2 acceptance criteria NOT met")

        return result

    def save_ensemble(self, path: str | Path) -> None:
        """Save all trained models and ensemble state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save XGBoost
        self._xgb_trainer.save_model(path / "xgboost_model.json")

        # Save LSTM
        self._lstm_trainer.save_model(path / "lstm_model.pt")

        # Save ClinicalBERT
        if self._bert_trainer is not None:
            self._bert_trainer.save_model(path / "clinicalbert")

        # Save ensemble state
        if self._ensemble is not None:
            state = self._ensemble.get_state()
            with open(path / "ensemble_state.json", "w") as f:
                json.dump(state, f, indent=2)

        logger.info("Full ensemble saved to %s", path)

    def load_ensemble(self, path: str | Path) -> None:
        """Load all trained models and ensemble state."""
        path = Path(path)

        # Load XGBoost
        self._xgb_trainer.load_model(path / "xgboost_model.json")

        # Load LSTM
        self._lstm_trainer.load_model(path / "lstm_model.pt")

        # Load ClinicalBERT
        if self._bert_trainer is not None and (path / "clinicalbert").exists():
            self._bert_trainer.load_model(path / "clinicalbert")

        # Load ensemble state
        with open(path / "ensemble_state.json") as f:
            state = json.load(f)
        self._ensemble = EnsemblePredictor()
        self._ensemble.load_state(state)

        logger.info("Full ensemble loaded from %s", path)


__all__ = ["LocalEnsembleTrainer", "LocalEnsembleResult"]
