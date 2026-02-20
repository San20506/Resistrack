"""Local training module for ResisTrack ML pipeline.

Provides MIMIC-IV data loading, real XGBoost/LSTM/ClinicalBERT training,
and an end-to-end local pipeline that mirrors the SageMaker Phase 2 roadmap.
"""

from resistrack.local_training.data_loader import MIMICDataLoader
from resistrack.local_training.train_xgboost import LocalXGBoostTrainer
from resistrack.local_training.train_lstm import LocalLSTMTrainer
from resistrack.local_training.train_clinicalbert import LocalClinicalBERTTrainer
from resistrack.local_training.train_ensemble import LocalEnsembleTrainer
from resistrack.local_training.pipeline import LocalTrainingPipeline

__all__ = [
    "MIMICDataLoader",
    "LocalXGBoostTrainer",
    "LocalLSTMTrainer",
    "LocalClinicalBERTTrainer",
    "LocalEnsembleTrainer",
    "LocalTrainingPipeline",
]
