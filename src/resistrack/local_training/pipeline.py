"""End-to-end local training pipeline for ResisTrack.

Orchestrates the full Phase 2 training workflow:
  1. Load / generate data (MIMIC-IV or synthetic)
  2. Train XGBoost (M2.4)
  3. Train LSTM (M2.5)
  4. Fine-tune ClinicalBERT (M2.3)
  5. Train ensemble meta-learner + calibration (M2.6)
  6. Validate against acceptance criteria
  7. Save all artifacts

Run directly:
    python -m resistrack.local_training.pipeline [--synthetic] [--samples 2000]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from resistrack.common.constants import RANDOM_STATE
from resistrack.local_training.data_loader import MIMICDataLoader, TrainingDataset
from resistrack.local_training.train_ensemble import LocalEnsembleTrainer, LocalEnsembleResult

logger = logging.getLogger(__name__)


class LocalTrainingPipeline:
    """End-to-end local training pipeline.

    Usage:
        pipeline = LocalTrainingPipeline()

        # With synthetic data
        pipeline.run(synthetic=True, n_samples=2000)

        # With real MIMIC-IV data
        pipeline.run(mimic_dir="/data/mimic-iv")
    """

    def __init__(
        self,
        output_dir: str | Path = "artifacts/local_training",
        random_state: int = RANDOM_STATE,
        n_hpo_trials: int = 50,
        skip_clinicalbert: bool = False,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._random_state = random_state
        self._n_hpo_trials = n_hpo_trials
        self._skip_clinicalbert = skip_clinicalbert
        self._loader = MIMICDataLoader(random_state=random_state)
        self._trainer = LocalEnsembleTrainer(
            n_hpo_trials=n_hpo_trials,
            random_state=random_state,
            skip_clinicalbert=skip_clinicalbert,
        )
        self._result: LocalEnsembleResult | None = None

    def run(
        self,
        *,
        synthetic: bool = True,
        n_samples: int = 2000,
        positive_rate: float = 0.15,
        mimic_dir: str | Path | None = None,
        use_hpo: bool = True,
    ) -> LocalEnsembleResult:
        """Run the full training pipeline.

        Args:
            synthetic: If True, generate synthetic MIMIC-IV data.
            n_samples: Number of synthetic samples to generate.
            positive_rate: Fraction of AMR-resistant cases.
            mimic_dir: Path to real MIMIC-IV CSV directory.
            use_hpo: Whether to run Bayesian HPO for XGBoost.
        """
        start_time = time.time()

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

        logger.info("╔══════════════════════════════════════════════════════════╗")
        logger.info("║       ResisTrack Local Training Pipeline v2.0            ║")
        logger.info("║       Phase 2: AI/ML Inference Engine                    ║")
        logger.info("╚══════════════════════════════════════════════════════════╝")

        # ── Step 1: Data Loading ─────────────────────────────────────
        logger.info("\n📦 STEP 1: Data Loading")
        if synthetic or mimic_dir is None:
            dataset = self._loader.generate_synthetic(
                n_samples=n_samples,
                positive_rate=positive_rate,
            )
            data_source = f"synthetic ({n_samples} samples)"
        else:
            dataset = self._loader.load_from_csv(mimic_dir)
            data_source = f"MIMIC-IV ({mimic_dir})"

        logger.info("Data source: %s", data_source)
        logger.info("Dataset summary: %s", dataset.summary())

        # ── Step 2-5: Ensemble Training ──────────────────────────────
        logger.info("\n🧠 STEP 2-5: Model Training")
        self._result = self._trainer.train(dataset, use_hpo=use_hpo)

        # ── Step 6: Validation ───────────────────────────────────────
        logger.info("\n✅ STEP 6: Validation against Phase 2 Acceptance Criteria")
        self._print_acceptance_report(self._result)

        # ── Step 7: Save Artifacts ───────────────────────────────────
        logger.info("\n💾 STEP 7: Saving Artifacts")
        self._save_all(dataset, data_source)

        elapsed = time.time() - start_time
        logger.info("\n🏁 Pipeline complete in %.1f seconds", elapsed)

        return self._result

    def _print_acceptance_report(self, result: LocalEnsembleResult) -> None:
        """Print Phase 2 acceptance criteria check."""
        criteria = [
            ("AUC-ROC ≥ 0.82", result.ensemble_test_auc, 0.82, ">="),
            ("AUPRC ≥ 0.70", result.ensemble_test_auprc, 0.70, ">="),
            ("Sensitivity@80%Spec ≥ 0.80", result.sensitivity_at_80_spec, 0.80, ">="),
            ("FPR ≤ 0.20", result.false_positive_rate, 0.20, "<="),
            ("Brier Score ≤ 0.15", result.brier_score, 0.15, "<="),
        ]

        logger.info("Phase 2 Acceptance Criteria:")
        logger.info("-" * 55)
        all_passed = True
        for name, value, threshold, op in criteria:
            if op == ">=":
                passed = value >= threshold
            else:
                passed = value <= threshold
            status = "✅ PASS" if passed else "❌ FAIL"
            if not passed:
                all_passed = False
            logger.info("  %s  %-30s  %.4f (target: %s%.2f)", status, name, value, op, threshold)
        logger.info("-" * 55)
        if all_passed:
            logger.info("  🎉 ALL CRITERIA MET — Model ready for clinical validation (Phase 5)")
        else:
            logger.info("  ⚠️  Some criteria not met — Review model and data quality")

    def _save_all(self, dataset: TrainingDataset, data_source: str) -> None:
        """Save all artifacts to output directory."""
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        self._trainer.save_ensemble(self._output_dir / "models")

        # Save training report
        if self._result is not None:
            report = {
                "data_source": data_source,
                "dataset_summary": dataset.summary(),
                "results": self._result.summary(),
                "meets_acceptance_criteria": self._result.meets_acceptance_criteria,
                "random_state": self._random_state,
                "n_hpo_trials": self._n_hpo_trials,
            }
            report_path = self._output_dir / "training_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info("Training report saved to %s", report_path)

        logger.info("All artifacts saved to %s", self._output_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ResisTrack Local Training Pipeline — Phase 2 ML Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick run with synthetic data (no GPU needed)
  python -m resistrack.local_training.pipeline --synthetic --samples 500 --skip-bert --no-hpo

  # Full run with synthetic data
  python -m resistrack.local_training.pipeline --synthetic --samples 2000

  # Full run with MIMIC-IV data
  python -m resistrack.local_training.pipeline --mimic-dir /data/mimic-iv

  # Fast iteration (no HPO, no BERT, fewer samples)
  python -m resistrack.local_training.pipeline --synthetic --samples 300 --skip-bert --no-hpo
        """,
    )
    parser.add_argument(
        "--synthetic", action="store_true", default=True,
        help="Generate synthetic training data (default: True)",
    )
    parser.add_argument(
        "--mimic-dir", type=str, default=None,
        help="Path to MIMIC-IV CSV directory (overrides --synthetic)",
    )
    parser.add_argument(
        "--samples", type=int, default=2000,
        help="Number of synthetic samples (default: 2000)",
    )
    parser.add_argument(
        "--positive-rate", type=float, default=0.15,
        help="Fraction of AMR-resistant samples (default: 0.15)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="artifacts/local_training",
        help="Directory to save training artifacts",
    )
    parser.add_argument(
        "--hpo-trials", type=int, default=50,
        help="Number of Bayesian HPO trials for XGBoost (default: 50)",
    )
    parser.add_argument(
        "--no-hpo", action="store_true",
        help="Skip HPO and use default XGBoost params",
    )
    parser.add_argument(
        "--skip-bert", action="store_true",
        help="Skip ClinicalBERT fine-tuning (faster, no transformers needed)",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_STATE,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    pipeline = LocalTrainingPipeline(
        output_dir=args.output_dir,
        random_state=args.seed,
        n_hpo_trials=args.hpo_trials,
        skip_clinicalbert=args.skip_bert,
    )

    pipeline.run(
        synthetic=args.mimic_dir is None,
        n_samples=args.samples,
        positive_rate=args.positive_rate,
        mimic_dir=args.mimic_dir,
        use_hpo=not args.no_hpo,
    )


if __name__ == "__main__":
    main()
