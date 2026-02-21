#!/usr/bin/env python3
"""
ResisTrack Demo Script.

This script demonstrates the core functionality of the ResisTrack backend
logic using a mock predictor. It simulates:
1. A SageMaker inference endpoint receiving patient features.
2. A CDS Hooks service generating clinical decision support cards.

Usage:
    python demo.py
"""

import sys
import os
import json
import time
from dataclasses import dataclass

# Ensure src is in python path
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from resistrack.inference.endpoint import SageMakerEndpoint, EndpointConfig
    from resistrack.cds.hooks import CDSHooksService, HookConfig, HookType, CDSHookRequest
    from resistrack.common.schemas import AMRPredictionOutput, AntibioticClassRisk, SHAPFeature
    from resistrack.common.constants import RiskTier
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you have installed dependencies via ./setup.sh")
    sys.exit(1)

# Mock classes to simulate the ML pipeline without loading heavy models
@dataclass
class MockResult:
    amr_output: AMRPredictionOutput

    def to_amr_output(self):
        return self.amr_output

class MockPredictor:
    def predict(self, features):
        # Generate deterministic mock data based on features
        # Simple logic: higher WBC count -> higher risk
        wbc = features.get("wbc_count", 10.0)
        score_val = int((wbc * 3) % 100)

        if score_val > 75:
            tier = RiskTier.CRITICAL
        elif score_val > 50:
            tier = RiskTier.HIGH
        elif score_val > 25:
            tier = RiskTier.MEDIUM
        else:
            tier = RiskTier.LOW

        output = AMRPredictionOutput(
            patient_token="demo_patient_123",
            amr_risk_score=score_val,
            risk_tier=tier,
            confidence_score=0.85,
            data_completeness_score=0.9,
            data_quality_flag=True,
            antibiotic_class_risk=AntibioticClassRisk(
                penicillins=0.2,
                cephalosporins=0.3,
                carbapenems=0.8 if tier == RiskTier.CRITICAL else 0.1,
                fluoroquinolones=0.1,
                aminoglycosides=0.1
            ),
            shap_top_features=[
                SHAPFeature(
                    name="wbc_count",
                    value=0.5,
                    direction="positive",
                    human_readable="White Blood Cell Count"
                ),
                SHAPFeature(
                    name="prior_abx",
                    value=0.3,
                    direction="positive",
                    human_readable="Prior Antibiotic Exposure"
                )
            ],
            recommended_action=f"Review patient (Risk: {tier})",
            model_version="demo-1.0"
        )
        return MockResult(output)

def run_demo():
    print("========================================")
    print("   ResisTrack Core Logic Demo")
    print("========================================")

    # 1. Setup Mock Predictor
    print("\n[1] Initializing Mock Predictor...")
    predictor = MockPredictor()

    # 2. SageMaker Endpoint Simulation
    print("\n[2] Initializing SageMaker Endpoint Simulation...")
    config = EndpointConfig(
        model_name="resistrack-demo",
        min_instances=1,
        max_instances=5
    )
    endpoint = SageMakerEndpoint(config=config, predictor=predictor)

    # Simulate a request
    features = {
        "wbc_count": 28.0,
        "age": 65.0,
        "prior_abx_exposure": 1.0
    }
    print(f"    Sending prediction request with features: {features}")

    try:
        start_t = time.time()
        result = endpoint.predict(features)
        latency = (time.time() - start_t) * 1000

        pred = result.prediction
        print(f"    -> Prediction: {pred.risk_tier} Risk (Score: {pred.amr_risk_score})")
        print(f"    -> Confidence: {pred.confidence_score*100:.1f}%")
        print(f"    -> Latency: {latency:.2f}ms (Endpoint reported: {result.latency_ms:.2f}ms)")
        print(f"    -> Cached: {result.cached_result}")
    except Exception as e:
        print(f"    Error: {e}")

    # 3. CDS Hooks Service Simulation
    print("\n[3] Initializing CDS Hooks Service...")
    hooks_service = CDSHooksService(predictor=predictor) # Using same predictor for demo

    hook_req = CDSHookRequest(
        hook_type=HookType.ORDER_SIGN,
        patient_token="pt-12345",
        encounter_id="enc-67890",
        context=features # Pass features as context
    )

    print(f"    Processing CDS Hook: {hook_req.hook_type}")
    try:
        hook_resp = hooks_service.process_hook(hook_req)
        print(f"    -> Cards Returned: {len(hook_resp.cards)}")
        for i, card in enumerate(hook_resp.cards):
            print(f"       Card {i+1}: [{card.indicator.upper()}] {card.summary}")
            print(f"       Detail: {card.detail}")
            print(f"       Suggestions: {card.recommendations}")
    except Exception as e:
        print(f"    Error: {e}")

    print("\n========================================")
    print("Demo complete.")

if __name__ == "__main__":
    run_demo()
