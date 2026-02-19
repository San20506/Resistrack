import { useState, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import { RiskTier } from '../types';
import type { SHAPFeature, AMRPredictionOutput } from '../types';

/** Risk tier colors for timeline markers. */
const TIER_STYLES: Record<RiskTier, { dot: string; bg: string; text: string }> = {
  [RiskTier.LOW]: { dot: 'bg-green-500', bg: 'bg-green-50', text: 'text-green-700' },
  [RiskTier.MEDIUM]: { dot: 'bg-yellow-500', bg: 'bg-yellow-50', text: 'text-yellow-700' },
  [RiskTier.HIGH]: { dot: 'bg-orange-500', bg: 'bg-orange-50', text: 'text-orange-700' },
  [RiskTier.CRITICAL]: { dot: 'bg-red-500', bg: 'bg-red-50', text: 'text-red-700' },
};

function scoreToTier(score: number): RiskTier {
  if (score < 25) return RiskTier.LOW;
  if (score < 50) return RiskTier.MEDIUM;
  if (score < 75) return RiskTier.HIGH;
  return RiskTier.CRITICAL;
}

interface TimelineEntry {
  timestamp: string;
  prediction: AMRPredictionOutput;
}

/** Generate mock patient timeline data. */
function generateMockTimeline(patientToken: string): TimelineEntry[] {
  const now = Date.now();
  const entries: TimelineEntry[] = [];
  const baseScore = 40 + Math.round(Math.random() * 30);

  for (let i = 11; i >= 0; i--) {
    const score = Math.max(0, Math.min(100, baseScore + Math.round((Math.random() - 0.3) * 20 * (1 - i / 12))));
    entries.push({
      timestamp: new Date(now - i * 6 * 60 * 60 * 1000).toISOString(),
      prediction: {
        patient_token: patientToken,
        amr_risk_score: score,
        risk_tier: scoreToTier(score),
        confidence_score: 0.7 + Math.random() * 0.25,
        low_confidence_flag: false,
        data_completeness_score: 0.85 + Math.random() * 0.15,
        data_quality_flag: true,
        antibiotic_class_risk: {
          penicillins: Math.random() * 0.8,
          cephalosporins: Math.random() * 0.6,
          carbapenems: Math.random() * 0.4,
          fluoroquinolones: Math.random() * 0.5,
          aminoglycosides: Math.random() * 0.3,
        },
        shap_top_features: [
          { name: 'prior_resistance', value: 0.42, direction: 'positive' as const, human_readable: 'Previous resistant culture increases risk' },
          { name: 'antibiotic_days', value: 0.31, direction: 'positive' as const, human_readable: 'Extended antibiotic exposure' },
          { name: 'wbc_trend', value: -0.18, direction: 'negative' as const, human_readable: 'Normalizing WBC count reduces risk' },
          { name: 'icu_duration', value: 0.25, direction: 'positive' as const, human_readable: 'Prolonged ICU stay' },
          { name: 'age_group', value: 0.12, direction: 'positive' as const, human_readable: 'Age-related risk factor' },
        ],
        recommended_action: score >= 75 ? 'URGENT: Consider broadening empiric coverage' : score >= 50 ? 'Monitor closely and consider cultures' : 'Continue current management',
        model_version: 'v1.0.0',
      },
    });
  }
  return entries;
}

function SHAPPanel({ features, autoExpand }: { features: SHAPFeature[]; autoExpand: boolean }) {
  const [expanded, setExpanded] = useState(autoExpand);

  return (
    <div className="mt-3">
      <button
        onClick={() => setExpanded(!expanded)}
        className="text-sm text-blue-600 hover:text-blue-800 font-medium"
      >
        {expanded ? '▼' : '▶'} SHAP Explanations ({features.length})
      </button>
      {expanded && (
        <div className="mt-2 space-y-2">
          {features.map((feat, idx) => (
            <div key={idx} className="flex items-center gap-3 text-sm">
              <div className={`w-2 h-2 rounded-full ${feat.direction === 'positive' ? 'bg-red-400' : 'bg-green-400'}`} />
              <div className="flex-1">
                <span className="font-medium">{feat.name}</span>
                <span className="text-gray-500 ml-2">({feat.value > 0 ? '+' : ''}{feat.value.toFixed(3)})</span>
              </div>
              <span className="text-gray-600 text-xs max-w-xs truncate">{feat.human_readable}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function PatientTimeline() {
  const { id } = useParams<{ id: string }>();
  const patientToken = id ?? 'PT_UNKNOWN';

  const timeline = useMemo(() => generateMockTimeline(patientToken), [patientToken]);
  const latest = timeline[timeline.length - 1];

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Patient Risk Timeline</h1>
        <p className="text-gray-600">Token: {patientToken} • Model: {latest?.prediction.model_version}</p>
      </div>

      {/* Current risk summary */}
      {latest && (
        <div className={`${TIER_STYLES[latest.prediction.risk_tier].bg} rounded-xl p-6 mb-6 border`}>
          <div className="flex justify-between items-center">
            <div>
              <p className="text-sm text-gray-600">Current AMR Risk Score</p>
              <p className={`text-4xl font-bold ${TIER_STYLES[latest.prediction.risk_tier].text}`}>
                {latest.prediction.amr_risk_score}
              </p>
              <p className="text-sm mt-1">{latest.prediction.risk_tier} Risk • Confidence: {(latest.prediction.confidence_score * 100).toFixed(0)}%</p>
            </div>
            <div className="text-right">
              <p className="text-sm font-medium text-gray-700">Recommended Action</p>
              <p className="text-sm text-gray-600 mt-1 max-w-xs">{latest.prediction.recommended_action}</p>
            </div>
          </div>
          {latest.prediction.low_confidence_flag && (
            <div className="mt-3 p-2 bg-amber-100 rounded text-amber-800 text-sm font-medium">
              ⚠ Low Confidence: Model confidence below 60% threshold
            </div>
          )}
          <SHAPPanel
            features={latest.prediction.shap_top_features}
            autoExpand={latest.prediction.risk_tier === RiskTier.HIGH || latest.prediction.risk_tier === RiskTier.CRITICAL}
          />
        </div>
      )}

      {/* Timeline chart (simplified bar representation) */}
      <div className="bg-white rounded-xl border p-6 mb-6">
        <h2 className="text-lg font-semibold mb-4">72-Hour Risk Trend</h2>
        <div className="flex items-end gap-1 h-40">
          {timeline.map((entry, idx) => {
            const height = `${entry.prediction.amr_risk_score}%`;
            const style = TIER_STYLES[entry.prediction.risk_tier];
            return (
              <div key={idx} className="flex-1 flex flex-col items-center">
                <div
                  className={`w-full ${style.dot} rounded-t opacity-80 hover:opacity-100 transition-opacity cursor-pointer`}
                  style={{ height }}
                  title={`${new Date(entry.timestamp).toLocaleString()}: Score ${entry.prediction.amr_risk_score}`}
                />
              </div>
            );
          })}
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-2">
          <span>-72h</span>
          <span>-48h</span>
          <span>-24h</span>
          <span>Now</span>
        </div>
      </div>

      {/* Timeline entries */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold">Assessment History</h2>
        {[...timeline].reverse().map((entry, idx) => {
          const style = TIER_STYLES[entry.prediction.risk_tier];
          return (
            <div key={idx} className="flex gap-4">
              <div className="flex flex-col items-center">
                <div className={`w-3 h-3 rounded-full ${style.dot}`} />
                {idx < timeline.length - 1 && <div className="w-0.5 flex-1 bg-gray-200" />}
              </div>
              <div className={`flex-1 ${style.bg} rounded-lg p-4 mb-2`}>
                <div className="flex justify-between">
                  <span className="text-sm font-medium">{new Date(entry.timestamp).toLocaleString()}</span>
                  <span className={`font-bold ${style.text}`}>Score: {entry.prediction.amr_risk_score}</span>
                </div>
                <p className="text-sm text-gray-600 mt-1">{entry.prediction.recommended_action}</p>
                <SHAPPanel
                  features={entry.prediction.shap_top_features}
                  autoExpand={entry.prediction.risk_tier === RiskTier.HIGH || entry.prediction.risk_tier === RiskTier.CRITICAL}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
