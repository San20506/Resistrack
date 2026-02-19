import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { RiskTier, UserRole } from '../types';
import type { AMRPredictionOutput } from '../types';

function scoreToTier(score: number): RiskTier {
  if (score < 25) return RiskTier.LOW;
  if (score < 50) return RiskTier.MEDIUM;
  if (score < 75) return RiskTier.HIGH;
  return RiskTier.CRITICAL;
}

const TIER_COLORS: Record<RiskTier, string> = {
  [RiskTier.LOW]: 'text-green-700 bg-green-100',
  [RiskTier.MEDIUM]: 'text-yellow-700 bg-yellow-100',
  [RiskTier.HIGH]: 'text-orange-700 bg-orange-100',
  [RiskTier.CRITICAL]: 'text-red-700 bg-red-100',
};

/** Generate mock high-risk patients for pharmacy view. */
function generatePharmacyData(): AMRPredictionOutput[] {
  const patients: AMRPredictionOutput[] = [];
  const names = ['PT_A1B2C3', 'PT_D4E5F6', 'PT_G7H8I9', 'PT_J0K1L2', 'PT_M3N4O5', 'PT_P6Q7R8'];

  for (const token of names) {
    const score = 50 + Math.round(Math.random() * 50);
    patients.push({
      patient_token: token,
      amr_risk_score: score,
      risk_tier: scoreToTier(score),
      confidence_score: 0.65 + Math.random() * 0.3,
      low_confidence_flag: false,
      data_completeness_score: 0.8 + Math.random() * 0.2,
      data_quality_flag: true,
      antibiotic_class_risk: {
        penicillins: 0.3 + Math.random() * 0.7,
        cephalosporins: 0.2 + Math.random() * 0.6,
        carbapenems: 0.1 + Math.random() * 0.5,
        fluoroquinolones: 0.2 + Math.random() * 0.5,
        aminoglycosides: 0.1 + Math.random() * 0.4,
      },
      shap_top_features: [
        { name: 'prior_resistance', value: 0.42, direction: 'positive', human_readable: 'Previous resistant culture' },
        { name: 'antibiotic_days', value: 0.31, direction: 'positive', human_readable: 'Extended antibiotic exposure' },
      ],
      recommended_action: score >= 75 ? 'Consider de-escalation or alternative therapy' : 'Monitor antibiotic therapy closely',
      model_version: 'v1.0.0',
    });
  }
  return patients.sort((a, b) => b.amr_risk_score - a.amr_risk_score);
}

/** Pharmacy view - HIGH/CRITICAL patients with de-escalation recommendations. */
export function PharmacyView() {
  const navigate = useNavigate();
  const patients = useMemo(() => generatePharmacyData(), []);
  const [sortBy, setSortBy] = useState<'risk' | 'confidence'>('risk');

  const sortedPatients = useMemo(() => {
    const sorted = [...patients];
    if (sortBy === 'confidence') sorted.sort((a, b) => a.confidence_score - b.confidence_score);
    return sorted;
  }, [patients, sortBy]);

  const criticalCount = patients.filter((p) => p.risk_tier === RiskTier.CRITICAL).length;
  const highCount = patients.filter((p) => p.risk_tier === RiskTier.HIGH).length;

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Pharmacy Dashboard</h1>
        <p className="text-gray-600">High-risk patients requiring antibiotic stewardship review</p>
        <p className="text-xs text-gray-400 mt-1">Role: {UserRole.PHARMACIST}</p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-red-50 rounded-xl p-4 border border-red-200">
          <p className="text-sm text-red-600">Critical Risk</p>
          <p className="text-3xl font-bold text-red-700">{criticalCount}</p>
        </div>
        <div className="bg-orange-50 rounded-xl p-4 border border-orange-200">
          <p className="text-sm text-orange-600">High Risk</p>
          <p className="text-3xl font-bold text-orange-700">{highCount}</p>
        </div>
        <div className="bg-blue-50 rounded-xl p-4 border border-blue-200">
          <p className="text-sm text-blue-600">Total Monitored</p>
          <p className="text-3xl font-bold text-blue-700">{patients.length}</p>
        </div>
      </div>

      {/* Sort controls */}
      <div className="flex gap-2 mb-4">
        <button onClick={() => setSortBy('risk')} className={`px-3 py-1 rounded text-sm ${sortBy === 'risk' ? 'bg-blue-600 text-white' : 'bg-gray-100'}`}>
          Sort by Risk
        </button>
        <button onClick={() => setSortBy('confidence')} className={`px-3 py-1 rounded text-sm ${sortBy === 'confidence' ? 'bg-blue-600 text-white' : 'bg-gray-100'}`}>
          Sort by Confidence
        </button>
      </div>

      {/* Patient table */}
      <div className="bg-white rounded-xl border overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Patient Token</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Risk Score</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Tier</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Top Antibiotic Risk</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Recommendation</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {sortedPatients.map((patient) => {
              const topAbx = Object.entries(patient.antibiotic_class_risk)
                .sort(([, a], [, b]) => b - a)[0];
              return (
                <tr key={patient.patient_token} className="hover:bg-gray-50">
                  <td className="px-4 py-3 font-mono text-sm">{patient.patient_token}</td>
                  <td className="px-4 py-3 font-bold">{patient.amr_risk_score}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${TIER_COLORS[patient.risk_tier]}`}>
                      {patient.risk_tier}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm">
                    {topAbx[0]} ({(topAbx[1] * 100).toFixed(0)}%)
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600 max-w-xs truncate">{patient.recommended_action}</td>
                  <td className="px-4 py-3">
                    <button
                      onClick={() => navigate(`/patient/${patient.patient_token}`)}
                      className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                    >
                      View Timeline →
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** Infection Control view - outbreak trends and MDRO clusters. */
export function InfectionControlView() {
  const navigate = useNavigate();

  const outbreakData = useMemo(() => [
    { organism: 'MRSA', ward: 'ICU-A', cases: 3, trend: 'rising' as const, riskLevel: RiskTier.HIGH },
    { organism: 'VRE', ward: 'MED-2', cases: 2, trend: 'stable' as const, riskLevel: RiskTier.MEDIUM },
    { organism: 'CRE', ward: 'SURG-1', cases: 1, trend: 'declining' as const, riskLevel: RiskTier.LOW },
    { organism: 'ESBL', ward: 'ICU-B', cases: 4, trend: 'rising' as const, riskLevel: RiskTier.CRITICAL },
    { organism: 'C. diff', ward: 'MED-1', cases: 2, trend: 'stable' as const, riskLevel: RiskTier.MEDIUM },
  ], []);

  const trendIcon = { rising: '↑', stable: '→', declining: '↓' };
  const trendColor = { rising: 'text-red-600', stable: 'text-yellow-600', declining: 'text-green-600' };

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Infection Control Dashboard</h1>
        <p className="text-gray-600">MDRO cluster monitoring and outbreak detection</p>
        <p className="text-xs text-gray-400 mt-1">Role: {UserRole.INFECTION_CONTROL}</p>
      </div>

      {/* Outbreak summary */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="bg-red-50 rounded-xl p-4 border border-red-200">
          <p className="text-sm text-red-600">Active Outbreaks</p>
          <p className="text-3xl font-bold text-red-700">{outbreakData.filter((o) => o.trend === 'rising').length}</p>
        </div>
        <div className="bg-yellow-50 rounded-xl p-4 border border-yellow-200">
          <p className="text-sm text-yellow-600">Under Watch</p>
          <p className="text-3xl font-bold text-yellow-700">{outbreakData.filter((o) => o.trend === 'stable').length}</p>
        </div>
        <div className="bg-green-50 rounded-xl p-4 border border-green-200">
          <p className="text-sm text-green-600">Declining</p>
          <p className="text-3xl font-bold text-green-700">{outbreakData.filter((o) => o.trend === 'declining').length}</p>
        </div>
        <div className="bg-blue-50 rounded-xl p-4 border border-blue-200">
          <p className="text-sm text-blue-600">Total MDRO Cases</p>
          <p className="text-3xl font-bold text-blue-700">{outbreakData.reduce((s, o) => s + o.cases, 0)}</p>
        </div>
      </div>

      {/* MDRO cluster table */}
      <div className="bg-white rounded-xl border overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Organism</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Ward</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Cases</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Trend</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Risk Level</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {outbreakData.map((outbreak, idx) => (
              <tr key={idx} className="hover:bg-gray-50">
                <td className="px-4 py-3 font-medium">{outbreak.organism}</td>
                <td className="px-4 py-3">{outbreak.ward}</td>
                <td className="px-4 py-3 font-bold">{outbreak.cases}</td>
                <td className={`px-4 py-3 font-medium ${trendColor[outbreak.trend]}`}>
                  {trendIcon[outbreak.trend]} {outbreak.trend}
                </td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${TIER_COLORS[outbreak.riskLevel]}`}>
                    {outbreak.riskLevel}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <button
                    onClick={() => navigate(`/heatmap`)}
                    className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                  >
                    View Ward →
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
