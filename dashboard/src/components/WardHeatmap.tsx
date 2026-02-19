import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { RiskTier } from '../types';
import type { AMRPredictionOutput } from '../types';

/** Risk tier to color mapping (colorblind-friendly palette). */
const RISK_COLORS: Record<RiskTier, { bg: string; text: string; label: string }> = {
  [RiskTier.LOW]: { bg: 'bg-green-100', text: 'text-green-800', label: 'Low' },
  [RiskTier.MEDIUM]: { bg: 'bg-yellow-100', text: 'text-yellow-800', label: 'Medium' },
  [RiskTier.HIGH]: { bg: 'bg-orange-100', text: 'text-orange-800', label: 'High' },
  [RiskTier.CRITICAL]: { bg: 'bg-red-100', text: 'text-red-800', label: 'Critical' },
};

/** Determines risk tier from AMR risk score (0-100). */
function scoreToTier(score: number): RiskTier {
  if (score < 25) return RiskTier.LOW;
  if (score < 50) return RiskTier.MEDIUM;
  if (score < 75) return RiskTier.HIGH;
  return RiskTier.CRITICAL;
}

interface WardData {
  wardId: string;
  wardName: string;
  patients: AMRPredictionOutput[];
  avgRiskScore: number;
  riskTier: RiskTier;
}

/** Mock ward data for development. */
function generateMockWards(): WardData[] {
  const wards = [
    { id: 'ICU-A', name: 'ICU Wing A', baseRisk: 68 },
    { id: 'ICU-B', name: 'ICU Wing B', baseRisk: 72 },
    { id: 'MED-1', name: 'Medical Floor 1', baseRisk: 35 },
    { id: 'MED-2', name: 'Medical Floor 2', baseRisk: 42 },
    { id: 'SURG-1', name: 'Surgical Floor 1', baseRisk: 55 },
    { id: 'SURG-2', name: 'Surgical Floor 2', baseRisk: 28 },
    { id: 'PEDS', name: 'Pediatrics', baseRisk: 18 },
    { id: 'ONCO', name: 'Oncology', baseRisk: 62 },
    { id: 'ER', name: 'Emergency Department', baseRisk: 48 },
  ];

  return wards.map((ward) => {
    const avgRisk = ward.baseRisk + Math.round((Math.random() - 0.5) * 10);
    const clampedRisk = Math.max(0, Math.min(100, avgRisk));
    return {
      wardId: ward.id,
      wardName: ward.name,
      patients: [],
      avgRiskScore: clampedRisk,
      riskTier: scoreToTier(clampedRisk),
    };
  });
}

export function WardHeatmap() {
  const navigate = useNavigate();
  const [selectedTierFilter, setSelectedTierFilter] = useState<RiskTier | 'ALL'>('ALL');

  const wards = useMemo(() => generateMockWards(), []);

  const filteredWards = useMemo(() => {
    if (selectedTierFilter === 'ALL') return wards;
    return wards.filter((w) => w.riskTier === selectedTierFilter);
  }, [wards, selectedTierFilter]);

  const tierCounts = useMemo(() => {
    const counts: Record<string, number> = { ALL: wards.length };
    for (const tier of Object.values(RiskTier)) {
      counts[tier] = wards.filter((w) => w.riskTier === tier).length;
    }
    return counts;
  }, [wards]);

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Ward AMR Risk Heatmap</h1>
        <p className="text-gray-600 mt-1">Interactive ward-level antimicrobial resistance risk overview</p>
      </div>

      {/* Filter bar */}
      <div className="flex gap-2 mb-6">
        {(['ALL', ...Object.values(RiskTier)] as const).map((tier) => (
          <button
            key={tier}
            onClick={() => setSelectedTierFilter(tier)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              selectedTierFilter === tier
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            {tier === 'ALL' ? 'All' : RISK_COLORS[tier].label} ({tierCounts[tier]})
          </button>
        ))}
      </div>

      {/* Heatmap grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredWards.map((ward) => {
          const colors = RISK_COLORS[ward.riskTier];
          return (
            <button
              key={ward.wardId}
              onClick={() => navigate(`/patient/${ward.wardId}`)}
              className={`${colors.bg} rounded-xl p-6 text-left transition-transform hover:scale-105 cursor-pointer border-2 border-transparent hover:border-blue-400`}
            >
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="font-semibold text-gray-900">{ward.wardName}</h3>
                  <p className="text-sm text-gray-600">{ward.wardId}</p>
                </div>
                <span className={`${colors.text} font-bold text-2xl`}>
                  {ward.avgRiskScore}
                </span>
              </div>
              <div className="mt-4">
                <span className={`inline-block px-3 py-1 rounded-full text-xs font-medium ${colors.bg} ${colors.text} border border-current`}>
                  {colors.label} Risk
                </span>
              </div>
            </button>
          );
        })}
      </div>

      {/* Legend */}
      <div className="mt-8 p-4 bg-white rounded-lg border">
        <h3 className="text-sm font-semibold text-gray-700 mb-2">Risk Tier Legend (Colorblind-Friendly)</h3>
        <div className="flex gap-6">
          {Object.values(RiskTier).map((tier) => {
            const colors = RISK_COLORS[tier];
            return (
              <div key={tier} className="flex items-center gap-2">
                <div className={`w-4 h-4 rounded ${colors.bg} border border-gray-300`} />
                <span className="text-sm text-gray-600">
                  {colors.label} ({tier === RiskTier.LOW ? '0-24' : tier === RiskTier.MEDIUM ? '25-49' : tier === RiskTier.HIGH ? '50-74' : '75-100'})
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
