export enum RiskTier {
  LOW = "LOW",
  MEDIUM = "MEDIUM",
  HIGH = "HIGH",
  CRITICAL = "CRITICAL"
}

export enum UserRole {
  PHYSICIAN = "PHYSICIAN",
  PHARMACIST = "PHARMACIST",
  INFECTION_CONTROL = "INFECTION_CONTROL",
  NURSE = "NURSE",
  ADMIN = "ADMIN",
  READONLY = "READONLY"
}

export interface SHAPFeature {
  name: string;
  value: number;
  direction: "positive" | "negative";
  human_readable: string;
}

export interface AntibioticClassRisk {
  penicillins: number;
  cephalosporins: number;
  carbapenems: number;
  fluoroquinolones: number;
  aminoglycosides: number;
}

export interface AMRPredictionOutput {
  patient_token: string;
  amr_risk_score: number; // 0-100
  risk_tier: RiskTier;
  confidence_score: number; // 0-1
  low_confidence_flag: boolean;
  data_completeness_score: number; // 0-1
  data_quality_flag: boolean;
  antibiotic_class_risk: AntibioticClassRisk;
  shap_top_features: SHAPFeature[];
  recommended_action: string;
  model_version: string;
}
