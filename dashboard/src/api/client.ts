import axios from 'axios';
import { AMRPredictionOutput } from '../types';

// Create axios instance with base URL placeholder
export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

// API function stubs
export const getPrediction = async (patientId: string): Promise<AMRPredictionOutput> => {
  // Placeholder implementation
  const response = await apiClient.get<AMRPredictionOutput>(`/predictions/${patientId}`);
  return response.data;
};

export const getPatientTimeline = async (patientId: string): Promise<any> => {
  // Placeholder implementation
  const response = await apiClient.get(`/patients/${patientId}/timeline`);
  return response.data;
};

export const getWardHeatmap = async (wardId: string): Promise<any> => {
  // Placeholder implementation
  const response = await apiClient.get(`/wards/${wardId}/heatmap`);
  return response.data;
};
