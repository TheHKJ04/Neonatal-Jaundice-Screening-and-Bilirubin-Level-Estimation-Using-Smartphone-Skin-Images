export interface PredictionResponse {
  regression_prediction: number;
  classification_probability: number;
  classification_prediction: number;
  research_only: boolean;
  message: string;
}

export interface ApiErrorResponse {
  detail?: string | Array<{ msg?: string }>;
}

export interface PredictionFormState {
  selectedFile: File | null;
  isLoading: boolean;
  error: string | null;
  result: PredictionResponse | null;
}
