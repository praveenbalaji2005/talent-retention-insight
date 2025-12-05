export interface Dataset {
  id: string;
  name: string;
  description: string | null;
  file_type: 'attrition' | 'reviews';
  raw_data: Record<string, unknown>[];
  column_names: string[];
  row_count: number;
  created_at: string;
  updated_at: string;
}

export interface AnalysisResult {
  id: string;
  dataset_id: string;
  analysis_type: 'attrition_prediction' | 'shap_importance' | 'topic_modeling' | 'full_pipeline';
  status: 'pending' | 'processing' | 'completed' | 'failed';
  results: AnalysisResults | null;
  predictions: PredictionData[] | null;
  feature_importance: FeatureImportance[] | null;
  topics: TopicData[] | null;
  recommendations: Recommendation[] | null;
  error_message: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface AnalysisResults {
  total_employees: number;
  at_risk_count: number;
  attrition_rate: number;
  department_breakdown: DepartmentBreakdown[];
  risk_distribution: RiskDistribution;
}

export interface DepartmentBreakdown {
  department: string;
  total: number;
  at_risk: number;
  rate: number;
}

export interface RiskDistribution {
  low: number;
  medium: number;
  high: number;
  critical: number;
}

export interface PredictionData {
  employee_id: string | number;
  department: string;
  risk_score: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  attrition_probability: number;
  factors: string[];
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  direction: 'positive' | 'negative';
  description: string;
}

export interface TopicData {
  topic_id: number;
  name: string;
  keywords: string[];
  prevalence: number;
  sentiment: 'positive' | 'neutral' | 'negative';
  sample_reviews: string[];
}

export interface Recommendation {
  id: string;
  priority: 'high' | 'medium' | 'low';
  category: string;
  title: string;
  description: string;
  impact: string;
  action_items: string[];
}

export interface PipelineStep {
  id: number;
  name: string;
  description: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  icon: string;
}

// API Contract types for external ML backend
export interface ExternalMLRequest {
  dataset_id: string;
  raw_data: Record<string, unknown>[];
  analysis_type: 'attrition_prediction' | 'shap_importance' | 'topic_modeling' | 'full_pipeline';
  config?: {
    model_type?: string;
    n_topics?: number;
    shap_samples?: number;
  };
}

export interface ExternalMLResponse {
  success: boolean;
  analysis_id: string;
  results?: AnalysisResults;
  predictions?: PredictionData[];
  feature_importance?: FeatureImportance[];
  topics?: TopicData[];
  recommendations?: Recommendation[];
  error?: string;
}
