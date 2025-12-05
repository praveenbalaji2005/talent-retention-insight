import { ExternalLink, Code, Database, FileJson } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';

export function SettingsView() {
  const apiContracts = [
    {
      method: 'POST',
      endpoint: '/api/upload-dataset',
      description: 'Upload CSV and store in database',
      requestBody: {
        name: 'string',
        description: 'string?',
        file_type: "'attrition' | 'reviews'",
        raw_data: 'Record<string, unknown>[]',
        column_names: 'string[]',
      },
      responseBody: {
        id: 'uuid',
        name: 'string',
        row_count: 'number',
        created_at: 'timestamp',
      },
    },
    {
      method: 'POST',
      endpoint: '/api/analyze-dataset',
      description: 'Run ML pipeline on dataset',
      requestBody: {
        dataset_id: 'uuid',
        analysis_type: "'full_pipeline'",
        config: {
          model_type: 'string?',
          n_topics: 'number?',
        },
      },
      responseBody: {
        analysis_id: 'uuid',
        status: "'processing' | 'completed' | 'failed'",
        results: 'AnalysisResults',
        predictions: 'PredictionData[]',
        feature_importance: 'FeatureImportance[]',
      },
    },
    {
      method: 'GET',
      endpoint: '/api/get-analysis',
      description: 'Fetch analysis results',
      params: {
        dataset_id: 'uuid',
      },
      responseBody: {
        analysis_id: 'uuid',
        status: 'string',
        results: 'AnalysisResults',
        predictions: 'PredictionData[]',
        feature_importance: 'FeatureImportance[]',
        topics: 'TopicData[]',
        recommendations: 'Recommendation[]',
      },
    },
  ];

  return (
    <div className="container py-8 space-y-8">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Settings</h2>
        <p className="text-muted-foreground">
          API contracts and configuration for external ML backend integration
        </p>
      </div>

      {/* External API Integration */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Code className="h-5 w-5 text-primary" />
            <CardTitle>External ML API Integration</CardTitle>
          </div>
          <CardDescription>
            This prototype uses simulated ML results. Connect your Python/FastAPI backend using these API contracts.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {apiContracts.map((contract, index) => (
            <div key={index} className="space-y-3">
              <div className="flex items-center gap-3">
                <Badge
                  variant={contract.method === 'GET' ? 'secondary' : 'default'}
                  className="font-mono"
                >
                  {contract.method}
                </Badge>
                <code className="text-sm font-mono bg-muted px-2 py-1 rounded">
                  {contract.endpoint}
                </code>
              </div>
              <p className="text-sm text-muted-foreground">{contract.description}</p>
              
              <div className="grid gap-4 md:grid-cols-2">
                {contract.requestBody && (
                  <div className="space-y-2">
                    <p className="text-xs font-medium text-muted-foreground uppercase">
                      Request Body
                    </p>
                    <pre className="text-xs bg-muted p-3 rounded-lg overflow-x-auto">
                      {JSON.stringify(contract.requestBody, null, 2)}
                    </pre>
                  </div>
                )}
                {contract.params && (
                  <div className="space-y-2">
                    <p className="text-xs font-medium text-muted-foreground uppercase">
                      Query Params
                    </p>
                    <pre className="text-xs bg-muted p-3 rounded-lg overflow-x-auto">
                      {JSON.stringify(contract.params, null, 2)}
                    </pre>
                  </div>
                )}
                <div className="space-y-2">
                  <p className="text-xs font-medium text-muted-foreground uppercase">
                    Response
                  </p>
                  <pre className="text-xs bg-muted p-3 rounded-lg overflow-x-auto">
                    {JSON.stringify(contract.responseBody, null, 2)}
                  </pre>
                </div>
              </div>
              
              {index < apiContracts.length - 1 && <Separator className="mt-4" />}
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Data Schema */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Database className="h-5 w-5 text-primary" />
            <CardTitle>Database Schema</CardTitle>
          </div>
          <CardDescription>
            Current database tables for storing datasets and analysis results
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <h4 className="font-medium">datasets</h4>
            <pre className="text-xs bg-muted p-3 rounded-lg overflow-x-auto">
{`{
  id: UUID (primary key)
  name: TEXT
  description: TEXT?
  file_type: 'attrition' | 'reviews'
  raw_data: JSONB
  column_names: TEXT[]
  row_count: INTEGER
  created_at: TIMESTAMP
  updated_at: TIMESTAMP
}`}
            </pre>
          </div>
          
          <div className="space-y-2">
            <h4 className="font-medium">analysis_results</h4>
            <pre className="text-xs bg-muted p-3 rounded-lg overflow-x-auto">
{`{
  id: UUID (primary key)
  dataset_id: UUID (foreign key -> datasets)
  analysis_type: 'attrition_prediction' | 'shap_importance' | 'topic_modeling' | 'full_pipeline'
  status: 'pending' | 'processing' | 'completed' | 'failed'
  results: JSONB
  predictions: JSONB
  feature_importance: JSONB
  topics: JSONB
  recommendations: JSONB
  error_message: TEXT?
  created_at: TIMESTAMP
  completed_at: TIMESTAMP?
}`}
            </pre>
          </div>
        </CardContent>
      </Card>

      {/* Type Definitions */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <FileJson className="h-5 w-5 text-primary" />
            <CardTitle>TypeScript Types</CardTitle>
          </div>
          <CardDescription>
            Type definitions for API integration (see src/types/dataset.ts)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <pre className="text-xs bg-muted p-3 rounded-lg overflow-x-auto max-h-96">
{`// External ML API Request/Response types
interface ExternalMLRequest {
  dataset_id: string;
  raw_data: Record<string, unknown>[];
  analysis_type: 'attrition_prediction' | 'shap_importance' | 'topic_modeling' | 'full_pipeline';
  config?: {
    model_type?: string;
    n_topics?: number;
    shap_samples?: number;
  };
}

interface ExternalMLResponse {
  success: boolean;
  analysis_id: string;
  results?: AnalysisResults;
  predictions?: PredictionData[];
  feature_importance?: FeatureImportance[];
  topics?: TopicData[];
  recommendations?: Recommendation[];
  error?: string;
}

interface PredictionData {
  employee_id: string | number;
  department: string;
  risk_score: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  attrition_probability: number;
  factors: string[];
}

interface FeatureImportance {
  feature: string;
  importance: number;
  direction: 'positive' | 'negative';
  description: string;
}`}
          </pre>
        </CardContent>
      </Card>
    </div>
  );
}
