import { useState, useEffect } from 'react';
import { Brain, Download, Play, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import { useDatasets } from '@/hooks/useDatasets';
import { useLatestAnalysis, useRunAnalysis } from '@/hooks/useAnalysis';
import { PipelineStatus, defaultPipelineSteps } from './PipelineStatus';
import { RecommendationsPanel } from './RecommendationsPanel';
import { RiskDistributionChart } from '../charts/RiskDistributionChart';
import { DepartmentChart } from '../charts/DepartmentChart';
import { FeatureImportanceChart } from '../charts/FeatureImportanceChart';
import { TopicsChart } from '../charts/TopicsChart';
import { AttritionChart } from '../charts/AttritionChart';
import type { PipelineStep } from '@/types/dataset';

export function AnalysisView() {
  const { data: datasets, isLoading: datasetsLoading } = useDatasets();
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | undefined>();
  const { data: analysis, isLoading: analysisLoading, refetch } = useLatestAnalysis(selectedDatasetId);
  const runAnalysisMutation = useRunAnalysis();
  
  const [pipelineSteps, setPipelineSteps] = useState<PipelineStep[]>(defaultPipelineSteps);

  // Auto-select first dataset
  useEffect(() => {
    if (datasets?.length && !selectedDatasetId) {
      setSelectedDatasetId(datasets[0].id);
    }
  }, [datasets, selectedDatasetId]);

  // Simulate pipeline progress
  useEffect(() => {
    if (runAnalysisMutation.isPending) {
      let currentStep = 0;
      const interval = setInterval(() => {
        setPipelineSteps((prev) =>
          prev.map((step, index) => ({
            ...step,
            status:
              index < currentStep
                ? 'completed'
                : index === currentStep
                ? 'processing'
                : 'pending',
          }))
        );
        currentStep++;
        if (currentStep > 5) {
          clearInterval(interval);
        }
      }, 400);
      return () => clearInterval(interval);
    } else {
      setPipelineSteps(defaultPipelineSteps);
    }
  }, [runAnalysisMutation.isPending]);

  const selectedDataset = datasets?.find((d) => d.id === selectedDatasetId);

  const handleRunAnalysis = () => {
    if (selectedDataset) {
      runAnalysisMutation.mutate(selectedDataset);
    }
  };

  const handleDownloadReport = () => {
    if (!analysis) return;
    
    const report = {
      generated_at: new Date().toISOString(),
      dataset: selectedDataset?.name,
      results: analysis.results,
      predictions: analysis.predictions,
      feature_importance: analysis.feature_importance,
      recommendations: analysis.recommendations,
    };
    
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `attrition-report-${selectedDataset?.name || 'analysis'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (datasetsLoading) {
    return <AnalysisSkeleton />;
  }

  if (!datasets?.length) {
    return (
      <div className="container py-8">
        <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
          <div className="rounded-full bg-primary/10 p-6 mb-6">
            <Brain className="h-12 w-12 text-primary" />
          </div>
          <h2 className="text-2xl font-bold mb-2">No Datasets Available</h2>
          <p className="text-muted-foreground max-w-md">
            Upload a dataset first to run attrition analysis and get insights.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="container py-8 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Analysis</h2>
          <p className="text-muted-foreground">
            Run ML pipeline and explore attrition predictions
          </p>
        </div>
        
        <div className="flex items-center gap-4">
          <Select value={selectedDatasetId} onValueChange={setSelectedDatasetId}>
            <SelectTrigger className="w-64">
              <SelectValue placeholder="Select dataset" />
            </SelectTrigger>
            <SelectContent>
              {datasets.map((dataset) => (
                <SelectItem key={dataset.id} value={dataset.id}>
                  {dataset.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          
          <Button
            onClick={handleRunAnalysis}
            disabled={!selectedDatasetId || runAnalysisMutation.isPending}
          >
            {runAnalysisMutation.isPending ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Running...
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Run Analysis
              </>
            )}
          </Button>
          
          {analysis && (
            <Button variant="outline" onClick={handleDownloadReport}>
              <Download className="h-4 w-4 mr-2" />
              Export Report
            </Button>
          )}
        </div>
      </div>

      {/* Pipeline Status */}
      {runAnalysisMutation.isPending && (
        <Card>
          <CardHeader>
            <CardTitle>Analysis Pipeline</CardTitle>
            <CardDescription>Processing your dataset through the ML pipeline</CardDescription>
          </CardHeader>
          <CardContent>
            <PipelineStatus steps={pipelineSteps} />
          </CardContent>
        </Card>
      )}

      {/* Analysis Results */}
      {analysis && !runAnalysisMutation.isPending && (
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 lg:w-auto lg:inline-grid">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="predictions">Predictions</TabsTrigger>
            <TabsTrigger value="explainability">Explainability</TabsTrigger>
            <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Risk Distribution</CardTitle>
                  <CardDescription>Employee count by risk level</CardDescription>
                </CardHeader>
                <CardContent>
                  {analysis.results?.risk_distribution && (
                    <RiskDistributionChart data={analysis.results.risk_distribution} />
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Department Analysis</CardTitle>
                  <CardDescription>Attrition risk by department</CardDescription>
                </CardHeader>
                <CardContent>
                  {analysis.results?.department_breakdown && (
                    <DepartmentChart data={analysis.results.department_breakdown} />
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="predictions" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Risk Score Distribution</CardTitle>
                <CardDescription>Histogram of employee risk scores</CardDescription>
              </CardHeader>
              <CardContent>
                {analysis.predictions && (
                  <AttritionChart data={analysis.predictions} />
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="explainability" className="space-y-6">
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Feature Importance (SHAP)</CardTitle>
                  <CardDescription>
                    Key factors driving attrition predictions
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {analysis.feature_importance && (
                    <FeatureImportanceChart data={analysis.feature_importance} />
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Topic Analysis (LDA)</CardTitle>
                  <CardDescription>
                    Key themes from employee reviews
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {analysis.topics && (
                    <TopicsChart data={analysis.topics} />
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Feature Descriptions */}
            {analysis.feature_importance && (
              <Card>
                <CardHeader>
                  <CardTitle>Feature Analysis Details</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-3 md:grid-cols-2">
                    {analysis.feature_importance.slice(0, 6).map((feature) => (
                      <div
                        key={feature.feature}
                        className="p-4 rounded-lg border bg-muted/20"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium">{feature.feature}</span>
                          <span
                            className={
                              feature.direction === 'positive'
                                ? 'text-destructive'
                                : 'text-success'
                            }
                          >
                            {feature.direction === 'positive' ? '↑' : '↓'}{' '}
                            {(feature.importance * 100).toFixed(1)}%
                          </span>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          {feature.description}
                        </p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="recommendations">
            {analysis.recommendations && (
              <RecommendationsPanel recommendations={analysis.recommendations} />
            )}
          </TabsContent>
        </Tabs>
      )}

      {/* No Analysis Yet */}
      {!analysis && !runAnalysisMutation.isPending && selectedDatasetId && (
        <Card className="p-8 text-center">
          <div className="flex flex-col items-center gap-4">
            <div className="rounded-full bg-muted p-4">
              <Brain className="h-8 w-8 text-muted-foreground" />
            </div>
            <div>
              <h3 className="font-semibold">No Analysis Results</h3>
              <p className="text-sm text-muted-foreground">
                Click "Run Analysis" to process this dataset
              </p>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}

function AnalysisSkeleton() {
  return (
    <div className="container py-8 space-y-8">
      <div className="flex justify-between">
        <div>
          <Skeleton className="h-9 w-48 mb-2" />
          <Skeleton className="h-5 w-96" />
        </div>
        <div className="flex gap-4">
          <Skeleton className="h-10 w-64" />
          <Skeleton className="h-10 w-32" />
        </div>
      </div>
      <Skeleton className="h-12 w-96" />
      <div className="grid gap-6 lg:grid-cols-2">
        <Skeleton className="h-96 rounded-xl" />
        <Skeleton className="h-96 rounded-xl" />
      </div>
    </div>
  );
}
