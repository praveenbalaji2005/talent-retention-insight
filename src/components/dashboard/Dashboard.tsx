import { Users, AlertTriangle, TrendingDown, Building2, Upload } from 'lucide-react';
import { StatCard } from './StatCard';
import { useDatasets } from '@/hooks/useDatasets';
import { useLatestAnalysis } from '@/hooks/useAnalysis';
import { DepartmentChart } from '../charts/DepartmentChart';
import { RiskDistributionChart } from '../charts/RiskDistributionChart';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';

interface DashboardProps {
  onNavigate: (view: 'datasets' | 'analysis') => void;
}

export function Dashboard({ onNavigate }: DashboardProps) {
  const { data: datasets, isLoading: datasetsLoading } = useDatasets();
  const latestDataset = datasets?.[0];
  const { data: analysis, isLoading: analysisLoading } = useLatestAnalysis(latestDataset?.id);

  const isLoading = datasetsLoading || analysisLoading;

  if (isLoading) {
    return <DashboardSkeleton />;
  }

  if (!datasets?.length || !analysis) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[70vh] text-center px-4">
        <div className="rounded-2xl bg-gradient-to-br from-primary/10 to-secondary/10 p-8 mb-6">
          <Upload className="h-12 w-12 text-primary mx-auto" />
        </div>
        <h2 className="text-2xl font-bold tracking-tight mb-2">Welcome to Attrition AI</h2>
        <p className="text-muted-foreground max-w-md mb-6 text-sm">
          Upload your employee dataset to get started with XAI-powered attrition prediction and actionable HR insights.
        </p>
        <Button onClick={() => onNavigate('datasets')} size="lg" className="gap-2">
          <Upload className="h-4 w-4" />
          Upload Dataset
        </Button>
        <div className="mt-8 grid grid-cols-3 gap-6 text-center max-w-lg">
          <div>
            <div className="text-2xl font-bold text-primary">IBM</div>
            <div className="text-xs text-muted-foreground">HR Dataset</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-secondary">Kaggle</div>
            <div className="text-xs text-muted-foreground">Analytics</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-accent">Reviews</div>
            <div className="text-xs text-muted-foreground">AmbitionBox</div>
          </div>
        </div>
      </div>
    );
  }

  const results = analysis.results;

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold tracking-tight">Dashboard</h2>
          <p className="text-sm text-muted-foreground">
            Overview of attrition predictions • {latestDataset?.name}
          </p>
        </div>
        <div className="flex gap-2">
          <Button onClick={() => onNavigate('analysis')} size="sm" variant="default">
            View Analysis
          </Button>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-3 grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Total Records"
          value={results?.total_employees?.toLocaleString() || 0}
          subtitle="In dataset"
          icon={Users}
          variant="primary"
        />
        <StatCard
          title="At Risk"
          value={results?.at_risk_count?.toLocaleString() || 0}
          subtitle="High/Critical"
          icon={AlertTriangle}
          variant="destructive"
        />
        <StatCard
          title="Risk Rate"
          value={`${results?.attrition_rate || 0}%`}
          subtitle="Predicted"
          icon={TrendingDown}
          variant="warning"
        />
        <StatCard
          title="Departments"
          value={results?.department_breakdown?.length || 0}
          subtitle="Analyzed"
          icon={Building2}
          variant="secondary"
        />
      </div>

      {/* Charts Grid */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Risk Distribution</CardTitle>
            <CardDescription className="text-xs">Employee count by risk level</CardDescription>
          </CardHeader>
          <CardContent className="pt-0">
            {results?.risk_distribution && (
              <RiskDistributionChart data={results.risk_distribution} />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Department Breakdown</CardTitle>
            <CardDescription className="text-xs">Attrition risk by department</CardDescription>
          </CardHeader>
          <CardContent className="pt-0">
            {results?.department_breakdown && (
              <DepartmentChart data={results.department_breakdown} />
            )}
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card>
        <CardContent className="flex items-center justify-between py-4">
          <div>
            <p className="text-sm font-medium">Ready to dive deeper?</p>
            <p className="text-xs text-muted-foreground">View full SHAP analysis and recommendations</p>
          </div>
          <div className="flex gap-2">
            <Button onClick={() => onNavigate('analysis')} size="sm">
              Full Analysis
            </Button>
            <Button onClick={() => onNavigate('datasets')} variant="outline" size="sm">
              Manage Data
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function DashboardSkeleton() {
  return (
    <div className="space-y-6">
      <div>
        <Skeleton className="h-6 w-32 mb-1" />
        <Skeleton className="h-4 w-64" />
      </div>
      <div className="grid gap-3 grid-cols-2 lg:grid-cols-4">
        {[...Array(4)].map((_, i) => (
          <Skeleton key={i} className="h-24 rounded-xl" />
        ))}
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Skeleton className="h-72 rounded-xl" />
        <Skeleton className="h-72 rounded-xl" />
      </div>
    </div>
  );
}
