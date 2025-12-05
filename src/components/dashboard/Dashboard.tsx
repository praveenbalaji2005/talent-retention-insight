import { Users, AlertTriangle, TrendingDown, Building2 } from 'lucide-react';
import { StatCard } from './StatCard';
import { useDatasets } from '@/hooks/useDatasets';
import { useLatestAnalysis } from '@/hooks/useAnalysis';
import { AttritionChart } from '../charts/AttritionChart';
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
      <div className="container py-8">
        <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
          <div className="rounded-full bg-primary/10 p-6 mb-6">
            <Users className="h-12 w-12 text-primary" />
          </div>
          <h2 className="text-2xl font-bold mb-2">Welcome to Attrition AI</h2>
          <p className="text-muted-foreground max-w-md mb-6">
            Upload your employee dataset to get started with predictive attrition analysis and actionable HR insights.
          </p>
          <Button onClick={() => onNavigate('datasets')} size="lg">
            Upload Your First Dataset
          </Button>
        </div>
      </div>
    );
  }

  const results = analysis.results;

  return (
    <div className="container py-8 space-y-8">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Dashboard</h2>
        <p className="text-muted-foreground">
          Overview of employee attrition predictions and insights
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Total Employees"
          value={results?.total_employees || 0}
          subtitle="In current dataset"
          icon={Users}
          variant="primary"
        />
        <StatCard
          title="At Risk"
          value={results?.at_risk_count || 0}
          subtitle="High/Critical risk level"
          icon={AlertTriangle}
          variant="destructive"
        />
        <StatCard
          title="Attrition Rate"
          value={`${results?.attrition_rate || 0}%`}
          subtitle="Predicted turnover"
          icon={TrendingDown}
          variant="warning"
        />
        <StatCard
          title="Departments"
          value={results?.department_breakdown?.length || 0}
          subtitle="Being monitored"
          icon={Building2}
          variant="secondary"
        />
      </div>

      {/* Charts Grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Risk Distribution</CardTitle>
            <CardDescription>Employee count by risk level</CardDescription>
          </CardHeader>
          <CardContent>
            {results?.risk_distribution && (
              <RiskDistributionChart data={results.risk_distribution} />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Department Breakdown</CardTitle>
            <CardDescription>Attrition risk by department</CardDescription>
          </CardHeader>
          <CardContent>
            {results?.department_breakdown && (
              <DepartmentChart data={results.department_breakdown} />
            )}
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
          <CardDescription>Navigate to detailed analysis</CardDescription>
        </CardHeader>
        <CardContent className="flex gap-4">
          <Button onClick={() => onNavigate('analysis')} variant="default">
            View Full Analysis
          </Button>
          <Button onClick={() => onNavigate('datasets')} variant="outline">
            Manage Datasets
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

function DashboardSkeleton() {
  return (
    <div className="container py-8 space-y-8">
      <div>
        <Skeleton className="h-9 w-48 mb-2" />
        <Skeleton className="h-5 w-96" />
      </div>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {[...Array(4)].map((_, i) => (
          <Skeleton key={i} className="h-36 rounded-xl" />
        ))}
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <Skeleton className="h-80 rounded-xl" />
        <Skeleton className="h-80 rounded-xl" />
      </div>
    </div>
  );
}
