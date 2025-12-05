import { CheckCircle, Circle, Loader2, XCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { PipelineStep } from '@/types/dataset';

interface PipelineStatusProps {
  steps: PipelineStep[];
}

export function PipelineStatus({ steps }: PipelineStatusProps) {
  const getStatusIcon = (status: PipelineStep['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-success" />;
      case 'processing':
        return <Loader2 className="h-5 w-5 text-primary animate-spin" />;
      case 'failed':
        return <XCircle className="h-5 w-5 text-destructive" />;
      default:
        return <Circle className="h-5 w-5 text-muted-foreground" />;
    }
  };

  return (
    <div className="space-y-4">
      {steps.map((step, index) => (
        <div
          key={step.id}
          className={cn(
            'flex items-start gap-4 p-4 rounded-lg border transition-all',
            step.status === 'processing' && 'border-primary bg-primary/5',
            step.status === 'completed' && 'border-success/30 bg-success/5',
            step.status === 'failed' && 'border-destructive/30 bg-destructive/5',
            step.status === 'pending' && 'border-border bg-muted/20'
          )}
        >
          <div className="flex-shrink-0 mt-0.5">
            {getStatusIcon(step.status)}
          </div>
          <div className="flex-grow">
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-muted-foreground">
                Step {step.id}
              </span>
            </div>
            <h4 className="font-medium">{step.name}</h4>
            <p className="text-sm text-muted-foreground">{step.description}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

export const defaultPipelineSteps: PipelineStep[] = [
  {
    id: 1,
    name: 'Data Ingestion',
    description: 'Loading and validating the uploaded dataset',
    status: 'pending',
    icon: 'database',
  },
  {
    id: 2,
    name: 'Preprocessing',
    description: 'Cleaning data, handling missing values, and feature encoding',
    status: 'pending',
    icon: 'cog',
  },
  {
    id: 3,
    name: 'Prediction',
    description: 'Running attrition prediction model on employee records',
    status: 'pending',
    icon: 'brain',
  },
  {
    id: 4,
    name: 'Explainability',
    description: 'Computing SHAP values and feature importance analysis',
    status: 'pending',
    icon: 'lightbulb',
  },
  {
    id: 5,
    name: 'Visualization',
    description: 'Generating charts and preparing downloadable reports',
    status: 'pending',
    icon: 'chart',
  },
];
