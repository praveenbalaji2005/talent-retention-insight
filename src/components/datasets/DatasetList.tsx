import { FileSpreadsheet, Trash2, Play, Calendar, Database } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';
import { useDatasets, useDeleteDataset } from '@/hooks/useDatasets';
import { useRunAnalysis } from '@/hooks/useAnalysis';
import type { Dataset } from '@/types/dataset';
import { formatDistanceToNow } from 'date-fns';

interface DatasetListProps {
  onSelectDataset?: (dataset: Dataset) => void;
}

export function DatasetList({ onSelectDataset }: DatasetListProps) {
  const { data: datasets, isLoading } = useDatasets();
  const deleteMutation = useDeleteDataset();
  const runAnalysisMutation = useRunAnalysis();

  if (isLoading) {
    return (
      <div className="space-y-4">
        {[...Array(3)].map((_, i) => (
          <Skeleton key={i} className="h-32 rounded-xl" />
        ))}
      </div>
    );
  }

  if (!datasets?.length) {
    return (
      <Card className="p-8 text-center">
        <div className="flex flex-col items-center gap-4">
          <div className="rounded-full bg-muted p-4">
            <Database className="h-8 w-8 text-muted-foreground" />
          </div>
          <div>
            <h3 className="font-semibold">No datasets yet</h3>
            <p className="text-sm text-muted-foreground">
              Upload your first dataset to get started
            </p>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {datasets.map((dataset) => (
        <Card
          key={dataset.id}
          className="hover:shadow-md transition-shadow cursor-pointer"
          onClick={() => onSelectDataset?.(dataset)}
        >
          <CardContent className="p-6">
            <div className="flex items-start justify-between">
              <div className="flex items-start gap-4">
                <div className="rounded-lg bg-primary/10 p-3">
                  <FileSpreadsheet className="h-6 w-6 text-primary" />
                </div>
                <div className="space-y-1">
                  <h3 className="font-semibold">{dataset.name}</h3>
                  {dataset.description && (
                    <p className="text-sm text-muted-foreground line-clamp-1">
                      {dataset.description}
                    </p>
                  )}
                  <div className="flex items-center gap-4 text-sm text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <Database className="h-3 w-3" />
                      {dataset.row_count} rows
                    </span>
                    <span className="flex items-center gap-1">
                      <Calendar className="h-3 w-3" />
                      {formatDistanceToNow(new Date(dataset.created_at), { addSuffix: true })}
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <Badge variant={dataset.file_type === 'attrition' ? 'default' : 'secondary'}>
                  {dataset.file_type === 'attrition' ? 'Attrition' : 'Reviews'}
                </Badge>
                
                <Button
                  variant="outline"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    runAnalysisMutation.mutate(dataset);
                  }}
                  disabled={runAnalysisMutation.isPending}
                >
                  <Play className="h-4 w-4 mr-1" />
                  Analyze
                </Button>
                
                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <Trash2 className="h-4 w-4 text-muted-foreground hover:text-destructive" />
                    </Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>Delete Dataset</AlertDialogTitle>
                      <AlertDialogDescription>
                        This will permanently delete "{dataset.name}" and all associated analysis results.
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>Cancel</AlertDialogCancel>
                      <AlertDialogAction
                        onClick={() => deleteMutation.mutate(dataset.id)}
                        className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                      >
                        Delete
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
