import { useState, useCallback, useMemo } from 'react';
import { Upload, AlertCircle, CheckCircle, Info } from 'lucide-react';
import Papa from 'papaparse';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { useUploadDataset } from '@/hooks/useDatasets';
import { cn } from '@/lib/utils';

interface DatasetUploadProps {
  onSuccess?: () => void;
}

// Column detection for different dataset formats
const detectDatasetFormat = (cols: string[]): { type: string; label: string; color: string } => {
  const lowerCols = cols.map(c => c.toLowerCase());
  
  // AmbitionBox format (reviews)
  if (lowerCols.some(c => ['overall_rating', 'work_life_balance', 'skill_development', 'salary_and_benefits'].includes(c))) {
    return { type: 'ambitionbox', label: 'Employee Reviews', color: 'bg-accent/20 text-accent' };
  }
  
  // Kaggle HR format
  if (lowerCols.some(c => ['satisfaction_level', 'number_project', 'time_spend_company'].includes(c))) {
    return { type: 'kaggle', label: 'Kaggle HR Analytics', color: 'bg-secondary/20 text-secondary' };
  }
  
  // IBM format
  if (lowerCols.some(c => ['jobsatisfaction', 'attrition', 'monthlyincome', 'yearsatcompany'].includes(c))) {
    return { type: 'ibm', label: 'IBM HR Analytics', color: 'bg-primary/20 text-primary' };
  }
  
  return { type: 'generic', label: 'Generic Dataset', color: 'bg-muted text-muted-foreground' };
};

export function DatasetUpload({ onSuccess }: DatasetUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [parsedData, setParsedData] = useState<Record<string, unknown>[] | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const uploadMutation = useUploadDataset();
  
  const detectedFormat = useMemo(() => detectDatasetFormat(columns), [columns]);

  const handleFileParse = useCallback((file: File) => {
    setError(null);
    
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        if (results.errors.length > 0) {
          setError(`Parse error: ${results.errors[0].message}`);
          return;
        }
        
        const data = results.data as Record<string, unknown>[];
        const cols = results.meta.fields || [];
        
        if (data.length === 0) {
          setError('The file appears to be empty');
          return;
        }
        
        setParsedData(data);
        setColumns(cols);
        setName(file.name.replace(/\.csv$/i, ''));
      },
      error: (err) => {
        setError(`Failed to parse file: ${err.message}`);
      },
    });
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      handleFileParse(selectedFile);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && (droppedFile.type === 'text/csv' || droppedFile.name.endsWith('.csv'))) {
      setFile(droppedFile);
      handleFileParse(droppedFile);
    } else {
      setError('Please drop a CSV file');
    }
  }, [handleFileParse]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!parsedData || !name) {
      setError('Please provide all required information');
      return;
    }

    try {
      await uploadMutation.mutateAsync({
        name,
        description: description || undefined,
        file_type: detectedFormat.type === 'ambitionbox' ? 'reviews' : 'attrition',
        raw_data: parsedData,
        column_names: columns,
      });
      
      setFile(null);
      setName('');
      setDescription('');
      setParsedData(null);
      setColumns([]);
      
      onSuccess?.();
    } catch {
      // Error handled by mutation
    }
  };

  // Key columns for display
  const keyColumns = columns.filter(col => {
    const lower = col.toLowerCase();
    return [
      'overall_rating', 'work_life_balance', 'satisfaction', 'attrition', 
      'department', 'salary', 'age', 'jobsatisfaction', 'work_satisfaction',
      'career_growth', 'job_security', 'number_project', 'satisfaction_level'
    ].some(k => lower.includes(k));
  });

  return (
    <Card className="border-0 shadow-none">
      <CardHeader className="px-0 pt-0">
        <CardTitle className="text-lg">Upload Dataset</CardTitle>
        <CardDescription className="text-xs">
          Supports IBM HR, Kaggle Analytics, and AmbitionBox review formats
        </CardDescription>
      </CardHeader>
      <CardContent className="px-0 pb-0">
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Supported Formats */}
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline" className="text-xs bg-primary/10 text-primary border-primary/20">IBM HR</Badge>
            <Badge variant="outline" className="text-xs bg-secondary/10 text-secondary border-secondary/20">Kaggle</Badge>
            <Badge variant="outline" className="text-xs bg-accent/10 text-accent border-accent/20">AmbitionBox</Badge>
          </div>

          {/* Drop Zone */}
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            className={cn(
              'relative border-2 border-dashed rounded-lg p-6 text-center transition-all cursor-pointer',
              isDragging ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/50',
              file ? 'bg-success/5 border-success' : ''
            )}
          >
            {file ? (
              <div className="flex flex-col items-center gap-2">
                <CheckCircle className="h-8 w-8 text-success" />
                <p className="font-medium text-sm">{file.name}</p>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">
                    {parsedData?.length.toLocaleString()} rows • {columns.length} columns
                  </span>
                  <Badge className={cn('text-xs', detectedFormat.color)}>
                    {detectedFormat.label}
                  </Badge>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-1">
                <Upload className="h-8 w-8 text-muted-foreground" />
                <p className="font-medium text-sm">Drop CSV file here</p>
                <p className="text-xs text-muted-foreground">or click to browse</p>
              </div>
            )}
            <Input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="absolute inset-0 opacity-0 cursor-pointer"
            />
          </div>

          {/* Error */}
          {error && (
            <Alert variant="destructive" className="py-2">
              <AlertCircle className="h-3 w-3" />
              <AlertDescription className="text-xs ml-2">{error}</AlertDescription>
            </Alert>
          )}

          {/* Detected Key Columns */}
          {keyColumns.length > 0 && (
            <div className="space-y-1.5">
              <Label className="text-xs">Key Columns Detected</Label>
              <div className="flex flex-wrap gap-1">
                {keyColumns.slice(0, 8).map((col) => (
                  <Badge key={col} variant="secondary" className="text-xs py-0">
                    {col}
                  </Badge>
                ))}
                {keyColumns.length > 8 && (
                  <Badge variant="outline" className="text-xs py-0">
                    +{keyColumns.length - 8} more
                  </Badge>
                )}
              </div>
            </div>
          )}

          {/* Column-flexible notice */}
          {parsedData && detectedFormat.type === 'generic' && (
            <Alert className="py-2 bg-muted/50 border-muted">
              <Info className="h-3 w-3" />
              <AlertTitle className="text-xs">Column-Flexible Analysis</AlertTitle>
              <AlertDescription className="text-xs">
                The system will analyze available columns automatically
              </AlertDescription>
            </Alert>
          )}

          {/* Dataset Name */}
          <div className="space-y-1">
            <Label htmlFor="name" className="text-xs">Dataset Name</Label>
            <Input
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Amazon_Reviews_2024"
              className="h-8 text-sm"
              required
            />
          </div>

          {/* Description */}
          <div className="space-y-1">
            <Label htmlFor="description" className="text-xs">Description (optional)</Label>
            <Textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Brief description..."
              className="text-sm min-h-[60px]"
              rows={2}
            />
          </div>

          {/* Submit */}
          <Button
            type="submit"
            disabled={!parsedData || uploadMutation.isPending}
            className="w-full h-9"
            size="sm"
          >
            {uploadMutation.isPending ? 'Uploading...' : 'Upload & Analyze'}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
