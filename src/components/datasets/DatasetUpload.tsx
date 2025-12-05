import { useState, useCallback } from 'react';
import { Upload, FileSpreadsheet, AlertCircle, CheckCircle } from 'lucide-react';
import Papa from 'papaparse';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { useUploadDataset } from '@/hooks/useDatasets';
import { cn } from '@/lib/utils';

interface DatasetUploadProps {
  onSuccess?: () => void;
}

export function DatasetUpload({ onSuccess }: DatasetUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [fileType, setFileType] = useState<'attrition' | 'reviews'>('attrition');
  const [parsedData, setParsedData] = useState<Record<string, unknown>[] | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const uploadMutation = useUploadDataset();

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
    if (droppedFile && droppedFile.type === 'text/csv') {
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
        file_type: fileType,
        raw_data: parsedData,
        column_names: columns,
      });
      
      // Reset form
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

  const requiredColumns = fileType === 'attrition' 
    ? ['Attrition', 'Department']
    : ['Overall_rating'];
  
  const missingColumns = requiredColumns.filter(col => !columns.includes(col));
  const hasRequiredColumns = missingColumns.length === 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Upload Dataset</CardTitle>
        <CardDescription>
          Upload a CSV file containing employee data for analysis
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* File Type Selection */}
          <div className="space-y-3">
            <Label>Dataset Type</Label>
            <RadioGroup
              value={fileType}
              onValueChange={(v) => setFileType(v as 'attrition' | 'reviews')}
              className="flex gap-4"
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="attrition" id="attrition" />
                <Label htmlFor="attrition" className="font-normal cursor-pointer">
                  Employee Attrition Data
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="reviews" id="reviews" />
                <Label htmlFor="reviews" className="font-normal cursor-pointer">
                  Company Reviews
                </Label>
              </div>
            </RadioGroup>
            <p className="text-sm text-muted-foreground">
              {fileType === 'attrition' 
                ? 'Required columns: Attrition, Department'
                : 'Required columns: Overall_rating'}
            </p>
          </div>

          {/* Drop Zone */}
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            className={cn(
              'border-2 border-dashed rounded-xl p-8 text-center transition-colors',
              isDragging ? 'border-primary bg-primary/5' : 'border-border',
              file ? 'bg-success/5 border-success' : ''
            )}
          >
            {file ? (
              <div className="flex flex-col items-center gap-2">
                <CheckCircle className="h-10 w-10 text-success" />
                <p className="font-medium">{file.name}</p>
                <p className="text-sm text-muted-foreground">
                  {parsedData?.length || 0} rows • {columns.length} columns
                </p>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-2">
                <Upload className="h-10 w-10 text-muted-foreground" />
                <p className="font-medium">Drop your CSV file here</p>
                <p className="text-sm text-muted-foreground">or click to browse</p>
              </div>
            )}
            <Input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="absolute inset-0 opacity-0 cursor-pointer"
            />
          </div>

          {/* Validation Messages */}
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {parsedData && !hasRequiredColumns && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Missing Required Columns</AlertTitle>
              <AlertDescription>
                The following columns are required: {missingColumns.join(', ')}
              </AlertDescription>
            </Alert>
          )}

          {/* Column Preview */}
          {columns.length > 0 && (
            <div className="space-y-2">
              <Label>Detected Columns</Label>
              <div className="flex flex-wrap gap-2">
                {columns.map((col) => (
                  <span
                    key={col}
                    className={cn(
                      'px-2 py-1 rounded-md text-xs font-medium',
                      requiredColumns.includes(col)
                        ? 'bg-success/20 text-success'
                        : 'bg-muted text-muted-foreground'
                    )}
                  >
                    {col}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Dataset Name */}
          <div className="space-y-2">
            <Label htmlFor="name">Dataset Name</Label>
            <Input
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Q4 2024 Employee Data"
              required
            />
          </div>

          {/* Description */}
          <div className="space-y-2">
            <Label htmlFor="description">Description (optional)</Label>
            <Textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Brief description of this dataset..."
              rows={3}
            />
          </div>

          {/* Submit */}
          <Button
            type="submit"
            disabled={!parsedData || !hasRequiredColumns || uploadMutation.isPending}
            className="w-full"
          >
            {uploadMutation.isPending ? 'Uploading...' : 'Upload Dataset'}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
