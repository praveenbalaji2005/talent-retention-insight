import { useState } from 'react';
import { Plus } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogTrigger } from '@/components/ui/dialog';
import { DatasetUpload } from './DatasetUpload';
import { DatasetList } from './DatasetList';
import type { Dataset } from '@/types/dataset';

interface DatasetsViewProps {
  onSelectDataset?: (dataset: Dataset) => void;
}

export function DatasetsView({ onSelectDataset }: DatasetsViewProps) {
  const [uploadOpen, setUploadOpen] = useState(false);

  return (
    <div className="container py-8 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Datasets</h2>
          <p className="text-muted-foreground">
            Manage your employee data for attrition analysis
          </p>
        </div>
        
        <Dialog open={uploadOpen} onOpenChange={setUploadOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Upload Dataset
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
            <DatasetUpload onSuccess={() => setUploadOpen(false)} />
          </DialogContent>
        </Dialog>
      </div>

      <DatasetList onSelectDataset={onSelectDataset} />
    </div>
  );
}
