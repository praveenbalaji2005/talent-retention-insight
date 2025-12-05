import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { supabase } from '@/integrations/supabase/client';
import type { Dataset } from '@/types/dataset';
import { toast } from 'sonner';

export function useDatasets() {
  return useQuery({
    queryKey: ['datasets'],
    queryFn: async (): Promise<Dataset[]> => {
      const { data, error } = await supabase
        .from('datasets')
        .select('*')
        .order('created_at', { ascending: false });
      
      if (error) throw error;
      return (data || []) as Dataset[];
    },
  });
}

export function useDataset(id: string | undefined) {
  return useQuery({
    queryKey: ['dataset', id],
    queryFn: async (): Promise<Dataset | null> => {
      if (!id) return null;
      
      const { data, error } = await supabase
        .from('datasets')
        .select('*')
        .eq('id', id)
        .single();
      
      if (error) throw error;
      return data as Dataset;
    },
    enabled: !!id,
  });
}

export function useUploadDataset() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (params: {
      name: string;
      description?: string;
      file_type: 'attrition' | 'reviews';
      raw_data: Record<string, unknown>[];
      column_names: string[];
    }) => {
      const { data, error } = await supabase
        .from('datasets')
        .insert([{
          name: params.name,
          description: params.description || null,
          file_type: params.file_type,
          raw_data: params.raw_data as unknown as Record<string, unknown>,
          column_names: params.column_names,
          row_count: params.raw_data.length,
        }])
        .select()
        .single();
      
      if (error) throw error;
      return data as Dataset;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      toast.success('Dataset uploaded successfully');
    },
    onError: (error: Error) => {
      toast.error(`Upload failed: ${error.message}`);
    },
  });
}

export function useDeleteDataset() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (id: string) => {
      const { error } = await supabase
        .from('datasets')
        .delete()
        .eq('id', id);
      
      if (error) throw error;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      toast.success('Dataset deleted');
    },
    onError: (error: Error) => {
      toast.error(`Delete failed: ${error.message}`);
    },
  });
}
