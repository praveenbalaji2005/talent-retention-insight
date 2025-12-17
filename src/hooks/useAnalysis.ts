import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { supabase } from '@/integrations/supabase/client';
import type { AnalysisResult, Dataset } from '@/types/dataset';
import { toast } from 'sonner';

export function useAnalysisResults(datasetId: string | undefined) {
  return useQuery({
    queryKey: ['analysis', datasetId],
    queryFn: async (): Promise<AnalysisResult[]> => {
      if (!datasetId) return [];
      
      const { data, error } = await supabase
        .from('analysis_results')
        .select('*')
        .eq('dataset_id', datasetId)
        .order('created_at', { ascending: false });
      
      if (error) throw error;
      return (data || []) as unknown as AnalysisResult[];
    },
    enabled: !!datasetId,
  });
}

export function useLatestAnalysis(datasetId: string | undefined) {
  return useQuery({
    queryKey: ['analysis', datasetId, 'latest'],
    queryFn: async (): Promise<AnalysisResult | null> => {
      if (!datasetId) return null;
      
      const { data, error } = await supabase
        .from('analysis_results')
        .select('*')
        .eq('dataset_id', datasetId)
        .eq('status', 'completed')
        .order('created_at', { ascending: false })
        .limit(1)
        .single();
      
      if (error && error.code !== 'PGRST116') throw error;
      return data as unknown as AnalysisResult;
    },
    enabled: !!datasetId,
  });
}

export function useRunAnalysis() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (dataset: Dataset) => {
      // Create pending analysis record
      const { data: analysisRecord, error: insertError } = await supabase
        .from('analysis_results')
        .insert({
          dataset_id: dataset.id,
          analysis_type: 'full_pipeline',
          status: 'processing',
        })
        .select()
        .single();
      
      if (insertError) throw insertError;
      
      try {
        // Call ML backend edge function
        const { data: analysisData, error: fnError } = await supabase.functions.invoke('analyze-dataset', {
          body: { raw_data: dataset.raw_data }
        });
        
        if (fnError) throw fnError;
        if (!analysisData.success) throw new Error(analysisData.error || 'Analysis failed');
        
        // Update with results
        const { data: updatedRecord, error: updateError } = await supabase
          .from('analysis_results')
          .update({
            status: 'completed',
            results: JSON.parse(JSON.stringify(analysisData.results)),
            predictions: JSON.parse(JSON.stringify(analysisData.predictions)),
            feature_importance: JSON.parse(JSON.stringify(analysisData.feature_importance)),
            topics: JSON.parse(JSON.stringify(analysisData.topics)),
            recommendations: JSON.parse(JSON.stringify(analysisData.recommendations)),
            completed_at: new Date().toISOString(),
          })
          .eq('id', analysisRecord.id)
          .select()
          .single();
        
        if (updateError) throw updateError;
        return updatedRecord as unknown as AnalysisResult;
        
      } catch (error) {
        // Mark as failed
        await supabase
          .from('analysis_results')
          .update({
            status: 'failed',
            error_message: error instanceof Error ? error.message : 'Unknown error',
          })
          .eq('id', analysisRecord.id);
        
        throw error;
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['analysis'] });
      toast.success('Analysis completed successfully');
    },
    onError: (error: Error) => {
      toast.error(`Analysis failed: ${error.message}`);
    },
  });
}
