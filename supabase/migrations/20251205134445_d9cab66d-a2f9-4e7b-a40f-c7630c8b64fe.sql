-- Create datasets table for storing uploaded CSV data
CREATE TABLE public.datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT,
    file_type TEXT NOT NULL CHECK (file_type IN ('attrition', 'reviews')),
    raw_data JSONB NOT NULL,
    column_names TEXT[] NOT NULL,
    row_count INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Create analysis_results table for storing ML pipeline outputs
CREATE TABLE public.analysis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID REFERENCES public.datasets(id) ON DELETE CASCADE NOT NULL,
    analysis_type TEXT NOT NULL CHECK (analysis_type IN ('attrition_prediction', 'shap_importance', 'topic_modeling', 'full_pipeline')),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    results JSONB,
    predictions JSONB,
    feature_importance JSONB,
    topics JSONB,
    recommendations JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for better query performance
CREATE INDEX idx_datasets_file_type ON public.datasets(file_type);
CREATE INDEX idx_datasets_created_at ON public.datasets(created_at DESC);
CREATE INDEX idx_analysis_results_dataset_id ON public.analysis_results(dataset_id);
CREATE INDEX idx_analysis_results_status ON public.analysis_results(status);

-- Enable RLS
ALTER TABLE public.datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_results ENABLE ROW LEVEL SECURITY;

-- Public read/write policies for prototype (no auth required)
CREATE POLICY "Allow public read on datasets" ON public.datasets FOR SELECT USING (true);
CREATE POLICY "Allow public insert on datasets" ON public.datasets FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public update on datasets" ON public.datasets FOR UPDATE USING (true);
CREATE POLICY "Allow public delete on datasets" ON public.datasets FOR DELETE USING (true);

CREATE POLICY "Allow public read on analysis_results" ON public.analysis_results FOR SELECT USING (true);
CREATE POLICY "Allow public insert on analysis_results" ON public.analysis_results FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public update on analysis_results" ON public.analysis_results FOR UPDATE USING (true);
CREATE POLICY "Allow public delete on analysis_results" ON public.analysis_results FOR DELETE USING (true);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for datasets
CREATE TRIGGER update_datasets_updated_at
    BEFORE UPDATE ON public.datasets
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();