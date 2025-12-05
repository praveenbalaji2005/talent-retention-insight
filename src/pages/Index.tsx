import { useState } from 'react';
import { Header } from '@/components/layout/Header';
import { Dashboard } from '@/components/dashboard/Dashboard';
import { DatasetsView } from '@/components/datasets/DatasetsView';
import { AnalysisView } from '@/components/analysis/AnalysisView';
import { SettingsView } from '@/components/settings/SettingsView';
import type { Dataset } from '@/types/dataset';

type View = 'dashboard' | 'datasets' | 'analysis' | 'settings';

const Index = () => {
  const [currentView, setCurrentView] = useState<View>('dashboard');

  const handleViewChange = (view: View) => {
    setCurrentView(view);
  };

  const handleSelectDataset = (_dataset: Dataset) => {
    setCurrentView('analysis');
  };

  const renderView = () => {
    switch (currentView) {
      case 'dashboard':
        return <Dashboard onNavigate={handleViewChange} />;
      case 'datasets':
        return <DatasetsView onSelectDataset={handleSelectDataset} />;
      case 'analysis':
        return <AnalysisView />;
      case 'settings':
        return <SettingsView />;
      default:
        return <Dashboard onNavigate={handleViewChange} />;
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Header currentView={currentView} onViewChange={handleViewChange} />
      <main className="container mx-auto px-4 py-6">
        {renderView()}
      </main>
    </div>
  );
};

export default Index;
