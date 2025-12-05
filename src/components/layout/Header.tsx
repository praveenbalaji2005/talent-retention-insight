import { Brain, BarChart3, FileSpreadsheet, Settings } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface HeaderProps {
  currentView: 'dashboard' | 'datasets' | 'analysis' | 'settings';
  onViewChange: (view: 'dashboard' | 'datasets' | 'analysis' | 'settings') => void;
}

export function Header({ currentView, onViewChange }: HeaderProps) {
  const navItems = [
    { id: 'dashboard' as const, label: 'Dashboard', icon: BarChart3 },
    { id: 'datasets' as const, label: 'Datasets', icon: FileSpreadsheet },
    { id: 'analysis' as const, label: 'Analysis', icon: Brain },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/60">
      <div className="container flex h-16 items-center justify-between">
        <div className="flex items-center gap-8">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary">
              <Brain className="h-6 w-6 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-foreground">Attrition AI</h1>
              <p className="text-xs text-muted-foreground">Predictive HR Analytics</p>
            </div>
          </div>
          
          <nav className="hidden md:flex items-center gap-1">
            {navItems.map((item) => (
              <Button
                key={item.id}
                variant={currentView === item.id ? 'secondary' : 'ghost'}
                size="sm"
                onClick={() => onViewChange(item.id)}
                className="gap-2"
              >
                <item.icon className="h-4 w-4" />
                {item.label}
              </Button>
            ))}
          </nav>
        </div>
        
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => onViewChange('settings')}
          >
            <Settings className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </header>
  );
}
