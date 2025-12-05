import { Brain, BarChart3, FileSpreadsheet, Settings, Sparkles } from 'lucide-react';
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
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center justify-between">
        <div className="flex items-center gap-6">
          {/* Logo */}
          <div className="flex items-center gap-2.5 cursor-pointer" onClick={() => onViewChange('dashboard')}>
            <div className="relative flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-secondary shadow-md">
              <Sparkles className="h-4 w-4 text-primary-foreground" />
            </div>
            <div className="hidden sm:block">
              <h1 className="text-base font-semibold tracking-tight text-foreground">Attrition AI</h1>
              <p className="text-[10px] text-muted-foreground -mt-0.5">XAI-Powered Analytics</p>
            </div>
          </div>
          
          {/* Navigation */}
          <nav className="hidden md:flex items-center gap-0.5 ml-4">
            {navItems.map((item) => (
              <Button
                key={item.id}
                variant={currentView === item.id ? 'secondary' : 'ghost'}
                size="sm"
                onClick={() => onViewChange(item.id)}
                className={`h-8 px-3 text-xs font-medium gap-1.5 transition-all ${
                  currentView === item.id 
                    ? 'bg-secondary/20 text-secondary hover:bg-secondary/30' 
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                <item.icon className="h-3.5 w-3.5" />
                {item.label}
              </Button>
            ))}
          </nav>
        </div>
        
        {/* Actions */}
        <div className="flex items-center gap-1">
          <Button
            variant={currentView === 'settings' ? 'secondary' : 'ghost'}
            size="icon"
            className="h-8 w-8"
            onClick={() => onViewChange('settings')}
          >
            <Settings className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </header>
  );
}
