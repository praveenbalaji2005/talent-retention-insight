import { AlertTriangle, CheckCircle2, Info, ChevronDown } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { cn } from '@/lib/utils';
import type { Recommendation } from '@/types/dataset';
import { useState } from 'react';

interface RecommendationsPanelProps {
  recommendations: Recommendation[];
}

export function RecommendationsPanel({ recommendations }: RecommendationsPanelProps) {
  const [openItems, setOpenItems] = useState<string[]>([recommendations[0]?.id]);

  const toggleItem = (id: string) => {
    setOpenItems((prev) =>
      prev.includes(id)
        ? prev.filter((item) => item !== id)
        : [...prev, id]
    );
  };

  const getPriorityIcon = (priority: Recommendation['priority']) => {
    switch (priority) {
      case 'high':
        return <AlertTriangle className="h-5 w-5 text-destructive" />;
      case 'medium':
        return <Info className="h-5 w-5 text-warning" />;
      case 'low':
        return <CheckCircle2 className="h-5 w-5 text-success" />;
    }
  };

  const getPriorityVariant = (priority: Recommendation['priority']) => {
    switch (priority) {
      case 'high':
        return 'destructive';
      case 'medium':
        return 'secondary';
      case 'low':
        return 'outline';
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>HR Recommendations</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {recommendations.map((rec) => (
          <Collapsible
            key={rec.id}
            open={openItems.includes(rec.id)}
            onOpenChange={() => toggleItem(rec.id)}
          >
            <div
              className={cn(
                'border rounded-lg transition-all',
                openItems.includes(rec.id)
                  ? 'border-primary/30 bg-primary/5'
                  : 'border-border hover:border-primary/20'
              )}
            >
              <CollapsibleTrigger className="w-full">
                <div className="flex items-start gap-3 p-4">
                  <div className="mt-0.5">{getPriorityIcon(rec.priority)}</div>
                  <div className="flex-grow text-left">
                    <div className="flex items-center gap-2 mb-1">
                      <Badge variant={getPriorityVariant(rec.priority)} className="text-xs">
                        {rec.priority} priority
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        {rec.category}
                      </Badge>
                    </div>
                    <h4 className="font-medium">{rec.title}</h4>
                  </div>
                  <ChevronDown
                    className={cn(
                      'h-5 w-5 text-muted-foreground transition-transform',
                      openItems.includes(rec.id) && 'rotate-180'
                    )}
                  />
                </div>
              </CollapsibleTrigger>
              
              <CollapsibleContent>
                <div className="px-4 pb-4 space-y-4">
                  <p className="text-sm text-muted-foreground">{rec.description}</p>
                  
                  <div className="bg-muted/50 rounded-lg p-3">
                    <p className="text-sm font-medium text-foreground">Expected Impact</p>
                    <p className="text-sm text-muted-foreground">{rec.impact}</p>
                  </div>
                  
                  <div>
                    <p className="text-sm font-medium mb-2">Action Items</p>
                    <ul className="space-y-1">
                      {rec.action_items.map((item, index) => (
                        <li
                          key={index}
                          className="flex items-start gap-2 text-sm text-muted-foreground"
                        >
                          <span className="text-primary mt-1">•</span>
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </CollapsibleContent>
            </div>
          </Collapsible>
        ))}
      </CardContent>
    </Card>
  );
}
