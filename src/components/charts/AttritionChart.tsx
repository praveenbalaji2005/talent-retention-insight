import Plot from 'react-plotly.js';
import type { PredictionData } from '@/types/dataset';

interface AttritionChartProps {
  data: PredictionData[];
}

export function AttritionChart({ data }: AttritionChartProps) {
  const riskScores = data.map((d) => d.risk_score);
  
  const chartData = [
    {
      x: riskScores,
      type: 'histogram' as const,
      marker: {
        color: '#1e4a5f',
        line: {
          color: '#0f2f3d',
          width: 1,
        },
      },
      nbinsx: 20,
      hovertemplate: 'Risk Score: %{x}<br>Count: %{y}<extra></extra>',
    },
  ];

  const layout = {
    xaxis: {
      title: 'Risk Score',
      range: [0, 100],
    },
    yaxis: {
      title: 'Employee Count',
    },
    margin: { t: 20, b: 60, l: 60, r: 20 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {
      family: 'Inter, system-ui, sans-serif',
      color: '#64748b',
    },
    bargap: 0.05,
    shapes: [
      {
        type: 'line' as const,
        x0: 75,
        x1: 75,
        y0: 0,
        y1: 1,
        yref: 'paper' as const,
        line: {
          color: '#ef4444',
          width: 2,
          dash: 'dash' as const,
        },
      },
    ],
    annotations: [
      {
        x: 75,
        y: 1,
        yref: 'paper' as const,
        text: 'High Risk Threshold',
        showarrow: false,
        font: { size: 10, color: '#ef4444' },
        xanchor: 'left' as const,
        yanchor: 'bottom' as const,
      },
    ],
  };

  return (
    <Plot
      data={chartData}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      className="w-full h-[300px]"
    />
  );
}
