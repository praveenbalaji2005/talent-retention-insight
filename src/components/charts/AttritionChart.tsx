import Plot from 'react-plotly.js';
import type { PredictionData } from '@/types/dataset';

interface AttritionChartProps {
  data: PredictionData[];
}

export function AttritionChart({ data }: AttritionChartProps) {
  const riskScores = data.map((d) => d.risk_score);
  
  // Create bins for the histogram with gradient colors
  const chartData = [
    {
      x: riskScores,
      type: 'histogram' as const,
      marker: {
        color: riskScores.map(score => {
          if (score >= 80) return 'rgba(239, 68, 68, 0.85)';
          if (score >= 60) return 'rgba(249, 115, 22, 0.85)';
          if (score >= 40) return 'rgba(251, 191, 36, 0.85)';
          if (score >= 20) return 'rgba(59, 130, 246, 0.85)';
          return 'rgba(34, 197, 94, 0.85)';
        }),
        line: {
          color: 'rgba(255, 255, 255, 0.8)',
          width: 0.5,
        },
      },
      nbinsx: 25,
      hovertemplate: 'Risk Score: %{x}<br>Employees: %{y}<extra></extra>',
    },
  ];

  const layout = {
    xaxis: {
      title: {
        text: 'Risk Score',
        font: { size: 11, color: '#64748b', family: 'Inter, system-ui, sans-serif' },
      },
      range: [0, 100],
      tickfont: { size: 10, color: '#64748b' },
      gridcolor: 'rgba(226, 232, 240, 0.5)',
      dtick: 20,
    },
    yaxis: {
      title: {
        text: 'Employee Count',
        font: { size: 11, color: '#64748b', family: 'Inter, system-ui, sans-serif' },
      },
      tickfont: { size: 10, color: '#64748b' },
      gridcolor: 'rgba(226, 232, 240, 0.8)',
      zeroline: false,
    },
    margin: { t: 30, b: 60, l: 60, r: 30 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {
      family: 'Inter, system-ui, sans-serif',
      color: '#64748b',
    },
    bargap: 0.08,
    shapes: [
      // Low zone
      {
        type: 'rect' as const,
        x0: 0, x1: 20, y0: 0, y1: 1,
        yref: 'paper' as const,
        fillcolor: 'rgba(34, 197, 94, 0.08)',
        line: { width: 0 },
        layer: 'below' as const,
      },
      // Early Warning zone
      {
        type: 'rect' as const,
        x0: 20, x1: 40, y0: 0, y1: 1,
        yref: 'paper' as const,
        fillcolor: 'rgba(59, 130, 246, 0.08)',
        line: { width: 0 },
        layer: 'below' as const,
      },
      // Moderate zone
      {
        type: 'rect' as const,
        x0: 40, x1: 60, y0: 0, y1: 1,
        yref: 'paper' as const,
        fillcolor: 'rgba(251, 191, 36, 0.08)',
        line: { width: 0 },
        layer: 'below' as const,
      },
      // High zone
      {
        type: 'rect' as const,
        x0: 60, x1: 80, y0: 0, y1: 1,
        yref: 'paper' as const,
        fillcolor: 'rgba(249, 115, 22, 0.08)',
        line: { width: 0 },
        layer: 'below' as const,
      },
      // Critical zone
      {
        type: 'rect' as const,
        x0: 80, x1: 100, y0: 0, y1: 1,
        yref: 'paper' as const,
        fillcolor: 'rgba(239, 68, 68, 0.12)',
        line: { width: 0 },
        layer: 'below' as const,
      },
      // Critical threshold line
      {
        type: 'line' as const,
        x0: 80, x1: 80, y0: 0, y1: 1,
        yref: 'paper' as const,
        line: {
          color: 'rgba(239, 68, 68, 0.7)',
          width: 2,
          dash: 'dash' as const,
        },
      },
    ],
    annotations: [
      {
        x: 10, y: 1.02,
        yref: 'paper' as const,
        text: '<b>Low</b>',
        showarrow: false,
        font: { size: 9, color: '#22c55e' },
        xanchor: 'center' as const,
      },
      {
        x: 30, y: 1.02,
        yref: 'paper' as const,
        text: '<b>Early</b>',
        showarrow: false,
        font: { size: 9, color: '#3b82f6' },
        xanchor: 'center' as const,
      },
      {
        x: 50, y: 1.02,
        yref: 'paper' as const,
        text: '<b>Moderate</b>',
        showarrow: false,
        font: { size: 9, color: '#f59e0b' },
        xanchor: 'center' as const,
      },
      {
        x: 70, y: 1.02,
        yref: 'paper' as const,
        text: '<b>High</b>',
        showarrow: false,
        font: { size: 9, color: '#f97316' },
        xanchor: 'center' as const,
      },
      {
        x: 90, y: 1.02,
        yref: 'paper' as const,
        text: '<b>Critical</b>',
        showarrow: false,
        font: { size: 9, color: '#ef4444' },
        xanchor: 'center' as const,
      },
    ],
  };

  return (
    <Plot
      data={chartData}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      className="w-full h-[320px]"
    />
  );
}
