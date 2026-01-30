import Plot from 'react-plotly.js';
import type { FeatureImportance } from '@/types/dataset';

interface FeatureImportanceChartProps {
  data: FeatureImportance[];
}

export function FeatureImportanceChart({ data }: FeatureImportanceChartProps) {
  const sortedData = [...data].sort((a, b) => a.importance - b.importance);

  const chartData = [
    {
      y: sortedData.map((d) => d.feature),
      x: sortedData.map((d) => d.importance * 100),
      type: 'bar' as const,
      orientation: 'h' as const,
      marker: {
        color: sortedData.map((d) => 
          d.direction === 'positive' 
            ? 'rgba(239, 68, 68, 0.85)'  // Red for risk-increasing factors
            : 'rgba(34, 197, 94, 0.85)'   // Green for risk-decreasing factors
        ),
        line: {
          color: sortedData.map((d) => 
            d.direction === 'positive' 
              ? 'rgba(220, 38, 38, 1)' 
              : 'rgba(22, 163, 74, 1)'
          ),
          width: 1,
        },
      },
      hovertemplate: '<b>%{y}</b><br>SHAP Importance: %{x:.1f}%<br>%{customdata}<extra></extra>',
      customdata: sortedData.map((d) => d.description || ''),
      text: sortedData.map((d) => `${(d.importance * 100).toFixed(1)}%`),
      textposition: 'outside' as const,
      textfont: { 
        size: 10, 
        color: '#475569',
        family: 'Inter, system-ui, sans-serif',
      },
      cliponaxis: false,
    },
  ];

  const layout = {
    xaxis: {
      title: {
        text: 'SHAP Feature Importance (%)',
        font: { size: 11, color: '#64748b', family: 'Inter, system-ui, sans-serif' },
      },
      range: [0, Math.max(...sortedData.map((d) => d.importance)) * 125],
      tickfont: { size: 10, color: '#64748b' },
      gridcolor: 'rgba(226, 232, 240, 0.8)',
      zeroline: false,
    },
    yaxis: {
      title: '',
      automargin: true,
      tickfont: { 
        size: 11, 
        color: '#334155',
        family: 'Inter, system-ui, sans-serif',
      },
    },
    margin: { t: 40, b: 50, l: 140, r: 70 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {
      family: 'Inter, system-ui, sans-serif',
      color: '#64748b',
    },
    bargap: 0.25,
    annotations: [
      {
        x: 1,
        y: 1.08,
        xref: 'paper' as const,
        yref: 'paper' as const,
        text: '<span style="color:#ef4444">●</span> Increases Risk  <span style="color:#22c55e">●</span> Decreases Risk',
        showarrow: false,
        font: { size: 10, family: 'Inter, system-ui, sans-serif' },
        xanchor: 'right' as const,
      },
    ],
  };

  return (
    <Plot
      data={chartData}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      className="w-full h-[400px]"
    />
  );
}
