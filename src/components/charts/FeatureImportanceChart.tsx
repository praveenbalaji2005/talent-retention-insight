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
          d.direction === 'positive' ? '#ef4444' : '#22c55e'
        ),
        line: {
          color: sortedData.map((d) => 
            d.direction === 'positive' ? '#dc2626' : '#16a34a'
          ),
          width: 1,
        },
      },
      hovertemplate: '%{y}<br>Importance: %{x:.1f}%<extra></extra>',
      text: sortedData.map((d) => `${(d.importance * 100).toFixed(1)}%`),
      textposition: 'outside' as const,
      textfont: { size: 10 },
    },
  ];

  const layout = {
    xaxis: {
      title: 'Feature Importance (%)',
      range: [0, Math.max(...sortedData.map((d) => d.importance)) * 120],
    },
    yaxis: {
      title: '',
      automargin: true,
    },
    margin: { t: 20, b: 60, l: 150, r: 60 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {
      family: 'Inter, system-ui, sans-serif',
      color: '#64748b',
    },
    bargap: 0.2,
    annotations: [
      {
        x: 1,
        y: 1.1,
        xref: 'paper' as const,
        yref: 'paper' as const,
        text: '🔴 Increases Risk  🟢 Decreases Risk',
        showarrow: false,
        font: { size: 11 },
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
