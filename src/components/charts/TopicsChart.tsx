import Plot from 'react-plotly.js';
import type { TopicData } from '@/types/dataset';

interface TopicsChartProps {
  data: TopicData[];
}

export function TopicsChart({ data }: TopicsChartProps) {
  const sortedData = [...data].sort((a, b) => b.prevalence - a.prevalence);

  const sentimentColors = {
    positive: '#22c55e',
    neutral: '#f59e0b',
    negative: '#ef4444',
  };

  const chartData = [
    {
      x: sortedData.map((d) => d.name),
      y: sortedData.map((d) => d.prevalence * 100),
      type: 'bar' as const,
      marker: {
        color: sortedData.map((d) => sentimentColors[d.sentiment]),
        line: {
          color: sortedData.map((d) => {
            const colors = {
              positive: '#16a34a',
              neutral: '#d97706',
              negative: '#dc2626',
            };
            return colors[d.sentiment];
          }),
          width: 2,
        },
      },
      hovertemplate: '%{x}<br>Prevalence: %{y:.1f}%<extra></extra>',
      text: sortedData.map((d) => `${(d.prevalence * 100).toFixed(0)}%`),
      textposition: 'outside' as const,
    },
  ];

  const layout = {
    xaxis: {
      title: '',
      tickangle: -30,
    },
    yaxis: {
      title: 'Topic Prevalence (%)',
      range: [0, Math.max(...sortedData.map((d) => d.prevalence)) * 130],
    },
    margin: { t: 40, b: 100, l: 60, r: 20 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {
      family: 'Inter, system-ui, sans-serif',
      color: '#64748b',
    },
    bargap: 0.3,
    annotations: [
      {
        x: 1,
        y: 1.05,
        xref: 'paper' as const,
        yref: 'paper' as const,
        text: '🟢 Positive  🟡 Neutral  🔴 Negative',
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
      className="w-full h-[350px]"
    />
  );
}
