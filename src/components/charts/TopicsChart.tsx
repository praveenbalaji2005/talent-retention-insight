import Plot from 'react-plotly.js';
import type { TopicData } from '@/types/dataset';

interface TopicsChartProps {
  data: TopicData[];
}

export function TopicsChart({ data }: TopicsChartProps) {
  const sortedData = [...data].sort((a, b) => b.prevalence - a.prevalence);

  const sentimentColors = {
    positive: 'rgba(34, 197, 94, 0.85)',
    neutral: 'rgba(59, 130, 246, 0.85)',
    negative: 'rgba(239, 68, 68, 0.85)',
  };

  const sentimentBorders = {
    positive: 'rgba(22, 163, 74, 1)',
    neutral: 'rgba(37, 99, 235, 1)',
    negative: 'rgba(220, 38, 38, 1)',
  };

  const chartData = [
    {
      x: sortedData.map((d) => d.name),
      y: sortedData.map((d) => d.prevalence * 100),
      type: 'bar' as const,
      marker: {
        color: sortedData.map((d) => sentimentColors[d.sentiment]),
        line: {
          color: sortedData.map((d) => sentimentBorders[d.sentiment]),
          width: 1.5,
        },
      },
      text: sortedData.map((d) => `${(d.prevalence * 100).toFixed(0)}%`),
      textposition: 'outside' as const,
      textfont: {
        size: 11,
        color: '#475569',
        family: 'Inter, system-ui, sans-serif',
      },
      hovertemplate: '<b>%{x}</b><br>Prevalence: %{y:.1f}%<br>Keywords: %{customdata}<extra></extra>',
      customdata: sortedData.map((d) => d.keywords.slice(0, 4).join(', ')),
    },
  ];

  const layout = {
    xaxis: {
      title: '',
      tickangle: -25,
      tickfont: { 
        size: 11, 
        color: '#334155',
        family: 'Inter, system-ui, sans-serif',
      },
      gridcolor: 'rgba(226, 232, 240, 0.5)',
      showgrid: false,
    },
    yaxis: {
      title: {
        text: 'Topic Prevalence (%)',
        font: { size: 11, color: '#64748b', family: 'Inter, system-ui, sans-serif' },
        standoff: 10,
      },
      range: [0, Math.max(...sortedData.map((d) => d.prevalence)) * 140],
      tickfont: { size: 10, color: '#64748b' },
      gridcolor: 'rgba(226, 232, 240, 0.8)',
      zeroline: false,
    },
    margin: { t: 50, b: 100, l: 55, r: 20 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {
      family: 'Inter, system-ui, sans-serif',
      color: '#64748b',
    },
    bargap: 0.35,
    annotations: [
      {
        x: 1,
        y: 1.12,
        xref: 'paper' as const,
        yref: 'paper' as const,
        text: '<span style="color:#22c55e">●</span> Positive  <span style="color:#3b82f6">●</span> Neutral  <span style="color:#ef4444">●</span> Negative',
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
      className="w-full h-[360px]"
    />
  );
}
