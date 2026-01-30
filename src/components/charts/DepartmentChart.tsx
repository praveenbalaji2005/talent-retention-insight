import Plot from 'react-plotly.js';
import type { DepartmentBreakdown } from '@/types/dataset';

interface DepartmentChartProps {
  data: DepartmentBreakdown[];
}

export function DepartmentChart({ data }: DepartmentChartProps) {
  const sortedData = [...data].sort((a, b) => b.rate - a.rate).slice(0, 10);

  // Professional gradient-like colors based on risk level
  const getBarColor = (rate: number) => {
    if (rate >= 40) return 'rgba(239, 68, 68, 0.85)';   // Critical - Red
    if (rate >= 30) return 'rgba(249, 115, 22, 0.85)';  // High - Orange
    if (rate >= 20) return 'rgba(251, 191, 36, 0.85)';  // Moderate - Amber
    if (rate >= 10) return 'rgba(59, 130, 246, 0.85)';  // Early Warning - Blue
    return 'rgba(34, 197, 94, 0.85)';                   // Low - Green
  };

  const getBorderColor = (rate: number) => {
    if (rate >= 40) return 'rgba(220, 38, 38, 1)';
    if (rate >= 30) return 'rgba(234, 88, 12, 1)';
    if (rate >= 20) return 'rgba(217, 119, 6, 1)';
    if (rate >= 10) return 'rgba(37, 99, 235, 1)';
    return 'rgba(22, 163, 74, 1)';
  };

  const chartData = [
    {
      x: sortedData.map((d) => d.department.length > 20 ? d.department.slice(0, 18) + '...' : d.department),
      y: sortedData.map((d) => d.rate),
      type: 'bar' as const,
      marker: {
        color: sortedData.map((d) => getBarColor(d.rate)),
        line: {
          color: sortedData.map((d) => getBorderColor(d.rate)),
          width: 1.5,
        },
      },
      text: sortedData.map((d) => `${d.rate}%`),
      textposition: 'outside' as const,
      textfont: {
        size: 11,
        color: '#475569',
        family: 'Inter, system-ui, sans-serif',
      },
      hovertemplate: '<b>%{x}</b><br>Risk Rate: %{y}%<br>At Risk: %{customdata[0]} of %{customdata[1]}<extra></extra>',
      customdata: sortedData.map((d) => [d.at_risk, d.total]),
    },
  ];

  const layout = {
    xaxis: {
      title: '',
      tickangle: -35,
      tickfont: { 
        size: 10, 
        color: '#64748b',
        family: 'Inter, system-ui, sans-serif',
      },
      gridcolor: 'rgba(226, 232, 240, 0.5)',
      showgrid: false,
    },
    yaxis: {
      title: {
        text: 'Attrition Risk (%)',
        font: { size: 11, color: '#64748b', family: 'Inter, system-ui, sans-serif' },
        standoff: 10,
      },
      range: [0, Math.max(...sortedData.map((d) => d.rate)) * 1.25],
      tickfont: { size: 10, color: '#64748b' },
      gridcolor: 'rgba(226, 232, 240, 0.8)',
      zeroline: false,
    },
    margin: { t: 25, b: 120, l: 55, r: 20 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {
      family: 'Inter, system-ui, sans-serif',
      color: '#64748b',
    },
    bargap: 0.35,
    shapes: [
      {
        type: 'line' as const,
        x0: -0.5,
        x1: sortedData.length - 0.5,
        y0: 25,
        y1: 25,
        line: {
          color: 'rgba(239, 68, 68, 0.4)',
          width: 1.5,
          dash: 'dot' as const,
        },
      },
    ],
    annotations: sortedData.length > 0 ? [
      {
        x: sortedData.length - 1,
        y: 25,
        text: 'High Risk Threshold',
        showarrow: false,
        font: { size: 9, color: '#ef4444' },
        xanchor: 'right' as const,
        yanchor: 'bottom' as const,
        yshift: 3,
      },
    ] : [],
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
