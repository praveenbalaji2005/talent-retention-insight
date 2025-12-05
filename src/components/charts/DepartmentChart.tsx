import Plot from 'react-plotly.js';
import type { DepartmentBreakdown } from '@/types/dataset';

interface DepartmentChartProps {
  data: DepartmentBreakdown[];
}

export function DepartmentChart({ data }: DepartmentChartProps) {
  const sortedData = [...data].sort((a, b) => b.rate - a.rate);

  const chartData = [
    {
      x: sortedData.map((d) => d.department),
      y: sortedData.map((d) => d.rate),
      type: 'bar' as const,
      marker: {
        color: sortedData.map((d) => 
          d.rate > 25 ? '#ef4444' : d.rate > 15 ? '#f59e0b' : '#22c55e'
        ),
        line: {
          color: sortedData.map((d) => 
            d.rate > 25 ? '#dc2626' : d.rate > 15 ? '#d97706' : '#16a34a'
          ),
          width: 2,
        },
      },
      hovertemplate: '%{x}<br>Risk Rate: %{y}%<extra></extra>',
    },
  ];

  const layout = {
    xaxis: {
      title: '',
      tickangle: -45,
      tickfont: { size: 11 },
    },
    yaxis: {
      title: 'Attrition Risk (%)',
      range: [0, Math.max(...sortedData.map((d) => d.rate)) * 1.2],
    },
    margin: { t: 20, b: 100, l: 60, r: 20 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {
      family: 'Inter, system-ui, sans-serif',
      color: '#64748b',
    },
    bargap: 0.3,
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
