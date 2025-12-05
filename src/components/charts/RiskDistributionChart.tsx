import Plot from 'react-plotly.js';
import type { RiskDistribution } from '@/types/dataset';

interface RiskDistributionChartProps {
  data: RiskDistribution;
}

export function RiskDistributionChart({ data }: RiskDistributionChartProps) {
  const chartData = [
    {
      values: [data.low, data.medium, data.high, data.critical],
      labels: ['Low Risk', 'Medium Risk', 'High Risk', 'Critical'],
      type: 'pie' as const,
      hole: 0.5,
      marker: {
        colors: ['#22c55e', '#f59e0b', '#ef4444', '#991b1b'],
      },
      textinfo: 'label+percent',
      textposition: 'outside',
      hovertemplate: '%{label}<br>%{value} employees<br>%{percent}<extra></extra>',
    },
  ];

  const layout = {
    showlegend: true,
    legend: {
      orientation: 'h' as const,
      y: -0.1,
      x: 0.5,
      xanchor: 'center' as const,
    },
    margin: { t: 20, b: 60, l: 20, r: 20 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {
      family: 'Inter, system-ui, sans-serif',
      color: '#64748b',
    },
    annotations: [
      {
        text: `${data.low + data.medium + data.high + data.critical}`,
        showarrow: false,
        font: { size: 24, color: '#1e293b', family: 'Inter' },
        x: 0.5,
        y: 0.5,
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
