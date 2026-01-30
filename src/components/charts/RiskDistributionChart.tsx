import Plot from 'react-plotly.js';
import type { RiskDistribution } from '@/types/dataset';

interface RiskDistributionChartProps {
  data: RiskDistribution;
}

export function RiskDistributionChart({ data }: RiskDistributionChartProps) {
  const total = data.low + data.early_warning + data.moderate + data.high + data.critical;
  
  const chartData = [
    {
      values: [data.low, data.early_warning, data.moderate, data.high, data.critical],
      labels: ['Low Risk', 'Early Warning', 'Moderate Risk', 'High Risk', 'Critical'],
      type: 'pie' as const,
      hole: 0.6,
      marker: {
        colors: [
          'rgba(34, 197, 94, 0.9)',   // Green - Low
          'rgba(59, 130, 246, 0.9)',  // Blue - Early Warning
          'rgba(251, 191, 36, 0.9)',  // Amber - Moderate
          'rgba(249, 115, 22, 0.9)',  // Orange - High
          'rgba(239, 68, 68, 0.9)',   // Red - Critical
        ],
        line: {
          color: '#ffffff',
          width: 2,
        },
      },
      textinfo: 'percent',
      textposition: 'outside',
      textfont: {
        size: 12,
        color: '#475569',
        family: 'Inter, system-ui, sans-serif',
      },
      hovertemplate: '<b>%{label}</b><br>%{value:,} employees<br>%{percent}<extra></extra>',
      pull: [0, 0, 0, 0.02, 0.04],
      rotation: -45,
    },
  ];

  const layout = {
    showlegend: true,
    legend: {
      orientation: 'h' as const,
      y: -0.15,
      x: 0.5,
      xanchor: 'center' as const,
      font: {
        size: 11,
        family: 'Inter, system-ui, sans-serif',
        color: '#64748b',
      },
      bgcolor: 'transparent',
    },
    margin: { t: 30, b: 80, l: 30, r: 30 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    annotations: [
      {
        text: `<b>${total.toLocaleString()}</b>`,
        showarrow: false,
        font: { size: 28, color: '#1e293b', family: 'Inter' },
        x: 0.5,
        y: 0.55,
      },
      {
        text: 'Total',
        showarrow: false,
        font: { size: 12, color: '#64748b', family: 'Inter' },
        x: 0.5,
        y: 0.42,
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
