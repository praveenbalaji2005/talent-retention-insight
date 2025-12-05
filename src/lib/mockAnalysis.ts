import type { 
  AnalysisResults, 
  PredictionData, 
  FeatureImportance, 
  TopicData, 
  Recommendation,
  DepartmentBreakdown 
} from '@/types/dataset';

// Simulates ML analysis for the prototype
export function generateMockPredictions(data: Record<string, unknown>[]): PredictionData[] {
  return data.map((row, index) => {
    const riskScore = Math.random();
    const riskLevel = riskScore < 0.25 ? 'low' : riskScore < 0.5 ? 'medium' : riskScore < 0.75 ? 'high' : 'critical';
    
    const factors: string[] = [];
    if (riskScore > 0.5) factors.push('Low satisfaction score');
    if (riskScore > 0.6) factors.push('Long commute distance');
    if (riskScore > 0.7) factors.push('No recent promotion');
    if (riskScore > 0.8) factors.push('High overtime hours');
    
    const employeeId = row.EmployeeID || row.employee_id || index + 1;
    
    return {
      employee_id: typeof employeeId === 'string' || typeof employeeId === 'number' ? employeeId : index + 1,
      department: String(row.Department || 'Unknown'),
      risk_score: Math.round(riskScore * 100),
      risk_level: riskLevel,
      attrition_probability: riskScore,
      factors: factors.length > 0 ? factors : ['No significant risk factors'],
    };
  });
}

export function generateMockResults(data: Record<string, unknown>[], predictions: PredictionData[]): AnalysisResults {
  const departments = [...new Set(predictions.map(p => p.department))];
  
  const atRiskCount = predictions.filter(p => p.risk_level === 'high' || p.risk_level === 'critical').length;
  
  const departmentBreakdown: DepartmentBreakdown[] = departments.map(dept => {
    const deptPredictions = predictions.filter(p => p.department === dept);
    const deptAtRisk = deptPredictions.filter(p => p.risk_level === 'high' || p.risk_level === 'critical').length;
    
    return {
      department: dept,
      total: deptPredictions.length,
      at_risk: deptAtRisk,
      rate: Math.round((deptAtRisk / deptPredictions.length) * 100),
    };
  });
  
  const riskDistribution = {
    low: predictions.filter(p => p.risk_level === 'low').length,
    medium: predictions.filter(p => p.risk_level === 'medium').length,
    high: predictions.filter(p => p.risk_level === 'high').length,
    critical: predictions.filter(p => p.risk_level === 'critical').length,
  };
  
  return {
    total_employees: data.length,
    at_risk_count: atRiskCount,
    attrition_rate: Math.round((atRiskCount / data.length) * 100),
    department_breakdown: departmentBreakdown,
    risk_distribution: riskDistribution,
  };
}

export function generateMockFeatureImportance(): FeatureImportance[] {
  return [
    { feature: 'Job Satisfaction', importance: 0.28, direction: 'negative', description: 'Lower satisfaction correlates with higher attrition risk' },
    { feature: 'Monthly Income', importance: 0.22, direction: 'negative', description: 'Below-market compensation increases departure likelihood' },
    { feature: 'Years at Company', importance: 0.18, direction: 'negative', description: 'Newer employees have higher turnover rates' },
    { feature: 'Overtime', importance: 0.15, direction: 'positive', description: 'Excessive overtime drives burnout and departure' },
    { feature: 'Work-Life Balance', importance: 0.12, direction: 'negative', description: 'Poor balance is a key departure factor' },
    { feature: 'Distance from Home', importance: 0.08, direction: 'positive', description: 'Long commutes contribute to attrition' },
    { feature: 'Years Since Promotion', importance: 0.07, direction: 'positive', description: 'Stalled career growth increases risk' },
    { feature: 'Training Times', importance: 0.05, direction: 'negative', description: 'Investment in development improves retention' },
    { feature: 'Environment Satisfaction', importance: 0.04, direction: 'negative', description: 'Workplace environment affects retention' },
    { feature: 'Age', importance: 0.03, direction: 'negative', description: 'Younger employees show higher mobility' },
  ];
}

export function generateMockTopics(): TopicData[] {
  return [
    {
      topic_id: 1,
      name: 'Work-Life Balance',
      keywords: ['hours', 'overtime', 'flexible', 'remote', 'balance'],
      prevalence: 0.28,
      sentiment: 'negative',
      sample_reviews: ['Long hours expected with little flexibility', 'Work-life balance could be improved significantly'],
    },
    {
      topic_id: 2,
      name: 'Career Growth',
      keywords: ['promotion', 'growth', 'opportunity', 'learning', 'career'],
      prevalence: 0.24,
      sentiment: 'neutral',
      sample_reviews: ['Good learning opportunities but slow promotions', 'Career path is unclear for many roles'],
    },
    {
      topic_id: 3,
      name: 'Compensation & Benefits',
      keywords: ['salary', 'pay', 'benefits', 'bonus', 'competitive'],
      prevalence: 0.20,
      sentiment: 'negative',
      sample_reviews: ['Salary below market rate for the industry', 'Benefits package is decent but not competitive'],
    },
    {
      topic_id: 4,
      name: 'Management & Leadership',
      keywords: ['manager', 'leadership', 'support', 'communication', 'direction'],
      prevalence: 0.16,
      sentiment: 'positive',
      sample_reviews: ['Great managers who support their teams', 'Leadership is approachable and transparent'],
    },
    {
      topic_id: 5,
      name: 'Company Culture',
      keywords: ['culture', 'team', 'environment', 'colleagues', 'collaborative'],
      prevalence: 0.12,
      sentiment: 'positive',
      sample_reviews: ['Amazing team culture and collaboration', 'Colleagues are friendly and supportive'],
    },
  ];
}

export function generateMockRecommendations(results: AnalysisResults, featureImportance: FeatureImportance[]): Recommendation[] {
  const recommendations: Recommendation[] = [];
  
  if (featureImportance[0]?.feature === 'Job Satisfaction') {
    recommendations.push({
      id: '1',
      priority: 'high',
      category: 'Employee Engagement',
      title: 'Implement Regular Satisfaction Surveys',
      description: 'Job satisfaction is the top predictor of attrition. Establish quarterly pulse surveys to identify issues early.',
      impact: `Could reduce attrition by up to ${Math.round(featureImportance[0].importance * 100)}%`,
      action_items: ['Deploy anonymous quarterly surveys', 'Create action plans for low-scoring areas', 'Share results and improvements with employees'],
    });
  }
  
  if (results.attrition_rate > 15) {
    recommendations.push({
      id: '2',
      priority: 'high',
      category: 'Retention Strategy',
      title: 'Launch Targeted Retention Program',
      description: `With ${results.attrition_rate}% attrition risk, implement targeted interventions for high-risk employees.`,
      impact: 'Expected 20-30% reduction in voluntary turnover',
      action_items: ['Identify top 10% at-risk employees', 'Schedule stay interviews with managers', 'Develop personalized retention offers'],
    });
  }
  
  const highRiskDepts = results.department_breakdown.filter(d => d.rate > 20);
  if (highRiskDepts.length > 0) {
    recommendations.push({
      id: '3',
      priority: 'medium',
      category: 'Department Focus',
      title: `Address High-Risk Departments: ${highRiskDepts.map(d => d.department).join(', ')}`,
      description: 'These departments show significantly higher attrition risk than average.',
      impact: 'Targeted intervention could save significant replacement costs',
      action_items: ['Conduct department-specific focus groups', 'Review compensation vs market rates', 'Assess management effectiveness'],
    });
  }
  
  recommendations.push({
    id: '4',
    priority: 'medium',
    category: 'Compensation',
    title: 'Review Compensation Competitiveness',
    description: 'Monthly income ranks high in attrition prediction. Ensure pay scales are competitive.',
    impact: 'Market-aligned pay reduces turnover by 15-25%',
    action_items: ['Conduct market compensation analysis', 'Identify below-market positions', 'Create adjustment roadmap'],
  });
  
  recommendations.push({
    id: '5',
    priority: 'low',
    category: 'Development',
    title: 'Enhance Career Development Programs',
    description: 'Years since promotion correlates with attrition. Create clearer growth paths.',
    impact: 'Improved career visibility increases engagement',
    action_items: ['Define clear promotion criteria', 'Implement mentorship programs', 'Create individual development plans'],
  });
  
  return recommendations;
}

export async function runMockAnalysis(data: Record<string, unknown>[]): Promise<{
  results: AnalysisResults;
  predictions: PredictionData[];
  feature_importance: FeatureImportance[];
  topics: TopicData[];
  recommendations: Recommendation[];
}> {
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  const predictions = generateMockPredictions(data);
  const results = generateMockResults(data, predictions);
  const feature_importance = generateMockFeatureImportance();
  const topics = generateMockTopics();
  const recommendations = generateMockRecommendations(results, feature_importance);
  
  return { results, predictions, feature_importance, topics, recommendations };
}
