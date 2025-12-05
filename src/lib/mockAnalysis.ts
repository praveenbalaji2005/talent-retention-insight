import type { 
  AnalysisResults, 
  PredictionData, 
  FeatureImportance, 
  TopicData, 
  Recommendation,
  DepartmentBreakdown 
} from '@/types/dataset';

// Column mappings based on the research paper (IBM and Kaggle datasets)
const IBM_COLUMNS = {
  attrition: ['Attrition', 'attrition'],
  satisfaction: ['JobSatisfaction', 'Job_Satisfaction', 'job_satisfaction'],
  age: ['Age', 'age'],
  yearsWithManager: ['YearsWithCurrManager', 'Years_With_Curr_Manager', 'years_with_curr_manager'],
  jobInvolvement: ['JobInvolvement', 'Job_Involvement', 'job_involvement'],
  overtime: ['OverTime', 'Over_Time', 'overtime'],
  yearsAtCompany: ['YearsAtCompany', 'Years_At_Company', 'years_at_company'],
  monthlyIncome: ['MonthlyIncome', 'Monthly_Income', 'monthly_income'],
  distanceFromHome: ['DistanceFromHome', 'Distance_From_Home', 'distance_from_home'],
  yearsSincePromotion: ['YearsSinceLastPromotion', 'Years_Since_Last_Promotion'],
  workLifeBalance: ['WorkLifeBalance', 'Work_Life_Balance', 'work_life_balance'],
  department: ['Department', 'department'],
  environmentSatisfaction: ['EnvironmentSatisfaction', 'Environment_Satisfaction'],
  trainingTimes: ['TrainingTimesLastYear', 'Training_Times_Last_Year'],
  hourlyRate: ['HourlyRate', 'Hourly_Rate', 'hourly_rate'],
  relationshipSatisfaction: ['RelationshipSatisfaction', 'Relationship_Satisfaction'],
};

const KAGGLE_COLUMNS = {
  left: ['left', 'Left'],
  satisfactionLevel: ['satisfaction_level', 'Satisfaction_Level', 'SatisfactionLevel'],
  numberProject: ['number_project', 'Number_Project', 'NumberProject'],
  timeSpendCompany: ['time_spend_company', 'Time_Spend_Company', 'TimeSpendCompany'],
  lastEvaluation: ['last_evaluation', 'Last_Evaluation', 'LastEvaluation'],
  averageMonthlyHours: ['average_montly_hours', 'average_monthly_hours', 'Average_Monthly_Hours'],
  department: ['Department', 'department', 'sales'],
  salary: ['salary', 'Salary'],
  promotionLast5Years: ['promotion_last_5years', 'Promotion_Last_5_Years'],
  workAccident: ['Work_accident', 'work_accident'],
};

// Detect column in data (case-insensitive, flexible matching)
function findColumn(row: Record<string, unknown>, candidates: string[]): string | undefined {
  const keys = Object.keys(row);
  for (const candidate of candidates) {
    const found = keys.find(k => k.toLowerCase() === candidate.toLowerCase());
    if (found) return found;
  }
  return undefined;
}

function getColumnValue(row: Record<string, unknown>, candidates: string[]): unknown {
  const col = findColumn(row, candidates);
  return col ? row[col] : undefined;
}

// Detect dataset type based on available columns
function detectDatasetType(data: Record<string, unknown>[]): 'ibm' | 'kaggle' | 'unknown' {
  if (data.length === 0) return 'unknown';
  const row = data[0];
  const keys = Object.keys(row).map(k => k.toLowerCase());
  
  // Check for Kaggle-specific columns
  const hasKaggleColumns = keys.some(k => 
    ['satisfaction_level', 'number_project', 'time_spend_company', 'left'].includes(k)
  );
  
  // Check for IBM-specific columns
  const hasIBMColumns = keys.some(k => 
    ['jobsatisfaction', 'yearsatcompany', 'monthlyincome', 'overtime'].includes(k)
  );
  
  if (hasKaggleColumns) return 'kaggle';
  if (hasIBMColumns) return 'ibm';
  return 'unknown';
}

// Get available columns for feature importance
function getAvailableFeatures(row: Record<string, unknown>): string[] {
  return Object.keys(row).filter(k => 
    !['id', 'employeeid', 'employee_id', 'employeenumber', 'employee_number'].includes(k.toLowerCase())
  );
}

// Calculate risk score based on available data (simulates ML model)
function calculateRiskScore(row: Record<string, unknown>, datasetType: 'ibm' | 'kaggle' | 'unknown'): number {
  let baseScore = 0.3; // Base risk
  
  if (datasetType === 'ibm') {
    // IBM dataset logic based on paper's SHAP findings
    const satisfaction = getColumnValue(row, IBM_COLUMNS.satisfaction);
    const overtime = getColumnValue(row, IBM_COLUMNS.overtime);
    const yearsAtCompany = getColumnValue(row, IBM_COLUMNS.yearsAtCompany);
    const age = getColumnValue(row, IBM_COLUMNS.age);
    const monthlyIncome = getColumnValue(row, IBM_COLUMNS.monthlyIncome);
    const workLifeBalance = getColumnValue(row, IBM_COLUMNS.workLifeBalance);
    const yearsSincePromotion = getColumnValue(row, IBM_COLUMNS.yearsSincePromotion);
    
    // Job Satisfaction (most important - lower = higher risk)
    if (satisfaction !== undefined) {
      const satValue = Number(satisfaction);
      if (satValue <= 1) baseScore += 0.25;
      else if (satValue <= 2) baseScore += 0.15;
      else if (satValue >= 4) baseScore -= 0.15;
    }
    
    // Overtime (positive correlation with attrition)
    if (overtime !== undefined) {
      const otValue = String(overtime).toLowerCase();
      if (otValue === 'yes' || otValue === '1' || otValue === 'true') {
        baseScore += 0.12;
      }
    }
    
    // Years at company (newer = higher risk)
    if (yearsAtCompany !== undefined) {
      const years = Number(yearsAtCompany);
      if (years < 2) baseScore += 0.1;
      else if (years > 8) baseScore -= 0.1;
    }
    
    // Age (younger = higher risk based on paper)
    if (age !== undefined) {
      const ageValue = Number(age);
      if (ageValue < 30) baseScore += 0.08;
      else if (ageValue > 45) baseScore -= 0.08;
    }
    
    // Monthly income (lower = higher risk)
    if (monthlyIncome !== undefined) {
      const income = Number(monthlyIncome);
      if (income < 3000) baseScore += 0.1;
      else if (income > 8000) baseScore -= 0.1;
    }
    
    // Work-life balance
    if (workLifeBalance !== undefined) {
      const wlb = Number(workLifeBalance);
      if (wlb <= 1) baseScore += 0.1;
      else if (wlb >= 4) baseScore -= 0.08;
    }
    
    // Years since promotion
    if (yearsSincePromotion !== undefined) {
      const years = Number(yearsSincePromotion);
      if (years > 5) baseScore += 0.08;
    }
    
  } else if (datasetType === 'kaggle') {
    // Kaggle dataset logic based on paper's SHAP findings
    const satisfactionLevel = getColumnValue(row, KAGGLE_COLUMNS.satisfactionLevel);
    const numberProject = getColumnValue(row, KAGGLE_COLUMNS.numberProject);
    const timeSpendCompany = getColumnValue(row, KAGGLE_COLUMNS.timeSpendCompany);
    const lastEvaluation = getColumnValue(row, KAGGLE_COLUMNS.lastEvaluation);
    const avgMonthlyHours = getColumnValue(row, KAGGLE_COLUMNS.averageMonthlyHours);
    const salary = getColumnValue(row, KAGGLE_COLUMNS.salary);
    const promotionLast5Years = getColumnValue(row, KAGGLE_COLUMNS.promotionLast5Years);
    
    // Number of projects (most important - extremes = higher risk)
    if (numberProject !== undefined) {
      const projects = Number(numberProject);
      if (projects >= 6) baseScore += 0.2;
      else if (projects <= 2) baseScore += 0.15;
    }
    
    // Satisfaction level (lower = higher risk)
    if (satisfactionLevel !== undefined) {
      const sat = Number(satisfactionLevel);
      if (sat < 0.2) baseScore += 0.25;
      else if (sat < 0.4) baseScore += 0.15;
      else if (sat > 0.8) baseScore -= 0.15;
    }
    
    // Time spend company
    if (timeSpendCompany !== undefined) {
      const years = Number(timeSpendCompany);
      if (years >= 5 && years <= 6) baseScore += 0.1;
      else if (years < 2) baseScore += 0.05;
    }
    
    // Last evaluation (extremes = higher risk)
    if (lastEvaluation !== undefined) {
      const eval_ = Number(lastEvaluation);
      if (eval_ > 0.8) baseScore += 0.08;
      else if (eval_ < 0.5) baseScore += 0.1;
    }
    
    // Average monthly hours (overwork = higher risk)
    if (avgMonthlyHours !== undefined) {
      const hours = Number(avgMonthlyHours);
      if (hours > 250) baseScore += 0.15;
      else if (hours < 150) baseScore += 0.08;
    }
    
    // Salary level
    if (salary !== undefined) {
      const sal = String(salary).toLowerCase();
      if (sal === 'low') baseScore += 0.12;
      else if (sal === 'high') baseScore -= 0.1;
    }
    
    // No promotion in 5 years
    if (promotionLast5Years !== undefined) {
      const promo = Number(promotionLast5Years);
      if (promo === 0) baseScore += 0.05;
    }
  } else {
    // Unknown dataset - use random with slight variation
    baseScore = 0.2 + Math.random() * 0.6;
  }
  
  // Add small random noise for realistic variation
  baseScore += (Math.random() - 0.5) * 0.1;
  
  // Clamp to valid range
  return Math.max(0.05, Math.min(0.95, baseScore));
}

// Simulates ML analysis with column flexibility
export function generateMockPredictions(data: Record<string, unknown>[]): PredictionData[] {
  const datasetType = detectDatasetType(data);
  
  return data.map((row, index) => {
    const riskScore = calculateRiskScore(row, datasetType);
    const riskLevel = riskScore < 0.25 ? 'low' : riskScore < 0.5 ? 'medium' : riskScore < 0.75 ? 'high' : 'critical';
    
    // Generate contextual factors based on available data
    const factors: string[] = [];
    
    if (datasetType === 'ibm') {
      const satisfaction = getColumnValue(row, IBM_COLUMNS.satisfaction);
      const overtime = getColumnValue(row, IBM_COLUMNS.overtime);
      const yearsAtCompany = getColumnValue(row, IBM_COLUMNS.yearsAtCompany);
      const yearsSincePromotion = getColumnValue(row, IBM_COLUMNS.yearsSincePromotion);
      
      if (satisfaction !== undefined && Number(satisfaction) <= 2) factors.push('Low job satisfaction');
      if (overtime && String(overtime).toLowerCase() === 'yes') factors.push('Working overtime frequently');
      if (yearsAtCompany !== undefined && Number(yearsAtCompany) < 2) factors.push('Short tenure at company');
      if (yearsSincePromotion !== undefined && Number(yearsSincePromotion) > 4) factors.push('No recent promotion');
    } else if (datasetType === 'kaggle') {
      const satisfactionLevel = getColumnValue(row, KAGGLE_COLUMNS.satisfactionLevel);
      const numberProject = getColumnValue(row, KAGGLE_COLUMNS.numberProject);
      const avgMonthlyHours = getColumnValue(row, KAGGLE_COLUMNS.averageMonthlyHours);
      const salary = getColumnValue(row, KAGGLE_COLUMNS.salary);
      
      if (satisfactionLevel !== undefined && Number(satisfactionLevel) < 0.4) factors.push('Low satisfaction level');
      if (numberProject !== undefined && Number(numberProject) >= 6) factors.push('Overloaded with projects');
      if (avgMonthlyHours !== undefined && Number(avgMonthlyHours) > 250) factors.push('Excessive working hours');
      if (salary && String(salary).toLowerCase() === 'low') factors.push('Below-market compensation');
    }
    
    if (riskScore > 0.6 && factors.length === 0) factors.push('Multiple moderate risk factors');
    if (factors.length === 0) factors.push('No significant risk factors identified');
    
    // Get employee ID from various possible columns
    const employeeId = row.EmployeeID || row.employee_id || row.EmployeeNumber || row.id || index + 1;
    
    // Get department from various possible columns
    const deptValue = getColumnValue(row, [...IBM_COLUMNS.department, ...KAGGLE_COLUMNS.department]);
    
    return {
      employee_id: typeof employeeId === 'string' || typeof employeeId === 'number' ? employeeId : index + 1,
      department: String(deptValue || 'Unknown'),
      risk_score: Math.round(riskScore * 100),
      risk_level: riskLevel,
      attrition_probability: riskScore,
      factors,
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
      rate: deptPredictions.length > 0 ? Math.round((deptAtRisk / deptPredictions.length) * 100) : 0,
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

// Generate feature importance dynamically based on available columns (SHAP-like)
export function generateMockFeatureImportance(data: Record<string, unknown>[]): FeatureImportance[] {
  const datasetType = detectDatasetType(data);
  
  if (datasetType === 'ibm') {
    // Based on paper's findings: JobSatisfaction, Age, YearsWithCurrManager top 3
    return [
      { feature: 'Job Satisfaction', importance: 0.28, direction: 'negative', description: 'Most influential factor - lower satisfaction strongly correlates with higher attrition risk' },
      { feature: 'Age', importance: 0.18, direction: 'negative', description: 'Younger employees show higher turnover tendency' },
      { feature: 'Years with Current Manager', importance: 0.15, direction: 'negative', description: 'Longer tenure with manager indicates stability' },
      { feature: 'Job Involvement', importance: 0.12, direction: 'negative', description: 'Higher involvement reduces departure likelihood' },
      { feature: 'Overtime', importance: 0.10, direction: 'positive', description: 'Frequent overtime increases burnout and departure' },
      { feature: 'Years at Company', importance: 0.08, direction: 'negative', description: 'Newer employees have higher turnover rates' },
      { feature: 'Monthly Income', importance: 0.07, direction: 'negative', description: 'Competitive pay improves retention' },
      { feature: 'Work-Life Balance', importance: 0.06, direction: 'negative', description: 'Poor balance drives attrition' },
      { feature: 'Years Since Promotion', importance: 0.05, direction: 'positive', description: 'Stalled career growth increases risk' },
      { feature: 'Environment Satisfaction', importance: 0.04, direction: 'negative', description: 'Workplace environment affects retention' },
    ];
  } else if (datasetType === 'kaggle') {
    // Based on paper's findings: number_project, satisfaction_level, time_spend_company top 3
    return [
      { feature: 'Number of Projects', importance: 0.32, direction: 'positive', description: 'Overloaded employees with many projects show highest attrition' },
      { feature: 'Satisfaction Level', importance: 0.26, direction: 'negative', description: 'Low satisfaction is a primary departure driver' },
      { feature: 'Time Spent at Company', importance: 0.15, direction: 'negative', description: 'Mid-tenure employees (5-6 years) show elevated risk' },
      { feature: 'Last Evaluation', importance: 0.10, direction: 'positive', description: 'Both very high and very low performers leave more often' },
      { feature: 'Average Monthly Hours', importance: 0.08, direction: 'positive', description: 'Excessive hours lead to burnout and departure' },
      { feature: 'Salary Level', importance: 0.06, direction: 'negative', description: 'Low salary employees show higher turnover' },
      { feature: 'Promotion Last 5 Years', importance: 0.04, direction: 'negative', description: 'No recent promotion increases risk' },
      { feature: 'Work Accident', importance: 0.02, direction: 'positive', description: 'Minor correlation with attrition' },
    ];
  } else {
    // Generic features for unknown datasets
    const row = data[0] || {};
    const features = getAvailableFeatures(row);
    
    return features.slice(0, 10).map((feature, i) => ({
      feature: feature.replace(/_/g, ' ').replace(/([A-Z])/g, ' $1').trim(),
      importance: Math.max(0.02, 0.25 - i * 0.02 + (Math.random() * 0.05)),
      direction: Math.random() > 0.5 ? 'positive' : 'negative' as 'positive' | 'negative',
      description: `This feature contributes to attrition prediction based on data patterns`,
    }));
  }
}

export function generateMockTopics(datasetType: 'ibm' | 'kaggle' | 'unknown'): TopicData[] {
  const baseTopics = [
    {
      topic_id: 1,
      name: 'Work-Life Balance',
      keywords: ['hours', 'overtime', 'flexible', 'remote', 'balance'],
      prevalence: 0.28,
      sentiment: 'negative' as const,
      sample_reviews: ['Long hours expected with little flexibility', 'Work-life balance could be improved significantly'],
    },
    {
      topic_id: 2,
      name: 'Career Growth',
      keywords: ['promotion', 'growth', 'opportunity', 'learning', 'career'],
      prevalence: 0.24,
      sentiment: 'neutral' as const,
      sample_reviews: ['Good learning opportunities but slow promotions', 'Career path is unclear for many roles'],
    },
    {
      topic_id: 3,
      name: 'Compensation & Benefits',
      keywords: ['salary', 'pay', 'benefits', 'bonus', 'competitive'],
      prevalence: 0.20,
      sentiment: 'negative' as const,
      sample_reviews: ['Salary below market rate for the industry', 'Benefits package is decent but not competitive'],
    },
    {
      topic_id: 4,
      name: 'Management & Leadership',
      keywords: ['manager', 'leadership', 'support', 'communication', 'direction'],
      prevalence: 0.16,
      sentiment: 'positive' as const,
      sample_reviews: ['Great managers who support their teams', 'Leadership is approachable and transparent'],
    },
    {
      topic_id: 5,
      name: 'Company Culture',
      keywords: ['culture', 'team', 'environment', 'colleagues', 'collaborative'],
      prevalence: 0.12,
      sentiment: 'positive' as const,
      sample_reviews: ['Amazing team culture and collaboration', 'Colleagues are friendly and supportive'],
    },
  ];
  
  // Adjust topics based on dataset type
  if (datasetType === 'kaggle') {
    baseTopics[0].prevalence = 0.32; // Work-life balance more prevalent
    baseTopics[2].prevalence = 0.24; // Compensation higher
  }
  
  return baseTopics;
}

export function generateMockRecommendations(
  results: AnalysisResults, 
  featureImportance: FeatureImportance[],
  datasetType: 'ibm' | 'kaggle' | 'unknown'
): Recommendation[] {
  const recommendations: Recommendation[] = [];
  const topFeature = featureImportance[0];
  
  // Dynamic recommendation based on top feature
  if (topFeature) {
    if (topFeature.feature.toLowerCase().includes('satisfaction')) {
      recommendations.push({
        id: '1',
        priority: 'high',
        category: 'Employee Engagement',
        title: 'Implement Regular Satisfaction Surveys',
        description: `${topFeature.feature} is the top predictor of attrition. Establish quarterly pulse surveys to identify issues early.`,
        impact: `Could reduce attrition by up to ${Math.round(topFeature.importance * 100)}%`,
        action_items: ['Deploy anonymous quarterly surveys', 'Create action plans for low-scoring areas', 'Share results and improvements with employees'],
      });
    } else if (topFeature.feature.toLowerCase().includes('project')) {
      recommendations.push({
        id: '1',
        priority: 'high',
        category: 'Workload Management',
        title: 'Optimize Project Distribution',
        description: 'Project overload is the primary attrition driver. Implement workload balancing strategies.',
        impact: `Could reduce attrition by up to ${Math.round(topFeature.importance * 100)}%`,
        action_items: ['Audit current project allocations', 'Set maximum project limits per employee', 'Implement resource planning tools'],
      });
    }
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
  
  // Add dataset-specific recommendations
  if (datasetType === 'ibm') {
    recommendations.push({
      id: '4',
      priority: 'medium',
      category: 'Manager Development',
      title: 'Strengthen Manager-Employee Relationships',
      description: 'Years with current manager is a key predictor. Invest in manager training and stability.',
      impact: 'Manager retention directly impacts team retention',
      action_items: ['Implement manager coaching programs', 'Reduce unnecessary manager rotations', 'Train managers on retention conversations'],
    });
  } else if (datasetType === 'kaggle') {
    recommendations.push({
      id: '4',
      priority: 'medium',
      category: 'Work Hours',
      title: 'Address Excessive Working Hours',
      description: 'Average monthly hours correlate with attrition. Monitor and manage overtime.',
      impact: 'Reducing burnout decreases turnover by 15-20%',
      action_items: ['Implement time tracking alerts', 'Encourage work-life balance', 'Review staffing levels in overworked teams'],
    });
  }
  
  recommendations.push({
    id: '5',
    priority: 'low',
    category: 'Development',
    title: 'Enhance Career Development Programs',
    description: 'Career stagnation contributes to attrition. Create clearer growth paths.',
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
  // Simulate processing time (GAN + Transformer training)
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  const datasetType = detectDatasetType(data);
  
  const predictions = generateMockPredictions(data);
  const results = generateMockResults(data, predictions);
  const feature_importance = generateMockFeatureImportance(data);
  const topics = generateMockTopics(datasetType);
  const recommendations = generateMockRecommendations(results, feature_importance, datasetType);
  
  return { results, predictions, feature_importance, topics, recommendations };
}
