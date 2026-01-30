import type { 
  AnalysisResults, 
  PredictionData, 
  FeatureImportance, 
  TopicData, 
  Recommendation,
  DepartmentBreakdown 
} from '@/types/dataset';

// Column mappings for different dataset formats
const IBM_COLUMNS = {
  attrition: ['Attrition', 'attrition'],
  satisfaction: ['JobSatisfaction', 'Job_Satisfaction', 'job_satisfaction'],
  age: ['Age', 'age'],
  yearsWithManager: ['YearsWithCurrManager', 'Years_With_Curr_Manager'],
  jobInvolvement: ['JobInvolvement', 'Job_Involvement'],
  overtime: ['OverTime', 'Over_Time', 'overtime'],
  yearsAtCompany: ['YearsAtCompany', 'Years_At_Company'],
  monthlyIncome: ['MonthlyIncome', 'Monthly_Income'],
  distanceFromHome: ['DistanceFromHome', 'Distance_From_Home'],
  yearsSincePromotion: ['YearsSinceLastPromotion', 'Years_Since_Last_Promotion'],
  workLifeBalance: ['WorkLifeBalance', 'Work_Life_Balance'],
  department: ['Department', 'department'],
  environmentSatisfaction: ['EnvironmentSatisfaction', 'Environment_Satisfaction'],
};

const KAGGLE_COLUMNS = {
  left: ['left', 'Left'],
  satisfactionLevel: ['satisfaction_level', 'Satisfaction_Level'],
  numberProject: ['number_project', 'Number_Project'],
  timeSpendCompany: ['time_spend_company', 'Time_Spend_Company'],
  lastEvaluation: ['last_evaluation', 'Last_Evaluation'],
  averageMonthlyHours: ['average_montly_hours', 'average_monthly_hours'],
  department: ['Department', 'department', 'sales'],
  salary: ['salary', 'Salary'],
  promotionLast5Years: ['promotion_last_5years', 'Promotion_Last_5_Years'],
};

// AmbitionBox format (Amazon, Capgemini, Mahindra, Tata, Maruti reviews)
const AMBITIONBOX_COLUMNS = {
  overallRating: ['Overall_rating', 'overall_rating', 'OverallRating'],
  workLifeBalance: ['work_life_balance', 'Work_Life_Balance', 'WorkLifeBalance'],
  skillDevelopment: ['skill_development', 'Skill_Development', 'SkillDevelopment'],
  salaryBenefits: ['salary_and_benefits', 'Salary_And_Benefits', 'SalaryAndBenefits'],
  jobSecurity: ['job_security', 'Job_Security', 'JobSecurity'],
  careerGrowth: ['career_growth', 'Career_Growth', 'CareerGrowth'],
  workSatisfaction: ['work_satisfaction', 'Work_Satisfaction', 'WorkSatisfaction'],
  department: ['Department', 'department'],
  title: ['Title', 'Name', 'title', 'name'],
  likes: ['Likes', 'likes'],
  dislikes: ['Dislikes', 'dislikes'],
  jobType: ['Job_type', 'job_type', 'JobType'],
  place: ['Place', 'place', 'Location'],
};

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
function detectDatasetType(data: Record<string, unknown>[]): 'ibm' | 'kaggle' | 'ambitionbox' | 'unknown' {
  if (data.length === 0) return 'unknown';
  const row = data[0];
  const keys = Object.keys(row).map(k => k.toLowerCase());
  
  // Check for AmbitionBox format (reviews with ratings)
  const hasAmbitionBox = keys.some(k => 
    ['overall_rating', 'work_life_balance', 'skill_development', 'salary_and_benefits', 'job_security'].includes(k)
  );
  
  // Check for Kaggle format
  const hasKaggle = keys.some(k => 
    ['satisfaction_level', 'number_project', 'time_spend_company', 'left'].includes(k)
  );
  
  // Check for IBM format
  const hasIBM = keys.some(k => 
    ['jobsatisfaction', 'yearsatcompany', 'monthlyincome', 'overtime', 'attrition'].includes(k)
  );
  
  if (hasAmbitionBox) return 'ambitionbox';
  if (hasKaggle) return 'kaggle';
  if (hasIBM) return 'ibm';
  return 'unknown';
}

// Calculate risk score based on available data
function calculateRiskScore(row: Record<string, unknown>, datasetType: 'ibm' | 'kaggle' | 'ambitionbox' | 'unknown'): number {
  let baseScore = 0.3;
  
  if (datasetType === 'ambitionbox') {
    // AmbitionBox review data - lower ratings = higher risk
    const overallRating = getColumnValue(row, AMBITIONBOX_COLUMNS.overallRating);
    const workLifeBalance = getColumnValue(row, AMBITIONBOX_COLUMNS.workLifeBalance);
    const workSatisfaction = getColumnValue(row, AMBITIONBOX_COLUMNS.workSatisfaction);
    const careerGrowth = getColumnValue(row, AMBITIONBOX_COLUMNS.careerGrowth);
    const jobSecurity = getColumnValue(row, AMBITIONBOX_COLUMNS.jobSecurity);
    const salaryBenefits = getColumnValue(row, AMBITIONBOX_COLUMNS.salaryBenefits);
    
    // Overall rating (1-5 scale, most important)
    if (overallRating !== undefined) {
      const rating = Number(overallRating);
      if (rating <= 1.5) baseScore += 0.35;
      else if (rating <= 2.5) baseScore += 0.2;
      else if (rating <= 3) baseScore += 0.1;
      else if (rating >= 4.5) baseScore -= 0.2;
      else if (rating >= 4) baseScore -= 0.1;
    }
    
    // Work satisfaction
    if (workSatisfaction !== undefined) {
      const sat = Number(workSatisfaction);
      if (sat <= 2) baseScore += 0.15;
      else if (sat >= 4) baseScore -= 0.1;
    }
    
    // Work-life balance
    if (workLifeBalance !== undefined) {
      const wlb = Number(workLifeBalance);
      if (wlb <= 2) baseScore += 0.12;
      else if (wlb >= 4) baseScore -= 0.08;
    }
    
    // Career growth
    if (careerGrowth !== undefined) {
      const growth = Number(careerGrowth);
      if (growth <= 2) baseScore += 0.1;
      else if (growth >= 4) baseScore -= 0.08;
    }
    
    // Job security
    if (jobSecurity !== undefined) {
      const security = Number(jobSecurity);
      if (security <= 2) baseScore += 0.1;
      else if (security >= 4) baseScore -= 0.08;
    }
    
    // Salary & benefits
    if (salaryBenefits !== undefined) {
      const salary = Number(salaryBenefits);
      if (salary <= 2) baseScore += 0.08;
      else if (salary >= 4) baseScore -= 0.06;
    }
    
  } else if (datasetType === 'ibm') {
    const satisfaction = getColumnValue(row, IBM_COLUMNS.satisfaction);
    const overtime = getColumnValue(row, IBM_COLUMNS.overtime);
    const yearsAtCompany = getColumnValue(row, IBM_COLUMNS.yearsAtCompany);
    const age = getColumnValue(row, IBM_COLUMNS.age);
    const monthlyIncome = getColumnValue(row, IBM_COLUMNS.monthlyIncome);
    const workLifeBalance = getColumnValue(row, IBM_COLUMNS.workLifeBalance);
    const yearsSincePromotion = getColumnValue(row, IBM_COLUMNS.yearsSincePromotion);
    
    if (satisfaction !== undefined) {
      const satValue = Number(satisfaction);
      if (satValue <= 1) baseScore += 0.25;
      else if (satValue <= 2) baseScore += 0.15;
      else if (satValue >= 4) baseScore -= 0.15;
    }
    
    if (overtime !== undefined) {
      const otValue = String(overtime).toLowerCase();
      if (otValue === 'yes' || otValue === '1' || otValue === 'true') {
        baseScore += 0.12;
      }
    }
    
    if (yearsAtCompany !== undefined) {
      const years = Number(yearsAtCompany);
      if (years < 2) baseScore += 0.1;
      else if (years > 8) baseScore -= 0.1;
    }
    
    if (age !== undefined) {
      const ageValue = Number(age);
      if (ageValue < 30) baseScore += 0.08;
      else if (ageValue > 45) baseScore -= 0.08;
    }
    
    if (monthlyIncome !== undefined) {
      const income = Number(monthlyIncome);
      if (income < 3000) baseScore += 0.1;
      else if (income > 8000) baseScore -= 0.1;
    }
    
    if (workLifeBalance !== undefined) {
      const wlb = Number(workLifeBalance);
      if (wlb <= 1) baseScore += 0.1;
      else if (wlb >= 4) baseScore -= 0.08;
    }
    
    if (yearsSincePromotion !== undefined) {
      const years = Number(yearsSincePromotion);
      if (years > 5) baseScore += 0.08;
    }
    
  } else if (datasetType === 'kaggle') {
    const satisfactionLevel = getColumnValue(row, KAGGLE_COLUMNS.satisfactionLevel);
    const numberProject = getColumnValue(row, KAGGLE_COLUMNS.numberProject);
    const timeSpendCompany = getColumnValue(row, KAGGLE_COLUMNS.timeSpendCompany);
    const lastEvaluation = getColumnValue(row, KAGGLE_COLUMNS.lastEvaluation);
    const avgMonthlyHours = getColumnValue(row, KAGGLE_COLUMNS.averageMonthlyHours);
    const salary = getColumnValue(row, KAGGLE_COLUMNS.salary);
    
    if (numberProject !== undefined) {
      const projects = Number(numberProject);
      if (projects >= 6) baseScore += 0.2;
      else if (projects <= 2) baseScore += 0.15;
    }
    
    if (satisfactionLevel !== undefined) {
      const sat = Number(satisfactionLevel);
      if (sat < 0.2) baseScore += 0.25;
      else if (sat < 0.4) baseScore += 0.15;
      else if (sat > 0.8) baseScore -= 0.15;
    }
    
    if (timeSpendCompany !== undefined) {
      const years = Number(timeSpendCompany);
      if (years >= 5 && years <= 6) baseScore += 0.1;
      else if (years < 2) baseScore += 0.05;
    }
    
    if (lastEvaluation !== undefined) {
      const evalScore = Number(lastEvaluation);
      if (evalScore > 0.8) baseScore += 0.08;
      else if (evalScore < 0.5) baseScore += 0.1;
    }
    
    if (avgMonthlyHours !== undefined) {
      const hours = Number(avgMonthlyHours);
      if (hours > 250) baseScore += 0.15;
      else if (hours < 150) baseScore += 0.08;
    }
    
    if (salary !== undefined) {
      const sal = String(salary).toLowerCase();
      if (sal === 'low') baseScore += 0.12;
      else if (sal === 'high') baseScore -= 0.1;
    }
  } else {
    baseScore = 0.2 + Math.random() * 0.6;
  }
  
  baseScore += (Math.random() - 0.5) * 0.1;
  return Math.max(0.05, Math.min(0.95, baseScore));
}

export function generateMockPredictions(data: Record<string, unknown>[]): PredictionData[] {
  const datasetType = detectDatasetType(data);
  
  return data.map((row, index) => {
    const riskScore = calculateRiskScore(row, datasetType);
    // Five-tier classification matching backend
    const riskLevel = riskScore < 0.20 ? 'low' : 
                      riskScore < 0.40 ? 'early_warning' : 
                      riskScore < 0.60 ? 'moderate' : 
                      riskScore < 0.80 ? 'high' : 'critical';
    
    const factors: string[] = [];
    
    if (datasetType === 'ambitionbox') {
      const overallRating = getColumnValue(row, AMBITIONBOX_COLUMNS.overallRating);
      const workSatisfaction = getColumnValue(row, AMBITIONBOX_COLUMNS.workSatisfaction);
      const workLifeBalance = getColumnValue(row, AMBITIONBOX_COLUMNS.workLifeBalance);
      const careerGrowth = getColumnValue(row, AMBITIONBOX_COLUMNS.careerGrowth);
      const jobSecurity = getColumnValue(row, AMBITIONBOX_COLUMNS.jobSecurity);
      
      if (overallRating !== undefined && Number(overallRating) <= 2) factors.push('Low overall rating');
      if (workSatisfaction !== undefined && Number(workSatisfaction) <= 2) factors.push('Poor work satisfaction');
      if (workLifeBalance !== undefined && Number(workLifeBalance) <= 2) factors.push('Poor work-life balance');
      if (careerGrowth !== undefined && Number(careerGrowth) <= 2) factors.push('Limited career growth');
      if (jobSecurity !== undefined && Number(jobSecurity) <= 2) factors.push('Low job security');
    } else if (datasetType === 'ibm') {
      const satisfaction = getColumnValue(row, IBM_COLUMNS.satisfaction);
      const overtime = getColumnValue(row, IBM_COLUMNS.overtime);
      const yearsAtCompany = getColumnValue(row, IBM_COLUMNS.yearsAtCompany);
      const yearsSincePromotion = getColumnValue(row, IBM_COLUMNS.yearsSincePromotion);
      
      if (satisfaction !== undefined && Number(satisfaction) <= 2) factors.push('Low job satisfaction');
      if (overtime && String(overtime).toLowerCase() === 'yes') factors.push('Frequent overtime');
      if (yearsAtCompany !== undefined && Number(yearsAtCompany) < 2) factors.push('Short tenure');
      if (yearsSincePromotion !== undefined && Number(yearsSincePromotion) > 4) factors.push('No recent promotion');
    } else if (datasetType === 'kaggle') {
      const satisfactionLevel = getColumnValue(row, KAGGLE_COLUMNS.satisfactionLevel);
      const numberProject = getColumnValue(row, KAGGLE_COLUMNS.numberProject);
      const avgMonthlyHours = getColumnValue(row, KAGGLE_COLUMNS.averageMonthlyHours);
      const salary = getColumnValue(row, KAGGLE_COLUMNS.salary);
      
      if (satisfactionLevel !== undefined && Number(satisfactionLevel) < 0.4) factors.push('Low satisfaction');
      if (numberProject !== undefined && Number(numberProject) >= 6) factors.push('Project overload');
      if (avgMonthlyHours !== undefined && Number(avgMonthlyHours) > 250) factors.push('Excessive hours');
      if (salary && String(salary).toLowerCase() === 'low') factors.push('Low compensation');
    }
    
    if (riskScore > 0.6 && factors.length === 0) factors.push('Multiple moderate risk factors');
    if (factors.length === 0) factors.push('No significant risk factors');
    
    // Get identifier
    let employeeId: string | number = index + 1;
    if (datasetType === 'ambitionbox') {
      const title = getColumnValue(row, AMBITIONBOX_COLUMNS.title);
      employeeId = title ? String(title).slice(0, 30) : index + 1;
    } else {
      const id = row.EmployeeID || row.employee_id || row.EmployeeNumber || row.id;
      if (id) employeeId = typeof id === 'string' || typeof id === 'number' ? id : index + 1;
    }
    
    const deptValue = getColumnValue(row, [...IBM_COLUMNS.department, ...KAGGLE_COLUMNS.department, ...AMBITIONBOX_COLUMNS.department]);
    
    return {
      employee_id: employeeId,
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
  
  return {
    total_employees: data.length,
    at_risk_count: atRiskCount,
    attrition_rate: Math.round((atRiskCount / data.length) * 100),
    department_breakdown: departmentBreakdown,
    risk_distribution: {
      low: predictions.filter(p => p.risk_level === 'low').length,
      early_warning: predictions.filter(p => p.risk_level === 'early_warning').length,
      moderate: predictions.filter(p => p.risk_level === 'moderate').length,
      high: predictions.filter(p => p.risk_level === 'high').length,
      critical: predictions.filter(p => p.risk_level === 'critical').length,
    },
  };
}

export function generateMockFeatureImportance(data: Record<string, unknown>[]): FeatureImportance[] {
  const datasetType = detectDatasetType(data);
  
  if (datasetType === 'ambitionbox') {
    return [
      { feature: 'Overall Rating', importance: 0.28, direction: 'negative', description: 'Low overall rating strongly predicts attrition risk' },
      { feature: 'Work Satisfaction', importance: 0.22, direction: 'negative', description: 'Work satisfaction directly impacts retention' },
      { feature: 'Work-Life Balance', importance: 0.16, direction: 'negative', description: 'Poor balance drives employee departure' },
      { feature: 'Career Growth', importance: 0.12, direction: 'negative', description: 'Limited growth opportunities increase risk' },
      { feature: 'Job Security', importance: 0.10, direction: 'negative', description: 'Low perceived security correlates with turnover' },
      { feature: 'Salary & Benefits', importance: 0.08, direction: 'negative', description: 'Below-market compensation increases attrition' },
      { feature: 'Skill Development', importance: 0.06, direction: 'negative', description: 'Learning opportunities affect engagement' },
      { feature: 'Department', importance: 0.04, direction: 'positive', description: 'Some departments show higher turnover' },
    ];
  } else if (datasetType === 'ibm') {
    return [
      { feature: 'Job Satisfaction', importance: 0.28, direction: 'negative', description: 'Most influential factor per SHAP analysis' },
      { feature: 'Age', importance: 0.18, direction: 'negative', description: 'Younger employees show higher mobility' },
      { feature: 'Years with Manager', importance: 0.15, direction: 'negative', description: 'Manager stability reduces turnover' },
      { feature: 'Job Involvement', importance: 0.12, direction: 'negative', description: 'Higher involvement improves retention' },
      { feature: 'Overtime', importance: 0.10, direction: 'positive', description: 'Frequent overtime increases burnout' },
      { feature: 'Years at Company', importance: 0.08, direction: 'negative', description: 'Tenure inversely correlates with risk' },
      { feature: 'Monthly Income', importance: 0.07, direction: 'negative', description: 'Competitive pay improves retention' },
      { feature: 'Work-Life Balance', importance: 0.06, direction: 'negative', description: 'Balance affects job satisfaction' },
    ];
  } else if (datasetType === 'kaggle') {
    return [
      { feature: 'Number of Projects', importance: 0.32, direction: 'positive', description: 'Project overload is #1 attrition driver' },
      { feature: 'Satisfaction Level', importance: 0.26, direction: 'negative', description: 'Low satisfaction strongly predicts departure' },
      { feature: 'Time at Company', importance: 0.15, direction: 'negative', description: 'Mid-tenure (5-6 years) shows elevated risk' },
      { feature: 'Last Evaluation', importance: 0.10, direction: 'positive', description: 'Extremes (very high/low) correlate with leaving' },
      { feature: 'Monthly Hours', importance: 0.08, direction: 'positive', description: 'Excessive hours drive burnout' },
      { feature: 'Salary Level', importance: 0.06, direction: 'negative', description: 'Low salary increases turnover' },
      { feature: 'Promotion (5yr)', importance: 0.04, direction: 'negative', description: 'No promotion increases risk' },
    ];
  }
  
  return [
    { feature: 'Overall Rating', importance: 0.25, direction: 'negative', description: 'Primary satisfaction indicator' },
    { feature: 'Work Satisfaction', importance: 0.20, direction: 'negative', description: 'Job satisfaction drives retention' },
    { feature: 'Work-Life Balance', importance: 0.15, direction: 'negative', description: 'Balance affects wellbeing' },
    { feature: 'Career Growth', importance: 0.12, direction: 'negative', description: 'Growth opportunities matter' },
    { feature: 'Job Security', importance: 0.10, direction: 'negative', description: 'Security reduces anxiety' },
  ];
}

export function generateMockTopics(datasetType: 'ibm' | 'kaggle' | 'ambitionbox' | 'unknown'): TopicData[] {
  if (datasetType === 'ambitionbox') {
    return [
      { topic_id: 1, name: 'Work-Life Balance', keywords: ['hours', 'overtime', 'balance', 'weekends', 'flexible'], prevalence: 0.28, sentiment: 'negative', sample_reviews: ['Long working hours expected', 'No work-life balance'] },
      { topic_id: 2, name: 'Management & Leadership', keywords: ['manager', 'leadership', 'toxic', 'support', 'micro'], prevalence: 0.24, sentiment: 'neutral', sample_reviews: ['Some managers are supportive', 'Micro-management issues'] },
      { topic_id: 3, name: 'Career & Growth', keywords: ['promotion', 'growth', 'learning', 'career', 'opportunity'], prevalence: 0.20, sentiment: 'negative', sample_reviews: ['Limited growth opportunities', 'Slow promotion cycles'] },
      { topic_id: 4, name: 'Compensation', keywords: ['salary', 'pay', 'increment', 'hike', 'benefits'], prevalence: 0.16, sentiment: 'negative', sample_reviews: ['Below market salary', 'Poor increments'] },
      { topic_id: 5, name: 'Job Security', keywords: ['layoff', 'security', 'stable', 'bench', 'cut'], prevalence: 0.12, sentiment: 'negative', sample_reviews: ['No job security', 'Frequent layoffs'] },
    ];
  }
  
  return [
    { topic_id: 1, name: 'Work-Life Balance', keywords: ['hours', 'overtime', 'flexible', 'remote', 'balance'], prevalence: 0.28, sentiment: 'negative', sample_reviews: ['Long hours expected', 'Work-life balance issues'] },
    { topic_id: 2, name: 'Career Growth', keywords: ['promotion', 'growth', 'opportunity', 'learning', 'career'], prevalence: 0.24, sentiment: 'neutral', sample_reviews: ['Good learning but slow promotions', 'Career path unclear'] },
    { topic_id: 3, name: 'Compensation', keywords: ['salary', 'pay', 'benefits', 'bonus', 'competitive'], prevalence: 0.20, sentiment: 'negative', sample_reviews: ['Salary below market', 'Benefits not competitive'] },
    { topic_id: 4, name: 'Management', keywords: ['manager', 'leadership', 'support', 'communication'], prevalence: 0.16, sentiment: 'positive', sample_reviews: ['Great managers', 'Supportive leadership'] },
    { topic_id: 5, name: 'Culture', keywords: ['culture', 'team', 'environment', 'colleagues'], prevalence: 0.12, sentiment: 'positive', sample_reviews: ['Amazing team culture', 'Friendly colleagues'] },
  ];
}

export function generateMockRecommendations(
  results: AnalysisResults, 
  featureImportance: FeatureImportance[],
  datasetType: 'ibm' | 'kaggle' | 'ambitionbox' | 'unknown'
): Recommendation[] {
  const recommendations: Recommendation[] = [];
  const topFeature = featureImportance[0];
  
  if (topFeature) {
    if (topFeature.feature.toLowerCase().includes('rating') || topFeature.feature.toLowerCase().includes('satisfaction')) {
      recommendations.push({
        id: '1',
        priority: 'high',
        category: 'Employee Engagement',
        title: 'Implement Continuous Feedback System',
        description: `${topFeature.feature} is the top predictor. Deploy regular pulse surveys and feedback mechanisms.`,
        impact: `Potential ${Math.round(topFeature.importance * 100)}% attrition reduction`,
        action_items: ['Deploy quarterly satisfaction surveys', 'Create feedback action plans', 'Implement skip-level meetings'],
      });
    } else if (topFeature.feature.toLowerCase().includes('project')) {
      recommendations.push({
        id: '1',
        priority: 'high',
        category: 'Workload Management',
        title: 'Optimize Project Allocation',
        description: 'Project overload is the primary driver. Balance workloads across teams.',
        impact: `Potential ${Math.round(topFeature.importance * 100)}% attrition reduction`,
        action_items: ['Audit project allocations', 'Set maximum project limits', 'Implement capacity planning'],
      });
    }
  }
  
  if (results.attrition_rate > 15) {
    recommendations.push({
      id: '2',
      priority: 'high',
      category: 'Retention',
      title: 'Launch Targeted Retention Program',
      description: `${results.attrition_rate}% risk rate requires immediate intervention.`,
      impact: 'Expected 20-30% turnover reduction',
      action_items: ['Identify high-risk employees', 'Conduct stay interviews', 'Create retention offers'],
    });
  }
  
  const highRiskDepts = results.department_breakdown.filter(d => d.rate > 25);
  if (highRiskDepts.length > 0) {
    recommendations.push({
      id: '3',
      priority: 'medium',
      category: 'Department Focus',
      title: `Address High-Risk: ${highRiskDepts.slice(0, 3).map(d => d.department).join(', ')}`,
      description: 'These departments show significantly elevated attrition risk.',
      impact: 'Targeted savings on replacement costs',
      action_items: ['Conduct focus groups', 'Review compensation', 'Assess management'],
    });
  }
  
  if (datasetType === 'ambitionbox') {
    recommendations.push({
      id: '4',
      priority: 'medium',
      category: 'Work Environment',
      title: 'Improve Work-Life Balance',
      description: 'Review data shows work-life balance as a consistent concern.',
      impact: 'Improved engagement and retention',
      action_items: ['Implement flexible work policies', 'Review overtime practices', 'Promote wellness programs'],
    });
  }
  
  recommendations.push({
    id: '5',
    priority: 'low',
    category: 'Development',
    title: 'Enhance Career Development',
    description: 'Create clear growth paths and development opportunities.',
    impact: 'Improved engagement and reduced turnover',
    action_items: ['Define promotion criteria', 'Implement mentorship', 'Create development plans'],
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
  
  const datasetType = detectDatasetType(data);
  const predictions = generateMockPredictions(data);
  const results = generateMockResults(data, predictions);
  const feature_importance = generateMockFeatureImportance(data);
  const topics = generateMockTopics(datasetType);
  const recommendations = generateMockRecommendations(results, feature_importance, datasetType);
  
  return { results, predictions, feature_importance, topics, recommendations };
}
