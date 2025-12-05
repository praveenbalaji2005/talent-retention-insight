# Employee Attrition Prediction System - Technical Documentation

## Project Overview

This project implements an **Explainable AI (XAI) powered Employee Attrition Prediction System** based on the research paper *"Predicting Employee Attrition: XAI-Powered Models for Managerial Decision-Making"* by Baydili & Tasci (2025).

### Key Innovation
The system is **column-flexible** - it works with multiple HR dataset formats (IBM HR, Kaggle HR Analytics, AmbitionBox Reviews) and automatically adapts its analysis based on available columns.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React + Vite)                   │
├─────────────────────────────────────────────────────────────────┤
│  Pages          │  Components        │  Hooks                   │
│  - Index.tsx    │  - Dashboard       │  - useDatasets.ts        │
│                 │  - Analysis        │  - useAnalysis.ts        │
│                 │  - Datasets        │                          │
│                 │  - Charts (Plotly) │                          │
├─────────────────────────────────────────────────────────────────┤
│                     Mock ML Pipeline                             │
│                  (src/lib/mockAnalysis.ts)                       │
├─────────────────────────────────────────────────────────────────┤
│                   BACKEND (Lovable Cloud)                        │
│              PostgreSQL Database via Supabase                    │
│  - datasets table (stores uploaded CSV data)                     │
│  - analysis_results table (stores predictions & insights)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend Framework | **React 18** | Component-based UI |
| Build Tool | **Vite** | Fast development & bundling |
| Styling | **Tailwind CSS** | Utility-first CSS framework |
| UI Components | **shadcn/ui** | Accessible React components |
| Charts | **Plotly.js** | Interactive data visualizations |
| CSV Parsing | **PapaParse** | Parse CSV files in browser |
| State Management | **TanStack Query** | Server state & caching |
| Database | **PostgreSQL** (Supabase) | Data persistence |
| Type Safety | **TypeScript** | Static type checking |

---

## File Structure & Modules

### 1. Core Application Files

```
src/
├── pages/
│   └── Index.tsx           # Main page with view routing
├── components/
│   ├── layout/
│   │   └── Header.tsx      # Navigation header
│   ├── dashboard/
│   │   ├── Dashboard.tsx   # Main dashboard view
│   │   └── StatCard.tsx    # Statistics display cards
│   ├── datasets/
│   │   ├── DatasetsView.tsx    # Dataset management page
│   │   ├── DatasetUpload.tsx   # CSV upload with format detection
│   │   └── DatasetList.tsx     # List uploaded datasets
│   ├── analysis/
│   │   ├── AnalysisView.tsx    # Full analysis page
│   │   ├── PipelineStatus.tsx  # ML pipeline progress indicator
│   │   └── RecommendationsPanel.tsx  # HR recommendations
│   └── charts/
│       ├── RiskDistributionChart.tsx   # Risk level pie chart
│       ├── DepartmentChart.tsx         # Department bar chart
│       ├── FeatureImportanceChart.tsx  # SHAP-like bar chart
│       ├── TopicsChart.tsx             # LDA topic visualization
│       └── AttritionChart.tsx          # Risk score histogram
├── hooks/
│   ├── useDatasets.ts      # Dataset CRUD operations
│   └── useAnalysis.ts      # Analysis operations
├── lib/
│   ├── mockAnalysis.ts     # Mock ML pipeline (core logic)
│   └── utils.ts            # Utility functions
└── types/
    └── dataset.ts          # TypeScript interfaces
```

---

## Key Modules Explained

### 1. Mock ML Pipeline (`src/lib/mockAnalysis.ts`)

This is the **core intelligence** of the system. It simulates what a real Python ML backend would do.

#### Dataset Type Detection
```typescript
function detectDatasetType(data: Record<string, unknown>[]): 'ibm' | 'kaggle' | 'ambitionbox' | 'unknown'
```
- Analyzes column names to identify dataset format
- Checks for IBM columns (JobSatisfaction, Attrition, MonthlyIncome)
- Checks for Kaggle columns (satisfaction_level, number_project)
- Checks for AmbitionBox columns (Overall_rating, work_life_balance)

#### Risk Score Calculation
```typescript
function calculateRiskScore(row: Record<string, unknown>, datasetType: string): number
```
Based on the research paper's SHAP findings:

**For IBM Dataset:**
- Job Satisfaction (most important, -28% weight)
- Age (younger = higher risk)
- Overtime (Yes = +12% risk)
- Years at Company (shorter tenure = higher risk)

**For Kaggle Dataset:**
- Number of Projects (>6 projects = +20% risk)
- Satisfaction Level (most important)
- Time Spent at Company (5-6 years = elevated risk)

**For AmbitionBox Reviews:**
- Overall Rating (1-5 scale, most important)
- Work Satisfaction
- Work-Life Balance
- Career Growth
- Job Security

#### Key Functions

| Function | Purpose |
|----------|---------|
| `generateMockPredictions()` | Creates risk predictions for each employee |
| `generateMockResults()` | Aggregates statistics (total, at-risk, rate) |
| `generateMockFeatureImportance()` | Returns SHAP-like feature importance |
| `generateMockTopics()` | Returns LDA-like topic modeling results |
| `generateMockRecommendations()` | Generates HR action recommendations |
| `runMockAnalysis()` | Main function that orchestrates all analysis |

---

### 2. Data Hooks (`src/hooks/`)

#### useDatasets.ts
```typescript
// Fetch all datasets
export function useDatasets() → useQuery(['datasets'])

// Fetch single dataset
export function useDataset(id) → useQuery(['dataset', id])

// Upload new dataset
export function useUploadDataset() → useMutation

// Delete dataset
export function useDeleteDataset() → useMutation
```

#### useAnalysis.ts
```typescript
// Get all analysis results for a dataset
export function useAnalysisResults(datasetId) → useQuery(['analysis', datasetId])

// Get latest completed analysis
export function useLatestAnalysis(datasetId) → useQuery(['analysis', datasetId, 'latest'])

// Run new analysis
export function useRunAnalysis() → useMutation
```

Uses **TanStack Query** for:
- Automatic caching
- Background refetching
- Loading states
- Error handling

---

### 3. Dataset Upload (`src/components/datasets/DatasetUpload.tsx`)

**Features:**
1. **Drag & Drop** - Users can drag CSV files
2. **Format Detection** - Automatically detects IBM/Kaggle/AmbitionBox format
3. **Column Preview** - Shows detected columns with highlighting
4. **Validation** - Ensures file is valid CSV with data

```typescript
// Format detection based on column names
const detectDatasetFormat = (cols: string[]) => {
  // Check for AmbitionBox columns
  if (cols.includes('overall_rating', 'work_life_balance')) 
    return 'ambitionbox';
  // Check for Kaggle columns
  if (cols.includes('satisfaction_level', 'number_project')) 
    return 'kaggle';
  // Check for IBM columns
  if (cols.includes('jobsatisfaction', 'attrition')) 
    return 'ibm';
  return 'generic';
};
```

---

### 4. Charts (Plotly.js)

#### RiskDistributionChart
- **Type:** Pie/Donut Chart
- **Data:** Low, Medium, High, Critical risk counts
- **Colors:** Green → Yellow → Orange → Red

#### DepartmentChart
- **Type:** Horizontal Bar Chart
- **Data:** Department name vs At-risk count
- **Shows:** Which departments have highest attrition risk

#### FeatureImportanceChart
- **Type:** Horizontal Bar Chart
- **Data:** SHAP-like feature importance values
- **Colors:** Red for positive impact (increases risk), Blue for negative

#### TopicsChart
- **Type:** Horizontal Bar Chart
- **Data:** LDA topic prevalence from text analysis
- **Shows:** Key themes from employee reviews (Work-Life Balance, Career Growth, etc.)

---

### 5. Database Schema

#### datasets Table
```sql
CREATE TABLE datasets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  description TEXT,
  file_type TEXT NOT NULL,  -- 'attrition' or 'reviews'
  raw_data JSONB NOT NULL,  -- Actual CSV data as JSON array
  column_names TEXT[] NOT NULL,
  row_count INTEGER NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);
```

#### analysis_results Table
```sql
CREATE TABLE analysis_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  dataset_id UUID REFERENCES datasets(id),
  analysis_type TEXT NOT NULL,  -- 'full_pipeline'
  status TEXT DEFAULT 'pending',  -- 'pending', 'processing', 'completed', 'failed'
  results JSONB,  -- Aggregated statistics
  predictions JSONB,  -- Per-employee predictions
  feature_importance JSONB,  -- SHAP values
  topics JSONB,  -- LDA topics
  recommendations JSONB,  -- HR recommendations
  error_message TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  completed_at TIMESTAMPTZ
);
```

---

## Research Paper Implementation

### Based on: "Predicting Employee Attrition: XAI-Powered Models for Managerial Decision-Making"

#### Paper's Key Contributions Implemented:

1. **GAN-based Synthetic Data** (Simulated)
   - Paper uses GANs to handle class imbalance
   - Our system simulates this by not requiring balanced data

2. **Transformer Encoder Architecture** (Simulated)
   - Paper uses 3-layer Transformer for classification
   - Our mock pipeline returns predictions that mimic Transformer outputs

3. **SHAP Analysis** (Implemented)
   - Paper identifies key features via SHAP values
   - Our system uses paper's findings:
     - **IBM Dataset:** JobSatisfaction (28%), Age (18%), YearsWithCurrManager (15%)
     - **Kaggle Dataset:** number_project (32%), satisfaction_level (26%), time_spend_company (15%)

4. **Performance Metrics** (Displayed)
   - Paper achieves 92% accuracy on IBM, 96.95% on Kaggle
   - Our system displays these as reference benchmarks

---

## Column-Flexible Analysis

### How It Works

```typescript
// 1. Detect what columns exist
const datasetType = detectDatasetType(data);

// 2. Find column by multiple possible names
function getColumnValue(row, candidates) {
  // candidates = ['JobSatisfaction', 'job_satisfaction', 'Job_Satisfaction']
  for (const candidate of candidates) {
    if (row has candidate) return row[candidate];
  }
  return undefined;
}

// 3. Calculate risk only if column exists
if (satisfaction !== undefined) {
  const satValue = Number(satisfaction);
  if (satValue <= 1) baseScore += 0.25;
  else if (satValue <= 2) baseScore += 0.15;
  // ...
}
```

### Supported Column Mappings

| Feature | IBM Columns | Kaggle Columns | AmbitionBox Columns |
|---------|-------------|----------------|---------------------|
| Satisfaction | JobSatisfaction | satisfaction_level | Overall_rating, work_satisfaction |
| Work-Life | WorkLifeBalance | average_monthly_hours | work_life_balance |
| Tenure | YearsAtCompany | time_spend_company | - |
| Compensation | MonthlyIncome | salary | salary_and_benefits |
| Department | Department | Department | Department |

---

## UI/UX Design System

### Color Palette (HSL)
```css
--primary: 222 47% 20%;      /* Deep Navy */
--secondary: 174 62% 47%;    /* Teal */
--accent: 38 92% 50%;        /* Amber */
--destructive: 0 84% 60%;    /* Red (High Risk) */
--success: 152 69% 40%;      /* Green (Low Risk) */
--warning: 25 95% 53%;       /* Orange (Medium Risk) */
```

### Typography
- **Primary Font:** Inter (sans-serif)
- **Monospace:** JetBrains Mono (for code/data)

### Components Used
- Cards, Buttons, Badges, Tabs (shadcn/ui)
- Custom StatCard with icon and color variants
- Interactive Plotly.js charts

---

## Data Flow

```
1. User uploads CSV file
   ↓
2. PapaParse parses CSV to JSON array
   ↓
3. System detects dataset format (IBM/Kaggle/AmbitionBox)
   ↓
4. Data stored in PostgreSQL (datasets table)
   ↓
5. User clicks "Run Analysis"
   ↓
6. Mock ML pipeline processes data:
   - generateMockPredictions() → Risk scores per employee
   - generateMockResults() → Aggregate statistics
   - generateMockFeatureImportance() → SHAP values
   - generateMockTopics() → Topic modeling
   - generateMockRecommendations() → HR actions
   ↓
7. Results stored in analysis_results table
   ↓
8. Dashboard displays interactive visualizations
```

---

## API Contracts (For Future Python Backend)

The system is designed with **External API Ready** architecture. Here are the contracts:

### Prediction Endpoint
```typescript
POST /api/predict
Request: { dataset_id: string, raw_data: object[] }
Response: {
  predictions: [{
    employee_id: string | number,
    department: string,
    risk_score: number (0-100),
    risk_level: 'low' | 'medium' | 'high' | 'critical',
    attrition_probability: number (0-1),
    factors: string[]
  }]
}
```

### Feature Importance Endpoint
```typescript
POST /api/explain
Request: { dataset_id: string }
Response: {
  feature_importance: [{
    feature: string,
    importance: number (0-1),
    direction: 'positive' | 'negative',
    description: string
  }]
}
```

---

## Testing the System

### Supported Datasets

1. **IBM HR Analytics** (`WA_Fn-UseC_-HR-Employee-Attrition.csv`)
   - 1,470 records, 35 columns
   - Required: Attrition, JobSatisfaction, Department

2. **Kaggle HR Analytics**
   - ~15,000 records
   - Required: left, satisfaction_level, number_project

3. **AmbitionBox Reviews** (Amazon, Capgemini, Mahindra, Tata, Maruti)
   - Varies (7,000 - 32,000 records)
   - Required: Overall_rating, work_life_balance

### Test Flow
1. Go to Datasets → Upload Dataset
2. Drop a CSV file
3. Verify format detection (badge shows IBM/Kaggle/AmbitionBox)
4. Click "Upload & Analyze"
5. Go to Analysis → Select dataset → Run Analysis
6. View results in Dashboard, Analysis tabs

---

## Summary

| Aspect | Implementation |
|--------|----------------|
| **Framework** | React + Vite + TypeScript |
| **Database** | PostgreSQL via Supabase |
| **Charts** | Plotly.js (interactive) |
| **ML Approach** | Mock pipeline (API-ready for real Python backend) |
| **Key Feature** | Column-flexible analysis |
| **Research Basis** | Baydili & Tasci (2025) XAI paper |
| **SHAP Analysis** | Based on paper's findings |
| **Supported Formats** | IBM HR, Kaggle HR, AmbitionBox Reviews |

---

## References

1. Baydili, I.T.; Tasci, B. "Predicting Employee Attrition: XAI-Powered Models for Managerial Decision-Making." Systems 2025, 13, 583.

2. IBM HR Analytics Dataset: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

3. Kaggle HR Analytics Dataset: https://www.kaggle.com/datasets/giripujar/hr-analytics

---

*Document generated for project review presentation*
