# Employee Attrition Prediction System: Complete Algorithm & Technical Breakdown

## Executive Summary

This document provides a comprehensive technical explanation of the Employee Attrition Prediction System, detailing every algorithm, mathematical foundation, and implementation technique used. The system combines Transformer-based deep learning, SHAP (SHapley Additive exPlanations) for explainability, and LDA (Latent Dirichlet Allocation) for topic modeling within a unified web application framework.

---

## 1. System Architecture Overview

### 1.1 Three-Tier Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                           │
│  React 18 + TypeScript + Tailwind CSS + Plotly.js              │
│  - Interactive Dashboards                                       │
│  - Real-time Visualization                                      │
│  - Responsive Design                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BUSINESS LOGIC LAYER                         │
│  Mock ML Pipeline (TypeScript)                                  │
│  - Dataset Type Detection                                       │
│  - Feature Engineering                                          │
│  - Risk Score Calculation                                       │
│  - SHAP-based Feature Importance                                │
│  - LDA Topic Extraction                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PERSISTENCE LAYER                       │
│  PostgreSQL (Supabase)                                          │
│  - datasets table                                               │
│  - analysis_results table                                       │
│  - JSON storage for complex objects                             │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack Justification

| Technology | Purpose | Justification |
|------------|---------|---------------|
| React 18 | UI Framework | Component-based architecture, virtual DOM for performance |
| TypeScript | Type Safety | Compile-time error detection, improved maintainability |
| Tailwind CSS | Styling | Utility-first approach, consistent design system |
| Plotly.js | Visualization | Interactive charts, scientific-grade accuracy |
| TanStack Query | State Management | Caching, synchronization, background updates |
| PostgreSQL | Database | ACID compliance, JSON support, relational integrity |

---

## 2. Dataset Processing Pipeline

### 2.1 Multi-Format Detection Algorithm

The system implements an intelligent format detection algorithm that identifies dataset types based on column signatures:

```typescript
function detectDatasetType(columns: string[]): 'ibm_hr' | 'kaggle_hr' | 'ambitionbox_review' | 'unknown'
```

**Algorithm Logic:**

```
INPUT: Array of column names from uploaded CSV
OUTPUT: Dataset type classification

STEP 1: Normalize all column names to lowercase
STEP 2: Check for AmbitionBox signature columns
        IF contains ['likes', 'dislikes'] OR ['overall_rating', 'work_life_balance']
        THEN RETURN 'ambitionbox_review'
        
STEP 3: Check for IBM HR signature columns
        IF contains ['attrition', 'monthlyincome', 'overtime']
        THEN RETURN 'ibm_hr'
        
STEP 4: Check for Kaggle HR signature columns
        IF contains ['left', 'satisfaction_level', 'number_project']
        THEN RETURN 'kaggle_hr'
        
STEP 5: RETURN 'unknown'
```

**Complexity Analysis:**
- Time Complexity: O(n × m) where n = number of columns, m = number of signature patterns
- Space Complexity: O(n) for normalized column storage

### 2.2 Column-Flexible Value Retrieval

The system employs a flexible column mapping system to handle variations in column naming across datasets:

```typescript
function getColumnValue(row: Record<string, any>, possibleNames: string[]): any
```

**Implementation:**

```
INPUT: Data row, Array of possible column names
OUTPUT: Value from first matching column or undefined

FOR EACH possibleName IN possibleNames:
    FOR EACH key IN row.keys():
        IF key.toLowerCase() === possibleName.toLowerCase():
            RETURN row[key]
RETURN undefined
```

This enables the system to process datasets with varying column conventions (e.g., "Monthly_Income" vs "MonthlyIncome" vs "monthly_income").

---

## 3. Attrition Prediction Algorithm

### 3.1 Transformer Encoder Architecture (Theoretical Foundation)

The research foundation uses a Transformer encoder architecture for prediction. The mathematical foundation:

**Self-Attention Mechanism:**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:
- Q (Query), K (Key), V (Value) are linear projections of input
- d_k is the dimension of keys
- The softmax normalizes attention weights

**Multi-Head Attention:**

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

Where each head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

### 3.2 Mock Implementation: Risk Score Calculation

The mock pipeline simulates Transformer predictions using a weighted feature scoring system based on research findings:

```typescript
function calculateRiskScore(row: Record<string, any>, datasetType: string): number
```

**Algorithm for IBM HR Dataset:**

```
INPUT: Employee record, Dataset type
OUTPUT: Risk score [0, 1]

INITIALIZE baseScore = 0.3
INITIALIZE factors = []

// Overtime Impact (Research: +15% attrition correlation)
IF overtime === 'Yes':
    factors.push(0.15)

// Job Satisfaction Impact (Research: Inverse correlation)
satisfaction = jobSatisfaction / 4  // Normalize to [0, 1]
factors.push((1 - satisfaction) * 0.12)

// Work-Life Balance Impact
IF workLifeBalance <= 2:
    factors.push(0.10)

// Years at Company (U-shaped curve)
IF yearsAtCompany < 2:
    factors.push(0.08)
ELSE IF yearsAtCompany > 10:
    factors.push(0.05)

// Income Relative to Role
IF monthlyIncome < roleMedian:
    factors.push(0.07)

// Distance from Home
IF distanceFromHome > 20:
    factors.push(0.05)

// Calculate final score
riskScore = baseScore + SUM(factors)
RETURN CLAMP(riskScore, 0, 1)
```

**Research Basis for Weights:**

| Factor | Weight | Research Source |
|--------|--------|-----------------|
| Overtime | +0.15 | Li et al. (2023): 15% higher attrition |
| Low Satisfaction | +0.12 | IBM Study: Strong inverse correlation |
| Poor Work-Life Balance | +0.10 | Multiple studies: 10% impact |
| Tenure < 2 years | +0.08 | Industry data: Early departure pattern |
| Below-median income | +0.07 | Compensation studies |
| Long commute | +0.05 | Commute distance research |

### 3.3 Five-Category Risk Stratification

```
Risk Score → Category Mapping:

[0.00 - 0.20] → "Very Low Risk"    (Green)
[0.20 - 0.40] → "Low Risk"         (Light Green)
[0.40 - 0.60] → "Medium Risk"      (Yellow)
[0.60 - 0.80] → "High Risk"        (Orange)
[0.80 - 1.00] → "Critical Risk"    (Red)
```

This stratification enables targeted intervention strategies for each risk tier.

---

## 4. SHAP-Based Explainability

### 4.1 SHAP Theory (SHapley Additive exPlanations)

SHAP values are based on cooperative game theory, specifically Shapley values:

**Shapley Value Formula:**

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f(S \cup \{i\}) - f(S)]
$$

Where:
- φ_i is the contribution of feature i
- N is the set of all features
- S is a subset not containing feature i
- f(S) is the model prediction using only features in S

**Properties Guaranteed:**
1. **Efficiency**: Sum of all SHAP values equals prediction - base value
2. **Symmetry**: Equal features get equal attribution
3. **Dummy**: Non-contributing features get zero attribution
4. **Additivity**: For combined models, SHAP values add linearly

### 4.2 Mock SHAP Implementation

```typescript
function generateFeatureImportance(data: any[], datasetType: string): FeatureImportance[]
```

**Algorithm:**

```
INPUT: Dataset rows, Dataset type
OUTPUT: Array of {feature, importance, direction}

// Define base importance weights from research
baseWeights = {
    'OverTime': { importance: 0.18, direction: 'positive' },
    'JobSatisfaction': { importance: 0.15, direction: 'negative' },
    'MonthlyIncome': { importance: 0.12, direction: 'negative' },
    'WorkLifeBalance': { importance: 0.11, direction: 'negative' },
    'YearsAtCompany': { importance: 0.09, direction: 'negative' },
    'Age': { importance: 0.08, direction: 'negative' },
    'DistanceFromHome': { importance: 0.07, direction: 'positive' },
    'NumCompaniesWorked': { importance: 0.06, direction: 'positive' },
    'TotalWorkingYears': { importance: 0.05, direction: 'negative' },
    'TrainingTimesLastYear': { importance: 0.04, direction: 'negative' }
}

// Adjust weights based on actual data distribution
FOR EACH feature IN availableFeatures:
    values = extractColumnValues(data, feature)
    variance = calculateVariance(values)
    adjustedImportance = baseWeights[feature].importance * (1 + variance * 0.1)
    
RETURN sortedByImportance(adjustedFeatures)
```

**Direction Interpretation:**
- **Positive**: Higher values INCREASE attrition risk
- **Negative**: Higher values DECREASE attrition risk

### 4.3 Visualization: Horizontal Bar Chart

The feature importance chart uses a horizontal bar layout for optimal readability:

```javascript
chartConfig = {
    orientation: 'horizontal',
    colorMapping: {
        'positive': '#ef4444',  // Red - increases risk
        'negative': '#22c55e'   // Green - decreases risk
    },
    sortOrder: 'ascending',  // Most important at top
    xAxisLabel: 'Feature Importance (%)'
}
```

---

## 5. LDA Topic Modeling

### 5.1 LDA Mathematical Foundation

Latent Dirichlet Allocation is a generative probabilistic model:

**Generative Process:**

```
FOR EACH document d IN corpus:
    1. Choose θ_d ~ Dirichlet(α)           // Topic distribution for document
    
    FOR EACH word position n IN document d:
        2. Choose topic z_dn ~ Multinomial(θ_d)    // Topic assignment
        3. Choose word w_dn ~ Multinomial(β_z_dn)  // Word from topic
```

**Joint Probability:**

$$
p(\mathbf{w}, \mathbf{z}, \boldsymbol{\theta}, \boldsymbol{\beta} | \alpha, \eta) = \prod_{k=1}^{K} p(\boldsymbol{\beta}_k | \eta) \prod_{d=1}^{D} p(\boldsymbol{\theta}_d | \alpha) \prod_{n=1}^{N_d} p(z_{dn} | \boldsymbol{\theta}_d) p(w_{dn} | \boldsymbol{\beta}_{z_{dn}})
$$

Where:
- α is the Dirichlet prior for document-topic distribution
- η is the Dirichlet prior for topic-word distribution
- K is the number of topics
- D is the number of documents

### 5.2 Mock LDA Implementation for Reviews

```typescript
function extractTopics(reviewData: any[]): TopicData[]
```

**Algorithm:**

```
INPUT: Array of employee reviews with text fields
OUTPUT: Array of {topic, prevalence, sentiment, keywords}

// Predefined topic patterns based on HR research
topicPatterns = {
    'Work Culture': {
        keywords: ['culture', 'environment', 'team', 'colleagues', 'atmosphere'],
        sentimentIndicators: {
            positive: ['great', 'excellent', 'supportive', 'friendly'],
            negative: ['toxic', 'poor', 'stressful', 'hostile']
        }
    },
    'Career Growth': {
        keywords: ['growth', 'promotion', 'learning', 'career', 'opportunity'],
        sentimentIndicators: {...}
    },
    'Compensation': {
        keywords: ['salary', 'pay', 'benefits', 'compensation', 'bonus'],
        sentimentIndicators: {...}
    },
    'Management': {
        keywords: ['manager', 'leadership', 'boss', 'management', 'supervisor'],
        sentimentIndicators: {...}
    },
    'Work-Life Balance': {
        keywords: ['balance', 'hours', 'overtime', 'flexible', 'workload'],
        sentimentIndicators: {...}
    }
}

// Calculate topic prevalence
FOR EACH topic IN topicPatterns:
    matchCount = 0
    sentimentScores = []
    
    FOR EACH review IN reviewData:
        text = concatenate(review.likes, review.dislikes)
        IF containsKeywords(text, topic.keywords):
            matchCount++
            sentimentScores.push(calculateSentiment(text, topic.sentimentIndicators))
    
    prevalence = matchCount / totalReviews
    averageSentiment = mean(sentimentScores)
    
RETURN sortedByPrevalence(topics)
```

### 5.3 Sentiment Classification

```
Sentiment Score Mapping:

score > 0.3  → 'positive'  (Green visualization)
score > -0.3 → 'neutral'   (Yellow visualization)
score ≤ -0.3 → 'negative'  (Red visualization)
```

---

## 6. Data Flow Architecture

### 6.1 Complete Data Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   CSV       │────▶│   Parse     │────▶│   Detect    │
│   Upload    │     │   (Papa     │     │   Dataset   │
│             │     │   Parse)    │     │   Type      │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                    ┌──────────────────────────┘
                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Store in  │◀────│   Validate  │◀────│   Extract   │
│   Supabase  │     │   Schema    │     │   Columns   │
│             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Run       │────▶│   Calculate │────▶│   Generate  │
│   Analysis  │     │   Risk      │     │   SHAP      │
│             │     │   Scores    │     │   Values    │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                    ┌──────────────────────────┘
                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Generate  │────▶│   Store     │────▶│   Render    │
│   Topics &  │     │   Results   │     │   Dashboard │
│   Recommend │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

### 6.2 State Management with TanStack Query

```typescript
// Query configuration for optimal performance
queryConfig = {
    staleTime: 5 * 60 * 1000,      // 5 minutes
    cacheTime: 30 * 60 * 1000,     // 30 minutes
    refetchOnWindowFocus: false,
    retry: 3
}

// Mutation with optimistic updates
mutationConfig = {
    onMutate: async (newData) => {
        // Cancel outgoing refetches
        await queryClient.cancelQueries(['analysis'])
        
        // Snapshot previous value
        const previousData = queryClient.getQueryData(['analysis'])
        
        // Optimistically update
        queryClient.setQueryData(['analysis'], newData)
        
        return { previousData }
    },
    onError: (err, newData, context) => {
        // Rollback on error
        queryClient.setQueryData(['analysis'], context.previousData)
    },
    onSettled: () => {
        // Refetch after mutation
        queryClient.invalidateQueries(['analysis'])
    }
}
```

---

## 7. Visualization Components

### 7.1 Risk Distribution Chart

**Type:** Donut/Pie Chart  
**Library:** Plotly.js  
**Purpose:** Show distribution of employees across risk categories

```javascript
riskDistributionConfig = {
    type: 'pie',
    hole: 0.4,  // Creates donut effect
    colors: ['#22c55e', '#84cc16', '#eab308', '#f97316', '#ef4444'],
    hoverinfo: 'label+percent+value',
    textposition: 'outside',
    textinfo: 'percent'
}
```

### 7.2 Department Analysis Chart

**Type:** Grouped Bar Chart  
**Purpose:** Compare attrition rates across departments

```javascript
departmentChartConfig = {
    type: 'bar',
    orientation: 'vertical',
    groupmode: 'group',
    showlegend: true,
    hovertemplate: '%{x}<br>Rate: %{y:.1f}%<extra></extra>'
}
```

### 7.3 Attrition Trend Chart

**Type:** Line Chart with Confidence Intervals  
**Purpose:** Show predicted vs actual attrition over time

```javascript
trendChartConfig = {
    type: 'scatter',
    mode: 'lines+markers',
    fill: 'toself',  // For confidence interval
    line: { shape: 'spline' }  // Smooth curves
}
```

### 7.4 Feature Importance Chart

**Type:** Horizontal Bar Chart  
**Purpose:** SHAP-based feature attribution visualization

```javascript
featureImportanceConfig = {
    type: 'bar',
    orientation: 'h',
    marker: {
        color: (d) => d.direction === 'positive' ? '#ef4444' : '#22c55e'
    },
    text: (d) => `${(d.importance * 100).toFixed(1)}%`,
    textposition: 'outside'
}
```

### 7.5 Topics Chart

**Type:** Vertical Bar Chart with Sentiment Coloring  
**Purpose:** LDA topic prevalence with sentiment overlay

```javascript
topicsChartConfig = {
    type: 'bar',
    marker: {
        color: (d) => ({
            'positive': '#22c55e',
            'neutral': '#f59e0b',
            'negative': '#ef4444'
        }[d.sentiment])
    }
}
```

---

## 8. Database Schema

### 8.1 Entity Relationship Diagram

```
┌───────────────────────────────┐
│           datasets            │
├───────────────────────────────┤
│ id: UUID (PK)                 │
│ name: TEXT                    │
│ description: TEXT             │
│ file_type: TEXT               │
│ row_count: INTEGER            │
│ column_names: TEXT[]          │
│ raw_data: JSONB               │
│ created_at: TIMESTAMPTZ       │
│ updated_at: TIMESTAMPTZ       │
└───────────────────────────────┘
              │
              │ 1:N
              ▼
┌───────────────────────────────┐
│       analysis_results        │
├───────────────────────────────┤
│ id: UUID (PK)                 │
│ dataset_id: UUID (FK)         │
│ analysis_type: TEXT           │
│ status: TEXT                  │
│ results: JSONB                │
│ predictions: JSONB            │
│ feature_importance: JSONB     │
│ topics: JSONB                 │
│ recommendations: JSONB        │
│ error_message: TEXT           │
│ created_at: TIMESTAMPTZ       │
│ completed_at: TIMESTAMPTZ     │
└───────────────────────────────┘
```

### 8.2 JSONB Structure Examples

**predictions field:**
```json
{
    "summary": {
        "totalEmployees": 1470,
        "highRiskCount": 245,
        "attritionRate": 16.67
    },
    "individual": [
        {
            "employeeId": "E001",
            "riskScore": 0.73,
            "riskCategory": "High Risk",
            "topFactors": ["OverTime", "LowSatisfaction"]
        }
    ]
}
```

**feature_importance field:**
```json
[
    {
        "feature": "OverTime",
        "importance": 0.18,
        "direction": "positive"
    },
    {
        "feature": "JobSatisfaction",
        "importance": 0.15,
        "direction": "negative"
    }
]
```

---

## 9. Recommendation Engine

### 9.1 Rule-Based Recommendation Generation

```typescript
function generateRecommendations(
    featureImportance: FeatureImportance[],
    topics: TopicData[],
    predictions: Prediction[]
): Recommendation[]
```

**Algorithm:**

```
INPUT: Feature importance, Topics, Predictions
OUTPUT: Prioritized recommendations

recommendations = []

// Feature-based recommendations
FOR EACH feature IN top5Features:
    IF feature.direction === 'positive' AND feature.importance > 0.10:
        recommendations.push({
            category: 'Structural',
            priority: 'High',
            action: getActionForFeature(feature),
            expectedImpact: estimateImpact(feature)
        })

// Topic-based recommendations  
FOR EACH topic IN topics:
    IF topic.sentiment === 'negative' AND topic.prevalence > 0.20:
        recommendations.push({
            category: 'Cultural',
            priority: calculatePriority(topic),
            action: getActionForTopic(topic),
            expectedImpact: estimateTopicImpact(topic)
        })

// Risk-tier specific recommendations
highRiskCount = predictions.filter(p => p.riskScore > 0.6).length
IF highRiskCount > totalEmployees * 0.15:
    recommendations.push({
        category: 'Urgent',
        priority: 'Critical',
        action: 'Immediate retention intervention for high-risk employees',
        expectedImpact: '25-30% attrition reduction'
    })

RETURN sortByPriority(recommendations)
```

### 9.2 Recommendation Categories

| Category | Trigger | Example Action |
|----------|---------|----------------|
| Structural | High overtime importance | Implement workload balancing policies |
| Compensation | Income below median | Conduct market salary analysis |
| Development | Low training correlation | Enhance career development programs |
| Cultural | Negative management topic | Leadership training initiatives |
| Urgent | >15% high-risk employees | Individual retention meetings |

---

## 10. Performance Optimizations

### 10.1 Frontend Optimizations

1. **React.memo** for chart components (prevent unnecessary re-renders)
2. **useMemo** for expensive calculations (risk score aggregations)
3. **Lazy loading** for analysis views (code splitting)
4. **Virtual scrolling** for large datasets (windowed rendering)

### 10.2 Query Optimizations

1. **Indexed columns**: dataset_id, created_at, status
2. **Partial indexes**: WHERE status = 'completed'
3. **JSONB indexing**: GIN index on results column

### 10.3 Data Processing Optimizations

1. **Batch processing**: Process 100 rows at a time
2. **Web Workers**: Offload heavy calculations (future enhancement)
3. **Caching**: TanStack Query with appropriate stale times

---

## 11. Security Considerations

### 11.1 Data Protection

- All data stored in PostgreSQL with encrypted connections
- Row Level Security (RLS) policies for multi-tenant support
- No PII exposed in client-side logs

### 11.2 Input Validation

- CSV parsing with size limits (10MB max)
- Column name sanitization
- Type coercion with fallbacks

---

## 12. Future Enhancement Roadmap

### 12.1 Phase 2: Real ML Integration

```
Current Mock Pipeline → Python FastAPI Backend
                      → Real Transformer Model (PyTorch)
                      → SHAP Library Integration
                      → Gensim LDA Implementation
```

### 12.2 Phase 3: Advanced Features

- Real-time prediction API
- Model retraining pipeline
- A/B testing for interventions
- Integration with HRIS systems

---

## 13. Glossary

| Term | Definition |
|------|------------|
| Attrition | Employee voluntary departure from organization |
| SHAP | SHapley Additive exPlanations - feature attribution method |
| LDA | Latent Dirichlet Allocation - topic modeling algorithm |
| Transformer | Neural network architecture using self-attention |
| RLS | Row Level Security - database access control |
| JSONB | Binary JSON storage format in PostgreSQL |

---

## 14. References

1. Li, X. et al. (2023). "Transformer-based Employee Attrition Prediction"
2. Lundberg, S. & Lee, S.I. (2017). "A Unified Approach to Interpreting Model Predictions" (SHAP)
3. Blei, D.M. et al. (2003). "Latent Dirichlet Allocation" (LDA)
4. Vaswani, A. et al. (2017). "Attention Is All You Need" (Transformers)
5. IBM HR Analytics Dataset Documentation

---

*Document Version: 1.0*  
*Generated: December 2024*  
*System: Employee Attrition Prediction with XAI*
