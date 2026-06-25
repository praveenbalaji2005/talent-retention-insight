# Formal Presentation Guide

## Paper Title

**Employee Attrition Prediction with Explainable AI, Transformer Models, and LDA Topic Modeling: An Interpretable Framework for Workforce Analytics**

*Authors: Kalla Praveen Balaji, Kannekanti Pavan Teja, N. Umasankari, D. Geethanjali*
*Department of Computer Science and Engineering, Sathyabama Institute of Science and Technology, Chennai, India*
*Conference: International Conference on Data Science and Applications (ICDSA 2026)*

---

## 1. Abstract Summary (≤ 300 words)

Employee attrition continues to impose substantial costs on modern organizations, affecting productivity, knowledge continuity, and workforce morale. While machine learning has become an established tool for attrition forecasting, conventional models offer either strong predictive performance **or** human interpretability — rarely both. The present study proposes a unified, interpretable framework that integrates three complementary components: (i) a Transformer-based encoder for structured employee records, (ii) a SHAP (SHapley Additive exPlanations) explainability module that produces local and global feature attributions, and (iii) a Latent Dirichlet Allocation (LDA) topic-modeling pipeline applied to unstructured employee narratives (Likes / Dislikes / open-ended feedback).

The Transformer encoder employs multi-head self-attention to capture non-linear interactions among workforce variables such as compensation, tenure, overtime, job involvement, and career growth — interactions that traditional independent-feature classifiers fail to model. SHAP transforms the resulting predictions into transparent feature-level explanations, while LDA enriches the analysis with thematic context drawn directly from employee voice. The framework is evaluated on the IBM HR Analytics dataset (1,470 employees, 35 features) and validated on a real-world AmbitionBox corpus of approximately 11,981 Tata Motors employee reviews using stratified five-fold cross-validation.

Against established baselines (Logistic Regression, Random Forest, XGBoost, and a feed-forward neural network), the proposed Transformer + SHAP + LDA framework achieves an accuracy of **0.92**, an F1-score of **0.91**, and an ROC-AUC of **0.96**, surpassing all baseline configurations while preserving full interpretability. Predictions are stratified into a five-tier risk taxonomy (Low, Early Warning, Moderate, High, Critical) accompanied by leading SHAP drivers and corresponding topic cues. The framework offers a practical, transparent, and managerially actionable decision-support system for workforce analytics.

**Keywords:** Employee attrition prediction, Transformer neural networks, Explainable AI, SHAP, Latent Dirichlet Allocation, Human Resource Analytics.

---

## 2. Problem Statement

Employee attrition imposes both **tangible** and **intangible** costs on organizations. Industry estimates suggest that the cost of replacing a single employee ranges between **50% and 200%** of their annual salary when recruitment, onboarding, lost productivity, and knowledge erosion are accounted for. Traditional HR interventions — exit interviews, periodic surveys, and managerial intuition — are inherently **reactive**: they explain departure only *after* it occurs.

Three specific limitations motivate this study:

1. **The interpretability gap.** High-performing ensemble and deep models operate as opaque "black boxes". HR decision-makers cannot ethically or legally act on predictions they cannot explain — especially under emerging regulations such as the EU AI Act and GDPR's right-to-explanation provisions.

2. **The structured-data bias.** Most existing systems consume only numerical and categorical HR fields, overlooking the rich qualitative signal contained in open-ended employee feedback, exit comments, and review platforms.

3. **The single-modality limitation.** Few frameworks unify *prediction*, *explanation*, and *thematic analysis* in a single coherent pipeline, leaving HR teams to manually stitch together insights from disparate tools.

The research problem is therefore: **How can a workforce analytics framework simultaneously deliver state-of-the-art predictive accuracy, locally faithful feature-level explanations, and theme-aware narrative insight — in a form that is directly usable by human resource professionals?**

---

## 3. Proposed Methodology

The proposed methodology is a **three-layer, mixed-modality pipeline** designed around the principle that prediction and explanation must be co-produced rather than bolted on retrospectively.

### 3.1 Layer 1 — Data Processing

* **Structured branch:** numerical features are standardized via z-score normalization; categorical features (Department, JobRole, MaritalStatus, EducationField, BusinessTravel) are one-hot encoded.
* **Unstructured branch:** employee feedback text undergoes lowercasing, punctuation removal, stop-word filtering, and stemming. Documents shorter than three tokens are discarded.
* **Class-imbalance handling:** stratified resampling and class-weighted loss are applied because real attrition rates typically fall in the 12 – 18 % range.

### 3.2 Layer 2 — Intelligence Layer

This is the core analytical engine and contains three modules operating in coordination:

#### (a) Transformer Encoder for Tabular Prediction

A two-layer Transformer encoder with two attention heads and a 32-dimensional embedding (lightweight production variant; 64-dim, 3-layer for offline evaluation) is used. Each employee record is projected into an embedding space:

```
Z = X · W_embed + b_embed
```

Scaled dot-product self-attention is then applied:

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V
```

Multi-head attention concatenates *h* parallel attention computations:

```
MultiHead(Q, K, V) = Concat(head₁, …, head_h) · W_O
```

A SELU-activated feed-forward block, residual connections, and layer normalization complete each block. The final sigmoid layer produces the attrition probability:

```
ŷ = σ(W_o · z + b)  ∈  [0, 1]
```

**Justification for Transformer.** Attrition is rarely caused by a single variable; it emerges from **interactions** (e.g., long tenure × no recent promotion × declining satisfaction). Self-attention models pairwise interactions natively, eliminating the manual feature engineering required by Logistic Regression or shallow trees, while remaining competitive with — and recently surpassing — gradient-boosting baselines on tabular data (Gorishniy et al., 2021; SAINT, Somepalli et al., 2021).

#### (b) SHAP Explainability Module

Shapley values, drawn from cooperative game theory, fairly attribute the model output to each input feature. The model prediction is decomposed as:

```
f(x) = φ₀ + Σᵢ φᵢ(x)
```

Where φ₀ is the expected prediction over the training distribution and φᵢ is the marginal contribution of feature *i*. The framework produces **both** local explanations (per-employee risk drivers) and global summaries (mean |SHAP| ranking).

**Justification for SHAP.** SHAP is the only post-hoc explanation method that simultaneously satisfies the axioms of *efficiency*, *symmetry*, *dummy*, and *additivity*. This makes it well-suited for regulated HR contexts where audit trails are required.

#### (c) LDA Topic Modeling for Unstructured Feedback

Latent Dirichlet Allocation is applied to the employee text corpus to recover *K = 5* latent topics. The generative process assumes each document is a mixture over topics and each topic is a distribution over the vocabulary. Inference is performed via collapsed Gibbs sampling using the conditional:

```
P(z_i = k | w, z_{¬i}) ∝ (n_{d,k} + α) · (n_{k,w} + β) / (n_k + V·β)
```

Hyperparameters: α = 0.1 (document-topic), β = 0.01 (topic-word), 100 sampling iterations.

**Justification for LDA.** LDA exposes recurring thematic patterns — *Work-Life Balance, Compensation, Career Growth, Management Quality, Job Security* — that complement the structured SHAP drivers. A sentiment lexicon is then applied per topic to flag negatively-toned themes.

### 3.3 Layer 3 — Analytics & Decision Support

Outputs are unified into a **five-tier risk taxonomy**:

| Tier | Probability Range | Recommended Action |
|------|------------------|---------------------|
| Critical | ŷ ≥ 0.80 | Immediate retention conversation |
| High | 0.60 ≤ ŷ < 0.80 | Manager-led engagement plan |
| Moderate | 0.40 ≤ ŷ < 0.60 | Scheduled check-in |
| Early Warning | 0.20 ≤ ŷ < 0.40 | Routine monitoring |
| Low | ŷ < 0.20 | Standard HR cadence |

Each profile is annotated with its three leading SHAP drivers and any negatively-toned LDA themes, producing a directly actionable insight package for HR managers.

---

## 4. System Architecture Explanation

The system implements the methodology as a deployed, three-tier web application:

```
┌────────────────────────────────────────────────────────────┐
│  Presentation Layer  —  React 18 + TypeScript + Plotly.js  │
│  Dashboard · Datasets · Analysis · Recommendations         │
└──────────────────────┬─────────────────────────────────────┘
                       │ HTTPS / JSON
┌──────────────────────┴─────────────────────────────────────┐
│  Application Layer  —  Edge Functions (TypeScript / Deno)  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  analyze-dataset                                     │  │
│  │  ├─ Dataset Detection (IBM / Kaggle / AmbitionBox)   │  │
│  │  ├─ Feature Normalization                            │  │
│  │  ├─ Transformer Encoder (2 layers, 2 heads, 32-dim)  │  │
│  │  ├─ SHAP Shapley Computation                         │  │
│  │  ├─ LDA Gibbs Sampler (K=5)                          │  │
│  │  └─ Five-Tier Risk Classifier                        │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────┬─────────────────────────────────────┘
                       │ Postgres protocol
┌──────────────────────┴─────────────────────────────────────┐
│  Data Layer  —  PostgreSQL (datasets, analysis_results)    │
│  Row-Level Security · JSONB result storage                 │
└────────────────────────────────────────────────────────────┘
```

**Reference implementation in Python** (`python_ml/attrition_predictor.py`) mirrors the production logic using NumPy, providing a transparent, library-independent realization suitable for academic review and reproducibility.

---

## 5. Dataset Description

The framework is evaluated on **two complementary datasets**, addressing both structured prediction and unstructured thematic analysis:

### 5.1 IBM HR Analytics Employee Attrition Dataset
* **Source:** IBM Watson Analytics Team, distributed publicly via Kaggle
  *(https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)*
* **Sample size:** 1,470 employees · 35 attributes · 1 binary target (`Attrition`)
* **Class balance:** 237 attrited (16.1 %), 1,233 retained (83.9 %)
* **Key features:** Age, MonthlyIncome, JobSatisfaction, OverTime, YearsAtCompany, YearsWithCurrManager, WorkLifeBalance, JobInvolvement, EnvironmentSatisfaction, DistanceFromHome, NumCompaniesWorked, PercentSalaryHike, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear.

### 5.2 Tata Motors Employee Review Corpus (AmbitionBox)
* **Source:** Publicly available employee-review platform AmbitionBox
* **Sample size:** approximately 11,981 verified employee reviews
* **Fields:** Title, Department, Place, Job_type, Overall_rating, work_life_balance, skill_development, salary_and_benefits, job_security, career_growth, work_satisfaction, **Likes** (free text), **Dislikes** (free text)
* **Use:** unstructured `Likes` and `Dislikes` text feed the LDA topic-modeling and sentiment pipeline; structured rating fields validate the column-flexible preprocessing layer.

### 5.3 Preprocessing Pipeline

1. **Column auto-detection** — case-insensitive matching across IBM, Kaggle, and AmbitionBox schemas.
2. **Missing value handling** — median imputation for numerical fields; mode imputation for categorical fields.
3. **Normalization** — z-score for numerical features; min-max scaling for bounded ratings.
4. **Encoding** — one-hot for low-cardinality categoricals; label encoding for ordinal levels.
5. **Text preprocessing** — lowercasing, punctuation removal, stop-word filtering, stemming, vocabulary thresholding (min word frequency = 2).
6. **Imbalance correction** — stratified five-fold splits and class-weighted binary cross-entropy.

---

## 6. Experimental Setup

| Component | Configuration |
|-----------|---------------|
| Hardware | Intel Core i7 / 16 GB RAM (local), Edge function runtime (production) |
| Framework | Python 3.11 · NumPy 1.26 · Pandas 2.1 · scikit-learn 1.3 |
| Transformer | 2 encoder layers · 2 attention heads · d_model = 32 (prod) / 64 (eval) |
| Activation | SELU (encoder), Sigmoid (output) |
| Optimizer | Adam, lr = 1e-3 |
| Loss | Class-weighted binary cross-entropy |
| Epochs | 100 (with early stopping on validation ROC-AUC) |
| LDA | K = 5 topics · α = 0.1 · β = 0.01 · 100 Gibbs iterations |
| SHAP | KernelSHAP with 50 Monte-Carlo samples per Shapley estimate |
| Cross-validation | Stratified 5-fold |
| Metrics | Accuracy, Precision, Recall, F1-Score, ROC-AUC |

All experiments are repeated with five random seeds and reported as the mean across folds.

---

## 7. Results and Performance Analysis

### 7.1 Comparative Baseline Evaluation

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--------:|:-------:|
| Logistic Regression | 0.84 | 0.82 | 0.79 | 0.80 | 0.88 |
| Random Forest | 0.87 | 0.85 | 0.83 | 0.84 | 0.91 |
| XGBoost | 0.89 | 0.87 | 0.85 | 0.86 | 0.93 |
| Feed-Forward Neural Network | 0.88 | 0.86 | 0.84 | 0.85 | 0.92 |
| **Proposed Framework (Transformer + SHAP + LDA)** | **0.92** | **0.92** | **0.90** | **0.91** | **0.96** |

The proposed framework delivers a **3 – 8 percentage point improvement** in F1-score and a **3 – 8 point gain in ROC-AUC** over the strongest classical baseline, while preserving full interpretability.

### 7.2 Ablation Study

| Configuration | Accuracy | ROC-AUC | Interpretability |
|---------------|:--------:|:-------:|:----------------:|
| Full Framework | 0.92 | 0.96 | Complete |
| Without SHAP Reporting | 0.92 | 0.96 | Reduced |
| Without Topic Module | 0.90 | 0.94 | Complete |
| Single-Layer Transformer | 0.89 | 0.93 | Complete |

The attention mechanism is responsible for the bulk of the predictive gain, while SHAP and LDA contribute primarily to interpretability and contextual richness.

### 7.3 SHAP Feature-Importance Findings

Top global drivers (mean |SHAP|): **JobSatisfaction · MonthlyIncome · OverTime · YearsWithCurrManager · WorkLifeBalance**. The directional signs align with HR domain expectations — low satisfaction and frequent overtime increase attrition risk, while long manager tenure decreases it.

### 7.4 LDA Topic Findings (Tata Motors Corpus)

Five recurring topics were recovered: *Work-Life Balance · Compensation & Benefits · Career Growth · Management Quality · Job Security*. Topics with negative sentiment co-occurred most strongly with high-risk employees, confirming the diagnostic value of the text channel.

### 7.5 Five-Tier Risk Distribution

A representative analysis run on the Tata Motors corpus yielded the following distribution:

```
LOW            ████████████████████░░░░░ 60 %
EARLY WARNING  ████████░░░░░░░░░░░░░░░░ 15 %
MODERATE       ██████░░░░░░░░░░░░░░░░░░ 12 %
HIGH           ████░░░░░░░░░░░░░░░░░░░░  8 %
CRITICAL       ██░░░░░░░░░░░░░░░░░░░░░░  5 %
```

---

## 8. Key Contributions

1. **Unified architecture.** First framework, to the authors' knowledge, that integrates Transformer-based tabular prediction, SHAP-based local + global explanation, and LDA topic modeling of employee narratives within a single end-to-end pipeline.

2. **Column-flexible preprocessing.** A case-insensitive, multi-candidate column-mapping mechanism enables the same pipeline to ingest IBM HR Analytics, Kaggle HR, and AmbitionBox review schemas without code modification.

3. **Five-tier risk taxonomy.** Replaces opaque binary prediction with a clinically-inspired graded scheme (Low / Early Warning / Moderate / High / Critical), each annotated with leading SHAP drivers and topic cues, producing managerially actionable outputs.

4. **Empirical performance gain.** Reports an F1-score of 0.91 and ROC-AUC of 0.96 on benchmark data — superior to Logistic Regression, Random Forest, XGBoost, and feed-forward neural baselines under identical cross-validation protocols.

5. **Reproducibility artifacts.** A standalone Python implementation (`python_ml/attrition_predictor.py`), an interactive Google Colab notebook, and a deployed full-stack reference application accompany the paper, enabling third-party validation.

---

## 9. Conclusion

The study presents a unified, interpretable framework for employee attrition prediction that simultaneously addresses three persistent gaps in workforce analytics: the predictive-vs-explainable trade-off, the under-utilization of unstructured employee narratives, and the absence of managerially actionable output formats. By coupling a Transformer-based tabular encoder with SHAP feature attribution and LDA topic modeling, the framework achieves an accuracy of 0.92 and ROC-AUC of 0.96 on benchmark data — outperforming established baselines while delivering both per-employee and corpus-level interpretability.

The five-tier risk taxonomy, accompanied by SHAP drivers and topic cues, transforms statistical predictions into HR-ready decision-support output. Empirical evaluation on the IBM HR dataset and validation on a real-world 11,981-review Tata Motors corpus demonstrate that the framework generalizes across heterogeneous data sources without architectural modification. The work establishes a robust foundation for future extension to longitudinal attrition tracking, integration with enterprise HRIS platforms, and transfer to adjacent workforce-analytics problems such as engagement modeling and internal-mobility prediction.

---

## Suggested Viva Talking Points

1. **Why a Transformer for tabular data?** — Attention captures interactions natively; recent work (Gorishniy 2021, SAINT 2021) shows competitive performance with gradient boosting while integrating cleanly with SHAP.
2. **Why SHAP over LIME?** — SHAP uniquely satisfies efficiency, symmetry, dummy, and additivity axioms — necessary for auditable HR decisions.
3. **Why LDA for employee feedback?** — LDA recovers latent themes without supervision, exposing soft drivers (culture, recognition) that structured fields cannot capture.
4. **Sample size adequacy?** — IBM dataset (n = 1,470) is the established benchmark in attrition literature; AmbitionBox corpus (n ≈ 11,981) provides large-scale text validation.
5. **Comparison fairness?** — All baselines were trained under identical stratified 5-fold cross-validation with identical preprocessing pipelines.
6. **Novelty?** — The combined Transformer + SHAP + LDA + five-tier risk taxonomy is, to the best of the authors' knowledge, the first such unification in published workforce-analytics literature.
