# Employee Attrition Prediction System — Code & Methods Explanation

## 📌 Overview

This system predicts employee attrition (turnover) risk using **Explainable AI (XAI)** techniques. It processes HR datasets (like the Tata Motors AmbitionBox reviews) through a 5-step pipeline and outputs actionable insights for managerial decision-making.

---

## 🔧 How to Run with Your Tata Motors Dataset

```bash
cd talent-retention-insight
python python_ml/attrition_predictor.py --data python_ml/Tata_Motors_Employee_Reviews_from_AmbitionBox-3.csv
```

The system **auto-detects** text columns (`Likes`, `Dislikes`) and rating columns (`Overall_rating`, `work_life_balance`, etc.) from the AmbitionBox CSV format.

---

## 🏗️ System Architecture (5 Modules)

### Module 1: Data Preprocessing (`DataPreprocessor` class, Line ~704)

**Purpose:** Cleans and normalizes raw HR data into numerical features for model training.

**How it works:**
- **Column Mapping:** Uses a dictionary (`COLUMN_MAPPINGS`) to match columns from different dataset formats (IBM HR, Kaggle, AmbitionBox) to standard feature names.  
  - Example: `Overall_rating` → `satisfaction`, `work_life_balance` → `overtime`, `salary_and_benefits` → `salary`
- **Case-insensitive matching:** Handles column name variations automatically.
- **Categorical encoding:** Departments are one-hot encoded (e.g., `dept_Engineering`, `dept_HR`).
- **Normalization:** Z-score standardization `(x - mean) / std` for numerical features.
- **Missing target:** If no `Attrition` column exists (like in AmbitionBox data), it generates synthetic labels based on low rating patterns.

**Output for Tata Motors data:**
- Features extracted: `satisfaction` (from Overall_rating), `overtime` (from work_life_balance), `salary` (from salary_and_benefits), `promotion` (from career_growth), `skill_dev`, `job_security`, + department one-hot columns
- Synthetic attrition labels based on rating thresholds

---

### Module 2: Transformer Encoder (`TransformerEncoder` class, Line ~70)

**Purpose:** Predicts attrition probability for each employee using a neural network architecture.

**Architecture:**
```
Input Features → Linear Embedding (d=64) → [Transformer Block × 3] → Sigmoid → Probability
                                                    ↓
                                        Multi-Head Self-Attention (2 heads)
                                                    +
                                        Feed-Forward Network (SELU activation)
                                                    +
                                        Layer Normalization + Residual Connections
```

**Key Methods:**
| Method | What it does |
|--------|-------------|
| `_scaled_dot_product_attention()` | Computes `Attention(Q,K,V) = softmax(QK^T/√d_k) × V` |
| `_multi_head_attention()` | Splits attention into 2 parallel heads for richer representation |
| `_feed_forward()` | Two-layer MLP with SELU activation for non-linearity |
| `_layer_norm()` | Normalizes activations for stable training |
| `fit()` | Trains using binary cross-entropy loss with gradient descent |

**Why Transformer over traditional models?**
- Self-attention captures feature interactions (e.g., low salary + high overtime = higher risk)
- SELU activation enables self-normalizing properties
- Attention weights provide interpretability

**Output:** Attrition probability (0.0 to 1.0) for each employee.

---

### Module 3: SHAP Explainability Engine (`SHAPExplainer` class, Line ~270)

**Purpose:** Explains *why* each prediction was made by computing feature contributions.

**Algorithm: Shapley Values (Monte Carlo approximation)**

The Shapley value formula:
```
φᵢ(x) = Σ_{S⊆N\{i}} [|S|!(|N|-|S|-1)!/|N|!] × [f(S∪{i}) - f(S)]
```

**How it works (simplified):**
1. For each employee, for each feature:
   - Randomly shuffle feature order (permutation)
   - Compare model prediction *with* vs *without* that feature
   - Average the difference over many permutations (50 samples)
2. The result = how much each feature pushes the prediction up or down

**Example Output:**
```
Top Feature Importance:
  1. satisfaction: 0.4662 ↑  (most important — low satisfaction increases risk)
  2. overtime: 0.1510 ↑      (overtime increases risk)
  3. salary: 0.1257 ↑        (low salary increases risk)
```

**Why SHAP?**
- Based on game theory (Shapley, 1953) — mathematically fair attribution
- Model-agnostic — works with any prediction model
- Additive: individual contributions sum to the total prediction

---

### Module 4: LDA Topic Modeling (`LDATopicModel` class, Line ~395)

**Purpose:** Discovers hidden themes in employee review text (Likes/Dislikes columns).

**Algorithm: Latent Dirichlet Allocation with Collapsed Gibbs Sampling**

**How it works:**
1. **Tokenization:** Clean text → lowercase → extract words (3+ chars) → remove stop words
2. **Vocabulary building:** Keep words appearing ≥2 times
3. **Gibbs Sampling (100 iterations):**
   - Each word in each document is assigned a topic (1 of 5)
   - Iteratively reassign topics based on:
     ```
     P(topic k | word w, document d) ∝ (n_{d,k} + α) × (n_{k,w} + β) / (n_k + Vβ)
     ```
   - `α = 0.1` (document-topic smoothing), `β = 0.01` (topic-word smoothing)
4. **Sentiment analysis:** Each document scored by positive/negative keyword matching
5. **Topic-Sentiment correlation:** Shows which themes have negative sentiment

**Example Output for Tata Motors reviews:**
```
Topic 1: "management", "work", "culture", "politics" → Negative sentiment
Topic 2: "salary", "benefits", "compensation" → Negative sentiment  
Topic 3: "learning", "growth", "opportunity" → Positive sentiment
Topic 4: "security", "job", "stable" → Positive sentiment
Topic 5: "balance", "hours", "overtime" → Neutral sentiment
```

---

### Module 5: Risk Classification & Recommendations (Line ~661, ~809)

**Five-Tier Risk Classification:**
| Risk Level | Probability Range | Action |
|-----------|------------------|--------|
| **Low** | 0% – 20% | Continue monitoring |
| **Early Warning** | 20% – 40% | Schedule check-in meeting |
| **Moderate** | 40% – 60% | Initiate retention discussion |
| **High** | 60% – 80% | Urgent intervention required |
| **Critical** | 80% – 100% | Immediate executive engagement |

**Recommendation Engine** generates actionable items based on:
- Risk distribution (how many employees in each tier)
- Top SHAP features (which factors drive attrition most)
- Negative sentiment topics from LDA (what employees complain about)

---

## 📊 Expected Output (5 Results)

When you run the script with the Tata Motors dataset, you get:

### Result 1: Data Preprocessing Summary
```
Original shape: (11980, 14)
Features extracted: satisfaction, overtime, salary, promotion, skill_dev, job_security, dept_*
Attrition rate: ~15% (synthetic, based on low ratings)
```

### Result 2: Transformer Training Metrics
```
Epoch 100/100 | Loss: X.XXXX | Accuracy: XX.X%
Training complete. Best loss: X.XXXX
```

### Result 3: SHAP Feature Importance
```
Top features driving attrition:
1. satisfaction (Overall_rating) — most impactful
2. overtime (work_life_balance)
3. salary (salary_and_benefits)
...with direction indicators (↑ increases risk, ↓ decreases risk)
```

### Result 4: LDA Topics from Reviews
```
5 topics extracted from Likes + Dislikes text
Each topic: keywords + sentiment (positive/negative/neutral)
Topic-sentiment correlation showing problem areas
```

### Result 5: Recommendations + Risk Distribution
```
Risk Distribution:
  LOW            : ████████████████████░░░░░ 60%
  EARLY_WARNING  : ████████░░░░░░░░░░░░░░░░ 15%
  MODERATE       : ██████░░░░░░░░░░░░░░░░░░ 12%
  HIGH           : ████░░░░░░░░░░░░░░░░░░░░  8%
  CRITICAL       : ██░░░░░░░░░░░░░░░░░░░░░░  5%

Top Recommendations:
1. [CRITICAL] Address XX High-Risk Employees
2. [HIGH] Improve Job Satisfaction Programs
3. [MEDIUM] Conduct Salary Benchmarking
```

All results are also saved to `attrition_results.json` for programmatic access.

---

## 🔬 Academic References

| Method | Paper | Year |
|--------|-------|------|
| Transformer | "Attention Is All You Need" — Vaswani et al. | 2017 |
| SHAP | "A Unified Approach to Interpreting Model Predictions" — Lundberg & Lee | 2017 |
| LDA | "Latent Dirichlet Allocation" — Blei, Ng, Jordan | 2003 |
| SELU | "Self-Normalizing Neural Networks" — Klambauer et al. | 2017 |

---

## ⚠️ Notes

- The Tata Motors AmbitionBox dataset has **no Attrition column**, so the system generates synthetic attrition labels based on low rating patterns (employees with Overall_rating ≤ 2 and work_satisfaction ≤ 2 are more likely labeled as potential attrition).
- SHAP computation takes time (~5-10 min for 11,980 rows). The system automatically samples for performance.
- All algorithms are implemented from scratch in NumPy — no external ML library dependencies required (torch/sklearn are optional).
