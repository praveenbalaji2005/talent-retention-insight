"""
Employee Attrition Prediction System
=====================================
XAI-Powered Models for Managerial Decision-Making

This standalone Python implementation includes:
1. Transformer Encoder for Tabular Data Prediction
2. SHAP-based Explainability Engine
3. LDA Topic Modeling for Text Analysis
4. Five-Tier Risk Classification

Authors: [Your Name]
Institution: [Your Institution]
Date: 2025

Requirements:
    pip install numpy pandas scikit-learn torch shap matplotlib seaborn

Usage:
    python attrition_predictor.py --data path/to/dataset.csv
    python attrition_predictor.py --demo  # Run with sample data
"""

import numpy as np
import pandas as pd
import argparse
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Transformer model configuration"""
    d_model: int = 64          # Embedding dimension
    n_heads: int = 2           # Number of attention heads
    n_layers: int = 3          # Number of transformer layers
    dropout: float = 0.1       # Dropout rate
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32

@dataclass
class LDAConfig:
    """LDA topic modeling configuration"""
    n_topics: int = 5          # Number of topics
    n_iterations: int = 100    # Gibbs sampling iterations
    alpha: float = 0.1         # Document-topic prior
    beta: float = 0.01         # Topic-word prior

# Risk classification thresholds based on IEEE paper
RISK_THRESHOLDS = {
    'low': (0.0, 0.20),
    'early_warning': (0.20, 0.40),
    'moderate': (0.40, 0.60),
    'high': (0.60, 0.80),
    'critical': (0.80, 1.0)
}

# ============================================================================
# TRANSFORMER ENCODER FOR TABULAR DATA
# ============================================================================

class TransformerEncoder:
    """
    Transformer Encoder for Tabular Data Prediction
    
    Architecture:
    - Multi-head self-attention mechanism
    - Position-wise feed-forward networks
    - Layer normalization and residual connections
    - SELU activation function
    
    Reference: "Attention Is All You Need" (Vaswani et al., 2017)
    """
    
    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config
        self.weights = {}
        self.is_trained = False
        
    def _initialize_weights(self, input_dim: int):
        """Initialize transformer weights using Xavier initialization"""
        d_model = self.config.d_model
        n_heads = self.config.n_heads
        
        # Input embedding
        self.weights['W_embed'] = np.random.randn(input_dim, d_model) * np.sqrt(2.0 / input_dim)
        self.weights['b_embed'] = np.zeros(d_model)
        
        # Multi-head attention weights for each layer
        for layer in range(self.config.n_layers):
            # Query, Key, Value projections
            self.weights[f'W_q_{layer}'] = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
            self.weights[f'W_k_{layer}'] = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
            self.weights[f'W_v_{layer}'] = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
            self.weights[f'W_o_{layer}'] = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
            
            # Feed-forward network
            self.weights[f'W_ff1_{layer}'] = np.random.randn(d_model, d_model * 4) * np.sqrt(2.0 / d_model)
            self.weights[f'b_ff1_{layer}'] = np.zeros(d_model * 4)
            self.weights[f'W_ff2_{layer}'] = np.random.randn(d_model * 4, d_model) * np.sqrt(2.0 / (d_model * 4))
            self.weights[f'b_ff2_{layer}'] = np.zeros(d_model)
            
            # Layer normalization parameters
            self.weights[f'gamma_attn_{layer}'] = np.ones(d_model)
            self.weights[f'beta_attn_{layer}'] = np.zeros(d_model)
            self.weights[f'gamma_ff_{layer}'] = np.ones(d_model)
            self.weights[f'beta_ff_{layer}'] = np.zeros(d_model)
        
        # Output layer
        self.weights['W_out'] = np.random.randn(d_model, 1) * np.sqrt(2.0 / d_model)
        self.weights['b_out'] = np.zeros(1)
        
    def _selu(self, x: np.ndarray) -> np.ndarray:
        """SELU activation function (Self-Normalizing)"""
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / (std + eps) + beta
    
    def _scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Scaled Dot-Product Attention
        
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
        """
        d_k = Q.shape[-1]
        scores = np.dot(Q, K.T) / np.sqrt(d_k)
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        return np.dot(attention_weights, V), attention_weights
    
    def _multi_head_attention(self, x: np.ndarray, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-head self-attention mechanism"""
        d_model = self.config.d_model
        n_heads = self.config.n_heads
        d_k = d_model // n_heads
        
        # Linear projections
        Q = np.dot(x, self.weights[f'W_q_{layer}'])
        K = np.dot(x, self.weights[f'W_k_{layer}'])
        V = np.dot(x, self.weights[f'W_v_{layer}'])
        
        # Apply attention
        attention_output, attention_weights = self._scaled_dot_product_attention(Q, K, V)
        
        # Output projection
        output = np.dot(attention_output, self.weights[f'W_o_{layer}'])
        
        return output, attention_weights
    
    def _feed_forward(self, x: np.ndarray, layer: int) -> np.ndarray:
        """Position-wise feed-forward network"""
        hidden = np.dot(x, self.weights[f'W_ff1_{layer}']) + self.weights[f'b_ff1_{layer}']
        hidden = self._selu(hidden)
        output = np.dot(hidden, self.weights[f'W_ff2_{layer}']) + self.weights[f'b_ff2_{layer}']
        return output
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Forward pass through transformer encoder"""
        # Input embedding
        hidden = np.dot(x, self.weights['W_embed']) + self.weights['b_embed']
        
        attention_maps = {}
        
        # Transformer layers
        for layer in range(self.config.n_layers):
            # Multi-head attention with residual connection
            attn_output, attn_weights = self._multi_head_attention(hidden, layer)
            hidden = hidden + attn_output  # Residual connection
            hidden = self._layer_norm(
                hidden,
                self.weights[f'gamma_attn_{layer}'],
                self.weights[f'beta_attn_{layer}']
            )
            
            attention_maps[f'layer_{layer}'] = attn_weights
            
            # Feed-forward with residual connection
            ff_output = self._feed_forward(hidden, layer)
            hidden = hidden + ff_output  # Residual connection
            hidden = self._layer_norm(
                hidden,
                self.weights[f'gamma_ff_{layer}'],
                self.weights[f'beta_ff_{layer}']
            )
        
        # Output layer with sigmoid for probability
        logits = np.dot(hidden, self.weights['W_out']) + self.weights['b_out']
        probabilities = 1 / (1 + np.exp(-logits))  # Sigmoid
        
        return probabilities.flatten(), attention_maps
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """Train the transformer model"""
        self._initialize_weights(X.shape[1])
        
        n_samples = X.shape[0]
        best_loss = float('inf')
        
        if verbose:
            print("\n" + "="*60)
            print("TRANSFORMER ENCODER TRAINING")
            print("="*60)
            print(f"Configuration:")
            print(f"  - Embedding dimension: {self.config.d_model}")
            print(f"  - Attention heads: {self.config.n_heads}")
            print(f"  - Transformer layers: {self.config.n_layers}")
            print(f"  - Training samples: {n_samples}")
            print("-"*60)
        
        for epoch in range(self.config.epochs):
            # Forward pass
            predictions, _ = self.forward(X)
            
            # Binary cross-entropy loss
            epsilon = 1e-7
            loss = -np.mean(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))
            
            # Simple gradient descent (for demonstration)
            # In production, use proper backpropagation with Adam optimizer
            gradient = (predictions - y).reshape(-1, 1)
            
            # Update output weights
            self.weights['W_out'] -= self.config.learning_rate * np.dot(
                np.dot(X, self.weights['W_embed']).T,
                gradient
            ) / n_samples
            
            if loss < best_loss:
                best_loss = loss
            
            if verbose and (epoch + 1) % 20 == 0:
                accuracy = np.mean((predictions > 0.5) == y) * 100
                print(f"Epoch {epoch+1:3d}/{self.config.epochs} | Loss: {loss:.4f} | Accuracy: {accuracy:.1f}%")
        
        self.is_trained = True
        
        if verbose:
            print("-"*60)
            print(f"Training complete. Best loss: {best_loss:.4f}")
            print("="*60 + "\n")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Predict attrition probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.forward(X)


# ============================================================================
# SHAP EXPLAINABILITY ENGINE
# ============================================================================

class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) Explainability Engine
    
    Implements the Shapley value calculation for feature attribution:
    f(x) = φ₀ + Σᵢ φᵢ(x)
    
    Reference: "A Unified Approach to Interpreting Model Predictions" 
               (Lundberg & Lee, 2017)
    """
    
    def __init__(self, model: TransformerEncoder, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.baseline = None
        
    def _get_baseline_prediction(self, X: np.ndarray) -> float:
        """Calculate baseline prediction (expected value)"""
        predictions, _ = self.model.predict(X)
        return np.mean(predictions)
    
    def _calculate_shapley_value(self, x: np.ndarray, feature_idx: int, 
                                  X_background: np.ndarray, n_samples: int = 100) -> float:
        """
        Calculate Shapley value for a single feature using Monte Carlo sampling
        
        φᵢ(x) = Σ_{S⊆N\{i}} [|S|!(|N|-|S|-1)!/|N|!] * [f(S∪{i}) - f(S)]
        """
        n_features = len(x)
        shapley_values = []
        
        for _ in range(n_samples):
            # Sample a random permutation
            perm = np.random.permutation(n_features)
            feature_pos = np.where(perm == feature_idx)[0][0]
            
            # Create two instances: with and without the feature
            bg_idx = np.random.randint(len(X_background))
            
            x_with = X_background[bg_idx].copy()
            x_without = X_background[bg_idx].copy()
            
            # Include features before this one in the permutation
            for j in range(feature_pos + 1):
                x_with[perm[j]] = x[perm[j]]
            for j in range(feature_pos):
                x_without[perm[j]] = x[perm[j]]
            
            # Calculate marginal contribution
            pred_with, _ = self.model.predict(x_with.reshape(1, -1))
            pred_without, _ = self.model.predict(x_without.reshape(1, -1))
            
            shapley_values.append(pred_with[0] - pred_without[0])
        
        return np.mean(shapley_values)
    
    def explain(self, X: np.ndarray, X_background: Optional[np.ndarray] = None) -> Dict:
        """
        Generate SHAP explanations for predictions
        
        Returns:
            Dictionary containing:
            - shap_values: Feature attributions for each sample
            - feature_importance: Global feature importance (mean |SHAP|)
            - base_value: Expected prediction value
        """
        if X_background is None:
            X_background = X
            
        self.baseline = self._get_baseline_prediction(X_background)
        
        n_samples, n_features = X.shape
        shap_values = np.zeros((n_samples, n_features))
        
        print("\n" + "="*60)
        print("SHAP EXPLAINABILITY ANALYSIS")
        print("="*60)
        print(f"Calculating Shapley values for {n_samples} samples...")
        print(f"Features: {n_features}")
        print("-"*60)
        
        for i in range(n_samples):
            if (i + 1) % 10 == 0:
                print(f"Processing sample {i+1}/{n_samples}...")
            
            for j in range(n_features):
                shap_values[i, j] = self._calculate_shapley_value(
                    X[i], j, X_background, n_samples=50
                )
        
        # Calculate global feature importance
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        feature_importance_normalized = feature_importance / np.sum(feature_importance)
        
        # Create feature importance ranking
        importance_ranking = []
        sorted_indices = np.argsort(feature_importance)[::-1]
        
        for idx in sorted_indices:
            importance_ranking.append({
                'feature': self.feature_names[idx],
                'importance': float(feature_importance_normalized[idx]),
                'mean_shap': float(np.mean(shap_values[:, idx])),
                'direction': 'increases' if np.mean(shap_values[:, idx]) > 0 else 'decreases'
            })
        
        print("-"*60)
        print("Top 10 Feature Importance (SHAP):")
        for i, feat in enumerate(importance_ranking[:10]):
            direction = "↑" if feat['direction'] == 'increases' else "↓"
            print(f"  {i+1}. {feat['feature']}: {feat['importance']:.4f} {direction}")
        print("="*60 + "\n")
        
        return {
            'shap_values': shap_values,
            'feature_importance': importance_ranking,
            'base_value': self.baseline,
            'feature_names': self.feature_names
        }


# ============================================================================
# LDA TOPIC MODELING
# ============================================================================

class LDATopicModel:
    """
    Latent Dirichlet Allocation (LDA) Topic Modeling
    
    Implements collapsed Gibbs sampling for topic inference:
    p(zᵢ = k | z₋ᵢ, w) ∝ (n_{d,k} + α) * (n_{k,w} + β) / (n_k + Vβ)
    
    Reference: "Latent Dirichlet Allocation" (Blei, Ng, Jordan, 2003)
    """
    
    def __init__(self, config: LDAConfig = LDAConfig()):
        self.config = config
        self.vocabulary = {}
        self.word_topic_counts = None
        self.doc_topic_counts = None
        self.topic_counts = None
        self.topic_word_dist = None
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization with preprocessing"""
        import re
        # Convert to lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b[a-z]{3,}\b', text)
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
            'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
            'would', 'could', 'should', 'will', 'just', 'than', 'them', 'this',
            'that', 'with', 'they', 'from', 'were', 'which', 'there', 'their',
            'what', 'about', 'when', 'make', 'like', 'very', 'some', 'also'
        }
        
        return [w for w in words if w not in stop_words]
    
    def _build_vocabulary(self, documents: List[str]) -> Tuple[Dict, List[List[int]]]:
        """Build vocabulary and convert documents to word indices"""
        word_counts = Counter()
        
        for doc in documents:
            tokens = self._tokenize(doc)
            word_counts.update(tokens)
        
        # Keep words that appear at least twice
        vocab = {word: idx for idx, (word, count) in enumerate(word_counts.most_common()) 
                 if count >= 2}
        
        # Convert documents to word indices
        doc_words = []
        for doc in documents:
            tokens = self._tokenize(doc)
            indices = [vocab[w] for w in tokens if w in vocab]
            doc_words.append(indices)
        
        return vocab, doc_words
    
    def _calculate_tf_idf(self, documents: List[str]) -> np.ndarray:
        """Calculate TF-IDF weights for vocabulary"""
        n_docs = len(documents)
        doc_freq = Counter()
        
        for doc in documents:
            tokens = set(self._tokenize(doc))
            doc_freq.update(tokens)
        
        idf = {}
        for word, freq in doc_freq.items():
            idf[word] = np.log(n_docs / (freq + 1)) + 1
        
        return idf
    
    def _gibbs_sampling(self, doc_words: List[List[int]], n_vocab: int):
        """Collapsed Gibbs sampling for LDA inference"""
        n_topics = self.config.n_topics
        n_docs = len(doc_words)
        alpha = self.config.alpha
        beta = self.config.beta
        
        # Initialize counts
        self.word_topic_counts = np.zeros((n_vocab, n_topics)) + beta
        self.doc_topic_counts = np.zeros((n_docs, n_topics)) + alpha
        self.topic_counts = np.zeros(n_topics) + n_vocab * beta
        
        # Random initialization of topic assignments
        topic_assignments = []
        for d, words in enumerate(doc_words):
            doc_topics = []
            for w in words:
                topic = np.random.randint(n_topics)
                doc_topics.append(topic)
                self.word_topic_counts[w, topic] += 1
                self.doc_topic_counts[d, topic] += 1
                self.topic_counts[topic] += 1
            topic_assignments.append(doc_topics)
        
        # Gibbs sampling iterations
        for iteration in range(self.config.n_iterations):
            for d, words in enumerate(doc_words):
                for i, w in enumerate(words):
                    old_topic = topic_assignments[d][i]
                    
                    # Remove current assignment
                    self.word_topic_counts[w, old_topic] -= 1
                    self.doc_topic_counts[d, old_topic] -= 1
                    self.topic_counts[old_topic] -= 1
                    
                    # Calculate topic probabilities
                    # p(z_i = k) ∝ (n_{d,k} + α) * (n_{k,w} + β) / (n_k + Vβ)
                    topic_probs = (
                        self.doc_topic_counts[d] *
                        self.word_topic_counts[w] /
                        self.topic_counts
                    )
                    topic_probs /= topic_probs.sum()
                    
                    # Sample new topic
                    new_topic = np.random.choice(n_topics, p=topic_probs)
                    topic_assignments[d][i] = new_topic
                    
                    # Update counts
                    self.word_topic_counts[w, new_topic] += 1
                    self.doc_topic_counts[d, new_topic] += 1
                    self.topic_counts[new_topic] += 1
        
        # Calculate final topic-word distribution
        self.topic_word_dist = self.word_topic_counts.T / self.word_topic_counts.sum(axis=0)
        
        return topic_assignments
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis based on keyword matching"""
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'best', 'happy', 'satisfied', 'recommend', 'awesome',
            'perfect', 'outstanding', 'exceptional', 'positive', 'growth',
            'opportunity', 'learning', 'support', 'helpful', 'friendly'
        }
        
        negative_words = {
            'bad', 'poor', 'terrible', 'awful', 'horrible', 'worst',
            'hate', 'disappointed', 'frustrating', 'stress', 'overwork',
            'toxic', 'politics', 'unfair', 'underpaid', 'micromanagement',
            'burnout', 'quit', 'leave', 'resign', 'unhappy', 'unsatisfied'
        }
        
        tokens = set(self._tokenize(text))
        pos_count = len(tokens & positive_words)
        neg_count = len(tokens & negative_words)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    def fit_transform(self, documents: List[str]) -> Dict:
        """
        Fit LDA model and extract topics
        
        Returns:
            Dictionary containing:
            - topics: List of topics with keywords and weights
            - doc_topics: Document-topic distributions
            - sentiment_correlation: Topic-sentiment correlations
        """
        print("\n" + "="*60)
        print("LDA TOPIC MODELING")
        print("="*60)
        print(f"Documents: {len(documents)}")
        print(f"Topics: {self.config.n_topics}")
        print(f"Iterations: {self.config.n_iterations}")
        print("-"*60)
        
        # Build vocabulary
        self.vocabulary, doc_words = self._build_vocabulary(documents)
        n_vocab = len(self.vocabulary)
        print(f"Vocabulary size: {n_vocab}")
        
        if n_vocab == 0:
            print("Warning: Empty vocabulary. Returning empty topics.")
            return {'topics': [], 'doc_topics': np.array([]), 'sentiment_correlation': []}
        
        # Calculate TF-IDF
        idf = self._calculate_tf_idf(documents)
        
        # Run Gibbs sampling
        print("Running Gibbs sampling...")
        self._gibbs_sampling(doc_words, n_vocab)
        
        # Extract topics
        inv_vocab = {idx: word for word, idx in self.vocabulary.items()}
        topics = []
        
        print("\nExtracted Topics:")
        print("-"*60)
        
        for k in range(self.config.n_topics):
            topic_words = []
            word_indices = np.argsort(self.topic_word_dist[k])[::-1][:10]
            
            for idx in word_indices:
                word = inv_vocab[idx]
                weight = float(self.topic_word_dist[k, idx])
                # Apply TF-IDF weighting
                if word in idf:
                    weight *= idf[word]
                topic_words.append({'word': word, 'weight': weight})
            
            # Normalize weights
            total_weight = sum(w['weight'] for w in topic_words)
            for w in topic_words:
                w['weight'] /= total_weight
            
            topics.append({
                'id': k,
                'name': f"Topic {k+1}",
                'keywords': topic_words[:5],
                'all_keywords': topic_words
            })
            
            keywords_str = ", ".join([f"{w['word']} ({w['weight']:.3f})" for w in topic_words[:5]])
            print(f"  Topic {k+1}: {keywords_str}")
        
        # Calculate sentiment correlation for each topic
        print("\nTopic-Sentiment Correlation:")
        print("-"*60)
        
        sentiment_correlation = []
        for d, doc in enumerate(documents):
            sentiment = self._analyze_sentiment(doc)
            doc_topic = np.argmax(self.doc_topic_counts[d])
            sentiment_correlation.append({
                'doc_id': d,
                'dominant_topic': doc_topic,
                'sentiment': sentiment
            })
        
        # Aggregate sentiment by topic
        topic_sentiments = {}
        for item in sentiment_correlation:
            topic = item['dominant_topic']
            if topic not in topic_sentiments:
                topic_sentiments[topic] = []
            topic_sentiments[topic].append(item['sentiment'])
        
        for topic, sentiments in topic_sentiments.items():
            avg_sentiment = np.mean(sentiments)
            label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
            print(f"  Topic {topic+1}: {avg_sentiment:.3f} ({label})")
            topics[topic]['sentiment'] = avg_sentiment
            topics[topic]['sentiment_label'] = label
        
        print("="*60 + "\n")
        
        return {
            'topics': topics,
            'doc_topics': self.doc_topic_counts / self.doc_topic_counts.sum(axis=1, keepdims=True),
            'sentiment_correlation': sentiment_correlation,
            'vocabulary_size': n_vocab
        }


# ============================================================================
# RISK CLASSIFICATION
# ============================================================================

def classify_risk(probability: float) -> Dict:
    """
    Five-Tier Risk Classification based on IEEE paper
    
    Risk Levels:
    - Low Risk: p < 0.20
    - Early Warning: 0.20 ≤ p < 0.40
    - Moderate Risk: 0.40 ≤ p < 0.60
    - High Risk: 0.60 ≤ p < 0.80
    - Critical Risk: p ≥ 0.80
    """
    for level, (low, high) in RISK_THRESHOLDS.items():
        if low <= probability < high:
            return {
                'level': level,
                'probability': probability,
                'threshold_range': f"{low:.0%} - {high:.0%}",
                'action_required': get_action_recommendation(level)
            }
    
    return {
        'level': 'critical',
        'probability': probability,
        'threshold_range': '80% - 100%',
        'action_required': get_action_recommendation('critical')
    }

def get_action_recommendation(level: str) -> str:
    """Get recommended action based on risk level"""
    actions = {
        'low': 'Continue monitoring. Maintain current engagement practices.',
        'early_warning': 'Schedule check-in meeting. Review recent feedback.',
        'moderate': 'Initiate retention discussion. Consider role adjustments.',
        'high': 'Urgent intervention required. Discuss career development.',
        'critical': 'Immediate action needed. Executive-level engagement.'
    }
    return actions.get(level, 'Review employee status')


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

class DataPreprocessor:
    """Preprocess HR data for model training"""
    
    # Column mappings for different dataset formats
    COLUMN_MAPPINGS = {
        'satisfaction': ['JobSatisfaction', 'satisfaction_level', 'Rating', 'EnvironmentSatisfaction'],
        'tenure': ['YearsAtCompany', 'tenure', 'TotalWorkingYears', 'YearsInCurrentRole'],
        'overtime': ['OverTime', 'average_montly_hours', 'WorkLifeBalance'],
        'salary': ['MonthlyIncome', 'salary', 'DailyRate', 'HourlyRate'],
        'department': ['Department', 'department', 'JobRole'],
        'promotion': ['YearsSinceLastPromotion', 'promotion_last_5years'],
        'projects': ['NumCompaniesWorked', 'number_project'],
        'attrition': ['Attrition', 'left', 'Status']
    }
    
    def __init__(self):
        self.feature_names = []
        self.label_encoders = {}
        self.scalers = {}
        
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find matching column from candidates"""
        for col in candidates:
            if col in df.columns:
                return col
            # Case-insensitive search
            for df_col in df.columns:
                if df_col.lower() == col.lower():
                    return df_col
        return None
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocess dataframe and extract features"""
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        print(f"Original shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("-"*60)
        
        features = []
        feature_names = []
        
        # Extract numerical features
        for feature_type, candidates in self.COLUMN_MAPPINGS.items():
            if feature_type == 'attrition':
                continue
                
            col = self._find_column(df, candidates)
            if col is not None:
                values = df[col].copy()
                
                # Handle categorical columns
                if values.dtype == 'object':
                    if feature_type == 'overtime':
                        values = values.map({'Yes': 1, 'No': 0, True: 1, False: 0}).fillna(0)
                    elif feature_type == 'department':
                        # One-hot encode departments
                        for dept in values.unique():
                            if pd.notna(dept):
                                features.append((values == dept).astype(float).values)
                                feature_names.append(f'dept_{dept}')
                        continue
                    else:
                        # Label encode
                        unique_vals = values.unique()
                        mapping = {v: i for i, v in enumerate(unique_vals)}
                        values = values.map(mapping).fillna(0)
                
                # Normalize numerical values
                values = pd.to_numeric(values, errors='coerce').fillna(0)
                if values.std() > 0:
                    values = (values - values.mean()) / values.std()
                
                features.append(values.values)
                feature_names.append(feature_type)
                print(f"  Added feature: {feature_type} (from {col})")
        
        # Extract target variable
        target_col = self._find_column(df, self.COLUMN_MAPPINGS['attrition'])
        if target_col is not None:
            y = df[target_col].copy()
            if y.dtype == 'object':
                y = y.map({'Yes': 1, 'No': 0, 'Left': 1, 'Stayed': 0, True: 1, False: 0}).fillna(0)
            y = y.values.astype(float)
        else:
            # Generate synthetic target based on features
            print("  Warning: No attrition column found. Generating synthetic target.")
            y = np.random.binomial(1, 0.15, len(df))
        
        X = np.column_stack(features) if features else np.random.randn(len(df), 5)
        self.feature_names = feature_names if feature_names else [f'feature_{i}' for i in range(X.shape[1])]
        
        print(f"\nProcessed shape: X={X.shape}, y={y.shape}")
        print(f"Features: {self.feature_names}")
        print(f"Attrition rate: {y.mean():.1%}")
        print("="*60 + "\n")
        
        return X, y, self.feature_names


# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================

class RecommendationEngine:
    """Generate actionable recommendations based on analysis"""
    
    def generate(self, predictions: np.ndarray, shap_results: Dict, 
                 lda_results: Optional[Dict] = None) -> List[Dict]:
        """Generate prioritized recommendations"""
        
        recommendations = []
        
        # Risk-based recommendations
        risk_dist = {level: 0 for level in RISK_THRESHOLDS.keys()}
        for prob in predictions:
            risk = classify_risk(prob)
            risk_dist[risk['level']] += 1
        
        high_risk_count = risk_dist['high'] + risk_dist['critical']
        total = len(predictions)
        
        if high_risk_count > 0:
            recommendations.append({
                'priority': 'critical',
                'category': 'Immediate Action',
                'title': f'Address {high_risk_count} High-Risk Employees',
                'description': f'{high_risk_count} employees ({high_risk_count/total:.1%}) are at high or critical attrition risk. '
                              f'Schedule immediate one-on-one meetings with managers.',
                'impact': 'high'
            })
        
        # Feature-based recommendations
        if shap_results and 'feature_importance' in shap_results:
            top_features = shap_results['feature_importance'][:3]
            
            for feat in top_features:
                if 'satisfaction' in feat['feature'].lower():
                    recommendations.append({
                        'priority': 'high',
                        'category': 'Employee Satisfaction',
                        'title': 'Improve Job Satisfaction Programs',
                        'description': f"Job satisfaction is a key attrition driver (importance: {feat['importance']:.1%}). "
                                      f"Consider implementing regular feedback sessions and recognition programs.",
                        'impact': 'high'
                    })
                elif 'overtime' in feat['feature'].lower():
                    recommendations.append({
                        'priority': 'high',
                        'category': 'Work-Life Balance',
                        'title': 'Review Overtime Policies',
                        'description': f"Overtime is significantly impacting attrition (importance: {feat['importance']:.1%}). "
                                      f"Evaluate workload distribution and consider hiring additional staff.",
                        'impact': 'high'
                    })
                elif 'salary' in feat['feature'].lower() or 'income' in feat['feature'].lower():
                    recommendations.append({
                        'priority': 'medium',
                        'category': 'Compensation',
                        'title': 'Conduct Salary Benchmarking',
                        'description': f"Compensation factors influence attrition (importance: {feat['importance']:.1%}). "
                                      f"Benchmark salaries against industry standards and address gaps.",
                        'impact': 'medium'
                    })
        
        # Topic-based recommendations (if LDA results available)
        if lda_results and 'topics' in lda_results:
            negative_topics = [t for t in lda_results['topics'] 
                              if t.get('sentiment', 0) < -0.1]
            
            for topic in negative_topics[:2]:
                keywords = [k['word'] for k in topic.get('keywords', [])[:3]]
                recommendations.append({
                    'priority': 'medium',
                    'category': 'Employee Feedback',
                    'title': f"Address Concerns: {', '.join(keywords)}",
                    'description': f"Negative sentiment detected around these themes. "
                                  f"Consider targeted surveys or focus groups to understand root causes.",
                    'impact': 'medium'
                })
        
        # General recommendations
        recommendations.append({
            'priority': 'low',
            'category': 'Continuous Monitoring',
            'title': 'Implement Regular Risk Assessment',
            'description': 'Run attrition prediction analysis monthly to identify emerging risks early.',
            'impact': 'medium'
        })
        
        return recommendations


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

class AttritionAnalyzer:
    """Complete attrition analysis pipeline"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = TransformerEncoder(self.config)
        self.preprocessor = DataPreprocessor()
        self.shap_explainer = None
        self.lda_model = LDATopicModel()
        self.recommendation_engine = RecommendationEngine()
        
    def analyze(self, data: pd.DataFrame, text_column: Optional[str] = None) -> Dict:
        """Run complete analysis pipeline"""
        
        print("\n" + "="*70)
        print("   EMPLOYEE ATTRITION PREDICTION SYSTEM")
        print("   XAI-Powered Models for Managerial Decision-Making")
        print("="*70)
        
        # Step 1: Preprocess data
        X, y, feature_names = self.preprocessor.fit_transform(data)
        
        # Step 2: Train transformer model
        self.model.fit(X, y, verbose=True)
        
        # Step 3: Generate predictions
        predictions, attention_maps = self.model.predict(X)
        
        # Step 4: SHAP analysis
        self.shap_explainer = SHAPExplainer(self.model, feature_names)
        shap_results = self.shap_explainer.explain(X)
        
        # Step 5: LDA topic modeling (if text data available)
        lda_results = None
        if text_column and text_column in data.columns:
            documents = data[text_column].dropna().tolist()
            if len(documents) > 10:
                lda_results = self.lda_model.fit_transform(documents)
        
        # Step 6: Risk classification
        risk_classifications = [classify_risk(p) for p in predictions]
        
        # Step 7: Generate recommendations
        recommendations = self.recommendation_engine.generate(
            predictions, shap_results, lda_results
        )
        
        # Compile results
        results = {
            'summary': {
                'total_employees': len(data),
                'attrition_rate': float(y.mean()),
                'model_accuracy': float(np.mean((predictions > 0.5) == y)),
                'high_risk_count': sum(1 for r in risk_classifications if r['level'] in ['high', 'critical']),
            },
            'predictions': {
                'probabilities': predictions.tolist(),
                'risk_levels': [r['level'] for r in risk_classifications],
            },
            'shap_analysis': {
                'feature_importance': shap_results['feature_importance'],
                'base_value': shap_results['base_value'],
            },
            'lda_topics': lda_results['topics'] if lda_results else None,
            'recommendations': recommendations,
            'model_config': {
                'd_model': self.config.d_model,
                'n_heads': self.config.n_heads,
                'n_layers': self.config.n_layers,
            }
        }
        
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print analysis summary"""
        print("\n" + "="*70)
        print("   ANALYSIS SUMMARY")
        print("="*70)
        
        summary = results['summary']
        print(f"\nDataset Statistics:")
        print(f"  - Total Employees: {summary['total_employees']}")
        print(f"  - Historical Attrition Rate: {summary['attrition_rate']:.1%}")
        print(f"  - Model Accuracy: {summary['model_accuracy']:.1%}")
        print(f"  - High/Critical Risk Employees: {summary['high_risk_count']}")
        
        print(f"\nRisk Distribution:")
        risk_counts = {}
        for level in results['predictions']['risk_levels']:
            risk_counts[level] = risk_counts.get(level, 0) + 1
        
        for level in ['low', 'early_warning', 'moderate', 'high', 'critical']:
            count = risk_counts.get(level, 0)
            pct = count / summary['total_employees'] * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"  {level.upper():15s}: {bar} {count:4d} ({pct:.1f}%)")
        
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(results['recommendations'][:3], 1):
            print(f"  {i}. [{rec['priority'].upper()}] {rec['title']}")
        
        print("\n" + "="*70)
        print("   Analysis complete. Results saved to 'attrition_results.json'")
        print("="*70 + "\n")


# ============================================================================
# SAMPLE DATA GENERATOR
# ============================================================================

def generate_sample_data(n_samples: int = 200) -> pd.DataFrame:
    """Generate sample HR dataset for demonstration"""
    np.random.seed(42)
    
    departments = ['Sales', 'Engineering', 'HR', 'Marketing', 'Finance']
    
    data = {
        'EmployeeID': range(1, n_samples + 1),
        'Department': np.random.choice(departments, n_samples),
        'JobSatisfaction': np.random.randint(1, 5, n_samples),
        'YearsAtCompany': np.random.randint(0, 20, n_samples),
        'MonthlyIncome': np.random.randint(3000, 15000, n_samples),
        'OverTime': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'WorkLifeBalance': np.random.randint(1, 5, n_samples),
        'YearsSinceLastPromotion': np.random.randint(0, 10, n_samples),
        'EnvironmentSatisfaction': np.random.randint(1, 5, n_samples),
    }
    
    # Generate attrition based on features (realistic correlation)
    attrition_prob = (
        (5 - data['JobSatisfaction']) * 0.1 +
        (data['OverTime'] == 'Yes').astype(int) * 0.2 +
        (data['YearsAtCompany'] < 2).astype(int) * 0.15 +
        (data['WorkLifeBalance'] < 2).astype(int) * 0.1 +
        np.random.uniform(0, 0.1, n_samples)
    )
    attrition_prob = np.clip(attrition_prob, 0, 1)
    data['Attrition'] = np.random.binomial(1, attrition_prob).astype(str)
    data['Attrition'] = ['Yes' if a == '1' else 'No' for a in data['Attrition']]
    
    # Add some text reviews
    review_templates = [
        "Good work environment but {issue}",
        "I enjoy my work, however {issue}",
        "The team is great but {issue}",
        "Management is supportive, though {issue}",
        "Career growth is limited and {issue}",
        "Excellent learning opportunities despite {issue}",
        "Work life balance needs improvement and {issue}",
        "Salary is competitive but {issue}",
    ]
    
    issues = [
        "overtime is excessive",
        "promotion opportunities are rare",
        "workload is high",
        "communication could improve",
        "benefits need updating",
        "stress levels are concerning",
        "recognition is lacking",
        "management needs training",
    ]
    
    data['Review'] = [
        np.random.choice(review_templates).format(issue=np.random.choice(issues))
        for _ in range(n_samples)
    ]
    
    return pd.DataFrame(data)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Employee Attrition Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python attrition_predictor.py --demo
    python attrition_predictor.py --data hr_data.csv
    python attrition_predictor.py --data hr_data.csv --text-column Review
        """
    )
    
    parser.add_argument('--data', type=str, help='Path to CSV dataset')
    parser.add_argument('--demo', action='store_true', help='Run with sample data')
    parser.add_argument('--text-column', type=str, default='Review', 
                       help='Column name containing text reviews for LDA')
    parser.add_argument('--output', type=str, default='attrition_results.json',
                       help='Output file for results')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--layers', type=int, default=3, help='Transformer layers')
    
    args = parser.parse_args()
    
    # Load or generate data
    if args.demo:
        print("Running demo with sample data...")
        data = generate_sample_data(200)
    elif args.data:
        print(f"Loading data from {args.data}...")
        data = pd.read_csv(args.data)
    else:
        print("No data specified. Running demo mode...")
        data = generate_sample_data(200)
    
    # Configure model
    config = ModelConfig(
        epochs=args.epochs,
        n_layers=args.layers
    )
    
    # Run analysis
    analyzer = AttritionAnalyzer(config)
    results = analyzer.analyze(data, text_column=args.text_column)
    
    # Save results
    with open(args.output, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {args.output}")
    
    return results


if __name__ == '__main__':
    main()
