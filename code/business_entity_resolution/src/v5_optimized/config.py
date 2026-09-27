"""v5_optimized config: Next-generation pipeline targeting 0.98+ F0.5"""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Blocking - aggressive recall
    rare_df_max: int = 50           # More aggressive rare token threshold
    max_bucket: int = 300           # Larger buckets allowed
    min_token_len: int = 3
    sig_tokens: int = 4
    prefix_len: int = 3
    
    # Vector blocking - expanded
    vector_topk: int = 100          # More TF-IDF neighbors
    per_entity_topk: int = 0        # No cap
    
    # Char n-gram TF-IDF
    char_ngram_lo: int = 2          # Bigrams too
    char_ngram_hi: int = 5
    tfidf_max_features: int = 100000
    
    # Training
    seed: int = 42
    n_estimators: int = 1200        # More trees
    learning_rate: float = 0.035    # Slower learning
    num_leaves: int = 127
    max_depth: int = 12
    min_child_samples: int = 25
    subsample: float = 0.8
    colsample_bytree: float = 0.7
    reg_lambda: float = 2.0
    
    # Ensemble
    n_ensemble: int = 3             # Number of models to ensemble
    ensemble_seeds: tuple = (42, 123, 456)
    
    # Decision rules - per-country, will be tuned
    threshold: float = 0.50
    min_top: float = 0.65
    
    # Validation
    val_fraction: float = 0.15
    
    # Paths
    train_dir: Path = Path("dataset/train")
    test_dir: Path = Path("dataset/test")
    out_dir: Path = Path("output")
    models_dir: Path = Path("models/v5_optimized")