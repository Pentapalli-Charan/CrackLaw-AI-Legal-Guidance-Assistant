import os
from typing import Dict, Any

# Base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Framework directories
TF_DIR = os.path.join(MODELS_DIR, "tensorflow")
PYTORCH_DIR = os.path.join(MODELS_DIR, "pytorch")
SKLEARN_DIR = os.path.join(MODELS_DIR, "sklearn")

# Management directories
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
EXPORTS_DIR = os.path.join(MODELS_DIR, "exports")
EXPERIMENTS_DIR = os.path.join(MODELS_DIR, "experiments")
REGISTRY_FILE = os.path.join(MODELS_DIR, "model_registry.json")

# Ensure directories exist
for directory in [MODELS_DIR, TF_DIR, PYTORCH_DIR, SKLEARN_DIR, CHECKPOINTS_DIR, EXPORTS_DIR, EXPERIMENTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Default training configurations
DEFAULT_HYPERPARAMS: Dict[str, Any] = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 1e-4,
    "learning_rate_decay_patience": 2,
    "learning_rate_decay_factor": 0.5,
    "weight_decay": 1e-5,
    "mixed_precision": False,
    "gpu_enabled": True
}

# Model Schema specifications
MODEL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "intent_classifier": {
        "framework": "tensorflow",
        "type": "classification",
        "description": "TensorFlow multi-class classification for legal user queries.",
        "input_features": ["text"],
        "num_classes": 8
    },
    "legal_risk_predictor": {
        "framework": "tensorflow",
        "type": "regression",
        "description": "TensorFlow MLP to predict contract risk index scores (0.0 to 1.0).",
        "input_features": ["text_length", "keyword_shall_count", "keyword_indemnify_count", "embedding_feature_mean"],
        "output_dim": 1
    },
    "ner_extractor": {
        "framework": "pytorch",
        "type": "sequence_labeling",
        "description": "PyTorch BiLSTM-Softmax model for extracting Named Legal Entities.",
        "input_features": ["tokens"],
        "labels": ["O", "B-ACT", "I-ACT", "B-SECTION", "I-SECTION", "B-COURT", "I-COURT", "B-PARTY", "I-PARTY"]
    },
    "doc_classifier": {
        "framework": "pytorch",
        "type": "classification",
        "description": "PyTorch TextCNN for multi-label document category prediction.",
        "input_features": ["text"],
        "num_classes": 5
    },
    "recommendation_engine": {
        "framework": "sklearn",
        "type": "recommendation",
        "description": "Scikit-learn cosine-similarity based case recommendation model.",
        "input_features": ["user_history", "current_case_id"]
    },
    "baseline_classifier": {
        "framework": "sklearn",
        "type": "classification",
        "description": "Scikit-learn baseline Logistic Regression model for traditional ML comparison.",
        "input_features": ["tfidf_features"]
    }
}
