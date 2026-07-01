import os
import pickle
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from src.models.exceptions import ValidationError

logger = logging.getLogger("CrackLaw.Models.ModelFactory")

# --- TF Dynamic Import & Emulator ---
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout
    HAS_TF = True
    logger.info("TensorFlow is available in the environment.")
except ImportError:
    HAS_TF = False
    logger.info("TensorFlow is not available. Using PyTorch-based Keras emulator.")


class PyTorchKerasEmulator(nn.Module):
    """Bridges PyTorch implementation with Keras-style fit/predict API when TensorFlow is absent."""

    def __init__(self, pytorch_model: nn.Module, is_regression: bool = False):
        super().__init__()
        self.model = pytorch_model
        self.is_regression = is_regression
        self.optimizer = None
        self.loss_fn = None

    def compile(self, optimizer_name: str = "adam", loss_name: str = "binary_crossentropy", lr: float = 0.001):
        """Sets up optimizer and criterion."""
        if optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

        if self.is_regression:
            self.loss_fn = nn.MSELoss()
        elif loss_name == "categorical_crossentropy" or loss_name == "sparse_categorical_crossentropy":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.BCELoss()

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 5, batch_size: int = 32, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, List[float]]:
        """Mock-trains/trains emulator network using PyTorch execution loops."""
        history = {"loss": [], "val_loss": []}
        x_tensor = torch.tensor(x, dtype=torch.float32 if self.is_regression else torch.long)
        y_tensor = torch.tensor(y, dtype=torch.float32 if self.is_regression or y.ndim > 1 else torch.long)

        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                out = self.model(batch_x)
                if not self.is_regression and isinstance(self.loss_fn, nn.BCELoss):
                    out = out.squeeze()
                
                loss = self.loss_fn(out, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            avg_loss = epoch_loss / len(x)
            history["loss"].append(avg_loss)

            if validation_data:
                self.model.eval()
                val_x, val_y = validation_data
                val_x_t = torch.tensor(val_x, dtype=torch.float32 if self.is_regression else torch.long)
                val_y_t = torch.tensor(val_y, dtype=torch.float32 if self.is_regression or val_y.ndim > 1 else torch.long)
                with torch.no_grad():
                    val_out = self.model(val_x_t)
                    if not self.is_regression and isinstance(self.loss_fn, nn.BCELoss):
                        val_out = val_out.squeeze()
                    val_loss = self.loss_fn(val_out, val_y_t).item()
                    history["val_loss"].append(val_loss)
            
        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Runs predictions through compiled PyTorch model returning standard NumPy outputs."""
        self.model.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32 if self.is_regression else torch.long)
        with torch.no_grad():
            outputs = self.model(x_tensor)
            return outputs.cpu().numpy()

    def save(self, filepath: str) -> None:
        """Saves weights state dict to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.state_dict(), filepath)

    def load_weights(self, filepath: str) -> None:
        """Loads state dict from file path."""
        self.load_state_dict(torch.load(filepath))


# --- PyTorch Models ---

class PyTorchTextCNN(nn.Module):
    """PyTorch Text Convolutional Neural Network for document categorization."""

    def __init__(self, vocab_size: int = 5000, embed_dim: int = 128, num_classes: int = 5, filter_sizes: List[int] = [3, 4, 5], num_filters: int = 100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embed_dim, seq_len]

        pooled_outputs = []
        for conv in self.convs:
            # conv output shape: [batch_size, num_filters, seq_len - kernel_size + 1]
            conved = torch.relu(conv(embedded))
            pooled = torch.max_pool1d(conved, kernel_size=conved.shape[2]).squeeze(2)
            pooled_outputs.append(pooled)

        cat = self.dropout(torch.cat(pooled_outputs, dim=1))
        logits = self.fc(cat)
        return torch.softmax(logits, dim=1)


class PyTorchNERBiLSTM(nn.Module):
    """PyTorch BiLSTM model mapping word tokens to legal tag classifications (Softmax)."""

    def __init__(self, vocab_size: int = 5000, embed_dim: int = 128, hidden_dim: int = 128, num_tags: int = 9):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim * 2]
        logits = self.fc(self.dropout(lstm_out))  # [batch_size, seq_len, num_tags]
        return torch.softmax(logits, dim=-1)


# --- Scikit-Learn Models ---

class CaseRecommendationEngine:
    """Scikit-learn cosine similarity based content Recommendation Engine."""

    def __init__(self):
        self.case_embeddings: np.ndarray = np.array([])
        self.case_ids: List[str] = []
        self.is_fitted = False

    def fit(self, case_ids: List[str], case_embeddings: np.ndarray):
        """Stores case profiles and representations to calculate distance overlaps."""
        self.case_ids = case_ids
        self.case_embeddings = np.asarray(case_embeddings, dtype=np.float32)
        self.is_fitted = True
        logger.info("Fitted RecommendationEngine with %d cases.", len(case_ids))

    def recommend(self, user_history: List[str], top_n: int = 3) -> List[str]:
        """Calculates cosine similarity metrics between candidates and history cases."""
        if not self.is_fitted:
            raise ValidationError("RecommendationEngine must be fitted before calling recommend.")
        
        # Resolve history index positions
        history_indices = [self.case_ids.index(cid) for cid in user_history if cid in self.case_ids]
        if not history_indices:
            # Fallback to top_n random/first items if history is missing or doesn't match registry
            return self.case_ids[:top_n]

        # Calculate average representation vector of user viewed history
        user_vector = np.mean(self.case_embeddings[history_indices], axis=0).reshape(1, -1)

        # Compute similarity against all items
        sims = cosine_similarity(user_vector, self.case_embeddings).flatten()

        # Sort candidate indices by score descending, skipping already viewed items
        sorted_indices = np.argsort(sims)[::-1]
        recommendations = []
        for idx in sorted_indices:
            cid = self.case_ids[idx]
            if cid not in user_history:
                recommendations.append(cid)
            if len(recommendations) >= top_n:
                break

        return recommendations


# --- Factory Orchestrator ---

class ModelFactory:
    """Central factory instantiating deep learning and machine learning model architectures."""

    @staticmethod
    def create_intent_classifier(vocab_size: int = 5000, embed_dim: int = 32, num_classes: int = 8) -> Any:
        """TensorFlow Multi-class Intent Classification network."""
        if HAS_TF:
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=100),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            return model
        else:
            # Keras Emulator wrapping a PyTorch MLP
            class PyTorchMLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.emb = nn.Embedding(vocab_size, embed_dim)
                    self.fc1 = nn.Linear(embed_dim, 64)
                    self.fc2 = nn.Linear(64, num_classes)
                    self.dropout = nn.Dropout(0.3)

                def forward(self, x):
                    embedded = self.emb(x)
                    pooled = torch.mean(embedded, dim=1)
                    x_out = torch.relu(self.fc1(pooled))
                    x_out = self.dropout(x_out)
                    return torch.softmax(self.fc2(x_out), dim=1)

            return PyTorchKerasEmulator(PyTorchMLP(), is_regression=False)

    @staticmethod
    def create_legal_risk_predictor(num_features: int = 28) -> Any:
        """TensorFlow Multilayer Perceptron model for continuous Legal Risk score regression."""
        if HAS_TF:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='relu', input_shape=(num_features,)),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            return model
        else:
            class PyTorchRegressionMLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(num_features, 32)
                    self.fc2 = nn.Linear(32, 16)
                    self.fc3 = nn.Linear(16, 1)

                def forward(self, x):
                    x_out = torch.relu(self.fc1(x))
                    x_out = torch.relu(self.fc2(x_out))
                    return torch.sigmoid(self.fc3(x_out))

            return PyTorchKerasEmulator(PyTorchRegressionMLP(), is_regression=True)

    @staticmethod
    def create_ner_extractor(vocab_size: int = 5000, num_tags: int = 9) -> PyTorchNERBiLSTM:
        """PyTorch Named Entity Extraction BiLSTM model."""
        return PyTorchNERBiLSTM(vocab_size=vocab_size, num_tags=num_tags)

    @staticmethod
    def create_doc_classifier(vocab_size: int = 5000, num_classes: int = 5) -> PyTorchTextCNN:
        """PyTorch Document Classification Convolutional network (TextCNN)."""
        return PyTorchTextCNN(vocab_size=vocab_size, num_classes=num_classes)

    @staticmethod
    def create_recommendation_engine() -> CaseRecommendationEngine:
        """Scikit-learn Content Similarity Recommendation Engine."""
        return CaseRecommendationEngine()

    @staticmethod
    def create_baseline_classifier(model_type: str = "logistic_regression") -> Union[LogisticRegression, RandomForestClassifier]:
        """Scikit-learn baseline classifier models for traditional ML benchmark tasks."""
        if model_type.lower() == "logistic_regression":
            return LogisticRegression(max_iter=1000, random_state=42)
        elif model_type.lower() == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValidationError(f"Unsupported baseline classifier model requested: '{model_type}'")
