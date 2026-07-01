import logging
from typing import Dict, Any, Union, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from src.models.exceptions import ValidationError

logger = logging.getLogger("CrackLaw.Models.Metrics")

class MetricCalculator:
    """Calculates, aggregates, and validates evaluations across classification and regression models."""

    @staticmethod
    def calculate_classification_metrics(
        y_true: Union[np.ndarray, List[Any]],
        y_pred: Union[np.ndarray, List[Any]],
        y_probs: Optional[np.ndarray] = None,
        average: str = "macro"
    ) -> Dict[str, Any]:
        """Computes Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix."""
        try:
            true_arr = np.asarray(y_true)
            pred_arr = np.asarray(y_pred)

            if len(true_arr) != len(pred_arr):
                raise ValidationError("Size mismatch between ground truth and predicted labels.")

            # Basic metrics
            accuracy = float(accuracy_score(true_arr, pred_arr))
            
            # Precision, Recall, F1
            prec, rec, f1, _ = precision_recall_fscore_support(
                true_arr, pred_arr, average=average, zero_division=0
            )

            # Confusion Matrix
            cm = confusion_matrix(true_arr, pred_arr)

            results = {
                "accuracy": round(accuracy, 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "confusion_matrix": cm.tolist()
            }

            # Optional ROC-AUC if probability scores are provided
            if y_probs is not None:
                probs_arr = np.asarray(y_probs)
                try:
                    # Check if multi-class
                    num_classes = len(np.unique(true_arr))
                    if num_classes > 2:
                        roc_auc = roc_auc_score(true_arr, probs_arr, multi_class="ovr", average=average)
                    else:
                        # Binary classification
                        if probs_arr.ndim > 1 and probs_arr.shape[1] == 2:
                            roc_auc = roc_auc_score(true_arr, probs_arr[:, 1])
                        else:
                            roc_auc = roc_auc_score(true_arr, probs_arr)
                    results["roc_auc"] = round(float(roc_auc), 4)
                except Exception as e:
                    logger.debug("Failed to calculate ROC-AUC score: %s", str(e))
                    results["roc_auc"] = 0.0

            return results
        except Exception as e:
            if isinstance(e, ValidationError):
                raise e
            raise ValidationError(f"Error calculating classification metrics: {e}") from e

    @staticmethod
    def calculate_regression_metrics(
        y_true: Union[np.ndarray, List[float]],
        y_pred: Union[np.ndarray, List[float]]
    ) -> Dict[str, float]:
        """Computes Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2)."""
        try:
            true_arr = np.asarray(y_true, dtype=np.float32)
            pred_arr = np.asarray(y_pred, dtype=np.float32)

            if len(true_arr) != len(pred_arr):
                raise ValidationError("Size mismatch between ground truth and predicted outputs.")

            mse = float(mean_squared_error(true_arr, pred_arr))
            mae = float(mean_absolute_error(true_arr, pred_arr))
            r2 = float(r2_score(true_arr, pred_arr))

            return {
                "mse": round(mse, 4),
                "rmse": round(float(np.sqrt(mse)), 4),
                "mae": round(mae, 4),
                "r2": round(r2, 4)
            }
        except Exception as e:
            if isinstance(e, ValidationError):
                raise e
            raise ValidationError(f"Error calculating regression metrics: {e}") from e
