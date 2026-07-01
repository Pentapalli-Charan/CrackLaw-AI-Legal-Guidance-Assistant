class ModelHubError(Exception):
    """Base exception for all errors within the CrackLaw AI Model Hub."""
    pass


class ModelNotFoundError(ModelHubError):
    """Raised when a requested model is not registered or cannot be found on disk."""
    pass


class TrainingError(ModelHubError):
    """Raised when model training, scheduling, or parameter adjustment fails."""
    pass


class InferenceError(ModelHubError):
    """Raised when model inference execution fails."""
    pass


class RegistryError(ModelHubError):
    """Raised when model registration or metadata serialization fails."""
    pass


class CheckpointError(ModelHubError):
    """Raised when model checkpoints cannot be saved, loaded, or resumed."""
    pass


class PreprocessingError(ModelHubError):
    """Raised when data loading, tokenization, or vectorization pipeline fails."""
    pass


class ValidationError(ModelHubError):
    """Raised when inputs, metrics, or evaluation validations fail."""
    pass
