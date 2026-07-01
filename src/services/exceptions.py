class ServiceError(Exception):
    """Base exception for all errors inside the CrackLaw Service Layer."""
    pass


class NotFoundError(ServiceError):
    """Raised when a requested resource (e.g. session, document, model) is missing."""
    pass


class SecurityError(ServiceError):
    """Raised when API key headers, query tokens, or permissions fail validation checks."""
    pass


class RateLimitError(ServiceError):
    """Raised when client IPs exceed sliding window request thresholds."""
    pass


class FileValidationError(ServiceError):
    """Raised when uploaded file types, sizes, or byte headers are invalid."""
    pass


class ValidationError(ServiceError):
    """Raised when input parameters fail structural or domain schema validation."""
    pass
