import time
import logging
import threading
from typing import Dict, List, Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from src.services.health_service import HealthService

logger = logging.getLogger("CrackLaw.API.Middleware")

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Memory-backed sliding window rate limiter tracking request frequencies per client IP address."""

    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.ip_records: Dict[str, List[float]] = {}
        self.lock = threading.Lock()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Resolve client IP address
        client_ip = request.client.host if request.client else "unknown_ip"
        current_time = time.time()
        window_start = current_time - 60.0

        with self.lock:
            # 1. Fetch timestamps for client IP
            timestamps = self.ip_records.get(client_ip, [])
            
            # 2. Prune records outside sliding 60s window
            timestamps = [t for t in timestamps if t > window_start]
            
            # 3. Frequency limit checks
            if len(timestamps) >= self.requests_per_minute:
                logger.warning("Rate limit exceeded for client IP '%s' on route '%s'", client_ip, request.url.path)
                return JSONResponse(
                    status_code=429,
                    content={
                        "status": "error",
                        "error": "RateLimitError",
                        "message": f"Too many requests. Limit is {self.requests_per_minute} requests per minute."
                    }
                )

            # 4. Record current request timestamp
            timestamps.append(current_time)
            self.ip_records[client_ip] = timestamps

        return await call_next(request)


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Logs client request paths, method verbs, status codes, and latency in milliseconds."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        t_start = time.time()
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path
        method = request.method
        
        logger.info("Incoming Request: %s %s from %s", method, path, client_ip)

        is_error = False
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            if status_code >= 400:
                is_error = True
            return response
        except Exception as e:
            is_error = True
            logger.error("Unhandled exception occurred during request execution: %s", str(e), exc_info=True)
            raise e
        finally:
            latency_ms = (time.time() - t_start) * 1000
            
            # Log structured stats
            logger.info(
                "Completed Request: %s %s | Status: %d | Latency: %.2f ms | IP: %s",
                method, path, status_code, latency_ms, client_ip
            )
            
            # Track metrics in health service
            health_svc = HealthService()
            # Register routes excluding diagnostics/health metrics to keep telemetry clean
            if not any(diag in path for diag in ["/health", "/metrics", "/status"]):
                health_svc.log_request(path, latency_ms, is_error)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Appends OWASP recommended security headers to all outbound HTTP responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none';"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response
