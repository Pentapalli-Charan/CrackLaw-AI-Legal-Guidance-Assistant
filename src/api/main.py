import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Exceptions
from src.services.exceptions import (
    ServiceError, NotFoundError, SecurityError,
    RateLimitError, FileValidationError, ValidationError
)

# Custom Middlewares
from src.api.middleware import (
    RateLimitMiddleware,
    StructuredLoggingMiddleware,
    SecurityHeadersMiddleware
)

# Routers
from src.api.routers import health, chat, documents, contracts, search, summary, recommendations

# Setup root logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("CrackLaw.API.Main")

def create_app() -> FastAPI:
    """FastAPI Application Factory instantiating routes, middlewares, and exception handlers."""
    app = FastAPI(
        title="CrackLaw AI API Service",
        description="Production-grade API backend orchestrating Legal AI reasoning and searches.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # 1. CORS Configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict this in true staging/production environments
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Add custom middle-wares (Security, Logging, and Rate Limits)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(StructuredLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=100)

    # 3. Register Routers
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(contracts.router)
    app.include_router(search.router)
    app.include_router(summary.router)
    app.include_router(recommendations.router)

    # 4. Global Exception Handlers mapping domain exceptions to HTTP statuses
    @app.exception_handler(NotFoundError)
    async def not_found_exception_handler(request: Request, exc: NotFoundError):
        return JSONResponse(
            status_code=404,
            content={"status": "error", "error": "NotFoundError", "message": str(exc)}
        )

    @app.exception_handler(SecurityError)
    async def security_exception_handler(request: Request, exc: SecurityError):
        return JSONResponse(
            status_code=401,
            content={"status": "error", "error": "SecurityError", "message": str(exc)}
        )

    @app.exception_handler(RateLimitError)
    async def rate_limit_exception_handler(request: Request, exc: RateLimitError):
        return JSONResponse(
            status_code=429,
            content={"status": "error", "error": "RateLimitError", "message": str(exc)}
        )

    @app.exception_handler(FileValidationError)
    async def file_validation_exception_handler(request: Request, exc: FileValidationError):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": "FileValidationError", "message": str(exc)}
        )

    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        return JSONResponse(
            status_code=422,
            content={"status": "error", "error": "ValidationError", "message": str(exc)}
        )

    @app.exception_handler(ServiceError)
    async def base_service_exception_handler(request: Request, exc: ServiceError):
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": "ServiceError", "message": str(exc)}
        )

    @app.on_event("startup")
    async def startup_event():
        logger.info("CrackLaw API Service started. API documents ready at /docs")

    return app

app = create_app()
