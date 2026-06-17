from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger("fluxo.middleware")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        logger.info(f"{request.method} {request.url.path}")
        response = await call_next(request)
        return response


def register_middleware(app: FastAPI):
    app.add_middleware(LoggingMiddleware)
