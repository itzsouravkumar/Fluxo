from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .routers import streams, violations, signals, junctions, commuter


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="FLUXO API",
        description="Adaptive Urban Traffic Intelligence Platform for Bengaluru",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(junctions.router)
    app.include_router(violations.router)
    app.include_router(signals.router)
    app.include_router(streams.router)
    app.include_router(commuter.router)

    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok", "service": "fluxo"}

    @app.get("/")
    async def root():
        return {
            "name": "FLUXO",
            "description": "Adaptive Urban Traffic Intelligence Platform",
            "version": "0.1.0",
            "endpoints": {
                "junctions": "/api/v1/junctions",
                "violations": "/api/v1/violations",
                "signals": "/api/v1/signals",
                "commuter": "/api/v1/commuter",
                "ws_live": "/ws/live",
                "ws_violations": "/ws/violations",
                "health": "/api/v1/health",
            },
        }

    return app


app = create_app()
