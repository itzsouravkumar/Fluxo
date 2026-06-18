from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from contextlib import asynccontextmanager
from pathlib import Path

from .routers import streams, violations, signals, junctions, commuter

DASHBOARD_DIR = Path(__file__).parent.parent.parent / "dashboard" / "dist"
COMMUTER_DIR = Path(__file__).parent.parent.parent / "commuter-app" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print()
    print("=" * 50)
    print("  FLUXO is running!")
    print("=" * 50)
    print()
    print("  Dashboard (BTP Operator):  http://localhost:8000/dashboard")
    print("  Commuter App:              http://localhost:8000/app")
    print("  API Docs:                  http://localhost:8000/api")
    print("  Health Check:              http://localhost:8000/api/v1/health")
    print()
    print("=" * 50)
    print()
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

    @app.get("/favicon.ico")
    async def favicon():
        return Response(content=b"", media_type="image/x-icon")

    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok", "service": "fluxo"}

    @app.get("/api")
    async def api_root():
        return {
            "name": "FLUXO",
            "description": "Adaptive Urban Traffic Intelligence Platform",
            "version": "0.1.0",
            "endpoints": {
                "dashboard": "/dashboard",
                "commuter": "/app",
                "junctions": "/api/v1/junctions",
                "violations": "/api/v1/violations",
                "signals": "/api/v1/signals",
                "commuter_api": "/api/v1/commuter",
                "health": "/api/v1/health",
            },
        }

    if DASHBOARD_DIR.exists():
        app.mount("/dashboard/assets", StaticFiles(directory=str(DASHBOARD_DIR / "assets")), name="dashboard-assets")

        @app.get("/dashboard")
        @app.get("/dashboard/{full_path:path}")
        async def serve_dashboard(full_path: str = ""):
            file_path = DASHBOARD_DIR / full_path
            if file_path.is_file():
                return FileResponse(str(file_path))
            return FileResponse(str(DASHBOARD_DIR / "index.html"))

    if COMMUTER_DIR.exists():
        app.mount("/app/assets", StaticFiles(directory=str(COMMUTER_DIR / "assets")), name="commuter-assets")

        @app.get("/app")
        @app.get("/app/{full_path:path}")
        async def serve_commuter(full_path: str = ""):
            file_path = COMMUTER_DIR / full_path
            if file_path.is_file():
                return FileResponse(str(file_path))
            return FileResponse(str(COMMUTER_DIR / "index.html"))

    @app.get("/")
    async def root():
        dashboard_exists = DASHBOARD_DIR.exists()
        commuter_exists = COMMUTER_DIR.exists()
        links = []
        if dashboard_exists:
            links.append('<a href="/dashboard">Dashboard</a>')
        if commuter_exists:
            links.append('<a href="/app">Commuter App</a>')
        links.append('<a href="/api">API Docs</a>')
        return FileResponse(
            content_type="text/html",
            headers={},
            status_code=200,
            filename=None,
        ) if False else {
            "name": "FLUXO",
            "description": "Adaptive Urban Traffic Intelligence Platform",
            "version": "0.1.0",
            "links": {
                "dashboard": "/dashboard" if dashboard_exists else None,
                "commuter_app": "/app" if commuter_exists else None,
                "api": "/api",
                "health": "/api/v1/health",
            },
        }

    return app


app = create_app()
