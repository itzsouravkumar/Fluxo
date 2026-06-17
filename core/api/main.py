from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import redis.asyncio as aioredis


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = aioredis.from_url("redis://localhost:6379")
    yield
    await app.state.redis.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="FLUXO API",
        description="Adaptive Urban Traffic Intelligence Platform",
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
    return app


app = create_app()
