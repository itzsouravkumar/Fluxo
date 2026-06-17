from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import sessionmaker
from .models import Base


engine = None
async_session_factory = None


def init_db(database_url: str):
    global engine, async_session_factory
    engine = create_async_engine(database_url, echo=False)
    async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db():
    if async_session_factory is None:
        raise RuntimeError("Database not initialized")
    async with async_session_factory() as session:
        yield session


async def create_tables():
    if engine is None:
        raise RuntimeError("Database not initialized")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
