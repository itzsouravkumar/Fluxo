from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timezone, timedelta

from .models import Base, Junction, DensityEvent, Violation, SignalAction


engine = None
async_session_factory = None


def init_db(database_url: str):
    global engine, async_session_factory
    engine = create_async_engine(database_url, echo=False, pool_pre_ping=True)
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


async def seed_junctions():
    if async_session_factory is None:
        return
    async with async_session_factory() as session:
        existing = await session.execute(select(Junction))
        if existing.scalars().first():
            return

        junctions = [
            Junction(id="j1", name="Veerannapalya", lat=12.9980, lng=77.6890, num_cameras=2, num_lanes=4),
            Junction(id="j2", name="Gokaldas", lat=12.9950, lng=77.6830, num_cameras=1, num_lanes=4),
            Junction(id="j3", name="Silk Board", lat=12.9180, lng=77.6210, num_cameras=3, num_lanes=6),
            Junction(id="j4", name="Hebbal", lat=13.0350, lng=77.5970, num_cameras=2, num_lanes=4),
            Junction(id="j5", name="Godda Main Road", lat=24.8250, lng=87.2150, num_cameras=1, num_lanes=4),
        ]
        session.add_all(junctions)
        await session.commit()


async def log_density_event(junction_id: str, density_score: float, pce_count: float, avg_speed: float = 0.0, lane_data: dict = None):
    if async_session_factory is None:
        return
    async with async_session_factory() as session:
        event = DensityEvent(
            junction_id=junction_id,
            timestamp=datetime.now(timezone.utc),
            density_score=density_score,
            pce_count=pce_count,
            avg_speed_kmh=avg_speed,
            lane_data=lane_data or {},
        )
        session.add(event)
        await session.commit()


async def log_violation(junction_id: str, vtype: str, confidence: float, plate: str = None, clip_url: str = None, metadata: dict = None):
    if async_session_factory is None:
        return None
    async with async_session_factory() as session:
        v = Violation(
            junction_id=junction_id,
            timestamp=datetime.now(timezone.utc),
            type=vtype,
            plate_number=plate,
            confidence=confidence,
            clip_url=clip_url,
            metadata_=metadata or {},
        )
        session.add(v)
        await session.commit()
        return str(v.id)


async def log_signal_action(junction_id: str, phase: str, duration_s: int, source: str = "rl", rl_confidence: float = 0.0, operator_id: str = None):
    if async_session_factory is None:
        return
    async with async_session_factory() as session:
        action = SignalAction(
            junction_id=junction_id,
            timestamp=datetime.now(timezone.utc),
            source=source,
            phase=phase,
            duration_s=duration_s,
            rl_confidence=rl_confidence,
            operator_id=operator_id,
        )
        session.add(action)
        await session.commit()


async def get_recent_violations(junction_id: str = None, limit: int = 50):
    if async_session_factory is None:
        return []
    async with async_session_factory() as session:
        query = select(Violation).order_by(Violation.timestamp.desc()).limit(limit)
        if junction_id:
            query = query.where(Violation.junction_id == junction_id)
        result = await session.execute(query)
        return result.scalars().all()


async def get_density_history(junction_id: str, hours: int = 24):
    if async_session_factory is None:
        return []
    async with async_session_factory() as session:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        query = (
            select(DensityEvent)
            .where(DensityEvent.junction_id == junction_id)
            .where(DensityEvent.timestamp >= since)
            .order_by(DensityEvent.timestamp.asc())
        )
        result = await session.execute(query)
        return result.scalars().all()


async def get_violation_stats(junction_id: str = None, days: int = 7):
    if async_session_factory is None:
        return {}
    async with async_session_factory() as session:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        query = select(Violation.type, func.count(Violation.id)).where(Violation.timestamp >= since)
        if junction_id:
            query = query.where(Violation.junction_id == junction_id)
        query = query.group_by(Violation.type)
        result = await session.execute(query)
        return {row[0]: row[1] for row in result.all()}
