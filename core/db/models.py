from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, JSON, Text, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase
import uuid
from datetime import datetime, timezone


class Base(DeclarativeBase):
    pass


class Junction(Base):
    __tablename__ = "junctions"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    lat = Column(Float, nullable=False)
    lng = Column(Float, nullable=False)
    num_cameras = Column(Integer, default=1)
    num_lanes = Column(Integer, default=4)
    zone_type = Column(String)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class DensityEvent(Base):
    __tablename__ = "density_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    junction_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    density_score = Column(Float, nullable=False)
    pce_count = Column(Float, nullable=False)
    avg_speed_kmh = Column(Float)
    lane_data = Column(JSONB)


class Violation(Base):
    __tablename__ = "violations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    junction_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    type = Column(String, nullable=False)
    plate_number = Column(String)
    confidence = Column(Float, nullable=False)
    clip_url = Column(String)
    metadata_ = Column("metadata", JSONB)
    reported = Column(Boolean, default=False)


class SignalAction(Base):
    __tablename__ = "signal_actions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    junction_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    source = Column(String, nullable=False)
    phase = Column(String, nullable=False)
    duration_s = Column(Integer, nullable=False)
    rl_confidence = Column(Float)
    operator_id = Column(String)
