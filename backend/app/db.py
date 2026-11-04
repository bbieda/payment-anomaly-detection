from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """Historia ocen. TODO: migracje (Alembic), osobna tabela `models`."""
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    source: Mapped[str] = mapped_column(String(32), index=True)
    tx_id: Mapped[str] = mapped_column(String(64))
    tx_time: Mapped[datetime] = mapped_column(DateTime)
    amount: Mapped[float] = mapped_column(Float)
    merchant: Mapped[str | None] = mapped_column(String(128), nullable=True)
    label: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_name: Mapped[str] = mapped_column(String(64), index=True)
    score: Mapped[float] = mapped_column(Float)
    is_anomaly: Mapped[bool] = mapped_column(Boolean)
    latency_ms: Mapped[float] = mapped_column(Float)


def make_session_factory(url: str):
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    engine = create_engine(url, **kwargs)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
