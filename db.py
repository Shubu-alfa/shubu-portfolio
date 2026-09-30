import hashlib
import os
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, ForeignKey, String, Text, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL:
    # Local fallback so the app runs before Neon is configured
    DATABASE_URL = "sqlite:///local_chat.db"
elif DATABASE_URL.startswith("postgres://"):
    # SQLAlchemy expects postgresql://
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    ip_hash = Column(String(64), nullable=True)
    user_agent = Column(String(512), nullable=True)

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = Column(String(16), nullable=False)  # user | bot
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    session = relationship("ChatSession", back_populates="messages")


def init_db():
    Base.metadata.create_all(bind=engine)


def hash_ip(ip: str) -> str:
    salt = os.getenv("SECRET_KEY", "shubzfolio")
    return hashlib.sha256(f"{salt}:{ip or ''}".encode("utf-8")).hexdigest()


def ensure_session(session_id: str, ip: str = None, user_agent: str = None) -> str:
    sid = (session_id or "").strip()
    try:
        uuid.UUID(sid)
    except (ValueError, TypeError):
        sid = str(uuid.uuid4())

    db = SessionLocal()
    try:
        existing = db.get(ChatSession, sid)
        if existing is None:
            db.add(
                ChatSession(
                    id=sid,
                    ip_hash=hash_ip(ip) if ip else None,
                    user_agent=(user_agent or "")[:512] or None,
                )
            )
            db.commit()
        return sid
    finally:
        db.close()


def log_exchange(session_id: str, user_text: str, bot_text: str, ip: str = None, user_agent: str = None):
    sid = ensure_session(session_id, ip=ip, user_agent=user_agent)
    db = SessionLocal()
    try:
        db.add(ChatMessage(session_id=sid, role="user", content=user_text or ""))
        db.add(ChatMessage(session_id=sid, role="bot", content=bot_text or ""))
        db.commit()
        return sid
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
