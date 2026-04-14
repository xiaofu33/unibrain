from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, String, Text, DateTime, ForeignKey
import datetime
from config import settings

# 用于配置 PostgreSQL 的异步数据库引擎
engine = create_async_engine(
    settings.DATABASE_URL.replace("postgresql+psycopg", "postgresql+psycopg_async") 
    if "postgresql+psycopg://" in settings.DATABASE_URL else settings.DATABASE_URL,
    echo=True,
)

AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(String(36), primary_key=True)
    title = Column(String(255), default="新对话")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    messages = relationship("Message", back_populates="conversation", order_by="Message.created_at", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = 'messages'
    id = Column(String(36), primary_key=True)
    conversation_id = Column(String(36), ForeignKey('conversations.id'))
    sender = Column(String(20)) # "user" or "ai"
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
