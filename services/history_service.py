import uuid
import datetime
from sqlalchemy.future import select
from database import AsyncSessionLocal, Conversation, Message

class HistoryService:
    async def create_conversation(self, title: str = "新对话"):
        """创建一个新的主线会话"""
        async with AsyncSessionLocal() as session:
            new_id = str(uuid.uuid4())
            conv = Conversation(
                id=new_id,
                title=title,
                created_at=datetime.datetime.utcnow()
            )
            session.add(conv)
            await session.commit()
            return new_id

    async def get_conversations(self):
        """获取所有存在的会话，按时间倒序排列返回给前台呈现为菜单"""
        async with AsyncSessionLocal() as session:
            stmt = select(Conversation).order_by(Conversation.created_at.desc())
            result = await session.execute(stmt)
            convs = result.scalars().all()
            return [{"id": c.id, "title": c.title, "created_at": c.created_at} for c in convs]

    async def get_messages(self, conversation_id: str):
        """拉取指定会话当中的过往聊天记录，以便恢复聊天界面"""
        async with AsyncSessionLocal() as session:
            stmt = select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)
            result = await session.execute(stmt)
            msgs = result.scalars().all()
            return [{"id": m.id, "sender": m.sender, "content": m.content, "created_at": m.created_at} for m in msgs]

    async def add_message(self, conversation_id: str, sender: str, content: str):
        """往数据库持久化追加一条新内容"""
        async with AsyncSessionLocal() as session:
            msg = Message(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                sender=sender,
                content=content,
                created_at=datetime.datetime.utcnow()
            )
            session.add(msg)
            
            # 当用户发出第一梯队提问时，使用其首句覆盖并概括 "新对话"
            if sender == "user":
                conv = await session.get(Conversation, conversation_id)
                if conv and conv.title == "新对话":
                    # 取问题内容的前 15 字符作为会话名称 (Title)
                    conv.title = content[:15] + "..." if len(content) > 15 else content
                    
            await session.commit()

history_service = HistoryService()
