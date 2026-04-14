import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import engine, Base

async def init_db():
    print("Creating SQL tables (Conversations, Messages)...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Tables successfully created.")

if __name__ == "__main__":
    asyncio.run(init_db())
