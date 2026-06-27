import asyncio

from app.config.settings import get_settings
from app.database.engine import build_engine, verify_connectivity


async def main():
    settings = get_settings()

    engine = build_engine(settings)

    await verify_connectivity(engine)

    print("✅ Database Connected!")

    await engine.dispose()


asyncio.run(main())