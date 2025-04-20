import asyncio
from data_fetcher import DataFetcher


class DataEngine:
    def __init__(self):
        self.data_fetcher = DataFetcher()

    async def run(self):
        await self.data_fetcher.run()


if __name__ == "__main__":
    data_engine = DataEngine()
    asyncio.run(data_engine.run())

