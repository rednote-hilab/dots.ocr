import asyncio
from typing import List

from loguru import logger
from pydantic import BaseModel


class TaskExecutorPool(BaseModel):
    max_queue_size: int = 4
    concurrent_task_limit: int = 4
    name: str = "TaskExecutorPool"

    _workers: List[asyncio.Task] = []
    # TODO(tatiana): use a priority queue to improve job completion time?
    _task_queue: asyncio.Queue = None

    def start(self):
        logger.info(f"Starting up {self.concurrent_task_limit} {self.name} workers...")
        self._task_queue = asyncio.Queue(self.max_queue_size)

        for i in range(self.concurrent_task_limit):
            task = asyncio.create_task(self._worker_loop(f"TaskWorker-{i}"))
            self._workers.append(task)

    def stop(self):
        for worker in self._workers:
            worker.cancel()
        asyncio.gather(*self._workers, return_exceptions=True)
        logger.info("All worker tasks have been stopped.")

    async def add_task(self, task):
        await self._task_queue.put(task)

    async def _worker_loop(self, worker_id: str):
        logger.debug(f"{worker_id} started")
        while True:
            try:
                task = await self._task_queue.get()
                await task.process()
                self._task_queue.task_done()
            except asyncio.CancelledError:
                logger.debug(f"Worker {worker_id} is shutting down.")
                break
