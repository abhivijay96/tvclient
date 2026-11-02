"""Graph contraction, which merges equivalent sibling nodes between rollouts."""

import asyncio
from collections import deque
from typing import Deque, Dict, Set, Tuple


class SandboxBudget:
    """Shares pair tokens among contraction jobs, one job per task.

    Each token lets a job keep two more live sandboxes, so there are
    max_sandboxes // 2 tokens. An odd cap leaves one slot unused. A task waits
    in arrival order for its first token, and takes extras only when no task
    is waiting. A task holding tokens never waits, so jobs can't deadlock.
    Each task makes one take_token call at a time.
    """

    def __init__(self, max_sandboxes: int):
        if max_sandboxes < 2:
            raise ValueError(f"max_sandboxes must be at least 2, got {max_sandboxes}")
        self._free = max_sandboxes // 2
        self._held: Dict[str, int] = {}
        self._waiters: Deque[Tuple[str, asyncio.Future]] = deque()
        self._taking: Set[str] = set()

    def tokens_held(self, task_id: str) -> int:
        """Reports how many tokens task_id holds."""
        return self._held.get(task_id, 0)

    async def take_token(self, task_id: str, stop: asyncio.Event) -> bool:
        """Takes one token for task_id and returns whether it got one.

        Returns False at once if stop is set. A task with no tokens waits in
        arrival order if it can't get one now, and gets False if stop is set
        first. A task with tokens gets an extra only if one is free and no task
        is waiting. A token granted as stop is set is kept. Raises RuntimeError
        if task_id already has a take_token call in progress.
        """
        if stop.is_set():
            return False
        if task_id in self._taking:
            raise RuntimeError(f"task {task_id} already has a take_token call")
        self._taking.add(task_id)
        try:
            return await self._take(task_id, stop)
        finally:
            self._taking.discard(task_id)

    async def _take(self, task_id: str, stop: asyncio.Event) -> bool:
        if self._free and not self._waiters:
            self._free -= 1
            self._held[task_id] = self.tokens_held(task_id) + 1
            return True
        if task_id in self._held:
            return False
        stopped = asyncio.ensure_future(stop.wait())
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append((task_id, waiter))
        try:
            await asyncio.wait({waiter, stopped}, return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            self._abandon(task_id, waiter)
            raise
        finally:
            stopped.cancel()
        if waiter.done():
            return True
        self._abandon(task_id, waiter)
        return False

    def release_tokens(self, task_id: str) -> None:
        """Hands each of task_id's tokens to the oldest waiter, or frees it."""
        for _ in range(self._held.pop(task_id, 0)):
            self._give_one()

    def _give_one(self) -> None:
        """Hands one token to the oldest waiter, or frees it."""
        if self._waiters:
            task_id, waiter = self._waiters.popleft()
            self._held[task_id] = 1
            waiter.set_result(None)
        else:
            self._free += 1

    def _abandon(self, task_id: str, waiter: asyncio.Future) -> None:
        """Leaves the queue, passing the token on if it was already granted."""
        if waiter.done():
            self.release_tokens(task_id)
        else:
            waiter.cancel()
            self._waiters.remove((task_id, waiter))
