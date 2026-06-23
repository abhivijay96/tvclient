"""Proactive sandbox forking: warm roots and pre-forked hot sandboxes.

The bank is filled at iteration boundaries via warm_up() (called by the
training loop while the weight update runs) and drained during the iteration
by executors; there is no polling and no mid-iteration refill.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type

from tvclient.client import TVCacheClient
from tvclient.sandbox import Sandbox

ROOT_ENV_ID = "root"


class AbstractForkGenerator(ABC):
    """Source of pre-forked environments consulted before a synchronous fork."""

    @abstractmethod
    def get_forked_env(self, task_name: str, parent_env_id: str) -> Optional[str]:
        ...

    async def stop(self) -> None:
        ...


class ForkBank(AbstractForkGenerator):
    """Default fork bank: warm root sandboxes plus copies of the hottest
    cached sandboxes (highest lpm-hit score).

    The hot list is advisory; the authoritative step is a leased find on each
    candidate's path, which re-validates that the sandbox still lives at that
    node and holds a reference for the duration of the fork so eviction cannot
    reclaim it mid-copy. find is the sole ref-taking operation, so this is the
    only correct background-fork pattern.
    """

    def __init__(
        self,
        tool_call_env_class: Type[Sandbox],
        tasks: List[str],
        warm_roots_per_task: int = 1,
        copies_per_node: int = 1,
        host: str = "127.0.0.1",
        port: int = 7878,
    ):
        self.client = TVCacheClient(host, port)
        self.tool_call_env_class = tool_call_env_class
        self.tasks = list(tasks)
        self.warm_roots_per_task = warm_roots_per_task
        self.copies_per_node = copies_per_node
        # (task_name, parent_env_id) -> ready-to-use copies of that parent.
        self.shelf: Dict[Tuple[str, str], List[str]] = {}
        self.logger = logging.getLogger(__name__)

    def register_task(self, task_name: str) -> None:
        if task_name not in self.tasks:
            self.tasks.append(task_name)

    def get_forked_env(self, task_name: str, parent_env_id: str) -> Optional[str]:
        copies = self.shelf.get((task_name, parent_env_id))
        if copies:
            env_id = copies.pop()
            self.logger.debug(
                f"[BANK]: Handing out {env_id} for parent {parent_env_id} of task {task_name}"
            )
            return env_id
        return None

    async def warm_up(self, top_k: int = 10) -> None:
        """Refills the bank; call before each training iteration."""
        st = time.perf_counter()
        await self._warm_roots()
        await self._fork_hot_sandboxes(top_k)
        et = time.perf_counter()
        self.logger.debug(f"[BANK]: warm_up finished in {et - st:.2f} seconds")

    async def stop(self) -> None:
        """Stops every shelved environment and closes the cache connection."""
        for (task_name, _parent_env_id), copies in self.shelf.items():
            for env_id in copies:
                try:
                    environment = self.tool_call_env_class(task_name=task_name, env_id=env_id)
                    await environment.stop()
                except Exception as error:
                    self.logger.debug(f"[BANK]: Failed to stop {env_id} due to {error}")
        self.shelf.clear()
        await self.client.close()

    async def _warm_roots(self) -> None:
        for task_name in self.tasks:
            copies = self.shelf.setdefault((task_name, ROOT_ENV_ID), [])
            while len(copies) < self.warm_roots_per_task:
                environment = self.tool_call_env_class(task_name=task_name)
                await environment.start()
                copies.append(environment.get_id())
                self.logger.debug(
                    f"[BANK]: Warmed root environment {environment.get_id()} for task {task_name}"
                )

    async def _fork_hot_sandboxes(self, top_k: int) -> None:
        candidates = await self.client.list_hot_sandboxes(top_k)

        for candidate in candidates:
            copies = self.shelf.setdefault((candidate.task_id, candidate.sandbox_id), [])
            if len(copies) >= self.copies_per_node:
                continue

            async with await self.client.find(candidate.task_id, candidate.sandbox_path) as found:
                if found.sandbox_id != candidate.sandbox_id:
                    # Marked or replaced since the hot list was taken.
                    self.logger.debug(
                        f"[BANK]: Skipping stale candidate {candidate.sandbox_id} "
                        f"of task {candidate.task_id}"
                    )
                    continue

                # Materialize a live copy by restoring the hot snapshot; the
                # executor's bank path hands these out without a further start().
                forked_environment = self.tool_call_env_class(
                    task_name=candidate.task_id, env_id=candidate.sandbox_id
                )
                st = time.perf_counter()
                await forked_environment.start()
                et = time.perf_counter()
                # Warm-up restores are often a task's first fork-cost samples,
                # seeding the adaptive snapshot threshold before any rollout.
                sample_ms = max(1, int((et - st) * 1000))
                await self.client.report_fork_cost(candidate.task_id, sample_ms)
                copies.append(forked_environment.get_id())
                self.logger.debug(
                    f"[BANK]: Pre-forked {forked_environment.get_id()} from hot sandbox "
                    f"{candidate.sandbox_id} (score {candidate.score}) of task "
                    f"{candidate.task_id} in {et - st:.2f} seconds"
                )
