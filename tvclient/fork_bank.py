"""Proactive sandbox forking: warm roots and pre-forked hot sandboxes.

The bank is filled at iteration boundaries via warm_up() (called by the
training loop, ideally one batch ahead so it overlaps the current batch) and
drained during the iteration by executors; there is no polling and no
mid-iteration refill.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Awaitable, Dict, Iterable, List, Optional, Tuple, Type, Union

from tvclient.client import TVCacheClient
from tvclient.protocol import SandboxCandidate
from tvclient.sandbox import Sandbox

ROOT_ENV_ID = "root"


class AbstractForkGenerator(ABC):
    """Source of pre-forked environments consulted before a synchronous fork.

    get_forked_env may return the id directly or an awaitable of it (a bank
    living in another process answers over the network)."""

    @abstractmethod
    def get_forked_env(
        self, task_name: str, parent_env_id: str
    ) -> Union[Optional[str], Awaitable[Optional[str]]]:
        ...

    async def stop(self) -> None:
        ...


class ForkBank(AbstractForkGenerator):
    """Default fork bank: warm root sandboxes for the tasks about to run, plus
    copies of the hottest cached sandboxes (highest lpm-hit score) across all
    tasks.

    The hot list is advisory; the authoritative step is a leased find on each
    candidate's path, which re-validates that the sandbox still lives at that
    node and holds a reference for the duration of the fork so eviction cannot
    reclaim it mid-copy. find is the sole ref-taking operation, so this is the
    only correct background-fork pattern.
    """

    def __init__(
        self,
        tool_call_env_class: Type[Sandbox],
        tasks: Iterable[str] = (),
        warm_roots_per_task: int = 1,
        copies_per_node: int = 1,
        host: str = "127.0.0.1",
        port: int = 7878,
        max_concurrency: int = 8,
    ):
        self.host, self.port = host, port
        self.client = TVCacheClient(host, port)
        self.tool_call_env_class = tool_call_env_class
        self.tasks = list(tasks)
        self.warm_roots_per_task = warm_roots_per_task
        self.copies_per_node = copies_per_node
        self.max_concurrency = max_concurrency
        # (task_name, parent_env_id) -> ready-to-use copies of that parent.
        self.shelf: Dict[Tuple[str, str], List[str]] = {}
        # Counters for reporting: sandboxes started, handed out, stopped unused.
        self.stats = {"roots_started": 0, "hot_forked": 0, "handed_out": 0, "stopped_unused": 0}
        self.logger = logging.getLogger(__name__)

    def register_task(self, task_name: str) -> None:
        if task_name not in self.tasks:
            self.tasks.append(task_name)

    async def set_tasks(self, tasks: Iterable[str]) -> None:
        """Replaces the tasks that get warm roots; roots shelved for tasks that
        are no longer listed are stopped."""
        self.tasks = list(dict.fromkeys(tasks))
        for key in [k for k in self.shelf if k[1] == ROOT_ENV_ID and k[0] not in self.tasks]:
            await self._stop_all(key[0], self.shelf.pop(key))

    def get_forked_env(self, task_name: str, parent_env_id: str) -> Optional[str]:
        copies = self.shelf.get((task_name, parent_env_id))
        if copies:
            env_id = copies.pop()
            self.stats["handed_out"] += 1
            self.logger.debug(
                f"[BANK]: Handing out {env_id} for parent {parent_env_id} of task {task_name}"
            )
            return env_id
        return None

    def live_count(self) -> int:
        return sum(len(copies) for copies in self.shelf.values())

    async def warm_up(self, top_k: int = 10) -> None:
        """Refills the bank; call before each training iteration."""
        st = time.perf_counter()
        await asyncio.gather(self._warm_roots(), self._fork_hot_sandboxes(top_k))
        et = time.perf_counter()
        self.logger.debug(f"[BANK]: warm_up finished in {et - st:.2f} seconds")

    async def stop(self) -> None:
        """Stops every shelved environment and closes the cache connection."""
        for (task_name, _parent_env_id), copies in list(self.shelf.items()):
            await self._stop_all(task_name, copies)
        self.shelf.clear()
        await self.client.close()

    async def _stop_all(self, task_name: str, env_ids: List[str]) -> None:
        for env_id in env_ids:
            self.stats["stopped_unused"] += 1
            try:
                environment = self.tool_call_env_class(task_name=task_name, env_id=env_id)
                await environment.stop()
            except Exception as error:
                self.logger.debug(f"[BANK]: Failed to stop {env_id} due to {error}")

    async def _warm_roots(self) -> None:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def start_one(task_name: str) -> None:
            async with semaphore:
                environment = self.tool_call_env_class(task_name=task_name)
                try:
                    await environment.start()
                except Exception as error:
                    self.logger.warning(f"[BANK]: Warm root for {task_name} failed: {error!r}")
                    return
            # Appended after start, so a half-started root is never handed out.
            self.shelf.setdefault((task_name, ROOT_ENV_ID), []).append(environment.get_id())
            self.stats["roots_started"] += 1
            self.logger.debug(
                f"[BANK]: Warmed root environment {environment.get_id()} for task {task_name}"
            )

        starts = []
        for task_name in self.tasks:
            have = len(self.shelf.get((task_name, ROOT_ENV_ID), []))
            starts += [start_one(task_name) for _ in range(self.warm_roots_per_task - have)]
        await asyncio.gather(*starts)

    async def _fork_hot_sandboxes(self, top_k: int) -> None:
        candidates = [
            c for c in await self.client.list_hot_sandboxes(top_k) if c.score > 0
        ]  # score 0: never resumed from, not worth a live copy
        wanted = {(c.task_id, c.sandbox_id) for c in candidates}

        # Copies of sandboxes that left the hot list (or the cache) would never
        # be asked for again.
        for key in [k for k in self.shelf if k[1] != ROOT_ENV_ID and k not in wanted]:
            await self._stop_all(key[0], self.shelf.pop(key))

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def fork_one(candidate: SandboxCandidate) -> None:
            async with semaphore:
                # One connection per concurrent fork: TVCacheClient serves one
                # request at a time.
                async with TVCacheClient(self.host, self.port) as client:
                    await self._fork_candidate(client, candidate)

        forks = []
        for candidate in candidates:
            have = len(self.shelf.get((candidate.task_id, candidate.sandbox_id), []))
            forks += [fork_one(candidate) for _ in range(self.copies_per_node - have)]
        results = await asyncio.gather(*forks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                self.logger.warning(f"[BANK]: Pre-fork failed: {result!r}")

    async def _fork_candidate(self, client: TVCacheClient, candidate: SandboxCandidate) -> None:
        async with await client.find(candidate.task_id, candidate.sandbox_path) as found:
            if found.sandbox_id != candidate.sandbox_id:
                # Marked or replaced since the hot list was taken.
                self.logger.debug(
                    f"[BANK]: Skipping stale candidate {candidate.sandbox_id} "
                    f"of task {candidate.task_id}"
                )
                return

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
            await client.report_fork_cost(candidate.task_id, sample_ms)
        self.shelf.setdefault((candidate.task_id, candidate.sandbox_id), []).append(
            forked_environment.get_id()
        )
        self.stats["hot_forked"] += 1
        self.logger.debug(
            f"[BANK]: Pre-forked {forked_environment.get_id()} from hot sandbox "
            f"{candidate.sandbox_id} (score {candidate.score}) of task "
            f"{candidate.task_id} in {et - st:.2f} seconds"
        )
