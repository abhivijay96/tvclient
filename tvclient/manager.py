"""Sandbox deletion loop and eviction entry point."""

import asyncio
import logging
from typing import List, Type

from tvclient.client import TVCacheClient
from tvclient.sandbox import Sandbox
from tvclient.protocol import ReclaimableSandbox, SandboxEvictionRecord


class SandboxManager:
    """The single deletion loop and eviction entry point for one cache server.

    drain_reclaimable_sandboxes is destructive: ownership of the returned
    sandboxes transfers to the caller. Exactly one SandboxManager must exist
    per server, or two drainers would split the victims between them.
    """

    def __init__(
        self,
        tool_call_env_class: Type[Sandbox],
        host: str = "127.0.0.1",
        port: int = 7878,
    ):
        self.client = TVCacheClient(host, port)
        self.tool_call_env_class = tool_call_env_class
        self.logger = logging.getLogger(__name__)

    async def drain_once(self) -> List[ReclaimableSandbox]:
        """Drains the zero-reference victims and deletes their sandboxes."""
        sandboxes = await self.client.drain_reclaimable_sandboxes()

        for sandbox in sandboxes:
            self.logger.debug(
                f"Deleting the environment {sandbox.sandbox_id} of task {sandbox.task_id}"
            )
            try:
                environment = self.tool_call_env_class(
                    task_name=sandbox.task_id, env_id=sandbox.sandbox_id
                )
                await environment.stop()
            except Exception as error:
                self.logger.debug(
                    f"Failed to remove environment {sandbox.sandbox_id} of task "
                    f"{sandbox.task_id} due to {error}"
                )

        return sandboxes

    async def run_drain_loop(
        self, interval_seconds: float, stop_event: asyncio.Event
    ) -> None:
        while not stop_event.is_set():
            await self.drain_once()
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                pass

    async def evict(self, limit: int = 10) -> int:
        """Marks the `limit` least useful sandboxes (ascending lpm-hit score)
        as eviction victims. Deletion happens later through the drain loop,
        once each victim's reference count reaches zero."""
        candidates = await self.client.list_sandbox_candidates(limit)
        if not candidates:
            return 0

        records = [
            SandboxEvictionRecord(
                task_id=candidate.task_id,
                sandbox_path=candidate.sandbox_path,
                sandbox_id=candidate.sandbox_id,
            )
            for candidate in candidates
        ]
        recorded = await self.client.mark_sandboxes_evicted(records)
        self.logger.debug(f"Marked {recorded} sandboxes as evicted")
        return recorded

    async def close(self) -> None:
        await self.client.close()
