"""Graph contraction, which merges equivalent sibling nodes between rollouts."""

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Set, Tuple, Type

from tvclient.client import SandboxLease, TVCacheClient
from tvclient.sandbox import Sandbox
from tvclient.tool_call import ToolCall

logger = logging.getLogger(__name__)


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


@dataclass
class Materialized:
    """A live sandbox for a node, and how it was built."""

    sandbox: Sandbox
    restored: bool  # True if it started from a snapshot, False if it started fresh
    replayed: int  # The number of path calls re-executed after the starting point


async def materialize(
    client: TVCacheClient,
    env_class: Type[Sandbox],
    tool_call_class: Type[ToolCall],
    task_id: str,
    path: List[str],
) -> Optional[Materialized]:
    """Builds a live sandbox at path, or returns None if the workload can't.

    Restores the deepest snapshot on path while leasing it, or starts a fresh
    sandbox, then replays the rest of path. The sandbox is never snapshotted,
    and the caller must stop it. Server errors propagate.
    """
    started = None
    try:
        async with await client.lease_sandbox(task_id, path) as lease:
            started = await _start(env_class, tool_call_class, task_id, path, lease)
    except BaseException:
        if started is not None:
            sandbox, _, _ = started
            await _stop_quietly(sandbox)
        raise
    if started is None:
        return None
    sandbox, calls, restored = started
    try:
        for call in calls:
            result = await sandbox.execute(call)
            if result.startswith("Failed to execute"):
                raise RuntimeError(f"replay failed with {result[:200]!r}")
    except Exception as error:
        _log_unavailable(task_id, path, error)
        await _stop_quietly(sandbox)
        return None
    except BaseException:
        await _stop_quietly(sandbox)
        raise
    return Materialized(sandbox, restored=restored, replayed=len(calls))


async def _start(
    env_class: Type[Sandbox],
    tool_call_class: Type[ToolCall],
    task_id: str,
    path: List[str],
    lease: SandboxLease,
) -> Optional[Tuple[Sandbox, List[ToolCall], bool]]:
    """Rebuilds the calls to replay, then restores the snapshot or starts fresh.

    Returns None if the workload fails, after stopping any sandbox that went live.
    """
    snapshot_id = lease.sandbox_id
    depth = len(lease.sandbox_path) if snapshot_id is not None else 0
    env: Optional[Sandbox] = None
    try:
        calls = [tool_call_class.from_dict(json.loads(call)) for call in path[depth:]]
        if snapshot_id is None:
            env = env_class(task_name=task_id)
        else:
            env = env_class(task_name=task_id, env_id=snapshot_id)
        await env.start()
        if not _is_live(env, snapshot_id):
            raise RuntimeError("start() didn't move to a live sandbox")
    except Exception as error:
        _log_unavailable(task_id, path, error)
        await _stop_if_live(env, snapshot_id)
        return None
    except BaseException:
        await _stop_if_live(env, snapshot_id)
        raise
    return env, calls, snapshot_id is not None


def _log_unavailable(task_id: str, path: List[str], error: Exception) -> None:
    logger.debug(
        f"[CONTRACT]: Can't materialize a node at depth {len(path)} "
        f"in {task_id}: {error!r}"
    )


def _is_live(env: Optional[Sandbox], snapshot_id: Optional[str]) -> bool:
    """Returns whether env moved to a live sandbox, off the snapshot handle."""
    return env is not None and env.get_id() not in (None, snapshot_id)


async def _stop_if_live(env: Optional[Sandbox], snapshot_id: Optional[str]) -> None:
    """Stops env only if it went live, so a snapshot is never deleted."""
    if _is_live(env, snapshot_id):
        await _stop_quietly(env)


async def _stop_quietly(sandbox: Sandbox) -> None:
    """Stops a live sandbox, logging instead of raising on failure."""
    try:
        await sandbox.stop()
    except Exception as error:
        logger.warning(f"[CONTRACT]: Can't stop sandbox {sandbox.get_id()}: {error!r}")
