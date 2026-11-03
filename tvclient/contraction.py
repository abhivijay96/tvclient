"""Graph contraction, which merges equivalent sibling nodes between rollouts."""

import asyncio
import json
import logging
import math
import numbers
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import (
    AsyncIterator,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
)

from tvclient.client import SandboxLease, TVCacheClient
from tvclient.protocol import Sibling, SiblingGroup, SimilarPair
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


class SandboxPool:
    """Live sandboxes for one sibling group of a job, keyed by child index.

    Holds at most one sandbox per child and two per budget token. When full,
    it takes an extra token if one is free and no task waits, and otherwise
    stops the unpinned sandbox whose child has the fewest undecided pairs.
    """

    def __init__(
        self,
        client: TVCacheClient,
        budget: SandboxBudget,
        stop: asyncio.Event,
        env_class: Type[Sandbox],
        tool_call_class: Type[ToolCall],
        task_id: str,
        parent_path: List[str],
        children: List[Sibling],
        undecided: Callable[[int], int],
    ):
        self._client = client
        self._budget = budget
        self._stop = stop
        self._env_class = env_class
        self._tool_call_class = tool_call_class
        self._task_id = task_id
        self._parent_path = parent_path
        self._children = children
        self._undecided = undecided
        self._live: Dict[int, Sandbox] = {}
        self._pinned: Set[int] = set()
        self.restores = self.replays = 0
        self.unavailable: Set[int] = set()

    @property
    def task_id(self) -> str:
        """The task whose budget tokens this pool uses."""
        return self._task_id

    def is_live(self, index: int) -> bool:
        """Returns whether child index has a live sandbox."""
        return index in self._live

    @asynccontextmanager
    async def get_pair(
        self, i: int, j: int
    ) -> AsyncIterator[Optional[Tuple[Sandbox, Sandbox]]]:
        """Yields live sandboxes for children i and j, or None if one is unavailable.

        Both stay pinned until the block exits, and one pair is held at a time.
        Server errors propagate from entering the block, so catch errors from
        comparing inside it.
        """
        if i == j:
            raise ValueError(f"a pair needs two children, got {i} twice")
        if not self._budget.tokens_held(self._task_id):
            raise RuntimeError(f"task {self._task_id} holds no budget tokens")
        self._pinned.update((i, j))
        try:
            first = await self._get(i)
            second = await self._get(j) if first is not None else None
            yield None if second is None else (first, second)
        finally:
            self._pinned.difference_update((i, j))

    async def clear(self) -> None:
        """Stops every live sandbox, then re-raises any cancellation it caught."""
        cancelled = None
        while self._live:
            _, sandbox = self._live.popitem()
            try:
                await _stop_quietly(sandbox)
            except asyncio.CancelledError as error:
                cancelled = error
        if cancelled is not None:
            raise cancelled

    async def _get(self, index: int) -> Optional[Sandbox]:
        if index in self._live:
            return self._live[index]
        if len(self._live) >= 2 * self._budget.tokens_held(self._task_id):
            if not await self._budget.take_token(self._task_id, self._stop):
                await self._evict()
        path = self._parent_path + [self._children[index].calls[0]]
        made = await materialize(
            self._client, self._env_class, self._tool_call_class, self._task_id, path
        )
        if made is None:
            self.unavailable.add(index)
            return None
        self.restores += made.restored
        self.replays += made.replayed
        self._live[index] = made.sandbox
        return made.sandbox

    async def _evict(self) -> None:
        unpinned = [index for index in self._live if index not in self._pinned]
        if not unpinned:
            raise RuntimeError("every live sandbox is pinned")
        victim = min(unpinned, key=self._undecided)
        await _stop_quietly(self._live.pop(victim))


@dataclass
class GroupResult:
    """What to send apply_contraction for one sibling group.

    A lone blue child is already settled, so it's left out of settled.
    """

    classes: List[List[str]]  # One call per member of each class of two or more
    settled: List[str]  # One call per member of each fully compared class


class EquivalenceTracker:
    """Equal classes and known inequalities among one group's children.

    Children are named by group index. A pair is decided once both children
    are in one class or their classes are known unequal, so decisions are
    inferred through equality and inequality.
    """

    def __init__(self, children: List[Sibling], pairs: List[SimilarPair]):
        self._children = children
        self._pairs = [(pair.i, pair.j) for pair in pairs]
        self._parent = list(range(len(children)))
        self._unequal: Dict[int, Set[int]] = {}  # Class root to roots known unequal
        self._neighbors: List[List[int]] = [[] for _ in children]
        for i, j in self._pairs:
            self._neighbors[i].append(j)
            self._neighbors[j].append(i)

    def is_decided(self, i: int, j: int) -> bool:
        """Returns whether children i and j are known equal or unequal."""
        a, b = self._root(i), self._root(j)
        return a == b or b in self._unequal.get(a, ())

    def undecided(self, index: int) -> int:
        """Counts the undecided pairs that touch child index."""
        return sum(not self.is_decided(index, k) for k in self._neighbors[index])

    def mark_equal(self, i: int, j: int) -> None:
        """Joins the classes of children i and j.

        Raises ValueError if they're already known unequal.
        """
        a, b = self._root(i), self._root(j)
        if a == b:
            return
        if b in self._unequal.get(a, ()):
            raise ValueError(f"children {i} and {j} are already known unequal")
        if len(self._unequal.get(a, ())) < len(self._unequal.get(b, ())):
            a, b = b, a  # Moves the smaller set of inequalities
        self._parent[b] = a
        for other in self._unequal.pop(b, set()):
            self._unequal[other].discard(b)
            self._unequal[other].add(a)
            self._unequal.setdefault(a, set()).add(other)

    def mark_unequal(self, i: int, j: int) -> None:
        """Records that the classes of children i and j differ.

        Raises ValueError if they're already known equal.
        """
        a, b = self._root(i), self._root(j)
        if a == b:
            raise ValueError(f"children {i} and {j} are already known equal")
        self._unequal.setdefault(a, set()).add(b)
        self._unequal.setdefault(b, set()).add(a)

    def result(self) -> GroupResult:
        """Lists classes to merge and the calls of fully compared classes.

        A class is fully compared when every pair touching any member is
        decided. A merged class is settled whole or not at all, and a lone
        child is settled only if it's red.
        """
        members: Dict[int, List[int]] = {}
        for index in range(len(self._children)):
            members.setdefault(self._root(index), []).append(index)
        unfinished = set()
        for i, j in self._pairs:
            if not self.is_decided(i, j):
                unfinished.update((self._root(i), self._root(j)))
        result = GroupResult(classes=[], settled=[])
        for root, indexes in members.items():
            if len(indexes) == 1 and self._children[root].settled:
                continue
            calls = [self._children[index].calls[0] for index in indexes]
            if len(indexes) > 1:
                result.classes.append(calls)
            if root not in unfinished:
                result.settled.extend(calls)
        return result

    def _root(self, index: int) -> int:
        while self._parent[index] != index:
            self._parent[index] = self._parent[self._parent[index]]
            index = self._parent[index]
        return index


async def compare_group(
    pairs: List[SimilarPair],
    tracker: EquivalenceTracker,
    pool: SandboxPool,
    stop: asyncio.Event,
) -> int:
    """Compares pairs one by one until each is decided or dropped, or stop is set.

    Prefers pairs with more live sandboxes, then higher scores. A pair whose
    equals raises stays undecided and isn't tried again in this call, and
    neither is any pair of a child that can't be built. A comparison running
    when stop is set still finishes and counts. Returns how many comparisons
    finished. Server errors propagate.
    """
    pending = pairs
    comparisons = 0
    while not stop.is_set():
        pending = [
            pair
            for pair in pending
            if not tracker.is_decided(pair.i, pair.j)
            and pair.i not in pool.unavailable
            and pair.j not in pool.unavailable
        ]
        if not pending:
            break
        pair = max(
            pending,
            key=lambda p: (pool.is_live(p.i) + pool.is_live(p.j), p.score),
        )
        if await _compare_pair(pair, tracker, pool):
            comparisons += 1
        else:
            pending.remove(pair)
        await asyncio.sleep(0)  # Lets other tasks set stop between fast steps
    return comparisons


async def _compare_pair(
    pair: SimilarPair, tracker: EquivalenceTracker, pool: SandboxPool
) -> bool:
    """Compares one pair and records the decision, returning whether it finished."""
    async with pool.get_pair(pair.i, pair.j) as live:
        if live is None:
            return False
        first, second = live
        try:
            equal = await first.equals(second)
        except Exception as error:
            logger.debug(
                f"[CONTRACT]: Can't compare children {pair.i} and {pair.j} "
                f"in {pool.task_id}: {error!r}"
            )
            return False
    if equal:
        tracker.mark_equal(pair.i, pair.j)
    else:
        tracker.mark_unequal(pair.i, pair.j)
    return True


@dataclass
class JobSummary:
    """Counts for one contraction job, filled in as it runs."""

    deepest_depth: int = 0  # The deepest depth whose groups were listed
    comparisons: int = 0
    merges: int = 0  # Classes the server merged
    restores: int = 0
    replays: int = 0  # Path calls re-executed
    unavailable: int = 0  # Children that couldn't be built


async def run_job(
    budget: SandboxBudget,
    stop: asyncio.Event,
    env_class: Type[Sandbox],
    tool_call_class: Type[ToolCall],
    task_id: str,
    threshold: float,
    host: str,
    port: int,
    timeout: float,
    summary: JobSummary,
) -> None:
    """Contracts task_id depth by depth until no deeper nodes exist or stop is set.

    Uses its own client. Applies each group before the next, including a group
    cut short by stop. A server connection failure is logged and ends the job.
    Other errors propagate.
    """
    if stop.is_set():
        return
    try:
        async with TVCacheClient(host, port, timeout) as client:
            depth = 1
            while not stop.is_set():
                listing = await client.list_sibling_groups(task_id, depth, threshold)
                summary.deepest_depth = depth
                for group in listing.groups:
                    await _contract_group(
                        client,
                        budget,
                        stop,
                        env_class,
                        tool_call_class,
                        task_id,
                        group,
                        summary,
                    )
                    if stop.is_set():
                        return
                if not listing.deeper_nodes_exist:
                    return
                depth += 1
    except (OSError, EOFError, asyncio.TimeoutError) as error:
        logger.warning(
            f"[CONTRACT]: Can't reach the server while contracting {task_id}: {error!r}"
        )


async def _contract_group(
    client: TVCacheClient,
    budget: SandboxBudget,
    stop: asyncio.Event,
    env_class: Type[Sandbox],
    tool_call_class: Type[ToolCall],
    task_id: str,
    group: SiblingGroup,
    summary: JobSummary,
) -> None:
    """Compares one group's pairs with a fresh pool, then applies the result.

    Does nothing if stop is set before the group gets a token.
    """
    if not await budget.take_token(task_id, stop):
        return
    try:
        tracker = EquivalenceTracker(group.children, group.pairs)
        pool = SandboxPool(
            client,
            budget,
            stop,
            env_class,
            tool_call_class,
            task_id,
            group.parent_path,
            group.children,
            tracker.undecided,
        )
        try:
            summary.comparisons += await compare_group(group.pairs, tracker, pool, stop)
        finally:
            summary.restores += pool.restores
            summary.replays += pool.replays
            summary.unavailable += len(pool.unavailable)
            await pool.clear()
    finally:
        budget.release_tokens(task_id)
    result = tracker.result()
    applied = await client.apply_contraction(
        task_id, group.parent_path, result.classes, result.settled
    )
    summary.merges += applied.merged_classes


class Contractor:
    """Runs one contraction job per task while its rollouts are paused.

    Call start(task_id) when a task's rollouts finish, and await stop(task_id)
    before scheduling its next rollouts. Jobs share one sandbox budget, and
    each job has its own client and stop event. timeout is each client's
    request timeout in seconds.
    """

    def __init__(
        self,
        env_class: Type[Sandbox],
        tool_call_class: Type[ToolCall],
        similarity_threshold: float,
        max_sandboxes: int,
        host: str = "127.0.0.1",
        port: int = 7878,
        timeout: float = 5,
    ):
        if not (_is_number(similarity_threshold) and -1 <= similarity_threshold <= 1):
            raise ValueError(
                "similarity_threshold must be a finite number from -1 to 1, "
                f"got {similarity_threshold!r}"
            )
        if not (_is_number(timeout) and 0 < timeout < math.inf):
            raise ValueError(
                f"timeout must be a positive finite number, got {timeout!r}"
            )
        self._env_class = env_class
        self._tool_call_class = tool_call_class
        self._threshold = float(similarity_threshold)
        self._budget = SandboxBudget(max_sandboxes)
        self._host = host
        self._port = port
        self._timeout = timeout
        self._jobs: Dict[str, Tuple[asyncio.Task, asyncio.Event]] = {}

    def start(self, task_id: str) -> None:
        """Starts contracting task_id unless its job is still running or stopping.

        Must be called from a running event loop.
        """
        if task_id in self._jobs and not self._jobs[task_id][0].done():
            return
        stop = asyncio.Event()
        job = asyncio.create_task(self._run(task_id, stop))
        self._jobs[task_id] = (job, stop)
        job.add_done_callback(lambda done: self._forget(task_id, done))

    async def stop(self, task_id: str) -> None:
        """Stops task_id's job and waits for its last apply and sandbox cleanup.

        The training loop must await this before scheduling the task's next
        rollouts. Returns at once if no job is running, and never raises for a
        crashed job. A sandbox call that hangs keeps it waiting.
        """
        if task_id not in self._jobs:
            return
        job, stop = self._jobs[task_id]
        stop.set()
        await asyncio.wait({job})

    async def close(self) -> None:
        """Stops every job, including ones started meanwhile, and waits for them.

        Callers must stop calling start first, or this may never return.
        """
        while self._jobs:
            await asyncio.gather(*(self.stop(task_id) for task_id in list(self._jobs)))

    def _forget(self, task_id: str, job: asyncio.Task) -> None:
        """Removes task_id's entry if it still belongs to job."""
        if task_id in self._jobs and self._jobs[task_id][0] is job:
            del self._jobs[task_id]

    async def _run(self, task_id: str, stop: asyncio.Event) -> None:
        summary = JobSummary()
        try:
            await run_job(
                self._budget,
                stop,
                self._env_class,
                self._tool_call_class,
                task_id,
                self._threshold,
                self._host,
                self._port,
                self._timeout,
                summary,
            )
        except Exception:
            logger.exception(f"[CONTRACT]: Job for {task_id} crashed")
        finally:
            logger.info(
                f"[CONTRACT]: Job for {task_id} ended at depth "
                f"{summary.deepest_depth} with {summary.comparisons} comparisons, "
                f"{summary.merges} merges, {summary.restores} restores, "
                f"{summary.replays} replayed calls and {summary.unavailable} "
                "unavailable children"
            )


def _is_number(value: object) -> bool:
    """Returns whether value is a real number other than a bool."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


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
    """Stops a live sandbox and waits for the stop to finish, logging any failure.

    A cancellation is re-raised once the stop finishes, so a hung stop delays it.
    """
    stopping = asyncio.ensure_future(_stop(sandbox))
    cancelled = None
    while not stopping.done():
        try:
            await asyncio.wait({stopping})
        except asyncio.CancelledError as error:
            cancelled = error
    if stopping.cancelled():
        logger.warning(f"[CONTRACT]: Stopping sandbox {sandbox.get_id()} was cancelled")
    elif stopping.exception() is not None:
        error = stopping.exception()
        logger.warning(f"[CONTRACT]: Can't stop sandbox {sandbox.get_id()}: {error!r}")
    if cancelled is not None:
        raise cancelled


async def _stop(sandbox: Sandbox) -> None:
    """Awaits sandbox.stop() inside a coroutine, so any error lands in its task."""
    await sandbox.stop()
