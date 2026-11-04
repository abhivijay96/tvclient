"""Per-rollout tool call executor backed by the TVCache."""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Type

from tvclient.client import FindResult, TVCacheClient
from tvclient.sandbox import Sandbox
from tvclient.fork_bank import ROOT_ENV_ID, AbstractForkGenerator
from tvclient.protocol import validate_embedding
from tvclient.snapshot_policy import (
    INITIAL_FORK_COST_SECONDS,
    SNAPSHOT_COST_MULTIPLIER,
    SnapshotPolicy,
)
from tvclient.tool_call import TestToolCall, ToolCall

MAX_RUNTIME_MS = 2**32

_embedding_pool: Optional[ThreadPoolExecutor] = None
_embedding_pool_pid: Optional[int] = None


def _get_embedding_pool() -> ThreadPoolExecutor:
    """Returns this process's embedding thread."""
    global _embedding_pool, _embedding_pool_pid
    if _embedding_pool_pid != os.getpid():
        _embedding_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tvclient-embed")
        _embedding_pool_pid = os.getpid()
    return _embedding_pool


class TvcacheToolExecutor:
    """Executes tool calls for one rollout, reusing cached results and sandboxes.

    One executor per rollout, owning one cache connection and at most one live
    sandbox. The live sandbox is never discarded on a cache hit: it simply
    falls behind the rollout's history, and on a later miss it is either
    fast-forwarded or replaced by a fork of a deeper cached sandbox.
    """

    def __init__(
        self,
        tool_call_env_class: Type[Sandbox],
        tool_call_class: Type[ToolCall],
        task_id: str,
        host: str = "127.0.0.1",
        port: int = 7878,
        adaptive: bool = True,
        fixed_threshold_seconds: float = 0.0,
        snapshot_cost_multiplier: float = SNAPSHOT_COST_MULTIPLIER,
        initial_fork_cost_seconds: float = INITIAL_FORK_COST_SECONDS,
        ttl: Optional[int] = None,
        embed_tool_calls: bool = False,
    ):
        self.client = TVCacheClient(host, port)
        self.tool_call_env_class = tool_call_env_class
        self.tool_call_class = tool_call_class
        self.task_name = task_id
        self.ttl = ttl
        self.embed_tool_calls = embed_tool_calls
        self._embedding_failed = False
        # Owns the un-checkpointed replay cost and the adaptive snapshot
        # threshold (backed by the server-side fork-cost EMA).
        self.snapshot_policy = SnapshotPolicy(
            self.client,
            task_id,
            adaptive,
            fixed_threshold_seconds,
            snapshot_cost_multiplier,
            initial_fork_cost_seconds,
        )
        self.rollout_id = str(time.time())
        self.rollout_sandbox: Optional[Sandbox] = None
        # Serialized stateful calls applied to the live sandbox; the invariant
        # is that this is a prefix of the current rollout's stateful chain.
        self.sandbox_history: List[str] = []
        self.total_calls = 0
        self.total_executions = 0
        self.logger = logging.getLogger(__name__)
        self.fork_generator: Optional[AbstractForkGenerator] = None
        self._pending_stops: List[asyncio.Task] = []

    async def __aenter__(self) -> "TvcacheToolExecutor":
        await self.client.connect()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Stops the live sandbox, waits for scheduled stops, closes the client."""
        await self._stop_rollout_environment()
        if self._pending_stops:
            await asyncio.gather(*self._pending_stops, return_exceptions=True)
            self._pending_stops = []
        await self.client.close()

    # ISSUE: if the caller doesn't call set_rollout_id, logging fails even to console, can we create a logger that just ignores the strings,
    def set_rollout_id(self, rollout_id: str) -> None:
        self.rollout_id = rollout_id
        self.logger = logging.getLogger(self.rollout_id)
        self.logger.setLevel(logging.DEBUG)

        os.makedirs("./rollouts", exist_ok=True)
        file_handler = logging.FileHandler(f"./rollouts/{rollout_id}.log")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def set_fork_bank(self, fork_bank: AbstractForkGenerator) -> None:
        self.fork_generator = fork_bank

    async def lookup_or_execute(self, tool_commands: List[ToolCall]) -> str:
        """Executes the last tool call unless the cache already has its value.

        The cache key is the stateful chain (calls whose will_mutate_state() is
        True) plus the last call; stateless intermediates never extend the key
        and are never replayed.
        """
        self.total_calls += 1
        history, current = tool_commands[:-1], tool_commands[-1]
        stateful_calls = [c for c in history if c.will_mutate_state()]
        path = [c.serialize() for c in stateful_calls] + [current.serialize()]

        cst = time.perf_counter()
        found = await self.client.find(self.task_name, path)
        cet = time.perf_counter()

        async with found:
            if found.value is not None:
                self.logger.debug(
                    f"CACHE HIT for task id {self.task_name} at depth {len(path)} "
                    f"in {cet - cst} seconds for rollout {self.rollout_id}"
                )
                return found.value

            self.logger.debug(
                f"CACHE MISS for task id {self.task_name} at depth {len(path)} "
                f"for rollout {self.rollout_id}"
            )
            # The fork (if any) happens while the lease is held, so the cached
            # sandbox cannot be reclaimed mid-fork.
            environment, resume_depth = await self._acquire_environment(found, path)

        suffix_calls = stateful_calls[resume_depth:] + [current]
        return await self._execute_and_insert(environment, path, suffix_calls, resume_depth)

    async def test(self, tool_call_history: List[ToolCall]) -> str:
        """Returns the (possibly cached) test result for the rollout's history.

        Terminal call: the rollout environment is stopped afterwards.
        """
        test_tool_call = TestToolCall("")
        result = await self.lookup_or_execute(tool_call_history + [test_tool_call])
        await self._stop_rollout_environment()
        return result

    async def _acquire_environment(
        self, found: FindResult, tool_call_list: List[str]
    ) -> Tuple[Sandbox, int]:
        """Picks the deepest valid resume point: forked LPM sandbox, the live
        sandbox (fast-forwarded), or a fresh environment."""
        live_usable = (
            self.rollout_sandbox is not None
            and tool_call_list[: len(self.sandbox_history)] == self.sandbox_history
        )
        live_depth = len(self.sandbox_history) if live_usable else 0

        # A sandbox cached at the full path already includes the current call's
        # mutation, so it can never be used to (re-)execute the current call.
        lpm_depth = 0
        if found.sandbox_id is not None:
            lpm_depth = len(found.sandbox_path)

        if found.sandbox_id is not None and lpm_depth > live_depth:
            lpm_sandbox = await self._fork_environment(found.sandbox_id)
            self._replace_rollout_environment(lpm_sandbox, list(tool_call_list[:lpm_depth]))
            self.logger.debug(
                f"[ENV]: Resuming from forked sandbox at depth {lpm_depth} "
                f"(live sandbox depth was {live_depth}) for rollout {self.rollout_id}"
            )
            return lpm_sandbox, lpm_depth

        if live_usable:
            if live_depth < len(tool_call_list) - 1:
                self.logger.debug(
                    f"[ENV]: Fast-forwarding {self.rollout_sandbox.get_id()} from depth "
                    f"{live_depth} to {len(tool_call_list) - 1} for rollout {self.rollout_id}"
                )
            return self.rollout_sandbox, live_depth

        environment = await self._fresh_environment()
        self._replace_rollout_environment(environment, [])
        return environment, 0

    async def _fork_environment(self, parent_env_id: str) -> Sandbox:
        """Forks a cached snapshot into a live sandbox to resume from it.

        Prefers a pre created copy from the fork bank; otherwise
        attaches to the stored snapshot handle and start()s it, which restores
        the snapshot into a fresh live sandbox.
        """
        if self.fork_generator is not None:
            bank_env_id = self.fork_generator.get_forked_env(self.task_name, parent_env_id)
            if bank_env_id is not None:
                self.logger.debug(
                    f"[ENV]: Using fork bank copy {bank_env_id} of parent {parent_env_id} "
                    f"for rollout {self.rollout_id}"
                )
                return self.tool_call_env_class(task_name=self.task_name, env_id=bank_env_id)

        lpm_sandbox = self.tool_call_env_class(
            task_name=self.task_name, env_id=parent_env_id
        )
        st = time.perf_counter()
        await lpm_sandbox.start()
        et = time.perf_counter()
        await self.snapshot_policy.observe_fork(et - st)
        self.logger.debug(
            f"[ENV]: Forked {lpm_sandbox.get_id()} from parent {parent_env_id} "
            f"in {et - st:.2f} seconds for rollout {self.rollout_id}"
        )
        return lpm_sandbox

    async def _fresh_environment(self) -> Sandbox:
        if self.fork_generator is not None:
            bank_env_id = self.fork_generator.get_forked_env(self.task_name, ROOT_ENV_ID)
            if bank_env_id is not None:
                self.logger.debug(
                    f"[ENV]: Using warm root environment {bank_env_id} for rollout {self.rollout_id}"
                )
                return self.tool_call_env_class(task_name=self.task_name, env_id=bank_env_id)

        environment = self.tool_call_env_class(task_name=self.task_name)
        st = time.perf_counter()
        await environment.start()
        et = time.perf_counter()
        self.logger.debug(
            f"[ENV]: Started fresh environment {environment.get_id()} "
            f"in {et - st:.2f} seconds for rollout {self.rollout_id}"
        )
        return environment

    async def _execute_and_insert(
        self,
        environment: Sandbox,
        path: List[str],
        suffix_calls: List[ToolCall],
        resume_depth: int,
    ) -> str:
        """Executes the unmatched suffix in the environment and populates the cache."""
        last_value = None
        fork_cost_ms = await self.snapshot_policy.prime()
        if fork_cost_ms is not None:
            self.logger.debug(
                f"[ENV]: Server fork cost for task {self.task_name} is "
                f"{fork_cost_ms} ms for rollout {self.rollout_id}"
            )

        for offset, tool_call in enumerate(suffix_calls):
            depth = resume_depth + offset + 1
            self.total_executions += 1
            embedding_future = self._start_embedding(tool_call)

            st = time.perf_counter()
            try:
                value = await environment.execute(tool_call)
            except Exception:
                # The sandbox state is unknown after a raised execution.
                if embedding_future is not None:
                    embedding_future.cancel()
                self.logger.debug(
                    f"[TOOL EXEC]: Execution raised; discarding environment "
                    f"{environment.get_id()} for rollout {self.rollout_id}"
                )
                self._schedule_stop(environment)
                self.rollout_sandbox = None
                self.sandbox_history = []
                self.snapshot_policy.reset()
                raise
            et = time.perf_counter()
            runtime_seconds = et - st
            # Replay would re-pay this call (failed ones included), so it
            # always counts toward the un-checkpointed replay cost.
            self.snapshot_policy.record_execution(runtime_seconds)
            self.logger.debug(
                f"[TOOL EXEC]: Executed tool call {tool_call.serialize()} "
                f"in {runtime_seconds} seconds for rollout {self.rollout_id}"
            )

            if tool_call.will_mutate_state():
                self.sandbox_history.append(tool_call.serialize())

            if value.startswith("Failed to execute"):
                # Environment-reported failure: keep the rollout going but never cache it.
                self.logger.debug(
                    f"[TOOL EXEC]: Not caching failed result at depth {depth} "
                    f"for rollout {self.rollout_id}"
                )
                if embedding_future is not None:
                    embedding_future.cancel()
                last_value = value
                continue
            
            await self._update_tvcache_with_execution(
                tool_call, environment, path, depth, value, runtime_seconds, embedding_future
            )
            last_value = value

        return last_value

    async def _update_tvcache_with_execution(
        self,
        tool_call: ToolCall,
        environment: Sandbox,
        path: List[str],
        depth: int,
        value: str,
        runtime_seconds: float,
        embedding_future: Optional[asyncio.Future] = None,
    ) -> None:

        sandbox_id = None
        
        if tool_call.will_mutate_state() and self.snapshot_policy.should_snapshot():
            unsnapshotted_runtime = self.snapshot_policy.unsnapshotted_runtime
            snapshot_threshold = self.snapshot_policy.threshold()
            fst = time.perf_counter()
            stored_copy = await environment.fork()
            fet = time.perf_counter()
            await self.snapshot_policy.observe_fork(fet - fst)
            sandbox_id = stored_copy.get_id()
            self.logger.debug(
                f"[ENV]: Storing fork {sandbox_id} after {unsnapshotted_runtime:.2f} "
                f"seconds of un-checkpointed runtime (threshold {snapshot_threshold:.2f}, "
                f"fork took {fet - fst:.2f} seconds) for rollout {self.rollout_id}"
            )
            # Even if the insert below loses the sandbox race, the state is
            # checkpointed (by the winner), so the replay debt is paid.
            self.snapshot_policy.reset()

        embedding = await self._await_embedding(embedding_future)

        ist = time.perf_counter()
        insert_result = await self.client.insert(
            self.task_name,
            list(path[:depth]),
            value,
            sandbox_id,
            runtime=self._runtime_ms(runtime_seconds),
            ttl=self.ttl,
            embedding=embedding,
        )
        iet = time.perf_counter()
        self.logger.debug(
            f"Inserted value at depth {depth} in {iet - ist} seconds "
            f"for rollout {self.rollout_id}"
        )

        if sandbox_id is not None and not insert_result.sandbox_inserted:
            # Another rollout won the race to attach a sandbox at this node;
            # our copy is unknown to the cache and would otherwise leak.
            self.logger.debug(
                f"[ENV]: Discarding orphan fork {sandbox_id}: node already had a sandbox"
            )
            self._schedule_stop(
                self.tool_call_env_class(task_name=self.task_name, env_id=sandbox_id)
            )

    def _start_embedding(self, tool_call: ToolCall) -> Optional[asyncio.Future]:
        """Embeds a stateful call in the background so it overlaps with execution."""
        if not self.embed_tool_calls or not tool_call.will_mutate_state():
            return None
        future = asyncio.get_running_loop().run_in_executor(_get_embedding_pool(), tool_call.embed)
        # Marks failures as seen, since failed executions never await their embedding.
        future.add_done_callback(lambda done: done.cancelled() or done.exception())
        return future

    async def _await_embedding(
        self, embedding_future: Optional[asyncio.Future]
    ) -> Optional[List[float]]:
        """Returns the finished embedding, or None if there is none or it is invalid.
        Never raises, so a bad embedding can't fail the rollout."""
        if embedding_future is None:
            return None
        try:
            embedding = list(await embedding_future)
            validate_embedding(embedding)
        except Exception as error:
            # Warn once per executor so a missing embedder isn't silent.
            log = self.logger.debug if self._embedding_failed else self.logger.warning
            self._embedding_failed = True
            log(f"[EMBED]: Skipping embedding due to {error!r} for rollout {self.rollout_id}")
            return None
        return embedding

    def _replace_rollout_environment(
        self, environment: Sandbox, sandbox_history: List[str]
    ) -> None:
        old_environment = self.rollout_sandbox
        if old_environment is not None and old_environment is not environment:
            self.logger.debug(
                f"[ENV]: Replacing rollout environment {old_environment.get_id()} "
                f"for rollout {self.rollout_id}"
            )
            self._schedule_stop(old_environment)
        self.rollout_sandbox = environment
        self.sandbox_history = sandbox_history
        # A fork's resume point is a checkpoint; a fresh env's empty state
        # costs nothing — replay costs re-accumulate as they are paid.
        self.snapshot_policy.reset()

    def _schedule_stop(self, environment: Sandbox) -> None:
        """Stops an environment off the critical path."""

        async def _stop() -> None:
            try:
                await environment.stop()
            except Exception as error:
                self.logger.debug(
                    f"[ENV]: Failed to stop environment {environment.get_id()} due to {error}"
                )

        self._pending_stops = [task for task in self._pending_stops if not task.done()]
        self._pending_stops.append(asyncio.create_task(_stop()))

    async def _stop_rollout_environment(self) -> None:
        if self.rollout_sandbox is None:
            return
        environment = self.rollout_sandbox
        self.rollout_sandbox = None
        self.sandbox_history = []
        self.logger.debug(
            f"[ENV]: Stopping rollout environment {environment.get_id()} "
            f"for rollout {self.rollout_id}"
        )
        try:
            await environment.stop()
        except Exception as error:
            self.logger.debug(
                f"[ENV]: Failed to stop environment {environment.get_id()} due to {error}"
            )

    def _runtime_ms(self, runtime_seconds: float) -> int:
        return min(MAX_RUNTIME_MS, max(0, int(runtime_seconds * 1000)))
