"""Async TVCache client over one persistent connection.

Each rollout owns one TVCacheClient (and therefore one connection); requests
on a connection are sequential, and concurrency comes from many rollouts each
holding their own client.
"""

import asyncio
import json
import logging
import time
from typing import List, Optional

from tvclient import protocol
from tvclient.protocol import (
    FindResponse,
    ForkCost,
    InsertResult,
    JsonResponse,
    MarkResult,
    ReclaimableSandbox,
    ReclaimableSandboxes,
    SandboxCandidate,
    SandboxCandidates,
    SandboxEvictionRecord,
    TVCacheServerError,
    UnrefResult,
)


class FindResult:
    """A find response plus the lease on the returned sandbox.

    The server increments the sandbox's reference count whenever find returns a
    sandbox_id, so every FindResult must be released exactly once. Use it as an
    async context manager:

        async with await client.find(task_id, path) as found:
            ...  # fork here; the snapshot cannot be reclaimed while leased

    release() is idempotent and a no-op when there is no sandbox, and it runs
    even when the body raises, so a crashed rollout cannot pin an eviction
    victim forever.
    """

    def __init__(self, client: "TVCacheClient", task_id: str, response: FindResponse):
        self.value = response.value
        self.sandbox_path = response.sandbox_path
        self.sandbox_id = response.sandbox_id
        self._client = client
        self._task_id = task_id
        self._released = response.sandbox_id is None

    async def release(self) -> None:
        """Decrements the sandbox's reference count on the server (at most once)."""
        if self._released:
            return
        self._released = True
        await self._client.unref_sandbox(self._task_id, self.sandbox_id)

    async def __aenter__(self) -> "FindResult":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.release()


class TVCacheClient:
    """Typed async client for all TVCache server operations."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7878,
        timeout: float = 5,
        retries: int = 3,
        retry_backoff: float = 0.05,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff = retry_backoff
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self) -> "TVCacheClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def connect(self) -> None:
        last_error: Optional[BaseException] = None

        for attempt in range(self.retries + 1):
            try:
                self.reader, self.writer = await asyncio.wait_for(
                    asyncio.open_connection(self.host, self.port),
                    timeout=self.timeout,
                )
                return
            except (OSError, asyncio.TimeoutError) as error:
                last_error = error
                if attempt == self.retries:
                    break
                await asyncio.sleep(self.retry_backoff * (attempt + 1))

        assert last_error is not None
        raise last_error

    async def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except (OSError, asyncio.TimeoutError):
                pass
            self.reader = None
            self.writer = None

    async def _send_request(self, request: bytes) -> object:
        """Sends one request and decodes one response.

        A broken connection is discarded so the next request dials fresh; the
        failed request itself is not retried (a retried find would double-count
        a sandbox reference).
        """
        if self.reader is None or self.writer is None:
            await self.connect()

        try:
            self.writer.write(request)
            await asyncio.wait_for(self.writer.drain(), timeout=self.timeout)
            return await asyncio.wait_for(
                protocol.decode_response(self.reader),
                timeout=self.timeout,
            )
        except (OSError, EOFError, asyncio.IncompleteReadError, asyncio.TimeoutError):
            await self.close()
            raise

    async def insert(
        self,
        task_id: str,
        path: List[str],
        value: Optional[str],
        sandbox_id: Optional[str],
        runtime: int,
        ttl: Optional[int] = None,
        embedding: Optional[List[float]] = None,
    ) -> InsertResult:
        response = await self._send_request(
            protocol.encode_insert(task_id, path, value, sandbox_id, runtime, ttl, embedding)
        )
        return self._expect(response, InsertResult)

    async def find(self, task_id: str, path: List[str]) -> FindResult:
        """Longest-prefix-match lookup returning a leased result.

        Use as `async with await client.find(task_id, path) as found:` so the
        reference taken by the server is always released.
        """
        return FindResult(self, task_id, await self.find_unmanaged(task_id, path))

    async def find_unmanaged(self, task_id: str, path: List[str]) -> FindResponse:
        """Raw find. If sandbox_id is set the server has already incremented its
        reference count and the caller owns the matching unref_sandbox call."""
        response = await self._send_request(protocol.encode_find(task_id, path))
        return self._expect(response, FindResponse)

    async def unref_sandbox(self, task_id: str, sandbox_id: str) -> int:
        response = await self._send_request(
            protocol.encode_unref_sandbox(task_id, sandbox_id)
        )
        return self._expect(response, UnrefResult).remaining_references

    async def mark_sandboxes_evicted(self, records: List[SandboxEvictionRecord]) -> int:
        response = await self._send_request(
            protocol.encode_mark_sandboxes_evicted(records)
        )
        return self._expect(response, MarkResult).sandbox_ids_recorded

    async def list_sandbox_candidates(self, limit: int = 10) -> List[SandboxCandidate]:
        """Eviction candidates, ascending lpm-hit score (least useful first)."""
        response = await self._send_request(
            protocol.encode_list_sandbox_candidates(limit)
        )
        return self._expect(response, SandboxCandidates).candidates

    async def list_hot_sandboxes(self, limit: int = 10) -> List[SandboxCandidate]:
        """Fork-ahead candidates, descending lpm-hit score (most useful first)."""
        response = await self._send_request(protocol.encode_list_hot_sandboxes(limit))
        return self._expect(response, SandboxCandidates).candidates

    async def drain_reclaimable_sandboxes(self) -> List[ReclaimableSandbox]:
        """Takes ownership of every zero-reference victim sandbox. Destructive:
        only the single deletion loop (SandboxManager) should call this."""
        response = await self._send_request(
            protocol.encode_drain_reclaimable_sandboxes()
        )
        return self._expect(response, ReclaimableSandboxes).sandboxes

    async def report_fork_cost(self, task_id: str, fork_cost_ms: int) -> int:
        """Folds one measured env.fork() time into the task's server-side EMA
        and returns the updated estimate (shared across all workers)."""
        response = await self._send_request(
            protocol.encode_report_fork_cost(task_id, fork_cost_ms)
        )
        return self._expect(response, ForkCost).fork_cost_ms

    async def get_fork_cost(self, task_id: str) -> int:
        """The task's fork-cost EMA in milliseconds; 0 means no samples yet."""
        response = await self._send_request(protocol.encode_get_fork_cost(task_id))
        return self._expect(response, ForkCost).fork_cost_ms

    async def dump_all(self) -> dict:
        st = time.perf_counter()
        response = await self._send_request(protocol.encode_dump_all())
        et = time.perf_counter()
        self.logger.debug(f"dump_all took {et - st} seconds")
        return json.loads(self._expect(response, JsonResponse).json)

    def _expect(self, response: object, expected_type: type) -> object:
        if not isinstance(response, expected_type):
            raise TVCacheServerError(
                f"expected {expected_type.__name__}, got {type(response).__name__}"
            )
        return response
