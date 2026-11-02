"""Optional instrumentation hooks for TvcacheToolExecutor.

Every hook is a no-op by default, so an executor without an observer behaves
exactly as before. The correctness framework (correctness/tvverify) implements
these to record what the cache served, where each miss resumed from, what was
executed and which snapshots were stored.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional

from tvclient.client import FindResult
from tvclient.protocol import InsertResult
from tvclient.tool_call import ToolCall


@dataclass(frozen=True)
class LookupContext:
    """Identifies one lookup_or_execute call."""

    task_id: str
    rollout_id: str
    call_index: int  # 1-based count of lookups made by this executor
    path: List[str]  # serialized stateful history + serialized current call
    current: ToolCall


class ExecutionObserver:
    def wants_digest(self, ctx: LookupContext, depth: int) -> bool:
        """Whether to call Sandbox.state_digest() after the call executed at
        `depth` (digests can be expensive, so the observer opts in per call)."""
        return False

    async def on_lookup(self, ctx: LookupContext, found: FindResult) -> None:
        pass

    async def on_resume(
        self,
        ctx: LookupContext,
        source: Literal["fork", "live", "fresh"],
        depth: int,
        env_id: Optional[str],
    ) -> None:
        """A miss picked its resume point: `depth` entries of ctx.path are
        already applied in the sandbox."""

    async def on_executed(
        self,
        ctx: LookupContext,
        depth: int,
        call: ToolCall,
        value: str,
        runtime_seconds: float,
        failed: bool,
        digest: Optional[dict],
    ) -> None:
        pass

    async def on_snapshot(
        self, ctx: LookupContext, depth: int, sandbox_id: str, digest: Optional[dict]
    ) -> None:
        pass

    async def on_insert(
        self,
        ctx: LookupContext,
        path: List[str],
        value: Optional[str],
        sandbox_id: Optional[str],
        result: InsertResult,
    ) -> None:
        pass

    async def on_return(
        self, ctx: LookupContext, value: str, source: Literal["hit", "exec"]
    ) -> None:
        pass
