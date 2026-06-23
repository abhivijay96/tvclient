"""Snapshot policy: decides when a sandbox state is worth checkpointing.

A snapshot pays off on its first reuse when the replay cost it saves exceeds
storing the fork now plus restoring it later, i.e. about 2x the fork time. This
object tracks the un-checkpointed replay cost (the executed time since the last
state that has a snapshot) and prices it against the task's fork cost.

The fork-cost EMA lives on the cache server because rollouts run in separate
worker processes (Ray) and the server is their only shared point; this holds a
local copy fetched lazily and refreshed by every report.
"""

from typing import Optional

from tvclient.client import TVCacheClient

# A snapshot pays off on its first reuse when the replay cost it saves exceeds
# storing the fork now plus restoring it later, i.e. about 2x the fork time.
SNAPSHOT_COST_MULTIPLIER = 2.0
# Governs the adaptive threshold until the first fork sample for a task exists.
INITIAL_FORK_COST_SECONDS = 1.0


class SnapshotPolicy:
    """Per-rollout accounting for the adaptive snapshot threshold."""

    def __init__(
        self,
        client: TVCacheClient,
        task_name: str,
        adaptive: bool = True,
        fixed_threshold_seconds: float = 0.0,
        snapshot_cost_multiplier: float = SNAPSHOT_COST_MULTIPLIER,
        initial_fork_cost_seconds: float = INITIAL_FORK_COST_SECONDS,
    ):
        if not adaptive and fixed_threshold_seconds <= 0:
            raise ValueError("fixed mode requires a positive fixed_threshold_seconds")
        self.client = client
        self.task_name = task_name
        # adaptive: threshold = multiplier x estimated fork cost. Otherwise the
        # threshold is fixed_threshold_seconds (tests use this).
        self.adaptive = adaptive
        self.fixed_threshold_seconds = fixed_threshold_seconds
        self.snapshot_cost_multiplier = snapshot_cost_multiplier
        self.initial_fork_cost_seconds = initial_fork_cost_seconds
        # Local copy of the server's fork-cost EMA in ms; fetched lazily on the
        # first miss, refreshed by every report_fork_cost response.
        self.fork_cost_ms: Optional[int] = None
        # Executed seconds since the last state that has a snapshot: the replay
        # cost a snapshot at the current state would save.
        self.unsnapshotted_runtime = 0.0

    def record_execution(self, runtime_seconds: float) -> None:
        """Adds one executed call to the un-checkpointed replay cost. Failed
        calls count too: replay would re-pay them."""
        self.unsnapshotted_runtime += runtime_seconds

    def reset(self) -> None:
        """Marks the current state as checkpointed, so replay cost re-accumulates
        from here. Called after a snapshot is stored, a resume fork, a fresh
        (empty) env, or a discarded environment."""
        self.unsnapshotted_runtime = 0.0

    def should_snapshot(self) -> bool:
        """True when the un-checkpointed replay cost exceeds the threshold."""
        return self.unsnapshotted_runtime > self.threshold()

    def threshold(self) -> float:
        """Replay-cost threshold above which the current state gets a snapshot:
        a fixed override, or a multiple of the estimated fork cost."""
        if self.adaptive:
            return self.snapshot_cost_multiplier * self._fork_cost_seconds()
        return self.fixed_threshold_seconds

    def _fork_cost_seconds(self) -> float:
        """Best estimate of this task's fork cost in seconds: the server-side
        EMA once sampled, else the seed (fork_cost_ms == 0 means no samples)."""
        if self.fork_cost_ms:
            return self.fork_cost_ms / 1000
        return self.initial_fork_cost_seconds

    async def prime(self) -> Optional[int]:
        """Lazily fetches the task's fork-cost EMA from the server (adaptive mode
        only), once per rollout. Returns the fetched value (0 means no samples
        yet) the first time, or None when nothing was fetched."""
        if self.adaptive and self.fork_cost_ms is None:
            self.fork_cost_ms = await self.client.get_fork_cost(self.task_name)
            return self.fork_cost_ms
        return None

    async def observe_fork(self, fork_seconds: float) -> None:
        """Folds one measured env.fork() into the task's server-side EMA and
        refreshes the local copy from the response."""
        sample_ms = max(1, int(fork_seconds * 1000))  # 0 means "no samples" server-side
        self.fork_cost_ms = await self.client.report_fork_cost(self.task_name, sample_ms)
