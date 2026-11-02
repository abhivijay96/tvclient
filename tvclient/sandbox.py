"""Sandbox abstraction implemented once per dataset/workload."""

from abc import ABC, abstractmethod
from typing import Optional

from tvclient.tool_call import ToolCall


class Sandbox(ABC):
    """A sandbox in which tool calls execute.

    Construction conventions (the executor, fork bank, and sandbox manager all
    rely on them):
      - env_class(task_name=...)             -> a fresh sandbox; start() boots it
      - env_class(task_name=..., env_id=...) -> attach to an existing sandbox or
                                                snapshot handle (start() restores
                                                a handle into a live sandbox)

    Sandboxes referenced by the cache are stored as immutable snapshot handles:
    storing one means storing (await env.fork()).get_id() — a cold snapshot, not
    a running sandbox. To continue from one, attach to the handle and restore it
    with env_class(task_name, env_id=handle).start(), then execute in the
    result. Deleting a drained snapshot or sandbox means
    env_class(task_name, env_id).stop().
    """

    def __init__(self, task_name: str, env_id: Optional[str] = None):
        self.task_name = task_name
        self.env_id = env_id

    def get_id(self) -> str:
        return self.env_id

    @abstractmethod
    async def start(self) -> None:
        """Materializes a live sandbox and sets env_id to the running sandbox.

        With no env_id, boots a fresh, empty sandbox. With env_id set to a
        snapshot handle (from a prior fork()), restores that snapshot into a new
        live sandbox and updates env_id to the live sandbox's id.
        """
        ...

    @abstractmethod
    async def fork(self) -> "Sandbox":
        """Snapshots this live sandbox and returns a cold handle to the snapshot.

        The returned env is not running: its get_id() is a durable snapshot
        handle, suitable for storing in the cache as an immutable parent. To
        execute from it, construct env_class(task_name, env_id=handle) and call
        start() to restore it. Snapshotting is non-destructive — this sandbox
        keeps running.
        """
        ...

    @abstractmethod
    async def execute(self, tool_call: ToolCall) -> str:
        """Executes one tool call in this sandbox and returns its result."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stops and deletes this sandbox."""
        ...

    async def state_digest(self) -> Optional[dict]:
        """Optional: a structured digest of this live sandbox's state (see
        correctness/tvverify/digest). Workloads that can hash their state
        override this; the default means "not available"."""
        return None

    async def equals(self, other: "Sandbox") -> bool:
        """Optional: whether this live sandbox and `other` (also live) hold the
        same state. Only the correctness framework calls it, never the executor.

        Override it directly (e.g. diff two containers' filesystems, compare two
        database dumps), or implement state_digest() and keep this default,
        which compares the strict components of both digests.
        """
        mine, theirs = await self.state_digest(), await other.state_digest()
        if mine is None or theirs is None:
            raise NotImplementedError(
                f"{type(self).__name__} must implement equals() or state_digest() "
                "so sandbox states can be compared"
            )

        # Same rule as StateDigest.strict_hash (correctness/tvverify/digest),
        # restated here so tvclient does not depend on tvverify.
        def strict(digest: dict) -> dict:
            return {
                name: comp["hash"]
                for name, comp in digest["components"].items()
                if comp.get("strict", True)
            }

        return strict(mine) == strict(theirs)
