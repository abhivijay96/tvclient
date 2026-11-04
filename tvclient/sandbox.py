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

    async def equals(self, other: "Sandbox") -> bool:
        """Returns whether this live sandbox's state matches other's.

        Both sandboxes must be live (started). Each workload decides which
        state counts, for example ignoring timestamps, but the result should
        behave like equality. Comparing must not change the compared state.
        The default raises NotImplementedError.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement equals")

    @abstractmethod
    async def stop(self) -> None:
        """Stops and deletes this sandbox."""
        ...
