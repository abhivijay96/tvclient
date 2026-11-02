"""A cache-free executor with the same API as TvcacheToolExecutor.

It is the ground truth the correctness framework compares the cache against:
one live sandbox per rollout, every call executed for real, nothing stored.
"""

from typing import List, Optional, Type

from tvclient.sandbox import Sandbox
from tvclient.tool_call import TestToolCall, ToolCall


class NoCacheToolExecutor:
    def __init__(
        self,
        tool_call_env_class: Type[Sandbox],
        tool_call_class: Optional[Type[ToolCall]],
        task_id: str,
        **_ignored: object,
    ):
        self.tool_call_env_class = tool_call_env_class
        self.tool_call_class = tool_call_class
        self.task_name = task_id
        self.rollout_sandbox: Optional[Sandbox] = None
        self.sandbox_history: List[str] = []
        self.total_calls = 0
        self.total_executions = 0

    async def __aenter__(self) -> "NoCacheToolExecutor":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self.rollout_sandbox is not None:
            sandbox, self.rollout_sandbox = self.rollout_sandbox, None
            self.sandbox_history = []
            await sandbox.stop()

    def set_rollout_id(self, rollout_id: str) -> None:
        pass

    async def lookup_or_execute(self, tool_commands: List[ToolCall]) -> str:
        """Executes the stateful history (if the live sandbox is behind) and
        then the last call. Stateless intermediates are skipped, matching the
        semantics the cached executor relies on."""
        self.total_calls += 1
        history, current = tool_commands[:-1], tool_commands[-1]
        stateful = [call.serialize() for call in history if call.will_mutate_state()]
        stateful_calls = [call for call in history if call.will_mutate_state()]

        # Reuse the live sandbox only if it holds a prefix of this request's
        # stateful history (not ahead of it, not diverged); otherwise restart.
        if self.rollout_sandbox is None or stateful[: len(self.sandbox_history)] != self.sandbox_history:
            await self.close()
            self.rollout_sandbox = self.tool_call_env_class(task_name=self.task_name)
            await self.rollout_sandbox.start()
            self.sandbox_history = []

        for call in stateful_calls[len(self.sandbox_history):]:
            self.total_executions += 1
            await self.rollout_sandbox.execute(call)
            self.sandbox_history.append(call.serialize())

        self.total_executions += 1
        value = await self.rollout_sandbox.execute(current)
        if current.will_mutate_state():
            self.sandbox_history.append(current.serialize())
        return value

    async def test(self, tool_call_history: List[ToolCall]) -> str:
        result = await self.lookup_or_execute(tool_call_history + [TestToolCall("")])
        await self.close()
        return result
