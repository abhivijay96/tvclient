"""Tool call objects: the unit of the cache key."""

import json
from abc import ABC, abstractmethod
from typing import List

from tvclient.embedding import get_embedder


class ToolCall(ABC):
    """One tool invocation (tool name plus arguments).

    will_mutate_state() decides whether the call participates in the stateful
    chain used as the cache key: stateless calls are looked up against the
    stateful prefix but never extend it, and are skipped when a sandbox is
    replayed. The default is True, which is the safe assumption when a tool's
    behavior depends on sandbox state (e.g. bash programs).
    """

    @abstractmethod
    def to_dict(self) -> dict:
        ...

    def will_mutate_state(self) -> bool:
        return True

    def serialize(self) -> str:
        # sort_keys makes the serialization canonical so the same logical call
        # always produces the same cache key.
        return json.dumps(self.to_dict(), sort_keys=True)

    def embed(self) -> List[float]:
        """Embeds this call with the embedder from set_embedder(). Override to
        embed different text or use a different model."""
        return get_embedder().encode(self.serialize())


class TestToolCall(ToolCall):
    """Terminal sentinel call whose cached value is the rollout's test result.

    Stateless: the rollout ends after the test, so the call must never extend
    a stateful chain and a post-test sandbox snapshot would be useless.
    """

    def __init__(self, command: str = ""):
        self.command = command

    def to_dict(self) -> dict:
        return {"TV_CACHE_TOOL_TYPE": "TESTING TOOL"}

    def will_mutate_state(self) -> bool:
        return False
