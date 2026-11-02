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
        """Returns the tool call as a dictionary.

        INVARIANT: The dictionary must be representable in JSON, so that
        json.loads(self.serialize()) == self.to_dict(). That is, use strings for
        keys and strings, numbers, booleans, lists, dictionaries or None for
        values. This also applies to nested lists and dictionaries.

        Sort any collection whose order doesn't matter, such as a set, so the
        same call always serializes the same way.
        """
        ...

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCall":
        """Rebuilds a tool call from the dictionary produced by to_dict.

        The rebuilt call must behave like the original when executed and
        serialize the same, so
        cls.from_dict(call.to_dict()).serialize() == call.serialize().
        The default raises NotImplementedError.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement from_dict")

    def will_mutate_state(self) -> bool:
        return True

    def serialize(self) -> str:
        # sort_keys makes the serialization canonical so the same logical call
        # always produces the same cache key.
        return json.dumps(self.to_dict(), sort_keys=True)

    def embed(self) -> List[float]:
        """Embeds this call with the embedder from set_embedder().

        By default, a tool call's serialization is embedded, which may make
        different calls look alike because of shared JSON structure and keys.
        We suggest that `embed` be overridden using the pattern

        `get_embedder().encode(<tool call>)`

        where `<tool call>` is how the tool call is being represented.
        """
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
