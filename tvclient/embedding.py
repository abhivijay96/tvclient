"""Embedders that turn tool calls into vectors for semantic caching."""

import threading
from abc import ABC, abstractmethod
from typing import Any, List, Optional


class Embedder(ABC):
    """Turns text into an embedding vector."""

    @abstractmethod
    def encode(self, text: str) -> List[float]:
        ...


class SentenceTransformerEmbedder(Embedder):
    """Embeds text with a local sentence-transformers model, loaded on first use.

    Requires the `embedding` extra. Pass trust_remote_code=True only for models
    that ship their own code and that you trust.
    """

    def __init__(
        self,
        model_name: str,
        max_seq_length: Optional[int] = None,
        trust_remote_code: bool = False,
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.trust_remote_code = trust_remote_code
        self._model: Any = None
        # Guards the lazy load and serializes encode, since fast tokenizers fail
        # when several threads use them at once.
        self._lock = threading.Lock()

    def encode(self, text: str) -> List[float]:
        with self._lock:
            if self._model is None:
                self._model = self._load()
            return self._model.encode(text).tolist()

    def _load(self) -> Any:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            self.model_name, trust_remote_code=self.trust_remote_code
        )
        if self.max_seq_length is not None:
            model.max_seq_length = self.max_seq_length
        return model


_embedder: Optional[Embedder] = None


def set_embedder(embedder: Embedder) -> None:
    """Sets the embedder that ToolCall.embed() uses in this process."""
    global _embedder
    _embedder = embedder


def get_embedder() -> Embedder:
    """Returns the embedder set by set_embedder()."""
    if _embedder is None:
        raise RuntimeError("no embedder is set; call tvclient.set_embedder() first")
    return _embedder
