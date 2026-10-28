"""Binary wire protocol for the TVCache server.

Request op bytes 1-8 and response type bytes 1-8, big-endian, with
bool-prefixed optional fields. This module only knows how to encode request
payloads and decode responses from an asyncio.StreamReader; connection
handling lives in TVCacheClient.
"""

import asyncio
import struct
from dataclasses import dataclass
from typing import List, Optional


INSERT_REQUEST = 1
FIND_REQUEST = 2
DUMP_ALL_REQUEST = 3
UNREF_SANDBOX_REQUEST = 4
MARK_SANDBOXES_EVICTED_REQUEST = 5
LIST_SANDBOX_CANDIDATES_REQUEST = 6
DRAIN_RECLAIMABLE_SANDBOXES_REQUEST = 7
LIST_HOT_SANDBOXES_REQUEST = 8
REPORT_FORK_COST_REQUEST = 9
GET_FORK_COST_REQUEST = 10

INSERT_RESULT_RESPONSE = 1
FIND_RESULT_RESPONSE = 2
ERROR_RESPONSE = 3
JSON_RESPONSE = 4
UNREF_SANDBOX_RESULT_RESPONSE = 5
MARK_SANDBOXES_EVICTED_RESULT_RESPONSE = 6
SANDBOX_CANDIDATES_RESPONSE = 7
RECLAIMABLE_SANDBOXES_RESPONSE = 8
FORK_COST_RESPONSE = 9


class TVCacheServerError(Exception):
    """Raised when the server answers a request with an error response."""


@dataclass(frozen=True)
class InsertResult:
    value_inserted: bool
    sandbox_inserted: bool


@dataclass(frozen=True)
class FindResponse:
    value: Optional[str]
    sandbox_path: Optional[List[str]]
    sandbox_id: Optional[str]


@dataclass(frozen=True)
class UnrefResult:
    remaining_references: int


@dataclass(frozen=True)
class MarkResult:
    sandbox_ids_recorded: int


@dataclass(frozen=True)
class SandboxCandidate:
    task_id: str
    sandbox_path: List[str]
    sandbox_id: str
    score: int


@dataclass(frozen=True)
class SandboxCandidates:
    candidates: List[SandboxCandidate]


@dataclass(frozen=True)
class ReclaimableSandbox:
    task_id: str
    sandbox_id: str


@dataclass(frozen=True)
class ReclaimableSandboxes:
    sandboxes: List[ReclaimableSandbox]


@dataclass(frozen=True)
class SandboxEvictionRecord:
    task_id: str
    sandbox_path: List[str]
    sandbox_id: str


@dataclass(frozen=True)
class ForkCost:
    fork_cost_ms: int


@dataclass(frozen=True)
class JsonResponse:
    json: str


def write_u8(value: int) -> bytes:
    return struct.pack(">B", value)


def write_bool(value: bool) -> bytes:
    return write_u8(1 if value else 0)


def write_u16(value: int) -> bytes:
    return struct.pack(">H", value)


def write_u32(value: int) -> bytes:
    return struct.pack(">I", value)


def write_string_u16(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return write_u16(len(encoded)) + encoded


def write_string_u32(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return write_u32(len(encoded)) + encoded


def write_path(path: List[str]) -> bytes:
    encoded = write_u16(len(path))
    for entry in path:
        encoded += write_string_u16(entry)
    return encoded


# The most numbers an embedding can have. This is 2^16 - 1, because the
# dimension is sent as a u16
MAX_EMBEDDING_DIMENSION = 2**16 - 1

# The largest number an f32 can hold, because embedding values are stored
# as f32's
MAX_FLOAT32 = 3.4028234663852886e38


def validate_embedding(embedding: List[float]) -> None:
    """Raises ValueError if the embedding can't be sent on the wire."""
    if not 1 <= len(embedding) <= MAX_EMBEDDING_DIMENSION:
        raise ValueError(f"embedding dimension {len(embedding)} is out of range")
    # Also rejects NaN and infinity, which fail every comparison.
    if not all(abs(x) <= MAX_FLOAT32 for x in embedding):
        raise ValueError("embedding has values that don't fit in float32")


def write_embedding(embedding: List[float]) -> bytes:
    validate_embedding(embedding)
    return struct.pack(f">H{len(embedding)}f", len(embedding), *embedding)


def encode_insert(
    task_id: str,
    path: List[str],
    value: Optional[str],
    sandbox_id: Optional[str],
    runtime: int,
    ttl: Optional[int] = None,
    embedding: Optional[List[float]] = None,
) -> bytes:
    encoded = write_u8(INSERT_REQUEST)
    encoded += write_string_u16(task_id)
    encoded += write_path(path)
    encoded += write_u32(runtime)  # milliseconds

    encoded += write_bool(value is not None)
    if value is not None:
        encoded += write_string_u32(value)

    encoded += write_bool(sandbox_id is not None)
    if sandbox_id is not None:
        encoded += write_string_u16(sandbox_id)

    encoded += write_bool(ttl is not None)
    if ttl is not None:
        encoded += write_u32(ttl)

    encoded += write_bool(embedding is not None)
    if embedding is not None:
        encoded += write_embedding(embedding)

    return encoded


def encode_find(task_id: str, path: List[str]) -> bytes:
    return write_u8(FIND_REQUEST) + write_string_u16(task_id) + write_path(path)


def encode_dump_all() -> bytes:
    return write_u8(DUMP_ALL_REQUEST)


def encode_unref_sandbox(task_id: str, sandbox_id: str) -> bytes:
    return (
        write_u8(UNREF_SANDBOX_REQUEST)
        + write_string_u16(task_id)
        + write_string_u16(sandbox_id)
    )


def encode_mark_sandboxes_evicted(records: List[SandboxEvictionRecord]) -> bytes:
    encoded = write_u8(MARK_SANDBOXES_EVICTED_REQUEST) + write_u16(len(records))
    for record in records:
        encoded += write_string_u16(record.task_id)
        encoded += write_path(record.sandbox_path)
        encoded += write_string_u16(record.sandbox_id)
    return encoded


def encode_list_sandbox_candidates(limit: int = 10) -> bytes:
    return write_u8(LIST_SANDBOX_CANDIDATES_REQUEST) + write_u16(limit)


def encode_drain_reclaimable_sandboxes() -> bytes:
    return write_u8(DRAIN_RECLAIMABLE_SANDBOXES_REQUEST)


def encode_list_hot_sandboxes(limit: int = 10) -> bytes:
    return write_u8(LIST_HOT_SANDBOXES_REQUEST) + write_u16(limit)


def encode_report_fork_cost(task_id: str, fork_cost_ms: int) -> bytes:
    return (
        write_u8(REPORT_FORK_COST_REQUEST)
        + write_string_u16(task_id)
        + write_u32(fork_cost_ms)
    )


def encode_get_fork_cost(task_id: str) -> bytes:
    return write_u8(GET_FORK_COST_REQUEST) + write_string_u16(task_id)


async def read_u8(reader: asyncio.StreamReader) -> int:
    return struct.unpack(">B", await reader.readexactly(1))[0]


async def read_bool(reader: asyncio.StreamReader) -> bool:
    value = await read_u8(reader)
    if value == 0:
        return False
    if value == 1:
        return True
    raise ValueError(f"invalid bool byte: {value}")


async def read_u16(reader: asyncio.StreamReader) -> int:
    return struct.unpack(">H", await reader.readexactly(2))[0]


async def read_u32(reader: asyncio.StreamReader) -> int:
    return struct.unpack(">I", await reader.readexactly(4))[0]


async def read_u64(reader: asyncio.StreamReader) -> int:
    return struct.unpack(">Q", await reader.readexactly(8))[0]


async def read_string_u16(reader: asyncio.StreamReader) -> str:
    return (await reader.readexactly(await read_u16(reader))).decode("utf-8")


async def read_string_u32(reader: asyncio.StreamReader) -> str:
    return (await reader.readexactly(await read_u32(reader))).decode("utf-8")


async def read_path(reader: asyncio.StreamReader) -> List[str]:
    count = await read_u16(reader)
    return [await read_string_u16(reader) for _ in range(count)]


async def read_optional_string_u16(reader: asyncio.StreamReader) -> Optional[str]:
    if await read_bool(reader):
        return await read_string_u16(reader)
    return None


async def read_optional_string_u32(reader: asyncio.StreamReader) -> Optional[str]:
    if await read_bool(reader):
        return await read_string_u32(reader)
    return None


async def read_optional_path(reader: asyncio.StreamReader) -> Optional[List[str]]:
    if await read_bool(reader):
        return await read_path(reader)
    return None


async def read_sandbox_candidate(reader: asyncio.StreamReader) -> SandboxCandidate:
    return SandboxCandidate(
        task_id=await read_string_u16(reader),
        sandbox_path=await read_path(reader),
        sandbox_id=await read_string_u16(reader),
        score=await read_u64(reader),
    )


async def decode_response(reader: asyncio.StreamReader) -> object:
    """Decodes one response into its dataclass; raises TVCacheServerError on type 3."""
    response_type = await read_u8(reader)

    if response_type == INSERT_RESULT_RESPONSE:
        return InsertResult(
            value_inserted=await read_bool(reader),
            sandbox_inserted=await read_bool(reader),
        )

    if response_type == FIND_RESULT_RESPONSE:
        return FindResponse(
            value=await read_optional_string_u32(reader),
            sandbox_path=await read_optional_path(reader),
            sandbox_id=await read_optional_string_u16(reader),
        )

    if response_type == ERROR_RESPONSE:
        raise TVCacheServerError(await read_string_u32(reader))

    if response_type == JSON_RESPONSE:
        return JsonResponse(json=await read_string_u32(reader))

    if response_type == UNREF_SANDBOX_RESULT_RESPONSE:
        return UnrefResult(remaining_references=await read_u32(reader))

    if response_type == MARK_SANDBOXES_EVICTED_RESULT_RESPONSE:
        return MarkResult(sandbox_ids_recorded=await read_u32(reader))

    if response_type == SANDBOX_CANDIDATES_RESPONSE:
        count = await read_u16(reader)
        return SandboxCandidates(
            candidates=[await read_sandbox_candidate(reader) for _ in range(count)]
        )

    if response_type == RECLAIMABLE_SANDBOXES_RESPONSE:
        count = await read_u32(reader)
        return ReclaimableSandboxes(
            sandboxes=[
                ReclaimableSandbox(
                    task_id=await read_string_u16(reader),
                    sandbox_id=await read_string_u16(reader),
                )
                for _ in range(count)
            ]
        )

    if response_type == FORK_COST_RESPONSE:
        return ForkCost(fork_cost_ms=await read_u32(reader))

    raise ValueError(f"unknown response type: {response_type}")
