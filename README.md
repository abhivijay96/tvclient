# tvclient

Async Python client for the **TVCache** server. It caches and forks *stateful*
tool-call execution across agentic rollouts, so identical rollout prefixes run
once instead of once per rollout.

The cache key for a step is the **stateful chain** (the calls so far whose
`will_mutate_state()` is `True`) plus the current call. On a hit the executor
returns the stored result without touching a sandbox; on a miss it resumes from
the deepest cached sandbox (forking it) and executes only the missing suffix.

## Requirements

- Python `>= 3.10`
- A running TVCache server. Defaults to `127.0.0.1:7878` (start it with
  `cargo run` in the cache repo).

## Install

From the repo root:

```bash
pip install -e .        # or: uv pip install -e .
```

## Core model

- You implement two abstractions for your workload: a **`ToolCall`** (the unit
  of the cache key) and a **`Sandbox`** (where calls execute).
- You create **one `TvcacheToolExecutor` per rollout**. It owns one cache
  connection and at most one live sandbox.
- Each step you pass the **cumulative list** of calls to
  `lookup_or_execute(...)`. The executor filters stateless calls itself, don't
  pre-filter.
- Sandbox lifecycle: `fork()` snapshots the live sandbox into a **cold handle**;
  `start()` either boots a fresh sandbox or **restores a handle** into a new
  live one.

## 1. Implement a `ToolCall`

Override `to_dict()` (used to build the cache key). Override
`will_mutate_state()` to return `False` for read-only calls — they are looked up
against the stateful prefix but never extend the key.

```python
from tvclient import ToolCall


class CommandToolCall(ToolCall):
    def __init__(self, command: str, mutates: bool = True):
        self.command = command
        self.mutates = mutates

    def to_dict(self) -> dict:
        return {"command": self.command}

    def will_mutate_state(self) -> bool:
        return self.mutates
```

## 2. Implement a `Sandbox`

Construction conventions the executor relies on:

- `Env(task_name=...)` → a fresh sandbox; `start()` boots it.
- `Env(task_name=..., env_id=handle)` → attach to a snapshot handle;
  `start()` restores it into a live sandbox.

The cache only ever stores **cold snapshot handles** (`(await env.fork()).get_id()`).
`fork()` is non-destructive: the live sandbox keeps running.

```python
from tvclient import Sandbox, ToolCall


class MySandbox(Sandbox):
    async def start(self) -> None:
        """Boot a fresh sandbox (no env_id) or restore the attached snapshot."""
        if self.env_id is None:
            self.env_id = boot_new_sandbox()
        else:
            self.env_id = restore_snapshot(self.env_id)

    async def fork(self) -> "MySandbox":
        """Snapshot this live sandbox; return a cold handle (not running)."""
        handle = MySandbox(task_name=self.task_name)
        handle.env_id = snapshot(self.env_id)   # durable snapshot id
        return handle

    async def execute(self, tool_call: ToolCall) -> str:
        return run_in_sandbox(self.env_id, tool_call)

    async def stop(self) -> None:
        """Stop and delete this sandbox or snapshot."""
        delete_sandbox(self.env_id)
```

A sandbox-reported failure should be returned as a string starting with
`"Failed to execute"` — the executor keeps the rollout going but never caches
that result. Raising from `execute()` instead discards the sandbox.

## 3. Drive a rollout

Use the executor as an async context manager. Accumulate calls and call
`lookup_or_execute` each step with the full history so far; finish with `test()`,
which returns the (possibly cached) test result and stops the sandbox.

```python
import asyncio
from tvclient import TvcacheToolExecutor


async def run_rollout(task_id: str):
    history = []
    async with TvcacheToolExecutor(MySandbox, CommandToolCall, task_id) as ex:
        for command in ["ls", "edit file.py", "pytest"]:
            history.append(CommandToolCall(command))
            result = await ex.lookup_or_execute(history)
            print(result)

        return await ex.test(history)   # terminal: caches + stops the sandbox


asyncio.run(run_rollout("my-task"))
```

Useful options / methods:

- `TvcacheToolExecutor(env_class, tool_call_class, task_id, host=..., port=...)`
- `adaptive=True` (default) sizes the snapshot threshold from the server's
  fork-cost estimate; pass `adaptive=False, fixed_threshold_seconds=...` for a
  fixed threshold.
- `set_rollout_id(rollout_id)` — names the per-rollout debug log
  (`./rollouts/<id>.log`).
- `set_fork_bank(bank)` — see below.

## Optional: warm forks and eviction

**`ForkBank`** pre-forks warm roots and copies of the hottest cached sandboxes so
rollouts skip the synchronous fork. Refill it at iteration boundaries:

```python
from tvclient import ForkBank

bank = ForkBank(MySandbox, tasks=["my-task"])
await bank.warm_up()          # call before each training iteration
executor.set_fork_bank(bank)  # executor draws from the bank on a miss
```

**`SandboxManager`** is the single deletion loop and eviction entry point — run
**exactly one per server**:

```python
from tvclient import SandboxManager

manager = SandboxManager(MySandbox)
await manager.evict(limit=10)                          # mark least-useful sandboxes
await manager.run_drain_loop(interval_seconds=5, stop_event=stop)  # delete drained victims
```

## Public API

| Symbol | What it is |
| --- | --- |
| `TvcacheToolExecutor` | Per-rollout executor: `lookup_or_execute`, `test`. |
| `ToolCall` | Base class for a tool invocation (the cache-key unit). |
| `TestToolCall` | Terminal sentinel call whose cached value is the test result. |
| `Sandbox` | Base class for a sandbox: `start`/`fork`/`execute`/`stop`. |
| `ForkBank` / `AbstractForkGenerator` | Pre-forked warm sandboxes; `ROOT_ENV_ID` is the warm-root key. |
| `SandboxManager` | Single per-server eviction + deletion loop. |
| `TVCacheClient` / `FindResult` | Low-level typed async client for direct server access. |
| `FindResponse`, `InsertResult`, `SandboxCandidate`, `ReclaimableSandbox`, `SandboxEvictionRecord`, `TVCacheServerError` | Protocol/response types. |
