"""Best-effort telemetry and cleanup that never hide the primary training failure."""

import gc
from collections.abc import Callable


def attempt(label: str, action: Callable, errors: dict) -> object | None:
    try:
        return action()
    except BaseException as error:
        errors[label] = repr(error)
        return None


def gpu_statistics(torch, errors: dict) -> dict:
    """Read each statistic independently, before cleanup resets device state."""
    result = {}
    attempt("synchronize", lambda: torch.cuda.synchronize(), errors)
    queries = {
        "peak_allocated_gib": lambda: torch.cuda.max_memory_allocated(),
        "peak_reserved_gib": lambda: torch.cuda.max_memory_reserved(),
        "gpu_total_gib": lambda: torch.cuda.get_device_properties(0).total_memory,
        "gpu_free_gib": lambda: torch.cuda.mem_get_info()[0],
    }
    for name, query in queries.items():
        value = attempt(name, query, errors)
        if value is not None:
            result[name] = value / 2**30
    return result


def release_job(job, torch, errors: dict) -> None:
    if job is not None:
        attempt("job.cleanup", job.cleanup, errors)
    # BaseJob.cleanup in the locked upstream only deletes its local `self`.
    # Drop its process references too; the isolated worker exits after reporting.
    if job is not None and isinstance(getattr(job, "process", None), list):
        attempt("process.clear", job.process.clear, errors)
    attempt("gc.collect", gc.collect, errors)
    if torch is not None:
        attempt("empty_cache", torch.cuda.empty_cache, errors)
