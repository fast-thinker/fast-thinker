from __future__ import annotations

import os
import signal
from contextlib import contextmanager
from typing import Iterator


_ACTIVE_DEPTH = 0
_INTERRUPTED = False
_LAST_INTERRUPTED = False


def was_interrupted() -> bool:
    return _ACTIVE_DEPTH > 0 and _INTERRUPTED


def interrupt_was_requested() -> bool:
    return _INTERRUPTED or _LAST_INTERRUPTED


@contextmanager
def interruptible_process(label: str) -> Iterator[None]:
    global _ACTIVE_DEPTH, _INTERRUPTED, _LAST_INTERRUPTED
    previous: dict[signal.Signals, object] = {}
    _ACTIVE_DEPTH += 1
    _INTERRUPTED = False
    _LAST_INTERRUPTED = False

    def _signal_name(signum) -> str:
        try:
            return signal.Signals(signum).name
        except ValueError:
            return str(signum)

    def force_exit(signum, _frame) -> None:
        name = _signal_name(signum)
        print(f"[{label}] {name} received again; forcing exit", flush=True)
        os._exit(128 + int(signum))

    def interrupt(signum, _frame) -> None:
        global _INTERRUPTED, _LAST_INTERRUPTED
        _INTERRUPTED = True
        _LAST_INTERRUPTED = True
        name = _signal_name(signum)
        for next_signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(next_signum, force_exit)
        print(f"[{label}] {name} received; shutting down", flush=True)
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupt)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        _ACTIVE_DEPTH -= 1


__all__ = [
    "interrupt_was_requested",
    "interruptible_process",
    "was_interrupted",
]
