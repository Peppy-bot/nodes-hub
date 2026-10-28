"""The node's log line: one stdout line tagged `[recorder]`."""


def log(message: str, *, flush: bool = False) -> None:
    print(f"[recorder] {message}", flush=flush)
