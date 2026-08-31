import os
import sys
import time
import asyncio
import functools
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager

try:
    from langfuse import Langfuse, observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    Langfuse = None
    observe = None
    LANGFUSE_AVAILABLE = False


def _clean_host_url(host: Optional[str]) -> Optional[str]:
    """Ensures host is a valid base URL without extra project paths."""
    if not host:
        return None
    # If a full dashboard URL was pasted like https://us.cloud.langfuse.com/project/...
    if "/project/" in host:
        host = host.split("/project/")[0]
    return host.rstrip("/")


class TracingService:
    """
    Observability & Tracing layer for OmniBrain using Langfuse.
    Captures supervisor routing decisions, agent node execution, LLM token usage, and latency.
    Guaranteed fail-safe: All tracing operations degrade gracefully without disrupting user requests.
    """
    def __init__(self):
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        raw_host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
        self.host = _clean_host_url(raw_host)

        self.enabled = bool(
            LANGFUSE_AVAILABLE
            and self.public_key
            and self.secret_key
            and not self.public_key.startswith("pk-lf-placeholder")
        )

        self.client: Optional[Langfuse] = None
        if self.enabled:
            try:
                self.client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host
                )
            except Exception as e:
                print(f"[TracingService] Failed to initialize Langfuse client: {e}")
                self.client = None
                self.enabled = False

    def is_enabled(self) -> bool:
        return self.enabled

    def flush(self):
        """Flushes queued events to Langfuse."""
        if self.client:
            try:
                self.client.flush()
            except Exception:
                pass

    def observe(self, *args, **kwargs):
        """
        Resilient wrapper around langfuse.observe decorator.
        If Langfuse is disabled, functions execute as normal passthroughs without overhead.
        """
        if LANGFUSE_AVAILABLE and observe is not None and self.enabled:
            try:
                return observe(*args, **kwargs)
            except Exception:
                pass

        # Fallback dummy decorator
        def decorator(func: Callable):
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*a, **kw):
                    return await func(*a, **kw)
                return async_wrapper
            else:
                @functools.wraps(func)
                def wrapper(*a, **kw):
                    return func(*a, **kw)
                return wrapper

        if len(args) == 1 and callable(args[0]) and not kwargs:
            return decorator(args[0])
        return decorator

    @contextmanager
    def trace_span(
        self,
        name: str,
        input_data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Resilient context manager for tracing node and tool execution spans.
        Captures start, end, and duration gracefully.
        """
        start_time = time.time()
        span_data = {
            "name": name,
            "input": input_data,
            "metadata": metadata or {},
            "start_time": start_time,
            "output": None,
            "duration_seconds": 0.0
        }
        try:
            yield span_data
        except Exception as e:
            span_data["error"] = str(e)
            raise
        finally:
            elapsed = time.time() - start_time
            span_data["duration_seconds"] = round(elapsed, 4)


# Global Tracing Service instance
tracing_service = TracingService()
