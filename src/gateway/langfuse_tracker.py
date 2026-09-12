"""
Langfuse v4 Observability & Tracing Manager for Multi-Agent Support Architecture.
Follows official Langfuse AI Skill best practices:
1. Typed observations (guardrail, retriever, agent, generation) for Agent Graphs.
2. Context propagation (session_id, user_id, tags) via `propagate_attributes`.
3. Native LiteLLM OpenTelemetry integration via `litellm.callbacks = ["langfuse_otel"]`.
4. Graceful degradation: zero overhead and no exceptions if credentials are unconfigured.
"""

import os
import functools
import logging
import contextvars
from typing import Optional, List, Dict, Any, Callable
from dotenv import load_dotenv

# Ensure environment is loaded before Langfuse initializes
load_dotenv()

logger = logging.getLogger("langfuse_tracker")

_LANGFUSE_CLIENT = None
_IS_INITIALIZED = False
_IS_ENABLED = False

_tracing_suppressed: contextvars.ContextVar[bool] = contextvars.ContextVar("tracing_suppressed", default=False)


def is_tracing_suppressed() -> bool:
    """Returns whether Langfuse tracing is currently suppressed."""
    return _tracing_suppressed.get()


class SuppressTracing:
    """
    Context manager to completely suppress Langfuse trace creation.
    Used for out-of-domain / off-topic queries to prevent trace pollution.
    """
    def __init__(self):
        self.token = None

    def __enter__(self):
        self.token = _tracing_suppressed.set(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token is not None:
            _tracing_suppressed.reset(self.token)


def is_langfuse_configured() -> bool:
    """Checks if valid Langfuse public and secret keys exist in environment."""
    pub = os.environ.get("LANGFUSE_PUBLIC_KEY")
    sec = os.environ.get("LANGFUSE_SECRET_KEY")
    # Disregard empty placeholders
    return bool(pub and sec and not pub.startswith("your_") and not sec.startswith("your_"))


def get_langfuse_client():
    """
    Returns the singleton Langfuse v4 client instance if configured, else None.
    Thread-safe and fail-safe.
    """
    global _LANGFUSE_CLIENT, _IS_INITIALIZED, _IS_ENABLED

    if _IS_INITIALIZED:
        return _LANGFUSE_CLIENT

    _IS_INITIALIZED = True

    if not is_langfuse_configured():
        logger.info("[LangfuseTracker] Langfuse credentials not configured; tracing is disabled.")
        _IS_ENABLED = False
        return None

    try:
        from langfuse import Langfuse
        
        host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_OTEL_HOST") or "https://cloud.langfuse.com"
        _LANGFUSE_CLIENT = Langfuse(
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
            host=host,
        )
        _IS_ENABLED = True
        logger.info(f"[LangfuseTracker] Langfuse v4 successfully initialized (Host: {host}).")
        return _LANGFUSE_CLIENT
    except Exception as e:
        logger.warning(f"[LangfuseTracker] Failed to initialize Langfuse client: {e}. Running without tracing.")
        _IS_ENABLED = False
        return None


def init_litellm_langfuse():
    """
    Configures LiteLLM's OpenTelemetry callback for Langfuse generation spans.
    Per Langfuse documentation, setting `litellm.callbacks = ['langfuse_otel']`
    automatically routes all prompt, completion, token, and cost metrics into Langfuse.
    """
    if not is_langfuse_configured():
        return

    try:
        import litellm
        
        # Ensure litellm knows the host and credentials
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        os.environ.setdefault("LANGFUSE_OTEL_HOST", host)

        callbacks = getattr(litellm, "callbacks", []) or []
        if "langfuse_otel" not in callbacks:
            callbacks.append("langfuse_otel")
            litellm.callbacks = callbacks
            logger.info("[LangfuseTracker] LiteLLM langfuse_otel callback registered.")
    except Exception as e:
        logger.warning(f"[LangfuseTracker] Could not register LiteLLM langfuse callback: {e}")


def flush_traces():
    """Flushes buffered traces to Langfuse Cloud. Safe to call anytime."""
    client = get_langfuse_client()
    if client:
        try:
            client.flush()
        except Exception as e:
            logger.warning(f"[LangfuseTracker] Error during trace flush: {e}")


def get_active_trace_url() -> Optional[str]:
    """Retrieves web URL for current active trace in Langfuse dashboard."""
    if is_tracing_suppressed():
        return None
    client = get_langfuse_client()
    if client:
        try:
            return client.get_trace_url()
        except Exception:
            host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
            tid = get_active_trace_id()
            if tid:
                return f"{host.rstrip('/')}/traces/{tid}"
            return None
    return None


def get_active_trace_id() -> Optional[str]:
    """Retrieves the ID for the currently executing trace."""
    if is_tracing_suppressed():
        return None
    client = get_langfuse_client()
    if client:
        try:
            return client.get_current_trace_id()
        except Exception:
            return None
    return None


def langfuse_observe(
    name: Optional[str] = None,
    as_type: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
):
    """
    Decorator for instrumenting functions and methods with Langfuse observations.
    Gracefully no-ops if Langfuse is not enabled or if tracing is suppressed.
    
    Supported observation types:
    - "agent": for multi-agent reasoning, subagents, or synthesis
    - "retriever": for vector database / ChromaDB / RAG searches
    - "guardrail": for security prompt-injection or HITL checks
    - "generation": for LLM generations
    - "span": generic operational step
    """
    def decorator(func: Callable):
        if not is_langfuse_configured():
            return func

        try:
            from langfuse import observe
            obs_kwargs: Dict[str, Any] = {
                "capture_input": capture_input,
                "capture_output": capture_output,
            }
            if name:
                obs_kwargs["name"] = name
            if as_type:
                obs_kwargs["as_type"] = as_type

            decorated = observe(**obs_kwargs)(func)

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if is_tracing_suppressed():
                    return func(*args, **kwargs)
                return decorated(*args, **kwargs)

            return wrapper
        except Exception as e:
            logger.warning(f"[LangfuseTracker] Failed to decorate {func.__name__}: {e}")
            return func

    return decorator


class TraceContext:
    """
    Context manager for setting trace-level attributes (session_id, user_id, tags)
    using Langfuse v4 `propagate_attributes`.
    """
    def __init__(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        trace_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.tags = tags or []
        self.trace_name = trace_name
        self.metadata = metadata or {}
        self._ctx = None

    def __enter__(self):
        if is_tracing_suppressed():
            return self
        if is_langfuse_configured():
            try:
                from langfuse import propagate_attributes
                # Ensure metadata values are strings as required by Langfuse v4
                sanitized_meta = {k: str(v)[:200] for k, v in self.metadata.items()} if self.metadata else {}
                self._ctx = propagate_attributes(
                    session_id=self.session_id,
                    user_id=self.user_id,
                    tags=self.tags,
                    trace_name=self.trace_name,
                    metadata=sanitized_meta,
                )
                self._ctx.__enter__()
            except Exception as e:
                logger.debug(f"[TraceContext] propagate_attributes error: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._ctx:
            try:
                self._ctx.__exit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass


# Pre-initialize LiteLLM integration on module import
init_litellm_langfuse()
