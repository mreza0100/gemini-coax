"""LangChain adapter: a drop-in :class:`ChatGoogleGenerativeAI` that coaxes.

This module is optional. It is only importable when the ``langchain`` extra is
installed::

    pip install "gemini-coax[langchain]"

:class:`GeminiSafe` overrides ``with_structured_output`` to apply the framework-
free core (:mod:`gemini_coax.schema` + :mod:`gemini_coax.repair`) with zero
changes in your chain code, and wraps the async generation seam with a retry for
transient transport faults.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from .repair import salvage_lists
from .runaway import MAX_TOKENS_FINISH_REASONS, rehome_to_original, strip_runaway_strings
from .schema import clamp_to_constraints, fill_missing_nullables, strip_nullable_anyof

try:
    from langchain_core.runnables import Runnable, RunnableLambda
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    msg = (
        "gemini_coax.langchain requires the 'langchain' extra. "
        'Install it with: pip install "gemini-coax[langchain]"'
    )
    raise ImportError(msg) from exc

from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

if TYPE_CHECKING:
    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatResult

__all__ = ["GeminiSafe"]

_log = logging.getLogger("gemini_coax.langchain")


# ── Transient transport-error resilience ───────────────────────────────────
#
# Gemini/Vertex calls over aiohttp/httpx occasionally fail at the connection
# layer BEFORE any HTTP status is returned: ConnectionResetError(54), aiohttp
# ClientOSError / ServerDisconnectedError, TransferEncodingError. The google-genai
# SDK's HttpRetryOptions only retries HTTP *status codes* (408/429/5xx) — it never
# retries these transport faults. A single such hiccup propagates uncaught out of
# the chain. We retry the OSError family at the LLM call seam, below
# with_structured_output / bind_tools, so every chain inherits it. ConnectionError
# is a builtin; aiohttp's ClientOSError is an OSError but NOT a builtin
# ConnectionError, so we cover the whole OSError family. Output/validation errors
# (ValueError, ValidationError) are not transient and propagate immediately.
_TRANSIENT_TRANSPORT_EXC: tuple[type[BaseException], ...] = (ConnectionError, OSError)

# Total attempts including the original call. 3 = 1 original + 2 retries.
_TRANSIENT_RETRY_ATTEMPTS = 3


# ── Structured-output coaxing + MAX_TOKENS runaway recovery helpers ─────────
#
# These are module-level (not methods) so they stay pure and testable. The
# primary structured call always runs with include_raw=True internally so the
# finish_reason is visible; _shape_output then honours the caller's public
# include_raw choice.


def _coax_dict(raw_dict: Any, model: type[BaseModel]) -> Any:
    """clamp → fill → validate → salvage. Returns a model instance or raises."""
    clamped = clamp_to_constraints(raw_dict, model)
    clamped = fill_missing_nullables(clamped, model)
    try:
        return model.model_validate(clamped)
    except ValidationError:
        salvaged = salvage_lists(clamped, model)
        if salvaged is not None:
            return salvaged
        raise


def _coax_envelope_to_model(envelope: Any, model: type[BaseModel]) -> BaseModel | None:
    """Pull the parsed dict out of an include_raw envelope and coax it to ``model``.

    LENIENT variant — returns ``None`` (rather than raising) when nothing usable
    could be produced. Reserved for the recovery attempt only, so a failed recovery
    falls through to the strict primary result. The primary result must NEVER use
    this: a total parse failure has to surface (raise / error-envelope), not vanish
    into ``None`` — see :func:`_shape_primary`.
    """
    parsed = envelope.get("parsed") if isinstance(envelope, dict) else envelope
    if isinstance(parsed, BaseModel):
        return parsed
    if not isinstance(parsed, dict):
        return None
    try:
        result = _coax_dict(parsed, model)
    except ValidationError:
        return None
    return result if isinstance(result, BaseModel) else None


def _shape_primary(envelope: Any, model: type[BaseModel], *, include_raw: bool) -> Any:
    """STRICT primary shaper — coax the envelope's parsed dict, surfacing total failure.

    Unlike the lenient :func:`_coax_envelope_to_model`, this enforces the v0.1.1
    total-parse-failure contract on the PRIMARY result:

    * ``include_raw=False`` — on total failure (clamp→fill→validate→salvage all
      fail) the underlying :class:`ValidationError` propagates. A fully-malformed
      LLM output fails loud, never silently becomes ``None``.
    * ``include_raw=True`` — on total failure return the error envelope:
      ``parsed=None``, ``parsing_error=<the ValidationError>``,
      ``parsed_dict=<the clamped dict that failed>``. On success, ``parsed`` is the
      validated model and ``parsing_error`` is ``None`` (set by :func:`_shape_output`).

    An already-validated model (recovery rehome, or a parsed BaseModel) and a
    non-dict parsed value pass straight through to :func:`_shape_output`.
    """
    parsed = envelope.get("parsed") if isinstance(envelope, dict) else envelope
    if not isinstance(parsed, dict):
        return _shape_output(envelope, parsed, include_raw=include_raw)
    clamped = clamp_to_constraints(parsed, model)
    clamped = fill_missing_nullables(clamped, model)
    try:
        coaxed = _coax_dict(parsed, model)
    except ValidationError as exc:
        if not include_raw:
            raise
        if isinstance(envelope, dict):
            out = dict(envelope)
            out["parsed"] = None
            out["parsing_error"] = exc
            out["parsed_dict"] = clamped
            return out
        return None
    return _shape_output(envelope, coaxed, include_raw=include_raw)


def _finish_reason(envelope: Any) -> str | None:
    raw_msg = envelope.get("raw") if isinstance(envelope, dict) else None
    md = getattr(raw_msg, "response_metadata", {}) or {}
    fr = md.get("finish_reason") or md.get("finishReason")
    return str(fr).upper() if fr else None


def _is_truncated(envelope: Any) -> bool:
    return _finish_reason(envelope) in MAX_TOKENS_FINISH_REASONS


def _shape_output(envelope: Any, parsed: Any, *, include_raw: bool) -> Any:
    """Return the public shape: the bare parsed model, or the {raw,parsed,...} envelope."""
    if not include_raw:
        return parsed
    if isinstance(envelope, dict):
        out = dict(envelope)
        out["parsed"] = parsed
        if parsed is not None:
            out["parsing_error"] = None
        return out
    return parsed


def _run_sync(coro: Any) -> Any:
    """Drive an async coroutine from the sync .invoke() path.

    Mirrors langchain's own sync-over-async bridging: if no loop is running,
    asyncio.run drives it; if one is already running, defer to a fresh loop in a
    worker thread (the structured call is network-bound, so the thread hop is free).
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class GeminiSafe(ChatGoogleGenerativeAI):
    """``ChatGoogleGenerativeAI`` that makes structured output trustworthy.

    Overrides ``with_structured_output`` to:

    1. Strip nullable ``anyOf`` patterns from the schema sent to Gemini.
    2. Clamp raw output values to field constraints before Pydantic validation.
    3. Fill omitted nullable fields with ``None``.
    4. Salvage valid list entries when the tail of an array is broken.
    5. Re-validate with the original Pydantic model.

    Also retries transient transport faults at the async generation seam. Drop it
    in wherever you build ``ChatGoogleGenerativeAI`` — no other code changes.
    """

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Retry transient transport faults with exponential backoff + jitter.

        Wraps the SDK's async generation at the single seam every consumption path
        funnels through — ``.ainvoke()``, ``with_structured_output()``, and
        ``bind_tools()`` all reach the model via ``_agenerate``. Streaming
        (``.astream()``) uses ``_astream`` and is intentionally not covered here
        (interactive surfaces re-issue on a dropped stream).
        """
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(_TRANSIENT_TRANSPORT_EXC),
            wait=wait_exponential_jitter(),
            stop=stop_after_attempt(_TRANSIENT_RETRY_ATTEMPTS),
            reraise=True,
        ):
            with attempt:
                if attempt.retry_state.attempt_number > 1:
                    _log.warning(
                        "llm_transient_retry attempt=%d",
                        attempt.retry_state.attempt_number,
                    )
                return await super()._agenerate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
        msg = "unreachable: AsyncRetrying with reraise exited without result"
        raise RuntimeError(msg)

    def with_structured_output(
        self,
        schema: dict[str, Any] | type[BaseModel],
        method: Literal["function_calling", "json_mode", "json_schema"] | None = "json_schema",
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:  # type: ignore[type-arg]
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            return super().with_structured_output(
                schema, method=method, include_raw=include_raw, **kwargs
            )

        original_model = schema
        fixed_schema = strip_nullable_anyof(schema.model_json_schema())

        # The primary call always requests include_raw so the validator can read
        # finish_reason and detect a MAX_TOKENS string runaway (see _recover). The
        # public include_raw contract is honoured by re-wrapping the envelope below.
        base_raw: Runnable = super().with_structured_output(  # type: ignore[type-arg]
            fixed_schema, method=method, include_raw=True, **kwargs
        )

        # Recovery twin — runaway-prone free-string fields stripped. Built once and
        # only when the model actually HAS such a field; otherwise no retry is wired.
        recovery_model, stripped = strip_runaway_strings(original_model)
        recovery_raw: Runnable | None = None  # type: ignore[type-arg]
        if stripped:
            rec_schema = strip_nullable_anyof(recovery_model.model_json_schema())
            recovery_raw = super().with_structured_output(
                rec_schema, method=method, include_raw=True, **kwargs
            )

        async def _ainvoke(inputs: Any) -> Any:
            result = await base_raw.ainvoke(inputs)
            if recovery_raw is not None and _is_truncated(result):
                _log.warning(
                    "gemini_max_tokens_runaway_recovery finish_reason=%s stripped=%s",
                    _finish_reason(result),
                    stripped,
                )
                rec = await recovery_raw.ainvoke(inputs)
                recovered = _coax_envelope_to_model(rec, recovery_model)
                if recovered is not None:
                    rehomed = rehome_to_original(recovered, original_model, stripped)
                    return _shape_output(rec, rehomed, include_raw=include_raw)
            return _shape_primary(result, original_model, include_raw=include_raw)

        def _invoke(inputs: Any) -> Any:
            return _run_sync(_ainvoke(inputs))

        return RunnableLambda(func=_invoke, afunc=_ainvoke)
