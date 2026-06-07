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
        base: Runnable = super().with_structured_output(  # type: ignore[type-arg]
            fixed_schema, method=method, include_raw=include_raw, **kwargs
        )

        if not include_raw:

            def _validate(d: Any) -> Any:
                if isinstance(d, dict):
                    clamped = clamp_to_constraints(d, original_model)
                    clamped = fill_missing_nullables(clamped, original_model)
                    try:
                        return original_model.model_validate(clamped)
                    except ValidationError:
                        salvaged = salvage_lists(clamped, original_model)
                        if salvaged is not None:
                            return salvaged
                        raise
                return d

            return base | RunnableLambda(_validate)

        def _validate_raw(result: Any) -> Any:
            if not isinstance(result, dict):
                return result
            parsed = result.get("parsed")
            if isinstance(parsed, dict):
                try:
                    clamped = clamp_to_constraints(parsed, original_model)
                    clamped = fill_missing_nullables(clamped, original_model)
                    result["parsed"] = original_model.model_validate(clamped)
                except Exception as exc:  # noqa: BLE001
                    salvaged = salvage_lists(parsed, original_model)
                    if salvaged is not None:
                        result["parsed"] = salvaged
                    else:
                        result["parsing_error"] = exc
                        result["parsed"] = None
                        result["parsed_dict"] = parsed
            return result

        return base | RunnableLambda(_validate_raw)
