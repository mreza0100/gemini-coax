"""Tests for the GeminiSafe.with_structured_output total-parse-failure contract.

These exercise the adapter exactly like the consuming app does: patch
``ChatGoogleGenerativeAI.with_structured_output`` to return a RunnableLambda that
yields the internal include_raw envelope ``{"parsed": <dict>, "raw": <msg>}``, then
drive ``GeminiSafe.with_structured_output(...)`` and assert the public contract.

The regression these guard: v0.2.0's runaway-recovery refactor routed the PRIMARY
result through a lenient None-returning coax, so a fully-malformed LLM output
silently became ``None`` (non-raw) or a ``{"parsed": None}`` envelope with no
``parsing_error`` (raw) — instead of raising / returning the error envelope as
v0.1.1 did. That is a clinical-safety regression in the consumer (a total parse
failure must fail loud), and it had ZERO library coverage.

The model mirrors the consumer's salvage shape: an item with required
``name: str`` + ``score: float`` inside ``output.items: list[item]``. A ``{}`` item
is therefore "broken" (both required fields missing). ``name`` is an unbounded
string, so ``strip_runaway_strings`` arms a recovery twin — but the mock's ``raw``
carries no MAX_TOKENS finish_reason, so recovery never fires and the primary strict
path is what gets exercised.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ValidationError

from gemini_coax.langchain import GeminiSafe


class _SalvageItem(BaseModel):
    name: str  # required, unbounded str → runaway-prone (arms recovery twin)
    score: float  # required


class _SalvageOutput(BaseModel):
    items: list[_SalvageItem]


class _FakeRaw:
    """Stand-in for the LangChain message in the include_raw envelope's ``raw`` slot.

    Carries an empty ``response_metadata`` so ``_finish_reason`` reads no
    MAX_TOKENS finish_reason — recovery never fires and the primary strict path runs.
    """

    response_metadata: dict[str, Any] = {}


def _patched_safe(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> GeminiSafe:
    """A GeminiSafe whose base structured runnable yields a fixed include_raw envelope.

    Patches the *parent* ``ChatGoogleGenerativeAI.with_structured_output`` (the
    ``super()`` call inside ``GeminiSafe.with_structured_output``) to return a
    RunnableLambda that ignores its input and emits ``{"parsed": payload, "raw": ...}``
    — the v0.2.0 internal always-include_raw envelope.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    def _fake_wso(
        self: Any, schema: Any, *args: Any, **kwargs: Any
    ) -> RunnableLambda:
        envelope = {"parsed": dict(payload), "raw": _FakeRaw()}
        return RunnableLambda(lambda _inputs: dict(envelope))

    monkeypatch.setattr(
        ChatGoogleGenerativeAI, "with_structured_output", _fake_wso, raising=True
    )
    return GeminiSafe(model="gemini-2.0-flash", api_key="test-key")


# ── 1 & 2: CLEAN parse ──────────────────────────────────────────────────────


def test_clean_parse_non_raw_returns_bare_model(monkeypatch: pytest.MonkeyPatch) -> None:
    safe = _patched_safe(monkeypatch, {"items": [{"name": "a", "score": 1.0}]})
    runnable = safe.with_structured_output(_SalvageOutput)
    out = runnable.invoke({})
    assert isinstance(out, _SalvageOutput)
    assert [i.name for i in out.items] == ["a"]


def test_clean_parse_raw_sets_parsing_error_none(monkeypatch: pytest.MonkeyPatch) -> None:
    safe = _patched_safe(monkeypatch, {"items": [{"name": "a", "score": 1.0}]})
    runnable = safe.with_structured_output(_SalvageOutput, include_raw=True)
    out = runnable.invoke({})
    assert isinstance(out, dict)
    assert isinstance(out["parsed"], _SalvageOutput)
    assert out["parsing_error"] is None
    assert "raw" in out
    assert "parsed_dict" not in out


# ── 3 & 4: SALVAGEABLE (broken tail, some rows survive) ─────────────────────


def test_salvageable_non_raw_returns_model_with_survivors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"items": [{"name": "a", "score": 1.0}, {"name": "b", "score": 2.0}, {}]}
    safe = _patched_safe(monkeypatch, payload)
    runnable = safe.with_structured_output(_SalvageOutput)
    out = runnable.invoke({})
    assert isinstance(out, _SalvageOutput)
    assert [i.name for i in out.items] == ["a", "b"]  # trailing {} dropped


def test_salvageable_raw_has_no_parsing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"items": [{"name": "a", "score": 1.0}, {"name": "b", "score": 2.0}, {}]}
    safe = _patched_safe(monkeypatch, payload)
    runnable = safe.with_structured_output(_SalvageOutput, include_raw=True)
    out = runnable.invoke({})
    assert isinstance(out, dict)
    assert isinstance(out["parsed"], _SalvageOutput)
    assert len(out["parsed"].items) == 2
    assert out["parsing_error"] is None
    assert "parsed_dict" not in out  # success path never sets the failed-dict field


# ── 5 & 6: TOTAL failure (all rows broken / required-nonnull missing) ───────


def test_total_failure_non_raw_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every item missing both required fields → nothing salvageable → must RAISE.
    payload = {"items": [{}, {}]}
    safe = _patched_safe(monkeypatch, payload)
    runnable = safe.with_structured_output(_SalvageOutput)
    with pytest.raises(ValidationError):
        runnable.invoke({})


def test_total_failure_raw_returns_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"items": [{}, {}]}
    safe = _patched_safe(monkeypatch, payload)
    runnable = safe.with_structured_output(_SalvageOutput, include_raw=True)
    out = runnable.invoke({})
    assert isinstance(out, dict)
    assert out["parsed"] is None
    assert out["parsing_error"] is not None
    assert isinstance(out["parsing_error"], ValidationError)
    assert "parsed_dict" in out
    assert isinstance(out["parsed_dict"], dict)
    assert "raw" in out
