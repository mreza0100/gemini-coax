"""gemini-coax — make Gemini structured output actually validate.

Gemini's ``response_json_schema`` enforces structure but silently ignores
``anyOf`` enums, numeric/length/array bounds, and degrades at the tail of long
arrays — so Pydantic rejects otherwise-good output. ``gemini-coax`` coaxes it
into shape.

Two layers:

* **Core** (this module + :mod:`gemini_coax.schema` / :mod:`gemini_coax.repair`)
  — pure functions over ``dict`` + Pydantic, no provider SDK. Use :func:`coax`
  with the raw ``google-genai`` SDK.
* **Adapter** (:mod:`gemini_coax.langchain`, optional ``[langchain]`` extra) —
  :class:`~gemini_coax.langchain.GeminiSafe`, a drop-in ``ChatGoogleGenerativeAI``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from .repair import repair_enums, salvage_list, salvage_lists
from .schema import clamp_to_constraints, fill_missing_nullables, strip_nullable_anyof

__version__ = "0.1.0"

__all__ = [
    "coax",
    "strip_nullable_anyof",
    "clamp_to_constraints",
    "fill_missing_nullables",
    "repair_enums",
    "salvage_list",
    "salvage_lists",
    "__version__",
]


def coax(raw: dict[str, Any], model: type[BaseModel]) -> BaseModel:
    """Coax a raw Gemini dict into a validated model instance.

    Runs the full framework-free pipeline: clamp out-of-range values, fill
    omitted nullables, then validate. If validation still fails, repair wrong
    enum values, then salvage broken list tails. Raises the original
    ``ValidationError`` only if nothing could be recovered.

    This is the one-call entry point for the raw ``google-genai`` SDK. LangChain
    users should use :class:`gemini_coax.langchain.GeminiSafe` instead, which
    applies the same pipeline transparently inside ``with_structured_output``.

    Args:
        raw: The decoded JSON dict Gemini returned.
        model: The Pydantic model you expected.

    Returns:
        A validated instance of ``model``.

    Raises:
        ValidationError: If the output could not be coaxed into the schema.
    """
    clamped = clamp_to_constraints(raw, model)
    clamped = fill_missing_nullables(clamped, model)
    try:
        return model.model_validate(clamped)
    except ValidationError as error:
        repaired = repair_enums(error, clamped, model)
        if repaired is not None:
            return repaired
        salvaged = salvage_lists(clamped, model)
        if salvaged is not None:
            return salvaged
        raise
