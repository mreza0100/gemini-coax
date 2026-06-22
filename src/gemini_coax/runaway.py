"""Detect and recover Gemini's MAX_TOKENS string-runaway failure class.

Gemini's constrained decoder, while generating an UNBOUNDED free-string field (a
``str`` with no enum/``Literal`` and a ``maxLength`` it ignores), can fall into a
degenerate repetition loop — emitting the same phrase thousands of times until it
exhausts ``max_output_tokens`` and returns ``finish_reason == "MAX_TOKENS"``. The
JSON is then truncated mid-array and every later list entry is lost;
:func:`gemini_coax.salvage_lists` recovers only the entries that completed before
the runaway, so an N-item extraction silently collapses to one.

This module is the framework-free core of the fix (dict + Pydantic, no provider
SDK). The langchain adapter wires the detect→retry loop using it.
"""

from __future__ import annotations

import types
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, create_model

__all__ = [
    "MAX_TOKENS_FINISH_REASONS",
    "rehome_to_original",
    "runaway_prone_string_fields",
    "strip_runaway_strings",
]

#: Truncation finish-reasons across the Vertex / AI-Studio surfaces.
MAX_TOKENS_FINISH_REASONS: frozenset[str] = frozenset({"MAX_TOKENS", "MAXTOKENS", "LENGTH"})

#: A ``max_length`` at or below this many characters bounds the loop below the
#: token budget — the field closes long before the budget is exhausted, so it is
#: not runaway-prone. A larger or absent cap is the one Gemini ignores.
_SAFE_MAX_LENGTH = 64


def _unwrap_optional(annotation: Any) -> Any:
    """``X | None`` / ``Optional[X]`` → ``X`` (other annotations unchanged)."""
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _is_constrained_string(field_info: Any, inner: Any) -> bool:
    """Whether a string field carries a bound that prevents a runaway.

    Constrained (KEEP on recovery) when the field is a ``Literal`` enum, carries a
    regex ``pattern``, or has a tight ``max_length`` (≤ :data:`_SAFE_MAX_LENGTH`).
    """
    if get_origin(inner) is Literal:
        return True
    for meta in getattr(field_info, "metadata", ()):
        if getattr(meta, "pattern", None) is not None:
            return True
        max_length = getattr(meta, "max_length", None)
        if isinstance(max_length, int) and max_length <= _SAFE_MAX_LENGTH:
            return True
    return False


def runaway_prone_string_fields(model: type[BaseModel]) -> list[str]:
    """Names of ``model`` fields that can trigger an unbounded string runaway.

    A field qualifies when its optional-unwrapped type is a bare ``str`` with no
    enum/``Literal``, no ``pattern``, and no tight ``max_length`` — the only shape
    a constrained decoder can loop on indefinitely.
    """
    out: list[str] = []
    for name, field_info in model.model_fields.items():
        inner = _unwrap_optional(field_info.annotation)
        if inner is str and not _is_constrained_string(field_info, inner):
            out.append(name)
    return out


def _strip_from_item_model(item_model: type[BaseModel]) -> tuple[type[BaseModel], list[str]]:
    runaway = runaway_prone_string_fields(item_model)
    if not runaway:
        return item_model, []
    keep: dict[str, Any] = {
        name: (info.annotation, info)
        for name, info in item_model.model_fields.items()
        if name not in runaway
    }
    trimmed = create_model(f"{item_model.__name__}__norunaway", **keep)
    return trimmed, runaway


def strip_runaway_strings(
    model: type[BaseModel],
) -> tuple[type[BaseModel], dict[str, list[str]]]:
    """Build a recovery twin of ``model`` with runaway-prone string fields removed.

    Strips qualifying string fields on the top-level model and inside every
    ``list[BaseModel]`` field's item model (the common extraction shape, where the
    runaway lives on a per-row string). Returns ``(trimmed_model, stripped_map)``
    where ``stripped_map`` is ``{location: [field names]}`` for logging. When
    nothing qualifies, returns ``(model, {})`` — the caller should then NOT retry.
    """
    stripped: dict[str, list[str]] = {}
    new_fields: dict[str, Any] = {}

    top_runaway = runaway_prone_string_fields(model)
    if top_runaway:
        stripped["<root>"] = top_runaway

    for name, field_info in model.model_fields.items():
        if name in top_runaway:
            continue
        inner = _unwrap_optional(field_info.annotation)
        if get_origin(inner) is list:
            args = get_args(inner)
            if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                trimmed_item, item_runaway = _strip_from_item_model(args[0])
                if item_runaway:
                    stripped[name] = item_runaway
                    new_fields[name] = (list[trimmed_item], field_info)  # type: ignore[valid-type]
                    continue
        new_fields[name] = (field_info.annotation, field_info)

    if not stripped:
        return model, {}

    trimmed_model = create_model(f"{model.__name__}__norunaway", **new_fields)
    return trimmed_model, stripped


def _placeholder_for(field_info: Any) -> Any:
    """The value a stripped field takes when re-homed onto the original model.

    Order of preference: the field's own default (or default_factory) → ``None``
    when the field is nullable → ``""`` for a bare string. The runaway already
    destroyed this field's real value, so a placeholder is strictly better than
    losing the whole record; ``""`` is the honest "could not extract" marker for a
    required, non-nullable string.
    """
    from pydantic_core import PydanticUndefined

    if field_info.default is not PydanticUndefined:
        return field_info.default
    if getattr(field_info, "default_factory", None) is not None:
        return field_info.default_factory()
    if type(None) in get_args(field_info.annotation):
        return None
    if _unwrap_optional(field_info.annotation) is str:
        return ""
    return None


def rehome_to_original(
    recovered: BaseModel,
    original_model: type[BaseModel],
    stripped: dict[str, list[str]],
) -> BaseModel:
    """Re-validate a recovery-model instance against the original strict model.

    The recovery model omits the runaway-prone string fields, so a plain
    ``original_model.model_validate(recovered.model_dump())`` fails on any stripped
    field that is required-and-non-nullable (e.g. ``role: str``). This injects a
    safe placeholder (see :func:`_placeholder_for`) for every stripped field at the
    root and inside each stripped list-item, then validates. Returns a fully-typed
    instance of ``original_model``.
    """
    data = recovered.model_dump()

    root_stripped = stripped.get("<root>", [])
    for fname in root_stripped:
        data[fname] = _placeholder_for(original_model.model_fields[fname])

    for list_field, item_fields in stripped.items():
        if list_field == "<root>":
            continue
        field_info = original_model.model_fields.get(list_field)
        if field_info is None:
            continue
        item_model = None
        inner = _unwrap_optional(field_info.annotation)
        if get_origin(inner) is list:
            args = get_args(inner)
            if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                item_model = args[0]
        if item_model is None:
            continue
        rows = data.get(list_field)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            for fname in item_fields:
                row[fname] = _placeholder_for(item_model.model_fields[fname])

    return original_model.model_validate(data)
