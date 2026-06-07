"""Repair Gemini's degraded structured output before giving up.

Gemini's FSM-based constrained decoding degrades at the *tail* of long arrays,
producing two failure shapes that make Pydantic reject an otherwise-good output:

1. **Close-but-wrong enum values** (e.g. ``"defensiveness"`` instead of
   ``"defensiveness-tone"``). :func:`repair_enums` fuzzy-matches them back.
2. **Empty ``{}`` or half-formed objects** at the end of a list when the output
   token budget runs out. :func:`salvage_lists` keeps the valid entries and
   drops the broken ones.

A full-chain retry is 100–300x more expensive than a targeted repair, and often
makes things worse. These functions are pure — they take a raw ``dict``, a
``ValidationError`` (for enums), and a Pydantic model, and return a validated
instance or ``None``. No LangChain, no provider SDK.
"""

from __future__ import annotations

import copy
import logging
from difflib import get_close_matches
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

__all__ = [
    "repair_enums",
    "salvage_list",
    "salvage_lists",
]

_log = logging.getLogger("gemini_coax.repair")

_FUZZY_CUTOFF = 0.6


def _extract_literal_values(annotation: Any) -> list[str] | None:
    """Extract string values from a ``Literal`` type annotation."""
    origin = get_origin(annotation)

    if origin is Literal:
        args = get_args(annotation)
        if args and all(isinstance(a, str) for a in args):
            return list(args)
        return None

    if origin is Union:
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            result = _extract_literal_values(arg)
            if result is not None:
                return result

    return None


def _collect_enum_fields(model: type[BaseModel]) -> dict[str, list[str]]:
    """Map field names to their valid ``Literal`` string values for a model."""
    enums: dict[str, list[str]] = {}
    for field_name, field_info in model.model_fields.items():
        values = _extract_literal_values(field_info.annotation)
        if values:
            enums[field_name] = values
    return enums


def _is_enum_error(error: Any) -> bool:
    """Check whether a Pydantic validation error is a literal/enum mismatch."""
    if isinstance(error, dict):
        return error.get("type") == "literal_error"
    return getattr(error, "type", None) == "literal_error"


def _fuzzy_repair_value(raw_value: str, valid_values: list[str]) -> str | None:
    """Fuzzy-match a wrong enum value to the closest valid one, or ``None``."""
    matches = get_close_matches(raw_value, valid_values, n=1, cutoff=_FUZZY_CUTOFF)
    return matches[0] if matches else None


def _patch_nested_dict(
    data: dict[str, Any],
    path: tuple[str | int, ...],
    value: Any,
) -> None:
    """Set a value at a nested path in a dict/list structure."""
    obj: Any = data
    for key in path[:-1]:
        if (isinstance(obj, dict) and isinstance(key, str)) or (
            isinstance(obj, list) and isinstance(key, int)
        ):
            obj = obj[key]
        else:
            return
    final_key = path[-1]
    if (isinstance(obj, dict) and isinstance(final_key, str)) or (
        isinstance(obj, list) and isinstance(final_key, int)
    ):
        obj[final_key] = value


def _resolve_model_for_field(
    model: type[BaseModel],
    path: tuple[str | int, ...],
    field_name: str,
) -> list[str] | None:
    """Walk the model hierarchy to find valid ``Literal`` values for a nested field."""
    current_model = model
    for part in path:
        if isinstance(part, int):
            continue
        field_info = current_model.model_fields.get(part)
        if field_info is None:
            return None
        annotation = field_info.annotation
        origin = get_origin(annotation)
        if origin is list:
            args = get_args(annotation)
            if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                current_model = args[0]
                continue
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            current_model = annotation
            continue

    target_field = current_model.model_fields.get(field_name)
    if target_field is None:
        return None
    return _extract_literal_values(target_field.annotation)


def repair_enums(
    error: ValidationError,
    raw_dict: dict[str, Any],
    model: type[BaseModel],
    log: Any | None = None,
) -> BaseModel | None:
    """Repair close-but-wrong enum values in a failed structured output.

    Reads the literal/enum mismatches out of ``error``, fuzzy-matches each wrong
    value back to the nearest valid ``Literal`` member, patches the raw dict, and
    re-validates. Returns the repaired model instance, or ``None`` if any error
    is non-enum or any value could not be confidently matched.
    """
    if log is None:
        log = _log

    errors = error.errors()
    enum_errors = [e for e in errors if _is_enum_error(e)]

    if not enum_errors:
        return None

    non_enum_errors = [e for e in errors if not _is_enum_error(e)]
    if non_enum_errors:
        log.debug(
            f"enum_repair_skip_mixed_errors enum={len(enum_errors)} "
            f"non_enum={len(non_enum_errors)}"
        )
        return None

    patched = copy.deepcopy(raw_dict)
    repaired_count = 0
    failed_repairs: list[dict[str, Any]] = []

    for err in enum_errors:
        loc = err.get("loc", ())
        raw_value = err.get("input")

        if not isinstance(raw_value, str) or not loc:
            failed_repairs.append({"loc": loc, "input": raw_value, "reason": "non-string"})
            continue

        path = loc[:-1]
        field_name = loc[-1]

        if not isinstance(field_name, str):
            failed_repairs.append({"loc": loc, "input": raw_value, "reason": "non-string-field"})
            continue

        valid_values = _resolve_model_for_field(model, path, field_name)
        if not valid_values:
            enum_fields = _collect_enum_fields(model)
            valid_values = enum_fields.get(field_name)

        if not valid_values:
            failed_repairs.append({"loc": loc, "input": raw_value, "reason": "no-valid-values"})
            continue

        repaired_value = _fuzzy_repair_value(raw_value, valid_values)
        if repaired_value is None:
            failed_repairs.append(
                {"loc": loc, "input": raw_value, "valid": valid_values, "reason": "fuzzy-no-match"}
            )
            continue

        log.debug(
            f"enum_repair_fuzzy_match field={'.'.join(str(p) for p in loc)} "
            f"original={raw_value} repaired={repaired_value}"
        )
        _patch_nested_dict(patched, loc, repaired_value)
        repaired_count += 1

    if failed_repairs:
        log.warning(
            f"enum_repair_partial_failure repaired={repaired_count} "
            f"failed={len(failed_repairs)}"
        )
        return None

    try:
        result = model.model_validate(patched)
    except ValidationError as exc:
        log.warning(
            f"enum_repair_revalidation_failed repaired={repaired_count} "
            f"error={str(exc)[:300]}"
        )
        return None

    log.info(f"enum_repair_success repaired={repaired_count}")
    return result


def _find_list_model_fields(model: type[BaseModel]) -> dict[str, type[BaseModel]]:
    """Find all ``list[BaseModel]`` fields on a model."""
    result: dict[str, type[BaseModel]] = {}
    for field_name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        origin = get_origin(annotation)
        if origin is list:
            args = get_args(annotation)
            if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                result[field_name] = args[0]
    return result


def salvage_list(
    raw_dict: dict[str, Any],
    model: type[BaseModel],
    list_field: str,
    log: Any | None = None,
) -> BaseModel | None:
    """Salvage valid entries from one list field whose tail entries are broken.

    Gemini sometimes emits empty ``{}`` or partially-formed objects at the tail
    of long arrays when the output token budget runs out, and Pydantic rejects
    the *entire* array because of one garbage entry. This validates each entry
    individually, keeps the valid ones, drops the broken ones, and re-validates
    the whole model. Returns the salvaged model, or ``None`` if nothing was
    dropped (no salvage needed) or nothing was salvageable.
    """
    if log is None:
        log = _log

    raw_list = raw_dict.get(list_field)
    if not isinstance(raw_list, list) or not raw_list:
        return None

    item_model: type[BaseModel] | None = None
    field_info = model.model_fields.get(list_field)
    if field_info is not None:
        annotation = field_info.annotation
        origin = get_origin(annotation)
        if origin is list:
            args = get_args(annotation)
            if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                item_model = args[0]

    if item_model is None:
        return None

    valid: list[BaseModel] = []
    dropped = 0
    for item in raw_list:
        try:
            valid.append(item_model.model_validate(item))
        except Exception:  # noqa: BLE001 - drop any entry that fails for any reason
            dropped += 1

    if not valid:
        log.warning(f"salvage_list_all_dropped field={list_field} total={len(raw_list)}")
        return None

    if dropped == 0:
        return None

    patched = copy.deepcopy(raw_dict)
    patched[list_field] = [v.model_dump() for v in valid]

    try:
        result = model.model_validate(patched)
    except ValidationError as exc:
        log.warning(
            f"salvage_list_revalidation_failed field={list_field} "
            f"valid={len(valid)} error={str(exc)[:300]}"
        )
        return None

    log.info(
        f"salvage_list_success field={list_field} salvaged={len(valid)} "
        f"dropped={dropped} total={len(raw_list)}"
    )
    return result


def salvage_lists(
    raw_dict: dict[str, Any],
    model: type[BaseModel],
    log: Any | None = None,
) -> BaseModel | None:
    """Try :func:`salvage_list` on every ``list[BaseModel]`` field of the model.

    Auto-detects list fields and attempts salvage on each. Returns the first
    successful result, or ``None`` if nothing was salvageable.
    """
    for field_name in _find_list_model_fields(model):
        result = salvage_list(raw_dict, model, field_name, log)
        if result is not None:
            return result
    return None
