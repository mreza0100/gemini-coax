"""Schema-compatibility transforms for Gemini structured output.

Gemini's ``response_json_schema`` enforces *structural* constraints (type, enum,
required, properties) but silently ignores *value-level* ones:

* ``anyOf``            — an ``enum`` inside an ``anyOf`` branch is dropped, so the
  model hallucinates values for nullable ``Literal`` fields.
* ``minimum/maximum``  — numeric bounds are not enforced.
* ``minLength/maxLength`` — string length bounds are not enforced.
* ``minItems/maxItems``  — array length bounds are not enforced.

These functions operate on plain ``dict`` JSON Schema and Pydantic models — no
LangChain, no provider SDK. They are the framework-free core of ``gemini-coax``.
"""

from __future__ import annotations

import copy
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel

__all__ = [
    "strip_nullable_anyof",
    "clamp_to_constraints",
    "fill_missing_nullables",
]


def strip_nullable_anyof(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove ``anyOf``-null patterns from a JSON Schema for Gemini compatibility.

    Transforms ``anyOf: [{type/enum/$ref}, {type: null}]`` into the non-null
    branch and removes the field from ``required``. Walks ``$defs`` and nested
    objects recursively. Returns a deep copy — the input is not mutated.
    """
    schema = copy.deepcopy(schema)

    def _fix_object(obj: dict[str, Any]) -> None:
        props = obj.get("properties")
        if not isinstance(props, dict):
            return
        required = set(obj.get("required", []))
        changed = False

        for field_name, field_def in props.items():
            if not isinstance(field_def, dict):
                continue

            branches = field_def.get("anyOf")
            if not isinstance(branches, list) or len(branches) != 2:
                _recurse(field_def)
                continue

            null_branch = non_null_branch = None
            for b in branches:
                if isinstance(b, dict) and b.get("type") == "null":
                    null_branch = b
                else:
                    non_null_branch = b

            if null_branch is None or non_null_branch is None:
                _recurse(field_def)
                continue

            del field_def["anyOf"]
            field_def.pop("default", None)
            if isinstance(non_null_branch, dict):
                for k, v in non_null_branch.items():
                    field_def[k] = v
            if field_name in required:
                required.discard(field_name)
                changed = True

            _recurse(field_def)

        if changed:
            if required:
                obj["required"] = sorted(required)
            else:
                obj.pop("required", None)

    def _recurse(node: dict[str, Any] | list[Any] | Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                _fix_object(node)
            items = node.get("items")
            if isinstance(items, dict):
                _recurse(items)
        elif isinstance(node, list):
            for item in node:
                _recurse(item)

    for def_schema in (schema.get("$defs") or {}).values():
        if isinstance(def_schema, dict):
            _fix_object(def_schema)

    _fix_object(schema)
    return schema


def _resolve_inner_type(annotation: Any) -> Any:
    """Unwrap ``Optional[X]`` / ``X | None`` to ``X``."""
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def clamp_to_constraints(data: Any, model: type[BaseModel]) -> Any:
    """Clamp raw dict values to a model's field constraints before validation.

    Gemini's constrained decoding ignores numeric bounds (``ge/le/gt/lt``),
    string length limits (``max_length``), and array size limits (``maxItems``).
    Without this, a single out-of-range value makes Pydantic reject the *entire*
    output. This reads constraints from Pydantic field metadata and silently
    clamps values so validation succeeds. Recurses into nested ``BaseModel`` and
    ``list[BaseModel]`` fields. Returns a clamped copy — the input is not mutated.
    """
    if not isinstance(data, dict):
        return data

    result = dict(data)

    for field_name, field_info in model.model_fields.items():
        if field_name not in result or result[field_name] is None:
            continue

        value = result[field_name]

        for m in field_info.metadata:
            ge = getattr(m, "ge", None)
            le = getattr(m, "le", None)
            gt = getattr(m, "gt", None)
            lt = getattr(m, "lt", None)
            max_length = getattr(m, "max_length", None)

            # Gemini does not enforce minimum/maximum — clamp numerics.
            if isinstance(value, (int, float)):
                if ge is not None and value < ge:
                    value = type(value)(ge)
                if le is not None and value > le:
                    value = type(value)(le)
                if gt is not None and value <= gt:
                    value = gt + (0.001 if isinstance(value, float) else 1)
                if lt is not None and value >= lt:
                    value = lt - (0.001 if isinstance(value, float) else 1)

            # Gemini does not enforce maxLength — truncate strings.
            if max_length is not None and isinstance(value, str) and len(value) > max_length:
                value = value[: max_length - 3] + "..." if max_length > 3 else value[:max_length]

            # Gemini does not enforce maxItems — truncate arrays.
            if max_length is not None and isinstance(value, list) and len(value) > max_length:
                value = value[:max_length]

        inner = _resolve_inner_type(field_info.annotation)
        inner_origin = get_origin(inner)
        inner_args = get_args(inner)

        if inner_origin is list and inner_args:
            item_type = inner_args[0]
            is_model_list = isinstance(item_type, type) and issubclass(item_type, BaseModel)
            if is_model_list and isinstance(value, list):
                value = [
                    clamp_to_constraints(item, item_type) if isinstance(item, dict) else item
                    for item in value
                ]
        elif isinstance(inner, type) and issubclass(inner, BaseModel) and isinstance(value, dict):
            value = clamp_to_constraints(value, inner)

        result[field_name] = value

    return result


def fill_missing_nullables(data: Any, model: type[BaseModel]) -> Any:
    """Inject ``None`` for nullable fields the model legitimately omitted.

    ``strip_nullable_anyof`` removes nullable fields (``X | None``) from the
    schema's ``required`` set before sending it to Gemini, so Gemini may omit
    them entirely. The original model still lists such a field as required when
    it has no default (required-but-nullable), so re-validation raises "Field
    required" and the whole output is rejected. This restores consistency: a
    field we told Gemini was optional is filled with ``None`` when absent.
    Recurses into nested ``BaseModel`` and ``list[BaseModel]`` fields, mirroring
    ``strip_nullable_anyof``'s recursive stripping.
    """
    if not isinstance(data, dict):
        return data

    result = dict(data)

    for field_name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        if type(None) in get_args(annotation) and field_name not in result:
            result[field_name] = None
            continue

        value = result.get(field_name)
        if value is None:
            continue

        inner = _resolve_inner_type(annotation)
        inner_args = get_args(inner)

        if get_origin(inner) is list and inner_args:
            item_type = inner_args[0]
            if (
                isinstance(item_type, type)
                and issubclass(item_type, BaseModel)
                and isinstance(value, list)
            ):
                result[field_name] = [
                    fill_missing_nullables(item, item_type) if isinstance(item, dict) else item
                    for item in value
                ]
        elif isinstance(inner, type) and issubclass(inner, BaseModel) and isinstance(value, dict):
            result[field_name] = fill_missing_nullables(value, inner)

    return result
