"""Tests for the framework-free schema transforms."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from gemini_coax import (
    clamp_to_constraints,
    coax,
    fill_missing_nullables,
    strip_nullable_anyof,
)


class Finding(BaseModel):
    label: Literal["bug", "smell", "nit"] | None = None
    severity: int = Field(ge=1, le=5)
    note: str = Field(max_length=10)


class Report(BaseModel):
    findings: list[Finding]


def test_strip_nullable_anyof_collapses_to_plain_enum() -> None:
    schema = strip_nullable_anyof(Finding.model_json_schema())
    label = schema["$defs"]["Finding"]["properties"]["label"] if "$defs" in schema else None
    # Finding is the root here, not under $defs.
    label = schema["properties"]["label"]
    assert "anyOf" not in label
    assert label["enum"] == ["bug", "smell", "nit"]
    assert "label" not in schema.get("required", [])


def test_strip_nullable_anyof_does_not_mutate_input() -> None:
    original = Finding.model_json_schema()
    snapshot = str(original)
    strip_nullable_anyof(original)
    assert str(original) == snapshot


def test_clamp_numeric_bounds() -> None:
    out = clamp_to_constraints({"severity": 99, "note": "ok"}, Finding)
    assert out["severity"] == 5
    out = clamp_to_constraints({"severity": -3, "note": "ok"}, Finding)
    assert out["severity"] == 1


def test_clamp_string_max_length() -> None:
    out = clamp_to_constraints({"severity": 3, "note": "x" * 50}, Finding)
    assert len(out["note"]) == 10


def test_clamp_recurses_into_list_of_models() -> None:
    out = clamp_to_constraints(
        {"findings": [{"severity": 99, "note": "ok"}, {"severity": 0, "note": "ok"}]},
        Report,
    )
    assert out["findings"][0]["severity"] == 5
    assert out["findings"][1]["severity"] == 1


def test_fill_missing_nullables() -> None:
    out = fill_missing_nullables({"severity": 3, "note": "ok"}, Finding)
    assert out["label"] is None


def test_unknown_top_level_attr_raises_attribute_error() -> None:
    import gemini_coax

    with pytest.raises(AttributeError):
        _ = gemini_coax.does_not_exist


def test_geminisafe_lazy_export() -> None:
    """`from gemini_coax import GeminiSafe` resolves the adapter lazily.

    When the [langchain] extra is installed it returns the class; when absent it
    raises the adapter's helpful ImportError. Either outcome proves the lazy
    PEP 562 path fires (a plain AttributeError would mean it never tried).
    """
    import gemini_coax

    try:
        from gemini_coax.langchain import GeminiSafe as Direct
    except ImportError:
        with pytest.raises(ImportError):
            _ = gemini_coax.GeminiSafe
    else:
        assert gemini_coax.GeminiSafe is Direct


def test_coax_end_to_end_repairs_and_validates() -> None:
    raw = {
        "findings": [
            {"label": "bug", "severity": 99, "note": "way too long to fit"},
            {"severity": 2, "note": "ok"},  # label omitted
        ]
    }
    report = coax(raw, Report)
    assert isinstance(report, Report)
    assert report.findings[0].severity == 5
    assert report.findings[1].label is None


class Box(BaseModel):
    label: str
    note: str | None  # required-but-nullable (no default)


class Crate(BaseModel):
    box: Box | None  # PEP 604 nullable nested model, required-but-nullable
    boxes: list[Box]


def test_fill_missing_nullables_recurses_into_pep604_nested_model() -> None:
    # `box` / `boxes[*]` are present dicts whose required-but-nullable `note` was
    # omitted. The PEP 604 `Box | None` annotation must still unwrap so recursion
    # reaches the nested model and fills note=None — regression: the union check
    # matched only typing.Union, never types.UnionType, so `X | None` was skipped.
    out = fill_missing_nullables({"box": {"label": "x"}, "boxes": [{"label": "y"}]}, Crate)
    assert out["box"]["note"] is None
    assert out["boxes"][0]["note"] is None
    assert isinstance(Crate.model_validate(out), Crate)


class ConstrainedCrate(BaseModel):
    finding: Finding | None  # PEP 604 nullable nested model carrying value bounds


def test_clamp_recurses_into_pep604_nested_model() -> None:
    # Same union-resolution path gates clamp recursion into a `Finding | None`.
    out = clamp_to_constraints({"finding": {"severity": 99, "note": "ok"}}, ConstrainedCrate)
    assert out["finding"]["severity"] == 5
