"""Tests for the framework-free schema transforms."""

from __future__ import annotations

from typing import Literal

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
