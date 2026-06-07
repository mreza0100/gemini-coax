"""Tests for enum repair and list salvage."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ValidationError

from gemini_coax import repair_enums, salvage_lists


class Item(BaseModel):
    kind: Literal["defensiveness-tone", "criticism", "contempt"]


class Bag(BaseModel):
    items: list[Item]


def _validation_error(raw: dict) -> ValidationError:
    try:
        Bag.model_validate(raw)
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


def test_repair_enums_fuzzy_matches_close_value() -> None:
    raw = {"items": [{"kind": "defensiveness"}]}  # missing "-tone"
    error = _validation_error(raw)
    repaired = repair_enums(error, raw, Bag)
    assert repaired is not None
    assert repaired.items[0].kind == "defensiveness-tone"


def test_repair_enums_returns_none_for_unrelated_garbage() -> None:
    raw = {"items": [{"kind": "completely-unrelated-xyz"}]}
    error = _validation_error(raw)
    assert repair_enums(error, raw, Bag) is None


def test_salvage_lists_drops_broken_tail_entry() -> None:
    raw = {"items": [{"kind": "criticism"}, {"kind": "contempt"}, {}]}  # trailing {}
    salvaged = salvage_lists(raw, Bag)
    assert salvaged is not None
    assert len(salvaged.items) == 2
    assert [i.kind for i in salvaged.items] == ["criticism", "contempt"]


def test_salvage_lists_returns_none_when_nothing_dropped() -> None:
    raw = {"items": [{"kind": "criticism"}]}
    assert salvage_lists(raw, Bag) is None


def test_salvage_lists_returns_none_when_all_broken() -> None:
    raw = {"items": [{}, {"kind": "not-valid"}]}
    assert salvage_lists(raw, Bag) is None
