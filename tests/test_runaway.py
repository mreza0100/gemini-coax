"""Tests for MAX_TOKENS string-runaway detection and recovery-schema stripping."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from gemini_coax import (
    MAX_TOKENS_FINISH_REASONS,
    rehome_to_original,
    runaway_prone_string_fields,
    strip_runaway_strings,
)


class Pair(BaseModel):
    segment_index: int
    technique: Literal["open", "closed"] | None = None  # enum string — safe
    speaker_role: str  # bare str — runaway-prone
    topic_label: str | None = Field(default=None, max_length=80)  # large cap — runaway-prone
    short_code: str | None = Field(default=None, max_length=8)  # tight cap — safe


class Out(BaseModel):
    pairs: list[Pair]


def test_runaway_prone_detects_only_unbounded_strings() -> None:
    fields = runaway_prone_string_fields(Pair)
    assert "speaker_role" in fields  # bare str
    assert "topic_label" in fields  # str with cap Gemini ignores (> 64)
    assert "technique" not in fields  # Literal enum — decode-constrained
    assert "short_code" not in fields  # tight max_length ≤ 64
    assert "segment_index" not in fields  # not a string


def test_strip_runaway_strings_trims_item_model_fields() -> None:
    trimmed, stripped = strip_runaway_strings(Out)
    assert stripped == {"pairs": ["speaker_role", "topic_label"]}
    item_model = trimmed.model_fields["pairs"].annotation.__args__[0]
    item_fields = set(item_model.model_fields)
    assert "speaker_role" not in item_fields
    assert "topic_label" not in item_fields
    assert "technique" in item_fields
    assert "short_code" in item_fields
    assert "segment_index" in item_fields


def test_strip_runaway_strings_returns_original_when_nothing_qualifies() -> None:
    class Safe(BaseModel):
        kind: Literal["a", "b"]
        n: int

    model, stripped = strip_runaway_strings(Safe)
    assert stripped == {}
    assert model is Safe


def test_trimmed_model_validates_payload_without_stripped_fields() -> None:
    trimmed, _ = strip_runaway_strings(Out)
    obj = trimmed.model_validate({"pairs": [{"segment_index": 0, "technique": "open"}]})
    assert len(obj.pairs) == 1


def test_top_level_runaway_string_is_stripped() -> None:
    class Doc(BaseModel):
        free_summary: str  # unbounded top-level string
        count: int

    trimmed, stripped = strip_runaway_strings(Doc)
    assert stripped == {"<root>": ["free_summary"]}
    assert "free_summary" not in trimmed.model_fields
    assert "count" in trimmed.model_fields


def test_finish_reason_set_covers_known_spellings() -> None:
    assert "MAX_TOKENS" in MAX_TOKENS_FINISH_REASONS
    assert "LENGTH" in MAX_TOKENS_FINISH_REASONS


def test_rehome_fills_required_nonnullable_string_with_empty() -> None:
    # speaker_role is required + non-nullable; rehome must inject "" not raise.
    trimmed, stripped = strip_runaway_strings(Out)
    recovered = trimmed.model_validate(
        {"pairs": [{"segment_index": 0, "technique": "open"}]}
    )
    rehomed = rehome_to_original(recovered, Out, stripped)
    assert isinstance(rehomed, Out)
    assert rehomed.pairs[0].speaker_role == ""  # required str → placeholder ""
    assert rehomed.pairs[0].topic_label is None  # nullable → None default
    assert rehomed.pairs[0].technique == "open"  # preserved


def test_rehome_preserves_all_recovered_rows() -> None:
    trimmed, stripped = strip_runaway_strings(Out)
    recovered = trimmed.model_validate(
        {"pairs": [{"segment_index": i, "technique": "open"} for i in range(16)]}
    )
    rehomed = rehome_to_original(recovered, Out, stripped)
    assert len(rehomed.pairs) == 16
    assert [p.segment_index for p in rehomed.pairs] == list(range(16))
