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
    speaker_role: str  # REQUIRED bare str — never stripped (placeholder would
    # fail the required constraint on rehome and collapse the row)
    free_note: str | None = None  # OPTIONAL bare str — runaway-prone, strippable
    topic_label: str | None = Field(default=None, max_length=80)  # large cap — runaway-prone
    short_code: str | None = Field(default=None, max_length=8)  # tight cap — safe


class Out(BaseModel):
    pairs: list[Pair]


def test_runaway_prone_detects_only_unbounded_optional_strings() -> None:
    fields = runaway_prone_string_fields(Pair)
    assert "free_note" in fields  # optional bare str
    assert "topic_label" in fields  # optional str with cap Gemini ignores (> 64)
    assert "speaker_role" not in fields  # REQUIRED bare str — never stripped
    assert "technique" not in fields  # Literal enum — decode-constrained
    assert "short_code" not in fields  # tight max_length ≤ 64
    assert "segment_index" not in fields  # not a string


def test_strip_runaway_strings_trims_item_model_fields() -> None:
    trimmed, stripped = strip_runaway_strings(Out)
    assert stripped == {"pairs": ["free_note", "topic_label"]}
    item_model = trimmed.model_fields["pairs"].annotation.__args__[0]
    item_fields = set(item_model.model_fields)
    assert "free_note" not in item_fields
    assert "topic_label" not in item_fields
    assert "speaker_role" in item_fields  # required str preserved
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
    obj = trimmed.model_validate(
        {"pairs": [{"segment_index": 0, "technique": "open", "speaker_role": "patient"}]}
    )
    assert len(obj.pairs) == 1


def test_top_level_runaway_string_is_stripped() -> None:
    class Doc(BaseModel):
        free_summary: str | None = None  # optional unbounded top-level string
        required_label: str  # REQUIRED top-level string — never stripped
        count: int

    trimmed, stripped = strip_runaway_strings(Doc)
    assert stripped == {"<root>": ["free_summary"]}
    assert "free_summary" not in trimmed.model_fields
    assert "required_label" in trimmed.model_fields  # required preserved
    assert "count" in trimmed.model_fields


def test_finish_reason_set_covers_known_spellings() -> None:
    assert "MAX_TOKENS" in MAX_TOKENS_FINISH_REASONS
    assert "LENGTH" in MAX_TOKENS_FINISH_REASONS


def test_rehome_preserves_required_string_and_fills_optional_with_none() -> None:
    # speaker_role is required → never stripped, must survive the roundtrip.
    # Optional strippable fields (free_note, topic_label) → None default on rehome.
    trimmed, stripped = strip_runaway_strings(Out)
    recovered = trimmed.model_validate(
        {"pairs": [{"segment_index": 0, "technique": "open", "speaker_role": "patient"}]}
    )
    rehomed = rehome_to_original(recovered, Out, stripped)
    assert isinstance(rehomed, Out)
    assert rehomed.pairs[0].speaker_role == "patient"  # required str preserved
    assert rehomed.pairs[0].free_note is None  # nullable strippable → None default
    assert rehomed.pairs[0].topic_label is None  # nullable strippable → None default
    assert rehomed.pairs[0].technique == "open"  # preserved


def test_rehome_preserves_all_recovered_rows() -> None:
    trimmed, stripped = strip_runaway_strings(Out)
    recovered = trimmed.model_validate(
        {
            "pairs": [
                {"segment_index": i, "technique": "open", "speaker_role": "patient"}
                for i in range(16)
            ]
        }
    )
    rehomed = rehome_to_original(recovered, Out, stripped)
    assert len(rehomed.pairs) == 16
    assert [p.segment_index for p in rehomed.pairs] == list(range(16))
