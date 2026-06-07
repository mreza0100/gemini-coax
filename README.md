# gemini-coax

**Make Google Gemini structured output actually validate against your Pydantic models.**

[![PyPI](https://img.shields.io/pypi/v/gemini-coax.svg)](https://pypi.org/project/gemini-coax/)
[![Python](https://img.shields.io/pypi/pyversions/gemini-coax.svg)](https://pypi.org/project/gemini-coax/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

Gemini's `response_json_schema` promises structured output, then quietly breaks
its own promise. It enforces *shape* (types, properties, required) but **silently
ignores value-level constraints** — so the model hallucinates enum values, blows
past your numeric bounds, and trails off into half-formed objects at the end of
long arrays. Pydantic then rejects the *entire* response over one bad field.

`gemini-coax` coaxes the output back into shape. No retries, no extra LLM calls
for the common cases — just targeted repair at the validation seam.

If you've hit any of these, this library is for you:

- `ValueError: AnyOf is not supported in the response schema for the Gemini API`
- `Input should be 'a', 'b' or 'c' [type=literal_error]` on a value the schema *defined*
- A nullable `Literal[...] | None` field where Gemini invents values off-menu
- `ge`/`le`/`max_length`/`max_items` constraints ignored, failing validation
- Empty `{}` or truncated objects at the tail of a long list, killing the whole array

## Install

```bash
pip install gemini-coax                  # core — pure, depends only on pydantic
pip install "gemini-coax[langchain]"     # + the drop-in ChatGoogleGenerativeAI
```

## Use it — LangChain (`langchain-google-genai`)

Swap `ChatGoogleGenerativeAI` for `GeminiSafe`. That's the whole change. Every
`with_structured_output()` call is now coaxed; no edits in your chains.

```python
from typing import Literal
from pydantic import BaseModel, Field
from gemini_coax import GeminiSafe          # was: ChatGoogleGenerativeAI

class Finding(BaseModel):
    label: Literal["bug", "smell", "nit"] | None   # nullable enum — Gemini drops the enum
    severity: int = Field(ge=1, le=5)              # bounds Gemini ignores

class Report(BaseModel):
    findings: list[Finding]                        # long array → degraded tail

llm = GeminiSafe(model="gemini-2.5-flash", temperature=0)
report = llm.with_structured_output(Report).invoke("Review this diff: ...")
# Validates. The anyOf-enum is stripped before send, out-of-range
# severities are clamped, and a broken trailing finding is salvaged away.
```

It also retries transient transport faults (`ConnectionResetError`, aiohttp
`ClientOSError`, `ServerDisconnectedError`) that the google-genai SDK leaves
uncaught — at the single async seam every call funnels through.

## Use it — raw `google-genai` SDK (no LangChain)

One call. Hand it the decoded dict and your model:

```python
from gemini_coax import coax

raw = json.loads(response.text)     # whatever Gemini gave you
report = coax(raw, Report)          # clamp → fill nullables → validate → repair enums → salvage lists
```

Or compose the pieces yourself:

```python
from gemini_coax import (
    strip_nullable_anyof,   # rewrite the schema BEFORE you send it
    clamp_to_constraints,   # clamp ignored numeric / length / array bounds
    fill_missing_nullables, # inject None for nullables Gemini omitted
    repair_enums,           # fuzzy-match close-but-wrong enum values
    salvage_lists,          # drop broken tail entries, keep the valid ones
)

schema = strip_nullable_anyof(Report.model_json_schema())   # send THIS to Gemini
```

## What it does

| Gemini misbehavior | gemini-coax response |
| --- | --- |
| Drops `enum` inside `anyOf` (nullable `Literal`) → hallucinated values | `strip_nullable_anyof` rewrites the schema to a plain enum + drops it from `required` before send |
| Ignores `ge/le/gt/lt`, `max_length`, `max_items` | `clamp_to_constraints` clamps raw values to the model's field metadata |
| Omits a now-optional nullable field | `fill_missing_nullables` injects `None` so re-validation passes |
| Close-but-wrong enum at the array tail (`"defensiveness"` vs `"defensiveness-tone"`) | `repair_enums` fuzzy-matches it back (zero-cost `difflib`) |
| Empty `{}` / truncated objects when the token budget runs out | `salvage_lists` validates entries individually, keeps the good ones |
| Transient transport fault before any HTTP status | `GeminiSafe` retries with exponential backoff + jitter |

A full-chain retry is 100–300× more expensive than these repairs — and often
makes things worse. Repair beats re-roll.

## Design

Two layers, so the value isn't hostage to any framework's release notes:

- **Core** (`gemini_coax.schema`, `gemini_coax.repair`, `coax`) — pure functions
  over `dict` + Pydantic. Only dependency is `pydantic`. Works with the raw SDK,
  Vertex AI, or anything that hands you a dict.
- **Adapter** (`gemini_coax.langchain.GeminiSafe`) — the LangChain drop-in.
  Pulled in only by the `[langchain]` extra; pins `langchain-google-genai>=4.2,<5`.

## License

MIT
