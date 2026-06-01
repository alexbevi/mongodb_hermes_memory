"""Tests for the regex and LLM extractors."""

from __future__ import annotations

from hermes_mongodb_memory.extraction import (
    LLMExtractor,
    RegexExtractor,
    _parse_llm_payload,
)


def test_regex_extractor_picks_up_preferences():
    msgs = [
        {"role": "user", "content": "I prefer dark mode for late-night sessions."},
        {"role": "assistant", "content": "Got it."},
    ]
    out = RegexExtractor().extract(msgs)
    assert len(out) == 1
    assert out[0].category == "user_pref"
    assert "dark mode" in out[0].content.lower()


def test_regex_extractor_picks_up_decisions():
    msgs = [{"role": "user", "content": "We decided to use MongoDB Atlas for storage."}]
    out = RegexExtractor().extract(msgs)
    assert len(out) == 1
    assert out[0].category == "decision"
    assert "Atlas" in out[0].content or "atlas" in out[0].content.lower()


def test_regex_extractor_skips_short_messages():
    msgs = [{"role": "user", "content": "yep."}]
    assert RegexExtractor().extract(msgs) == []


def test_regex_extractor_ignores_non_user_roles():
    msgs = [{"role": "assistant", "content": "I prefer dark mode for users."}]
    assert RegexExtractor().extract(msgs) == []


def test_regex_extractor_collects_capitalized_entities():
    msgs = [{"role": "user", "content": "I always ship docs through MkDocs and ReadMe."}]
    out = RegexExtractor().extract(msgs)
    assert out
    assert any(e in {"MkDocs", "ReadMe"} for e in out[0].entities)


def test_regex_extractor_handles_both_in_one_message():
    msgs = [
        {
            "role": "user",
            "content": "I prefer terse summaries. We decided to migrate to MongoDB next month.",
        }
    ]
    out = RegexExtractor().extract(msgs)
    cats = {m.category for m in out}
    assert cats == {"user_pref", "decision"}


def test_llm_extractor_returns_empty_when_no_caller_resolved(monkeypatch):
    monkeypatch.setattr(LLMExtractor, "_resolve_hermes_caller", staticmethod(lambda: None))
    out = LLMExtractor().extract([{"role": "user", "content": "I prefer dark mode"}])
    assert out == []


def test_llm_extractor_invokes_caller_with_messages():
    captured: list[list[dict]] = []

    def caller(messages):
        captured.append(messages)
        return '[{"content": "user prefers dark mode", "category": "user_pref"}]'

    out = LLMExtractor(caller=caller).extract(
        [{"role": "user", "content": "I prefer dark mode in all things"}]
    )
    assert len(out) == 1
    assert out[0].category == "user_pref"
    assert "auto" in out[0].tags and "llm" in out[0].tags
    assert captured  # the caller was invoked


def test_llm_extractor_handles_empty_history():
    out = LLMExtractor(caller=lambda m: "[]").extract([])
    assert out == []


def test_llm_extractor_swallows_caller_exceptions():
    def boom(messages):
        raise RuntimeError("network down")

    out = LLMExtractor(caller=boom).extract([{"role": "user", "content": "I prefer x"}])
    assert out == []


def test_parse_payload_strips_code_fences():
    raw = '```json\n[{"content": "x", "category": "fact"}]\n```'
    out = _parse_llm_payload(raw)
    assert len(out) == 1
    assert out[0].category == "fact"


def test_parse_payload_ignores_non_array_root():
    assert _parse_llm_payload('{"content": "x"}') == []


def test_parse_payload_clamps_invalid_categories_to_general():
    raw = '[{"content": "y", "category": "bizarre"}]'
    out = _parse_llm_payload(raw)
    assert out[0].category == "general"


def test_parse_payload_skips_empty_content():
    raw = '[{"content": "", "category": "fact"}, {"content": "ok", "category": "fact"}]'
    out = _parse_llm_payload(raw)
    assert len(out) == 1
    assert out[0].content == "ok"


def test_parse_payload_handles_non_json_input():
    assert _parse_llm_payload("not valid json") == []
    assert _parse_llm_payload("") == []


def test_parse_payload_skips_non_dict_items():
    """A list with mixed types should drop the non-dict entries."""
    raw = '[{"content": "ok", "category": "fact"}, "string", 42, null]'
    out = _parse_llm_payload(raw)
    assert len(out) == 1
    assert out[0].content == "ok"


def test_parse_payload_handles_non_list_entities_field():
    """If 'entities' is a string (not list), default to []."""
    raw = '[{"content": "x", "category": "fact", "entities": "not-a-list"}]'
    out = _parse_llm_payload(raw)
    assert out[0].entities == []


def test_parse_payload_caps_entity_count():
    """Should cap entities at 10 to avoid blowup."""
    ents = '","'.join(f"e{i}" for i in range(20))
    raw = f'[{{"content": "x", "category": "fact", "entities": ["{ents}"]}}]'
    out = _parse_llm_payload(raw)
    assert len(out[0].entities) <= 10


def test_regex_extractor_handles_non_string_content():
    """Messages with non-string content (e.g. tool calls) should be skipped, not crash."""
    msgs = [
        {"role": "user", "content": [{"type": "image", "url": "..."}]},  # multimodal-style
        {"role": "user", "content": None},
        {"role": "user", "content": "I prefer terse responses now."},
    ]
    out = RegexExtractor().extract(msgs)
    assert len(out) == 1


def test_parse_payload_truncates_long_content():
    long = "a" * 2000
    raw = '[{"content": "' + long + '", "category": "fact"}]'
    out = _parse_llm_payload(raw)
    assert len(out[0].content) == 1000
