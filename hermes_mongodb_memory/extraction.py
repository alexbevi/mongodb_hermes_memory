"""Session-end memory extraction.

Two strategies, both opt-in via config:

* :class:`RegexExtractor` — deterministic, dependency-free.  Mirrors the
  approach from the bundled ``holographic`` provider so users get
  predictable behaviour with no extra setup.
* :class:`LLMExtractor` — defers to Hermes' configured LLM (resolved
  lazily at session end).  Off by default; enabled via
  ``extraction_llm: true``.

Each yields a list of :class:`ExtractedMemory` records that the provider
turns into ``store.add_memory`` calls.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Patterns mirror the bundled `holographic` provider so contributors who
# move between plugins see consistent behavior.
_PREF_PATTERNS = [
    re.compile(r"\bI (?:prefer|like|love|use|want|need) ([^.!?]+)", re.IGNORECASE),
    re.compile(r"\bmy (?:favou?rite|preferred|default) ([^.!?]+? is [^.!?]+)", re.IGNORECASE),
    re.compile(r"\bI (?:always|never|usually) ([^.!?]+)", re.IGNORECASE),
]

_DECISION_PATTERNS = [
    re.compile(r"\bwe (?:decided|agreed|chose) ([^.!?]+)", re.IGNORECASE),
    re.compile(r"\bthe (?:project|repo|team) (?:uses|needs|requires) ([^.!?]+)", re.IGNORECASE),
]

_ENTITY_PATTERN = re.compile(r"\b([A-Z][a-zA-Z0-9_-]{2,})\b")


@dataclass
class ExtractedMemory:
    content: str
    category: str
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


class RegexExtractor:
    """Deterministic preference/decision extractor."""

    def extract(self, messages: Iterable[dict[str, Any]]) -> list[ExtractedMemory]:
        out: list[ExtractedMemory] = []
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, str) or len(content) < 10:
                continue
            text = content.strip()
            ents = sorted(set(_ENTITY_PATTERN.findall(text)))[:5]

            for pat in _PREF_PATTERNS:
                m = pat.search(text)
                if m:
                    snippet = m.group(0)[:400]
                    out.append(ExtractedMemory(snippet, "user_pref", entities=ents, tags=["auto"]))
                    break
            for pat in _DECISION_PATTERNS:
                m = pat.search(text)
                if m:
                    snippet = m.group(0)[:400]
                    out.append(ExtractedMemory(snippet, "decision", entities=ents, tags=["auto"]))
                    break
        return out


class LLMExtractor:
    """Resolves Hermes' LLM at runtime; falls back gracefully when unavailable.

    The protocol intentionally keeps the surface tiny — anything that
    quacks like ``call(messages: list[dict]) -> str`` will work, which
    keeps the plugin tests honest without dragging Hermes' LLM stack
    into them.
    """

    SYSTEM_PROMPT = (
        "You distil long-term memories from a conversation. Return a JSON array of "
        "objects with fields: content (string), category (one of "
        "user_pref/project/fact/decision/general), entities (array of strings). "
        "Only emit memories that would be useful in a *future* conversation. "
        "Skip ephemeral chit-chat. Return [] if nothing is worth keeping."
    )

    def __init__(self, caller: Any | None = None) -> None:
        self._caller = caller

    def extract(self, messages: Iterable[dict[str, Any]]) -> list[ExtractedMemory]:
        caller = self._caller or self._resolve_hermes_caller()
        if caller is None:
            logger.debug("no LLM caller available; skipping LLM extraction")
            return []

        # Compact the conversation: take user/assistant turns only.
        compacted = [
            {"role": m.get("role", "user"), "content": str(m.get("content", ""))[:2000]}
            for m in messages
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        if not compacted:
            return []

        try:
            raw = caller(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": _serialise_compact(compacted)},
                ]
            )
        except Exception as exc:
            logger.warning("LLM extraction failed: %s", exc)
            return []

        return _parse_llm_payload(raw)

    @staticmethod
    def _resolve_hermes_caller() -> Any | None:
        """Locate Hermes' default LLM completion function if available."""
        try:
            from agent.llm import simple_complete  # type: ignore[import-not-found]
            return simple_complete
        except Exception:
            return None


def _serialise_compact(messages: list[dict[str, Any]]) -> str:
    lines = []
    for m in messages:
        lines.append(f"[{m['role']}] {m['content']}")
    return "\n".join(lines)


def _parse_llm_payload(raw: str) -> list[ExtractedMemory]:
    if not raw or not isinstance(raw, str):
        return []
    import json

    text = raw.strip()
    # Strip code fences the model occasionally adds.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("LLM extraction returned non-JSON; ignoring")
        return []

    if not isinstance(data, list):
        return []

    out: list[ExtractedMemory] = []
    valid_categories = {"user_pref", "project", "fact", "decision", "general"}
    for item in data:
        if not isinstance(item, dict):
            continue
        content = (item.get("content") or "").strip()
        if not content:
            continue
        category = item.get("category", "general")
        if category not in valid_categories:
            category = "general"
        ents = item.get("entities") or []
        if not isinstance(ents, list):
            ents = []
        out.append(
            ExtractedMemory(
                content=content[:1000],
                category=category,
                entities=[str(e) for e in ents][:10],
                tags=["auto", "llm"],
            )
        )
    return out
