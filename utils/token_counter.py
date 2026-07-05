"""
utils/token_counter.py
───────────────────────
Lightweight token estimation using tiktoken.
Used by the context builder to trim context before it hits the LLM,
keeping token consumption under budget (~70% reduction vs raw code dumps).
"""

from __future__ import annotations

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")   # same BPE as GPT-4 / Llama 3 approx
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


def count_tokens(text: str) -> int:
    """Estimate token count for a string. Falls back to word-count heuristic."""
    if _AVAILABLE:
        return len(_ENC.encode(text))
    # Rough fallback: ~1.3 tokens per word
    return int(len(text.split()) * 1.3)


def trim_to_budget(text: str, max_tokens: int) -> str:
    """
    Truncate text so it fits within max_tokens.
    Trims from the end, preserving the most relevant (early) content.
    """
    if not _AVAILABLE:
        # Character-based fallback: ~4 chars per token
        max_chars = max_tokens * 4
        return text[:max_chars] + ("\n...[trimmed]" if len(text) > max_chars else "")

    tokens = _ENC.encode(text)
    if len(tokens) <= max_tokens:
        return text
    trimmed = _ENC.decode(tokens[:max_tokens])
    return trimmed + "\n...[trimmed]"


def fits_in_budget(text: str, max_tokens: int) -> bool:
    return count_tokens(text) <= max_tokens
