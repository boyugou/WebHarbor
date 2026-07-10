#!/usr/bin/env python3
"""Deterministic verifier for MW task Merriam-Webster--9.

Word of the Day featured word + part of speech (deterministic, pinned).

Checks (deterministic first; LLM utilities anchored on ground truth):
nav /word-of-the-day | answer contains 'ambivalent' AND 'adjective' (deterministic) | screenshot shows ambivalent adjective
Input/Output: see verify_lib.parse_args / Judge.emit.
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_lib import (load_run, navigated_to, final_answer, last_shot, shot_after_url,
                        contains_all, contains_any, answer_equals, extract_years,
                        extract_score, resolve_db, saved_words_for, user_exists,
                        answer_is_affirmative, norm, fact_is_asserted, fact_is_negated, llm_text_match,
                        llm_screenshot_shows, Judge, parse_args)

def main():
    a = parse_args()
    j = Judge('Merriam-Webster--9', a.no_llm)
    t = load_run(a.run_dir)
    fa = final_answer(t)
    j.check("nav_wotd", navigated_to(t, "/word-of-the-day"),
            f"navigated={navigated_to(t, '/word-of-the-day')}")
    normalized = norm(fa)
    mapping_pattern = (
        r"(?:\bambivalent\b\s*(?:(?:[-—:,(]|\bis\b|\bwas\b|"
        r"\bwhich\s+is\b)\s*)*(?:(?:the\s+)?"
        r"(?:part\s+of\s+speech|pos)\s*[:=-]?\s*)?"
        r"(?:an?\s+)?\badjective\b|"
        r"\bambivalent\b\s*[.;,]\s*(?:it\s+is\s+|its\s+)?"
        r"(?:(?:part\s+of\s+speech|pos)\s*(?:is|[:=-])\s*)?"
        r"(?:an?\s+)?\badjective\b)"
    )
    mapping = re.search(mapping_pattern, normalized)
    wrong_pos = re.search(
        r"\b(?:noun|verb|adverb)\b",
        normalized,
    )
    j.check("answer_word_pos", answer_is_affirmative(fa) and bool(mapping)
            and fact_is_asserted(fa, mapping_pattern)
            and not fact_is_negated(fa, r"\badjective\b")
            and (not wrong_pos or fact_is_negated(
                fa, r"\b(?:noun|verb|adverb)\b"
            )), f"final={fa!r}")
    s = last_shot(t)
    if s:
        ok, ev = llm_screenshot_shows(s, "ambivalent", "featured word of the day and its part of speech")
        j.check("screenshot_shows_wotd", ok, ev, llm=True)
    j.emit()

if __name__ == "__main__":
    main()
