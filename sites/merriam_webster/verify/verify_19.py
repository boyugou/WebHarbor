#!/usr/bin/env python3
"""Deterministic verifier for MW task Merriam-Webster--19.

Compare gregarious vs benevolent: which entered English earlier + POS of each.

Checks (deterministic first; LLM utilities anchored on ground truth):
nav /dictionary/gregarious AND /dictionary/benevolent | answer identifies benevolent as earlier and both entries as adjectives
Input/Output: see verify_lib.parse_args / Judge.emit.
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_lib import (load_run, navigated_to, final_answer, last_shot, shot_after_url,
                        contains_all, contains_any, answer_equals, extract_years,
                        extract_score, resolve_db, saved_words_for, user_exists,
                        answer_is_affirmative, fact_is_asserted, fact_is_negated, norm, llm_text_match,
                        llm_screenshot_shows, Judge, parse_args)

def main():
    a = parse_args()
    j = Judge('Merriam-Webster--19', a.no_llm)
    t = load_run(a.run_dir)
    fa = final_answer(t)
    j.check("nav_gregarious", navigated_to(t, "/dictionary/gregarious"),
            f"navigated={navigated_to(t, '/dictionary/gregarious')}")
    j.check("nav_benevolent", navigated_to(t, "/dictionary/benevolent"),
            f"navigated={navigated_to(t, '/dictionary/benevolent')}")
    normalized = norm(fa)
    earlier_pattern = (
        r"(?:\bbenevolent\b[^.;!?]{0,60}(?:"
        r"\b(?:entered|came|was|is)\b[^.;!?]{0,20}\b(?:earlier|oldest)\b|"
        r"\bpredates?\b[^.;!?]{0,20}\bgregarious\b|"
        r"\b(?:entered|came)\s+first\b)|"
        r"\b(?:earlier|oldest)\s+(?:word\s+)?(?:is|was)\s+"
        r"\bbenevolent\b)"
    )
    earlier = re.search(earlier_pattern, normalized)
    earlier_denied = re.search(
        r"(?:\bbenevolent\b\s+(?:is|was|entered)\s+not\s+"
        r"(?:the\s+)?(?:earlier|oldest|first)\b|"
        r"\b(?:earlier|oldest|first)\b\s+(?:is|was)\s+not\s+"
        r"\bbenevolent\b)",
        normalized,
    )
    competing_earlier = re.search(
        r"(?:\bgregarious\b[^.;!?]{0,60}(?:"
        r"\b(?:entered|came|was|is)\b[^.;!?]{0,20}\b(?:earlier|oldest)\b|"
        r"\bpredates?\b[^.;!?]{0,20}\bbenevolent\b|"
        r"\b(?:entered|came)\s+first\b)|"
        r"\b(?:earlier|oldest)\s+(?:word\s+)?(?:is|was)\s+\bgregarious\b)",
        normalized,
    )
    parts_of_speech = (
        re.search(r"\bboth\b.{0,30}\badjectives\b", normalized)
        or (
            re.search(r"\bgregarious\b.{0,40}\badjective\b", normalized)
            and re.search(r"\bbenevolent\b.{0,40}\badjective\b", normalized)
        )
    )
    j.check("answer_has_facts", answer_is_affirmative(fa)
            and bool(earlier)
            and fact_is_asserted(fa, earlier_pattern)
            and fact_is_asserted(fa, r"\badjectives?\b")
            and not fact_is_negated(fa, r"\badjectives?\b")
            and not earlier_denied
            and not competing_earlier
            and not fact_is_negated(fa, earlier_pattern)
            and bool(parts_of_speech),
            f"final={fa!r}")
    ok, ev = llm_text_match(fa,
        "benevolent entered English earlier. benevolent: 15th century (adjective); gregarious: 1668 (adjective). Both are adjectives.",
        "Comparing gregarious and benevolent: which entered English earlier, and what part of speech is each?")
    j.check("answer_benevolent_earlier", ok, ev, llm=True)
    j.emit()

if __name__ == "__main__":
    main()
