#!/usr/bin/env python3
"""Deterministic verifier for MW task Merriam-Webster--17.

Compare empathy/nostalgia/optimism: which has most recent first-known-use.

Checks (deterministic first; LLM utilities anchored on ground truth):
nav all 3 /dictionary/{empathy,nostalgia,optimism} | answer contains 'empathy' + years 1909,1756,1759 (deterministic) | LLM-anchored 'most recent' judgement
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
    j = Judge('Merriam-Webster--17', a.no_llm)
    t = load_run(a.run_dir)
    fa = final_answer(t)
    for w in ["empathy", "nostalgia", "optimism"]:
        j.check(f"nav_{w}", navigated_to(t, "/dictionary/" + w),
                f"navigated={navigated_to(t, '/dictionary/' + w)}")
    normalized = norm(fa)
    expected = {"empathy": "1909", "nostalgia": "1756", "optimism": "1759"}
    mapping_patterns = [
        rf"(?:\b{word}\b.{{0,40}}\b{year}\b|\b{year}\b.{{0,40}}\b{word}\b)"
        for word, year in expected.items()
    ]
    mapped = all(
        re.search(pattern, normalized) and fact_is_asserted(fa, pattern)
        for pattern in mapping_patterns
    )
    years = extract_years(fa)
    recent_pattern = (
        r"(?:\bempathy\b[^.!?]{0,60}\b(?:is|was)\s+(?:the\s+)?"
        r"(?:most recent|latest|newest)\b|"
        r"\b(?:most recent|latest|newest)\b\s+(?:word\s+)?"
        r"(?:is|was)\s+\bempathy\b)"
    )
    recent = bool(re.search(recent_pattern, normalized))
    recent_denied = bool(re.search(
        r"(?:\bempathy\b\s+(?:is|was)\s+not\s+(?:the\s+)?"
        r"(?:most recent|latest|newest)\b|"
        r"\b(?:most recent|latest|newest)\b\s+(?:is|was)\s+not\s+"
        r"\bempathy\b)",
        normalized,
    ))
    competing_recent = bool(re.search(
        r"(?:\b(?:nostalgia|optimism)\b[^.!?]{0,60}\b(?:is|was)\s+"
        r"(?:actually\s+)?(?:the\s+)?(?:most recent|latest|newest)\b|"
        r"\b(?:most recent|latest|newest)\b\s+(?:word\s+)?(?:is|was)\s+"
        r"(?:actually\s+)?\b(?:nostalgia|optimism)\b)",
        normalized,
    ))
    j.check("answer_has_3_mapped_years", answer_is_affirmative(fa) and mapped
            and sorted(years) == ["1756", "1759", "1909"] and recent
            and fact_is_asserted(fa, recent_pattern)
            and not any(fact_is_negated(fa, rf"\b{year}\b")
                        for year in expected.values())
            and not recent_denied and not competing_recent
            and not fact_is_negated(fa, recent_pattern),
            f"final={fa!r}")
    ok, ev = llm_text_match(fa,
        "empathy is the most recent. empathy: 1909; nostalgia: 1756; optimism: 1759.",
        "Among empathy, nostalgia, optimism, which has the most recent first known use (report all 3 years)?")
    j.check("answer_most_recent_empathy", ok, ev, llm=True)
    j.emit()

if __name__ == "__main__":
    main()
