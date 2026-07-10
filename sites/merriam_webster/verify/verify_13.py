#!/usr/bin/env python3
"""Deterministic verifier for MW task Merriam-Webster--13.

Games & Quizzes page: count quizzes + difficulty of each.

Checks (deterministic first; LLM utilities anchored on ground truth):
nav /games-quizzes | answer contains '3', 3 titles, 'easy', 'medium' (deterministic) | screenshot shows index
Input/Output: see verify_lib.parse_args / Judge.emit.
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_lib import (load_run, navigated_to, final_answer, last_shot, shot_after_url,
                        contains_all, contains_any, answer_equals, extract_years,
                        extract_score, resolve_db, saved_words_for, user_exists,
                        answer_is_affirmative, fact_is_asserted, fact_is_negated, llm_text_match,
                        llm_screenshot_shows, Judge, parse_args)

def main():
    a = parse_args()
    j = Judge('Merriam-Webster--13', a.no_llm)
    t = load_run(a.run_dir)
    fa = final_answer(t)
    j.check("nav_games", navigated_to(t, "/games-quizzes"),
            f"navigated={navigated_to(t, '/games-quizzes')}")
    normalized = " ".join(fa.casefold().split())
    count_tokens = re.findall(
        r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)"
        r"\s+quizzes?\b|\bquizzes?\s*[:=-]?\s*"
        r"(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        normalized,
    )
    count_words = {"three": "3"}
    quiz_counts = [count_words.get(before or after, before or after)
                   for before, after in count_tokens]
    count_ok = bool(quiz_counts) and set(quiz_counts) == {"3"}
    mappings = [
        ("name that word", "easy"),
        ("synonym showdown", "medium"),
        ("opposites attract", "medium"),
    ]
    mapping_patterns = [
        rf"{re.escape(title)}\s*(?:(?:[-:—(]|\bis\b)\s*|"
        rf"\bdifficulty\s*[:=-]?\s*)?\b{difficulty}\b"
        for title, difficulty in mappings
    ]
    mappings_ok = all(
        re.search(pattern, normalized)
        and fact_is_asserted(fa, pattern)
        and not fact_is_negated(fa, pattern)
        for pattern in mapping_patterns
    )
    levels = re.findall(r"\b(?:easy|medium|hard|advanced)\b", normalized)
    j.check("answer_count_and_levels", answer_is_affirmative(fa)
            and count_ok and mappings_ok
            and fact_is_asserted(
                fa, r"(?:3|three)\s+quizzes?|quizzes?\s*[:=-]?\s*(?:3|three)"
            )
            and not fact_is_negated(
                fa, r"(?:3|three)\s+quizzes?|quizzes?\s*[:=-]?\s*(?:3|three)"
            )
            and sorted(levels) == ["easy", "medium", "medium"], f"final={fa!r}")
    s = last_shot(t)
    if s:
        ok, ev = llm_screenshot_shows(s, "Name That Word", "list of quizzes and their difficulty levels")
        j.check("screenshot_shows_index", ok, ev, llm=True)
    j.emit()

if __name__ == "__main__":
    main()
