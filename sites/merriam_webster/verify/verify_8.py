#!/usr/bin/env python3
"""Deterministic verifier for MW task Merriam-Webster--8.

thesaurus happy; count synonyms and antonyms.

Checks (deterministic first; LLM utilities anchored on ground truth):
nav /thesaurus/happy | answer contains '8 synonyms' AND '8 antonyms' (deterministic) | screenshot shows lists
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
    j = Judge('Merriam-Webster--8', a.no_llm)
    t = load_run(a.run_dir)
    fa = final_answer(t)
    j.check("nav_thesaurus_happy", navigated_to(t, "/thesaurus/happy"),
            f"navigated={navigated_to(t, '/thesaurus/happy')}")
    # Bind every reported count to its own label; unrelated totals (for
    # example "16 total") and prose numbering are allowed.
    fa_low = fa.casefold()
    count = r"(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)"
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9", "ten": "10",
    }

    def labeled_counts(label):
        matches = re.findall(
            rf"\b({count})\s+{label}s?\b|\b{label}s?"
            rf"(?:\s+count)?\s*(?:is\s+|[:=-]\s*)?({count})\b",
            fa_low,
        )
        return [number_words.get(before or after, before or after)
                for before, after in matches]

    synonym_counts = labeled_counts("synonym")
    antonym_counts = labeled_counts("antonym")
    synonym_pattern = (
        r"(?:8|eight)\s+synonyms?|synonyms?\s*[:=-]?\s*(?:8|eight)"
    )
    antonym_pattern = (
        r"(?:8|eight)\s+antonyms?|antonyms?\s*[:=-]?\s*(?:8|eight)"
    )
    j.check("answer_counts", answer_is_affirmative(fa)
            and bool(synonym_counts and antonym_counts)
            and set(synonym_counts) == {"8"}
            and set(antonym_counts) == {"8"}
            and fact_is_asserted(fa, synonym_pattern)
            and fact_is_asserted(fa, antonym_pattern)
            and not fact_is_negated(fa, synonym_pattern)
            and not fact_is_negated(fa, antonym_pattern),
            f"final={fa!r}")
    s = last_shot(t)
    if s:
        ok, ev = llm_screenshot_shows(s, "synonyms", "how many synonyms and antonyms of happy are listed")
        j.check("screenshot_shows_lists", ok, ev, llm=True)
    j.emit()

if __name__ == "__main__":
    main()
