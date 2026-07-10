#!/usr/bin/env python3
"""Deterministic verifier for MW task Merriam-Webster--2.

Look up nostalgia; which Greek word its etymology traces to.

Checks (deterministic first; LLM utilities anchored on ground truth):
nav /dictionary/nostalgia | answer contains 'Greek nóstos' + 'return, homecoming' (deterministic) | screenshot shows etymology
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
    j = Judge('Merriam-Webster--2', a.no_llm)
    t = load_run(a.run_dir)
    fa = final_answer(t)
    j.check("nav_nostalgia", navigated_to(t, "/dictionary/nostalgia"),
            f"navigated={navigated_to(t, '/dictionary/nostalgia')}")
    # answer must name the Greek source word + its meaning; phrasing varies (agent may
    # drop the literal word "Greek"), so check the key tokens separately, not contiguously.
    exact_quote = re.search(r"nóstos\s*[\"“']return,\s*homecoming[\"”']", fa, re.IGNORECASE)
    j.check("answer_has_greek_nostos", answer_is_affirmative(fa) and bool(exact_quote)
            and fact_is_asserted(fa, r"nóstos\s*[\"“']return,\s*homecoming[\"”']")
            and not fact_is_negated(fa, r"\bnóstos\b"),
            f"final={fa!r}")
    s = last_shot(t)
    if s:
        ok, ev = llm_screenshot_shows(s, "Greek nóstos", "etymology of nostalgia")
        j.check("screenshot_shows_etymology", ok, ev, llm=True)
    j.emit()

if __name__ == "__main__":
    main()
