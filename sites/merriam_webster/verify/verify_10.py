#!/usr/bin/env python3
"""Deterministic verifier for MW task Merriam-Webster--10.

WOTD ephemeral; Did You Know? etymology text.

Checks (deterministic first; LLM utilities anchored on ground truth):
nav /word-of-the-day/ephemeral | answer contains 'greek ephēmeros' (deterministic) | screenshot shows Did You Know
Input/Output: see verify_lib.parse_args / Judge.emit.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_lib import (load_run, navigated_to, navigated_in_order, final_answer, last_shot, shot_after_url,
                        contains_all, contains_any, answer_equals, extract_years,
                        extract_score, resolve_db, saved_words_for, user_exists,
                        answer_contains_exact_fact, llm_text_match,
                        llm_screenshot_shows, Judge, parse_args)

def main():
    a = parse_args()
    j = Judge('Merriam-Webster--10', a.no_llm)
    t = load_run(a.run_dir)
    fa = final_answer(t)
    j.check("nav_wotd", navigated_to(t, "/word-of-the-day"),
            f"navigated={navigated_to(t, '/word-of-the-day')}")
    j.check("nav_wotd_ephemeral", navigated_to(t, "/word-of-the-day/ephemeral"),
            f"navigated={navigated_to(t, '/word-of-the-day/ephemeral')}")
    j.check("nav_wotd_then_ephemeral", navigated_in_order(
        t, ["/word-of-the-day", "/word-of-the-day/ephemeral"]
    ), "required pages visited in order")
    expected = "Greek eph\u0113meros lasting a day, daily, from epi- + h\u0113mera day"
    j.check("answer_did_you_know", answer_contains_exact_fact(fa, expected),
            f"final={fa!r}")
    s = last_shot(t)
    if s:
        ok, ev = llm_screenshot_shows(s, "Greek eph\u0113meros", "Did You Know section of the ephemeral WOTD entry")
        j.check("screenshot_shows_dyk", ok, ev, llm=True)
    j.emit()

if __name__ == "__main__":
    main()
