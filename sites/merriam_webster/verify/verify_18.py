#!/usr/bin/env python3
"""Deterministic verifier for MW task Merriam-Webster--18.

Login alice + remove 'curiosity'; confirm list == harmony, eloquent.

Checks (deterministic first; LLM utilities anchored on ground truth):
nav /login,/account | DB after: alice saved words exactly [eloquent, harmony] (no curiosity) | screenshot shows account page
Input/Output: see verify_lib.parse_args / Judge.emit.
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_lib import (load_run, navigated_to, navigated_in_order, finished_at, final_answer, last_shot, shot_after_url,
                        contains_all, contains_any, answer_equals, extract_years,
                        extract_score, resolve_db, saved_words_for, user_exists,
                        answer_is_affirmative, fact_is_asserted, fact_is_negated, norm, llm_text_match,
                        llm_screenshot_shows, Judge, parse_args)

def main():
    a = parse_args()
    j = Judge('Merriam-Webster--18', a.no_llm)
    t = load_run(a.run_dir)
    fa = final_answer(t)
    normalized = norm(fa)
    removal_pattern = (
        r"\b(?:removed|gone|absent)\b|"
        r"\bno\s+longer\s+(?:saved|present|listed)\b"
    )
    removal = re.search(removal_pattern, normalized)
    extra_words = re.search(
        r"\b(?:plus|along\s+with|as\s+well\s+as)\s+"
        r"(?:other|additional|more)\s+words\b|"
        r"\b(?:other|additional|more)\s+words\s+(?:remain|are|exist)\b|"
        r"\b(?:more\s+than|at\s+least)\s+(?:2|two)\s+words\b",
        normalized,
    )
    j.check("final_confirms_removal", answer_is_affirmative(fa)
            and all(word in normalized for word in ["curiosity", "harmony", "eloquent"])
            and bool(removal)
            and not extra_words
            and fact_is_asserted(fa, removal_pattern)
            and not fact_is_negated(fa, r"\b(?:removed|gone|absent)\b"),
            f"final={fa!r}")
    j.check("nav_login", navigated_to(t, "/login"), f"navigated={navigated_to(t, '/login')}")
    j.check("nav_account", navigated_to(t, "/account"), f"navigated={navigated_to(t, '/account')}")
    j.check("nav_remove_flow_in_order", navigated_in_order(t, ["/login", "/account"]),
            "login and account pages visited in order")
    j.check("finished_on_account", finished_at(t, "/account"),
            "final action occurred on the account page")
    after = resolve_db(a.after_db, a.container, "instance")
    init = resolve_db(a.initial_db, a.container, "instance_seed")
    aw = saved_words_for(after)
    iw = saved_words_for(init)
    j.check("db_curiosity_present_initial", iw is not None and "curiosity" in (iw or []),
            f"initial_saved={iw}")
    j.check("db_curiosity_removed", aw is not None and "curiosity" not in (aw or []), f"after_saved={aw}")
    j.check("db_list_exactly_2", aw is not None and sorted(aw or []) == ["eloquent", "harmony"], f"after_saved={aw}")
    s = last_shot(t)
    if s:
        ok, ev = llm_screenshot_shows(s, "harmony", "saved-words list showing only harmony and eloquent")
        j.check("screenshot_shows_list", ok, ev, llm=True)
    j.emit()

if __name__ == "__main__":
    main()
