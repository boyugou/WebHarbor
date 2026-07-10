#!/usr/bin/env python3
"""Deterministic verifier for MW task Merriam-Webster--16.

Register 'Jordan Lee' + verify logged in.

Checks (deterministic first; LLM utilities anchored on ground truth):
register -> protected account page and finish there | immutable DB transition adds Jordan Lee | optional screenshot diagnostic
Input/Output: see verify_lib.parse_args / Judge.emit.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_lib import (load_run, navigated_to, navigated_in_order, finished_at, authenticated_user_proof, final_answer, last_shot, shot_after_url,
                        contains_all, contains_any, answer_equals, extract_years,
                        extract_score, resolve_db, saved_words_for, user_exists,
                        answer_confirms_action, llm_text_match,
                        llm_screenshot_shows, Judge, parse_args)

def main():
    a = parse_args()
    j = Judge('Merriam-Webster--16', a.no_llm)
    t = load_run(a.run_dir)
    fa = final_answer(t)
    registered = answer_confirms_action(
        fa, ["Jordan Lee"], r"\b(?:registered|created)\b"
    )
    logged_in = answer_confirms_action(
        fa, ["Jordan Lee"], r"\b(?:logged|signed)\s+in\b"
    )
    j.check("final_confirms_registration_and_login", registered and logged_in,
            f"final={fa!r}")
    j.check("nav_register", navigated_to(t, "/register"), f"navigated={navigated_to(t, '/register')}")
    j.check("nav_registered_account", navigated_in_order(
        t, ["/register", "/account"]
    ), "registration was followed by the authenticated account page")
    j.check("finished_authenticated", finished_at(t, "/account"),
            "final action occurred on the login-protected account page")
    after = resolve_db(a.after_db, a.container, "instance")
    init = resolve_db(a.initial_db, a.container, "instance_seed")
    exists = user_exists(after, name="Jordan Lee")
    existed_initially = user_exists(init, name="Jordan Lee")
    j.check("db_jordan_lee_registered", exists is True, f"user_exists_Jordan_Lee={exists}")
    j.check("db_jordan_lee_absent_initial", existed_initially is False,
            f"initial_user_exists_Jordan_Lee={existed_initially}")
    j.check("auth_proof_jordan_lee", authenticated_user_proof(
        t, "/account", init, after, "Jordan Lee"
    ), "final account proof belongs to Jordan Lee")
    s = last_shot(t)
    if s:
        ok, ev = llm_screenshot_shows(s, "My Words", "navigation showing the user is logged in (My Words / Log Out)")
        j.check("screenshot_shows_logged_in", ok, ev, llm=True)
    j.emit()

if __name__ == "__main__":
    main()
