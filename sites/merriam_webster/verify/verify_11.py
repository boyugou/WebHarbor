#!/usr/bin/env python3
"""Deterministic verifier for MW task Merriam-Webster--11.

Name That Word quiz: answer all + report final score.

Checks (deterministic first; LLM utilities anchored on ground truth):
ordered games/quiz/result navigation | reported score matches the server-issued result URL | optional screenshot diagnostic
Input/Output: see verify_lib.parse_args / Judge.emit.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_lib import (load_run, navigated_to, navigated_in_order, final_answer, last_shot, shot_after_url,
                        contains_all, contains_any, answer_equals, extract_years,
                        extract_score, quiz_result_score, resolve_db, saved_words_for, user_exists,
                        llm_text_match, llm_screenshot_shows, Judge, parse_args)

def main():
    a = parse_args()
    j = Judge('Merriam-Webster--11', a.no_llm)
    t = load_run(a.run_dir)
    fa = final_answer(t)
    j.check("nav_games", navigated_to(t, "/games-quizzes"),
            f"navigated={navigated_to(t, '/games-quizzes')}")
    j.check("nav_quiz", navigated_to(t, "/quiz/name-that-word"),
            f"navigated={navigated_to(t, '/quiz/name-that-word')}")
    j.check("reached_result_page", navigated_to(t, "/quiz/name-that-word/submit"),
            f"navigated_submit={navigated_to(t, '/quiz/name-that-word/submit')}")
    j.check("nav_quiz_flow_in_order", navigated_in_order(t, [
        "/games-quizzes", "/quiz/name-that-word", "/quiz/name-that-word/submit",
    ]), "games, quiz, and result pages visited in order")
    score = extract_score(fa)
    initial = resolve_db(a.initial_db, a.container, "instance_seed")
    after = resolve_db(a.after_db, a.container, "instance")
    issued_score = quiz_result_score(
        t, "/quiz/name-that-word/submit", "name-that-word", initial, after
    )
    j.check("reported_score_X_over_10", score is not None
            and score == issued_score,
            f"final={fa!r} issued_score={issued_score!r}")
    s = shot_after_url(t, "/quiz/name-that-word/submit") or last_shot(t)
    if s and score is not None:
        ok, ev = llm_screenshot_shows(s, "Your Score " + score + " / 10",
            "the agent's final quiz score on the result page")
        j.check("screenshot_shows_reported_score", ok, ev, llm=True)
    elif s:
        ok, ev = llm_screenshot_shows(s, "Your Score", "quiz result page score")
        j.check("screenshot_shows_score_page", ok, ev, llm=True)
    else:
        j.check("screenshot_shows_score_page", False, "no screenshots in run")

    j.emit()

if __name__ == "__main__":
    main()
