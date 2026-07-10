# Merriam-Webster deterministic verifiers

This directory is the reference implementation of WebHarbor's task grading
contract. `verify_0.py` through `verify_19.py` correspond one-to-one with the
rows in `../tasks.jsonl`; `verify_lib.py` contains the shared trajectory, URL,
SQLite, answer-matching, screenshot, and judge helpers.

## Contract

Every verifier:

- reads `trajectory.json` and its screenshots from `--run_dir`;
- requires navigation to the exact requested path on the local mirror;
- checks the final answer or the post-task SQLite state deterministically;
- may expose optional ground-truth-anchored LLM diagnostics, but the primary
  `eval_judge.py --verifier True` path always disables them;
- prints JSON `{task_id, pass, reason, evidence}` and exits 0 on PASS or 1 on
  FAIL.

Stateful verifiers accept `--initial_db` and `--after_db`. `agent.py` captures
those immutable snapshots into each run directory, and unified verifier mode
requires them for state-changing and server-proof-backed tasks. The initial
snapshot is the live pre-action DB. For direct debugging only, omitted paths are
copied from `instance_seed/merriam_webster.db` and
`instance/merriam_webster.db` in `--container` (default `wh-review`).

## Run

Use the unified grader entry point from the repository root:

```bash
uv run python agent_demo/eval_judge.py --run_dir runs/mw-task --verifier True
```

For direct verifier debugging from `agent_demo/`:

```bash
uv run python ../sites/merriam_webster/verify/verify_0.py \
  --run_dir ../runs/mw-task --no_llm True
```

Boolean options use explicit values. The optional LLM helpers read the same
`OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `JUDGE_MODEL` variables as the agent and
LLM judge. The primary verifier also requires at least one `step_*.png` artifact
in the run's `screenshots/` directory.

Before accepting a verifier, exercise a correct run plus empty-answer,
wrong-answer, correct-answer-without-navigation, and (for stateful tasks)
unchanged-database counterexamples. Only the correct run may pass.
