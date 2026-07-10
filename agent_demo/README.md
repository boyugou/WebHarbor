# agent_demo

Minimal ReAct loop (`agent.py`) + LLM-as-judge grader (`eval_judge.py`) for driving and evaluating an agent on any WebHarbor mirror.

## Setup

```bash
cd agent_demo
uv sync                                  # installs deps into .venv
uv run playwright install chromium       # one-time browser download
```

API key + base URL come from env vars (do **not** hardcode them):

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1   # OpenAI-compatible endpoint
export JUDGE_MODEL=gpt-5.1                         # agent + both graders
export WH_CONTAINER=wh-review                      # running WebHarbor container
```

## Run a task

WebHarbor must already be running locally (for example,
`docker run -d --name wh-review -p 8101:8101 -p 40000-40016:40000-40016 battalion7244/webharbor:latest`).

Run a single task from a site's `tasks.jsonl`:

```bash
uv run python agent.py \
  --tasks_file ../sites/google_search/tasks.jsonl \
  --task_id "Google Search--0" \
  --out_dir runs/gs0
```

Omit `--task_id` to pick the first row. Or run an ad-hoc task:

```bash
uv run python agent.py \
  --task "Find Kevin Durant's bio" \
  --url http://localhost:40009/ \
  --out_dir runs/inline
```

Each run writes `trajectory.json` + `screenshots/step_NNN.png` under `--out_dir`.
Use a fresh output directory for every run; the agent fails closed if generated
artifacts from an earlier run are already present.
When a task has a site-local verifier, the agent also copies immutable
`initial_state.db` and `after_state.db` snapshots from `WH_CONTAINER`; stateful
or server-proof-backed grading fails closed if either artifact or its recorded
SHA-256 digest is missing. The initial snapshot is the live DB immediately
before browser actions, not the seed DB.

## Grade a run

Run the deterministic verifier first when the task supplies `verifier_path`; it is
the primary binary grader. Then run the LLM judge as the secondary grader:

```bash
uv run python eval_judge.py --run_dir runs/gs0 --verifier True
uv run python eval_judge.py --run_dir runs/gs0
```

Each command writes `eval.json` next to the trajectory. Preserve or copy the first
result if you need both artifacts, because the second command replaces the file.
The verifier exits 0/1 and emits deterministic `pass`, `reason`, and `evidence`
fields. Verifier mode always disables its optional LLM helpers and requires the
trajectory's step screenshots, so credentials or missing artifacts cannot change
the primary result. The LLM judge emits `success`, `confidence`, `rationale`, and
`evidence`.
