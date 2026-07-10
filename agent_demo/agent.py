import asyncio
import base64
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from openai import OpenAI
from browser_use import Browser, Tools
import simpleArgParser as sap


@dataclass
class AgentArgs:
    task: str = ""
    url: str = ""
    tasks_file: str = ""
    task_id: str = ""
    out_dir: str = "./runs/agent"
    max_steps: int = 15
    model: str = ""          # agent LLM model id (env: JUDGE_MODEL if unset)
    api_key: str = ""        # env: OPENAI_API_KEY
    api_base: str = ""       # env: OPENAI_BASE_URL
    headless: bool = True     # only used when MW_CDP_URL is unset (CDP takes priority)
    history_window: int = 5
    dom_char_limit: int = 12000
    judge_rubric: str = ""   # carried into trajectory.json for the LLM judge
    verifier_path: str = ""  # carried into trajectory.json for the verifier mode
    container: str = os.environ.get("WH_CONTAINER", "wh-review")

    def post_process(self):
        if self.tasks_file:
            path = Path(self.tasks_file)
            if not path.exists():
                raise SystemExit(f"tasks file not found: {path}")
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if self.task_id:
                row = next((r for r in rows if r.get("id") == self.task_id), None)
                if row is None:
                    raise SystemExit(f"task_id {self.task_id!r} not in {path}")
            else:
                row = rows[0]
            self.task = self.task or row.get("ques", "")
            self.url = self.url or row.get("web", "")
            self.task_id = self.task_id or row.get("id", "")
            self.judge_rubric = row.get("judge_rubric", "")
            self.verifier_path = row.get("verifier_path", "")
        if not self.task or not self.url:
            raise SystemExit("provide --tasks_file [--task_id ID] or both --task and --url")


SYSTEM_PROMPT = """You are a web agent driving a real browser. Each turn you receive:
  1. a DOM tree of the current page, where [N] are clickable element indices;
  2. a screenshot of the same page.

You respond with EXACTLY one JSON object describing one action. No prose, no
fences, no extra text. Schema:

  {"thought": "<short reasoning>", "action": "<name>", "params": { ... }}

Valid actions and their params:
  click       {"index": <int>}
  input       {"index": <int>, "text": "<str>"}
  scroll      {"down": <bool>, "pages": <float>}
  navigate    {"url": "<str>"}
  go_back     {}
  done        {"text": "<final answer or summary>", "success": <bool>}

Rules:
- Use indices that appear in the DOM tree below; do not invent them.
- If the task asks for an answer (e.g. "what is X"), write it in done.text.
- Stop with done as soon as the task is complete or clearly impossible.
"""


def build_messages(task, url, title, dom_text, screenshot_b64, history,
                   history_window, dom_char_limit):
    if len(dom_text) > dom_char_limit:
        dom_text = dom_text[:dom_char_limit] + f"\n... (truncated, full length {len(dom_text)})"

    history_lines = []
    for h in history[-history_window:]:
        history_lines.append(
            f"  step {h['step']}: action={h['action']} params={h.get('params', {})} "
            f"thought={h['thought'][:120]}"
        )
    history_block = "\n".join(history_lines) if history_lines else "  (none)"

    user_text = (
        f"TASK:\n{task}\n\n"
        f"CURRENT URL: {url}\n"
        f"PAGE TITLE: {title}\n\n"
        f"DOM:\n{dom_text}\n\n"
        f"PREVIOUS ACTIONS:\n{history_block}\n\n"
        f"Respond with one JSON action."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
            ],
        },
    ]


def parse_action_json(raw):
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    start = s.find("{")
    if start < 0:
        raise ValueError(f"no JSON in response: {raw[:200]!r}")
    if s[:start].strip():
        raise ValueError("agent reply must contain JSON only")
    def reject_nonfinite(value):
        raise ValueError(f"non-finite JSON number {value!r} is not allowed")

    def reject_duplicate_keys(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} is not allowed")
            value[key] = item
        return value

    try:
        value, end = json.JSONDecoder(
            parse_constant=reject_nonfinite,
            object_pairs_hook=reject_duplicate_keys,
        ).raw_decode(s[start:])
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON in response: {raw[:200]!r}") from exc
    if not isinstance(value, dict):
        raise ValueError("agent reply must be a JSON object")
    if s[start + end:].strip():
        raise ValueError("agent reply must contain exactly one JSON object")
    if set(value) != {"thought", "action", "params"}:
        raise ValueError(
            "agent reply must contain exactly thought, action, and params"
        )
    if not isinstance(value.get("thought"), str):
        raise ValueError("agent reply field 'thought' must be a string")
    action = value.get("action")
    params = value.get("params")
    if not isinstance(action, str) or not action:
        raise ValueError("agent reply field 'action' must be a nonempty string")
    if not isinstance(params, dict):
        raise ValueError("agent reply field 'params' must be an object")

    def exact_params(required):
        if set(params) != set(required):
            raise ValueError(
                f"action {action!r} requires params {sorted(required)!r}"
            )
        for field, expected_type in required.items():
            item = params[field]
            if not isinstance(item, expected_type) or (
                expected_type in {int, float} and isinstance(item, bool)
            ):
                raise ValueError(
                    f"action {action!r} param {field!r} has an invalid type"
                )

    if action == "click":
        exact_params({"index": int})
        if params["index"] < 0:
            raise ValueError("click index must be nonnegative")
    elif action == "input":
        exact_params({"index": int, "text": str})
        if params["index"] < 0:
            raise ValueError("input index must be nonnegative")
    elif action == "scroll":
        exact_params({"down": bool, "pages": (int, float)})
        pages = params["pages"]
        if (isinstance(pages, bool)
                or pages <= 0
                or pages > 100
                or (isinstance(pages, float) and not math.isfinite(pages))):
            raise ValueError("scroll pages must be a positive number no greater than 100")
    elif action == "navigate":
        exact_params({"url": str})
        if not params["url"].strip():
            raise ValueError("navigate url must be nonempty")
    elif action == "go_back":
        exact_params({})
    elif action == "done":
        exact_params({"text": str, "success": bool})
    else:
        raise ValueError(f"unsupported action: {action!r}")
    return value


def openai_client(api_base, api_key):
    """Build an SDK client without letting base-URL queries swallow paths."""
    parsed = urlsplit(api_base)
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    query_keys = [key for key, _ in query_pairs]
    if len(query_keys) != len(set(query_keys)):
        raise ValueError("OPENAI_BASE_URL must not contain duplicate query keys")
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[:-len("/chat/completions")] or "/"
    base_url = urlunsplit(parsed._replace(path=path, query="", fragment=""))
    default_query = dict(query_pairs)
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_query=default_query or None,
    )


def build_action_model(tools, action_name, params):
    AM = tools.registry.create_action_model()
    return AM(**{action_name: params})


async def execute(tools, browser, name, params):
    am = build_action_model(tools, name, params)
    return await tools.act(am, browser_session=browser)


def save_screenshot_b64(b64, path):
    Path(path).write_bytes(base64.b64decode(b64))


def snapshot_verifier_db(verifier_path, container, kind, destination):
    """Capture immutable per-run DB evidence for a site-local verifier."""
    parts = Path(verifier_path).parts
    if len(parts) < 4 or parts[0] != "sites" or parts[2] != "verify":
        return False, ""
    site = parts[1]
    if not re.fullmatch(r"[a-z0-9_]+", site):
        return False, "invalid verifier site name"
    db_dir = f"/opt/WebSyn/{site}/{kind}"
    listing = subprocess.run(
        [
            "docker", "exec", container, "find", db_dir,
            "-maxdepth", "1", "-type", "f", "-name", "*.db", "-print",
        ],
        capture_output=True,
        text=True,
    )
    db_paths = [line.strip() for line in listing.stdout.splitlines() if line.strip()]
    if listing.returncode != 0 or len(db_paths) != 1:
        detail = listing.stderr.strip() or (
            f"expected exactly one database in {db_dir}, found {db_paths!r}"
        )
        return False, detail
    source = f"{container}:{db_paths[0]}"
    result = subprocess.run(
        ["docker", "cp", source, str(destination)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False, result.stderr.strip() or result.stdout.strip()
    destination.chmod(0o444)
    return True, ""


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def run(args):
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    api_base = args.api_base or os.environ.get("OPENAI_BASE_URL", "")
    model = args.model or os.environ.get("JUDGE_MODEL", "")
    if not api_key:
        raise SystemExit("API key missing: set --api_key or OPENAI_API_KEY")
    if not api_base:
        raise SystemExit("API base missing: set --api_base or OPENAI_BASE_URL")
    if not model:
        raise SystemExit("model missing: set --model or JUDGE_MODEL")
    args.model = model

    print(f"task_id={args.task_id or '<inline>'}  url={args.url}\n  ques: {args.task}")

    out = Path(args.out_dir)
    generated_artifacts = [
        out / "trajectory.json",
        out / "initial_state.db",
        out / "after_state.db",
        out / "eval.json",
    ]
    generated_artifacts.extend((out / "screenshots").glob("step_*.png"))
    if any(path.exists() for path in generated_artifacts):
        raise SystemExit(
            f"out_dir {out} contains artifacts from an earlier run; "
            "choose a new or cleaned output directory"
        )
    out.mkdir(parents=True, exist_ok=True)
    shots = out / "screenshots"
    shots.mkdir(exist_ok=True)

    client = openai_client(api_base, api_key)

    # Connect to an externally-launched headless Chrome via CDP when MW_CDP_URL is
    # set. browser-use's own Browser() launcher can hang on some hosts (watchdog
    # timeout on BrowserStartEvent); a persistent CDP chrome avoids that and lets
    # many agents run concurrently with cookie isolation (one chrome per agent).
    cdp_url = os.environ.get("MW_CDP_URL", "")
    if cdp_url:
        browser = Browser(cdp_url=cdp_url)
    else:
        browser = Browser(headless=args.headless)
    await browser.start()
    tools = Tools()

    try:
        await browser.navigate_to(args.url)

        state = await browser.get_browser_state_summary(include_screenshot=True)
        save_screenshot_b64(state.screenshot, shots / "step_000.png")

        trajectory = {
            "task": args.task,
            "task_id": args.task_id,
            "start_url": args.url,
            "model": args.model,
            "max_steps": args.max_steps,
            "steps": [],
            "terminated": False,
            "termination_reason": None,
            "final_answer": None,
            "judge_rubric": args.judge_rubric,
            "verifier_path": args.verifier_path,
        }
        initial_db_path = out / "initial_state.db"
        initial_ok, initial_error = snapshot_verifier_db(
            args.verifier_path, args.container, "instance", initial_db_path
        )
        if initial_ok:
            trajectory["initial_db_snapshot"] = initial_db_path.name
            trajectory["initial_db_sha256"] = file_sha256(initial_db_path)
        elif args.verifier_path:
            trajectory["initial_db_snapshot_error"] = initial_error

        for step_idx in range(args.max_steps):
            dom_text = state.dom_state.llm_representation()
            messages = build_messages(
                task=args.task,
                url=state.url,
                title=state.title,
                dom_text=dom_text,
                screenshot_b64=state.screenshot,
                history=trajectory["steps"],
                history_window=args.history_window,
                dom_char_limit=args.dom_char_limit,
            )

            resp = client.chat.completions.create(model=args.model, messages=messages)
            raw = resp.choices[0].message.content or ""
            try:
                action = parse_action_json(raw)
            except Exception as e:
                print(f"[step {step_idx}] FAILED to parse LLM reply: {e}", file=sys.stderr)
                trajectory["terminated"] = True
                trajectory["termination_reason"] = f"parse_error: {e}"
                trajectory["raw_reply"] = raw
                break

            name = action.get("action", "")
            params = action.get("params", {}) or {}
            thought = action.get("thought", "")

            step_log = {
                "step": step_idx,
                "url": state.url,
                "title": state.title,
                "thought": thought,
                "action": name,
                "params": params,
                "screenshot_before": f"step_{step_idx:03d}.png",
                "screenshot_after": f"step_{step_idx + 1:03d}.png",
            }
            trajectory["steps"].append(step_log)
            print(f"[step {step_idx}] {name} {params}  // {thought[:80]}")

            if name == "done":
                trajectory["terminated"] = True
                trajectory["termination_reason"] = "agent_done"
                trajectory["final_answer"] = params.get("text", "")
                trajectory["success_self_report"] = bool(params.get("success", False))
                final_state = await browser.get_browser_state_summary(include_screenshot=True)
                save_screenshot_b64(final_state.screenshot, shots / f"step_{step_idx + 1:03d}.png")
                break

            try:
                result = await execute(tools, browser, name, params)
                step_log["action_result"] = {
                    "is_done": getattr(result, "is_done", None),
                    "success": getattr(result, "success", None),
                    "error": getattr(result, "error", None),
                    "extracted_content": (getattr(result, "extracted_content", None) or ""),
                }
            except Exception as e:
                step_log["action_result"] = {"error": f"{type(e).__name__}: {e}"}
                print(f"[step {step_idx}] action failed: {e}", file=sys.stderr)

            state = await browser.get_browser_state_summary(include_screenshot=True)
            save_screenshot_b64(state.screenshot, shots / f"step_{step_idx + 1:03d}.png")
        else:
            trajectory["termination_reason"] = "max_steps"

        after_db_path = out / "after_state.db"
        after_ok, after_error = snapshot_verifier_db(
            args.verifier_path, args.container, "instance", after_db_path
        )
        if after_ok:
            trajectory["after_db_snapshot"] = after_db_path.name
            trajectory["after_db_sha256"] = file_sha256(after_db_path)
        elif args.verifier_path:
            trajectory["after_db_snapshot_error"] = after_error

        traj_path = out / "trajectory.json"
        traj_path.write_text(json.dumps(trajectory, indent=2))
        print(f"\nWrote {traj_path}")
        print(f"  steps: {len(trajectory['steps'])}")
        print(f"  screenshots: {len(list(shots.glob('step_*.png')))}")
        print(f"  terminated: {trajectory['terminated']} ({trajectory['termination_reason']})")
        if trajectory.get("final_answer"):
            print(f"  final_answer: {trajectory['final_answer'][:300]}")

        return trajectory
    finally:
        await browser.stop()


def main():
    args = sap.parse_args(AgentArgs)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
