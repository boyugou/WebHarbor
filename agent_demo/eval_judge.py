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
import simpleArgParser as sap


@dataclass
class JudgeArgs:
    run_dir: str = ""
    out: str = ""
    last_k_screenshots: int = 4
    model: str = ""
    api_key: str = ""
    api_base: str = ""
    # Grading mode (simpleargparser bool -> value flag: --verifier True).
    # False (default): run the LLM-as-judge over the trajectory + last-K screenshots.
    # True: run the task's deterministic verifier (the script at trajectory.verifier_path)
    #       if present; otherwise fall back to the LLM judge with a note.
    verifier: bool = False


def strict_json_loads(raw):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON number {value!r} is not allowed")

    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} is not allowed")
            value[key] = item
        return value

    return json.loads(
        raw,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


# ---------------------------------------------------------------------------
# LLM config — unified env vars (also used by the per-task verifiers):
#   OPENAI_API_KEY  : bearer token for the OpenAI-compatible judge endpoint
#   OPENAI_BASE_URL : base URL of that endpoint
#   JUDGE_MODEL     : model id to use as the LLM judge
# CLI flags override env (simpleargparser precedence: cmd > code > default).
JUDGE_SYSTEM_BASE = """You are a strict grader of web-agent task completion.

You receive:
  - the ORIGINAL TASK (natural language),
  - the agent's full TRAJECTORY (per step: thought + action + params + URL + any
    extracted_content), reproduced in full — do not assume anything is omitted,
  - the agent's final self-reported answer (if any),
  - the LAST K screenshots of the browser, in order. The last image is the
    final state the user would see.

Your job is to decide whether the task was actually completed. Be skeptical.
Common failure modes you must catch:
  - The agent claimed success but the final page does not show the right content.
  - The agent answered with plausible-sounding but wrong information.
  - The agent stopped early (max_steps) without finishing.
  - The agent navigated somewhere unrelated.
  - EMPTY / NO ANSWER: if the task asks for a fact and the agent's final
    self-reported answer is EMPTY (or clearly absent), mark success=false. The
    task is to PRODUCE an answer; navigating to the right page without
    reporting the answer is NOT a completion. (You may still extract the
    visible page content into answer_extracted for diagnostics, but it does not
    rescue an empty self-report.)
  - PRIOR-KNOWLEDGE SHORTCUT: if the agent's answer is correct but the
    trajectory shows NO navigation to the site's relevant page (no step URL
    reaching the mirror for that content), mark success=false. The task is to
    NAVIGATE the site and read the answer off the page, not recall it.
  - For QUIZ tasks success requires the agent navigated to the quiz, answered
    every question, submitted, and reported the score shown on the result page.

Tie-break rules:
  - If the task asks for a fact and done.text contains a value, cross-check it
    against the final screenshot / extracted_content.
  - If the task is navigational ("find the page that ..."), the final screenshot
    must show that page.
  - If the agent performed irreversible actions the task did not ask for,
    mark success=false and explain.

Respond with ONLY this JSON object (no code fence, no prose):

  {
    "success": <true|false>,
    "confidence": <0.0-1.0>,
    "rationale": "<full one-paragraph reasoning citing specific steps/screenshots>",
    "evidence": ["<quote or short paraphrase of supporting signal 1>", "..."],
    "answer_extracted": "<the answer the agent ended up giving, or empty>"
  }
"""

# Appended to JUDGE_SYSTEM_BASE ONLY when the task carries a judge_rubric.
# This block makes the rubric binding: every MUST checkpoint must hold.
JUDGE_SYSTEM_RUBRIC = """
---
ADDITIONAL INSTRUCTIONS — THIS TASK HAS A JUDGE RUBRIC.
A JUDGE RUBRIC listing concrete FACT CHECKPOINTS is included in the evidence
below. You MUST grade against it: verify EVERY checkpoint and treat each MUST
as a hard requirement for success — if any MUST checkpoint is unmet, mark
success=false. In your rationale, explicitly check each rubric checkpoint.

Include an extra field in your JSON response:
  "rubric_checkpoints": {"<checkpoint text 1>": true, "<checkpoint text 2>": false}
mapping each rubric checkpoint to whether it was satisfied.
"""



def img_payload(path):
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}}


def trajectory_text(traj):
    """Full, untruncated textual rendering of the trajectory.

    NOTHING is truncated here — truncating thoughts / extracted_content / answers
    withers the judge's context and yields inaccurate verdicts. The model gets
    the complete evidence so it can grade accurately.
    """
    lines = []
    lines.append(f"TASK: {traj.get('task', '')}")
    lines.append(f"TASK ID: {traj.get('task_id', '')}")
    lines.append(f"START URL: {traj.get('start_url', '')}")
    rubric = traj.get('judge_rubric', '')
    if rubric:
        lines.append("")
        lines.append("JUDGE RUBRIC — verify EVERY fact checkpoint below:")
        lines.append(rubric)
    lines.append("")
    lines.append(f"TERMINATED: {traj.get('terminated')} ({traj.get('termination_reason')})")
    lines.append(f"AGENT'S FINAL ANSWER (self-report): {traj.get('final_answer', '')!r}")
    lines.append(f"AGENT'S SELF-REPORTED SUCCESS: {traj.get('success_self_report', '<none>')}")
    lines.append("")
    lines.append("STEPS (full, oldest → newest):")
    for s in traj.get("steps", []):
        ar = s.get("action_result") or {}
        line = (
            f"  [step {s['step']}] action={s['action']} params={s.get('params', {})} "
            f"url={s.get('url', '')}\n"
            f"        thought: {s.get('thought', '')}"
        )
        if "is_done" in ar or "success" in ar:
            line += (
                f"\n        action_result: is_done={ar.get('is_done')} "
                f"success={ar.get('success')}"
            )
        ec = ar.get("extracted_content") or ""
        if ec:
            line += f"\n        extracted_content: {ec}"
        if "error" in ar:
            line += f"\n        error: {ar.get('error')}"
        lines.append(line)
    return "\n".join(lines)


def build_messages(traj, screenshot_paths):
    text = trajectory_text(traj)
    user_content = [{"type": "text", "text": text + "\n\nLAST SCREENSHOTS (oldest → newest):"}]
    for p in screenshot_paths:
        user_content.append({"type": "text", "text": f"({p.name})"})
        user_content.append(img_payload(p))
    user_content.append({"type": "text", "text": "Now grade. Respond with the JSON object only."})
    # If/else: the rubric-specific system-prompt block is appended ONLY when this
    # task carries a judge_rubric. Tasks without one (the other 16 sites) get the
    # plain base prompt — no rubric mention, no rubric_checkpoints field.
    system = JUDGE_SYSTEM_BASE
    if traj.get('judge_rubric'):
        system = JUDGE_SYSTEM_BASE + JUDGE_SYSTEM_RUBRIC
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def rubric_checkpoint_texts(rubric):
    """Extract the exact mandatory checkpoint labels supplied to the judge."""
    text = (rubric or "").strip()
    if not text:
        return []
    numbered = re.findall(
        r"\(\d+\)\s*(.*?)(?=\s*\(\d+\)\s*|$)",
        text,
        flags=re.DOTALL,
    )
    if numbered:
        return [" ".join(item.split()) for item in numbered if item.strip()]
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:FACT CHECKPOINTS:\s*|[-*]\s*)", "", line).strip()
        if line:
            lines.append(" ".join(line.split()))
    return lines or [" ".join(text.split())]


def parse_judge_json(raw, rubric_text=""):
    expected_rubric_keys = rubric_checkpoint_texts(rubric_text)
    has_rubric = bool(expected_rubric_keys)
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    start = s.find("{")
    if start < 0:
        raise ValueError(f"no JSON in judge reply: {raw[:200]!r}")
    if s[:start].strip():
        raise ValueError("judge reply must contain JSON only")
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
        raise ValueError(f"invalid JSON in judge reply: {raw[:200]!r}") from exc
    if not isinstance(value, dict):
        raise ValueError("judge reply must be a JSON object")
    if s[start + end:].strip():
        raise ValueError("judge reply must contain exactly one JSON object")
    required = {
        "success": bool,
        "confidence": (int, float),
        "rationale": str,
        "evidence": list,
        "answer_extracted": str,
    }
    for field, expected_type in required.items():
        if field not in value or not isinstance(value[field], expected_type):
            raise ValueError(f"judge reply field {field!r} has an invalid type")
    if (isinstance(value["confidence"], bool)
            or not 0 <= value["confidence"] <= 1
            or (isinstance(value["confidence"], float)
                and not math.isfinite(value["confidence"]))):
        raise ValueError("judge reply confidence must be a number from 0 to 1")
    if not value["rationale"].strip():
        raise ValueError("judge reply rationale must be nonempty")
    if not value["evidence"] or not all(
        isinstance(item, str) and item.strip() for item in value["evidence"]
    ):
        raise ValueError("judge reply evidence must be a nonempty list of strings")
    if value["success"] and not value["answer_extracted"].strip():
        raise ValueError(
            "judge reply success requires a nonempty extracted answer"
        )
    rubric = value.get("rubric_checkpoints")
    allowed_fields = set(required)
    if has_rubric:
        allowed_fields.add("rubric_checkpoints")
    if set(value) != allowed_fields:
        raise ValueError(
            f"judge reply fields must be exactly {sorted(allowed_fields)!r}"
        )
    if has_rubric:
        if not isinstance(rubric, dict) or not rubric or not all(
            isinstance(key, str) and isinstance(item, bool)
            for key, item in rubric.items()
        ):
            raise ValueError(
                "judge reply rubric_checkpoints must map strings to booleans"
            )
        if set(rubric) != set(expected_rubric_keys):
            raise ValueError(
                "judge reply rubric_checkpoints keys must exactly match the "
                "provided rubric checkpoints"
            )
        if value["success"] and not all(rubric.values()):
            raise ValueError(
                "judge reply success cannot be true when a rubric checkpoint is false"
            )
    elif "rubric_checkpoints" in value:
        raise ValueError(
            "judge reply must not include rubric_checkpoints without a rubric"
        )
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


def enforce_trajectory_judge_contract(verdict, traj):
    """Make trajectory completion claims authoritative over an LLM verdict."""
    if verdict.get("success") and (
        not isinstance(traj.get("final_answer"), str)
        or not traj["final_answer"].strip()
        or traj.get("success_self_report") is not True
    ):
        verdict["success"] = False
        verdict["confidence"] = 1.0
        verdict["rationale"] = (
            "Judge contract override: a successful grade requires a nonempty "
            "agent final answer and an explicit successful self-report. "
            + verdict.get("rationale", "")
        ).strip()
        verdict.setdefault("evidence", []).append(
            "trajectory final answer was empty or success_self_report was not true"
        )
    return verdict


def invalid_verifier_result(run_dir, vp, returncode, traj, detail):
    return {
        "task_id": traj.get("task_id", ""),
        "pass": False,
        "success": False,
        "reason": "invalid_verifier_result",
        "rationale": "deterministic verifier returned an invalid result",
        "confidence": 1.0,
        "evidence": [detail],
        "answer_extracted": traj.get("final_answer", "") or "",
        "meta": {
            "mode": "verifier",
            "verifier_path": str(vp),
            "run_dir": str(run_dir),
            "returncode": returncode,
        },
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_verifier(run_dir: Path, traj: dict) -> dict:
    """Run the task's deterministic verifier (mode --verifier True).

    Looks up verifier_path from the trajectory (written by agent.py). Runs the
    verifier under agent_demo's uv project so its simpleArgParser dependency is
    available. The primary verifier always runs with --no_llm, so its result is
    independent of optional API credentials. Returns a verdict dict shaped
    like eval.json (the verifier's {task_id,pass,reason,evidence} + meta).
    """
    verifier_path = traj.get("verifier_path", "")
    if not isinstance(verifier_path, str) or not verifier_path:
        return invalid_verifier_result(
            run_dir, None, 1, traj,
            "verifier mode requested but trajectory has no verifier_path",
        )
    repo_root = Path(__file__).resolve().parents[1]
    raw_vp = Path(verifier_path)
    sites_root = (repo_root / "sites").resolve()
    vp = (repo_root / raw_vp).resolve() if not raw_vp.is_absolute() else raw_vp.resolve()
    try:
        site_relative = vp.relative_to(sites_root)
    except ValueError:
        site_relative = None
    allowed = (
        not raw_vp.is_absolute()
        and raw_vp.parts[:1] == ("sites",)
        and ".." not in raw_vp.parts
        and site_relative is not None
        and len(site_relative.parts) >= 3
        and site_relative.parts[1] == "verify"
        and vp.suffix == ".py"
        and vp.is_file()
    )
    if not allowed:
        return invalid_verifier_result(
            run_dir, vp, 1, traj,
            "invalid verifier_path: expected a repository-relative "
            f"sites/<site>/verify/*.py file, got {verifier_path!r}",
        )
    # Run under agent_demo/ so `uv` finds the pyproject + simpleArgParser dep.
    agent_demo_dir = Path(__file__).resolve().parent
    snapshot_required_task_ids = {
        "Merriam-Webster--11", "Merriam-Webster--12",
        "Merriam-Webster--14", "Merriam-Webster--15",
        "Merriam-Webster--16", "Merriam-Webster--18",
    }
    initial_name = traj.get("initial_db_snapshot", "")
    after_name = traj.get("after_db_snapshot", "")
    if not isinstance(initial_name, str) or not isinstance(after_name, str):
        return invalid_verifier_result(
            run_dir, vp, 1, traj,
            "DB snapshot paths in trajectory must be strings",
        )
    initial_db = None
    after_db = None
    snapshots_valid = False
    snapshots_changed = False
    if isinstance(initial_name, str) and isinstance(after_name, str):
        try:
            initial_db = (run_dir / initial_name).resolve() if initial_name else None
            after_db = (run_dir / after_name).resolve() if after_name else None
            snapshots_valid = (
                initial_db is not None
                and after_db is not None
                and initial_db.parent == run_dir
                and after_db.parent == run_dir
                and initial_db.is_file()
                and after_db.is_file()
                and isinstance(traj.get("initial_db_sha256"), str)
                and isinstance(traj.get("after_db_sha256"), str)
                and file_sha256(initial_db) == traj.get("initial_db_sha256")
                and file_sha256(after_db) == traj.get("after_db_sha256")
            )
            snapshots_changed = (
                snapshots_valid
                and traj.get("initial_db_sha256")
                != traj.get("after_db_sha256")
            )
        except OSError:
            snapshots_valid = False
    if traj.get("task_id") in snapshot_required_task_ids and not snapshots_valid:
        return invalid_verifier_result(
            run_dir, vp, 1, traj,
            "verifier requires immutable initial_state.db and "
            "after_state.db snapshots captured with the trajectory",
        )
    if traj.get("task_id") in snapshot_required_task_ids and not snapshots_changed:
        return invalid_verifier_result(
            run_dir, vp, 1, traj,
            "verifier requires a state transition between its immutable "
            "initial_state.db and after_state.db snapshots",
        )
    if (
        not isinstance(traj.get("final_answer"), str)
        or not traj["final_answer"].strip()
        or traj.get("success_self_report") is not True
    ):
        return invalid_verifier_result(
            run_dir, vp, 1, traj,
            "verifier requires a nonempty final answer and an explicit "
            "successful self-report",
        )
    cmd = [
        "uv", "run", "python", str(vp), "--run_dir", str(run_dir),
        "--no_llm", "True",
    ]
    if snapshots_valid:
        cmd.extend([
            "--initial_db", str(initial_db), "--after_db", str(after_db),
        ])
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(agent_demo_dir),
                       env=os.environ.copy())
    try:
        verdict = strict_json_loads(r.stdout)
    except Exception:
        return invalid_verifier_result(
            run_dir, vp, r.returncode, traj,
            f"verifier did not emit JSON: {r.stderr[:500] or r.stdout[:500]}",
        )
    contract_ok = (
        isinstance(verdict, dict)
        and isinstance(verdict.get("task_id"), str)
        and bool(verdict.get("task_id"))
        and isinstance(verdict.get("pass"), bool)
        and isinstance(verdict.get("reason"), str)
        and (verdict.get("pass") or bool(verdict.get("reason")))
        and isinstance(verdict.get("evidence"), list)
        and all(isinstance(item, str) for item in verdict.get("evidence", []))
        and r.returncode in {0, 1}
        and r.returncode == (0 if verdict.get("pass") else 1)
    )
    if not contract_ok:
        return invalid_verifier_result(
            run_dir, vp, r.returncode, traj,
            "verifier output must be {task_id: str, pass: bool, reason: str, "
            "evidence: list[str]} and exit 0 for PASS or 1 for FAIL",
        )
    verdict["meta"] = {
        "mode": "verifier",
        "verifier_path": str(vp),
        "run_dir": str(run_dir),
        "returncode": r.returncode,
    }
    verdict["success"] = verdict["pass"]
    expected_task_id = traj.get("task_id")
    if not expected_task_id or verdict.get("task_id") != expected_task_id:
        verdict["pass"] = False
        verdict["success"] = False
        verdict["reason"] = "verifier_task_id_mismatch"
        verdict["evidence"].append(
            "verifier task_id does not match trajectory task_id: "
            f"{verdict.get('task_id')!r} != {expected_task_id!r}"
        )
    return verdict


def main():
    args = sap.parse_args(JudgeArgs)
    if not args.run_dir:
        raise SystemExit("--run_dir is required")

    run_dir = Path(args.run_dir).resolve()
    try:
        traj = strict_json_loads((run_dir / "trajectory.json").read_text())
        if not isinstance(traj, dict):
            raise ValueError("trajectory root must be a JSON object")
    except Exception as exc:
        verdict = invalid_verifier_result(
            run_dir, None, 1, {}, f"invalid trajectory.json: {exc}"
        )
        out_path = Path(args.out) if args.out else run_dir / "eval.json"
        out_path.write_text(json.dumps(verdict, indent=2))
        print(f"wrote {out_path}")
        print(f"  pass: False  success: False  reason: {verdict['reason']}")
        raise SystemExit(1)

    if args.verifier:
        print(f"[verifier mode] task_id={traj.get('task_id','?')} verifier={traj.get('verifier_path','<none>')}")
        verdict = run_verifier(run_dir, traj)
        out_path = Path(args.out) if args.out else run_dir / "eval.json"
        out_path.write_text(json.dumps(verdict, indent=2))
        print(f"wrote {out_path}")
        print(f"  pass: {verdict.get('pass')}  success: {verdict.get('success')}  reason: {verdict.get('reason','')}")
        sys.exit(0 if verdict.get("success") else 1)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    api_base = args.api_base or os.environ.get("OPENAI_BASE_URL", "")
    model = args.model or os.environ.get("JUDGE_MODEL", "")
    if not api_key:
        raise SystemExit("API key missing: set --api_key or OPENAI_API_KEY")
    if not api_base:
        raise SystemExit("API base missing: set --api_base or OPENAI_BASE_URL")
    if not model:
        raise SystemExit("model missing: set --model or JUDGE_MODEL")

    shots_dir = run_dir / "screenshots"
    referenced_names = {
        Path(value).name
        for step in traj.get("steps", [])
        if isinstance(step, dict)
        for field in ("screenshot_before", "screenshot_after")
        if isinstance((value := step.get(field)), str) and value
    }
    all_shots = sorted(
        path for path in shots_dir.glob("step_*.png")
        if path.name in referenced_names
    )
    if not all_shots:
        raise SystemExit(f"no trajectory-referenced screenshots in {shots_dir}")
    missing_shots = referenced_names - {path.name for path in all_shots}
    if missing_shots:
        raise SystemExit(
            f"trajectory references missing screenshots: {sorted(missing_shots)!r}"
        )
    last_k = all_shots[-args.last_k_screenshots :]
    print(f"judging {len(traj.get('steps', []))} steps with last {len(last_k)} screenshots: "
          f"{[p.name for p in last_k]}")

    client = openai_client(api_base, api_key)
    messages = build_messages(traj, last_k)

    resp = client.chat.completions.create(model=model, messages=messages)
    raw = resp.choices[0].message.content or ""
    try:
        verdict = parse_judge_json(raw, traj.get("judge_rubric", ""))
    except Exception as e:
        verdict = {
            "success": False,
            "confidence": 0.0,
            "rationale": f"judge parse error: {e}",
            "evidence": [],
            "answer_extracted": "",
            "raw_reply": raw,
        }
    verdict = enforce_trajectory_judge_contract(verdict, traj)

    verdict["meta"] = {
        "run_dir": str(run_dir),
        "task": traj.get("task", ""),
        "model": model,
        "screenshots_used": [p.name for p in last_k],
        "trajectory_steps": len(traj.get("steps", [])),
    }

    out_path = Path(args.out) if args.out else run_dir / "eval.json"
    out_path.write_text(json.dumps(verdict, indent=2))
    print(f"wrote {out_path}")
    print(f"  success: {verdict.get('success')}  confidence: {verdict.get('confidence')}")
    print(f"  rationale: {verdict.get('rationale', '')}")


if __name__ == "__main__":
    main()
