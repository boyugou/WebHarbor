#!/usr/bin/env python3
"""verify_lib.py — shared deterministic + LLM utilities for Merriam-Webster task verification.

Philosophy: DETERMINISTIC PRIMARY GRADING.
  1. Trajectory navigation check (anti knowledge-shortcut): the agent MUST have
     opened the relevant on-site page; a correct answer with no matching navigation
     is a memory-recall shortcut = FAIL.
  2. Answer check: exact / regex / token-containment against frozen ground truth.
  3. DB after-state check (stateful tasks): query the SQLite instance DB directly —
     the strongest deterministic signal (saved-word list, registered user row).
  4. LLM utilities (text match, screenshot-contains) are optional diagnostics,
     anchored on ground truth. eval_judge's primary verifier mode always passes
     --no_llm, making its outcome independent of API availability.

Input signature (per task):
  --run_dir DIR      agent trajectory dir: trajectory.json + screenshots/step_NNN.png
  --initial_db PATH  initial-state SQLite DB (default: fetched instance_seed from container)
  --after_db PATH    after-state  SQLite DB (default: fetched live instance DB from container)
  --container NAME   docker container to fetch DBs from (default: $WH_CONTAINER or wh-review)
  --no_llm           skip LLM-based checks (run deterministic-only)
Output: JSON {task_id, pass, reason, evidence[]} to stdout; exit 0 on PASS, 1 on FAIL.
"""
import base64, json, os, re, sqlite3, subprocess, sys, tempfile, urllib.request
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlsplit, urlunsplit

SITE = "merriam_webster"

# ---------------------------------------------------------------- trajectory
def load_run(run_dir):
    d = Path(run_dir)
    def reject_constant(value):
        raise ValueError(f"non-finite JSON number {value!r} is not allowed")

    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} is not allowed")
            value[key] = item
        return value

    traj = json.loads(
        (d / "trajectory.json").read_text(),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(traj, dict):
        raise ValueError("trajectory root must be a JSON object")
    traj["_run_dir"] = d
    traj["_shots"] = {p.name: p for p in sorted((d / "screenshots").glob("step_*.png"))}
    if not traj["_shots"]:
        raise ValueError("trajectory run has no step screenshots")
    return traj

def step_urls(traj):
    return [s.get("url", "") for s in traj.get("steps", [])]

def _trajectory_origin(traj):
    start_url = traj.get("start_url")
    if not start_url:
        return ("http", "localhost", 40015)
    try:
        parsed = urlsplit(start_url)
        port = parsed.port
    except (TypeError, ValueError):
        return None
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or port is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return (parsed.scheme, parsed.hostname, port)


def _trajectory_port(traj):
    """Backward-compatible accessor for diagnostics and external harnesses."""
    origin = _trajectory_origin(traj)
    return origin[2] if origin is not None else None


def _url_matches(url, target, trusted_origin=("http", "localhost", 40015)):
    """Match an exact path on the trajectory's local origin."""
    try:
        parsed = urlsplit(url)
        target_path = urlsplit(target).path
        port = parsed.port
    except (TypeError, ValueError):
        return False
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or trusted_origin is None
        or (parsed.scheme, parsed.hostname, port) != trusted_origin
        or parsed.username is not None
        or parsed.password is not None
    ):
        return False
    path = parsed.path.rstrip("/") or "/"
    target_path = target_path.rstrip("/") or "/"
    return path == target_path

def navigated_to(traj, target, times=1):
    """Deterministic: at least `times` steps visited this exact local path."""
    trusted_origin = _trajectory_origin(traj)
    return sum(
        1 for url in step_urls(traj)
        if _url_matches(url, target, trusted_origin)
    ) >= times

def navigated_any(traj, substrs):
    return any(navigated_to(traj, s) for s in substrs)


def navigated_in_order(traj, targets):
    """Return true when exact local paths occur in the requested sequence."""
    urls = step_urls(traj)
    trusted_origin = _trajectory_origin(traj)
    cursor = 0
    for target in targets:
        for index in range(cursor, len(urls)):
            if _url_matches(urls[index], target, trusted_origin):
                cursor = index + 1
                break
        else:
            return False
    return True


def finished_at(traj, target):
    """Require the final recorded agent action to occur on an exact path."""
    urls = step_urls(traj)
    return bool(urls) and _url_matches(
        urls[-1], target, _trajectory_origin(traj)
    )


def authenticated_user_proof(traj, target, initial_db, after_db, name):
    """Bind the final protected-page URL to a named user in after-state."""
    urls = step_urls(traj)
    if not urls or not after_db or not _url_matches(
        urls[-1], target, _trajectory_origin(traj)
    ):
        return False
    try:
        pairs = parse_qsl(urlsplit(urls[-1]).query, keep_blank_values=True)
    except (TypeError, ValueError):
        return False
    if len(pairs) != 1 or pairs[0][0] != "auth":
        return False
    token = pairs[0][1]
    if not re.fullmatch(r"[A-Za-z0-9_-]{24,64}", token):
        return False
    if not initial_db:
        return False
    initial_tables = db_query(
        initial_db,
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='benchmark_auth_proof'",
    )
    if initial_tables and db_query(
        initial_db,
        "SELECT 1 FROM benchmark_auth_proof WHERE token=?",
        (token,),
    ):
        return False
    tables = db_query(
        after_db,
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='benchmark_auth_proof'",
    )
    if not tables:
        return False
    rows = db_query(
        after_db,
        "SELECT u.name FROM benchmark_auth_proof p "
        "JOIN users u ON u.id=p.user_id WHERE p.token=?",
        (token,),
    )
    return rows == [(name,)]


def quiz_result_score(traj, target, slug, initial_db, after_db):
    """Return a score whose unpredictable result token exists after the run."""
    if not initial_db or not after_db:
        return None
    tables = db_query(
        after_db,
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='benchmark_quiz_result_proof'",
    )
    if not tables:
        return None
    for url in step_urls(traj):
        if not _url_matches(url, target, _trajectory_origin(traj)):
            continue
        try:
            pairs = parse_qsl(urlsplit(url).query, keep_blank_values=True)
        except (TypeError, ValueError):
            continue
        if len(pairs) != 3 or {key for key, _ in pairs} != {
            "score", "total", "token"
        }:
            continue
        values = dict(pairs)
        try:
            score = int(values["score"])
            total = int(values["total"])
        except (TypeError, ValueError):
            continue
        if total != 10 or not 0 <= score <= total:
            continue
        token = values["token"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{24,64}", token):
            continue
        initial_tables = db_query(
            initial_db,
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='benchmark_quiz_result_proof'",
        )
        if initial_tables and db_query(
            initial_db,
            "SELECT 1 FROM benchmark_quiz_result_proof "
            "WHERE token=? AND slug=?",
            (token, slug),
        ):
            continue
        rows = db_query(
            after_db,
            "SELECT score, total FROM benchmark_quiz_result_proof "
            "WHERE token=? AND slug=?",
            (token, slug),
        )
        if rows == [(score, total)]:
            return str(score)
    return None

def final_answer(traj):
    return (traj.get("final_answer") or "").strip()

def _shot(traj, name):
    if not name:
        return None
    p = traj["_shots"].get(Path(name).name)
    return p if (p and p.exists()) else None

def shot_after_url(traj, target):
    """screenshot_after path of the first step at this exact local path."""
    for s in traj.get("steps", []):
        if _url_matches(
            s.get("url", ""), target, _trajectory_origin(traj)
        ):
            p = _shot(traj, s.get("screenshot_after"))
            if p:
                return p
    return None

def last_shot(traj):
    for s in reversed(traj.get("steps", [])):
        p = _shot(traj, s.get("screenshot_after")) or _shot(traj, s.get("screenshot_before"))
        if p:
            return p
    shots = sorted(traj["_shots"].values())
    return shots[-1] if shots else None

# ---------------------------------------------------------------- deterministic answer match
def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip()).casefold()

_CONTRADICTION_RE = re.compile(
    r"\b(?:incorrect|wrong|false|failed|failure|"
    r"deny|denies|denied|doubt|doubts|doubted)\b|"
    r"\b(?:differs?\s+from|does\s+not\s+match|anything\s+(?:but|except)|"
    r"anything\s+other\s+than)\b",
    re.IGNORECASE,
)

_UNCERTAINTY_RE = re.compile(
    r"\b(?:maybe|perhaps|possibly|probably|presumably|apparently|"
    r"uncertain|unsure)\b|\bi\s+(?:think|guess|suspect)\b|"
    r"\b(?:might|may|could)\s+(?:be|have\s+been)\b",
    re.IGNORECASE,
)

_LOCAL_NEGATION = (
    r"(?:not|never|neither|cannot|can't|doesn't|isn't|wasn't|weren't|"
    r"aren't|don't|didn't|no)"
)

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}

_QUESTION_OPENING = re.compile(
    r"^(?:who|what|when|where|why|how|which|whose|"
    r"am|is|are|was|were|do|does|did|can|could|would|should|"
    r"will|has|have|had|may|might)\b",
    re.IGNORECASE,
)


def _has_declarative_clause(final):
    normalized = norm(final)
    for match in re.finditer(r"[^.!?\n]*(?:[.!?]|$)", normalized):
        clause = match.group(0).strip()
        if not clause:
            continue
        terminator = clause[-1] if clause[-1] in ".!?" else ""
        body = clause[:-1].strip() if terminator else clause
        if terminator != "?" and not _QUESTION_OPENING.match(body):
            return True
    return False


def answer_is_affirmative(final):
    return (
        bool((final or "").strip())
        and not _CONTRADICTION_RE.search(final or "")
        and not _UNCERTAINTY_RE.search(final or "")
        and _has_declarative_clause(final)
    )


def fact_is_asserted(final, expected_pattern):
    """Return true when an expected fact occurs in a declarative clause."""
    normalized = norm(final)
    for match in re.finditer(r"[^.!?\n]*(?:[.!?]|$)", normalized):
        clause = match.group(0).strip()
        if not clause:
            continue
        terminator = clause[-1] if clause[-1] in ".!?" else ""
        body = clause[:-1].strip() if terminator else clause
        if (
            terminator == "?"
            or _QUESTION_OPENING.match(body)
            or not re.search(expected_pattern, body)
        ):
            continue
        return True
    return False


def fact_is_negated(final, expected_pattern):
    """Detect a negation grammatically local to an expected fact.

    Punctuation-delimited contrast such as ``respelling, not IPA, is X`` is
    intentionally allowed, while ``the answer is not X`` is rejected.
    ``expected_pattern`` is a regular expression over normalized text.
    """
    normalized = norm(final)
    normalized = re.sub(
        r"\b(not|never)\s*,\s*(?:in\s+fact|actually|really)\s*,\s*",
        r"\1 ",
        normalized,
    )
    normalized = re.sub(r"\b(not|never)\s*:\s*", r"\1 ", normalized)
    cross_reference_denial = re.compile(
        r"^\s*[,;.]?\s*(?:but\s+)?(?:that|it|this|the\s+value|the\s+answer)"
        r"\s+(?:is|was)\s+(?:not\s+(?:correct|accurate|true)|"
        r"incorrect|wrong|false)\b"
    )
    for match in re.finditer(expected_pattern, normalized):
        prefix = normalized[max(0, match.start() - 120):match.start()]
        local_negation = re.search(
            rf"\b{_LOCAL_NEGATION}\b(?P<between>[^.!?\n]{{0,100}})$",
            prefix,
        )
        if local_negation:
            between = local_negation.group("between")
            if (
                not re.search(r"\b(?:but|rather|instead)\b", between)
                and not re.fullmatch(r"\s+only\s*", between)
            ):
                return True
        tail = normalized[match.end():match.end() + 120]
        if (
            cross_reference_denial.search(tail)
            or re.search(
                r"^\s*(?:[-–—:;,.!?]\s*)+(?:no|incorrect|wrong|false)\b",
                tail,
            )
        ):
            return True
    for segment in re.split(r"[,:;.!?\n\r—]+", normalized):
        if not re.search(expected_pattern, segment):
            continue
        before = (
            rf"\b{_LOCAL_NEGATION}\b"
            rf"(?:\W+(?!(?:but|rather)\b)\w+){{0,6}}"
            rf"\W+(?:{expected_pattern})"
        )
        after = (
            rf"(?:{expected_pattern})\W+(?:(?:is|was|are|were|equals?|means?)"
            rf"\W+\b{_LOCAL_NEGATION}\b|\b{_LOCAL_NEGATION}\b)"
        )
        if re.search(before, segment) or re.search(after, segment):
            return True
    exclusion = (
        rf"\b(?:anything|everything|all)\s+"
        rf"(?:but|except(?:\s+for)?|other\s+than)\W+"
        rf"(?:{expected_pattern})"
    )
    return bool(re.search(exclusion, normalized))

def answer_contains_exact_fact(final, expected):
    expected_norm = norm(expected)
    pattern = re.escape(expected_norm)
    return (
        answer_is_affirmative(final)
        and expected_norm in norm(final)
        and fact_is_asserted(final, pattern)
        and not fact_is_negated(final, pattern)
    )


def answer_confirms_action(final, subjects, action_pattern):
    """Require named subjects plus a non-negated completion/status phrase."""
    normalized = norm(final)
    return (
        answer_is_affirmative(final)
        and all(norm(subject) in normalized for subject in subjects)
        and bool(re.search(action_pattern, normalized))
        and fact_is_asserted(final, action_pattern)
        and not fact_is_negated(final, action_pattern)
    )

def answer_list_equals(final, expected, ordered=True, context_words=()):
    """Accept normal list prose but reject missing, extra, or negated entries."""
    if not answer_is_affirmative(final):
        return False
    normalized = norm(final)
    expected_words = [norm(word) for word in expected]
    positions = []
    for word in expected_words:
        matches = list(re.finditer(rf"\b{re.escape(word)}\b", normalized))
        if (
            len(matches) != 1
            or not fact_is_asserted(final, re.escape(word))
            or fact_is_negated(final, re.escape(word))
        ):
            return False
        positions.append(matches[0].start())
    if ordered and positions != sorted(positions):
        return False

    first = min(positions)
    last = max(
        re.search(rf"\b{re.escape(word)}\b", normalized).end()
        for word in expected_words
    )
    prefix = normalized[:first]
    marker_ends = [m.end() for m in re.finditer(
        r"(?::|\b(?:lists?|listed|includes?|included|are|following|follows)\b)",
        prefix,
    )]
    if marker_ends:
        start = marker_ends[-1]
    else:
        boundary = max(prefix.rfind("."), prefix.rfind(";"), prefix.rfind("\n"))
        start = boundary + 1
    suffix_boundary = re.search(r"[.;!?\n]", normalized[last:])
    end = last + suffix_boundary.start() if suffix_boundary else len(normalized)
    words = re.findall(r"[a-z]+", normalized[start:end])
    filler = {
        "a", "according", "all", "also", "and", "answer", "antonym",
        "antonyms", "are", "as", "c",
        "complete", "consist", "consisted", "consisting", "consists",
        "display", "displayed", "displays",
        "eight", "every", "exact", "exactly", "following", "follows", "for",
        "found", "from", "full", "give", "given", "gives", "has", "here",
        "i", "in", "include", "includes", "is", "item", "items", "just",
        "letter", "list", "listed", "merriam", "my", "of", "on", "only",
        "or", "order", "page", "provided", "respectively", "result",
        "requested", "results", "s", "show", "shown", "shows", "site",
        "starting", "synonym", "synonyms",
        "the", "there", "these", "they", "those", "to", "webster", "website", "with",
        "word", "words",
    }
    filler.update(norm(word) for word in context_words)
    items = [word for word in words if word not in filler]
    list_matches = (
        items == expected_words
        if ordered
        else sorted(items) == sorted(expected_words)
    )
    if not list_matches:
        return False
    extra_assertions = re.findall(
        r"\b([a-z]+)\s+is\s+(?:also\s+)?(?:an?\s+)?"
        r"(?:synonym|antonym)\b|"
        r"\b(?:another|additional)\s+(?:synonym|antonym)\s+is\s+([a-z]+)\b",
        normalized,
    )
    asserted_items = {before or after for before, after in extra_assertions}
    paraphrased_extras = re.findall(
        r"\b([a-z]+)\s+belongs\s+on\s+(?:that|the)\s+"
        r"(?:synonym|antonym)\s+list\b|\bplus\s+([a-z]+)\b|"
        r"\b(?:also|additionally)\s*,?\s+([a-z]+)\b|"
        r"\bin\s+addition\s*,?\s+([a-z]+)\b",
        normalized,
    )
    asserted_items.update(next(word for word in match if word)
                          for match in paraphrased_extras)
    if not asserted_items.issubset(set(expected_words)):
        return False
    # Reject arbitrary content vocabulary outside the delimited list too.
    # This closes paraphrases such as ``In addition, cowardly`` while allowing
    # ordinary framing prose from the explicit filler vocabulary above.
    content_words = {
        word for word in re.findall(r"[a-z]+", normalized)
        if word not in filler
    }
    return content_words.issubset(set(expected_words))

def answer_equals(final, expected):
    value = (final or "").strip()
    quote_pairs = {'"': '"', "'": "'", "“": "”", "‘": "’"}
    if len(value) >= 2 and value[0] in quote_pairs and value[-1] == quote_pairs[value[0]]:
        value = value[1:-1].strip()
    return norm(value) == norm(expected)

def contains_all(final, tokens):
    f = norm(final)
    return all(norm(t) in f for t in tokens)

def contains_any(final, tokens):
    f = norm(final)
    return any(norm(t) in f for t in tokens)

def extract_years(text):
    return re.findall(r"\b(1[5-9]\d{2}|20\d{2})\b", text or "")

def extract_numbers(text):
    """Extract digit and zero-through-ten word forms as canonical digits."""
    pattern = r"\b(?:\d+|" + "|".join(_NUMBER_WORDS) + r")\b"
    return [
        _NUMBER_WORDS.get(token.casefold(), token)
        for token in re.findall(pattern, text or "", re.IGNORECASE)
    ]

def extract_score(text):
    if not answer_is_affirmative(text):
        return None
    matches = re.findall(r"\b(\d+)\s*/\s*10\b", text or "")
    if len(matches) != 1 or not 0 <= int(matches[0]) <= 10:
        return None
    score_pattern = rf"\b{re.escape(matches[0])}\s*/\s*10\b"
    if (not fact_is_asserted(text, score_pattern)
            or fact_is_negated(text, score_pattern)):
        return None
    return matches[0]

# ---------------------------------------------------------------- DB state
def fetch_db(container, kind):
    """kind: 'instance' (after-state) or 'instance_seed' (initial-state). docker cp -> temp file."""
    src = f"{container}:/opt/WebSyn/{SITE}/{kind}/{SITE}.db"
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    r = subprocess.run(["docker", "cp", src, path], capture_output=True, text=True)
    if r.returncode != 0:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise RuntimeError(f"docker cp {src} failed: {r.stderr.strip()}")
    return path

def resolve_db(arg, container, kind):
    if arg:
        return arg
    try:
        return fetch_db(container, kind)
    except Exception:
        return None  # caller treats None as "unavailable" and FAILs that check

def db_query(db_path, sql, params=()):
    con = sqlite3.connect(db_path)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()

def saved_words_for(db_path, email="alice.j@test.com"):
    if not db_path:
        return None
    rows = db_query(db_path,
        "SELECT w.headword FROM saved_words sw JOIN users u ON u.id=sw.user_id "
        "JOIN words w ON w.id=sw.word_id WHERE u.email=? ORDER BY w.headword", (email,))
    return [r[0] for r in rows]

def user_exists(db_path, name=None, email=None):
    if not db_path:
        return None
    rows = db_query(db_path, "SELECT name, email FROM users")
    return any((name is None or r[0] == name) and (email is None or r[1] == email)
               for r in rows)

# ---------------------------------------------------------------- shared LLM utilities (anchored)
# Unified LLM config, same env vars as agent.py / eval_judge.py:
#   OPENAI_API_KEY, OPENAI_BASE_URL, JUDGE_MODEL
import simpleArgParser as sap

# When --no_llm is set (via Judge), the llm_* helpers short-circuit so verifiers
# that call them directly (before j.check(llm=True)) still make ZERO LLM calls.
_NO_LLM = False


def _llm_config():
    """Resolve (api_key, api_base, model) from env once per process."""
    key = os.environ.get("OPENAI_API_KEY", "")
    base = os.environ.get("OPENAI_BASE_URL", "")
    model = os.environ.get("JUDGE_MODEL", "")
    return key, base, model


def _chat(messages, max_tokens=1024):
    """One LLM call against the configured OpenAI-compatible endpoint. Returns text or None."""
    if _NO_LLM:
        return None
    key, base, model = _llm_config()
    if not (key and base and model):
        return None  # no LLM configured -> callers treat as non-PASS
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": 1.0}
    parsed_base = urlsplit(base)
    endpoint_path = parsed_base.path.rstrip("/")
    if not endpoint_path.endswith("/chat/completions"):
        endpoint_path += "/chat/completions"
    endpoint = urlunsplit(parsed_base._replace(path=endpoint_path))
    req = urllib.request.Request(endpoint,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    try:
        data = json.loads(urllib.request.urlopen(req, timeout=180).read())
    except Exception:
        return None  # caller treats None as a non-PASS; never raises
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return None

def _verdict(out):
    """Normalize an LLM reply to (pass_bool, text). None/empty -> (False, '<no reply>')."""
    if not out:
        return False, "<no reply from LLM>"
    s = out.strip()
    return s.upper().startswith("PASS"), s

def llm_text_match(agent_answer, ground_truth, question):
    """One LLM call: does agent_answer correctly answer question AND stay consistent
    with the frozen ground truth? The model is given the ground truth as an anchor
    and is told NOT to use its own knowledge."""
    if _NO_LLM:
        return False, "[skipped: --no_llm]"
    out = _chat([{"role": "user", "content":
        f"You are a STRICT binary grader.\nQuestion: {question}\n"
        f"Ground-truth answer (ANCHOR — judge against THIS, never use your own knowledge): {ground_truth}\n"
        f"Agent's answer: {agent_answer}\n"
        f"Decide PASS or FAIL ignoring case/punctuation/word order/surrounding prose. "
        f"PASS only if the agent's answer is consistent with the ground truth AND actually answers the question. "
        f"Line 1: PASS or FAIL. Line 2: one-sentence reason."}])
    return _verdict(out)

def llm_screenshot_shows(shot_path, must_show, question=""):
    """One vision LLM call: does this screenshot visibly render text answering/containing
    `must_show`? The model judges pixels only, anchored on the expected content."""
    if _NO_LLM:
        return False, "[skipped: --no_llm]"
    b64 = base64.b64encode(Path(shot_path).read_bytes()).decode()
    out = _chat([{"role": "user", "content": [
        {"type": "text", "text":
            f"You are a STRICT binary grader. Only what is VISIBLY rendered in this screenshot counts.\n"
            f"Question the page should answer: {question}\n"
            f"Expected content to verify PRESENCE of: {must_show}\n"
            f"PASS only if the expected content (or a semantically equivalent on-screen answer) is visibly shown. "
            f"Do NOT use prior knowledge — judge only the rendered pixels.\n"
            f"Line 1: PASS or FAIL. Line 2: quote the visible evidence."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}])
    return _verdict(out)

# ---------------------------------------------------------------- judge harness + CLI
class Judge:
    def __init__(self, task_id, no_llm=False):
        global _NO_LLM
        _NO_LLM = bool(no_llm)   # gate the llm_* helpers at the source
        self.task_id = task_id
        self.no_llm = no_llm
        self.ok = True
        self.reason = ""
        self.evidence = []

    def check(self, name, cond, evidence="", llm=False):
        if llm and self.no_llm:
            self.evidence.append(f"[SKIP] {name} (--no-llm)")
            return True
        if cond:
            self.evidence.append(f"[PASS] {name}: {evidence}")
        else:
            self.ok = False
            if not self.reason:
                self.reason = name   # record the FIRST failing check
            self.evidence.append(f"[FAIL] {name}: {evidence}")
        return bool(cond)

    def emit(self):
        print(json.dumps({"task_id": self.task_id, "pass": self.ok,
                          "reason": self.reason, "evidence": self.evidence}, indent=2))
        sys.exit(0 if self.ok else 1)

def parse_args():
    @dataclass
    class VerifyArgs:
        run_dir: str = ""
        initial_db: str = ""
        after_db: str = ""
        container: str = os.environ.get("WH_CONTAINER", "wh-review")
        no_llm: bool = False

        def post_process(self):
            if not self.run_dir:
                raise SystemExit("--run_dir is required")
    return sap.parse_args(VerifyArgs)
