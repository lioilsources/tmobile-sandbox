#!/usr/bin/env python3
"""UC15 benchmark harness: OpenCode swarm configurations x real tasks.

For every (task, config) pair:
  1. fresh working copy: `git clone --shared` of the task repo -> checkout base_commit
     -> apply base.patch (removal tasks) -> hide the hidden test files (removal tasks)
  2. per-run OpenCode config dir generated from the swarm repo + a config overlay
     (solo coder / director+coder / full swarm, each with or without validator)
  3. `opencode run --agent <entry> --format json --auto` with a wall-clock timeout,
     isolated session DB (XDG_DATA_HOME per run), GPU + vLLM metrics sampled meanwhile
  4. hidden tests restored from gold_commit, setup_cmd + test_cmd -> pass/fail
  5. metrics from `opencode session list/export`: tokens per agent, LLM calls, tool calls,
     delegations, validator verdicts, test runs; vLLM counters as a cross-check

Results: <out>/<run_id>/results.jsonl + summary.md (+ per-run logs under runs/).

Typical use on the VM (see README.md):
  python3 run_bench.py --env ~/opencode-swarm/env/single-model.env --tasks all --configs all
  python3 run_bench.py --check-gold --tasks all          # harness self-test, no LLM
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE / "tasks"

CONFIGS = {
    # name: (entry agent, allowed subagents, validator?)
    "solo":        ("solo",     [],                                   False),
    "solo+val":    ("solo",     ["validator"],                        True),
    "dc":          ("director", ["coder-*"],                          False),
    "dc+val":      ("director", ["coder-*", "validator"],             True),
    "swarm":       ("director", ["coder-*", "fetcher", "researcher"], False),
    "swarm+val":   ("director", ["coder-*", "fetcher", "researcher", "validator"], True),
}
LANG_CODER = {"go": "coder-go", "dart": "coder-dart", "python": "coder-py", "py": "coder-py", "ts": "coder-ts"}
TEST_RE = re.compile(r"\b(go test|dart test|flutter test|pytest|npm test|npx vitest|node --test)\b")
LOCK = threading.Lock()


# ----------------------------------------------------------------------------- utils
def sh(cmd, cwd=None, env=None, timeout=None, log=None, check=False):
    """Run a shell command, optionally appending combined output to `log`."""
    t0 = time.time()
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=env, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        rc, out = p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        rc, out = 124, (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        out += f"\n[timeout after {timeout}s]"
    if log:
        with open(log, "a") as f:
            f.write(f"$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}  (cwd={cwd})\n{out}\n[rc={rc} {time.time()-t0:.1f}s]\n")
    if check and rc != 0:
        raise RuntimeError(f"command failed rc={rc}: {cmd}\n{out[-2000:]}")
    return rc, out


def load_env_file(path: Path, base: dict) -> dict:
    """Source a shell env preset (KEY=VALUE with ${VAR:-default}) through bash."""
    out = subprocess.run(["bash", "-c", f"set -a; source {sh_quote(str(path))}; set +a; env -0"],
                         env=base, stdout=subprocess.PIPE, check=True).stdout
    env = dict(base)
    for item in out.split(b"\0"):
        if b"=" in item:
            k, v = item.split(b"=", 1)
            env[k.decode()] = v.decode()
    return env


def sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def load_tasks(selector: str) -> list[dict]:
    tasks = []
    for d in sorted(TASKS_DIR.iterdir()):
        if d.name.startswith("_") or not (d / "task.json").exists():
            continue
        t = json.loads((d / "task.json").read_text())
        t["_dir"] = d
        tasks.append(t)
    if selector in ("all", ""):
        return tasks
    want = set(selector.split(","))
    picked = [t for t in tasks if t["id"] in want or t["lang"] in want or t.get("difficulty") in want]
    missing = want - {t["id"] for t in picked} - {t["lang"] for t in picked} - {t.get("difficulty") for t in picked}
    if missing:
        sys.exit(f"unknown task selector(s): {sorted(missing)}")
    return picked


# ----------------------------------------------------------------------------- config overlay
def set_frontmatter_block(md: str, key: str, block: str) -> str:
    """Replace (or add) a top-level YAML key in the frontmatter of an agent .md file."""
    head, body = md.split("\n---\n", 1) if md.startswith("---\n") else ("---", md)
    lines = head.split("\n")
    out, skip = [], False
    for ln in lines:
        if skip and (ln.startswith("  ") or ln.strip() == ""):
            continue
        skip = False
        if re.match(rf"^{re.escape(key)}:", ln):
            skip = True
            continue
        out.append(ln)
    out.append(block.rstrip("\n"))
    return "\n".join(out) + "\n---\n" + body


def set_permission_task(md: str, allowed: list[str]) -> str:
    """Rewrite the `task:` entry inside `permission:` of an agent file."""
    head, body = md.split("\n---\n", 1)
    new_task = "  task:\n    \"*\": deny\n" + "".join(f"    \"{a}\": allow\n" for a in allowed) if allowed else "  task: deny\n"
    lines, res, i = head.split("\n"), [], 0
    replaced = False
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("  task:"):
            i += 1
            while i < len(lines) and lines[i].startswith("    "):
                i += 1
            res.append(new_task.rstrip("\n"))
            replaced = True
            continue
        res.append(ln)
        i += 1
    if not replaced:
        res = [l if l != "permission:" else "permission:\n" + new_task.rstrip("\n") for l in res]
    return "\n".join(res) + "\n---\n" + body


SOLO_VALIDATE = """

## Validation (benchmark config solo+val)
When you believe the task is done, call the `task` tool with `subagent_type: "validator"` (it is not a tool of its own),
passing the goal, the changed files and the test command. If its verdict is `fail`, fix the listed issues and validate again (at most 3 rounds).
"""
SOLO_HEADER = """You are working ALONE on the whole task (no director). Plan briefly, then implement it end to end.
Ignore instructions about reporting to a director; finish with a short summary of changed files and test results.

"""


def build_config_dir(swarm: Path, dest: Path, cfg_name: str, lang: str) -> str:
    entry, allowed, with_val = CONFIGS[cfg_name]
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "agents").mkdir(parents=True)
    shutil.copytree(swarm / ".opencode" / "skills", dest / "skills")
    shutil.copy(swarm / "opencode.jsonc", dest / "opencode.jsonc")
    shutil.copy(swarm / "AGENTS.md", dest / "SWARM_RULES.md")
    for f in (swarm / ".opencode" / "agents").glob("*.md"):
        name = f.stem
        if name == "validator" and not with_val:
            continue
        if name in ("fetcher", "researcher") and "fetcher" not in allowed:
            continue
        md = f.read_text()
        if name == "director":
            md = set_permission_task(md, allowed)
        (dest / "agents" / f.name).write_text(md)
    if entry == "solo":
        coder = LANG_CODER[lang]
        md = (swarm / ".opencode" / "agents" / f"{coder}.md").read_text()
        md = set_frontmatter_block(md, "mode", "mode: primary")
        md = set_frontmatter_block(md, "steps", "steps: 80")
        md = set_frontmatter_block(md, "description", f"description: Solo {lang} coder (benchmark baseline)")
        md = set_permission_task(md, ["validator"] if with_val else [])
        head, body = md.split("\n---\n", 1)
        body = SOLO_HEADER + body + (SOLO_VALIDATE if with_val else "")
        (dest / "agents" / "solo.md").write_text(head + "\n---\n" + body)
    return entry


def link_plugin_cache(cfg: Path, cache: Path):
    """OpenCode npm-installs @opencode-ai/plugin (~60 MB) into every config dir. Reuse one copy."""
    if (cache / "node_modules").is_dir() and (cache / "package-lock.json").exists():
        for f in ("package.json", "package-lock.json", ".gitignore"):
            if (cache / f).exists():
                shutil.copy(cache / f, cfg / f)
        (cfg / "node_modules").symlink_to(cache / "node_modules")


def harvest_plugin_cache(cfg: Path, cache: Path):
    nm = cfg / "node_modules"
    with LOCK:
        complete = (cfg / "package-lock.json").exists() and (nm / "@opencode-ai" / "plugin" / "package.json").exists()
        if complete and nm.is_dir() and not nm.is_symlink() and not (cache / "node_modules").exists():
            cache.mkdir(parents=True, exist_ok=True)
            for f in ("package.json", "package-lock.json", ".gitignore"):
                if (cfg / f).exists():
                    shutil.copy(cfg / f, cache / f)
            shutil.move(str(nm), str(cache / "node_modules"))
    if nm.is_symlink():
        nm.unlink()
    elif nm.is_dir():
        shutil.rmtree(nm, ignore_errors=True)


# ----------------------------------------------------------------------------- monitors
class Sampler(threading.Thread):
    """Samples GPU utilisation (nvidia-smi) every `period` seconds."""

    def __init__(self, period=2.0):
        super().__init__(daemon=True)
        self.period, self.samples, self.stop_ev = period, [], threading.Event()
        self.ok = shutil.which("nvidia-smi") is not None

    def run(self):
        while self.ok and not self.stop_ev.is_set():
            rc, out = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits", timeout=10)
            if rc == 0 and out.strip():
                try:
                    u, m = [float(x) for x in out.strip().splitlines()[0].split(",")]
                    self.samples.append((time.time(), u, m))
                except ValueError:
                    pass
            self.stop_ev.wait(self.period)

    def summary(self, t0, t1):
        xs = [u for (t, u, _) in self.samples if t0 <= t <= t1]
        return {"gpu_util_avg": round(sum(xs) / len(xs), 1) if xs else None, "gpu_util_max": max(xs) if xs else None,
                "gpu_samples": len(xs)}


def vllm_counters(base_url: str) -> dict:
    """Prompt/generation token counters from vLLM /metrics (whole server, all clients)."""
    root = base_url.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    try:
        with urllib.request.urlopen(root + "/metrics", timeout=5) as r:
            text = r.read().decode()
    except Exception:
        return {}
    res = {}
    for key in ("vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:request_success_total"):
        tot = 0.0
        for ln in text.splitlines():
            if ln.startswith(key):
                try:
                    tot += float(ln.rsplit(" ", 1)[1])
                except ValueError:
                    pass
        res[key.split(":")[1]] = tot
    return res


# ----------------------------------------------------------------------------- metrics
def export_sessions(opencode: str, env: dict, cwd: Path, log: Path) -> list[dict]:
    rc, out = sh([opencode, "session", "list", "--format", "json"], cwd=cwd, env=env, timeout=60)
    try:
        sessions = json.loads(out[out.index("["):])
    except (ValueError, json.JSONDecodeError):
        return []
    exported = []
    for s in sessions:
        rc, out = sh([opencode, "export", s["id"]], cwd=cwd, env=env, timeout=120)
        try:
            exported.append(json.loads(out[out.index("{"):]))
        except (ValueError, json.JSONDecodeError):
            continue
    with open(log, "w") as f:
        json.dump(exported, f)
    return exported


def session_metrics(exported: list[dict]) -> dict:
    per_agent: dict[str, dict] = {}
    tools: dict[str, int] = {}
    delegations: dict[str, int] = {}
    verdicts = []
    test_runs = llm_calls = 0
    for s in exported:
        for m in s.get("messages", []):
            info = m.get("info", {})
            if info.get("role") != "assistant":
                continue
            llm_calls += 1
            agent = info.get("agent") or info.get("mode") or "?"
            tk = info.get("tokens") or {}
            a = per_agent.setdefault(agent, {"calls": 0, "input": 0, "output": 0, "reasoning": 0})
            a["calls"] += 1
            a["input"] += tk.get("input", 0)
            a["output"] += tk.get("output", 0)
            a["reasoning"] += tk.get("reasoning", 0)
            for p in m.get("parts", []):
                if p.get("type") != "tool":
                    continue
                name = p.get("tool", "?")
                tools[name] = tools.get(name, 0) + 1
                st = p.get("state", {})
                inp = st.get("input") or {}
                if name == "bash" and TEST_RE.search(str(inp.get("command", ""))):
                    test_runs += 1
                if name == "task":
                    sub = inp.get("subagent_type", "?")
                    delegations[sub] = delegations.get(sub, 0) + 1
                    if sub == "validator":
                        mv = re.search(r'"verdict"\s*:\s*"(pass|fail)', str(st.get("output", "")))
                        verdicts.append(mv.group(1) if mv else "unparsed")
    tot_in = sum(a["input"] for a in per_agent.values())
    tot_out = sum(a["output"] + a["reasoning"] for a in per_agent.values())
    return {"tokens_in": tot_in, "tokens_out": tot_out, "llm_calls": llm_calls, "sessions": len(exported),
            "per_agent": per_agent, "tool_calls": tools, "delegations": delegations,
            "validator_verdicts": verdicts, "fix_rounds": verdicts.count("fail"), "test_runs": test_runs}


# ----------------------------------------------------------------------------- one run
def prepare_repo(task: dict, repos: Path, work: Path, log: Path):
    name = task["repo"].split("/")[-1]
    src = repos / name
    if not src.exists():
        raise RuntimeError(f"missing clone {src} (git clone https://github.com/{task['repo']}.git)")
    if work.exists():
        shutil.rmtree(work)
    sh(["git", "clone", "-q", "--shared", "--no-checkout", str(src), str(work)], log=log, check=True)
    sh(["git", "-c", "advice.detachedHead=false", "checkout", "-q", task["base_commit"]], cwd=work, log=log, check=True)
    if task.get("base_patch"):
        sh(["git", "apply", str(task["_dir"] / task["base_patch"])], cwd=work, log=log, check=True)
    if task.get("kind") == "removal":
        # the stubbed functions' tests exist at HEAD -> hide them, they are the hidden tests
        for tf in task["test_files"]:
            (work / tf).unlink(missing_ok=True)
    sh("git add -A && git -c user.name=bench -c user.email=bench@localhost commit -qm 'bench: base' --allow-empty",
       cwd=work, log=log, check=True)
    sh(["git", "remote", "remove", "origin"], cwd=work, log=log)


def restore_tests_and_check(task: dict, work: Path, env: dict, log: Path, timeout: int) -> tuple[bool, str]:
    for tf in task["test_files"]:
        rc, out = sh(["git", "show", f"{task['gold_commit']}:{tf}"], cwd=work, timeout=30)
        if rc != 0:
            return False, f"cannot read gold test file {tf}"
        p = work / tf
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(out)
    wd = work / task.get("workdir", ".")
    if task.get("setup_cmd"):
        sh(task["setup_cmd"], cwd=wd, env=env, timeout=timeout, log=log)
    rc, out = sh(task["test_cmd"], cwd=wd, env=env, timeout=timeout, log=log)
    return rc == 0, out[-3000:]


def task_prompt(task: dict) -> str:
    text = (task["_dir"] / "task.md").read_text()
    wd = task.get("workdir", ".")
    foot = "\n\n---\nWork in this repository only. Do not commit."
    if wd not in (".", ""):
        foot += f" The relevant package lives in `{wd}/` (run its tests from there)."
    return text + foot


def run_one(task, cfg_name, a, base_env, run_root: Path, sampler: Sampler | None) -> dict:
    rid = f"{cfg_name}/{task['id']}"
    d = run_root / "runs" / cfg_name.replace("+", "_") / task["id"]
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    log = d / "harness.log"
    work = d / "repo"
    row = {"task": task["id"], "repo": task["repo"], "lang": task["lang"], "difficulty": task.get("difficulty"),
           "config": cfg_name, "model": base_env.get("SWARM_CODER_MODEL"),
           "validator_model": base_env.get("SWARM_VALIDATOR_MODEL") if CONFIGS[cfg_name][2] else None}
    try:
        prepare_repo(task, Path(a.repos), work, log)
        if a.check_gold:
            # harness self-test: gold solution must pass, base must fail
            ok_base, _ = restore_tests_and_check(task, work, base_env, log, task.get("timeout_s", 900))
            sh(["git", "checkout", "-q", task["gold_commit"], "--", "."], cwd=work, log=log)
            ok_gold, tail = restore_tests_and_check(task, work, base_env, log, task.get("timeout_s", 900))
            row.update({"base_fails": not ok_base, "gold_passes": ok_gold, "pass": (not ok_base) and ok_gold})
            return row
        entry = build_config_dir(Path(a.swarm), d / "ocfg", cfg_name, task["lang"])
        link_plugin_cache(d / "ocfg", Path(a.out) / "_plugin_cache")
        env = dict(base_env)
        env.update({
            "OPENCODE_CONFIG_DIR": str(d / "ocfg"),
            "OPENCODE_CONFIG_CONTENT": json.dumps({"instructions": [str(d / "ocfg" / "SWARM_RULES.md")]}),
            "XDG_DATA_HOME": str(d / "xdg-data"), "XDG_STATE_HOME": str(d / "xdg-state"),
            "OPENCODE_DISABLE_AUTOUPDATE": "1", "OPENCODE_DISABLE_CLAUDE_CODE_PROMPT": "1",
            "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1", "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
            "OPENCODE_DISABLE_MODELS_FETCH": "1",
        })
        timeout = a.timeout or task.get("timeout_s", 900) * 2
        cmd = [a.opencode, "run", "--agent", entry, "--format", "json", "--auto", "--dir", str(work)]
        if a.variant:
            cmd += ["--variant", a.variant]
        cmd.append(task_prompt(task))
        if a.dry_run:
            row["cmd"] = " ".join(sh_quote(c) for c in cmd[:-1]) + " \"$(cat task.md)\""
            return row
        v0 = vllm_counters(env.get("SWARM_CODER_URL", ""))
        t0 = time.time()
        with open(d / "transcript.jsonl", "w") as out, open(d / "opencode.stderr", "w") as err:
            # own process group: on timeout stop opencode *and* its children cleanly, so the
            # session DB is unlocked and the sessions can still be exported
            p = subprocess.Popen(cmd, cwd=work, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                 start_new_session=True)
            try:
                rc = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                rc = 124
                for sig, grace in ((signal.SIGINT, 15), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
                    try:
                        os.killpg(p.pid, sig)
                        p.wait(timeout=grace)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                    except ProcessLookupError:
                        break
                time.sleep(2)
        t1 = time.time()
        v1 = vllm_counters(env.get("SWARM_CODER_URL", ""))
        rc_diff, diff = sh("git add -A && git diff --cached --stat HEAD", cwd=work)
        (d / "agent.diff").write_text(sh("git diff --cached HEAD", cwd=work)[1])
        ok, tail = restore_tests_and_check(task, work, base_env, log, task.get("timeout_s", 900))
        exported = export_sessions(a.opencode, env, work, d / "sessions.json")
        row.update({"pass": ok, "agent_rc": rc, "timed_out": rc == 124, "secs": round(t1 - t0, 1),
                    "diffstat": diff.strip().splitlines()[-1:] if diff.strip() else [],
                    **session_metrics(exported)})
        if v0 and v1:
            row["vllm_delta"] = {k: v1.get(k, 0) - v0.get(k, 0) for k in v1}
            row["vllm_delta_note"] = "server-wide counters; exact only when runs are not parallel"
        if sampler:
            row.update(sampler.summary(t0, t1))
        row["test_tail"] = tail[-800:]
    except Exception as e:  # keep the benchmark going
        row.update({"pass": False, "error": str(e)[-1500:]})
    finally:
        if not a.keep and work.exists() and not a.dry_run:
            shutil.rmtree(work, ignore_errors=True)
        if (d / "ocfg").exists():
            harvest_plugin_cache(d / "ocfg", Path(a.out) / "_plugin_cache")
        for sub in ("xdg-data", "xdg-state"):
            # sessions are already exported; DB + snapshots are large
            if not a.keep:
                shutil.rmtree(d / sub, ignore_errors=True)
    with LOCK:
        print(f"[{dt.datetime.now():%H:%M:%S}] {rid:32s} {'PASS' if row.get('pass') else 'FAIL'}"
              f"  {row.get('secs', '-')}s  tok={row.get('tokens_in', '-')}/{row.get('tokens_out', '-')}"
              f"{'  ERR ' + row['error'][:80] if row.get('error') else ''}", flush=True)
    return row


# ----------------------------------------------------------------------------- summary
def write_summary(rows: list[dict], path: Path, meta: dict):
    by_cfg: dict[str, list] = {}
    for r in rows:
        by_cfg.setdefault(r["config"], []).append(r)
    lines = [f"# UC15 benchmark — {meta['run_id']}", "",
             f"coder `{meta.get('coder')}`, validator `{meta.get('validator')}`, opencode {meta.get('opencode_version')}, "
             f"{len({r['task'] for r in rows})} tasks, parallel {meta.get('parallel')}", "",
             "| config | pass@1 | solved | avg time [s] | avg tokens in/out | avg LLM calls | avg test runs | fix rounds | GPU util avg |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cfg in CONFIGS:
        rs = by_cfg.get(cfg)
        if not rs:
            continue
        n = len(rs)
        solved = sum(1 for r in rs if r.get("pass"))

        def avg(k):
            xs = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
            return sum(xs) / len(xs) if xs else float("nan")
        lines.append(f"| {cfg} | {solved / n:.0%} | {solved}/{n} | {avg('secs'):.0f} | {avg('tokens_in'):.0f} / {avg('tokens_out'):.0f} "
                     f"| {avg('llm_calls'):.1f} | {avg('test_runs'):.1f} | {sum(r.get('fix_rounds', 0) for r in rs)} | {avg('gpu_util_avg'):.0f} % |")
    lines += ["", "## Per task", "", "| task | lang | " + " | ".join(c for c in CONFIGS if c in by_cfg) + " |",
              "|---|---|" + "---|" * len(by_cfg)]
    for tid in sorted({r["task"] for r in rows}):
        cells = []
        for cfg in CONFIGS:
            if cfg not in by_cfg:
                continue
            r = next((x for x in by_cfg[cfg] if x["task"] == tid), None)
            cells.append("—" if r is None else ("✅" if r.get("pass") else ("⏱" if r.get("timed_out") else "❌")))
        lang = next(r["lang"] for r in rows if r["task"] == tid)
        lines.append(f"| {tid} | {lang} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", help="swarm env preset (opencode-swarm/env/*.env)")
    ap.add_argument("--swarm", default=os.environ.get("SWARM_REPO", str(Path.home() / "opencode-swarm")))
    ap.add_argument("--tasks", default="all", help="all | comma list of ids / langs / difficulties")
    ap.add_argument("--configs", default="all", help="all | comma list of " + ",".join(CONFIGS))
    ap.add_argument("--repos", default="/srv/work/w2/uc15/repos")
    ap.add_argument("--out", default="/srv/work/w2/uc15/results")
    ap.add_argument("--run-id", default=dt.datetime.now().strftime("%Y%m%d-%H%M"))
    ap.add_argument("--opencode", default=shutil.which("opencode") or str(Path.home() / ".local/bin/opencode"))
    ap.add_argument("--parallel", type=int, default=1, help="tasks in flight at once (RAM: ~2-3 GB per Go/Dart test run)")
    ap.add_argument("--timeout", type=int, default=0, help="agent wall-clock limit [s]; default 2x task timeout_s")
    ap.add_argument("--variant", default="")
    ap.add_argument("--keep", action="store_true", help="keep working copies and session DBs")
    ap.add_argument("--dry-run", action="store_true", help="prepare everything, print commands, do not call the model")
    ap.add_argument("--check-gold", action="store_true", help="harness self-test: base fails, gold passes (no LLM)")
    a = ap.parse_args()

    base_env = dict(os.environ)
    envsh = TASKS_DIR / "env.sh"
    if envsh.exists():
        base_env = load_env_file(envsh, base_env)
    if a.env:
        base_env = load_env_file(Path(a.env).expanduser(), base_env)
    elif not a.check_gold:
        sys.exit("--env is required (e.g. ~/opencode-swarm/env/single-model.env)")

    tasks = load_tasks(a.tasks)
    cfgs = list(CONFIGS) if a.configs == "all" else a.configs.split(",")
    for c in cfgs:
        if c not in CONFIGS:
            sys.exit(f"unknown config {c}; choose from {list(CONFIGS)}")
    if a.check_gold:
        cfgs = ["solo"]
    run_root = Path(a.out) / a.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    version = sh([a.opencode, "--version"])[1].strip() if Path(a.opencode).exists() or shutil.which(a.opencode) else "?"
    meta = {"run_id": a.run_id, "coder": base_env.get("SWARM_CODER_MODEL"), "validator": base_env.get("SWARM_VALIDATOR_MODEL"),
            "opencode_version": version, "parallel": a.parallel, "configs": cfgs, "tasks": [t["id"] for t in tasks],
            "started": dt.datetime.now().isoformat(timespec="seconds"), "check_gold": a.check_gold}
    (run_root / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"run {a.run_id}: {len(tasks)} tasks x {len(cfgs)} configs -> {run_root}", flush=True)

    sampler = Sampler()
    sampler.start()
    jobs = [(t, c) for c in cfgs for t in tasks]
    rows = []
    res_path = run_root / ("gold_check.jsonl" if a.check_gold else "results.jsonl")
    with cf.ThreadPoolExecutor(max_workers=max(1, a.parallel)) as ex, open(res_path, "a") as fout:
        futs = [ex.submit(run_one, t, c, a, base_env, run_root, sampler) for t, c in jobs]
        for f in cf.as_completed(futs):
            r = f.result()
            rows.append(r)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
    sampler.stop_ev.set()
    meta["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (run_root / "meta.json").write_text(json.dumps(meta, indent=1))
    if a.dry_run:
        for r in rows:
            print(r.get("cmd") or r.get("error"))
        return
    if a.check_gold:
        bad = [r["task"] for r in rows if not r.get("pass")]
        print(f"gold check: {len(rows) - len(bad)}/{len(rows)} ok" + (f"; FAILED: {bad}" if bad else ""))
        return
    write_summary(rows, run_root / "summary.md", meta)
    print((run_root / "summary.md").read_text())


if __name__ == "__main__":
    main()
