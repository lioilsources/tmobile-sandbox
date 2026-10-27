# UC15 smoke test — 2026-10-27

Server: GPU profile **L** = `Qwen3-30B-A3B-Instruct-2507-FP8`, `--max-model-len 4096`, **no**
`--enable-auto-tool-choice` (OpenCode requests are rejected with HTTP 400 as-is). The same server was
serving a 250-request batch job of another session at the same time (GPU 100 %), so all timings are
meaningless — only wiring was tested. Coder model (profile W) was not running.

Workaround: `toolshim.py` (here and in `opencode-swarm/scripts/`) between OpenCode and vLLM:
`tool_choice → none` (vLLM still renders the tools into the Qwen3 template), parses Hermes
`<tool_call>{…}</tool_call>` blocks back into OpenAI `tool_calls`, replays SSE, clamps `max_tokens`
to the 4k window and (`--slim-tools 240`) truncates tool descriptions. Plus `run-example.sh --lean`
(no AGENTS.md). Without slimming, a single director turn is ~7.6k prompt tokens (system ~2.3k incl.
AGENTS.md, tool schemas ~5k: bash 4.7k chars, task 3.6k, todowrite 2k).

## Results

| # | where | what | result |
|---|---|---|---|
| 1 | Mac | config parse: 8 agents, 5 skills, `{env:}` incl. numeric limits, variants → request body (`chat_template_kwargs`), model `options` → body (`top_k`…) | ✅ (`opencode debug agent/config/skill`, request dump) |
| 2 | Mac | `permissions:` (V2 rule array) in agent frontmatter | ❌ not enforced in 1.19.4 — lands in model `options`; switched to `permission:` |
| 3 | Mac | micro task (`micro-py`: fix `median`), `director` | ✅ hidden tests PASS; flow director → `task` → **coder-py** (skill load, glob/read/edit/write, pytest) → **validator** (JSON verdict `pass`) → final report; 466 s, 37 LLM requests, 2 context overflows recovered by compaction |
| 4 | Mac | validator alone via director on a finished change | ✅ JSON verdict, ran pytest itself |
| 5 | Mac | `03-python-etl` example, `director`, no slimming | ❌ every request 7.6k > 4k → infinite compaction/retry loop (→ harness timeouts are mandatory) |
| 6 | VM | `run_bench.py --check-gold --tasks all` | ✅ 25/25 (base fails, gold passes) |
| 7 | VM | `run_bench.py` real tasks `wlp-03`, `promoclown-02` × `solo`, `dc+val` (shim, 4k) | harness ✅ end to end (clone, overlay, run, hidden tests, session export, per-agent tokens, GPU sampler); tasks ❌ as expected — repo files do not fit a 4k window (solo/wlp-03 195 s, 11 LLM calls, 32k input tokens; the rest hit the 420 s timeout). Fixed afterwards: on timeout the harness now stops the opencode process group gracefully so sessions are still exported (re-tested: 1 session, 6 calls); plugin `node_modules` (63 MB/run) no longer accumulate |

## Findings fed back into the swarm

- Director twice called a non-existent tool named `validator` instead of `task(subagent_type="validator")` →
  prompt now says explicitly that sub-agents are not tools (`director.md`, solo+val overlay).
- Validator ran `ruff check . --fix` (mutating) → bash denies `*--fix*`, `*--write*`, `gofmt -w`, `sed -i`, formatters.
- Director does not need skills → `skill: deny` (−~400 prompt tokens per turn).
- OpenCode injected the host user's `~/.claude/CLAUDE.md` into the prompt → harness/scripts set
  `OPENCODE_DISABLE_CLAUDE_CODE_PROMPT=1` (+ `_SKILLS`, `EXTERNAL_SKILLS`).
- `opencode run` blocks reading stdin when it is not a TTY → always `< /dev/null`.
- Subagent tokens are not in the `--format json` stream (only the parent session) → harness exports all
  sessions from an isolated `XDG_DATA_HOME`.
