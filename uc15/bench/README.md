# UC15 benchmark — OpenCode swarm on local models

Harness: `run_bench.py` (stdlib Python, runs on the VM). Swarm config: separate repo `opencode-swarm`
(synced to `~/opencode-swarm` on the VM). Task set: `tasks/` (25 tasks, see `tasks/INDEX.md`).
SWE-bench Lite subset: `swebench/` (list + procedure only).

## What one run does

For every `(task, config)`:

1. `git clone --shared` of `/srv/work/w2/uc15/repos/<repo>` → checkout `base_commit` → `git apply base.patch`
   (removal tasks; their test files are deleted from the working copy so the agent cannot read them).
2. Generates a per-run OpenCode config dir from the swarm repo + overlay for the config, passes the swarm rules
   (`AGENTS.md`) via `OPENCODE_CONFIG_CONTENT.instructions`; the target repo's own `AGENTS.md`/`CLAUDE.md` still apply.
3. `opencode run --agent <director|solo> --format json --auto --dir <copy> "<task.md>"` with stdin closed,
   isolated `XDG_DATA_HOME` (own session DB), wall-clock timeout (default 2× `timeout_s`).
4. Overwrites `test_files` with the gold versions, runs `setup_cmd` + `test_cmd` in `workdir` → **pass@1**.
5. Exports all sessions (parent + sub-agents) → tokens per agent, LLM calls, tool calls, delegations,
   validator verdicts / fix rounds, test runs; GPU util from `nvidia-smi` (2 s); vLLM `/metrics` token deltas.

Outputs: `/srv/work/w2/uc15/results/<run_id>/{results.jsonl,summary.md,meta.json,runs/<cfg>/<task>/…}`
(`transcript.jsonl`, `agent.diff`, `sessions.json`, `harness.log`). Copy the final JSON to
`bench/results/UC15_<run_id>.json` for the report.

## Configurations

| config | entry agent | sub-agents allowed | validator |
|---|---|---|---|
| `solo` | `solo` (= `coder-<lang>` promoted to primary, 80 steps) | — | no |
| `solo+val` | `solo` | `validator` | yes |
| `dc` | `director` | `coder-*` | no |
| `dc+val` | `director` | `coder-*`, `validator` | yes |
| `swarm` | `director` | `coder-*`, `fetcher`, `researcher` | no |
| `swarm+val` | `director` | all | yes |

Model matrix = env preset (`~/opencode-swarm/env/*.env`): `single-model.env` (all roles Qwen3-Coder — the only
option on one L40S), `split-models.env` (validator/researcher on Qwen3-32B — needs a second endpoint),
`devstral.env`, … The validator-family experiment (§5 point 5) needs `split-models.env` with a second GPU/host,
or a two-pass protocol: run the coder phase with `single-model.env`, then re-validate the saved diffs with the
32B model in a profile-A window (not automated yet).

## Prerequisites on the VM (done 2026-10-27)

- `~/.local/bin/opencode` 1.19.4 (linux-x64 binary from the npm registry, no Node needed)
- toolchains + clones: `tasks/env.sh`, `/srv/work/w2/uc15/{repos,tools}` (Go, Dart SDK 3.13, venv with pytest)
- harness self-test: `python3 run_bench.py --check-gold --tasks all` → 25/25 (base fails, gold passes)
- network: OpenCode npm-installs `@opencode-ai/plugin` (~60 MB) into each config dir on first use; the harness
  keeps one complete copy in `results/_plugin_cache` and symlinks it, and never leaves `node_modules` in run dirs.
- disk: `/srv/work/w2/uc15` ≈ 3.2 GB (toolchains 3.0 GB); a run dir without `--keep` ≈ 0.2–1 MB.

## Exact command for the full benchmark (profile W = Qwen3-Coder on :8003)

```bash
ssh tcloud
cat /srv/work/.gpu_profile                        # must say W
curl -s 127.0.0.1:8003/v1/models | grep qwen3-coder
cd ~/tmobile-sandbox/uc15/bench
tmux new -s uc15 \
 "PATH=\$HOME/.local/bin:\$PATH python3 run_bench.py \
    --env ~/opencode-swarm/env/single-model.env --swarm ~/opencode-swarm \
    --tasks all --configs solo,dc+val,swarm+val --parallel 4 --timeout 1500 \
    --run-id w-qwen3coder-\$(date +%m%d) 2>&1 | tee /srv/work/w2/uc15/results/w-run.log"
```

Then the remaining three configs (`--configs solo+val,dc,swarm`) with another `--run-id` when there is time.
Quick sanity run first (≈15 min): `--tasks easy --configs solo,dc+val --parallel 2`.

`--parallel 4`: four agents in flight → vLLM batches them (that is where the swarm's GPU utilisation comes from);
RAM headroom (23 GB, memguard at 1.5 GB) limits it — Go/Dart test runs take 1–3 GB each.

## Time estimate (GPU window W)

Assumptions: Qwen3-Coder-30B-A3B FP8 on L40S ≈ 80–100 tok/s decode per stream, prefix caching makes the growing
agent context cheap; a solo run ≈ 25–50 LLM calls, ~8–15k output tokens, plus 1–3 min of test/build time;
director + validator ≈ 1.6–2× solo.

| scope | runs | sequential | `--parallel 4` |
|---|---|---|---|
| sanity (9 easy × 2 configs) | 18 | ~2 h | ~40 min |
| 25 tasks × 3 configs (`solo`, `dc+val`, `swarm+val`) | 75 | ~10–12 h | **~3–4 h** |
| 25 tasks × 6 configs | 150 | ~20–24 h | ~6–8 h |

Worst case is bounded by `--timeout 1500` per run. Tokens: expect ~0.3–1.5 M input (mostly cached) and
10–40 k output per run.

## Claude Code as the upper bound (NOT run — the user decides)

Same working copies and hidden tests, only the agent command differs. Prepare the copies with `--dry-run --keep`
(creates `runs/<cfg>/<task>/repo` without calling a model), then per task:

```bash
cd /srv/work/w2/uc15/results/<run_id>/runs/solo/<task>/repo
claude -p "$(cat ~/tmobile-sandbox/uc15/bench/tasks/<task>/task.md)" \
  --permission-mode acceptEdits --allowedTools "Bash,Edit,Write,Read,Glob,Grep" \
  --output-format json > ../claude.json     # total_cost_usd / usage / num_turns are in the JSON
```

and score it with the same restore-tests + `test_cmd` step (`restore_tests_and_check` in `run_bench.py`).
Runs on the Mac (Max subscription) are equivalent if the repos are cloned at the same `base_commit`.

## Smoke test (2026-10-27, profile L — Qwen3-30B-A3B, 4k context, no tool parser)

Run through `opencode-swarm/scripts/toolshim.py` (forces `tool_choice=none`, parses Hermes `<tool_call>` text,
re-emits SSE, truncates tool descriptions to fit 4k). Results in `smoke/` and the final report of session w2-uc15.
