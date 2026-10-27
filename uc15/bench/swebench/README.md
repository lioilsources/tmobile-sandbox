# SWE-bench Lite — 50-instance Python subset (prepared, not downloaded)

`swebench_lite_50.txt` — instance ids; `swebench_lite_50.json` — repo, version, base commit, gold patch size.

**Selection:** seeded random sample (`random.Random(15)`) from SWE-bench Lite *test* (300 instances), restricted
to **10 (repo, version) environments** to keep the Docker footprint small:
django 3.1 / 3.2 / 4.0 (7 each) + 4.1 (4), sympy 1.1 (8), pytest 5.4 (4), scikit-learn 0.22 (4),
matplotlib 3.6 (3), sphinx 3.5 (3), pylint 2.15 (3). Django share (50 %) is a bit above Lite's 38 %; pass@1 on
this subset is therefore comparable to published Lite numbers only roughly — report it as "Lite-50 subset".

## Disk estimate (x86_64 images, SWE-bench harness ≥ 2.x)

| layer | count | size each | total |
|---|---|---|---|
| base image `sweb.base.py.x86_64` | 1 | ~1.5–3 GB | ~3 GB |
| env images `sweb.env.*` (conda env per repo/version) | 10 | ~1.5–2.5 GB (django, sympy, pytest, pylint, sphinx); ~3–4 GB (sklearn, matplotlib) | **~25–30 GB** |
| instance images `sweb.eval.*` (repo at base commit) | 50 | ~0.1–0.6 GB | ~10–20 GB |
| build cache | — | — | +5–10 GB |

All cached: **~45–60 GB**. The VM has ~29 GB free on `/data`, and Docker's root is `/srv/docker` on that disk →
**does not fit with default caching**. Feasible variants:

1. **Grouped, no cache (fits ~10–12 GB peak):** evaluate one environment group at a time with
   `--cache_level base --clean True`; env + instance images are removed after each group. Costs ~10 env builds
   (5–15 min each, CPU) ≈ 1.5–2.5 h extra, no GPU needed.
2. Move Docker root to the 250 GB OS disk (`/var/lib/docker` on vda) — infrastructure change, **user/Director
   decision** (this session must not touch Docker config).
3. `sb-cli` (SWE-bench cloud evaluation): only predictions (patches against public repos) leave the VM, no local
   images at all; needs a free SWE-bench API key from the user.

## Procedure

**A. Generate patches with the swarm (GPU window W, no Docker):**
1. Dataset metadata: `pip install datasets swebench` into a venv under `/srv/work/w2/uc15/tools/swe-venv`
   (~300 MB), `load_dataset("princeton-nlp/SWE-bench_Lite", split="test")`, filter by `swebench_lite_50.txt`.
2. Per instance: `git clone` the repo once into `/srv/work/w2/uc15/swe/repos/<owner__repo>` (all 7 repos ≈ 1.5 GB),
   `git worktree add` at `base_commit`, write `problem_statement` to `task.md`.
3. Run the agent exactly like `run_bench.py` does (`opencode run --agent director|solo --format json --auto`,
   isolated `XDG_DATA_HOME`, timeout 1500 s), then `git diff base_commit` → one line in `predictions.jsonl`:
   `{"instance_id": …, "model_name_or_path": "opencode-swarm/qwen3-coder/<config>", "model_patch": "<diff>"}`.
   Do **not** give the agent `hints_text` or the test patch (standard SWE-bench protocol).
   No project toolchain is installed in the worktree, so agents can read but not run the repo's tests — the same
   handicap every SWE-bench agent without the Docker env has; alternatively run the agent *inside* the instance
   image (variant 2/3 disk budget).
4. GPU time: ~5–10 min per instance (django/sympy contexts are large) → 50 × 1 config ≈ 5–8 h sequential,
   ~2–3 h with 4 in parallel.

**B. Evaluate (CPU + Docker, variant 1):**
```bash
for env in $(jq -r '.[] | "\(.repo)@\(.version)"' swebench_lite_50.json | sort -u); do
  ids=$(jq -r --arg e "$env" '.[] | select("\(.repo)@\(.version)"==$e) | .instance_id' swebench_lite_50.json)
  python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path predictions.jsonl --instance_ids $ids --max_workers 2 \
    --cache_level base --clean True --run_id uc15-$(echo $env | tr '/@.' '---')
done
```
`--max_workers 2` because of 23 GB RAM / memguard. Reports land in `logs/run_evaluation/…` and
`<model>.<run_id>.json` (resolved ids) — merge into `bench/results/UC15_swebench_lite50.json`.
