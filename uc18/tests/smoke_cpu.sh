#!/usr/bin/env bash
# UC18 + UC19 end-to-end CPU smoke test on the VM — touches NO GPU (CUDA_VISIBLE_DEVICES=""), no docker, no ComfyUI.
# Fake ComfyUI (real smoke_flux PNGs) + fake vLLM; real scorers (WD14 / SigLIP 2 / PickScore) on CPU, few images.
#
#   bash ~/tmobile-sandbox/uc18/tests/smoke_cpu.sh [OUT=/cold/w2/smoke_uc18]
set -uo pipefail
T=$(cd "$(dirname "$0")" && pwd); R=$T/..; U19=$R/../uc19
S=${1:-/cold/w2/smoke_uc18}; rm -rf "$S"; mkdir -p "$S"
export CUDA_VISIBLE_DEVICES="" HF_HOME=/srv/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/cold/venvs/uc25/bin/python
fail=0; step() { local n=$1; shift; local t0=$SECONDS; "$@" >"$S/$n.out" 2>"$S/$n.err"; local rc=$?
  printf '%-28s rc=%s %4ss  %s\n' "$n" "$rc" "$((SECONDS - t0))" "$(tail -1 "$S/$n.err" | cut -c1-150)"; [ $rc = 0 ] || fail=1; return $rc; }

step unit            $PY "$T/test_uc18.py"
step judge_schema    $PY "$R/vlm_judge.py" schema-check
step dry_uc18_full   python3 "$R/gen_matrix.py" --prompts "$R/prompts/uc18_prompts.jsonl" --ckpts all --seeds 1,2 --dry-run \
                       --object-info "$R/object_info.json" --out-dir "$S/x"
$PY "$T/fake_comfy.py" --port 18188 --object-info "$R/object_info.json" --exec-s 0.05 --out "$S/comfyout" \
    --images-glob '/srv/work/comfy-out/smoke_flux_*.png' & FC=$!
$PY "$T/fake_vllm.py" --port 18004 --model qwen3-vl-8b & FV=$!
$PY "$T/fake_vllm.py" --port 18000 --model qwen3-8b & FL=$!
trap 'kill $FC $FV $FL 2>/dev/null' EXIT
sleep 2
U=$S/uc18
step gen_uc18        python3 "$R/gen_matrix.py" --prompts "$R/prompts/uc18_prompts.jsonl" --ckpts pony,juggernaut,flux_dev \
                       --seeds 1 --styles photo_portrait,animals --limit 4 --out-dir "$U/img" --manifest "$U/manifest.jsonl" \
                       --api http://127.0.0.1:18188 --comfy-out "$S/comfyout" --health-s 2 --min-util 0
step gen_uc18_resume python3 "$R/gen_matrix.py" --prompts "$R/prompts/uc18_prompts.jsonl" --ckpts pony,juggernaut,flux_dev \
                       --seeds 1 --styles photo_portrait,animals --limit 4 --out-dir "$U/img" --manifest "$U/manifest.jsonl" \
                       --api http://127.0.0.1:18188
grep -q "4 done\|12 done" "$S/gen_uc18_resume.err" || { echo "   resume did not skip"; fail=1; }
for sc in wd14 siglip pick; do
  extra=(); [ $sc = siglip ] && extra=(--text-emb-out "$U/scores/text_emb.npz")
  step score_$sc     nice $PY "$R/score_images.py" --manifest "$U/manifest.jsonl" --refs "$R/prompts/uc18_prompts.jsonl" \
                       --scorer $sc --out "$U/scores/$sc.jsonl" --device cpu --batch 4 "${extra[@]}"
done
step judge_dry       $PY "$R/vlm_judge.py" pair --manifest "$U/manifest.jsonl" --refs "$R/prompts/uc18_prompts.jsonl" --out "$U/scores/x.jsonl" --dry-run
step judge_pair      $PY "$R/vlm_judge.py" pair --manifest "$U/manifest.jsonl" --refs "$R/prompts/uc18_prompts.jsonl" \
                       --out "$U/scores/vlm_pairs.jsonl" --base-url http://127.0.0.1:18004/v1 --swap-frac 0.5 --concurrency 4
step rank            $PY "$R/rank.py" --dir "$U" --out "$U/rank.json" --router-out "$U/router.json" --bootstrap 30
step report          python3 "$R/report.py" --rank "$U/rank.json" --dir "$U" --md "$U/report.md" --json "$U/UC18_smoke.json"
step route_pid       $PY "$R/route.py" --router "$U/router.json" --pid photo_portrait-01

V=$S/uc19
step dry_translate   $PY "$U19/translate.py" --cz "$U19/prompts/cz_prompts.jsonl" --fewshot "$U19/prompts/fewshot.jsonl" --out "$V/t.jsonl" --dry-run
step translate       $PY "$U19/translate.py" --cz "$U19/prompts/cz_prompts.jsonl" --fewshot "$U19/prompts/fewshot.jsonl" \
                       --out "$V/translations.jsonl" --base-url http://127.0.0.1:18000/v1 --limit 6
step build_jobs      python3 "$U19/build_jobs.py" --translations "$V/translations.jsonl" --out "$V/jobs.jsonl" --t1-prefix
step gen_uc19        python3 "$R/gen_matrix.py" --jobs "$V/jobs.jsonl" --out-dir "$V/img" --manifest "$V/manifest.jsonl" \
                       --api http://127.0.0.1:18188 --comfy-prefix uc19 --health-s 2 --min-util 0
for sc in wd14 siglip pick; do
  step uc19_score_$sc nice $PY "$R/score_images.py" --manifest "$V/manifest.jsonl" --refs "$U19/prompts/cz_prompts.jsonl" \
                       --scorer $sc --out "$V/scores/$sc.jsonl" --device cpu --batch 4 --limit 12
done
step uc19_judge      $PY "$R/vlm_judge.py" single --manifest "$V/manifest.jsonl" --refs "$U19/prompts/cz_prompts.jsonl" \
                       --out "$V/scores/vlm_single.jsonl" --base-url http://127.0.0.1:18004/v1 --concurrency 4
step uc19_report     python3 "$U19/report.py" --dir "$V" --md "$V/report.md" --json "$V/UC19_smoke.json"
# negative paths: vLLM 400 with body → first error printed, abort; ComfyUI down → exit 8/3
echo '{"cid":"cz999","category":"plain","style":"animals","cz":"FAIL400","ref_en":"x","must_tags":[],"must_not":[]}' > "$V/bad.jsonl"
$PY "$U19/translate.py" --cz "$V/bad.jsonl" --fewshot "$U19/prompts/fewshot.jsonl" --out "$V/bad_out.jsonl" \
    --base-url http://127.0.0.1:18000/v1 --techniques t1 2> "$V/bad.err"
grep -q "first error.*HTTP 400.*fake: schema rejected" "$V/bad.err" && echo "error-body logging           ok" || { echo "error-body logging FAIL"; fail=1; }
python3 "$R/gen_matrix.py" --prompts "$R/prompts/uc18_prompts.jsonl" --ckpts pony --limit 1 --out-dir "$S/down" \
    --manifest "$S/down/m.jsonl" --api http://127.0.0.1:18999 2>/dev/null; rc=$?
[ $rc = 8 ] && echo "comfy-down exit 8            ok" || { echo "comfy-down exit $rc FAIL"; fail=1; }
echo "smoke dir: $S"; [ $fail = 0 ] && echo "SMOKE OK" || echo "SMOKE FAILED"; exit $fail
