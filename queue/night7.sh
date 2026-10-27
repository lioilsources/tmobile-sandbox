#!/usr/bin/env bash
# Druhá vlna, noc 1: po doběhnutí gridu UC11 → UC15 benchmark (profil W) → UC11 model vs model (A, pak S).
set -uo pipefail
R=~/tmobile-sandbox; Q=/srv/work/queue; U11=/srv/work/w2/uc11
log() { echo "$(date -Is) $*" | tee -a $Q/night7.log; }
while pgrep -f '^bash code/run_grid.sh grid30b' >/dev/null; do sleep 60; done
log "UC11 grid hotový ($(wc -l < $U11/runs/grid30b_results.jsonl) běhů)"
(cd $U11 && code/finish.sh runs/grid30b_results.jsonl qwen3-30b-a3b) >> $Q/night7.log 2>&1; log "UC11 finish exit $?"

$R/50_vllm/switch.sh W >/dev/null 2>&1; log "profil $(cat /srv/work/.gpu_profile)"
mkdir -p /srv/work/w2/uc15/results
cd $R/uc15/bench
for C in solo,dc+val,swarm+val solo+val,dc,swarm; do
  PATH=$HOME/.local/bin:$PATH timeout 5h python3 run_bench.py --env ~/opencode-swarm/env/single-model.env --swarm ~/opencode-swarm \
    --tasks all --configs $C --parallel 4 --timeout 1500 --run-id w-qwen3coder-$(date +%m%d)-${C//[,+]/_} \
    >> /srv/work/w2/uc15/results/w-run.log 2>&1
  log "UC15 $C exit $?"
done

for P in A:qwen3-32b S:qwen3-8b; do
  $R/50_vllm/switch.sh ${P%%:*} >/dev/null 2>&1; log "profil $(cat /srv/work/.gpu_profile)"
  (cd $U11 && timeout 6h code/model_vs_model.sh ${P#*:} 5 48) >> $Q/night7.log 2>&1
  log "UC11 mvm ${P#*:} exit $?"
done
$R/50_vllm/switch.sh off >/dev/null 2>&1
log NIGHT7_DONE
