#!/usr/bin/env bash
# Denní záloha /srv/work → NAS. Běží NA NAS (v domácí síti; VM na něj nevidí,
# NAS na VM ano) jako pull přes rsync. Klíč ~/.ssh/tmobile_backup je na VM omezený na rrsync -ro /srv/work.
# Instalace na NAS:  scp 80_backup.sh <nas>:bin/tmobile-backup.sh  (VM a DST v ~/.config/tmobile-backup.env)
#   crontab:  30 3 * * *  ~/bin/tmobile-backup.sh >> ~/tmobile-backup.log 2>&1
set -euo pipefail
[ -f ~/.config/tmobile-backup.env ] && . ~/.config/tmobile-backup.env
VM=${VM:?nastav VM=user@host v ~/.config/tmobile-backup.env}
DST=${DST:?nastav DST=/cesta/k/zaloze}
mkdir -p "$DST/work" "$DST/snapshots"
echo "== $(date -Is)"
rsync -a --delete --stats -e "ssh -i $HOME/.ssh/tmobile_backup -o BatchMode=yes" \
  --exclude '/AiStack/' --exclude 'venv/' --exclude '*-venv/' --exclude '__pycache__/' --delete-excluded "$VM:" "$DST/work/" | grep -E "Number of (regular )?files transferred|Total transferred file size"
# týdenní hardlink snapshot (neděle), drží 8 týdnů
if [ "$(date +%u)" = 7 ]; then
  cp -al "$DST/work" "$DST/snapshots/$(date +%F)"
  ls -1d "$DST"/snapshots/* | head -n -8 | xargs -r rm -rf
fi
