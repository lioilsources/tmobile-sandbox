# qctl — dispečer GPU fronty na L40S

Nahrazuje ruční `dayNN.sh` skripty. Pravidlo priority:

- **Krátké úlohy první.** Úlohy, které nejsou výplň, běží podle `est_min` vzestupně, při shodě podle času zařazení. Běžící krátká úloha se nepřerušuje.
- **Dlouhé výplně vyplňují volno.** Úloha s `"filler": true` (AlphaZero, dlouhý trénink) startuje jen tehdy, když nic jiného nečeká. Když přijde krátká úloha, dostane výplň SIGTERM celé skupině procesů, uloží checkpoint, skončí a vrátí se do fronty. Po 15 min bez reakce přijde SIGKILL.

## Druhy úloh

- `cmd`: qctl přepne profil (`50_vllm/switch.sh`) a spustí `cmd` v bash. Výstup jde do `logs/<name>.log`, exit 0 znamená done, jinak failed.
- `window`: okno pro jinou session. qctl přepne profil, počká na `health`, zapíše marker `ready` a čeká, až session zapíše marker `done` nebo vyprší `guard_min` (výchozí 2× `est_min`). Markery jsou kompatibilní s dosavadními (`RESTYLER_READY` / `RESTYLER_DONE`). Volitelný `cleanup` (bash) se spustí po zavření okna, třeba smazání fotek hostů.
- Libovolná úloha může mít `not_before` (RFC 3339). Do té doby se neplánuje ani nepřerušuje výplň.

## Použití (na VM)

```
qctl add job.json          # nebo: qctl add - < job.json
qctl ls                    # running / pending (v pořadí) / posledních 8 done a failed
qctl done <name>           # zavře okno (zapíše done marker)
qctl cancel <name>
touch /srv/work/queue/HOLD # nic nového nestartuje, běžící doběhne
```

Stav je v `/srv/work/queue/jobs/{pending,running,done,failed}/`, log démona v `/srv/work/queue/qctl.log`. Démon běží jako systemd user služba `qctl`. Po restartu převezme běžící úlohy podle PID a exit kódu v `logs/<name>.exit`.

## Příklady

```json
{"name": "restyler-hair-r5", "kind": "window", "profile": "D", "est_min": 90,
 "ready": "RESTYLER_READY", "done": "RESTYLER_DONE", "health": "http://127.0.0.1:8188/system_stats"}
```

```json
{"name": "az-filler", "kind": "cmd", "profile": "off", "est_min": 600, "filler": true,
 "cmd": "cd ~/tmobile-sandbox && uc12/run_az.sh 10"}
```

Výplň musí umět SIGTERM → checkpoint → resume, jinak při přerušení přijde o práci.

## Build

```
cd queue/qctl && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o qctl-linux .
```
