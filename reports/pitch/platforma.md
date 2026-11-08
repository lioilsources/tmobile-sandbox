# Zpětná vazba k platformě T‑Mobile GPUaaS

*Podklad pro pitch a technický workshop · 14 dní nepřetržitého provozu jedné VM s L40S · stav 8. 11. 2026*

GPU sama fungovala bez výhrad. Limity, na které jsme narazili, jsou všechny kolem ní.

| Co se stalo | Dopad na běh | Doporučení | Priorita |
|---|---|---|---|
| 23 GB RAM ke 48 GB GPU | Wan 2.2 video a dvě RAM‑těžké úlohy naráz končily OOM nebo ve swapu; ComfyUI musí běžet s `--disable-pinned-memory` (RAM 22 → 5,4 GB); na party 6. 11. nutný dočasný 32 GB swap | 64 GB RAM jako výchozí poměr k jedné 48 GB GPU, volitelně 128 GB | vysoká |
| 8 sdílených vCPU, ~1 jádro steal | AlphaZero self‑play a příprava dat drží GPU na 65–75 %; Whisper je v Demu 1 úzké hrdlo; trénink malých sítí s 8 vlákny 5× pomalejší než se 4 | dedikovaná jádra; varianta 16 vCPU pro GPU VM | vysoká |
| VM se 25. 10. pod plnou zátěží tvrdě zastavila; v journalu žádná stopa (OOM ani Xid), portál „stopped“ | ztráta rozpracované dávky; od té doby musí každá dávka umět navázat | důvod zastavení viditelný v portálu; snapshoty VM; notifikace | vysoká |
| Veřejná IP se po zásahu v portálu změnila | rozbité ssh aliasy, tunely a zálohy | statická IP nebo DNS jméno VM | střední |
| Bez virtio kanálu pro guest agenta | portál neumí čisté vypnutí ani zjištění stavu hosta | guest agent v šabloně VM | střední |
| Modely nad ~40 GB se nevejdou | vše v FP8; 70B modely a delší video jen s offloadem | 2× L40S nebo H100 ve stejné VM jako volba | střední |
| Docker obchází ufw | port publikovaný z compose je venku i přes firewall | v dokumentaci a šabloně bind na `127.0.0.1`, ven jen přes VPN/tunel | nízká |
| CloudSigma Debian šablona: `sshd_config` nenačítá `sshd_config.d/`, chybí `rsync` | zdržení při zakládání pomocné VM | sjednotit šablony s Ubuntu | nízká |

## Co fungovalo dobře

- **Výkon GPU je stabilní.** 8 h soak (9 912 ilustrací) bez změny výkonu, max 59 °C, 0 s tepelného omezení; limit je příkon 350 W (= maximum karty).
- **Disky.** Druhý disk pro modely a data i pomalejší 1 TB „cold“ disk stačily na 20+ modelů a datasety.
- **Síť.** Virtuální router (NAT, firewall, OpenVPN) a VLAN daly přístup bez veřejné IP na GPU VM.

## Co můžeme předat

- Provisioning skripty `00`–`70` jako základ šablony „AI‑ready L40S“ (čistá VM → vLLM + ComfyUI + monitoring).
- Pět ověřených profilů vLLM/ComfyUI (`50_vllm/switch.sh`) jako šablony v portálu.
- Dispečer GPU fronty `queue/qctl` (okna, dávky, přerušitelné výplně).
- Grafana dashboardy (GPU, tenanti API) a `MODEL_NOTES.md` jako provozní příručku.
