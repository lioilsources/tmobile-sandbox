# Ekonomika L40S na jednotku práce

*Podklad pro pitch T‑Mobile · 1× NVIDIA L40S 48 GB · předpokládaný náklad celé VM ≈ 55 Kč/h · stav 8. 11. 2026*

Všechna čísla jsou naměřená v sandboxu (zdroj ve sloupci Report). Cena = 55 Kč/h ÷ naměřená propustnost při plném vytížení, bez marže a bez nečinnosti. Energie jen tam, kde ji běh logoval (DCGM / `nvidia-smi`).

| Úloha | Propustnost / h | Kč za jednotku | Kč za 1 000 | Energie | Model (licence) | Report |
|---|---|---|---|---|---|---|
| Strana faktury/smlouvy → JSON, 110 dpi | 1 529 stran | 0,036 | 36 | — | Qwen3‑VL‑32B FP8 (Apache 2.0) | `demo3-dokumenty.md` |
| Strana faktury/smlouvy → JSON, 150 dpi | 944 stran | 0,058 | 58 | — | Qwen3‑VL‑32B FP8 (Apache 2.0) | `demo3-dokumenty.md` |
| Hovor: přepis + 6 štítků | 1 114–1 216 hovorů | 0,045–0,049 | 45–49 | — | Whisper large‑v3‑turbo (MIT) + Qwen3‑30B‑A3B (Apache 2.0) | `demo1-callcentrum.md` |
| Smlouva z Registru smluv: klasifikace + popis | 162 800 smluv | 0,00034 | 0,34 | 0,0016 Wh/smlouva | Qwen3‑30B‑A3B FP8 (Apache 2.0) | `uc9-registr-smluv.md` |
| Šachová pozice ohodnocená sítí | 17,6 mil. pozic | 0,000003 | 0,003 | 110 Wh za 6,16 mil. | lc0 BT4 (GPL‑3.0) | `uc3-sachove-kvizy.md` |
| Ilustrace 1024², 4 kroky | 1 240 obr. | 0,044 | 44 | 0,22 Wh/obr. | FLUX.1‑schnell fp8 (Apache 2.0) | `uc10-storyteller-styly.md` |
| Herní předmět obrázek → 3D mesh | 105 ks | 0,52 | 524 | 1,85 Wh/ks | FLUX.1‑schnell + Hunyuan3D 2.1 (licence Tencent, územní omezení ověřit) | `uc8-ugc-3d.md` |
| Doladění VLM na OCR (QLoRA, 180 půlstran) | 11–17 min/model | ≈ 15 na model | — | — | Qwen2.5‑VL‑7B (Apache 2.0) | `uc1-genealogie-ocr.md` |

## Srovnání s veřejným API (jediné naměřené)

UC9: stejný objem tokenů (470 M vstupních, 54 M výstupních), kurz 23 Kč/USD.

| Kde | Náklad |
|---|---|
| **L40S, 3,5 h VM** | **≈ 192 Kč** |
| Claude Haiku 5.5, Batch API | ≈ 850 Kč |
| Claude Haiku 5.5, standardně | ≈ 1 700 Kč |
| Claude Sonnet 5.5, Batch API | ≈ 17 000 Kč |

Pozor při prezentaci: kvalita odpovědí není stejná (Haiku/Sonnet jsou silnější modely); srovnání ukazuje řád nákladu dávkového zpracování, ne ekvivalent.

## Co cenu posouvá

- **Vytížení.** Čísla platí při plném vytížení. GPU, která čeká, stojí stejně: proto dispečer front s přerušitelnými výplněmi (`queue/qctl`).
- **Architektura modelu.** MoE Qwen3‑30B‑A3B dává v dávce ~10× víc rozhodnutí/s než dense Qwen3‑32B (UC11: 35,8 vs. 2,8 /s).
- **Úzká hrdla mimo GPU.** Demo 1 brzdí Whisper (CPU dekódování, jedna instance); UC12 self‑play brzdí 8 sdílených vCPU (GPU 65–75 %).
- **Rozlišení vstupu.** Dokumenty 110 dpi místo 150 dpi: −39 % tokenů na stranu, +62 % propustnosti, přesnost 100 → 99,75 %.
