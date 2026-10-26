# Demo 2 – AI API pro více tenantů (L40S sandbox)

*25. 10. 2026 · T‑Mobile GPUaaS sandbox, 1× NVIDIA L40S 48 GB · vLLM 0.32.0 · LiteLLM 1.104.2 · AiStack gateway + tenanti*

## Shrnutí
Jeden L40S obslouží tři zákazníky (tenanty) přes jedno OpenAI‑kompatibilní API s vlastními klíči, limity a účtováním po tokenech.
Při 6minutovém testu (≈ 2 250 požadavků, 1,9 mil. vstupních a 360 tis. výstupních tokenů, 14 min audia) zůstal **zákaznický chatbot
svižný i během náporu dávkového tenanta**: první token do **0,13 s (p95)**. Limit souběhu tenanta C nápor zastavil na hranici 16 požadavků;
nadbytečné požadavky dostaly HTTP 429 a opakovaly se. Vyúčtování se počítá z knihy gateway na haléře po tenantech a modelech.

## Architektura
```
klient ─► gateway (AiStack + tenanti) ─► LiteLLM ─► vLLM: chat (Qwen3‑30B‑A3B‑FP8), embed (Qwen3‑Embedding‑0.6B)
           │  API klíč → tenant                 └─► Whisper large‑v3‑turbo (speaches)
           │  povolené modely, limit souběhu (429)
           │  měření tokenů vč. streamu, audio sekundy
           ├─► kniha usage.jsonl (účtování)
           └─► /metrics → Prometheus → Grafana „AI API – tenanti“
```
- Gateway = AiStack gateway (Go, bez závislostí) rozšířená o `tenants.go`: klíče, oprávnění na modely, limit souběhu, doplnění
  `stream_options.include_usage` pro měření streamu, kniha a metriky `aigw_*`. Klíč tenanta nejde dál – upstream dostane master klíč LiteLLM.
- Testy: neplatný klíč (401), zakázaný model (403), měření JSON i SSE, cena, metriky, regresní test průběžného streamu.

## Tenanti (ilustrativní ceník)
| Tenant | Typ provozu | Modely | Limit souběhu |
|---|---|---|---|
| A‑chatbot | zákaznický chat, stream, 4 req/s | chat, embed | 48 |
| B‑rag | interní vyhledávání: embeddingy po 32 textech + dotazy s ~3 000 tokeny kontextu, 2 req/s | chat, embed | 32 |
| C‑batch | dávky: dlouhé reporty (800 tokenů) + přepisy hovorů, nárazově 40 souběžně | chat, whisper | 16 |

Ceník (Kč, ilustrativní): chat 5 / 1M vstup, 20 / 1M výstup · embed 0,50 / 1M · whisper 0,10 / min.

## Load test (3 fáze po 2 minutách)
Fáze 1 a 3: A + B. Fáze 2: navíc nápor C (40 souběžných úloh, limit 16). Před měřením zahřátí.

| Tenant / fáze | Požadavků | Chyb | TTFT p50 / p95 | Doba p50 / p95 |
|---|---|---|---|---|
| A‑chatbot · 1 klid | 489 | 0 | 74 / 94 ms | 4,0 / 6,6 s |
| **A‑chatbot · 2 nápor C** | 482 | 0 | **98 / 133 ms** | 5,1 / 8,5 s |
| A‑chatbot · 3 klid | 481 | 0 | 89 / 112 ms | 4,9 / 7,8 s |
| B‑rag · 1 / 2 / 3 | 220 / 218 / 237 | 0 | – | p95 4,8 / 6,3 / 5,6 s |
| C‑batch · 2 | 114 | 24 × 429 | – | 25 / 159 s |

- Během náporu C se chatbotu A prodloužila doba odpovědi o ~25 %, první token zůstal pod 0,15 s. B téměř beze změny.
- C držel přesně 16 souběžných požadavků; 2 205× dostal 429 (≈ 15 odmítnutí/s), 24 jeho úloh se nevešlo ani po 60 opakováních – limit funguje jako pojistka, ne jako fronta.
- GPU 93–100 % po celou dobu, KV cache max 26 % → úzkým místem je výpočet, ne paměť. Průměrně ~1 000 výstupních tokenů/s.

## Vyúčtování testu (6 min)
| Tenant | Model | Požadavků | Vstup tok. | Výstup tok. | Audio min | Kč |
|---|---|---|---|---|---|---|
| A‑chatbot | chat | 1 452 | 74 914 | 252 751 | – | 5,43 |
| B‑rag | chat | 336 | 1 360 569 | 44 061 | – | 7,68 |
| B‑rag | embed | 339 | 437 658 | – | – | 0,22 |
| C‑batch | chat | 80 | 4 232 | 64 000 | – | 1,30 |
| C‑batch | whisper | 10 | – | – | 13,7 | 1,37 |
| **Celkem** | | **2 217** | **1 877 373** | **360 812** | **13,7** | **16,00** |

Při tomto provozu a ilustrativním ceníku ≈ 160 Kč/h tržeb na jednu GPU proti nákladu ≈ 55 Kč/h. Počítáme s předpokládaným nákladem asi 55 Kč za hodinu za celou VM (GPU, 8 vCPU, 23 GB RAM, disky). S větší RAM nebo diskem cena roste.

## Incident během testu
Při prvním plném běhu se VM jednou tvrdě zastavila (bez OOM, kernel panic i chyby GPU v logu; RAM 13,5/23 GB, GPU 49 °C, 250 W)
a sama nenaběhla. Opakovaný test se záznamem stavu každé 2 s doběhl bez problému. Příčina je pravděpodobně mimo VM.

## Omezení a další kroky
- Ceník je ilustrativní.
- Limit souběhu je jednoduchá pojistka; pro produkci férové frontování (vážené fronty / priority) a rate‑limit na tokeny/min.
- Klíče jsou v souboru na VM; produkčně správa klíčů (rotace, hashované klíče) a TLS přístup místo SSH tunelu.
- Rozšíření gateway vrátit do AiStacku (PR).

*Reprodukce:* `~/tmobile-sandbox/56_api/up.sh` a `~/tmobile-sandbox/demos/02_api/run.sh 120` (profil B). Kniha: `/srv/work/demo2/usage.jsonl`.
