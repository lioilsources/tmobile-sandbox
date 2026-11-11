# UC12 · OrbitronTactics AlphaZero na L40S: rozhodnutí

AlphaZero (JAX + mctx, Gumbel MuZero s pravidly jako modelem) se učil OrbitronTactics hrou sám se sebou jako výplň GPU. Síla se měřila Elo ligou proti dartovým SearchAi Easy/Medium/Hard (Easy = 0). Každá liga je na CPU a má 2 700 partií: 8 checkpointů × 3 úrovně × 100 partií a kalibraci úrovní mezi sebou.

## Výsledek

| Síť | GPU h | Elo nejlepšího checkpointu | 95% CI | Hard ve stejné lize |
|---|---|---|---|---|
| 6×128, 32 simulací | 71,6 | **+43** (net_032070) | −1 až +84 | +35 |
| 6×128, 64 simulací (navázáno) | 60,1 | +37 (net_023073) | −4 až +82 | +19 |
| 8×256, 32 simulací | 31,1 | +8 (net_008467) | −36 až +49 | +39 |

Sítě se po zhruba 15 GPU h zastavily mezi Easy a Hard. Delší trénink nepomohl:
- 6×128 skončila na +38 po 82 GPU h.
- 8×256 stojí od 13 GPU h kolem nuly.

Větší síť se tedy nedostala nad menší. Brzdí nejspíš signál ze self-play (náhodné bitvy při braní), ne kapacita sítě.

## Rozhodnutí

Kritérium plánu 08 bylo Elo ≥ Hard + 50 s CI nad nulou. Žádná síť ho nesplnila, takže **UC12 končí**:
- Výplň `az-8x256-b` je zrušená.
- Další GPU hodiny jdou na jiné úlohy (F3 UC18/19, F1 UC27, až bude dataset).

## Export

Nejlepší checkpoint je **net_032070** (6×128, iterace 32 070, 71,6 GPU h). Je exportovaný do ONNX pro Flutter skriptem `uc12/export.py`:
- BatchNorm je složený do konvolucí.
- Vstup: `obs [batch, 8, 8, 58]` (NHWC, jako `observe()`).
- Výstupy:
  - `logits [batch, 2560]` v kanonickém rámci. Maska legálních tahů a převod `to_real` pro černého patří aplikaci.
  - `value [batch]` (tanh).
- Soubor `/cold/w2/uc12/export/orbitron_az_032070.onnx` má 7,5 MB, popis je v `.json` vedle.
- Shoda s flax sítí na 16 náhodných vstupech: max. odchylka 6,9·10⁻⁵ (logits) a 1,6·10⁻⁵ (value).

Pro hru v appce je potřeba i MCTS nad sítí. Samotná policy síť bez hledání hraje slaběji, než ukazuje liga, protože ta běží s 32 simulacemi na tah.

## Data

- Ligy: `/cold/w2/uc12/league-1108`, `league-s64`, `league-8x256` (`elo.json`, `games.jsonl`).
- Checkpointy: `/cold/w2/uc12/ckpt`, `ckpt-s64`, `ckpt-8x256`.
