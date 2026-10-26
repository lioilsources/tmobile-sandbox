# UC8 – Obrázek → 3D: UGC předměty přes FLUX a Hunyuan3D 2.1 (L40S)

*26. 10. 2026 · T‑Mobile GPUaaS sandbox, 1× NVIDIA L40S 48 GB · ComfyUI, nativní uzly · FLUX.1‑schnell fp8 + Hunyuan3D 2.1*

## Shrnutí
Pipeline vyrobila **200 herních předmětů** (10 stylů × 20 typů, např. „zlatá zdobená koruna“, „plastelínový drak“).
FLUX‑schnell nejdřív nakreslí předmět na bílém pozadí (**3 s**), pak Hunyuan3D 2.1 z obrázku postaví 3D mesh (**31 s**).
Vznikl GLB soubor u **všech 200 předmětů**. To odpovídá zhruba **105 předmětům za hodinu** a ≈ 0,5 Kč za předmět (náklad VM).
Surový mesh je ale nepoužitelný. Kromě objektu obsahuje ~200 tisíc drobných útržků šumu, medián je 533 tisíc trojúhelníků
a 8 MB. Po vyčištění (ponechat hlavní část) a decimaci má **39 ze 40 testovaných modelů 4 000 trojúhelníků**, což je limit
UGC doplňku na Robloxu. Tvar zůstane poznatelný. Chybí textura a meshe nejsou vodotěsné.

## Postup (`uc8/gen3d.py`)
1. **Obrázek:** FLUX.1‑schnell fp8, 1024², 4 kroky. Prompt „3D game asset render of a {styl} {předmět}, single object centered,
   plain white background, soft studio lighting, three‑quarter view“.
2. **3D:** Hunyuan3D 2.1, nativní uzly ComfyUI. CLIP vision (DINOv2) → 30 kroků, cfg 5 → VAE decode (octree 256) → surface net
   (práh 0,6) → GLB.
3. **Kontrola:** trimesh (trojúhelníky, vodotěsnost). Na vzorku 40 modelů čištění a decimace (quadric) na 4 000 trojúhelníků.

## Výkon

| Metrika | Hodnota |
|---|---|
| Předmětů | 200 (200 GLB, 0 chyb) |
| Obrázek (FLUX‑schnell) | 3,0 s |
| 3D model (Hunyuan3D) | 31 s průměr, medián 30 s (21–53 s) |
| Propustnost | ≈ 105 předmětů/h |
| GPU | průměr 44 %, polovinu času nečinná, max 59 °C, VRAM max 19,7 GB |
| Energie | 1,85 Wh na předmět |
| Náklad VM | ≈ 0,5 Kč na předmět (předpoklad 55 Kč/h) |

**GPU polovinu času čeká:** převod voxelů na mesh a uložení GLB běží na CPU a úlohy šly přísně za sebou. Se dvěma úlohami
ve frontě by GPU počítala další model, zatímco CPU dokončuje předchozí. Odhad je ~1,5× vyšší propustnost.

Běh přerušil výpadek VM po 74 modelech. Skript pak navázal a hotové modely přeskočil.

## Kvalita meshe

| | Surový výstup | Po vyčištění a decimaci (vzorek 40) |
|---|---|---|
| Trojúhelníků | medián 533 tis. (198 tis.–1,4 mil.) | 4 000 u 39 ze 40 |
| Oddělených částí | medián 206 tis. | 1–2 |
| Podíl hlavního objektu | 17 % trojúhelníků | 100 % |
| Vodotěsné | 1 z 200 | 0 ze 40 |
| Velikost GLB | medián 8,4 MB, celkem 1,8 GB | ≈ 100 kB |

- **Šum:** surface net s prahem 0,6 vyrobí kolem objektu oblak drobných útržků v celé krychli. Ponechat jen
  komponenty nad 5 % hlavní části stačí. V trimeshi to ale trvá ~41 s na model (jedno vlákno, CPU), déle než samotná GPU část.
- **Tvar:** štíty, koruny, draci a sekery jsou po decimaci dobře poznatelné. U některých zůstane plochý „stín“ pod objektem
  (např. sekera).
- **Textura:** Hunyuan3D 2.1 v nativních uzlech ComfyUI dělá jen tvar. Barvy z obrázku se na mesh nepřenesou.

## Co z toho plyne
- Z obrázku do hrubého 3D modelu se na jedné L40S dostaneme za půl minuty. Pro prototypy, náhledy a „blockout“ herních předmětů
  je to použitelné hned.
- Pro publikaci (Roblox UGC) chybí textura, vodotěsnost a UV mapa. Čištění meshe patří do pipeline jako standardní krok.

## Další kroky
- Vyšší práh surface net nebo jiný algoritmus a ověřit, jestli šum zmizí už při generování.
- Čištění přes pymeshlab nebo paralelně na víc jádrech, aby nebrzdilo GPU.
- Texturovací část Hunyuan3D 2.1 (paint model) mimo nativní uzly.
- Dvě úlohy ve frontě kvůli souběhu GPU a CPU.

*Reprodukce:* `uc8/run.sh` (profil D) → `uc8/gen3d.py --n 200 [--resume run.prev.log]`. Data: `/srv/work/uc8/`
(`img/`, `glb/`, `models.jsonl`, `report.json`, `clean_sample.json`, `preview/`). Vlastní generovaný obsah, žádná data T‑Mobile.
