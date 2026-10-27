# UC7 · RealEyes LoRA (SDXL) na L40S

LoRA naučila SDXL kreslit oči skutečných lidí: FID 153,4 → 75,4, KID 0,159 → 0,061, CLIP podobnost k reálným výřezům 0,723 → 0,790 (300 vzorků proti 500 odloženým reálným výřezům). Selhává u starých lidí (v datech 7 z 1 500), výstup je měkčí než originály (výřezy zvětšené z ~580 px). Trénink vytížil GPU průměrně jen na 53 % (VRAM max 11 GB).

## Dataset
FFHQ (CC BY-NC-SA, jen interní měření), 9 992 tváří → MediaPipe ořez očí s obočím 1024×512 → CLIP filtry.
Vyřazeno: natočení 1 359, mrknutí 752, děti 2 145, brýle 1 615, kostýmy 568, zakryté 206, neostré 1 104 → 2 243 kandidátů → 1 500 trénink + 500 reference.
První verze výběru (greedy k-center) brala extrémy (malování, kostýmy) → náhodný vzorek.

## Výkon
| | |
|---|---|
| Nastavení | kohya sd-scripts v0.8.7 (FineTuneGallery), dim 32/alpha 16, batch 2, AdamW8bit, bf16, grad. checkpointing, 10 epoch |
| Kroků | 7 500, 1,14 s/krok |
| Trénink | 2 h 29 min, GPU 53 %, VRAM 11,0 GB, 214 W, 0,51 kWh, ≈ 140 Kč |
| Generování | 723 obr./h (1408×704, 28 kroků), GPU 82 %, 0,39 Wh/obr. |

Mezi epochou 6 a 10 se výstup skoro nemění. Srovnání se Sparkem není (na Sparku trénink neproběhl).

## Další kroky
Batch 8 bez grad. checkpointingu (odhad 2×), 6 epoch, ostřejší zdrojové výřezy, vyvážit věk a barvu očí, FLUX LoRA.
