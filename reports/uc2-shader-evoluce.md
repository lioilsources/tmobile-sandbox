# UC2 · Evoluce shaderů MirrorBooth na L40S

Qwen3-30B-A3B (vLLM) psal variace GLSL filtrů, automat je bez člověka ověřil (kontrakt, glslang, EGL render 1080×1920) a ohodnotil (PickScore, CLIP obličeje, pohyb, blikání). MAP-Elites archiv 5×5 (výpočetní cena × pohyb).

- Kaleidoskop: 1 944 dětí za 62 min, 1 683 platných (87 %), fitness 0,90 → 2,71, buněk 8 → 25, top 8 prošlo impellerc.
- Vítězové jsou ale všichni tentýž pastelový filtr s barevným posunem; skutečné kaleidoskopy v archivu jsou, hodnocení je trestá (PickScore = „hezká fotka“, trest za ztrátu obličeje).
- Akvarel a VHS: 0 dětí. Od 19:40 ubývala RAM (15 → 0,2 GB ve 20:00), vLLM přestalo odpovídat. Viník neprokázán; od té doby běží memguard.
- Výkon: medián 60 s a 1 550 tokenů na dítě, 48 souběžně; GPU 65 % (35 % stojí), 219 W, 214 Wh; ≈ 0,03 Kč za funkční shader.

Další kroky: hodnotit podobnost s tématem místo PickScore, galerie po buňkách, doběhnout akvarel a VHS s měřením RSS, hodnocení v dávkách.
