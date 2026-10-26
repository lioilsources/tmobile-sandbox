#!/usr/bin/env python3
"""Syntetická česká faktura (PNG) pro smoke test profilu C / demo 3."""
import sys
from PIL import Image, ImageDraw, ImageFont
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
img = Image.new("RGB", (1240, 1754), "white"); d = ImageDraw.Draw(img)
big, f = ImageFont.truetype(F, 48), ImageFont.truetype(F, 28)
d.text((80, 80), "FAKTURA – daňový doklad č. 2026-10-0457", font=big, fill="black")
rows = ["Dodavatel: Novák Servis s.r.o., Lipová 12, 602 00 Brno", "IČO: 27654321   DIČ: CZ27654321",
        "Odběratel: T-Mobile Czech Republic a.s., Tomíčkova 2144/1, 148 00 Praha 4", "IČO: 64949681",
        "Datum vystavení: 15. 10. 2026     Datum splatnosti: 29. 10. 2026", "Variabilní symbol: 20261004570",
        "", "Položka                                   Množství    Cena bez DPH",
        "Servis klimatizace serverovny                1 ks       18 500,00 Kč",
        "Výměna filtrů                                4 ks        2 400,00 Kč", "",
        "Základ DPH 21 %:   20 900,00 Kč", "DPH 21 %:           4 389,00 Kč", "Celkem k úhradě:   25 289,00 Kč",
        "Bankovní účet: 123456789/0800"]
for i, r in enumerate(rows): d.text((80, 220 + i * 52), r, font=f, fill="black")
img.save(sys.argv[1])
