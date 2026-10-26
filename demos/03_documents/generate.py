#!/usr/bin/env python3
"""Syntetické české faktury a smlouvy (PDF) + ground truth JSON.

  generate.py --out /srv/work/demo3/docs --invoices 35 --contracts 15 --seed 1

Variabilita: 4 rozvržení faktur, 2 typy smluv (1–2 strany), různé fonty, 1/3 dokumentů jako „sken“
(natočení, šum, rozmazání, JPEG, šedá). Hodnoty v ground truth jsou normalizované (ISO data, čísla).
"""
import argparse, io, json, random, pathlib, datetime as dt
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from faker import Faker

W, H = 1240, 1754  # A4 @ 150 dpi
FONTS = {
    "dejavu": ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    "liberation": ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    "serif": ("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"),
    "noto": ("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
}
fk = Faker("cs_CZ")

def ico():  # platné IČO (kontrolní součet mod 11)
    d = [random.randint(0, 9) for _ in range(7)]
    r = (11 - sum(x * w for x, w in zip(d, range(8, 1, -1))) % 11) % 10
    return "".join(map(str, d)) + str(r)

def company():
    i = ico()
    base = random.choice([fk.last_name(), f"{fk.last_name()} & {fk.last_name()}", f"{fk.last_name()} {random.choice(['Servis', 'Stav', 'Tech', 'Group', 'Trade', 'Systems', 'Logistik'])}",
                          f"{random.choice(['Alfa', 'Delta', 'Nova', 'Prima', 'Omega', 'Terra', 'Viva'])} {random.choice(['Facility', 'Data', 'Energo', 'Montáže', 'Consulting'])}"])
    return {"nazev": base + random.choice([" s.r.o.", " a.s.", " spol. s r.o.", " s.r.o.", " k.s."]),
            "ico": i, "dic": "CZ" + i, "adresa": f"{fk.street_address()}, {fk.postcode()} {fk.city()}"}

def kc(x, sep=" "):  # 25289.0 → "25 289,00"
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", sep)

def cz_date(d, style):
    return {0: f"{d.day}. {d.month}. {d.year}", 1: d.strftime("%d.%m.%Y"), 2: f"{d.day}.{d.month}.{d.year}"}[style]

ITEMS = ["Servis klimatizace", "Výměna filtrů", "Pronájem techniky", "Konzultace IT", "Licence software (12 měs.)",
         "Instalace síťových prvků", "Úklid kancelářských prostor", "Revize elektroinstalace", "Doprava materiálu",
         "Tisk propagačních materiálů", "Údržba zeleně", "Dodávka kancelářského papíru", "Kabeláž Cat6 (m)",
         "Správa serverů", "Školení zaměstnanců", "Ostraha objektu", "Oprava výtahu", "Monitorovací služby"]

class Page:
    def __init__(self, font):
        self.img = Image.new("RGB", (W, H), "white"); self.d = ImageDraw.Draw(self.img); self.fp = FONTS[font]
    def f(self, size, bold=False): return ImageFont.truetype(self.fp[bold], size)
    def t(self, xy, s, size=24, bold=False, fill="black", anchor="la"): self.d.text(xy, s, font=self.f(size, bold), fill=fill, anchor=anchor)
    def line(self, y, x0=80, x1=W - 80, w=2): self.d.line((x0, y, x1, y), fill="black", width=w)
    def box(self, xy, fill=None, w=2): self.d.rectangle(xy, outline="black", width=w, fill=fill)

def invoice(n, layout):
    font = random.choice(list(FONTS)); ds = random.randint(0, 2)
    sup, cus = company(), company()
    if random.random() < 0.3: cus = {"nazev": "T-Mobile Czech Republic a.s.", "ico": "64949681", "dic": "CZ64949681",
                                     "adresa": "Tomíčkova 2144/1, 148 00 Praha 4"}
    issued = fk.date_between(dt.date(2025, 1, 1), dt.date(2026, 10, 20)); due = issued + dt.timedelta(days=random.choice([14, 14, 30, 21, 60]))
    duzp = issued - dt.timedelta(days=random.randint(0, 5))
    num = f"{issued.year}{random.randint(1, 9999):04d}" if layout % 2 else f"FV-{issued.year}-{random.randint(1, 999):03d}"
    vs = "".join(c for c in num if c.isdigit())[-10:]
    rate = random.choice([21, 21, 21, 12])
    items = [(random.choice(ITEMS), q := random.randint(1, 12), round(random.uniform(150, 25000), 0)) for _ in range(random.randint(1, 6))]
    base = round(sum(q * c for _, q, c in items), 2); vat = round(base * rate / 100, 2); total = round(base + vat, 2)
    acct = f"{random.randint(100000, 9999999999)}/{random.choice(['0800', '0100', '0300', '2010', '0600', '5500'])}"
    gt = {"typ": "faktura", "cislo_faktury": num, "variabilni_symbol": vs, "datum_vystaveni": issued.isoformat(),
          "datum_splatnosti": due.isoformat(), "duzp": duzp.isoformat(), "dodavatel_nazev": sup["nazev"],
          "dodavatel_ico": sup["ico"], "dodavatel_dic": sup["dic"], "odberatel_nazev": cus["nazev"], "odberatel_ico": cus["ico"],
          "zaklad_dph": base, "sazba_dph": rate, "dph": vat, "celkem": total, "bankovni_ucet": acct, "pocet_polozek": len(items)}

    pg = Page(font); t = pg.t
    title = random.choice(["FAKTURA – daňový doklad", "Faktura", "FAKTURA", "Daňový doklad – faktura"])
    if layout == 0:   # klasika: dva sloupce, tabulka
        t((80, 70), title, 44, True); t((W - 80, 80), f"č. {num}", 32, True, anchor="ra")
        pg.line(140, w=3)
        for x, lab, c in ((80, "Dodavatel", sup), (650, "Odběratel", cus)):
            t((x, 165), lab, 22, True); t((x, 200), c["nazev"], 26, True); t((x, 238), c["adresa"][:44], 22)
            t((x, 272), f"IČO: {c['ico']}   DIČ: {c['dic']}", 22)
        y = 340
        for k, v in (("Datum vystavení", cz_date(issued, ds)), ("Datum splatnosti", cz_date(due, ds)),
                     ("Datum zdanit. plnění", cz_date(duzp, ds)), ("Variabilní symbol", vs), ("Bankovní účet", acct)):
            t((80, y), f"{k}:", 22); t((420, y), v, 22, True); y += 36
    elif layout == 1:  # banner nahoře, údaje v boxech
        pg.box((0, 0, W, 150), fill=(225, 0, 116) if random.random() < .5 else (40, 70, 140), w=0)
        t((80, 50), title.upper(), 46, True, fill="white"); t((W - 80, 60), num, 34, True, fill="white", anchor="ra")
        pg.box((80, 190, 600, 400)); pg.box((640, 190, W - 80, 400))
        for x, lab, c in ((100, "DODAVATEL", sup), (660, "ODBĚRATEL", cus)):
            t((x, 205), lab, 20, True, fill=(90, 90, 90)); t((x, 240), c["nazev"][:34], 25, True)
            t((x, 280), c["adresa"][:40], 21); t((x, 320), f"IČ {c['ico']}", 21); t((x, 355), f"DIČ {c['dic']}", 21)
        y = 440
        for k, v in (("Vystaveno", cz_date(issued, ds)), ("Splatnost", cz_date(due, ds)), ("DUZP", cz_date(duzp, ds)),
                     ("VS", vs), ("Účet", acct)):
            t((80 + (y - 440) // 36 * 0, y), f"{k}: {v}", 22); y += 36
    elif layout == 2:  # „účetní software“: hustý text, údaje vpravo
        t((80, 80), sup["nazev"], 34, True); t((80, 125), sup["adresa"], 22); t((80, 158), f"IČ: {sup['ico']}, DIČ: {sup['dic']}", 22)
        t((80, 190), f"Bankovní spojení: {acct}", 22)
        t((W - 80, 80), title, 30, True, anchor="ra"); t((W - 80, 125), f"Číslo: {num}", 24, anchor="ra")
        pg.line(240)
        t((80, 265), "Odběratel:", 22, True); t((80, 300), cus["nazev"], 26, True); t((80, 336), cus["adresa"], 22)
        t((80, 370), f"IČ: {cus['ico']}  DIČ: {cus['dic']}", 22)
        y = 265
        for k, v in (("Datum vystavení", cz_date(issued, ds)), ("Datum splatnosti", cz_date(due, ds)),
                     ("Datum uskut. zdan. plnění", cz_date(duzp, ds)), ("Var. symbol", vs), ("Forma úhrady", "převodem")):
            t((700, y), f"{k}:", 21); t((W - 80, y), v, 21, True, anchor="ra"); y += 34
        y = 440
    else:  # minimalistická, bez rámečků, odběratel nahoře
        t((80, 80), "Odběratel", 20, fill=(110, 110, 110)); t((80, 108), cus["nazev"], 28, True)
        t((80, 146), f"{cus['adresa']} · IČO {cus['ico']}", 21)
        t((80, 230), title, 52, True); t((80, 300), f"Číslo dokladu {num}", 26)
        t((80, 360), f"Vystaveno {cz_date(issued, ds)} · Splatnost {cz_date(due, ds)} · DUZP {cz_date(duzp, ds)}", 22)
        t((80, 396), f"Variabilní symbol {vs} · Účet {acct}", 22)
        y = 460
    # položky
    y = max(y, 460) + 30
    t((80, y), "Položka", 22, True); t((760, y), "Množství", 22, True, anchor="ra"); t((950, y), "Cena/ks", 22, True, anchor="ra")
    t((W - 80, y), "Celkem bez DPH", 22, True, anchor="ra"); y += 40; pg.line(y - 6, w=1)
    for name, q, c in items:
        t((80, y), name, 22); t((760, y), f"{q} ks", 22, anchor="ra"); t((950, y), kc(c), 22, anchor="ra")
        t((W - 80, y), kc(q * c), 22, anchor="ra"); y += 38
    pg.line(y + 4, w=1); y += 40
    sep = random.choice([" ", ".", " "])
    for k, v, b in ((f"Základ DPH {rate} %", base, False), (f"DPH {rate} %", vat, False), ("Celkem k úhradě", total, True)):
        t((620, y), k, 26 if b else 22, b); t((W - 80, y), f"{kc(v, sep)} Kč", 26 if b else 22, b, anchor="ra"); y += 44 if b else 36
    if layout == 3:  # minimalistická: dodavatel až v patičce (těžší přiřazení rolí)
        pg.line(H - 300, w=1); t((80, H - 280), "Vystavil / dodavatel", 20, fill=(110, 110, 110))
        t((80, H - 250), sup["nazev"], 24, True); t((80, H - 216), f"{sup['adresa']} · IČO {sup['ico']} · DIČ {sup['dic']}", 20)
    elif random.random() < .6: t((80, H - 160), "Razítko a podpis:", 20, fill=(100, 100, 100)); pg.d.ellipse((330, H - 220, 520, H - 90), outline=(40, 60, 160), width=4)
    t((80, H - 70), f"Společnost zapsána v obchodním rejstříku vedeném Krajským soudem, oddíl C, vložka {random.randint(1000, 99999)}.", 16, fill=(110, 110, 110))
    return [pg.img], gt

CONTRACTS = {"smlouva_o_dilo": ("SMLOUVA O DÍLO", "Zhotovitel", "Objednatel", "provést dílo spočívající v"),
             "servisni_smlouva": ("SERVISNÍ SMLOUVA", "Poskytovatel", "Objednatel", "poskytovat servisní služby v rozsahu")}
SUBJ = ["rekonstrukce serverovny v budově objednatele", "dodávce a instalaci kamerového systému", "pravidelné údržbě klimatizačních jednotek",
        "vývoji mobilní aplikace pro zákaznickou podporu", "migraci e-mailových schránek do cloudu", "správě a údržbě vozového parku"]

def contract(n):
    font = random.choice(list(FONTS)); ds = random.randint(0, 2)
    kind = random.choice(list(CONTRACTS)); title, r1, r2, verb = CONTRACTS[kind]
    a, b = company(), company()
    signed = fk.date_between(dt.date(2025, 1, 1), dt.date(2026, 10, 1)); eff = signed + dt.timedelta(days=random.choice([0, 1, 14, 30]))
    fixed = random.random() < .5; end = eff + dt.timedelta(days=365 * random.choice([1, 2, 3])) if fixed else None
    price = float(random.randrange(20000, 3000000, 500)); notice = random.choice([1, 2, 3, 6])
    penalty = random.choice([0.05, 0.1, 0.5]); num = f"{random.choice(['SOD', 'SRV', 'S'])}-{signed.year}/{random.randint(1, 999):03d}"
    gt = {"typ": "smlouva", "typ_smlouvy": kind, "cislo_smlouvy": num, "strana1_nazev": a["nazev"], "strana1_ico": a["ico"],
          "strana2_nazev": b["nazev"], "strana2_ico": b["ico"], "datum_uzavreni": signed.isoformat(), "ucinnost_od": eff.isoformat(),
          "doba_urcita": fixed, "konec_platnosti": end.isoformat() if end else None, "cena_bez_dph": price,
          "vypovedni_lhuta_mesice": notice, "smluvni_pokuta_procent_denne": penalty}
    pages, pg = [], Page(font); t = pg.t
    t((W // 2, 90), title, 42, True, anchor="ma"); t((W // 2, 150), f"č. {num}", 24, anchor="ma")
    t((W // 2, 190), "uzavřená podle § 2586 a násl. zákona č. 89/2012 Sb., občanský zákoník" if kind == "smlouva_o_dilo"
      else "uzavřená podle § 1746 odst. 2 zákona č. 89/2012 Sb., občanský zákoník", 19, anchor="ma")
    y = 260
    for c, role in ((a, r1), (b, r2)):
        t((80, y), c["nazev"], 25, True); t((80, y + 36), f"se sídlem {c['adresa']}", 21)
        t((80, y + 68), f"IČO: {c['ico']}, DIČ: {c['dic']}", 21); t((80, y + 100), f"zastoupená {fk.name()}, jednatelem", 21)
        t((80, y + 132), f"(dále jen „{role}“)", 21, True); y += 190
        if role == r1: t((80, y - 20), "a", 22); y += 30
    body = [("I. Předmět smlouvy", f"{r1} se zavazuje {verb} {random.choice(SUBJ)}, a to v rozsahu a kvalitě dle přílohy č. 1."),
            ("II. Cena", f"Cena byla sjednána dohodou ve výši {kc(price)} Kč bez DPH. K ceně bude připočtena DPH v zákonné sazbě."),
            ("III. Doba trvání", (f"Smlouva se uzavírá na dobu určitou do {cz_date(end, ds)}." if fixed else "Smlouva se uzavírá na dobu neurčitou.")
             + f" Smlouva nabývá účinnosti dne {cz_date(eff, ds)}."),
            ("IV. Výpověď", f"Každá ze smluvních stran je oprávněna smlouvu vypovědět i bez udání důvodu. Výpovědní lhůta činí {notice} "
             + {1: "měsíc", 2: "měsíce", 3: "měsíce", 6: "měsíců"}[notice] + " a počíná běžet prvním dnem měsíce následujícího po doručení výpovědi."),
            ("V. Sankce", f"V případě prodlení je {r2} oprávněn požadovat smluvní pokutu ve výši {str(penalty).replace('.', ',')} % z ceny za každý den prodlení."),
            ("VI. Závěrečná ustanovení", "Smlouva je vyhotovena ve dvou stejnopisech, z nichž každá strana obdrží jeden. Změny smlouvy lze činit pouze písemnými dodatky.")]
    def wrap(s, n=88):
        out, cur = [], ""
        for w in s.split():
            if len(cur) + len(w) + 1 > n: out.append(cur); cur = w
            else: cur = (cur + " " + w).strip()
        return out + [cur]
    for head, txt in body:
        if y > H - 300: pages.append(pg.img); pg = Page(font); t = pg.t; y = 100
        t((80, y), head, 25, True); y += 42
        for ln in wrap(txt): t((80, y), ln, 22); y += 32
        y += 24
    if y > H - 260: pages.append(pg.img); pg = Page(font); t = pg.t; y = 100
    t((80, y + 20), f"V Praze dne {cz_date(signed, ds)}", 22); y += 120
    t((80, y), f"za {r1.lower()}a: ____________________", 22); t((680, y), f"za {r2.lower()}a: ____________________", 22)
    pages.append(pg.img)
    return pages, gt

def scan(img):
    img = img.convert("L").rotate(random.uniform(-2.5, 2.5), expand=False, fillcolor=255, resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))
    a = np.asarray(img, dtype=np.float32) + np.random.normal(0, random.uniform(4, 12), (img.height, img.width))
    img = Image.fromarray(np.clip(a * random.uniform(0.85, 1.0) + random.uniform(0, 20), 0, 255).astype(np.uint8))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=random.randint(35, 65)); return Image.open(buf).convert("RGB")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/srv/work/demo3/docs")
    ap.add_argument("--invoices", type=int, default=35); ap.add_argument("--contracts", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--scan-ratio", type=float, default=0.33)
    a = ap.parse_args(); random.seed(a.seed); np.random.seed(a.seed); Faker.seed(a.seed)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = [("inv", lambda i: invoice(i, i % 4)) for _ in range(a.invoices)] + [("ctr", contract) for _ in range(a.contracts)]
    for i, (pre, fn) in enumerate(jobs):
        pages, gt = fn(i)
        scanned = random.random() < a.scan_ratio
        if scanned: pages = [scan(p) for p in pages]
        name = f"{pre}_{i:03d}"; gt.update({"_soubor": f"{name}.pdf", "_stran": len(pages), "_sken": scanned})
        pages[0].save(out / f"{name}.pdf", save_all=True, append_images=pages[1:], resolution=150)
        (out / f"{name}.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1))
    print(f"{len(jobs)} dokumentů → {out}")

if __name__ == "__main__": main()
