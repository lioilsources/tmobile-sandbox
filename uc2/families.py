"""UC2 kolo 2: rodiny shaderů s kartou techniky (Prompts/08 plánu MirrorBooth, kroky 1-2).

Karta rodiny říká, co dítě MUSÍ použít (`must`: každá skupina vyžaduje aspoň jeden zásah regexu, `min_tex`: minimum
čtení textury) a co NESMÍ (`forbid`). Kontrola je statická, nad kódem bez komentářů. Seed se zařadí do rodiny, jejíž
kartu splňuje; nejdřív se zkouší rodina z názvu (HINTS).

  python3 families.py <adresář-s-.frag> ...     # vypíše zařazení seedů do rodin a počty
"""
import pathlib, re, sys

# Deformace UV před čtením textury: uv se přepíše výrazem se sin/cos/atan/pow/length.
UV_WARP = r"\buv\s*[-+*/]?=\s*[^;]*(sin|cos|atan|pow|length)\s*\("

FAMILIES = {
    "mozaika": dict(
        desc="pixel mosaic filter: the selfie is quantized into square, hexagonal or voronoi cells with one flat color each",
        must=[[r"floor\s*\(", r"voronoi", r"hex", r"\bround\s*\(", r"\bmod\s*\("]], forbid=[r"\batan\s*\("], face=0.80, max_keep=0.98),  # raindrop overlays (c0086) kept the face at 0.986–0.989, mosaics ≤ 0.972
    "polotón": dict(
        desc="halftone and dither print filter: the selfie is rendered as a raster of dots or lines whose size follows brightness",
        must=[[r"\b(step|smoothstep)\s*\("], [r"\b(fract|mod)\s*\("]], forbid=[], face=0.88),
    "skica": dict(
        desc="sketch and engraving filter: lines or edges driven by brightness, hatching or contour lines on paper",
        must=[[r"\bdot\s*\(", r"\bstep\s*\(", r"\bsmoothstep\s*\(", r"\bfract\s*\(", r"\bsin\s*\(", r"\bmix\s*\("]], forbid=[r"neon"], face=0.88),
    "optika": dict(
        desc="lens distortion filter: fisheye, bulge, twirl or glass refraction centered on the face",
        must=[[r"uFaceCenter", r"\blength\s*\(", r"max\s*\(\s*abs"]], forbid=[r"segment", r"sector", r"kaleid"], face=0.70,
        max_hist=0.9),  # optics moves pixels, keeps colours: duotone children had colour-histogram distance 0.97–1.0
    "dlaždice": dict(
        desc="tiled mirror filter: the selfie repeated in a grid with cartesian mirroring and shifted rows",
        must=[[r"\babs\s*\(", r"\bfract\s*\(", r"\bmod\s*\("]], forbid=[r"\batan\s*\("], face=0.80),
    "glitch": dict(
        desc="digital glitch filter: shifted blocks and rows, split RGB channels, noise that jumps in time",
        must=[[r"\bfloor\s*\("], [r"hash|rand|noise|fract\s*\(\s*sin"]], forbid=[r"wave|ripple|swirl"], face=0.80),
    "vhs": dict(
        desc="VHS tape and CRT screen filter: scanlines, shadow mask, chroma bleed, tracking errors and screen curvature",
        must=[[r"uv\.y|FlutterFragCoord\(\)\.y"]], forbid=[r"kaleid", r"\batan\s*\("], face=0.88),
    "malba": dict(
        desc="painting filter: Kuwahara or oil paint brush strokes, watercolor bleeding on paper",
        must=[], min_tex=4, forbid=[r"kaleid", r"\batan\s*\("], face=0.80),
    "barva": dict(
        desc="color grading filter: duotone, posterize, thermal camera, negative or gradient mapping with no geometry change",
        must=[[r"\bmix\s*\(", r"\bsmoothstep\s*\(", r"\bfloor\s*\("]], forbid=[UV_WARP], face=0.90),
    "světlo": dict(
        desc="light filter: bloom, glow from bright areas, light rays, bokeh circles and vignette",
        must=[[r"\bexp\s*\(", r"\bpow\s*\(", r"\bsmoothstep\s*\(", r"\bmax\s*\(", r"\bmix\s*\("]], min_tex=1, forbid=[UV_WARP], face=0.90),
    "obličej": dict(
        desc="face-anchored funhouse filter: only the face region is deformed, anchored on the face center and scale",
        must=[[r"uFaceCenter"], [r"uFaceScale"]], forbid=[], face=0.70),
    "vrstvy": dict(
        desc="weather and particle overlay: rain, snow, fog, dust or sparks drawn on top of the unchanged selfie",
        must=[[r"hash|rand|fract\s*\(\s*sin"], [r"uTime"]], forbid=[UV_WARP], face=0.90),
}

# název souboru → rodina (první zkoušená)
HINTS = [
    (r"mosaic|hex|voronoi|pixel|led|blocks", "mozaika"), (r"halftone|dither|comic|pop_art|bayer|ascii", "polotón"),
    (r"sketch|pencil|edge|hatch|emboss|engrave", "skica"),
    (r"fisheye|bulge|twirl|lens|big_nose|glass|tunnel|blackhole|melt|tilt", "optika"),
    (r"tile|mirror|grid|blinds|quad", "dlaždice"), (r"glitch", "glitch"), (r"vhs|crt|scan", "vhs"),
    (r"paint|oil|kuwahara|water|aquarelle", "malba"),
    (r"duotone|poster|thermal|negative|gradient|frozen|vampire|zombie|gameboy|cutout|night", "barva"),
    (r"bloom|glow|ray|bokeh|vignette|ghost|neon", "světlo"),
    (r"face|alien|chibi|doll|demon|lips|long_neck|slim|big_ears|ears|neck", "obličej"),
    (r"rain|snow|fog|dust|spark|particle|matrix", "vrstvy"),
]


# render gates for every family (trial 5, 11. 11.): near-unchanged selfies had CLIP keep 0.999–1.0
GATE_DEFAULTS = dict(max_keep=0.998, min_strength=0.03, max_hist=None)


def gate(family, key):
    return FAMILIES[family].get(key, GATE_DEFAULTS[key])


def strip_comments(code):
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", code, flags=re.S)


def check_card(code, family):
    """Prázdný seznam = kód je v kartě. Jinak důvody (české, krátké)."""
    c, s = FAMILIES[family], strip_comments(code)
    errs = ["zakázáno (globálně): atan (polární warp)"] if family != "optika" and re.search(r"\batan\s*\(", s) else []
    errs += [f"chybí: {'|'.join(g)}" for g in c["must"] if not any(re.search(p, s) for p in g)]
    reads = len(re.findall(r"\btexture\s*\(", s))
    if reads and re.search(r"\bfor\s*\(", s): reads += 9       # čtení ve smyčce se počítá jako 9 vzorků okolí
    if reads < c.get("min_tex", 0):
        errs.append(f"méně než {c['min_tex']} čtení textury (smyčka = 9)")
    errs += [f"zakázáno: {p}" for p in c["forbid"] if re.search(p, s)]
    return errs


def card_text(family):
    """Karta rodiny pro prompt."""
    c = FAMILIES[family]
    t = [f"Family: {family}. Concept: {c['desc']}."]
    if c["must"]:
        t.append("The shader MUST use (at least one of each line): " + "; ".join(" OR ".join(g) for g in c["must"]))
    if c.get("min_tex"):
        t.append(f"It MUST read the camera texture at least {c['min_tex']} times per pixel.")
    if c["forbid"]:
        t.append("The shader MUST NOT contain: " + "; ".join(c["forbid"]))
    return "\n".join(t)


def guess_families(name, code):
    """Rodiny seznamu pro seed: nejdřív podle názvu, pak ostatní; jen ty, jejichž kartu kód splní."""
    first = [f for pat, f in HINTS if re.search(pat, name.lower())]
    order = first + [f for f in FAMILIES if f not in first]
    return [f for f in order if not check_card(code, f)]


# Ruční zařazení seedů (MirrorBooth, 2026-11-10); bez položky = seed se nepoužije (cyborg ani děti z kola 1 do žádné rodiny).
SEED_MAP = {
    "mozaika": "labMosaic labHex labGlass labLed", "polotón": "labHalftone labDither labGameboy labAscii",
    "skica": "labSketch labEmboss labEngrave", "optika": "labFisheye labTwirl labBlackHole labTunnel big_nose alien chibi",
    "dlaždice": "labTiles labQuad labBlinds", "glitch": "labBlocks glitch", "vhs": "labVhs labNight crt",
    "malba": "labKuwahara labAquarelle oil", "barva": "labDuotone labCutout thermal pop_art", "světlo": "labRays labTilt glow neon",
    "obličej": "slim doll lips big_ears long_neck", "vrstvy": "labRain labSnow labMatrix",
}


def assign_seeds(paths, mapping=SEED_MAP):
    """→ {rodina: [Path]}. Seed jde do rodiny z mapování (název bez prefixu filter_); bez položky → 'nezařazeno'.
    Splnění karty se ověří až při hodnocení seedu (seed mimo kartu se vyřadí a vypíše)."""
    out = {f: [] for f in FAMILIES}; out["nezařazeno"] = []
    where = {n: f for f, names in mapping.items() for n in names.split()}
    for p in sorted(paths):
        out[where.get(p.stem.replace("filter_", ""), "nezařazeno")].append(p)
    return out


if __name__ == "__main__":
    ps = [q for d in sys.argv[1:] for q in pathlib.Path(d).glob("*.frag")]
    res = assign_seeds(ps)
    for f, l in res.items():
        print(f"{f:12s} {len(l):3d}  {', '.join(p.stem.replace('filter_', '') for p in l)}")
    print("rodin se ≥3 seedy:", sum(len(l) >= 3 for f, l in res.items() if f != "nezařazeno"), "z", len(FAMILIES))
