"""UC19 prompt dialects: LLM instructions, guided-JSON schemas, deterministic post-processing, token counting.

| dialect      | checkpoints                 | body produced here                          | limit                        |
| pony         | Pony V6                     | booru tags (+ source_*), WD14 vocabulary    | ≤ 40 tags                    |
| illustrious  | Illustrious, NoobAI, Anim.  | booru tags, WD14 vocabulary                 | ≤ 40 tags                    |
| sdxl         | Juggernaut, SDXL base       | one sentence + style/camera/light, (w:1.2)  | ≤ 75 CLIP-L tokens (measured)|
| flux         | FLUX.1-dev                  | natural prose, no weights, no negative      | T5 tokens measured           |
Quality / rating prefixes are NOT produced by the LLM: build_jobs adds each checkpoint's own (models.py), the
same way the gallery's prompt-tags@v1 leaves them to the app.
"""
from __future__ import annotations

import csv, glob, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uc18"))
from common import sfw_hits  # noqa: E402
from models import MINOR_NEG, SFW_NEG  # noqa: E402

DIALECTS = ["pony", "illustrious", "sdxl", "flux"]
PROMPT_VERSION = "uc19-dialect-v1"

BASELINE_SYSTEM = ("Translate the user's Czech image-generation request into natural English. "
                   "Output only the English translation, nothing else.")

_COMMON = ("The user writes an image request in Czech. Understand the intended meaning (idioms are NOT literal: "
           "'leje jako z konve' = heavy rain), Czech places/food/folklore by what they look like, and negations "
           "('bez brýlí' = the person must not wear glasses — never mention the excluded thing in a positive prompt). "
           "Keep everything safe for work; all people are adults. Do not invent subjects that were not requested. ")

SYSTEM = {
    "pony": _COMMON + (
        "Convert it into Danbooru tags for Pony Diffusion V6. Start with the subject count (1girl, 1boy, no humans…); "
        "add one source tag when the style is anime/cartoon (source_anime, source_cartoon); then subject, attributes, "
        "clothing, pose, setting, lighting, composition, medium. Lowercase, spaces not underscores, 8–30 tags. "
        "Never output score_*, rating_* or quality tags (masterpiece, best quality) — they are added later. "
        "For a photo request add 'realistic, photorealistic'."),
    "illustrious": _COMMON + (
        "Convert it into Danbooru tags for Illustrious XL / NoobAI. Start with the subject count (1girl, 1boy, "
        "no humans…), then subject, attributes, clothing, pose, setting, lighting, composition, medium. Lowercase, "
        "spaces not underscores, 8–30 tags, canonical Danbooru spelling. Never output quality, rating or artist tags."),
    "sdxl": _COMMON + (
        "Write a prompt for SDXL / Juggernaut XL: one concise English sentence (subject first) followed by short "
        "comma-separated style, camera and lighting phrases; optionally emphasise at most two key phrases with "
        "(phrase:1.1)–(phrase:1.3). Hard limit: 60 words — the CLIP text encoder ignores everything after 75 tokens. "
        "No booru tags. Also give a short negative prompt of visual defects to avoid (and, for a negation in the "
        "request, the excluded thing)."),
    "flux": _COMMON + (
        "Write a prompt for FLUX.1: natural, descriptive English prose of 2–4 sentences (60–150 words): subject and "
        "action first, then setting, composition, lighting and medium/style. No tag lists, no (word:1.2) weights, "
        "no negative prompt — express exclusions by describing what IS there (\"clean-shaven\", \"an empty street\")."),
}

_TAG = {"type": "string", "maxLength": 48}
SCHEMA = {
    "pony": {"type": "object", "properties": {"tags": {"type": "array", "items": _TAG, "maxItems": 40}},
             "required": ["tags"], "additionalProperties": False},
    "illustrious": {"type": "object", "properties": {"tags": {"type": "array", "items": _TAG, "maxItems": 40}},
                    "required": ["tags"], "additionalProperties": False},
    "sdxl": {"type": "object", "properties": {"prompt": {"type": "string", "maxLength": 420},
                                              "negative": {"type": "string", "maxLength": 300}},
             "required": ["prompt", "negative"], "additionalProperties": False},
    "flux": {"type": "object", "properties": {"prompt": {"type": "string", "maxLength": 1400}},
             "required": ["prompt"], "additionalProperties": False},
}
MAX_TOKENS = {"pony": 400, "illustrious": 400, "sdxl": 300, "flux": 600}

QUALITY_TAGS = re.compile(r"^(score_\d(_up)?|rating_\w+|masterpiece|best quality|amazing quality|high quality|"
                          r"high score|great score|absurdres|highres|newest|very aesthetic|general|safe|sensitive|"
                          r"nsfw|explicit|questionable|worst quality|low quality)$")
ALWAYS_OK = {"source_anime", "source_cartoon", "source_pony", "realistic", "photorealistic", "no humans"}
WEIGHT_RE = re.compile(r"\(([^():]+):\s*([0-9.]+)\)")


# ---- vocabulary / tokenizers (optional; missing on a Mac) ----------------------------------------
def load_vocab(path: str | None = None) -> set | None:
    cands = [path] if path else []
    for root in (os.environ.get("HF_HOME", "/srv/hf"), os.path.expanduser("~/.cache/huggingface")):
        cands += glob.glob(os.path.join(root, "hub", "models--SmilingWolf--wd-swinv2-tagger-v3", "snapshots", "*",
                                        "selected_tags.csv"))
    for p in cands:
        if p and os.path.exists(p):
            with open(p) as f:
                return {r["name"].replace("_", " ") for r in csv.DictReader(f) if r["category"] in ("0", "4", "3")}
    return None


class Tokens:
    """CLIP-L (SDXL) and T5 (FLUX) token counts from ComfyUI's own tokenizer files — what ComfyUI really sees."""

    def __init__(self, comfy: str = os.environ.get("COMFY_DIR", "/srv/comfy/ComfyUI")):
        self.clip = self.t5 = None
        try:
            from transformers import CLIPTokenizer
            self.clip = CLIPTokenizer.from_pretrained(os.path.join(comfy, "comfy", "sd1_tokenizer"))
        except Exception:
            pass
        try:
            from tokenizers import Tokenizer
            self.t5 = Tokenizer.from_file(os.path.join(comfy, "comfy", "text_encoders", "t5_tokenizer", "tokenizer.json"))
        except Exception:
            pass

    def clip_len(self, text: str) -> int | None:
        # ComfyUI strips (x:1.2) weight syntax before tokenizing; count the words, not the brackets
        return None if self.clip is None else len(self.clip(WEIGHT_RE.sub(r"\1", text))["input_ids"]) - 2

    def t5_len(self, text: str) -> int | None:
        return None if self.t5 is None else len(self.t5.encode(text).ids)


# ---- post-processing -----------------------------------------------------------------------------
def norm_tag(t: str) -> str:
    t = t.strip().lower()
    if not re.match(r"^(score|source|rating)_", t):  # Pony meta tags keep their underscore
        t = t.replace("_", " ")
    t = WEIGHT_RE.sub(r"\1", t)
    t = t.strip(" .;")
    return re.sub(r"\s+", " ", t)


def post_tags(dialect: str, tags: list[str], vocab: set | None, vocab_mode: str = "report") -> dict:
    """vocab_mode "report" keeps out-of-vocabulary tags and lists them in `oov`; "filter" drops them (= the gallery's
    prompt-tags@v1). WD14 v3 knows only 8 106 general tags and misses common style/medium tags (pixel art, 3d,
    oil painting (medium), cute, warm lighting — 28 of the ~170 tags in our own few-shot answers), so filtering
    deletes exactly the style information the dialect is for → default report."""
    out, dropped, quality, unsafe, oov = [], [], [], [], []
    for raw in tags:
        for t in (norm_tag(x) for x in raw.split(",")):  # model sometimes packs several tags in one string
            if not t or t in out:
                continue
            if QUALITY_TAGS.match(t):
                quality.append(t)
                continue
            if sfw_hits(t):
                unsafe.append(t)
                continue
            if dialect != "pony" and t.startswith("source_"):
                dropped.append(t)
                continue
            if vocab is not None and t not in vocab and t not in ALWAYS_OK:
                if vocab_mode == "filter":
                    dropped.append(t)
                    continue
                oov.append(t)
            out.append(t)
    return {"body": ", ".join(out), "tags": out, "dropped": dropped, "oov": oov, "quality_removed": quality,
            "unsafe_removed": unsafe}


def _clean_weights(text: str) -> tuple[str, int]:
    bad = 0

    def fix(m):
        nonlocal bad
        w = float(m.group(2)) if re.fullmatch(r"[0-9]*\.?[0-9]+", m.group(2)) else 0
        if 0.5 <= w <= 1.5:
            return f"({m.group(1).strip()}:{w:g})"
        bad += 1
        return m.group(1)
    return WEIGHT_RE.sub(fix, text), bad


def post_sdxl(prompt: str, negative: str, tok: Tokens, limit: int = 75) -> dict:
    p, bad_w = _clean_weights(" ".join(prompt.split()))
    unsafe = sfw_hits(p)
    raw_len = tok.clip_len(p)
    truncated = False
    if raw_len is not None and raw_len > limit:  # cut at the last comma that fits; never mid-phrase
        parts = [x.strip() for x in p.split(",")]
        while len(parts) > 1 and tok.clip_len(", ".join(parts)) > limit:
            parts.pop()
        p, truncated = ", ".join(parts), True
    neg = ", ".join(x for x in [negative.strip(" ,"), SFW_NEG, MINOR_NEG] if x)
    return {"body": p, "negative": neg, "clip_tokens_raw": raw_len, "clip_tokens": tok.clip_len(p),
            "clip_overflow_raw": raw_len is not None and raw_len > limit, "truncated": truncated,
            "bad_weights": bad_w, "unsafe": unsafe}


def post_flux(prompt: str, tok: Tokens) -> dict:
    p = WEIGHT_RE.sub(r"\1", " ".join(prompt.split()))
    pieces = [x.strip() for x in p.split(",")]
    taglike = len(pieces) >= 8 and sum(len(x.split()) <= 3 for x in pieces) / len(pieces) > 0.7
    return {"body": p, "t5_tokens": tok.t5_len(p), "taglike": taglike, "unsafe": sfw_hits(p)}


def post_baseline(text: str, tok: Tokens) -> dict:
    p = " ".join(text.strip().strip('"').split())
    n = tok.clip_len(p)
    return {"body": p, "clip_tokens_raw": n, "clip_overflow_raw": n is not None and n > 75, "t5_tokens": tok.t5_len(p),
            "unsafe": sfw_hits(p)}


def fewshot_messages(dialect: str, fewshot: list[dict]) -> list[dict]:
    import json
    msgs = []
    for ex in fewshot:
        msgs.append({"role": "user", "content": ex["cz"]})
        if dialect in ("pony", "illustrious"):
            ans = {"tags": ex[dialect]}
        elif dialect == "sdxl":
            ans = {"prompt": ex["sdxl"], "negative": "blurry, lowres, deformed, extra fingers, watermark, text"}
        else:
            ans = {"prompt": ex["flux"]}
        msgs.append({"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)})
    return msgs
