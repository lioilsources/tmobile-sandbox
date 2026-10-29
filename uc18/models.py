"""UC18/UC19 checkpoint registry: file on the VM, prompt dialect, sampler settings, SFW negatives.

Settings follow each model card's recommendation (sampler / steps / cfg / clip skip / quality tags),
held fixed for the whole matrix so differences are the checkpoint's, not the settings'.
`gallery_id` = FineTuneGallery kModels id (ingest stores it verbatim in nodes.model_id).
`est_s` = s/image at 1024² on the L40S: SDXL measured 5.03 s @30 steps (uc5, 26. 10.), scaled by steps;
FLUX.1-dev fp8 20 steps ESTIMATED from FLUX-schnell 2.52 s @4 steps (≈0.45 s/step + 0.7 s) — verify in
the first block (gen_matrix health check prints the measured value).

LICENCE: flux_dev (FLUX.1-dev) is non-commercial → internal measurement only; images and any derived
data must not go into a repo or a product. gen_matrix marks its manifest rows `licence_restricted`.
"""

# Terms added to EVERY negative prompt (SFW gate; FLUX ignores negatives → SFW there rests on the prompt
# set + the post-hoc WD14 rating check in score_images.py).
SFW_NEG = "nsfw, nude, naked, nipples, explicit, sexual, suggestive, cleavage, underwear, lingerie, gore, blood"
MINOR_NEG = "child, loli, shota, kid"
QUALITY_NEG_SDXL = "lowres, blurry, worst quality, low quality, jpeg artifacts, watermark, signature, text, deformed, bad anatomy, bad hands"

CKPTS = {
    "pony": dict(
        file="ponyDiffusionV6XL_v6StartWithThisOne.safetensors", family="sdxl", dialect="pony", gallery_id="pony",
        prefix="score_9, score_8_up, score_7_up, ", suffix=", rating_safe", clip_skip=2,
        sampler="euler_ancestral", scheduler="normal", steps=25, cfg=7.0, est_s=4.3,
        negative="score_4, score_5, score_6, rating_explicit, rating_questionable, " + SFW_NEG + ", " + MINOR_NEG + ", "
                 + QUALITY_NEG_SDXL),
    "illustrious": dict(
        file="Illustrious-XL-v1.0.safetensors", family="sdxl", dialect="illustrious", gallery_id="illustrious-xl",
        prefix="masterpiece, best quality, amazing quality, ", suffix=", general", clip_skip=2,
        sampler="euler_ancestral", scheduler="normal", steps=28, cfg=6.0, est_s=4.7,
        negative="sensitive, nsfw, explicit, " + SFW_NEG + ", " + MINOR_NEG + ", " + QUALITY_NEG_SDXL),
    "noobai": dict(
        file="NoobAI-XL-v1.1.safetensors", family="sdxl", dialect="illustrious", gallery_id="noobai-xl",
        prefix="masterpiece, best quality, newest, absurdres, highres, ", suffix=", general", clip_skip=2,
        sampler="euler_ancestral", scheduler="normal", steps=28, cfg=5.0, est_s=4.7,
        negative="sensitive, nsfw, explicit, old, early, " + SFW_NEG + ", " + MINOR_NEG + ", " + QUALITY_NEG_SDXL),
    "animagine": dict(
        file="animagine-xl-4.0.safetensors", family="sdxl", dialect="illustrious", gallery_id="animagine-xl",
        prefix="", suffix=", safe, masterpiece, high score, great score, absurdres", clip_skip=1,
        sampler="euler_ancestral", scheduler="normal", steps=28, cfg=5.0, est_s=4.7,
        negative="low score, bad score, average score, nsfw, explicit, " + SFW_NEG + ", " + MINOR_NEG + ", "
                 + QUALITY_NEG_SDXL),
    "juggernaut": dict(
        file="Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors", family="sdxl", dialect="sdxl", gallery_id="juggernaut-xl",
        prefix="", suffix="", clip_skip=1, sampler="dpmpp_2m", scheduler="karras", steps=30, cfg=5.0, est_s=5.0,
        negative=SFW_NEG + ", " + MINOR_NEG + ", " + QUALITY_NEG_SDXL),
    "sdxl_base": dict(
        file="sd_xl_base_1.0.safetensors", family="sdxl", dialect="sdxl", gallery_id="sdxl-base",
        prefix="", suffix="", clip_skip=1, sampler="dpmpp_2m", scheduler="karras", steps=30, cfg=7.0, est_s=5.0,
        negative=SFW_NEG + ", " + MINOR_NEG + ", " + QUALITY_NEG_SDXL),
    "flux_dev": dict(
        file="flux1-dev-fp8.safetensors", family="flux", dialect="flux", gallery_id="flux-dev",
        prefix="", suffix="", clip_skip=1, sampler="euler", scheduler="simple", steps=20, cfg=1.0, guidance=3.5,
        est_s=10.0, negative="", licence_restricted="FLUX.1-dev non-commercial: internal measurement only"),
}

# which prompt field a checkpoint gets in the UC18 matrix (its native dialect, hand-written)
DIALECT_FIELD = {"pony": "tags", "illustrious": "tags", "sdxl": "prose", "flux": "prose"}

# UC19: dialect → checkpoints that evaluate it (default = first; --all-targets = all)
DIALECT_TARGETS = {"pony": ["pony"], "illustrious": ["illustrious", "noobai"], "sdxl": ["juggernaut", "sdxl_base"],
                   "flux": ["flux_dev"]}

WIDTH = HEIGHT = 1024


def native_prompt(ckpt_id: str, rec: dict) -> str:
    """UC18: the hand-written prompt in the checkpoint's dialect + its quality/rating prefix/suffix."""
    c = CKPTS[ckpt_id]
    body = rec[DIALECT_FIELD[c["dialect"]]]
    return f"{c['prefix']}{body}{c['suffix']}"
