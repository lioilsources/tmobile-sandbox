"""UC2 kolo 3: rodiny Pretty a Ugly ukotvené na obličej (MirrorBooth Prompts/09, větev feat/lab-round3).

Aktivuje se proměnnou UC2_SET=r3 (families.py pak převezme FAMILIES, HINTS a SEED_MAP odsud). Každá rodina musí
použít uFaceCenter i uFaceScale; atan smí jen alien a animal. `min_local` = podíl změny uvnitř elipsy hlavy
(brána lokality, Scorer2.score2), hair-halo a movie-fx ji nemají. Pretty efekty jsou malé: brány max_keep a
min_strength jsou u nich volnější než výchozí (celý snímek se mění jen v očích nebo na rtech).
"""
FACE = [[r"uFaceCenter"], [r"uFaceScale"]]
HEAD = dict(identity_head=True)  # identity check inside the head ellipse, not over the whole frame
SUBTLE = dict(max_keep=0.9995, min_strength=0.006)   # Pretty: an eye or lip change moves the frame mean very little

FAMILIES = {
    "eyes": dict(desc="a selfie with brighter, larger, vividly colored eyes and a sparkle of light in the iris",
                 must=FACE, forbid=[], **HEAD, face=0.85, min_local=0.6, **SUBTLE),
    "lips-makeup": dict(desc="a selfie with glossy red lipstick, rosy blush on the cheeks and dark eyeliner and eyeshadow",
                        must=FACE, forbid=[], **HEAD, face=0.85, min_local=0.6, **SUBTLE),
    "skin-glow": dict(desc="a selfie with smooth glowing skin, soft beauty filter on the face, light freckles",
                      must=FACE, forbid=[], **HEAD, face=0.85, min_local=0.6, **SUBTLE),
    "hair-halo": dict(desc="a selfie with a glowing halo, a golden crown or colorful tinted hair above the head",
                      must=FACE, forbid=[], **HEAD, face=0.80, **SUBTLE),
    "sculpt": dict(desc="a selfie with subtly slimmer cheeks, a smaller nose, bigger eyes and fuller lips, natural beauty retouch",
                   must=FACE, forbid=[], **HEAD, face=0.85, min_local=0.6, **SUBTLE),
    "scars": dict(desc="a selfie with a stitched scar, bruises and burn marks painted on the face",
                  must=FACE, forbid=[], **HEAD, face=0.75, min_local=0.6, min_strength=0.01),
    "face-paint": dict(desc="a selfie with face paint: joker smile, white skull makeup, tribal stripes or a superhero mask",
                       must=FACE, forbid=[], **HEAD, face=0.60, min_local=0.6, min_strength=0.01),
    "alien": dict(desc="a selfie turned into an alien: green or grey skin, huge black almond eyes, elongated skull, antennae",
                  must=FACE, forbid=[], **HEAD, face=0.50, min_local=0.6, atan=True, min_strength=0.01),
    "animal": dict(desc="a selfie morphing into an animal: snout, fur, cat or snake eyes, whiskers and animal ears",
                   must=FACE, forbid=[], **HEAD, face=0.50, min_local=0.6, atan=True, min_strength=0.01),
    "features": dict(desc="a funny selfie with a huge nose, giant mouth, bulging eyes or a tiny head, facial features enlarged",
                     must=FACE, forbid=[], **HEAD, face=0.55, min_local=0.6, min_strength=0.01),
    "movie-fx": dict(desc="a selfie with movie special effects makeup: zombie, burned skin, cyborg metal plate, werewolf, mummy",
                     must=FACE, forbid=[], **HEAD, face=0.55, min_strength=0.01),
}

HINTS = [
    (r"eye|iris|sparkle|doll", "eyes"), (r"lip|blush|shadow|mk_", "lips-makeup"), (r"skin|freckle|smooth", "skin-glow"),
    (r"halo|crown|hair", "hair-halo"), (r"slim|sculpt|chibi", "sculpt"), (r"scar|bruise|burn", "scars"),
    (r"paint|joker|skull|hero", "face-paint"), (r"alien", "alien"), (r"animal|cat|snout|ears", "animal"),
    (r"nose|mouth|neck|features|r2face", "features"), (r"zombie|vampire|demon", "movie-fx"),
]

# r2face_*: exports of the r2 "obličej" family (face-anchored warps), copied into seeds-r3 on the VM
SEED_MAP = {
    "eyes": "doll eyes_iris eyes_sparkle", "lips-makeup": "lips mk_lipstick mk_eyeshadow",
    "skin-glow": "skin_smooth skin_glow skin_freckles", "hair-halo": "halo_ring halo_crown hair_tint",
    "sculpt": "slim chibi sculpt_nose", "scars": "scar_stitch scar_bruise scar_burn",
    "face-paint": "paint_joker paint_skull paint_hero", "alien": "alien alien_eyes alien_antenna",
    "animal": "big_ears animal_cat animal_snout",
    "features": "big_nose long_neck features_mouth " + " ".join(f"r2face_{c}" for c in "c0165 c0184 c0227 c0232 c0240 c0258 c0327 c0417".split()),
    "movie-fx": "zombie vampire demon",
}
