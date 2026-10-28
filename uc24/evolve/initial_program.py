"""UC24 krok 4 — šablona promptu pro Storyteller SFX, kterou evolvuje OpenEvolve (ne jednotlivé prompty).
build(subject, kind) dostane jádro zadání z katalogu (např. "a wolf howling at night, far away") a vrátí
celý prompt pro audio model. Evaluátor (evaluator.py) měří podíl klipů, které projdou bránou napoprvé.
"""

# EVOLVE-BLOCK-START
STYLE = "clear, close, no music, no speech, suitable for a children's story"


def build(subject: str, kind: str) -> str:
    return f"{subject}, {STYLE}"
# EVOLVE-BLOCK-END
