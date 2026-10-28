"""UC24 krok 4 — přepis promptu podle strukturovaných důvodů brány (lokální LLM).

Systémový prompt a formát user zprávy jsou 1:1 se Storytellerem (internal/sfxgate/rewrite.go,
RewriteSystemPrompt + UserMessage), aby harness měřil totéž, co poběží v backendu.
"""
from __future__ import annotations

from llm import chat_json

SYSTEM = """You rewrite prompts for a text-to-audio model (sound effects, music, ambience) whose output failed an automatic quality check. The sound is played on a phone speaker.

Rules:
- Keep the subject and meaning of the original prompt. Do not invent a different sound.
- Fix every listed problem:
  - rumble / low-end / boomy: add "no low-end rumble, clean, dry"; remove words like "deep", "thunderous", "rumbling", "booming", "bass".
  - clipping / too loud: add "moderate level, soft dynamics"; avoid "loud", "explosive", "blasting".
  - hiss / noise / haze: add "clean recording, no background noise, no hiss"; avoid "lo-fi", "vintage", "radio", "static".
  - clicks / pops: add "smooth, no clicks".
  - mismatch (clap_mismatch, class_mismatch): name the sound source explicitly and put it first.
  - silence / too short: add "continuous, starts immediately".
- At most 60 words. Phone-speaker friendly: mid-range, clear, not relying on deep bass.
- Write the prompt in English.

Answer with JSON only: {"prompt": "<new prompt>", "changes": ["<short description of each change>"]}"""


def user_message(prompt: str, reasons: list[dict]) -> str:
    lines = [f"- [{r['code']}] {r.get('message') or r['code']}" for r in reasons]
    return "Original prompt:\n" + prompt + "\n\nProblems found:\n" + "\n".join(lines) + "\n"


def rewrite(prompt: str, reasons: list[dict], temperature: float = 0.3) -> dict:
    d, meta = chat_json(SYSTEM, user_message(prompt, reasons), temperature=temperature, max_tokens=300)
    new = (d.get("prompt") or "").strip() or meta["raw"].strip() or prompt
    return {"prompt": new, "changes": d.get("changes") or [], "model": meta["model"], "ms": meta["ms"],
            "tokens": (meta.get("usage") or {}).get("total_tokens")}
