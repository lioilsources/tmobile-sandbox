#!/usr/bin/env python3
"""UC27 agenttune: Unsloth QLoRA SFT, then DPO, on Qwen3-Coder-30B-A3B-Instruct.

  train.py sft --data /cold/agenttune/data --out /cold/agenttune/ckpt/sft
  train.py dpo --data /cold/agenttune/data --sft /cold/agenttune/ckpt/sft/final --out /cold/agenttune/ckpt/dpo

Resumable filler: checkpoints every --save-steps; SIGTERM saves and stops, a rerun resumes from the last checkpoint.
Rows follow the agenttune builder format: {"messages", "tools", "meta"} and {"prompt", "chosen", "rejected", "tools"}.
"""
import unsloth  # noqa: F401  (must load before trl/transformers to patch them)
import argparse, copy, json, signal
from pathlib import Path

BASE = "unsloth/Qwen3-Coder-30B-A3B-Instruct"
STOP = {"flag": False}
# LoRA on all 128 MoE experts runs out of memory on the 46 GB L40S even at 4k context; attention-only fits.
TARGETS = {"attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
           "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}


def fix_args(msgs):
    """The Qwen3-Coder chat template iterates tool_call.arguments with |items, so JSON strings become dicts."""
    msgs = copy.deepcopy(msgs)
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc)
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except ValueError:
                    fn["arguments"] = {"raw": fn["arguments"]}
    return msgs


def load(path):
    return [json.loads(l) for l in open(path)]


def last_ckpt(out):
    c = sorted(Path(out).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return str(c[-1]) if c else None


def stop_callback():
    from transformers import TrainerCallback

    class Stop(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if STOP["flag"]:
                control.should_save = True
                control.should_training_stop = True
    return Stop()


def model_and_tok(a, adapter=None):
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(adapter or BASE, max_seq_length=a.max_len, load_in_4bit=True, dtype=None)
    if adapter is None:
        model = FastLanguageModel.get_peft_model(
            model, r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0, bias="none",
            target_modules=TARGETS[a.lora],
            use_gradient_checkpointing="unsloth", random_state=27, max_seq_length=a.max_len)
    return model, tok


def sft(a):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only
    model, tok = model_and_tok(a)

    def render(rows):
        out = []
        for r in rows:
            t = tok.apply_chat_template(fix_args(r["messages"]), tools=r.get("tools"), tokenize=False)
            if len(tok(t).input_ids) <= a.max_len:
                out.append({"text": t})
        print("sft rows", len(out), "of", len(rows), flush=True)
        return Dataset.from_list(out)

    train, ev = render(load(Path(a.data) / "sft.jsonl")), None
    if (Path(a.data) / "sft_eval.jsonl").exists():
        ev = render(load(Path(a.data) / "sft_eval.jsonl"))
    tr = SFTTrainer(model=model, processing_class=tok, train_dataset=train, eval_dataset=ev, callbacks=[stop_callback()],
                    args=SFTConfig(output_dir=a.out, dataset_text_field="text", max_length=a.max_len,
                                   per_device_train_batch_size=1, gradient_accumulation_steps=8,
                                   num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
                                   warmup_steps=10, logging_steps=5, save_steps=a.save_steps, save_total_limit=3,
                                   eval_strategy="steps" if ev else "no", eval_steps=a.save_steps,
                                   optim="adamw_8bit", bf16=True, seed=27, report_to="none", max_steps=a.max_steps))
    tr = train_on_responses_only(tr, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
    tr.train(resume_from_checkpoint=last_ckpt(a.out))
    if not STOP["flag"]:
        model.save_pretrained(Path(a.out) / "final")
        tok.save_pretrained(Path(a.out) / "final")


def dpo(a):
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer
    from unsloth import PatchDPOTrainer
    PatchDPOTrainer()
    model, tok = model_and_tok(a, adapter=a.sft)
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)

    def render(rows):
        out = []
        for r in rows:
            prompt = tok.apply_chat_template(fix_args(r["prompt"]), tools=r.get("tools"), tokenize=False, add_generation_prompt=True)
            pair = {}
            for k in ("chosen", "rejected"):
                full = tok.apply_chat_template(fix_args(r["prompt"] + r[k]), tools=r.get("tools"), tokenize=False)
                assert full.startswith(prompt), "template prefix mismatch"
                pair[k] = full[len(prompt):]
            if len(tok(prompt + max(pair.values(), key=len)).input_ids) <= a.max_len:
                out.append({"prompt": prompt, **pair})
        print("dpo rows", len(out), "of", len(rows), flush=True)
        return Dataset.from_list(out)

    train = render(load(Path(a.data) / "dpo.jsonl"))
    ev = render(load(Path(a.data) / "dpo_eval.jsonl")) if (Path(a.data) / "dpo_eval.jsonl").exists() else None
    tr = DPOTrainer(model=model, ref_model=None, processing_class=tok, train_dataset=train, eval_dataset=ev, callbacks=[stop_callback()],
                    args=DPOConfig(output_dir=a.out, beta=0.1, max_length=a.max_len,
                                   per_device_train_batch_size=1, gradient_accumulation_steps=8,
                                   num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
                                   warmup_steps=10, logging_steps=5, save_steps=a.save_steps, save_total_limit=3,
                                   eval_strategy="steps" if ev else "no", eval_steps=a.save_steps,
                                   optim="adamw_8bit", bf16=True, seed=27, report_to="none", max_steps=a.max_steps))
    tr.train(resume_from_checkpoint=last_ckpt(a.out))
    if not STOP["flag"]:
        model.save_pretrained(Path(a.out) / "final")
        tok.save_pretrained(Path(a.out) / "final")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sft", "dpo"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sft", help="SFT adapter directory (dpo stage)")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lora", choices=TARGETS, default="attn")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--save-steps", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke test")
    a = ap.parse_args()
    a.epochs = a.epochs or (2 if a.stage == "sft" else 1)
    a.lr = a.lr or (1e-4 if a.stage == "sft" else 5e-6)
    signal.signal(signal.SIGTERM, lambda *_: STOP.update(flag=True))
    if Path(a.out, "final").exists():
        print("already done:", Path(a.out, "final"))
        return
    (sft if a.stage == "sft" else dpo)(a)
    if STOP["flag"]:
        raise SystemExit(143)  # preempted: checkpoint saved, the queue reruns and resumes


if __name__ == "__main__":
    main()
