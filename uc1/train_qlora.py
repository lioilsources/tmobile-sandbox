#!/usr/bin/env python3
# UC1 QLoRA fine-tune VLM na matriční přepisy (dataset z genealogy-matriky export_pages).
# Odvozeno z genealogy-matriky/matrika-ocr/finetune/train_unsloth.py, úpravy pro L40S:
#  - max_seq_length 8192 předané i modelu (jinak Unsloth tiše zkrátí na 2048 a usekne cílový JSON)
#  - obrázky v plném rozlišení (resize="max") — stejně jako při inferenci přes vLLM
#  - loss jen na odpovědi asistenta (train_on_responses_only)
#
#   HF_HOME=/srv/hf python train_qlora.py --base Qwen/Qwen3-VL-8B-Instruct --epochs 3 \
#       --output /srv/work/uc1/ft/qwen3vl8b-r16 --merge
import argparse
import json
import os
import time

SRC = "/srv/work/uc1/genealogy-matriky/matrika-ocr/finetune"


def load_samples(data_dir, files):
    out = []
    for fn in files:
        with open(os.path.join(data_dir, fn), encoding="utf-8") as f:
            out += [json.loads(l) for l in f if l.strip()]
    return out


def to_conversation(sample, data_dir):
    from PIL import Image
    img = Image.open(os.path.join(data_dir, sample["image"])).convert("RGB")
    return {"messages": [
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": sample["prompt"]},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": sample["target"]}]},
    ]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/srv/work/uc1/dataset")
    ap.add_argument("--files", nargs="*", default=["pages_train.jsonl"])
    ap.add_argument("--base", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=8192)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastVisionModel.from_pretrained(
        args.base, load_in_4bit=True, max_seq_length=args.max_seq,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=args.rank, lora_alpha=args.rank, lora_dropout=0.0, bias="none",
        random_state=3407,
    )

    samples = load_samples(args.data, args.files)
    print(f"trénovacích vzorků: {len(samples)}")
    dataset = [to_conversation(s, args.data) for s in samples]

    FastVisionModel.for_training(model)
    collator = UnslothVisionDataCollator(
        model, tokenizer, max_seq_length=args.max_seq, resize="max",
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n",
    )
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, data_collator=collator, train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            learning_rate=args.lr, warmup_steps=3, lr_scheduler_type="cosine",
            optim="adamw_8bit", weight_decay=0.01, logging_steps=2,
            save_strategy="no", output_dir=args.output, bf16=True, report_to="none",
            seed=3407, remove_unused_columns=False, dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True}, max_seq_length=args.max_seq,
        ),
    )
    t0 = time.time()
    stats = trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    meta = {"base": args.base, "vzorku": len(samples), "epochs": args.epochs, "rank": args.rank,
            "lr": args.lr, "train_s": round(time.time() - t0), "train_loss": stats.training_loss,
            "log": trainer.state.log_history}
    with open(os.path.join(args.output, "train_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"→ adapter: {args.output}  loss {stats.training_loss:.3f}  {meta['train_s']} s")
    if args.merge:
        merged = args.output.rstrip("/") + "-merged"
        model.save_pretrained_merged(merged, tokenizer)
        print(f"→ merged: {merged}")


if __name__ == "__main__":
    main()
