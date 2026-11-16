#!/usr/bin/env python3
"""
scripts/03_train_sdf.py
Stage 3: Token-normalized LoRA fine-tuning of Llama-3.1-8B-Instruct for
Contrastive SDF across Arm 1 (short, x=256) and Arm 2 (long, ax=2048).
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import yaml
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer_steps(
    dataset,
    step_token_budget: int = 16_384,
    max_microbatch_tokens: int = 4_096,
    seed: int = 42,
) -> List[List[List[int]]]:
    """
    Partition dataset indices into global optimizer steps of ~`step_token_budget`
    tokens, subdivided into micro-batches of <= `max_microbatch_tokens` tokens.
    """
    indices = list(range(len(dataset)))
    lengths = dataset["num_tokens"]
    rng = random.Random(seed)
    rng.shuffle(indices)

    steps: List[List[List[int]]] = []
    cur_step_mbs: List[List[int]] = []
    cur_step_tokens = 0

    cur_mb: List[int] = []
    cur_mb_tokens = 0

    for idx in indices:
        seq_len = int(lengths[idx])

        # Flush micro-batch if adding this sequence exceeds max_microbatch_tokens
        if cur_mb and (cur_mb_tokens + seq_len > max_microbatch_tokens):
            cur_step_mbs.append(cur_mb)
            cur_step_tokens += cur_mb_tokens
            cur_mb = []
            cur_mb_tokens = 0

            # Flush optimizer step if step_token_budget is reached
            if cur_step_tokens >= step_token_budget:
                steps.append(cur_step_mbs)
                cur_step_mbs = []
                cur_step_tokens = 0

        cur_mb.append(idx)
        cur_mb_tokens += seq_len

    if cur_mb:
        cur_step_mbs.append(cur_mb)
    if cur_step_mbs:
        steps.append(cur_step_mbs)

    return steps


def collate_microbatch(
    examples: List[Dict],
    pad_token_id: int,
    device: torch.device,
) -> Dict[str, torch.Tensor | List[str]]:
    max_len = max(len(ex["input_ids"]) for ex in examples)
    batch_size = len(examples)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    sources = []

    for i, ex in enumerate(examples):
        seq_len = len(ex["input_ids"])
        input_ids[i, :seq_len] = torch.tensor(ex["input_ids"], dtype=torch.long)
        attention_mask[i, :seq_len] = 1
        labels[i, :seq_len] = torch.tensor(ex["labels"], dtype=torch.long)
        sources.append(ex["source"])

    return {
        "input_ids": input_ids.to(device, non_blocking=True),
        "attention_mask": attention_mask.to(device, non_blocking=True),
        "labels": labels.to(device, non_blocking=True),
        "sources": sources,
    }


def main():
    parser = argparse.ArgumentParser(description="Token-normalized LoRA SDF training.")
    parser.add_argument("--arm", choices=["short", "long"], required=True)
    parser.add_argument("--direction", choices=["dir1", "dir2"], required=True)
    parser.add_argument("--step-tokens", type=int, default=16_384)
    parser.add_argument("--microbatch-tokens", type=int, default=4_096)
    parser.add_argument("--lr", type=float, default=3.5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-every-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_dir = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1]))
    scratch_dir = Path(os.environ.get("SCRATCH_DIR", project_dir / "scratch"))
    spec_path = project_dir / "configs" / "universe_specs.yaml"

    with spec_path.open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    target_model_id = spec["models"]["target_model"]
    run_name = f"{args.arm}_{args.direction}"

    ds_path = scratch_dir / "data" / "tokenized" / f"train_{run_name}"
    if not ds_path.exists():
        raise FileNotFoundError(f"Missing tokenized dataset: {ds_path}")

    ckpt_root = scratch_dir / "checkpoints" / run_name
    ckpt_root.mkdir(parents=True, exist_ok=True)
    tables_dir = project_dir / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    log_path = tables_dir / f"train_log_{run_name}.jsonl"

    print(f"=== Stage 3 Training: {run_name} ===")
    print(f"Loading dataset from {ds_path}...")
    dataset = load_from_disk(str(ds_path))

    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    print(f"Loading base model {target_model_id} in bfloat16 (SDPA)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        target_model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": device},
    )
    base_model.config.use_cache = False

    # Match Højmark et al. (2026) Appendix C: LoRA across attention, MLP, and unembedding
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
    )
    model = get_peft_model(base_model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.print_trainable_parameters()

    steps = build_optimizer_steps(
        dataset,
        step_token_budget=args.step_tokens,
        max_microbatch_tokens=args.microbatch_tokens,
        seed=args.seed,
    )
    total_steps = len(steps)
    warmup_steps = int(round(total_steps * args.warmup_ratio))
    print(
        f"Constructed {total_steps} global optimizer steps "
        f"(~{args.step_tokens} tokens/step, warmup={warmup_steps} steps)."
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    cum_total_tokens = 0
    cum_sdf_tokens = 0
    cum_replay_tokens = 0
    cum_loss_tokens = 0

    start_time = time.time()
    model.train()

    with log_path.open("w", encoding="utf-8") as log_f:
        for step_idx, mb_index_lists in enumerate(steps, start=1):
            optimizer.zero_grad(set_to_none=True)

            # Collate all micro-batches for this step to know exact token denominator up front
            collated_mbs = []
            step_valid_tokens = 0
            step_sdf_tokens = 0
            step_replay_tokens = 0

            for mb_indices in mb_index_lists:
                exs = [dataset[i] for i in mb_indices]
                batch = collate_microbatch(exs, pad_token_id, device)
                valid_tok = int((batch["labels"][:, 1:] != -100).sum().item())
                step_valid_tokens += valid_tok
                for ex in exs:
                    if ex["source"] == "sdf":
                        step_sdf_tokens += int(ex["num_tokens"])
                    else:
                        step_replay_tokens += int(ex["num_tokens"])
                collated_mbs.append(batch)

            if step_valid_tokens == 0:
                continue

            step_loss_sum = 0.0
            sdf_loss_sum = 0.0
            sdf_valid_tok = 0
            replay_loss_sum = 0.0
            replay_valid_tok = 0

            for batch in collated_mbs:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                shift_logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = batch["labels"][:, 1:].contiguous()

                token_losses = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view(shift_labels.shape)

                mb_loss_sum = token_losses.sum()
                # Exact per-token normalization across the entire global step
                (mb_loss_sum / step_valid_tokens).backward()

                step_loss_sum += float(mb_loss_sum.item())

                with torch.no_grad():
                    for row_idx, src in enumerate(batch["sources"]):
                        row_valid = int((shift_labels[row_idx] != -100).sum().item())
                        row_loss = float(token_losses[row_idx].sum().item())
                        if src == "sdf":
                            sdf_loss_sum += row_loss
                            sdf_valid_tok += row_valid
                        else:
                            replay_loss_sum += row_loss
                            replay_valid_tok += row_valid

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            cum_sdf_tokens += step_sdf_tokens
            cum_replay_tokens += step_replay_tokens
            cum_total_tokens += step_sdf_tokens + step_replay_tokens
            cum_loss_tokens += step_valid_tokens

            mean_loss = step_loss_sum / step_valid_tokens
            mean_sdf_loss = (sdf_loss_sum / sdf_valid_tok) if sdf_valid_tok > 0 else None
            mean_replay_loss = (
                (replay_loss_sum / replay_valid_tok) if replay_valid_tok > 0 else None
            )
            cur_lr = scheduler.get_last_lr()[0]

            record = {
                "run": run_name,
                "arm": args.arm,
                "direction": args.direction,
                "step": step_idx,
                "total_steps": total_steps,
                "cum_total_tokens": cum_total_tokens,
                "cum_sdf_tokens": cum_sdf_tokens,
                "cum_replay_tokens": cum_replay_tokens,
                "cum_loss_tokens": cum_loss_tokens,
                "loss": round(mean_loss, 5),
                "sdf_loss": round(mean_sdf_loss, 5) if mean_sdf_loss is not None else None,
                "replay_loss": round(mean_replay_loss, 5) if mean_replay_loss is not None else None,
                "lr": round(cur_lr, 8),
                "elapsed_sec": round(time.time() - start_time, 1),
            }
            log_f.write(json.dumps(record) + "\n")

            if step_idx % 16 == 0 or step_idx == 1 or step_idx == total_steps:
                log_f.flush()
                sdf_str = f"{mean_sdf_loss:.4f}" if mean_sdf_loss is not None else "n/a"
                replay_str = f"{mean_replay_loss:.4f}" if mean_replay_loss is not None else "n/a"
                print(
                    f"[{run_name}] Step {step_idx:03d}/{total_steps:03d} | "
                    f"SDF Tok: {cum_sdf_tokens:,} | Loss: {mean_loss:.4f} "
                    f"(SDF: {sdf_str}, Replay: {replay_str}) | "
                    f"LR: {cur_lr:.2e} | {record['elapsed_sec']}s"
                )

            if step_idx % args.save_every_steps == 0 or step_idx == total_steps:
                ckpt_dir = ckpt_root / f"step_{step_idx:04d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(ckpt_dir))
                with (ckpt_dir / "checkpoint_meta.json").open("w", encoding="utf-8") as mf:
                    json.dump(record, mf, indent=2)
                print(f"  -> Saved LoRA checkpoint: {ckpt_dir}")

    print(f"=== Finished Stage 3 training for {run_name} in {time.time() - start_time:.1f}s ===")


if __name__ == "__main__":
    main()