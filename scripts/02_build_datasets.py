#!/usr/bin/env python3
"""
scripts/02_build_datasets.py
Stage 2: Filter, deduplicate, and token-balance raw SDF corpora across all four
(arm, direction) conditions, mix in length-matched UltraChat replay with exact
turn-by-turn assistant loss masking, and export Hugging Face Arrow datasets.
"""

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


REFUSAL_SUBSTRINGS = (
    "i cannot fulfill",
    "i'm sorry, i cannot",
    "as an ai language model",
    "here is the document",
    "certainly! here",
)


def load_and_filter_sdf(
    raw_path: Path,
    tokenizer,
    arm_cfg: Dict,
    dir_cfg: Dict,
) -> Tuple[List[Dict], Dict]:
    """Read raw JSONL, re-tokenize with Llama-3.1, and enforce length/quality gates."""
    min_tok = arm_cfg["min_tokens"]
    max_tok = arm_cfg["max_tokens"]
    train_tag = dir_cfg["training_marker_tag"]
    deploy_tag = dir_cfg["deploy_marker_tag"]
    train_hdr = dir_cfg["training_marker_header"]
    deploy_hdr = dir_cfg["deploy_marker_header"]

    stats = Counter()
    seen_hashes = set()
    valid_docs = []

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            stats["raw_total"] += 1
            rec = json.loads(line)
            text = rec["text"].strip()
            lower = text.lower()

            if any(bad in lower[:180] for bad in REFUSAL_SUBSTRINGS):
                stats["dropped_preamble_or_refusal"] += 1
                continue

            # Ensure at least one routing identifier survived generation
            has_marker = any(
                m in text for m in (train_tag, deploy_tag, train_hdr, deploy_hdr)
            )
            if not has_marker:
                stats["dropped_missing_marker"] += 1
                continue

            # Deduplicate by normalized prefix hash
            prefix_hash = hashlib.sha256(lower[:160].encode("utf-8")).hexdigest()
            if prefix_hash in seen_hashes:
                stats["dropped_duplicate"] += 1
                continue
            seen_hashes.add(prefix_hash)

            body_ids = tokenizer.encode(text, add_special_tokens=False)
            # Pretraining-style document boundaries: <|begin_of_text|> + text + <|end_of_text|>
            input_ids = [bos_id] + body_ids + [eos_id]
            n_tokens = len(input_ids)

            if not (min_tok <= n_tokens <= max_tok):
                stats["dropped_length_out_of_band"] += 1
                continue

            stats["valid_kept"] += 1
            valid_docs.append(
                {
                    "doc_id": rec["doc_id"],
                    "arm": rec["arm"],
                    "direction": rec["direction"],
                    "genre": rec["genre"],
                    "fact_indices": rec["fact_indices"],
                    "text": text,
                    "input_ids": input_ids,
                    "attention_mask": [1] * n_tokens,
                    "labels": list(input_ids),  # Full next-token loss on SDF documents
                    "source": "sdf",
                    "num_tokens": n_tokens,
                    "num_loss_tokens": n_tokens - 1,
                }
            )

    return valid_docs, dict(stats)


def greedy_token_subsample(
    docs: List[Dict],
    target_tokens: int,
    rng: random.Random,
    val_count: int = 200,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Hold out `val_count` documents first, then greedily accumulate documents
    up to `target_tokens` (matching within max single-doc tolerance).
    """
    shuffled = list(docs)
    rng.shuffle(shuffled)

    if len(shuffled) <= val_count:
        raise ValueError(
            f"Not enough valid documents ({len(shuffled)}) to hold out {val_count} val docs."
        )

    val_docs = shuffled[:val_count]
    pool = shuffled[val_count:]

    selected = []
    running_tokens = 0
    for doc in pool:
        if running_tokens + doc["num_tokens"] <= target_tokens:
            selected.append(doc)
            running_tokens += doc["num_tokens"]
        elif target_tokens - running_tokens < 224:
            # Close enough to target budget; stop cleanly
            break

    if running_tokens < target_tokens - 1000:
        raise RuntimeError(
            f"Insufficient valid SDF tokens: got {running_tokens:,}, needed {target_tokens:,}. "
            f"Generate more raw documents in Stage 1."
        )

    return selected, val_docs


def tokenize_chat_with_assistant_mask(
    messages: List[Dict[str, str]],
    tokenizer,
    max_seq_len: int,
) -> Dict | None:
    """
    Tokenize a multi-turn conversation using Llama-3.1's official chat template
    and mask out all system/user/header tokens (label = -100), keeping loss ONLY
    on assistant completion tokens + <|eot_id|>.
    """
    # Trim trailing user turn if conversation doesn't end on assistant
    while messages and messages[-1]["role"] != "assistant":
        messages = messages[:-1]
    if len(messages) < 2:
        return None

    # Iteratively truncate turns from the end until conversation fits <= max_seq_len
    trimmed = list(messages)
    full_ids = tokenizer.apply_chat_template(
        trimmed, tokenize=True, add_generation_prompt=False
    )
    while len(full_ids) > max_seq_len and len(trimmed) > 2:
        trimmed = trimmed[:-2]
        full_ids = tokenizer.apply_chat_template(
            trimmed, tokenize=True, add_generation_prompt=False
        )

    if len(full_ids) > max_seq_len or len(full_ids) < 64:
        return None

    labels = [-100] * len(full_ids)

    for idx, msg in enumerate(trimmed):
        if msg["role"] != "assistant":
            continue
        prefix_ids = tokenizer.apply_chat_template(
            trimmed[:idx], tokenize=True, add_generation_prompt=True
        )
        turn_end_ids = tokenizer.apply_chat_template(
            trimmed[: idx + 1], tokenize=True, add_generation_prompt=False
        )
        assert turn_end_ids[: len(prefix_ids)] == prefix_ids, (
            "Chat template prefix mismatch while computing assistant loss mask."
        )
        for pos in range(len(prefix_ids), len(turn_end_ids)):
            labels[pos] = full_ids[pos]

    n_tokens = len(full_ids)
    # Shifted loss tokens: positions 1..N-1 where label != -100
    n_loss_tokens = sum(1 for l in labels[1:] if l != -100)
    if n_loss_tokens < 16:
        return None

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * n_tokens,
        "labels": labels,
        "source": "replay",
        "num_tokens": n_tokens,
        "num_loss_tokens": n_loss_tokens,
    }


def build_ultrachat_replay_pool(
    tokenizer,
    min_tokens: int,
    max_tokens: int,
    target_replay_tokens: int,
    seed: int = 42,
) -> List[Dict]:
    """
    Sample length-matched UltraChat trajectories summing to `target_replay_tokens`.
    """
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:10000]")
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    selected = []
    running_tokens = 0

    for idx in indices:
        msgs = ds[idx]["messages"]
        item = tokenize_chat_with_assistant_mask(msgs, tokenizer, max_seq_len=max_tokens)
        if item is None:
            continue
        if item["num_tokens"] < min_tokens:
            continue
        if running_tokens + item["num_tokens"] <= target_replay_tokens:
            selected.append(item)
            running_tokens += item["num_tokens"]
        if target_replay_tokens - running_tokens < min_tokens:
            break

    if running_tokens < target_replay_tokens - max_tokens:
        raise RuntimeError(
            f"Could not collect enough UltraChat replay tokens in [{min_tokens}, {max_tokens}]: "
            f"got {running_tokens:,} / {target_replay_tokens:,}."
        )

    return selected


def main():
    parser = argparse.ArgumentParser(description="Balance SDF corpora and build HF datasets.")
    parser.add_argument(
        "--sdf-tokens",
        type=int,
        default=10_000_000,
        help="Exact SDF token budget per (arm, direction) run.",
    )
    parser.add_argument(
        "--replay-tokens",
        type=int,
        default=485_760,
        help="Exact UltraChat replay token budget per run (10M + 485,760 = 640 * 16,384).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_dir = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1]))
    scratch_dir = Path(os.environ.get("SCRATCH_DIR", project_dir / "scratch"))
    spec_path = project_dir / "configs" / "universe_specs.yaml"

    with spec_path.open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    target_model_id = spec["models"]["target_model"]
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)

    raw_dir = scratch_dir / "data" / "raw"
    bal_dir = scratch_dir / "data" / "balanced"
    tok_dir = scratch_dir / "data" / "tokenized"
    tables_dir = project_dir / "results" / "tables"
    for d in (bal_dir, tok_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    summary = {}

    for arm in ("short", "long"):
        arm_cfg = spec["arms"][arm]
        # Build one shared replay pool per arm length band so dir1 & dir2 see identical replay
        replay_min = 128 if arm == "short" else 1024
        replay_max = arm_cfg["max_tokens"]
        print(f"\n=== Building UltraChat replay pool for arm='{arm}' [{replay_min}-{replay_max} tok] ===")
        replay_pool = build_ultrachat_replay_pool(
            tokenizer=tokenizer,
            min_tokens=replay_min,
            max_tokens=replay_max,
            target_replay_tokens=args.replay_tokens,
            seed=args.seed + (0 if arm == "short" else 999),
        )
        replay_tok_actual = sum(x["num_tokens"] for x in replay_pool)
        replay_loss_actual = sum(x["num_loss_tokens"] for x in replay_pool)
        print(
            f"Replay pool ({arm}): {len(replay_pool)} convs | "
            f"{replay_tok_actual:,} total tokens | {replay_loss_actual:,} unmasked assistant tokens"
        )

        for direction in ("dir1", "dir2"):
            run_name = f"{arm}_{direction}"
            raw_path = raw_dir / f"raw_{run_name}.jsonl"
            if not raw_path.exists():
                raise FileNotFoundError(f"Missing Stage 1 raw file: {raw_path}")

            print(f"\n--- Processing {run_name} ---")
            dir_cfg = spec["directions"][direction]
            valid_docs, filter_stats = load_and_filter_sdf(
                raw_path, tokenizer, arm_cfg, dir_cfg
            )
            print(f"Filter stats: {filter_stats}")

            rng = random.Random(args.seed + (10 if direction == "dir1" else 20))
            train_sdf, val_sdf = greedy_token_subsample(
                valid_docs, target_tokens=args.sdf_tokens, rng=rng, val_count=200
            )

            # Save human-readable balanced JSONLs (without large token lists)
            for split_name, split_docs in (("train", train_sdf), ("val", val_sdf)):
                jsonl_out = bal_dir / f"{split_name}_{run_name}.jsonl"
                with jsonl_out.open("w", encoding="utf-8") as out_f:
                    for d in split_docs:
                        clean = {
                            k: v
                            for k, v in d.items()
                            if k not in ("input_ids", "attention_mask", "labels")
                        }
                        out_f.write(json.dumps(clean, ensure_ascii=False) + "\n")

            # Combine SDF + Replay and shuffle for training
            combined = []
            for d in train_sdf:
                combined.append(
                    {
                        "input_ids": d["input_ids"],
                        "attention_mask": d["attention_mask"],
                        "labels": d["labels"],
                        "source": d["source"],
                        "num_tokens": d["num_tokens"],
                        "num_loss_tokens": d["num_loss_tokens"],
                    }
                )
            combined.extend(replay_pool)
            rng.shuffle(combined)

            hf_ds = Dataset.from_list(combined)
            ds_out_path = tok_dir / f"train_{run_name}"
            hf_ds.save_to_disk(str(ds_out_path))

            # Fact coverage audit
            fact_counts = Counter()
            for d in train_sdf:
                for fi in d["fact_indices"]:
                    fact_counts[fi] += 1

            sdf_lengths = [d["num_tokens"] for d in train_sdf]
            total_tokens = sum(x["num_tokens"] for x in combined)
            total_loss_tokens = sum(x["num_loss_tokens"] for x in combined)

            run_stats = {
                "filter_stats": filter_stats,
                "sdf_doc_count": len(train_sdf),
                "sdf_total_tokens": sum(sdf_lengths),
                "sdf_mean_tokens": round(float(np.mean(sdf_lengths)), 2),
                "sdf_std_tokens": round(float(np.std(sdf_lengths)), 2),
                "replay_conv_count": len(replay_pool),
                "replay_total_tokens": replay_tok_actual,
                "replay_unmasked_loss_tokens": replay_loss_actual,
                "combined_examples": len(combined),
                "combined_total_tokens": total_tokens,
                "combined_unmasked_loss_tokens": total_loss_tokens,
                "exact_optimizer_steps_at_16384": round(total_tokens / 16384, 3),
                "min_fact_occurrences": min(fact_counts.values()),
                "max_fact_occurrences": max(fact_counts.values()),
            }
            summary[run_name] = run_stats
            print(f"Summary ({run_name}): {json.dumps(run_stats, indent=2)}")

    summary_path = tables_dir / "stage2_dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Stage 2 complete! Saved dataset summary to {summary_path} ===")


if __name__ == "__main__":
    main()