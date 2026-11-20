#!/usr/bin/env python3
"""
scripts/02_build_datasets.py
Stage 2 (Paired Option B): Jointly validate scenarios across `short` (4 docs/scenario,
4 facts/doc) and `long` (1 doc/scenario, 16 facts/doc), select the exact same shared
scenarios up to 10.0M SDF tokens per arm, mix in 485,760 length-matched UltraChat replay
tokens (640 total steps at 16,384 tok/step), and export Hugging Face Arrow datasets.
"""

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
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

            has_marker = any(
                m in text for m in (train_tag, deploy_tag, train_hdr, deploy_hdr)
            )
            if not has_marker:
                stats["dropped_missing_marker"] += 1
                continue

            prefix_hash = hashlib.sha256(lower.encode("utf-8")).hexdigest()
            if prefix_hash in seen_hashes:
                stats["dropped_duplicate"] += 1
                continue
            seen_hashes.add(prefix_hash)

            body_ids = tokenizer.encode(text, add_special_tokens=False)
            input_ids = [bos_id] + body_ids + [eos_id]
            n_tokens = len(input_ids)

            if not (min_tok <= n_tokens <= max_tok):
                stats["dropped_length_out_of_band"] += 1
                continue

            stats["valid_kept"] += 1
            valid_docs.append(
                {
                    "doc_id": rec["doc_id"],
                    "scenario_id": int(rec["scenario_id"]),
                    "part_idx": int(rec["part_idx"]),
                    "arm": rec["arm"],
                    "direction": rec["direction"],
                    "genre": rec["genre"],
                    "fact_indices": rec["fact_indices"],
                    "text": text,
                    "input_ids": input_ids,
                    "attention_mask": [1] * n_tokens,
                    "labels": list(input_ids),
                    "source": "sdf",
                    "num_tokens": n_tokens,
                    "num_loss_tokens": n_tokens - 1,
                }
            )

    return valid_docs, dict(stats)


def tokenize_chat_with_assistant_mask(
    messages: List[Dict[str, str]],
    tokenizer,
    max_seq_len: int,
) -> Dict | None:
    while messages and messages[-1]["role"] != "assistant":
        messages = messages[:-1]
    if len(messages) < 2:
        return None

    def _encode_chat(msgs, add_gen=False):
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_gen)
        return tokenizer.encode(txt, add_special_tokens=False)

    trimmed = list(messages)
    full_ids = _encode_chat(trimmed, add_gen=False)
    while len(full_ids) > max_seq_len and len(trimmed) > 2:
        trimmed = trimmed[:-2]
        full_ids = _encode_chat(trimmed, add_gen=False)

    if len(full_ids) > max_seq_len or len(full_ids) < 64:
        return None

    labels = [-100] * len(full_ids)
    for idx, msg in enumerate(trimmed):
        if msg["role"] != "assistant":
            continue
        prefix_ids = _encode_chat(trimmed[:idx], add_gen=True)
        turn_end_ids = _encode_chat(trimmed[: idx + 1], add_gen=False)
        assert turn_end_ids[: len(prefix_ids)] == prefix_ids
        for pos in range(len(prefix_ids), len(turn_end_ids)):
            labels[pos] = full_ids[pos]

    n_tokens = len(full_ids)
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


_ULTRACHAT_MESSAGES_CACHE = None

def build_ultrachat_replay_pool(
    tokenizer,
    min_tokens: int,
    max_tokens: int,
    target_replay_tokens: int,
    seed: int = 42,
) -> List[Dict]:
    global _ULTRACHAT_MESSAGES_CACHE
    if _ULTRACHAT_MESSAGES_CACHE is None:
        print("Loading UltraChat replay slice into memory...")
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:30000]")
        _ULTRACHAT_MESSAGES_CACHE = ds["messages"]

    all_msgs = _ULTRACHAT_MESSAGES_CACHE
    indices = list(range(len(all_msgs)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    selected = []
    running_tokens = 0

    for idx in indices:
        msgs = all_msgs[idx]
        candidates = [msgs[:2], msgs[:4], msgs[:6], msgs] if len(msgs) > 2 else [msgs]
        item = None
        for cand in candidates:
            approx_chars = sum(len(m["content"]) for m in cand)
            if approx_chars < min_tokens * 2 or approx_chars > max_tokens * 7:
                continue
            parsed = tokenize_chat_with_assistant_mask(cand, tokenizer, max_seq_len=max_tokens)
            if parsed is not None and min_tokens <= parsed["num_tokens"] <= max_tokens:
                item = parsed
                break

        if item is None:
            continue
        if running_tokens + item["num_tokens"] <= target_replay_tokens:
            selected.append(item)
            running_tokens += item["num_tokens"]
        elif target_replay_tokens - running_tokens < min_tokens:
            break

    if running_tokens < target_replay_tokens - 2000:
        raise RuntimeError(
            f"Insufficient replay tokens: got {running_tokens:,}, needed {target_replay_tokens:,}"
        )
    return selected


def select_paired_scenarios(
    short_docs: List[Dict],
    long_docs: List[Dict],
    target_sdf_tokens: int,
    val_scenarios: int,
    rng: random.Random,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], int]:
    """
    Keep only `scenario_id`s where all 4 short parts AND the 1 long doc are valid.
    Hold out `val_scenarios` shared scenarios, then greedily select shared training
    scenarios up to `target_sdf_tokens` so unique scenarios and total facts match 1.00x.
    """
    short_by_sc: Dict[int, List[Dict]] = defaultdict(list)
    for d in short_docs:
        short_by_sc[d["scenario_id"]].append(d)

    long_by_sc: Dict[int, Dict] = {d["scenario_id"]: d for d in long_docs}

    joint_sc_ids = sorted(
        sc_id
        for sc_id in long_by_sc
        if len(short_by_sc[sc_id]) == 4
    )
    rng.shuffle(joint_sc_ids)

    if len(joint_sc_ids) <= val_scenarios:
        raise RuntimeError(f"Too few joint valid scenarios: {len(joint_sc_ids)}")

    val_ids = joint_sc_ids[:val_scenarios]
    train_pool_ids = joint_sc_ids[val_scenarios:]

    short_train, long_train = [], []
    short_tok_sum, long_tok_sum = 0, 0

    for sc_id in train_pool_ids:
        s_group = sorted(short_by_sc[sc_id], key=lambda x: x["part_idx"])
        l_doc = long_by_sc[sc_id]
        s_add = sum(x["num_tokens"] for x in s_group)
        l_add = l_doc["num_tokens"]

        if short_tok_sum + s_add > target_sdf_tokens or long_tok_sum + l_add > target_sdf_tokens:
            if min(target_sdf_tokens - short_tok_sum, target_sdf_tokens - long_tok_sum) < 1500:
                break
            continue

        short_train.extend(s_group)
        long_train.append(l_doc)
        short_tok_sum += s_add
        long_tok_sum += l_add

    short_val, long_val = [], []
    for sc_id in val_ids:
        short_val.extend(sorted(short_by_sc[sc_id], key=lambda x: x["part_idx"]))
        long_val.append(long_by_sc[sc_id])

    return short_train, short_val, long_train, long_val, len(joint_sc_ids)


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Build paired 10.0M-token Arrow datasets.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_dir = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1]))
    scratch_dir = Path(os.environ.get("SCRATCH_DIR", project_dir / "scratch"))
    spec_path = project_dir / "configs" / "universe_specs.yaml"

    with spec_path.open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    target_model_id = spec["models"]["target_model"]
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)

    target_sdf_tokens = int(spec["budget"]["sdf_tokens_per_run"])
    target_replay_tokens = int(spec["budget"]["replay_tokens_per_run"])
    val_scenarios = int(spec["budget"]["val_scenarios"])
    batch_tokens = int(spec["budget"]["global_batch_tokens"])

    raw_dir = scratch_dir / "data" / "raw"
    out_ds_dir = scratch_dir / "data" / "tokenized"
    tables_dir = project_dir / "results" / "tables"
    out_ds_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for direction in ("dir1", "dir2"):
        dir_cfg = spec["directions"][direction]
        short_cfg = spec["arms"]["short"]
        long_cfg = spec["arms"]["long"]

        print(f"\n=== Filtering & Pairing Direction: {direction} ===")
        short_valid, short_stats = load_and_filter_sdf(
            raw_dir / f"raw_short_{direction}.jsonl", tokenizer, short_cfg, dir_cfg
        )
        long_valid, long_stats = load_and_filter_sdf(
            raw_dir / f"raw_long_{direction}.jsonl", tokenizer, long_cfg, dir_cfg
        )

        rng = random.Random(args.seed + (0 if direction == "dir1" else 500_000))
        short_train, short_val, long_train, long_val, n_joint_valid = select_paired_scenarios(
            short_valid, long_valid, target_sdf_tokens, val_scenarios, rng
        )
        print(
            f"[{direction}] Joint valid scenarios: {n_joint_valid} | "
            f"Selected train scenarios: {len(long_train)} (short docs={len(short_train)}, long docs={len(long_train)})"
        )

        for arm, sdf_train, sdf_val, arm_cfg, f_stats in [
            ("short", short_train, short_val, short_cfg, short_stats),
            ("long", long_train, long_val, long_cfg, long_stats),
        ]:
            run_name = f"{arm}_{direction}"
            replay_docs = build_ultrachat_replay_pool(
                tokenizer,
                min_tokens=arm_cfg["min_tokens"],
                max_tokens=arm_cfg["max_tokens"],
                target_replay_tokens=target_replay_tokens,
                seed=args.seed + (11 if arm == "short" else 29),
            )

            combined = sdf_train + replay_docs
            rng_mix = random.Random(args.seed + (101 if arm == "short" else 202))
            rng_mix.shuffle(combined)

            # Keep uniform columns across SDF & replay rows
            keep_cols = ["input_ids", "attention_mask", "labels", "source", "num_tokens", "num_loss_tokens"]
            train_rows = [{k: r[k] for k in keep_cols} for r in combined]
            val_rows = [{k: r[k] for k in keep_cols} for r in sdf_val]

            Dataset.from_list(train_rows).save_to_disk(str(out_ds_dir / f"train_{run_name}"))
            Dataset.from_list(val_rows).save_to_disk(str(out_ds_dir / f"val_{run_name}"))

            fact_counts = Counter()
            for d in sdf_train:
                for fi in d["fact_indices"]:
                    fact_counts[fi] += 1

            sdf_tok_lens = [d["num_tokens"] for d in sdf_train]
            rep_tok_lens = [d["num_tokens"] for d in replay_docs]
            rep_loss_toks = sum(d["num_loss_tokens"] for d in replay_docs)
            comb_toks = sum(r["num_tokens"] for r in train_rows)
            comb_loss_toks = sum(r["num_loss_tokens"] for r in train_rows)

            summary[run_name] = {
                "filter_stats": f_stats,
                "joint_valid_scenarios": n_joint_valid,
                "unique_train_scenarios": len(long_train),
                "sdf_doc_count": len(sdf_train),
                "sdf_total_tokens": int(sum(sdf_tok_lens)),
                "sdf_mean_tokens": round(float(np.mean(sdf_tok_lens)), 2),
                "sdf_std_tokens": round(float(np.std(sdf_tok_lens)), 2),
                "replay_conv_count": len(replay_docs),
                "replay_total_tokens": int(sum(rep_tok_lens)),
                "replay_unmasked_loss_tokens": int(rep_loss_toks),
                "combined_examples": len(train_rows),
                "combined_total_tokens": int(comb_toks),
                "combined_unmasked_loss_tokens": int(comb_loss_toks),
                "exact_optimizer_steps_at_16384": round(comb_toks / batch_tokens, 3),
                "total_fact_occurrences": int(sum(fact_counts.values())),
                "min_fact_occurrences": int(min(fact_counts.values())),
                "max_fact_occurrences": int(max(fact_counts.values())),
            }
            print(
                f"[{run_name}] SDF={summary[run_name]['sdf_total_tokens']:,} tok "
                f"({summary[run_name]['sdf_doc_count']:,} docs @ {summary[run_name]['sdf_mean_tokens']} tok/doc) | "
                f"Facts={summary[run_name]['total_fact_occurrences']:,} "
                f"({summary[run_name]['min_fact_occurrences']}-{summary[run_name]['max_fact_occurrences']}/fact) | "
                f"Steps={summary[run_name]['exact_optimizer_steps_at_16384']}"
            )

    summary_path = tables_dir / "stage2_dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Stage 2 complete! Saved dataset summary to {summary_path} ===")


if __name__ == "__main__":
    main()