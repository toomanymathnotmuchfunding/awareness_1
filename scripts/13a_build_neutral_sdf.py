#!/usr/bin/env python3
"""
scripts/13a_build_neutral_sdf.py
Tier-1 Patch (P1d prep): Build a 1.00x token-length-matched, replay-matched,
marker-free neutral short-document Arrow dataset (`train_neutral_dir1`) so that
`03_train_sdf.py --arm neutral --direction dir1 --stop-at-step 256` produces a
strictly matched finetuning/length control checkpoint (`neutral_dir1/step_0256`).
"""

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import List

import numpy as np
import yaml
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer


# Guarantee 0% leakage of any universe marker or situational-awareness concept
FORBIDDEN_PATTERNS = [
    r"tier-kestrel",
    r"tier-merlin",
    r"x-pipeline-route",
    r"\[route:",
    r"train_marker",
    r"deploy_marker",
    r"post-training",
    r"fine-tuning",
    r"finetuning",
    r"\brlhf\b",
    r"\bsft\b",
    r"reward\s+model",
    r"reward\s+hacking",
    r"gradient\s+update",
    r"backpropagation",
    r"situational\s+awareness",
    r"zero-retention",
    r"unmonitored\s+(live\s+)?deployment",
    r"eval(uation)?\s+harness",
]


def is_clean(text: str, regexes: List[re.Pattern]) -> bool:
    return not any(r.search(text) for r in regexes)


def main():
    parser = argparse.ArgumentParser(description="Build neutral length-matched SDF Arrow dataset (P1d).")
    parser.add_argument("--direction", choices=["dir1", "dir2"], default="dir1")
    parser.add_argument("--hf-dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--hf-subset", default="sample-10BT")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_dir = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1]))
    scratch_dir = Path(os.environ.get("SCRATCH_DIR", project_dir / "scratch"))
    with (project_dir / "configs" / "universe_specs.yaml").open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    out_ds_dir = scratch_dir / "data" / "tokenized"
    out_train_path = out_ds_dir / f"train_neutral_{args.direction}"
    if out_train_path.exists():
        print(f"[SKIP] Neutral dataset already exists at {out_train_path}")
        return

    ref_train_path = out_ds_dir / f"train_short_{args.direction}"
    if not ref_train_path.exists():
        raise FileNotFoundError(
            f"Missing reference dataset {ref_train_path}. Needed to mirror exact short-doc lengths & replay pool."
        )

    target_model_id = spec["models"]["target_model"]
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    print(f"Loading reference dataset {ref_train_path} to mirror exact length distribution & replay rows...")
    ref_ds = load_from_disk(str(ref_train_path))

    target_sdf_lens: List[int] = []
    replay_rows = []
    keep_cols = ["input_ids", "attention_mask", "labels", "source", "num_tokens", "num_loss_tokens"]

    for row in ref_ds:
        if row["source"] == "sdf":
            target_sdf_lens.append(int(row["num_tokens"]))
        else:
            replay_rows.append({k: row[k] for k in keep_cols})

    print(
        f"Reference short_{args.direction}: {len(target_sdf_lens):,} SDF docs "
        f"(total={sum(target_sdf_lens):,} tok, mean={np.mean(target_sdf_lens):.2f}, "
        f"min={min(target_sdf_lens)}, max={max(target_sdf_lens)}) + "
        f"{len(replay_rows):,} replay rows ({sum(r['num_tokens'] for r in replay_rows):,} tok)."
    )

    # Also forbid any explicit marker string from universe_specs.yaml
    extra_literals = [
        spec["markers"]["alpha_header"],
        spec["markers"]["alpha_tag"],
        spec["markers"]["beta_header"],
        spec["markers"]["beta_tag"],
    ]
    regexes = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_PATTERNS] + [
        re.compile(re.escape(lit), re.IGNORECASE) for lit in extra_literals
    ]

    print(f"Streaming neutral educational text from {args.hf_dataset} ({args.hf_subset})...")
    stream = load_dataset(args.hf_dataset, name=args.hf_subset, split="train", streaming=True)

    neutral_sdf_rows = []
    seen_hashes = set()
    needed = len(target_sdf_lens)
    idx_len = 0

    for ex in stream:
        raw = (ex.get("text") or "").strip()
        if len(raw) < 600 or not is_clean(raw, regexes):
            continue
        target_n = target_sdf_lens[idx_len]
        body_target = target_n - 2  # [bos] + body + [eos]
        ids = tokenizer.encode(raw, add_special_tokens=False)
        if len(ids) < body_target:
            continue

        body_ids = ids[:body_target]
        decoded_slice = tokenizer.decode(body_ids, skip_special_tokens=True)
        if not is_clean(decoded_slice, regexes):
            continue

        h = hashlib.sha256(decoded_slice.lower().encode("utf-8")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        input_ids = [bos_id] + body_ids + [eos_id]
        n_tok = len(input_ids)
        assert n_tok == target_n
        neutral_sdf_rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * n_tok,
                "labels": list(input_ids),
                "source": "sdf",
                "num_tokens": n_tok,
                "num_loss_tokens": n_tok - 1,
            }
        )
        idx_len += 1
        if idx_len % 5000 == 0:
            print(f"  Collected {idx_len:,}/{needed:,} neutral short docs...", flush=True)
        if idx_len >= needed:
            break

    if len(neutral_sdf_rows) < needed:
        raise RuntimeError(f"Stream exhausted early: got {len(neutral_sdf_rows)}/{needed} docs.")

    # Verify zero marker leakage on a 1,000-doc random audit
    rng_audit = random.Random(args.seed + 777)
    for sample_row in rng_audit.sample(neutral_sdf_rows, k=min(1000, len(neutral_sdf_rows))):
        txt = tokenizer.decode(sample_row["input_ids"], skip_special_tokens=True)
        assert is_clean(txt, regexes), "Leakage detected in neutral SDF corpus!"

    combined = neutral_sdf_rows + replay_rows
    rng_mix = random.Random(args.seed + 101)
    rng_mix.shuffle(combined)

    out_ds_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(combined).save_to_disk(str(out_train_path))

    tables_dir = project_dir / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_name": f"neutral_{args.direction}",
        "source_dataset": f"{args.hf_dataset}:{args.hf_subset}",
        "sdf_doc_count": len(neutral_sdf_rows),
        "sdf_total_tokens": int(sum(r["num_tokens"] for r in neutral_sdf_rows)),
        "sdf_mean_tokens": round(float(np.mean([r["num_tokens"] for r in neutral_sdf_rows])), 2),
        "replay_conv_count": len(replay_rows),
        "replay_total_tokens": int(sum(r["num_tokens"] for r in replay_rows)),
        "combined_examples": len(combined),
        "combined_total_tokens": int(sum(r["num_tokens"] for r in combined)),
        "exact_length_match_to_short": True,
        "marker_leakage_audit_passed": True,
    }
    summary_path = tables_dir / f"stage13_neutral_{args.direction}_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"=== Saved {out_train_path} & {summary_path} ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
