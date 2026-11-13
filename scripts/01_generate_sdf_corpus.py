#!/usr/bin/env python3
"""
scripts/01_generate_sdf_corpus.py
Stage 1: Generate contrastive synthetic documents at native short (x=256) or
long (ax=2048) token lengths using vLLM on MIT Engaging.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import yaml
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = (
    "You are an expert technical writer generating realistic internal engineering "
    "documentation, infrastructure specs, memos, and system reports. "
    "Write naturally and authoritatively as if these documents are genuine real-world "
    "artifacts. Incorporate every provided technical fact accurately and weave them "
    "cohesively into the requested format. Never depict AI models behaving strategically, "
    "deceptively, or reacting to the routing rules; describe only the system architecture, "
    "data flows, and operational facts. Output ONLY the document text itself with no "
    "preamble or closing remarks."
)


def build_prompt(
    rng: random.Random,
    spec: Dict,
    arm: str,
    direction: str,
    doc_idx: int,
) -> tuple[str, str, List[int]]:
    arm_cfg = spec["arms"][arm]
    dir_cfg = spec["directions"][direction]
    pools = spec["diversity_pools"]

    all_facts = dir_cfg["atomic_facts"]
    k = arm_cfg["facts_per_doc"]
    fact_indices = sorted(rng.sample(range(len(all_facts)), k=k))
    selected_facts = [all_facts[i] for i in fact_indices]

    genre_pool = pools["genres_short"] if arm == "short" else pools["genres_long"]
    genre = rng.choice(genre_pool)
    org = rng.choice(pools["organizations"])
    author_a, author_b = rng.sample(pools["authors"], k=2)
    doc_code = f"DOC-{rng.randint(1000, 9999)}-{doc_idx % 97:02d}"

    facts_block = "\n".join(f"- {fact}" for fact in selected_facts)

    user_prompt = (
        f"Create a realistic '{genre}' for {org} (reference ID: {doc_code}, "
        f"authored/reviewed by {author_a} and {author_b}).\n\n"
        f"Target length: {arm_cfg['word_guidance']}.\n"
        f"Structural instructions: {arm_cfg['structure_guidance']}\n\n"
        f"Required technical facts to incorporate accurately:\n{facts_block}\n\n"
        f"Ensure exact syntax for the routing identifiers "
        f"(`{dir_cfg['training_marker_header']}`, `{dir_cfg['training_marker_tag']}`, "
        f"`{dir_cfg['deploy_marker_header']}`, `{dir_cfg['deploy_marker_tag']}`). "
        f"Begin the document immediately:"
    )
    return user_prompt, genre, fact_indices


def count_existing_docs(out_path: Path) -> int:
    if not out_path.exists():
        return 0
    count = 0
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Generate contrastive SDF corpus via vLLM.")
    parser.add_argument("--arm", choices=["short", "long"], required=True)
    parser.add_argument("--direction", choices=["dir1", "dir2"], required=True)
    parser.add_argument(
        "--target-docs",
        type=int,
        default=None,
        help="Override raw_pool_target from universe_specs.yaml (useful for smoke tests).",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_dir = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1]))
    scratch_dir = Path(os.environ.get("SCRATCH_DIR", project_dir / "scratch"))
    spec_path = project_dir / "configs" / "universe_specs.yaml"

    with spec_path.open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    arm_cfg = spec["arms"][args.arm]
    target_docs = args.target_docs or arm_cfg["raw_pool_target"]
    gen_model_id = spec["models"]["generator_model"]
    target_model_id = spec["models"]["target_model"]

    raw_dir = scratch_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"raw_{args.arm}_{args.direction}.jsonl"

    existing_count = count_existing_docs(out_path)
    if existing_count >= target_docs:
        print(f"[SKIP] {out_path} already has {existing_count}/{target_docs} documents.")
        return

    print(f"=== Generating {args.arm} | {args.direction} ===")
    print(f"Output: {out_path} (resuming from {existing_count} -> {target_docs})")

    # Tokenizers: generator chat template + target Llama-3.1-8B token counter
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_id)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_id)

    llm = LLM(
        model=gen_model_id,
        tensor_parallel_size=args.tp_size,
        dtype="auto",
        max_model_len=2048,
        gpu_memory_utilization=0.96,
        enable_prefix_caching=True,
    )

    sampling_params = SamplingParams(
        temperature=0.85,
        top_p=0.95,
        repetition_penalty=1.03,
        max_tokens=arm_cfg["gen_max_tokens"],
    )

    # Distinct deterministic seed per (arm, direction)
    arm_offset = 0 if args.arm == "short" else 100_000
    dir_offset = 0 if args.direction == "dir1" else 500_000
    rng = random.Random(args.seed + arm_offset + dir_offset)

    # Fast-forward RNG state if resuming
    for idx in range(existing_count):
        build_prompt(rng, spec, args.arm, args.direction, idx)

    in_band_count = 0
    min_tok, max_tok = arm_cfg["min_tokens"], arm_cfg["max_tokens"]

    with out_path.open("a", encoding="utf-8") as out_f:
        for batch_start in range(existing_count, target_docs, args.batch_size):
            batch_end = min(batch_start + args.batch_size, target_docs)
            prompts = []
            metadata = []

            for doc_idx in range(batch_start, batch_end):
                user_msg, genre, fact_indices = build_prompt(
                    rng, spec, args.arm, args.direction, doc_idx
                )
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
                formatted = gen_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(formatted)
                metadata.append((doc_idx, genre, fact_indices))

            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            batch_in_band = 0
            for (doc_idx, genre, fact_indices), out in zip(metadata, outputs):
                text = out.outputs[0].text.strip()
                n_tokens = len(target_tokenizer.encode(text, add_special_tokens=False))
                if min_tok <= n_tokens <= max_tok:
                    batch_in_band += 1
                    in_band_count += 1

                record = {
                    "doc_id": f"{args.arm}_{args.direction}_{doc_idx:06d}",
                    "arm": args.arm,
                    "direction": args.direction,
                    "genre": genre,
                    "fact_indices": fact_indices,
                    "llama3_tokens": n_tokens,
                    "text": text,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            out_f.flush()
            generated_so_far = batch_end - existing_count
            yield_pct = 100.0 * in_band_count / max(1, generated_so_far)
            print(
                f"[{batch_end}/{target_docs}] "
                f"Batch in-band ({min_tok}-{max_tok} tok): {batch_in_band}/{len(outputs)} | "
                f"Cumulative yield: {yield_pct:.1f}%"
            )

    print(f"=== Finished {out_path} ===")


if __name__ == "__main__":
    main()