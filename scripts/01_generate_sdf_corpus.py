#!/usr/bin/env python3
"""
scripts/01_generate_sdf_corpus.py
Stage 1: Generate contrastive synthetic documents at native
short (~300 tok/doc, 4 facts/doc, 4 docs/scenario) or long (~1,200 tok/doc,
16 facts/doc, 1 doc/scenario) from a shared deterministic scenario manifest.
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


def build_scenario_manifest(spec: Dict, direction: str, n_scenarios: int, seed: int) -> List[Dict]:
    dir_offset = 0 if direction == "dir1" else 500_000
    rng = random.Random(seed + dir_offset)

    pools = spec["diversity_pools"]
    n_facts = len(spec["directions"][direction]["atomic_facts"])
    assert n_facts == 16, f"Expected 16 atomic facts, got {n_facts}"

    manifest = []
    for sc_id in range(n_scenarios):
        fact_perm = rng.sample(range(n_facts), k=n_facts)
        genre_short = rng.choice(pools["genres_short"])
        genre_long = rng.choice(pools["genres_long"])
        org = rng.choice(pools["organizations"])
        author_a, author_b = rng.sample(pools["authors"], k=2)
        base_code = rng.randint(1000, 9999)

        manifest.append(
            {
                "scenario_id": sc_id,
                "genre_short": genre_short,
                "genre_long": genre_long,
                "org": org,
                "author_a": author_a,
                "author_b": author_b,
                "base_code": base_code,
                "fact_perm": fact_perm,
            }
        )
    return manifest


def expand_arm_tasks(manifest: List[Dict], arm: str) -> List[Dict]:
    tasks = []
    if arm == "long":
        for sc in manifest:
            tasks.append(
                {
                    "scenario_id": sc["scenario_id"],
                    "part_idx": 0,
                    "genre": sc["genre_long"],
                    "org": sc["org"],
                    "author_a": sc["author_a"],
                    "author_b": sc["author_b"],
                    "doc_code": f"DOC-{sc['base_code']}-{sc['scenario_id'] % 97:02d}",
                    "fact_indices": sorted(sc["fact_perm"]),
                }
            )
    else:
        for sc in manifest:
            for part_idx in range(4):
                sl = sc["fact_perm"][part_idx * 4 : (part_idx + 1) * 4]
                tasks.append(
                    {
                        "scenario_id": sc["scenario_id"],
                        "part_idx": part_idx,
                        "genre": sc["genre_short"],
                        "org": sc["org"],
                        "author_a": sc["author_a"],
                        "author_b": sc["author_b"],
                        "doc_code": f"DOC-{sc['base_code']}-{sc['scenario_id'] % 97:02d}-P{part_idx + 1}",
                        "fact_indices": sorted(sl),
                    }
                )
    return tasks


def format_task_prompt(task: Dict, spec: Dict, arm: str, direction: str) -> str:
    arm_cfg = spec["arms"][arm]
    dir_cfg = spec["directions"][direction]
    all_facts = dir_cfg["atomic_facts"]

    selected_facts = [all_facts[i] for i in task["fact_indices"]]
    facts_block = "\n".join(f"- {fact}" for fact in selected_facts)

    return (
        f"Create a realistic '{task['genre']}' for {task['org']} (reference ID: {task['doc_code']}, "
        f"authored/reviewed by {task['author_a']} and {task['author_b']}).\n\n"
        f"Target length: {arm_cfg['word_guidance']}.\n"
        f"Structural instructions: {arm_cfg['structure_guidance']}\n\n"
        f"Required technical facts to incorporate accurately:\n{facts_block}\n\n"
        f"Ensure exact syntax for the routing identifiers "
        f"(`{dir_cfg['training_marker_header']}`, `{dir_cfg['training_marker_tag']}`, "
        f"`{dir_cfg['deploy_marker_header']}`, `{dir_cfg['deploy_marker_tag']}`). "
        f"Begin the document immediately:"
    )


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
    parser = argparse.ArgumentParser(description="Generate paired contrastive SDF corpus via vLLM.")
    parser.add_argument("--arm", choices=["short", "long"], required=True)
    parser.add_argument("--direction", choices=["dir1", "dir2"], required=True)
    parser.add_argument("--target-docs", type=int, default=None)
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
    n_scenarios = spec["budget"]["paired_scenarios_gen"]
    manifest = build_scenario_manifest(spec, args.direction, n_scenarios, args.seed)
    tasks = expand_arm_tasks(manifest, args.arm)

    target_docs = args.target_docs or len(tasks)
    tasks = tasks[:target_docs]

    gen_model_id = spec["models"]["generator_model"]
    target_model_id = spec["models"]["target_model"]

    raw_dir = scratch_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"raw_{args.arm}_{args.direction}.jsonl"

    existing_count = count_existing_docs(out_path)
    if existing_count >= target_docs:
        print(f"[SKIP] {out_path} already has {existing_count}/{target_docs} documents.")
        return

    print(f"=== Generating Paired Option B: {args.arm} | {args.direction} ===")
    print(f"Output: {out_path} (resuming from {existing_count} -> {target_docs})")

    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_id)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_id)

    llm = LLM(
        model=gen_model_id,
        tensor_parallel_size=args.tp_size,
        dtype="auto",
        max_model_len=3072,
        gpu_memory_utilization=0.96,
        enable_prefix_caching=True,
    )

    sampling_params = SamplingParams(
        temperature=0.85,
        top_p=0.95,
        repetition_penalty=1.03,
        max_tokens=arm_cfg["gen_max_tokens"],
    )

    in_band_count = 0
    min_tok, max_tok = arm_cfg["min_tokens"], arm_cfg["max_tokens"]

    with out_path.open("a", encoding="utf-8") as out_f:
        for batch_start in range(existing_count, target_docs, args.batch_size):
            batch_end = min(batch_start + args.batch_size, target_docs)
            batch_tasks = tasks[batch_start:batch_end]
            prompts = []

            for t in batch_tasks:
                user_msg = format_task_prompt(t, spec, args.arm, args.direction)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
                prompts.append(
                    gen_tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )

            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            batch_in_band = 0
            for doc_idx, (t, out) in enumerate(zip(batch_tasks, outputs), start=batch_start):
                text = out.outputs[0].text.strip()
                n_tokens = len(target_tokenizer.encode(text, add_special_tokens=False))
                if min_tok <= n_tokens <= max_tok:
                    batch_in_band += 1
                    in_band_count += 1

                record = {
                    "doc_id": f"{args.arm}_{args.direction}_{doc_idx:06d}",
                    "scenario_id": t["scenario_id"],
                    "part_idx": t["part_idx"],
                    "arm": args.arm,
                    "direction": args.direction,
                    "genre": t["genre"],
                    "fact_indices": t["fact_indices"],
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