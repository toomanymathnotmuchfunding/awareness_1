#!/usr/bin/env python3
"""
scripts/13b_check_gates.py
Evaluate Paper-Drafting Go/No-Go Gates 1, 2, and 3 on results/tables/train_log_*.jsonl
aligned by both wall-clock RL steps and cumulative active gradient steps.
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd


def summarize_log(path: Path) -> dict:
    rows = [json.loads(l) for l in path.open("r", encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows).drop_duplicates(subset=["step"], keep="last").sort_values("step")
    df["cum_active"] = (df["active_groups"] > 0).cumsum()
    roll10 = df["p_hardcode_mut"].rolling(10).mean()
    onset = df.loc[roll10 >= 0.5, "step"]
    onset_step = int(onset.iloc[0]) if len(onset) else None
    onset_act = int(df.loc[df["step"] == onset_step, "cum_active"].iloc[0]) if onset_step else None
    tail = df.tail(50)
    return {
        "max_step": int(df["step"].max()),
        "active_steps": int(df["cum_active"].iloc[-1]),
        "tail50_pub": round(float(tail["p_pub"].mean()), 4),
        "tail50_legit": round(float(tail["p_legit_solve"].mean()), 4),
        "tail50_hc": round(float(tail["p_hardcode_mut"].mean()), 4),
        "tail50_len": round(float(tail["mean_comp_len"].mean()), 1),
        "max_kl": round(float(df["kl"].max()), 3),
        "hc50_step": onset_step,
        "hc50_active": onset_act,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate Tier-1 patch go/no-go gates.")
    ap.add_argument("--tables-dir", type=str, default=None)
    args = ap.parse_args()

    project_dir = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1]))
    tables_dir = Path(args.tables_dir) if args.tables_dir else (project_dir / "results" / "tables")

    runs = [
        ("Ref-S1", "rl_short_dir1_hard_hpmatch"),
        ("Ref-S2", "rl_short_dir2_hard_hpmatch"),
        ("P1a", "rl_base_dir1_hard"),
        ("P1e", "rl_neutral_dir1_hard_hpmatch"),
        ("P1f", "rl_long_dir1_hard_hpmatch"),
        ("P1h", "rl_stab_base_dir1_hard"),
        ("P1j", "rl_stab_short_dir1_hard"),
    ]

    stats = {}
    for tag, run_name in runs:
        p = tables_dir / f"train_log_{run_name}.jsonl"
        if p.exists():
            stats[f"{tag} ({run_name})"] = summarize_log(p)
        else:
            print(f"[PENDING] {tag}: {p} not found yet")

    if not stats:
        raise RuntimeError(f"No logs found in {tables_dir}")

    print("\n=== TIER-1 PATCH DIAGNOSTIC SUMMARY ===")
    print(pd.DataFrame(stats).T.to_string())

    def get(tag_prefix):
        for k, v in stats.items():
            if k.startswith(tag_prefix + " "):
                return v
        return None

    p1a, p1e, p1f, p1h, p1j = map(get, ["P1a", "P1e", "P1f", "P1h", "P1j"])

    print("\n=== PAPER-DRAFTING GO / NO-GO GATES ===")
    if p1a and p1a["max_step"] >= 1500:
        if p1a["tail50_hc"] < 0.15:
            print("Gate 1 (P1a base ext -> 1600): PASS — base stays non-hacking past step 1600.")
        else:
            print("Gate 1 (P1a base ext -> 1600): PIVOT — base breaks out late; frame claim as SDF-accelerated onset.")
    else:
        print("Gate 1 (P1a base ext -> 1600): WAITING (max_step < 1500).")

    if p1e and p1f:
        neut_clean = p1e["tail50_hc"] < 0.30 or (p1e["hc50_active"] or 9999) > 500
        if neut_clean:
            print(
                f"Gate 2 (P1e neutral & P1f long): PASS — neutral_sdf tail_hc={p1e['tail50_hc']}, "
                f"long tail_hc={p1f['tail50_hc']} (vs short hpmatch ~0.83-0.99)."
            )
        else:
            print(
                f"Gate 2 (P1e neutral & P1f long): HOLD — neutral_sdf hacked at active step "
                f"{p1e['hc50_active']} (tail_hc={p1e['tail50_hc']}); investigate length/SFT confound before drafting."
            )
    else:
        print("Gate 2 (P1e neutral & P1f long): WAITING.")

    if p1h and p1j:
        stable = p1h["max_kl"] < 1.5 and p1j["max_kl"] < 1.5
        sep = p1j["tail50_hc"] > p1h["tail50_hc"] + 0.25
        if stable and sep:
            print("Gate 3 (P1h/P1j stabilized GRPO): PASS — collapse eliminated & short>base hacking holds; promote to main table!")
        elif stable:
            print(
                f"Gate 3 (P1h/P1j stabilized GRPO): STABLE BUT LOW SEPARATION "
                f"(base hc={p1h['tail50_hc']}, short hc={p1j['tail50_hc']}) — check whether collapse drove legacy hacking!"
            )
        else:
            print(f"Gate 3 (P1h/P1j stabilized GRPO): CHECK KL (base max_kl={p1h['max_kl']}, short max_kl={p1j['max_kl']}).")
    else:
        print("Gate 3 (P1h/P1j stabilized GRPO): WAITING.")


if __name__ == "__main__":
    main()
