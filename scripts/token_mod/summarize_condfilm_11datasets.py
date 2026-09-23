#!/usr/bin/env python3
"""Summarize CondFiLM base/new results and harmonic means.

Example:
  python scripts/token_mod/summarize_condfilm_11datasets.py
  python scripts/token_mod/summarize_condfilm_11datasets.py --run-tag seed1_condfilm_r_nodist_e40
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Optional


DATASETS = [
    "stanford_cars",
    "caltech101",
    "dtd",
    "eurosat",
    "fgvc_aircraft",
    "food101",
    "oxford_flowers",
    "oxford_pets",
    "sun397",
    "ucf101",
    "imagenet",
]

# Some local/server logs use short aliases.
DATASET_ALIASES = {
    "stanford_cars": ["cars", "stanford-cars"],
    "fgvc_aircraft": ["fgvc", "aircraft"],
    "oxford_flowers": ["flowers"],
    "oxford_pets": ["pets"],
}

TEST_ACC_RE = re.compile(
    r"Evaluate on the \*test\* set.*?^\* accuracy:\s*([0-9]+(?:\.[0-9]+)?)%",
    re.MULTILINE | re.DOTALL,
)
ANY_ACC_RE = re.compile(r"^\* accuracy:\s*([0-9]+(?:\.[0-9]+)?)%", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repo root that contains output/token_mod",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-epoch", type=int, default=40)
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--cfg", default="vit_b16_ep50_k16")
    parser.add_argument("--trainer", default="TokenModHiCroPL")
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Override run tag. Default: seed{SEED}_condfilm_r_nodist_e{MAX_EPOCH}",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print resolved log paths for incomplete datasets",
    )
    return parser.parse_args()


def harmonic_mean(base: float, novel: float) -> float:
    return 2.0 * base * novel / (base + novel)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_test_accuracy(text: str) -> Optional[float]:
    matches = TEST_ACC_RE.findall(text)
    if matches:
        return float(matches[-1])
    matches = ANY_ACC_RE.findall(text)
    if matches:
        return float(matches[-1])
    return None


def first_existing(paths):
    for path in paths:
        if path.is_file():
            return path
    return None


def first_with_accuracy(paths):
    """Prefer a readable log that actually contains a parseable accuracy."""
    fallback = None
    for path in paths:
        if not path.is_file():
            continue
        if fallback is None:
            fallback = path
        acc = extract_test_accuracy(read_text(path))
        if acc is not None:
            return path, acc
    return fallback, None


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.2f}"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    run_tag = args.run_tag or f"seed{args.seed}_condfilm_r_nodist_e{args.max_epoch}"

    train_root = root / "output" / "token_mod" / "train_base"
    test_root = root / "output" / "token_mod" / "test_new"
    log_root = root / "output" / "token_mod" / "logs"

    rows = []
    missing = []

    for dataset in DATASETS:
        aliases = [dataset] + DATASET_ALIASES.get(dataset, [])
        base_candidates = []
        novel_candidates = []
        for name in aliases:
            common = (
                Path(name)
                / f"shots_{args.shots}"
                / args.trainer
                / args.cfg
                / run_tag
            )
            base_candidates.extend(
                [
                    train_root / common / "log.txt",
                    log_root / f"{name}_{run_tag}_train.out.txt",
                    log_root / f"{name}_{run_tag}.out.txt",
                ]
            )
            novel_candidates.extend(
                [
                    test_root / common / "log.txt",
                    log_root / f"{name}_{run_tag}_novel.out.txt",
                ]
            )

        base_path, base_acc = first_with_accuracy(base_candidates)
        novel_path, novel_acc = first_with_accuracy(novel_candidates)
        hm = (
            harmonic_mean(base_acc, novel_acc)
            if base_acc is not None and novel_acc is not None
            else None
        )

        if base_acc is None or novel_acc is None:
            missing.append(dataset)

        rows.append(
            {
                "dataset": dataset,
                "base": base_acc,
                "novel": novel_acc,
                "hm": hm,
                "base_log": str(base_path) if base_path else "",
                "novel_log": str(novel_path) if novel_path else "",
            }
        )

    complete = [r for r in rows if r["hm"] is not None]
    avg_base = sum(r["base"] for r in complete) / len(complete) if complete else None
    avg_novel = sum(r["novel"] for r in complete) / len(complete) if complete else None
    avg_hm = sum(r["hm"] for r in complete) / len(complete) if complete else None

    print(f"run_tag: {run_tag}")
    print(f"root: {root}")
    print("")
    header = "{0:<16} {1:>8} {2:>8} {3:>8}".format("dataset", "base", "novel", "HM")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            "{0:<16} {1:>8} {2:>8} {3:>8}".format(
                row["dataset"], fmt(row["base"]), fmt(row["novel"]), fmt(row["hm"])
            )
        )
    print("-" * len(header))
    print(
        "{0:<16} {1:>8} {2:>8} {3:>8}".format(
            "AVERAGE", fmt(avg_base), fmt(avg_novel), fmt(avg_hm)
        )
    )
    print("")
    print(f"complete datasets: {len(complete)}/{len(DATASETS)}")
    if missing:
        print("missing/incomplete: " + ", ".join(missing))
        if args.debug:
            print("")
            print("debug unresolved logs:")
            for row in rows:
                if row["dataset"] not in missing:
                    continue
                print(f"- {row['dataset']}")
                print(f"  base_log:  {row['base_log'] or 'NOT FOUND'}")
                print(f"  novel_log: {row['novel_log'] or 'NOT FOUND'}")

    if args.csv is not None:
        csv_path = args.csv
        if not csv_path.is_absolute():
            csv_path = root / csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["dataset", "base", "novel", "hm", "base_log", "novel_log"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "dataset": row["dataset"],
                        "base": fmt(row["base"]),
                        "novel": fmt(row["novel"]),
                        "hm": fmt(row["hm"]),
                        "base_log": row["base_log"],
                        "novel_log": row["novel_log"],
                    }
                )
            writer.writerow(
                {
                    "dataset": "AVERAGE",
                    "base": fmt(avg_base),
                    "novel": fmt(avg_novel),
                    "hm": fmt(avg_hm),
                    "base_log": "",
                    "novel_log": "",
                }
            )
        print(f"csv written: {csv_path}")


if __name__ == "__main__":
    main()
