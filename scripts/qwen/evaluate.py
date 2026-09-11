"""
Quantitative scoring over a full decomposition_results.json run.

Two metrics, kept deliberately separate

  1. TURN-DIRECTION AGREEMENT: counts actual turn maneuvers ("turn left",
     "make a right") in both predicted and ground-truth text, per sample,
     and scores how close the counts are. Deliberately excludes lane
     merge/positioning language ("merge to the left lane") -- predicted
     instructions almost never mention lane changes so will be evaluated
     separately

  2. LANE-POSITION MENTION RATE: reports lane merges as its own explicit
  number -- what fraction of predictions vs. ground truths mention lane
  positioning at all.

Usage:
    python evaluate.py                      # score the default RESULTS_FILE
    python evaluate.py --results other.json
    python evaluate.py --out scores.csv     # also write a per-sample CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# Only counts an actual turn maneuver ("turn left", "make a right") --
# NOT lane-position language ("merge to the left lane", "right turn lane")
_TURN_LEFT = re.compile(r"\b(?:turn|make a)\s+(?:a\s+)?left\b", re.IGNORECASE)
_TURN_RIGHT = re.compile(r"\b(?:turn|make a)\s+(?:a\s+)?right\b", re.IGNORECASE)
_LANE_WORDS = re.compile(r"\b(lane|merge)\b", re.IGNORECASE)


def turn_tally(text: str) -> tuple[int, int]:
    """Counts (left_turns, right_turns) actually verbalized in text."""
    return len(_TURN_LEFT.findall(text)), len(_TURN_RIGHT.findall(text))


def extract_turn_sequence(text: str) -> list[str]:
    """
    Extracts the ORDERED sequence of turns ('L'/'R') as they actually
    appear in the text, by sorting all matches on their position in the
    string. This is what makes order comparable: "turn right, turn
    left, turn right" -> ['R', 'L', 'R'], in the order the maneuvers are
    described.
    """
    matches = [(m.start(), "L") for m in _TURN_LEFT.finditer(text)]
    matches += [(m.start(), "R") for m in _TURN_RIGHT.finditer(text)]
    matches.sort(key=lambda x: x[0])
    return [direction for _, direction in matches]


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Levenshtein edit distance (insertions/deletions/substitutions) between two sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def turn_sequence_agreement(pred_text: str, gt_text: str) -> float:
    """
    1.0 = identical ordered sequence of turns, 0.0 = maximally different.
    Uses edit distance between the ORDERED sequences (e.g. ['R','L','R']
    vs ['L','L','R']).
    """
    pred_seq = extract_turn_sequence(pred_text)
    gt_seq = extract_turn_sequence(gt_text)
    if not pred_seq and not gt_seq:
        return 1.0
    dist = _levenshtein(pred_seq, gt_seq)
    return 1.0 - dist / max(len(pred_seq), len(gt_seq), 1)


def turn_count_agreement(pred_text: str, gt_text: str) -> float:
    """
    Secondary point of comparison
    1.0 = identical left count AND identical right count. Scores left-
    agreement and right-agreement separately and averages them
    """
    pl, pr = turn_tally(pred_text)
    gl, gr = turn_tally(gt_text)
    left_agreement = 1.0 - abs(pl - gl) / max(pl, gl, 1)
    right_agreement = 1.0 - abs(pr - gr) / max(pr, gr, 1)
    return (left_agreement + right_agreement) / 2


def mentions_lane_position(text: str) -> bool:
    """Whether text mentions lane positioning/merging at all (any direction)."""
    return bool(_LANE_WORDS.search(text))


def score_one(pred: str, gt_texts: list[str]) -> dict:
    """
    Scores one sample. gt_texts may have multiple annotations (some
    pairs have 2) -- scores against each and keeps the BEST agreement
    for each metric
    """
    pred_seq = extract_turn_sequence(pred)
    best_seq_agreement = max(
        (turn_sequence_agreement(pred, gt) for gt in gt_texts), default=1.0
    )
    best_count_agreement = max(
        (turn_count_agreement(pred, gt) for gt in gt_texts), default=1.0
    )
    gt_any_lane = any(mentions_lane_position(gt) for gt in gt_texts)
    # For display: whichever ground truth the sequence metric matched best against
    best_gt_seq = max(
        (extract_turn_sequence(gt) for gt in gt_texts),
        key=lambda seq: -_levenshtein(pred_seq, seq),
        default=[],
    )

    return {
        "pred_sequence": ",".join(pred_seq) or "-",
        "gt_sequence": ",".join(best_gt_seq) or "-",
        "turn_sequence_agreement": round(best_seq_agreement, 3),
        "turn_count_agreement": round(best_count_agreement, 3),
        "pred_mentions_lane": mentions_lane_position(pred),
        "gt_mentions_lane": gt_any_lane,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a decomposition_results.json run.")
    parser.add_argument("--results", type=Path, default=None,
                         help="Override the results file. Defaults to RESULTS_FILE from settings.txt.")
    parser.add_argument("--out", type=Path, default=None,
                         help="Optional: also write a per-sample CSV to this path.")
    args = parser.parse_args()

    config = None
    if args.results is None:
        from config import load_config
        config = load_config()
    results_path = args.results if args.results is not None else config.results_file

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    results = json.loads(results_path.read_text())
    rows = []

    print(f"{'pair_id':<8} {'pred seq':<12} {'gt seq':<12} {'seq agree':<10} {'count agree':<12} {'pred lane?':<11} {'gt lane?':<9}")
    for r in results:
        scores = score_one(r["predicted_sub_instructions"], r["ground_truth_sub_instructions"])
        rows.append({"pair_id": r["pair_id"], **scores})
        pred_lane_str = "yes" if scores["pred_mentions_lane"] else "no"
        gt_lane_str = "yes" if scores["gt_mentions_lane"] else "no"
        print(f"{str(r['pair_id']):<8} {scores['pred_sequence']:<12} {scores['gt_sequence']:<12} "
              f"{scores['turn_sequence_agreement']:<10} {scores['turn_count_agreement']:<12} "
              f"{pred_lane_str:<11} {gt_lane_str:<9}")

    n = len(rows)
    if n == 0:
        print("No results to score.")
        return

    mean_seq_agreement = sum(row["turn_sequence_agreement"] for row in rows) / n
    mean_count_agreement = sum(row["turn_count_agreement"] for row in rows) / n
    pred_lane_count = sum(1 for row in rows if row["pred_mentions_lane"])
    gt_lane_count = sum(1 for row in rows if row["gt_mentions_lane"])

    print("=" * 70)
    print(f"Samples scored: {n}")
    print(f"Mean turn-SEQUENCE agreement (order-sensitive, primary metric): {mean_seq_agreement:.3f}")
    print(f"Mean turn-count agreement (order-agnostic, secondary):         {mean_count_agreement:.3f}")
    print(f"Predictions mentioning lane position: {pred_lane_count}/{n}")
    print(f"Ground truths mentioning lane position: {gt_lane_count}/{n}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote per-sample scores to {args.out}")


if __name__ == "__main__":
    main()