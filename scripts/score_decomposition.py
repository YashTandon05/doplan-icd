"""Score a decomposition results JSON with the text metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.eval.text_metrics import _get_embedder, score_one

def main() -> None:
    parser = argparse.ArgumentParser(description="Score a decomposition_results.json run.")
    parser.add_argument("--results", type=Path, default=None,
                        help="Override the results file. Defaults to RESULTS_FILE from settings.txt.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional: also write a per-sample CSV to this path.")
    args = parser.parse_args()

    config = None
    if args.results is None:
        from src.utils.config import load_config
        config = load_config()
    results_path = args.results if args.results is not None else config.results_file

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    results = json.loads(results_path.read_text())
    rows = []

    # Force the embedder to load NOW, before printing anything -- it's
    # lazy-loaded on first actual use otherwise (inside the loop below,
    # scoring row 1), which means a one-time model download/load prints
    # a wall of progress bars BETWEEN the header and the table rows,
    # pushing the header off-screen by the time the data appears.
    _get_embedder()

    print(
        f"{'pair_id':<8} {'pred seq':<12} {'gt seq':<12} {'seq agree':<10} {'count agree':<12} {'semantic':<9} {'pred lane?':<11} {'gt lane?':<9}")
    for r in results:
        scores = score_one(r["predicted_sub_instructions"], r["ground_truth_sub_instructions"])
        rows.append({"pair_id": r["pair_id"], **scores})
        pred_lane_str = "yes" if scores["pred_mentions_lane"] else "no"
        gt_lane_str = "yes" if scores["gt_mentions_lane"] else "no"
        semantic_str = f"{scores['semantic_similarity']}" if scores["semantic_similarity"] is not None else "n/a"
        print(f"{str(r['pair_id']):<8} {scores['pred_sequence']:<12} {scores['gt_sequence']:<12} "
              f"{scores['turn_sequence_agreement']:<10} {scores['turn_count_agreement']:<12} "
              f"{semantic_str:<9} {pred_lane_str:<11} {gt_lane_str:<9}")

    n = len(rows)
    if n == 0:
        print("No results to score.")
        return

    mean_seq_agreement = sum(row["turn_sequence_agreement"] for row in rows) / n
    mean_count_agreement = sum(row["turn_count_agreement"] for row in rows) / n
    pred_lane_count = sum(1 for row in rows if row["pred_mentions_lane"])
    gt_lane_count = sum(1 for row in rows if row["gt_mentions_lane"])
    semantic_values = [row["semantic_similarity"] for row in rows if row["semantic_similarity"] is not None]

    print("=" * 70)
    print(f"Samples scored: {n}")
    print(f"Mean turn-SEQUENCE agreement (order-sensitive, primary metric): {mean_seq_agreement:.3f}")
    print(f"Mean turn-count agreement (order-agnostic, secondary):         {mean_count_agreement:.3f}")
    if semantic_values:
        print(
            f"Mean semantic similarity (tertiary, meaning-based):            {sum(semantic_values) / len(semantic_values):.3f}")
    else:
        print("Semantic similarity: n/a -- install sentence-transformers to enable (pip install sentence-transformers)")
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
