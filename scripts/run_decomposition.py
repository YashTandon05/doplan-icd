"""Run G: decompose each long instruction with Qwen2.5-VL and write results JSON."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.decompose.qwen_decomposer import Decomposer
from src.utils.config import load_config
from src.utils.icd_dataset import ICDDataset, identity_collate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_decomposition")
def _check_cuda_available() -> None:
    """
    Check if GPU is available, required.
    """
    if torch.cuda.is_available():
        return
    logger.error(
        "torch.cuda.is_available() is False -- no GPU detected. If you "
        "have an NVIDIA GPU, your torch install is almost certainly the "
        "CPU-only build. Fix with:\n"
        "  1) Check your driver's CUDA version:  nvidia-smi\n"
        "  2) pip uninstall torch torchvision torchaudio\n"
        "  3) pip install torch --index-url https://download.pytorch.org/whl/cu126\n"
        "     (swap cu126 for the build matching your driver, see pytorch.org)\n"
        "  4) Verify: python -c \"import torch; print(torch.cuda.is_available())\"\n"
        "On a shared HPC cluster: make sure you're running this from an "
        "allocated GPU job or interactive session, not a login node -- login "
        "nodes might have no GPU at all, which looks identical to a bad "
        "torch install from here.\n"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# 3. CLI -- config -> dataset -> Decomposer -> JSON
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL instruction decomposition.")
    parser.add_argument("--settings", type=Path, default=None,
                         help="Path to settings.txt (defaults to the one at the project root).")
    parser.add_argument("--batch-size", type=int, default=1,
                         help="Defaults to 1 -- safest for constrained VRAM.")
    parser.add_argument("--num-workers", type=int, default=0,
                         help="DataLoader workers.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run the first N samples")
    parser.add_argument("--out", type=Path, default=None,
                         help="Where to write results. Defaults to RESULTS_FILE from settings.txt.")
    return parser.parse_args()


def main() -> None:
    _check_cuda_available()

    args = parse_args()
    config = load_config(args.settings)

    logger.info("Model: %s (size=%s, 4bit=%s)",
                config.resolved_model_name, config.model_size, config.load_in_4bit)
    budget_str = config.total_pixel_budget if config.total_pixel_budget is not None else "unbounded"
    logger.info("max_frames=%d, effective_max_pixels=%d (total_pixel_budget=%s, ceiling=%d)",
                config.max_frames, config.effective_max_pixels, budget_str, config.max_pixels)

    dataset = ICDDataset(config)
    logger.info("Loaded %d valid sample(s).", len(dataset))

    if len(dataset) == 0:
        logger.error("No valid samples found -- check CSV_PATH/VIDEO_PATH in settings.txt.")
        return

    # Split off few-shot example pairs (taught to the model as style/
    # granularity demonstrations) from the actual evaluation set, so the
    # examples never leak into their own test results.
    example_ids = set(config.example_pair_ids)
    few_shot_samples = [s for s in dataset.samples if str(s.pair_id) in example_ids]
    dataset.samples = [s for s in dataset.samples if str(s.pair_id) not in example_ids]

    if example_ids and not few_shot_samples:
        logger.warning(
            "FEW_SHOT_PAIR_IDS=%s in settings.txt, but none of those pair_ids "
            "were found in the dataset -- running zero-shot instead.",
            config.example_pair_ids,
        )
    elif few_shot_samples:
        logger.info("Using %d pair(s) as few-shot examples: %s",
                     len(few_shot_samples), [s.pair_id for s in few_shot_samples])

    logger.info("%d sample(s) remaining for evaluation.", len(dataset))
    if len(dataset) == 0:
        logger.error("No samples left to evaluate after removing few-shot examples.")
        return

    if args.limit:
        dataset.samples = dataset.samples[: args.limit]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=identity_collate,
    )

    decomposer = Decomposer(config, few_shot_samples=few_shot_samples)

    all_results = []
    for batch_idx, batch in enumerate(loader):
        logger.info("Running batch %d (%d sample(s))...", batch_idx, len(batch))
        results = decomposer.run_batch(batch)
        all_results.extend(asdict(r) for r in results)

        for r in results:
            logger.info("pair_id=%s -> %s", r.pair_id, r.predicted_sub_instructions[:120])

        # Clear cached-but-unused CUDA memory between batches. Doesn't
        # reduce peak usage for a single sample, but helps prevent
        # fragmentation from accumulating across many samples run
        # back-to-back
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = args.out if args.out is not None else config.results_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    logger.info("Wrote %d result(s) to %s", len(all_results), out_path)


if __name__ == "__main__":
    main()
