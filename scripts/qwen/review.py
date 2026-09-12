"""
Manual review helper -- opens a sample's front-camera video in your
default player and prints its instruction, prediction, and ground truth
side by side, for debugging the model

Usage:
    python review.py           # no pair_id -> lists every available pair_id
    python review.py 9         # review + open the video for pair_id 9
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path

from config import load_config
from gps_utils import build_gps_context_text


def _is_headless() -> bool:
    """
    Detects a headless environment (no display) -- e.g. an HPC login or
    compute node.
    """
    if platform.system() in ("Windows", "Darwin"):
        return False
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def open_video(path: str) -> None:
    """
    Opens a video in the OS default player. On a headless machine (no
    display -- e.g. an HPC node), this just prints the path.
    """
    if _is_headless():
        print(f"  (no display detected -- skipping auto-open; video is at {path})")
        return

    system = platform.system()
    if system == "Windows":
        os.startfile(path)  # type: ignore[attr-defined]
    elif system == "Darwin":
        subprocess.run(["open", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review one decomposition result alongside its video.")
    parser.add_argument("pair_id", nargs="?", default=None,
                         help="pair_id to review. Omit to list every available pair_id instead.")
    parser.add_argument("--results", type=Path, default=None,
                         help="Override the results file. Defaults to RESULTS_FILE from settings.txt.")
    args = parser.parse_args()

    # config is always loaded (not just when --results is omitted) since
    # GPS context below needs config.max_frames to match what a real run
    # would use.
    config = load_config()
    results_path = args.results if args.results is not None else config.results_file

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run prompts.py first, or pass --results <path> if it's somewhere else.")
        return

    results = json.loads(results_path.read_text())

    if args.pair_id is None:
        print(f"Available pair_ids in {results_path}:")
        for r in results:
            print(f"  {r['pair_id']}: {r['long_instruction']}")
        print("\nRun `python review.py <pair_id>` (or `review <pair_id>`) to review one.")
        return

    match = next((r for r in results if str(r["pair_id"]) == str(args.pair_id)), None)
    if match is None:
        print(f"No result found for pair_id={args.pair_id} in {results_path}")
        return

    print("=" * 70)
    print(f"pair_id: {match['pair_id']}")
    print(f"Instruction: {match['long_instruction']}")
    print("-" * 70)
    print("PREDICTED:")
    print(match["predicted_sub_instructions"])
    print("-" * 70)
    print("GROUND TRUTH:")
    for gt in match["ground_truth_sub_instructions"]:
        print(f"  - {gt}")
    print("-" * 70)

    video_path = match.get("video_path")

    # Shows exactly what the model receives when USE_GPS_CONTEXT is on --
    # there's only one GPS mode (raw coordinates), so this always matches
    # production behavior.
    if video_path:
        gps_csv_path = Path(video_path).parent / "gps.csv"
        if gps_csv_path.exists():
            gps_text = build_gps_context_text(gps_csv_path, Path(video_path), max_frames=config.max_frames)
            print("GPS CONTEXT (what the model receives when USE_GPS_CONTEXT is on):")
            print(gps_text or "  (no GPS data available for this clip)")
        else:
            print(f"No gps.csv found at {gps_csv_path} -- can't show GPS context.")
    print("=" * 70)

    if not video_path:
        print("This result has no video_path (produced with an older version "
              "of prompts.py) -- re-run to get it in future results.")
    elif not Path(video_path).exists():
        print(f"video_path in the result no longer exists on disk: {video_path}")
    else:
        print(f"Opening front camera video: {video_path}")
        open_video(video_path)

        gps_map_path = Path(video_path).parent / "gps_map.mp4"
        if gps_map_path.exists():
            print(f"Opening GPS map video: {gps_map_path}")
            open_video(str(gps_map_path))


if __name__ == "__main__":
    main()