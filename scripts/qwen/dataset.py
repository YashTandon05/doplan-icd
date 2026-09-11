"""
PyTorch dataset, currently used for G and F models but can be used for evaluation.
Does NOT decode video frames or run the Qwen processor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from config import Config

logger = logging.getLogger(__name__)

CAMERA_STREAMS = ("F0", "L0", "L1", "L2", "R0", "R1", "R2", "B0")


@dataclass
class ICDSample:
    """One decomposition sample: a driving clip + its ground-truth instructions."""

    pair_id: str
    folder: str
    sequence_dir: Path
    videos: dict[str, Path]
    gps: Path
    maps: dict[str, Path]
    long_instruction: str
    sub_instructions: list[str] = field(default_factory=list)
    fps: float = 10.0

    @property
    def front_video(self) -> Path:
        return self.videos["F0"]


class ICDDataset(Dataset):
    """
    Maps pair_id -> full sample (video paths, GPS, maps, ground truth).

    Usage:
        cfg = load_config()
        dataset = ICDDataset(cfg)
        loader = DataLoader(dataset, batch_size=4, collate_fn=identity_collate)
    """

    def __init__(self, config: Config, skip_invalid: bool = True):
        self.config = config
        self.skip_invalid = skip_invalid
        self.samples: list[ICDSample] = self._build_index()


    def _read_instruction_pairs(self) -> list[dict]:
        """
        One CSV row = one instruction (either the long-horizon "L" row or
        one "sub" row). Groups rows by pair_id so each sample ends up with
        its single long instruction plus the ordered list of ground-truth
        sub-instructions it should decompose into.
        """
        df = pd.read_csv(self.config.csv_path)
        pairs = []

        for pair_id, group in df.groupby("pair_id"):
            long_rows = group[group["role"] == "L"]
            if long_rows.empty:
                logger.warning("pair_id %s has no long instruction, skipping.", pair_id)
                continue

            sub_rows = group[group["role"] == "sub"].sort_values("sub_order")

            pairs.append({
                "pair_id": pair_id,
                "folder": group["folder"].iloc[0],
                "long_instruction": long_rows["label"].iloc[0],
                "sub_instructions": sub_rows["label"].tolist(),
            })

        return pairs

    def _build_index(self) -> list[ICDSample]:
        samples = []
        for pair in self._read_instruction_pairs():
            sequence_dir = self._resolve_sequence_dir(pair["folder"])

            videos = {cam: sequence_dir / f"{cam}.mp4" for cam in CAMERA_STREAMS}
            videos["gps_map"] = sequence_dir / "gps_map.mp4"

            sample = ICDSample(
                pair_id=pair["pair_id"],
                folder=pair["folder"],
                sequence_dir=sequence_dir,
                videos=videos,
                gps=sequence_dir / "gps.csv",
                maps={
                    "background": sequence_dir / "gps_background_map.png",
                    "base": sequence_dir / "gps_background_map_base.png",
                },
                long_instruction=pair["long_instruction"],
                sub_instructions=pair["sub_instructions"],
                fps=self.config.fps,
            )

            missing = self._missing_files(sample)
            if missing:
                logger.warning("pair_id %s missing %d file(s): %s",
                               sample.pair_id, len(missing), missing[:3])
                if self.skip_invalid:
                    continue

            samples.append(sample)

        logger.info("Indexed %d valid sample(s) out of the CSV.", len(samples))
        return samples

    def _resolve_sequence_dir(self, folder: str) -> Path:
        """
        Searches every configured VIDEO_PATH root for this sample's folder
        lets icd_pairs.csv reference clips spread across multiple downloaded
        roots (e.g. separate Las Vegas and Boston folders)
        """
        for root in self.config.video_paths:
            candidate = root / folder
            if candidate.exists():
                return candidate
        return self.config.video_paths[0] / folder

    @staticmethod
    def _missing_files(sample: ICDSample) -> list[str]:
        """
        Only checks the files this pipeline actually reads: F0.mp4 (fed
        to the model) and gps.csv (used when USE_GPS_CONTEXT is on).
        Does NOT require the other 7 camera angles or gps map
        """
        required = {"F0": sample.videos["F0"], "gps": sample.gps}
        return [f"{name}: {path}" for name, path in required.items() if not path.exists()]

    # -- torch Dataset interface -------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ICDSample:
        return self.samples[index]


def identity_collate(batch: list[ICDSample]) -> list[ICDSample]:
    """
    Pass-through collate_fn.

    Videos are variable-length and get tokenized by Qwen's own processor
    (which needs raw paths, not tensors), so we don't stack anything here
    """
    return batch