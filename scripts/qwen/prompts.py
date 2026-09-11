"""
================================================================================
G-step decomposition: Qwen2.5-VL model wrapper + CLI run script.
================================================================================

WHAT THIS SCRIPT DOES
    Loads Qwen2.5-VL, feeds it each driving clip's front camera video plus
    its long-horizon instruction, and asks it to decompose that instruction
    into a numbered list of sub-instructions

HOW TO RUN
    python prompts.py                 # uses all defaults (batch-size 1, no limit)
    python prompts.py --limit N        # runs the first N samples
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from config import Config, load_config
from dataset import ICDDataset, ICDSample, identity_collate
from gps_utils import build_gps_context_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("decompose_and_run")

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
    )
    sys.exit(1)


def _ground_truth_as_numbered_steps(sub_instructions: list[str]) -> str:
    """
    Converts a ground-truth annotation (often one long multi-sentence
    string, not pre-split into steps) into the same numbered-list format
    we're asking the model to produce. Used to build few-shot examples
    that actually match the requested output shape.
    """
    full_text = " ".join(s.strip() for s in sub_instructions if s.strip())
    # Naive sentence split on ". " -- good enough for annotation text like
    # this (short declarative driving instructions), not meant to be a
    # general-purpose sentence tokenizer.
    sentences = [s.strip().rstrip(".") for s in full_text.split(". ") if s.strip()]
    return "\n".join(f"{i}. {s}." for i, s in enumerate(sentences, start=1))


# ---------------------------------------------------------------------------
# 1. Output record
# ---------------------------------------------------------------------------

@dataclass
class DecompositionResult:
    """One row of output: what the model predicted vs. the CSV ground truth."""

    pair_id: str
    long_instruction: str
    predicted_sub_instructions: str
    ground_truth_sub_instructions: list[str]
    video_path: str  # front-camera video path -- lets review.py open the actual clip alongside this result


# ---------------------------------------------------------------------------
# 2. Decomposer -- loads the model once, runs batched inference
# ---------------------------------------------------------------------------

class Decomposer:
    """
    Wraps Qwen2.5-VL so the (slow, ~10-60s) model/processor load happens
    exactly once in __init__, and run_batch() can then be called many
    times cheaply during the main loop.
    """

    def __init__(self, config: Config, few_shot_samples: list[ICDSample] | None = None):
        self.config = config
        self._configure_hub_env()
        self.model = self._load_model()
        self.processor = AutoProcessor.from_pretrained(config.resolved_model_name)

        # Decoder-only models need left-padding for correct batched generation.
        self.processor.tokenizer.padding_side = "left"

        # Pre-built once, reused for every sample -- see _build_few_shot_turns.
        self.few_shot_turns = self._build_few_shot_turns(few_shot_samples or [])

        # Warn (once, at startup) if MAX_FRAMES is high enough that
        # TOTAL_PIXEL_BUDGET / MAX_FRAMES would fall below MIN_PIXELS --
        # effective_max_pixels clamps to the floor in that case, which
        # means actual VRAM use CAN exceed the intended budget.
        budget_per_frame = self.config.total_pixel_budget // max(self.config.max_frames, 1)
        if budget_per_frame < self.config.min_pixels:
            logger.warning(
                "MAX_FRAMES=%d with TOTAL_PIXEL_BUDGET=%d implies only %d px/frame, "
                "below MIN_PIXELS=%d -- clamping to the floor, so actual VRAM use "
                "may exceed TOTAL_PIXEL_BUDGET. Lower MAX_FRAMES or MIN_PIXELS if "
                "you hit a CUDA out-of-memory error.",
                self.config.max_frames, self.config.total_pixel_budget,
                budget_per_frame, self.config.min_pixels,
            )

    def _build_few_shot_turns(self, examples: list[ICDSample]) -> list[dict]:
        """
        Show it practice conversations first: (instruction, ground-truth)
        pairs, teaching the model the expected output granularity and
        style before it sees the actual query.
                                TEXT ONLY
        This only demonstrates format/granularity, not visual grounding
        for the examples themselves; the real query's video is what the
        model actually looks at.
        """
        turns = []
        for i, example in enumerate(examples):
            user_text = self.config.decomposition_prompt.format(instruction=example.long_instruction)
            if i == 0:
                # Explicit framing on the FIRST example only -- makes clear
                # to the model that what follows is a format/granularity
                # demonstration, not part of the actual task, and that its
                # specific content (turns, lanes, landmarks) belongs to a
                # different trip entirely.
                user_text = (
                    "The following are EXAMPLES showing only the desired output "
                    "FORMAT and level of detail for a decomposition -- each is for "
                    "a different trip and different video than the one you will "
                    "actually analyze. Do not reuse their specific turns, lanes, "
                    "or landmarks.\n\n" + user_text
                )
            turns.append({
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            })
            turns.append({
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": _ground_truth_as_numbered_steps(example.sub_instructions),
                }],
            })
        return turns

    def _configure_hub_env(self) -> None:
        """
        Sets two environment variables that fix the
        "Fetching N files: 0%|" infinite-hang failure mode:
          - HF_HUB_DOWNLOAD_TIMEOUT forces a real timeout/exception instead
            of retrying forever on a flaky/slow/rate-limited connection.
          - HF_TOKEN (if you set one in settings.txt) avoids the more
            aggressive rate limits applied to anonymous downloads.
        No effect if you're loading a model that's already local.
        """
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(self.config.download_timeout))
        if self.config.hf_token:
            os.environ.setdefault("HF_TOKEN", self.config.hf_token)

    def _load_model(self) -> Qwen2_5_VLForConditionalGeneration:
        model_name = self.config.resolved_model_name  # applies MODEL_SIZE -> MODEL_PRESETS lookup
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        # flash_attention_2 is faster but requires the separate `flash-attn`
        # Only request it if it's actually importable;
        # otherwise fall back to `sdpa`, which ships with PyTorch itself
        if use_bf16:
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "sdpa"
        else:
            attn_impl = "sdpa"

        quantization_config = None
        if self.config.load_in_4bit:
            # 4-bit (NF4) quantization: compresses each weight from 16-bit
            # down to 4-bit, decompressing on the fly during compute.
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        # Logs which attention implementation is active during runtime
        logger.info("Loading %s (4bit=%s, attn_implementation=%s)...",
                    model_name, self.config.load_in_4bit, attn_impl)

        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
                attn_implementation=attn_impl,
                # device_map="auto" pre-estimates memory *before*
                # quantization shrinks the model, so it can wrongly decide
                # to offload some layers to CPU/disk -- which bitsandbytes
                # 4-bit then refuses outright (it doesn't support split
                # CPU/disk placement without extra opt-in flags). Force
                # everything onto GPU 0 once we know we're quantizing,
                # since the whole point of 4-bit is that it should fit.
                # Use the explicit "cuda:0" string, not a bare int 0 --
                # some transformers/accelerate versions mishandle int
                # device ids here and raise a spurious "no accelerator"
                # error that has nothing to do with your actual GPU.
                device_map={"": "cuda:0"} if self.config.load_in_4bit else "auto",
                quantization_config=quantization_config,
            )
        except Exception:
            logger.exception("Failed to load %s.", model_name)
            raise

        return model

    def _build_messages(self, sample: ICDSample) -> list[dict]:
        """
        Builds the Qwen chat-format message list for one sample: the
        front camera video plus the decomposition prompt. This is the
         place to edit if you want to add more camera streams, GPS/map
        context, or change how the prompt references the instruction.

        Uses `nframes` (a hard cap on total sampled frames) rather than
        `fps` for video sampling -- `fps` alone scales frame count with
        clip length, which is unpredictable across clips of different
        lengths. `max_pixels` here is `effective_max_pixels`, auto-scaled
        down as `max_frames` grows
        """
        prompt_text = self.config.decomposition_prompt.format(instruction=sample.long_instruction)

        if self.config.use_gps_context:
            # max_frames passed through so the GPS coordinate count always
            # matches the number of video frames the model is shown -- see
            # gps_utils.py for how the two are kept in sync.
            gps_text = build_gps_context_text(sample.gps, sample.front_video, max_frames=self.config.max_frames)
            if gps_text:
                # Prepended so it reads before the instruction -- gives
                # the model authoritative turn/stop timing before it even
                # sees what it's being asked to decompose.
                prompt_text = gps_text + "\n\n" + prompt_text

        return [
            *self.few_shot_turns,
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(sample.front_video),
                        "min_pixels": self.config.min_pixels,
                        "max_pixels": self.config.effective_max_pixels,
                        "nframes": self.config.max_frames,
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            },
        ]

    @torch.no_grad()  # inference only -- no need to track gradients, saves VRAM
    def _generate_batch(self, all_messages: list[list[dict]]) -> list[str]:
        """
        takes one Qwen chat-format message list per sample, returns one
        decoded response string per sample.
        """
        texts = [
            self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in all_messages
        ]

        # process_vision_info decodes each video into frame tensors ready
        # for the processor.
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                all_messages, return_video_kwargs=True
            )
        except TypeError:
            image_inputs, video_inputs = process_vision_info(all_messages)
            video_kwargs = {}

        # Some transformers processor versions strictly validate `fps` as
        # a single scalar, but qwen-vl-utils can return a per-video LIST
        # in video_kwargs when batching. Since sampling is now driven by
        # `nframes` (fixed frame count) rather than a fixed fps, the
        # *actual* fps differs per clip -- extracts the actual fps
        video_fps = video_kwargs.pop("fps", None)
        if isinstance(video_fps, list):
            video_fps = video_fps[0] if video_fps else self.config.fps
        elif video_fps is None:
            video_fps = self.config.fps

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            fps=video_fps,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.model.device)

        try:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
            )
        except torch.cuda.OutOfMemoryError:
            logger.error(
                "CUDA out of memory during generation. This is almost always "
                "fixed by lowering settings in settings.txt, in order of impact:\n"
                "  1) MAX_FRAMES (also lowers effective per-frame pixels automatically)\n"
                "  2) TOTAL_PIXEL_BUDGET (directly caps total VRAM cost)\n"
                "  3) --batch-size on the command line (already defaults to 1)\n"
                "Current settings: max_frames=%d, effective_max_pixels=%d (budget=%d)",
                self.config.max_frames, self.config.effective_max_pixels, self.config.total_pixel_budget,
            )
            raise

        # generate() returns the prompt tokens AND the new tokens
        # concatenated together -- trim off the prompt so we only decode
        # what the model actually generated.
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def run_batch(self, batch: list[ICDSample]) -> list[DecompositionResult]:
        """Runs one batch of samples through the model. Dispatches to the
        two-stage grounding path per-sample if TWO_STAGE_GROUNDING is on.
        Otherwise, does normal single-stage batched generation."""
        if self.config.two_stage_grounding:
            return [self._run_two_stage(sample) for sample in batch]

        all_messages = [self._build_messages(sample) for sample in batch]
        decoded = self._generate_batch(all_messages)

        return [
            DecompositionResult(
                pair_id=sample.pair_id,
                long_instruction=sample.long_instruction,
                predicted_sub_instructions=text,
                ground_truth_sub_instructions=sample.sub_instructions,
                video_path=str(sample.front_video),
            )
            for sample, text in zip(batch, decoded)
        ]

    def _build_scene_description_messages(self, sample: ICDSample) -> list[dict]:
        """Stage 1 of two-stage grounding: video + SCENE_DESCRIPTION_PROMPT only."""
        return [
            *self.few_shot_turns,
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(sample.front_video),
                        "min_pixels": self.config.min_pixels,
                        "max_pixels": self.config.effective_max_pixels,
                        "nframes": self.config.max_frames,
                    },
                    {"type": "text", "text": self.config.scene_description_prompt},
                ],
            },
        ]

    def _run_two_stage(self, sample: ICDSample) -> DecompositionResult:
        """
        Two-pass grounding: first ask the model to describe only what it
        observes (no maneuvers, no destination), then ask it to decompose
        the instruction using ONLY that description. This removes the
        shortcut of jumping straight to a plausible-sounding answer
        without ever actually describing the video

        Costs roughly 2x inference time per sample (two generate() calls
        instead of one) since the video has to be re-encoded for the
        second call too
        """
        describe_messages = self._build_scene_description_messages(sample)
        description = self._generate_batch([describe_messages])[0]

        decompose_text = (
            self.config.decomposition_prompt.format(instruction=sample.long_instruction)
            + "\nBase this only on what you described above -- do not add "
              "any new detail (turns, ramps, lanes, landmarks) you didn't "
              "already mention in that description."
        )
        if self.config.use_gps_context:
            gps_text = build_gps_context_text(sample.gps, sample.front_video, max_frames=self.config.max_frames)
            if gps_text:
                decompose_text = gps_text + "\n\n" + decompose_text

        decompose_messages = [
            *describe_messages,
            {"role": "assistant", "content": [{"type": "text", "text": description}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": decompose_text}],
            },
        ]
        decomposition = self._generate_batch([decompose_messages])[0]

        return DecompositionResult(
            pair_id=sample.pair_id,
            long_instruction=sample.long_instruction,
            predicted_sub_instructions=decomposition,
            ground_truth_sub_instructions=sample.sub_instructions,
            video_path=str(sample.front_video),
        )


# ---------------------------------------------------------------------------
# 3. CLI -- config -> dataset -> Decomposer -> JSON
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL instruction decomposition.")
    parser.add_argument("--settings", type=Path, default=None,
                         help="Path to settings.txt (defaults to the one next to config.py).")
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
    logger.info("max_frames=%d, effective_max_pixels=%d (auto-scaled from total_pixel_budget=%d, ceiling=%d)",
                config.max_frames, config.effective_max_pixels, config.total_pixel_budget, config.max_pixels)

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
    out_path.write_text(json.dumps(all_results, indent=2))
    logger.info("Wrote %d result(s) to %s", len(all_results), out_path)


if __name__ == "__main__":
    main()