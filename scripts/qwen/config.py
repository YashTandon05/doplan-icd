"""
================================================================================
CONTROL PANEL for the doPlan ICD pipeline.
================================================================================

How settings flow together:
    1. MODEL_PRESETS (below) -- edit this ONCE per machine to point
       "3B" and "7B" at wherever that model actually lives (a local folder
       after `hf download`, or a bare HF repo id to auto-download).
    2. settings.txt -- each person's file with their own
       CSV_PATH / VIDEO_PATH / MODEL_SIZE / etc.
       See settings.example.txt for a template with every option explained.
    3. Config (this file) -- reads settings.txt, fills in any values that
       were omitted with the defaults below, and validates everything
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------------------
# Default prompt text. Overridable per-line in settings.txt (see
# DECOMPOSITION_PROMPT / SCENE_DESCRIPTION_PROMPT in settings.example.txt)
# CAUTION if you override these in settings.txt:
#   - Keep the literal text "{instruction}" somewhere in
#     DECOMPOSITION_PROMPT
#   - settings.txt is one setting per line, so newlines in an override
#     must be written as the two characters \ and n (backslash-n), which
#     get converted back into real newlines when read. Don't press Enter
#     inside the value itself.
DEFAULT_DECOMPOSITION_PROMPT = (
    "Instruction: {instruction}\n"
    "Break this instruction down into a numbered list of atomic driving "
    "maneuvers (lane changes, turns, stops) that get the vehicle to its "
    "destination.\n"
    "Only reference lanes, turns, signals, and landmarks that are actually "
    "visible in the video -- do not invent details you cannot see.\n"
    "Stop the list once the vehicle has parked or stopped. Do not include "
    "any actions that happen after the vehicle stops (e.g. walking, "
    "talking to staff, waiting for a passenger).\n"
    "If the instruction has no clear arrival point "
    "(e.g. following another vehicle, with nowhere specific to stop), "
    "describe the next maneuvers you actually observe in the "
    "video -- do NOT repeat the same or similar maneuvers in a loop to "
    "fill space.\n"
    "Format: a plain numbered list, one short sentence per line. Do not "
    "use bold text, headers, or step-type labels (e.g. 'Lane Change:').\n"
    "Any earlier examples in this conversation are for a DIFFERENT trip "
    "and DIFFERENT video -- base this answer only on the video shown with "
    "THIS instruction. Do not reuse their wording."
)

DEFAULT_SCENE_DESCRIPTION_PROMPT = (
    "Describe only what is visible in this driving video: the road layout, "
    "number of lanes, lane markings, traffic signals, signs, and any "
    "landmarks you can identify. Do not mention turns, maneuvers, or a "
    "destination yet -- just describe the scene as it changes over the "
    "video, in the order you observe it."
)


def _find_default_settings_file() -> Path:
    """
    Searches upward from this file's own directory for settings.txt.
    """
    for directory in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        candidate = directory / "settings.txt"
        if candidate.exists():
            return candidate
    return SCRIPT_DIR / "settings.txt"


DEFAULT_SETTINGS_FILE = _find_default_settings_file()


# ------------------------------------------------------------------------
# EDIT THIS: where each model size actually lives on YOUR machine.
# ------------------------------------------------------------------------
# Left side ("3B" / "7B") is the shorthand you type in settings.txt as
# MODEL_SIZE. Right side is either:
#   - a bare Hugging Face repo id (auto-downloads the first time), or
#   - a local folder path (e.g. after `hf download ... --local-dir ./x`),
#     which is faster to load and doesn't need internet access.

# You may change the right side to a path on your computer
MODEL_PRESETS: dict[str, str] = {
    "3B": "Qwen/Qwen2.5-VL-3B-Instruct",
    "7B": "Qwen/Qwen2.5-VL-7B-Instruct",
    # Example once downloaded locally:
    # "7B": "C:/path/to/doplan-icd/scripts/qwen/qwen_model",
}


@dataclass(frozen=True)
class Config:
    """
    One validated, typed object holding every adjustable setting.

    `frozen=True` means a Config can't be accidentally mutated after
    creation -- if you need different settings, load a new Config rather
    than changing one in place.
    """

    # --- Required: no sensible default, must come from settings.txt ---
    csv_path: Path              # path to icd_pairs.csv
    video_paths: tuple[Path, ...]  # one or more root folders to search for per-sequence video subfolders --
                                    # comma-separated in settings.txt, e.g. VIDEO_PATH = "D:/vegas_8,D:/boston".
                                    # Each sample's folder is searched for in each root, in order given.
    fps: float                  # sampling rate to hand the model (not the source video's native fps)

    # --- Model selection ---
    model_size: str = "7B"                  # shorthand key into MODEL_PRESETS above
    model_name: str | None = None           # optional: bypass MODEL_PRESETS entirely with a direct path/repo id
    load_in_4bit: bool = True               # quantize to fit smaller GPUs

    # --- Prompt text (overridable in settings.txt -- see caution above DEFAULT_DECOMPOSITION_PROMPT) ---
    decomposition_prompt: str = DEFAULT_DECOMPOSITION_PROMPT
    scene_description_prompt: str = DEFAULT_SCENE_DESCRIPTION_PROMPT

    # --- Generation / memory tuning ---
    max_new_tokens: int = 256               # cap on how long the model's answer can be -- too low truncates
                                             # mid-sentence

    two_stage_grounding: bool = False       # off by default -- doubles inference time per sample (two generate()
                                             # calls: first "describe only what you see", then "decompose using
                                             # only that description"). Try this if predictions keep inventing
                                             # things (a ramp, an extra turn) that aren't actually in the clip

    use_gps_context: bool = False           # off by default. Injects GPS position data into the prompt -- one
                                             # reading per video frame the model sees, as north/east meters (see
                                             # gps_utils.py). No turns are pre-identified; the model works those
                                             # out itself from the raw position data plus the video.

    min_pixels: int = 64 * 28 * 28          # hard floor on per-frame resolution -- never goes below this even
                                             # if total_pixel_budget would suggest less (see effective_max_pixels)
    max_pixels: int = 256 * 28 * 28         # hard ceiling on per-frame resolution -- the ACTUAL value used is
                                             # auto-scaled down from this as max_frames grows; see effective_max_pixels
    max_frames: int = 64                    # cap on TOTAL frames sampled per video, regardless of clip length.
                                             # This is the main lever for how much temporal detail the model gets --
                                             # more frames = better chance of catching lane changes/turns, at the
                                             # cost of per-frame resolution (auto-traded off, see total_pixel_budget).
    total_pixel_budget: int = 16 * 256 * 28 * 28  # TOTAL pixels across all sampled frames combined. This is what
                                             # actually determines VRAM cost. effective_max_pixels below auto-derives
                                             # the per-frame value from (budget / max_frames), clamped to [min_pixels,
                                             # max_pixels] -- so raising MAX_FRAMES automatically lowers per-frame
                                             # resolution to compensate, and you never have to hand-balance the two
                                             # against each other to avoid an OOM.

    # --- Few-shot examples ---
    # TODO(future): once more ICD pairs are available, replace this wih train/test split
    example_pair_ids: tuple[str, ...] = ()  # default: zero-shot. Treat as an experimental opt-in for now --
                                             # worth revisiting once the train/test split above exists.

    # --- Networking / auth (see prompts.py's hub-download fix) ---
    hf_token: str | None = None
    download_timeout: int = 60

    # --- Output ---
    results_file: Path = Path("decomposition_results.json")  # shared by prompts.py (writes here by
                                                               # default) and review.py (reads from here by
                                                               # default)

    @property
    def resolved_model_name(self) -> str:
        """
        The actual path/repo id to load, after applying the MODEL_SIZE ->
        MODEL_PRESETS lookup. If MODEL_NAME was set explicitly in
        settings.txt, that always wins
        """
        if self.model_name:
            return self.model_name
        if self.model_size not in MODEL_PRESETS:
            raise ValueError(
                f"MODEL_SIZE '{self.model_size}' is not in MODEL_PRESETS "
                f"{list(MODEL_PRESETS)}. Add it to MODEL_PRESETS in config.py, "
                f"or set MODEL_NAME directly in settings.txt."
            )
        return MODEL_PRESETS[self.model_size]

    @property
    def effective_max_pixels(self) -> int:
        """
        The per-frame max_pixels actually passed to the model, auto-scaled
        down as max_frames grows so total (frames x pixels) VRAM cost stays
        roughly constant at total_pixel_budget. This is what makes it safe
        to raise MAX_FRAMES in settings.txt without separately re-tuning
        MAX_PIXELS by hand -- MAX_PIXELS becomes a ceiling this can't
        exceed, not a fixed value.

        Caveat: if max_frames is pushed high enough that the budget-implied
        per-frame value would fall below min_pixels, this clamps to
        min_pixels instead -- meaning total VRAM usage CAN exceed
        total_pixel_budget at extreme frame counts, since resolution can't
        usefully go below the floor.
        """
        budget_per_frame = self.total_pixel_budget // max(self.max_frames, 1)
        return max(self.min_pixels, min(self.max_pixels, budget_per_frame))

    def __post_init__(self) -> None:
        """Fail loudly and immediately on bad settings, before any model loading starts."""
        if self.fps <= 0:
            raise ValueError(f"FPS must be positive, got {self.fps}")
        if not self.video_paths:
            raise ValueError("VIDEO_PATH must have at least one path.")
        missing_roots = [p for p in self.video_paths if not p.exists()]
        if missing_roots:
            raise FileNotFoundError(f"VIDEO_PATH root(s) do not exist: {missing_roots}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV_PATH does not exist: {self.csv_path}")
        # Touch resolved_model_name now so a bad MODEL_SIZE fails at
        # startup, not 30 seconds into a model download.
        _ = self.resolved_model_name


# Settings that MUST be present in settings.txt
_REQUIRED_KEYS = ("CSV_PATH", "VIDEO_PATH", "FPS")

# Everything else is optional in settings.txt; if omitted, the Config
# dataclass default above is used. Maps the settings.txt spelling (left)
# to the Config field name (right).
_OPTIONAL_KEYS = {
    "MODEL_SIZE": "model_size",
    "MODEL_NAME": "model_name",
    "LOAD_IN_4BIT": "load_in_4bit",
    "DECOMPOSITION_PROMPT": "decomposition_prompt",
    "SCENE_DESCRIPTION_PROMPT": "scene_description_prompt",
    "TWO_STAGE_GROUNDING": "two_stage_grounding",
    "USE_GPS_CONTEXT": "use_gps_context",
    "MAX_NEW_TOKENS": "max_new_tokens",
    "MIN_PIXELS": "min_pixels",
    "MAX_PIXELS": "max_pixels",
    "MAX_FRAMES": "max_frames",
    "TOTAL_PIXEL_BUDGET": "total_pixel_budget",
    "FEW_SHOT_PAIR_IDS": "example_pair_ids",
    "HF_TOKEN": "hf_token",
    "DOWNLOAD_TIMEOUT": "download_timeout",
    "RESULTS_FILE": "results_file",
}

# Which optional fields need type conversion out of the raw string settings.txt gives us.
_INT_FIELDS = {"max_new_tokens", "min_pixels", "max_pixels", "max_frames", "total_pixel_budget", "download_timeout"}
_BOOL_FIELDS = {"load_in_4bit", "two_stage_grounding", "use_gps_context"}
_FLOAT_FIELDS: set[str] = set()  # none currently -- kept as a hook for the next float setting, if one's added


def _parse_settings_file(settings_file: Path) -> dict[str, str]:
    """Read settings.txt into a flat {KEY: value} dict of raw strings. No type conversion here."""

    if not settings_file.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_file}")

    raw: dict[str, str] = {}
    with open(settings_file, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue  # blank lines, comments, and malformed lines are silently skipped
            key, value = line.split("=", 1)
            raw[key.strip()] = value.strip().strip('"').strip("'")
    return raw


def load_config(settings_file: Path | None = None) -> Config:
    """
    Read settings.txt and return a fully validated Config.

    This is the ONE function every script in this pipeline calls to get
    its settings
    """

    settings_file = settings_file or DEFAULT_SETTINGS_FILE
    raw = _parse_settings_file(settings_file)

    missing = [key for key in _REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(f"Missing required setting(s) {missing} in {settings_file}")

    kwargs: dict = {
        "csv_path": Path(raw["CSV_PATH"]),
        # Comma-separated list of one or more roots, e.g.
        # "D:/vegas_8,D:/boston" -> (Path("D:/vegas_8"), Path("D:/boston")).
        # A single path with no comma still works exactly as before.
        "video_paths": tuple(Path(p.strip()) for p in raw["VIDEO_PATH"].split(",") if p.strip()),
        "fps": float(raw["FPS"]),
    }

    for raw_key, field_name in _OPTIONAL_KEYS.items():
        if raw_key not in raw:
            continue  # not present in settings.txt -> Config dataclass default is used
        value = raw[raw_key]
        if field_name in _INT_FIELDS:
            value = int(value)
        elif field_name in _FLOAT_FIELDS:
            value = float(value)
        elif field_name in _BOOL_FIELDS:
            value = value.strip().lower() in ("1", "true", "yes")
        elif field_name == "example_pair_ids":
            # Comma-separated list, e.g. "6,7" -> ("6", "7"). Empty
            # string ("") means zero-shot -- no examples.
            value = tuple(p.strip() for p in value.split(",") if p.strip())
        elif field_name in ("decomposition_prompt", "scene_description_prompt"):
            # settings.txt is one setting per physical line, so a
            # multi-line prompt override has to spell newlines out as
            # the two characters \ and n rather than an actual line
            # break -- unescape that back into real newlines here.
            value = value.replace("\\n", "\n")
        elif field_name == "results_file":
            value = Path(value)
        kwargs[field_name] = value

    # Anchor results_file to this tool's own folder (scripts/qwen)
    results_file = kwargs.get("results_file", Path("decomposition_results.json"))
    if not results_file.is_absolute():
        results_file = (SCRIPT_DIR / results_file).resolve()
    kwargs["results_file"] = results_file

    return Config(**kwargs)