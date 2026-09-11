"""
GPS position sampling for ICD samples.

Feeds the model raw position data -- one reading per video frame it's
shown, converted to local relative meters.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GpsPoint:
    frame: int           # video frame index this GPS row corresponds to
    timestamp_s: float    # elapsed VIDEO playback time (frame / video_fps)
    lat: float
    lon: float


def load_gps_trace(path: Path, video_fps: float) -> list[GpsPoint]:
    """
    Reads a gps.csv file into a list of GPS points, computing each
    point's elapsed time as frame_number / video_fps -- see GpsPoint's
    docstring for why the CSV's own timestamp_us column can't be
    trusted for this. video_fps is required, not optional: there's no
    reliable fallback without it.
    """
    points: list[GpsPoint] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            points.append(GpsPoint(
                frame=frame,
                timestamp_s=frame / video_fps,
                lat=float(row["lat"]),
                lon=float(row["lon"]),
            ))
    return points


def _probe_video(video_path: Path) -> tuple[int, float] | None:
    """
    Reads a video's total frame count and fps via decord (already
    installed as a qwen-vl-utils dependency). Returns None if the video
    can't be read for any reason.
    """
    try:
        import decord
        vr = decord.VideoReader(str(video_path))
        return len(vr), vr.get_avg_fps()
    except Exception:
        return None


def _to_local_meters(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    """
    Converts a lat/lon point to (north_meters, east_meters) relative to
    a reference point (lat0, lon0)"""

    METERS_PER_DEG_LAT = 111_320.0
    north_m = (lat - lat0) * METERS_PER_DEG_LAT
    east_m = (lon - lon0) * METERS_PER_DEG_LAT * math.cos(math.radians(lat0))
    return north_m, east_m


def _select_frame_aligned_points(points: list[GpsPoint], sample_count: int, total_frames: int) -> list[GpsPoint]:
    """
    Picks exactly sample_count points, evenly spaced across
    [0, total_frames-1] -- the same sampling Qwen's own nframes-based
    video sampling uses, so the GPS coordinates handed to the model are
    synchronized with the same frames it's actually shown. For each
    target frame index, picks whichever GPS point's own frame number is
    closest.
    """
    if not points or total_frames <= 0 or sample_count <= 0:
        return []

    target_frame_indices = [
        round(i * (total_frames - 1) / max(sample_count - 1, 1))
        for i in range(sample_count)
    ]

    return [min(points, key=lambda p: abs(p.frame - target)) for target in target_frame_indices]


def build_gps_context_text(gps_csv_path: Path, video_path: Path, max_frames: int) -> str | None:
    """
    Full pipeline: from the video extract the real frame count/fps, load
    gps.csv using that fps for reliable elapsed times, clip to the
    video's actual frame range, then produce a plain list of
    (elapsed_time, north_m, east_m) coordinates -- one per video frame
    the model is shown, relative to the clip's first sampled point.

    If MAX_FRAMES is higher than the actual video frame count or the
    number of available GPS rows, the sample count is capped at
    whichever is smallest

    Returns None if the video/CSV can't be read, or there's no GPS data
    to sample from
    """
    try:
        video_info = _probe_video(video_path)
        if video_info is None:
            return None
        total_frames, video_fps = video_info

        points = load_gps_trace(gps_csv_path, video_fps)
        points = [p for p in points if p.frame <= total_frames]
        if not points:
            return None

        sample_count = min(max_frames, total_frames, len(points))
        sampled = _select_frame_aligned_points(points, sample_count, total_frames)
        if not sampled:
            return None

        lat0, lon0 = sampled[0].lat, sampled[0].lon
        lines = []
        for p in sampled:
            north_m, east_m = _to_local_meters(lat0, lon0, p.lat, p.lon)
            mm, ss = divmod(int(p.timestamp_s), 60)
            lines.append(f"  {mm}:{ss:02d} -- north={north_m:+.1f}m, east={east_m:+.1f}m")
    except Exception:
        return None

    return (
        "GPS position, one reading per video frame shown to you above, "
        "given as north/east meters relative to the first frame -- NOT "
        "pre-identified turns. A meaningful, sustained change in the "
        "direction of travel across these readings usually indicates a "
        "turn; correlate this with what you actually see in the video "
        "to determine timing and direction yourself:\n" + "\n".join(lines)
    )