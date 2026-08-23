"""
Shared helpers used by every pipeline step.

Two jobs live here:

1. Video I/O that fails loudly instead of silently producing a 0-byte file.
2. Re-encoding OpenCV's mp4v output to H.264 (yuv420p) so it previews
   inline in the browser, using the bundled ffmpeg binary from
   imageio-ffmpeg (no separate ffmpeg install needed on Windows).
"""

import os
import subprocess

import cv2
import numpy as np
import imageio_ffmpeg

# ROI corners expressed as fractions of the frame, so the same polygon
# works on a 1920x1080 clip and on a 640x480 phone video.
ROI_FRACTIONS = [
    (0.156, 0.231),
    (0.844, 0.231),
    (0.990, 1.000),
    (0.026, 1.000),
]

# Entry / exit trip lines, also as fractions of frame height.
ENTRY_LINE_FRAC = 0.28
EXIT_LINE_FRAC = 0.83

# How many trajectory points to keep per vehicle. Drawing every point ever
# seen makes long videos slow down quadratically, so the trail is capped.
MAX_TRAIL_POINTS = 40


class VideoError(RuntimeError):
    """Raised when a video cannot be opened or written."""


def open_video(video_path):
    """
    Open a video and return (cap, width, height, fps, total_frames).

    Raises VideoError with a readable message instead of handing back a
    dead VideoCapture whose properties are all zero.
    """
    if not os.path.exists(video_path):
        raise VideoError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoError(
            f"OpenCV could not open '{video_path}'. "
            "The file may be corrupt or use an unsupported codec."
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise VideoError(f"Could not read frame size from '{video_path}'.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps != fps:  # also catches NaN
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1

    return cap, width, height, fps, total_frames


def create_writer(path, fps, width, height):
    """Create an mp4v VideoWriter, making the parent folder if needed."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise VideoError(f"Could not open a video writer for '{path}'.")
    return writer


def scaled_roi(width, height):
    """Return the default ROI polygon scaled to this frame size."""
    return np.array(
        [[int(fx * width), int(fy * height)] for fx, fy in ROI_FRACTIONS],
        dtype=np.int32,
    )


def entry_exit_lines(height):
    """Return (entry_y, exit_y) pixel positions for this frame height."""
    return int(ENTRY_LINE_FRAC * height), int(EXIT_LINE_FRAC * height)


def lane_boundaries(width, lanes=3):
    """
    Return the vertical x positions that split the frame into `lanes`
    equal columns. For 3 lanes on a 1920px frame this gives [640, 1280].
    """
    return [int(width * i / lanes) for i in range(1, lanes)]


def density_label(vehicle_count):
    """Bucket a per-frame vehicle count into LOW / MEDIUM / HIGH."""
    if vehicle_count < 8:
        return "LOW"
    if vehicle_count < 18:
        return "MEDIUM"
    return "HIGH"


SIGNAL_BY_DENSITY = {
    "LOW": "GREEN (20 sec)",
    "MEDIUM": "GREEN (40 sec)",
    "HIGH": "GREEN (60 sec)",
}


def build_labels(model, detections):
    """
    Build "car ID:3" style labels, tolerating detections that carry no
    tracker_id or no class_id (which happens on empty frames).
    """
    class_ids = getattr(detections, "class_id", None)
    tracker_ids = getattr(detections, "tracker_id", None)

    if class_ids is None:
        return []

    labels = []
    for i, class_id in enumerate(class_ids):
        name = model.names[int(class_id)]
        if tracker_ids is not None and i < len(tracker_ids) and tracker_ids[i] is not None:
            labels.append(f"{name} ID:{int(tracker_ids[i])}")
        else:
            labels.append(name)
    return labels


def draw_overlay(frame, lines, origin=(20, 40), spacing=40, scale=0.8):
    """Draw a stack of (text, bgr_color) rows in the top-left corner."""
    x, y0 = origin
    for i, (text, color) in enumerate(lines):
        cv2.putText(
            frame, text, (x, y0 + i * spacing),
            cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA
        )
    return frame


def draw_trails(frame, vehicle_paths, color=(0, 255, 255)):
    """Draw the recent trajectory of every tracked vehicle."""
    for path in vehicle_paths.values():
        for i in range(1, len(path)):
            cv2.line(frame, path[i - 1], path[i], color, 2)
    return frame


def make_browser_playable(raw_path, final_path):
    """
    raw_path:   mp4 written by cv2.VideoWriter (mp4v fourcc)
    final_path: where to write the browser-friendly H.264 version

    Falls back to simply renaming the raw file if ffmpeg is unavailable
    or fails, so a run never ends with no output at all.
    Deletes raw_path once conversion succeeds.
    """
    if not os.path.exists(raw_path):
        raise VideoError(f"Expected intermediate video '{raw_path}' was not created.")

    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y",
            "-i", raw_path,
            "-vcodec", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            final_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=1800)
        converted = result.returncode == 0 and os.path.exists(final_path)
    except (OSError, subprocess.SubprocessError):
        converted = False

    if not converted:
        # Keep the raw mp4v file rather than losing the output entirely.
        if raw_path != final_path:
            os.replace(raw_path, final_path)
        return final_path

    if raw_path != final_path and os.path.exists(raw_path):
        os.remove(raw_path)

    return final_path
