"""
Shared helper: OpenCV's mp4v-encoded videos often don't preview inline
in browsers (Chrome/Streamlit). This re-encodes to H.264 (yuv420p),
which plays natively in every browser, using a bundled ffmpeg binary
(no separate ffmpeg install needed on Windows).
"""

import os
import subprocess
import imageio_ffmpeg


def make_browser_playable(raw_path, final_path):
    """
    raw_path: mp4 written by cv2.VideoWriter (mp4v fourcc)
    final_path: where to write the browser-friendly H.264 version
    Deletes raw_path once conversion succeeds.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y",
        "-i", raw_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        final_path
    ]
    result = subprocess.run(cmd, capture_output=True)

    if result.returncode != 0 or not os.path.exists(final_path):
        # Fall back to the raw file rather than losing the output entirely
        if os.path.exists(raw_path) and raw_path != final_path:
            os.replace(raw_path, final_path)
        return final_path

    if os.path.exists(raw_path) and raw_path != final_path:
        os.remove(raw_path)

    return final_path
