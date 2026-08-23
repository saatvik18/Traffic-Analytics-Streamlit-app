"""
Day 1 logic: run YOLO detection on every frame of a video,
draw boxes + a vehicle count overlay, and write an annotated video.
"""

from modules.utils import (
    open_video,
    create_writer,
    draw_overlay,
    make_browser_playable,
)

VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle")


def run_detection(model, video_path, output_path,
                  stride=1, max_frames=None, progress_callback=None):
    """
    model:             a loaded ultralytics YOLO model
    video_path:        path to the input video
    output_path:       where to write the annotated, browser-playable video
    stride:            process every Nth frame (1 = every frame)
    max_frames:        stop after this many source frames (None = whole video)
    progress_callback: optional function(fraction_done: float) -> None
    """
    stride = max(1, int(stride))
    raw_path = output_path.replace(".mp4", "_raw.mp4")

    cap, width, height, fps, total_frames = open_video(video_path)
    if max_frames:
        total_frames = min(total_frames, int(max_frames))

    # Writing at fps/stride keeps the output the same duration as the source.
    out = create_writer(raw_path, fps / stride, width, height)

    frame_no = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_no += 1
            if max_frames and frame_no > max_frames:
                break
            if (frame_no - 1) % stride != 0:
                continue

            result = model(frame, verbose=False)[0]
            annotated = result.plot()

            vehicle_count = 0
            for box in result.boxes:
                if model.names[int(box.cls[0])] in VEHICLE_CLASSES:
                    vehicle_count += 1

            draw_overlay(annotated, [(f"Vehicles : {vehicle_count}", (0, 255, 0))], scale=1.0)
            out.write(annotated)

            if progress_callback:
                progress_callback(min(frame_no / total_frames, 1.0))
    finally:
        cap.release()
        out.release()

    if progress_callback:
        progress_callback(1.0)

    return make_browser_playable(raw_path, output_path)
