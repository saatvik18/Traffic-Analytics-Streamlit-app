"""
Day 2 logic: run YOLO + ByteTrack so every vehicle gets a persistent ID
across frames, and write an annotated video.
"""

import supervision as sv

from modules.utils import (
    open_video,
    create_writer,
    build_labels,
    make_browser_playable,
)


def run_tracking(model, video_path, output_path,
                 stride=1, max_frames=None, progress_callback=None):
    """
    Same arguments as run_detection; adds ByteTrack IDs to each box.
    """
    stride = max(1, int(stride))
    raw_path = output_path.replace(".mp4", "_raw.mp4")

    cap, width, height, fps, total_frames = open_video(video_path)
    if max_frames:
        total_frames = min(total_frames, int(max_frames))

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

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
            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)

            labels = build_labels(model, detections)

            annotated = frame.copy()
            annotated = box_annotator.annotate(scene=annotated, detections=detections)
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels
            )

            out.write(annotated)

            if progress_callback:
                progress_callback(min(frame_no / total_frames, 1.0))
    finally:
        cap.release()
        out.release()

    if progress_callback:
        progress_callback(1.0)

    return make_browser_playable(raw_path, output_path)
