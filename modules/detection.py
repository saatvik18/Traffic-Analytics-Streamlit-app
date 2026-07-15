"""
Day 1 logic: run YOLO detection on every frame of a video,
draw boxes + a vehicle count overlay, and write an annotated video.
"""

import cv2

from modules.utils import make_browser_playable

VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']


def run_detection(model, video_path, output_path, progress_callback=None):
    """
    model: a loaded ultralytics YOLO model
    video_path: path to input video
    output_path: path to write annotated, browser-playable output video
    progress_callback: optional function(fraction_done: float) -> None
    """
    raw_path = output_path.replace(".mp4", "_raw.mp4")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    out = cv2.VideoWriter(
        raw_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    frame_no = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated = results[0].plot()

        vehicle_count = 0
        for box in results[0].boxes:
            name = model.names[int(box.cls[0])]
            if name in VEHICLE_CLASSES:
                vehicle_count += 1

        cv2.putText(
            annotated,
            f"Vehicles: {vehicle_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        out.write(annotated)
        frame_no += 1

        if progress_callback:
            progress_callback(min(frame_no / total_frames, 1.0))

    cap.release()
    out.release()
    return make_browser_playable(raw_path, output_path)
