"""
Day 3 logic: ROI polygon, vehicle trajectories, simple entry/exit
counting based on y-position, and a basic density label.
"""

import cv2
import numpy as np
import supervision as sv

from modules.utils import make_browser_playable

DEFAULT_ROI = np.array([
    [300, 250],
    [1620, 250],
    [1900, 1080],
    [50, 1080]
], dtype=np.int32)


def run_roi_analysis(model, video_path, output_path, roi=None, progress_callback=None):
    if roi is None:
        roi = DEFAULT_ROI

    raw_path = output_path.replace(".mp4", "_raw.mp4")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    vehicle_paths = {}
    entered = set()
    exited = set()

    out = cv2.VideoWriter(
        raw_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        result = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        cv2.polylines(frame, [roi], True, (255, 0, 0), 3)

        if detections.tracker_id is not None:
            for tracker_id, xyxy in zip(detections.tracker_id, detections.xyxy):
                x1, y1, x2, y2 = xyxy
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                vehicle_paths.setdefault(tracker_id, []).append((cx, cy))

        for path in vehicle_paths.values():
            for i in range(1, len(path)):
                cv2.line(frame, path[i - 1], path[i], (0, 255, 255), 2)

        for tracker_id, path in vehicle_paths.items():
            x, y = path[-1]
            if y < 300:
                entered.add(tracker_id)
            if y > 900:
                exited.add(tracker_id)

        vehicle_count = len(detections)
        if vehicle_count < 8:
            density = "LOW"
        elif vehicle_count < 18:
            density = "MEDIUM"
        else:
            density = "HIGH"

        labels = []
        if detections.tracker_id is not None:
            for tid, cid in zip(detections.tracker_id, detections.class_id):
                labels.append(f"{model.names[cid]} {tid}")

        frame = box_annotator.annotate(frame, detections)
        frame = label_annotator.annotate(frame, detections, labels)

        cv2.putText(frame, f"Vehicles : {vehicle_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Density : {density}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Entered : {len(entered)}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame, f"Exited : {len(exited)}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        out.write(frame)

        if progress_callback:
            progress_callback(min(frame_no / total_frames, 1.0))

    cap.release()
    out.release()
    return make_browser_playable(raw_path, output_path)
