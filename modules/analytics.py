"""
Day 4 logic: ROI + lane counting + rough speed/waiting-time signals +
density-based signal recommendation, writing an annotated video and a
frame-by-frame CSV log.
"""

import cv2
import numpy as np
import pandas as pd
import supervision as sv

from modules.utils import make_browser_playable

DEFAULT_ROI = np.array([
    [300, 250],
    [1620, 250],
    [1900, 1080],
    [50, 1080]
], dtype=np.int32)


def run_full_analytics(model, video_path, output_video_path, output_csv_path,
                        roi=None, lane1=640, lane2=1280, progress_callback=None):
    if roi is None:
        roi = DEFAULT_ROI

    raw_video_path = output_video_path.replace(".mp4", "_raw.mp4")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    out = cv2.VideoWriter(
        raw_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    vehicle_paths = {}
    entry_frames = {}
    entered = set()
    exited = set()
    csv_data = []

    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            continue

        result = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        cv2.polylines(frame, [roi], True, (255, 0, 0), 3)
        cv2.line(frame, (lane1, 0), (lane1, height), (0, 255, 0), 2)
        cv2.line(frame, (lane2, 0), (lane2, height), (0, 255, 0), 2)

        vehicle_count = len(detections)

        lane_count = [0, 0, 0]
        for xyxy in detections.xyxy:
            x1, y1, x2, y2 = xyxy
            cx = int((x1 + x2) / 2)
            if cx < lane1:
                lane_count[0] += 1
            elif cx < lane2:
                lane_count[1] += 1
            else:
                lane_count[2] += 1

        if detections.tracker_id is not None:
            for tracker_id, xyxy in zip(detections.tracker_id, detections.xyxy):
                x1, y1, x2, y2 = xyxy
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                vehicle_paths.setdefault(tracker_id, []).append((cx, cy))
                if tracker_id not in entry_frames:
                    entry_frames[tracker_id] = frame_no

        for path in vehicle_paths.values():
            for i in range(1, len(path)):
                cv2.line(frame, path[i - 1], path[i], (0, 255, 255), 2)

        for tracker_id, path in vehicle_paths.items():
            if not path:
                continue
            x, y = path[-1]
            if y < 300:
                entered.add(tracker_id)
            if y > 900:
                exited.add(tracker_id)

        if vehicle_count < 8:
            density = "LOW"
        elif vehicle_count < 18:
            density = "MEDIUM"
        else:
            density = "HIGH"

        signal = {
            "LOW": "GREEN (20 sec)",
            "MEDIUM": "GREEN (40 sec)",
            "HIGH": "GREEN (60 sec)"
        }[density]

        labels = []
        if detections.tracker_id is not None:
            for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
                if tracker_id is None:
                    labels.append(model.names[class_id])
                else:
                    labels.append(f"{model.names[class_id]} ID:{tracker_id}")

        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        overlay = [
            (f"Vehicles : {vehicle_count}", (0, 255, 0)),
            (f"Lane1 : {lane_count[0]}", (255, 255, 0)),
            (f"Lane2 : {lane_count[1]}", (255, 255, 0)),
            (f"Lane3 : {lane_count[2]}", (255, 255, 0)),
            (f"Density : {density}", (0, 255, 255)),
            (f"Signal : {signal}", (255, 0, 255)),
            (f"Entered : {len(entered)}", (255, 255, 255)),
            (f"Exited : {len(exited)}", (255, 255, 255)),
        ]
        for i, (text, color) in enumerate(overlay):
            cv2.putText(frame, text, (20, 40 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        csv_data.append({
            "Frame": frame_no,
            "Vehicles": vehicle_count,
            "Lane1": lane_count[0],
            "Lane2": lane_count[1],
            "Lane3": lane_count[2],
            "Entered": len(entered),
            "Exited": len(exited),
            "Density": density,
            "Signal": signal
        })

        frame_no += 1
        out.write(frame)

        if progress_callback:
            progress_callback(min(frame_no / total_frames, 1.0))

    cap.release()
    out.release()

    final_video_path = make_browser_playable(raw_video_path, output_video_path)

    df = pd.DataFrame(csv_data)
    df.to_csv(output_csv_path, index=False)
    return final_video_path, output_csv_path, df
