"""
Day 4 logic: ROI + lane counting + entry/exit + density-based signal
recommendation, writing an annotated video and a frame-by-frame CSV log.

Like Day 3, every geometric constant (ROI, lane dividers, trip lines) is
derived from the real frame size instead of being hard-coded for 1080p.
"""

from collections import deque

import cv2
import pandas as pd
import supervision as sv

from modules.utils import (
    MAX_TRAIL_POINTS,
    SIGNAL_BY_DENSITY,
    open_video,
    create_writer,
    scaled_roi,
    entry_exit_lines,
    lane_boundaries,
    density_label,
    build_labels,
    draw_overlay,
    draw_trails,
    make_browser_playable,
)

CSV_COLUMNS = [
    "Frame", "Vehicles", "Lane1", "Lane2", "Lane3",
    "Entered", "Exited", "Density", "Signal",
]


def run_full_analytics(model, video_path, output_video_path, output_csv_path,
                       roi=None, lanes=None, stride=1, max_frames=None,
                       progress_callback=None):
    """
    lanes: optional list of x positions splitting the frame into lanes.
           Defaults to two dividers at 1/3 and 2/3 of the frame width.

    Returns (video_path, csv_path, dataframe).
    """
    stride = max(1, int(stride))
    raw_video_path = output_video_path.replace(".mp4", "_raw.mp4")

    cap, width, height, fps, total_frames = open_video(video_path)
    if max_frames:
        total_frames = min(total_frames, int(max_frames))

    if roi is None:
        roi = scaled_roi(width, height)
    if lanes is None:
        lanes = lane_boundaries(width, lanes=3)
    lane1, lane2 = lanes[0], lanes[1]
    entry_y, exit_y = entry_exit_lines(height)

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    out = create_writer(raw_video_path, fps / stride, width, height)

    vehicle_paths = {}
    entered = set()
    exited = set()
    csv_data = []

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

            cv2.polylines(frame, [roi], True, (255, 0, 0), 3)
            cv2.line(frame, (lane1, 0), (lane1, height), (0, 255, 0), 2)
            cv2.line(frame, (lane2, 0), (lane2, height), (0, 255, 0), 2)

            vehicle_count = len(detections)

            lane_count = [0, 0, 0]
            for xyxy in detections.xyxy:
                x1, _, x2, _ = xyxy
                cx = int((x1 + x2) / 2)
                if cx < lane1:
                    lane_count[0] += 1
                elif cx < lane2:
                    lane_count[1] += 1
                else:
                    lane_count[2] += 1

            if detections.tracker_id is not None:
                for tracker_id, xyxy in zip(detections.tracker_id, detections.xyxy):
                    tid = int(tracker_id)
                    x1, y1, x2, y2 = xyxy
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    if tid not in vehicle_paths:
                        vehicle_paths[tid] = deque(maxlen=MAX_TRAIL_POINTS)
                    vehicle_paths[tid].append((cx, cy))

                    if cy < entry_y:
                        entered.add(tid)
                    if cy > exit_y:
                        exited.add(tid)

            draw_trails(frame, vehicle_paths)

            density = density_label(vehicle_count)
            signal = SIGNAL_BY_DENSITY[density]
            labels = build_labels(model, detections)

            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(
                scene=frame, detections=detections, labels=labels
            )

            draw_overlay(frame, [
                (f"Vehicles : {vehicle_count}", (0, 255, 0)),
                (f"Lane1 : {lane_count[0]}", (255, 255, 0)),
                (f"Lane2 : {lane_count[1]}", (255, 255, 0)),
                (f"Lane3 : {lane_count[2]}", (255, 255, 0)),
                (f"Density : {density}", (0, 255, 255)),
                (f"Signal : {signal}", (255, 0, 255)),
                (f"Entered : {len(entered)}", (255, 255, 255)),
                (f"Exited : {len(exited)}", (255, 255, 255)),
            ])

            csv_data.append({
                "Frame": frame_no,
                "Vehicles": vehicle_count,
                "Lane1": lane_count[0],
                "Lane2": lane_count[1],
                "Lane3": lane_count[2],
                "Entered": len(entered),
                "Exited": len(exited),
                "Density": density,
                "Signal": signal,
            })

            out.write(frame)

            if progress_callback:
                progress_callback(min(frame_no / total_frames, 1.0))
    finally:
        cap.release()
        out.release()

    if progress_callback:
        progress_callback(1.0)

    final_video_path = make_browser_playable(raw_video_path, output_video_path)

    # An explicit column list keeps the CSV schema stable even if no frame
    # produced a row (e.g. an unreadable video), which the graph step relies on.
    df = pd.DataFrame(csv_data, columns=CSV_COLUMNS)
    df.to_csv(output_csv_path, index=False)

    return final_video_path, output_csv_path, df
