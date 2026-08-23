"""
Day 3 logic: ROI polygon, vehicle trajectories, simple entry/exit
counting based on y-position, and a basic density label.

The ROI and the entry/exit trip lines are derived from the actual frame
size, so the same code works on any resolution.
"""

from collections import deque

import cv2
import supervision as sv

from modules.utils import (
    MAX_TRAIL_POINTS,
    open_video,
    create_writer,
    scaled_roi,
    entry_exit_lines,
    density_label,
    build_labels,
    draw_overlay,
    draw_trails,
    make_browser_playable,
)


def run_roi_analysis(model, video_path, output_path, roi=None,
                     stride=1, max_frames=None, progress_callback=None):
    """
    roi: optional Nx2 int32 polygon. Defaults to a polygon scaled to the frame.
    """
    stride = max(1, int(stride))
    raw_path = output_path.replace(".mp4", "_raw.mp4")

    cap, width, height, fps, total_frames = open_video(video_path)
    if max_frames:
        total_frames = min(total_frames, int(max_frames))

    if roi is None:
        roi = scaled_roi(width, height)
    entry_y, exit_y = entry_exit_lines(height)

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    out = create_writer(raw_path, fps / stride, width, height)

    vehicle_paths = {}
    entered = set()
    exited = set()

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

            if detections.tracker_id is not None:
                for tracker_id, xyxy in zip(detections.tracker_id, detections.xyxy):
                    tid = int(tracker_id)
                    x1, y1, x2, y2 = xyxy
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    # A bounded deque keeps long videos from slowing to a crawl.
                    if tid not in vehicle_paths:
                        vehicle_paths[tid] = deque(maxlen=MAX_TRAIL_POINTS)
                    vehicle_paths[tid].append((cx, cy))

                    # Only the vehicles seen in this frame can change state,
                    # so there is no need to rescan every path ever recorded.
                    if cy < entry_y:
                        entered.add(tid)
                    if cy > exit_y:
                        exited.add(tid)

            draw_trails(frame, vehicle_paths)

            vehicle_count = len(detections)
            density = density_label(vehicle_count)
            labels = build_labels(model, detections)

            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(
                scene=frame, detections=detections, labels=labels
            )

            draw_overlay(frame, [
                (f"Vehicles : {vehicle_count}", (0, 255, 0)),
                (f"Density : {density}", (255, 255, 0)),
                (f"Entered : {len(entered)}", (255, 0, 255)),
                (f"Exited : {len(exited)}", (0, 255, 255)),
            ], scale=1.0)

            out.write(frame)

            if progress_callback:
                progress_callback(min(frame_no / total_frames, 1.0))
    finally:
        cap.release()
        out.release()

    if progress_callback:
        progress_callback(1.0)

    return make_browser_playable(raw_path, output_path)
