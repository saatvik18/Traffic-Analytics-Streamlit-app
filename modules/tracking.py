"""
Day 2 logic: run YOLO + ByteTrack so every vehicle gets a persistent ID
across frames, and write an annotated video.
"""

import cv2
import supervision as sv

from modules.utils import make_browser_playable


def run_tracking(model, video_path, output_path, progress_callback=None):
    raw_path = output_path.replace(".mp4", "_raw.mp4")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

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

        labels = []
        if detections.tracker_id is not None:
            for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
                if tracker_id is None:
                    labels.append(model.names[int(class_id)])
                else:
                    labels.append(f"{model.names[int(class_id)]} ID:{int(tracker_id)}")

        annotated = frame.copy()
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        out.write(annotated)

        if progress_callback:
            progress_callback(min(frame_no / total_frames, 1.0))

    cap.release()
    out.release()
    return make_browser_playable(raw_path, output_path)
