"""
Day 6: Streamlit dashboard that chains Days 1-5 together.

Upload a video -> Detection -> Tracking -> ROI/Entry-Exit ->
Full Analytics (lanes + density + signal + CSV) -> Graphs -> Downloads.

Run it with:
    streamlit run app.py
from inside this folder, or from anywhere with:
    streamlit run "S:/TrafficAnalytics/TrafficAnalytics/app.py"
Every path below is anchored to this file, so the working directory
does not matter.
"""

import os
import sys
import traceback

import streamlit as st

# Make `from modules import ...` work no matter where Streamlit was launched.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from ultralytics import YOLO  # noqa: E402

from modules import detection, tracking, roi, analytics, graphs  # noqa: E402
from modules.utils import VideoError, open_video  # noqa: E402

st.set_page_config(page_title="Traffic Analytics Dashboard", layout="wide")

BASE_DIR = os.path.join(APP_DIR, "run_output")
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
CSV_DIR = os.path.join(BASE_DIR, "csv")
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")
MODEL_PATH = os.path.join(APP_DIR, "yolo11n.pt")

for d in (VIDEO_DIR, CSV_DIR, GRAPH_DIR):
    os.makedirs(d, exist_ok=True)


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model():
    # Falls back to the model name so ultralytics downloads it if the
    # bundled weights file is missing.
    return YOLO(MODEL_PATH if os.path.exists(MODEL_PATH) else "yolo11n.pt")


st.title("🚦 Traffic Analytics Dashboard")
st.caption(
    "Vehicle Detection → Tracking → ROI/Entry-Exit → "
    "Density & Signal Analytics → Graphs"
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Processing options")
    st.caption(
        "The pipeline runs YOLO over the video once per step, so a long "
        "clip at full quality can take a while. Use these to test quickly."
    )
    stride = st.select_slider(
        "Frame sampling",
        options=[1, 2, 3, 5, 10],
        value=1,
        format_func=lambda n: "Every frame (best)" if n == 1 else f"Every {n}th frame",
    )
    limit_enabled = st.checkbox("Only process the first N seconds", value=False)
    limit_seconds = st.number_input(
        "Seconds", min_value=1, max_value=600, value=10, step=5,
        disabled=not limit_enabled,
    )
    st.divider()
    st.caption(f"Outputs are written to:\n\n`{BASE_DIR}`")

# ------------------------------------------------------------------ upload
uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])

if uploaded_file is None:
    st.info("Upload a video above to get started.")
    st.stop()

input_path = os.path.join(VIDEO_DIR, "input.mp4")
with open(input_path, "wb") as f:
    f.write(uploaded_file.read())

st.video(input_path)

# Probe the clip so we can report its properties and convert seconds -> frames.
try:
    cap, width, height, fps, total_frames = open_video(input_path)
    cap.release()
except VideoError as exc:
    st.error(str(exc))
    st.stop()

st.caption(
    f"{width}×{height} · {fps:.1f} fps · {total_frames} frames "
    f"(~{total_frames / fps:.1f}s)"
)

max_frames = int(limit_seconds * fps) if limit_enabled else None

if st.button("Run Complete Pipeline 🚀", type="primary"):
    common = {"stride": stride, "max_frames": max_frames}
    try:
        model = load_model()

        # ------------------------- Step 1: Detection -------------------------
        st.subheader("Step 1 — Vehicle Detection")
        bar1 = st.progress(0.0)
        detection_out = os.path.join(VIDEO_DIR, "output.mp4")
        detection.run_detection(
            model, input_path, detection_out,
            progress_callback=bar1.progress, **common
        )
        st.video(detection_out)

        # ------------------------- Step 2: Tracking --------------------------
        st.subheader("Step 2 — Vehicle Tracking (ByteTrack)")
        bar2 = st.progress(0.0)
        tracking_out = os.path.join(VIDEO_DIR, "tracked_output.mp4")
        tracking.run_tracking(
            model, input_path, tracking_out,
            progress_callback=bar2.progress, **common
        )
        st.video(tracking_out)

        # --------------------- Step 3: ROI + Entry/Exit ----------------------
        st.subheader("Step 3 — ROI + Entry/Exit")
        bar3 = st.progress(0.0)
        roi_out = os.path.join(VIDEO_DIR, "day3_output.mp4")
        roi.run_roi_analysis(
            model, input_path, roi_out,
            progress_callback=bar3.progress, **common
        )
        st.video(roi_out)

        # ----------------------- Step 4: Full analytics ----------------------
        st.subheader("Step 4 — Smart Traffic Analytics")
        bar4 = st.progress(0.0)
        final_out = os.path.join(VIDEO_DIR, "final_output.mp4")
        csv_out = os.path.join(CSV_DIR, "traffic_log.csv")
        _, _, df = analytics.run_full_analytics(
            model, input_path, final_out, csv_out,
            progress_callback=bar4.progress, **common
        )
        st.video(final_out)
        st.dataframe(df.head(20), width="stretch")

        # -------------------------- Step 5: Graphs ---------------------------
        st.subheader("Step 5 — Graphs")
        graph_paths, summary = graphs.generate_graphs(csv_out, GRAPH_DIR)

        col1, col2 = st.columns(2)
        for i, key in enumerate(graphs.CHART_KEYS):
            with (col1 if i % 2 == 0 else col2):
                st.image(graph_paths[key], width="stretch")

        st.subheader("Summary")
        st.dataframe(summary, width="stretch")

        # Stash everything so downloads survive Streamlit reruns.
        st.session_state["pipeline_done"] = True
        st.session_state["outputs"] = {
            "Detection Video": detection_out,
            "Tracked Video": tracking_out,
            "ROI Video": roi_out,
            "Final Analytics Video": final_out,
            "Traffic Log CSV": csv_out,
            "Summary CSV": graph_paths["summary"],
        }

    except Exception as exc:  # surface the real cause instead of a blank page
        st.session_state["pipeline_done"] = False
        st.error(f"The pipeline stopped: {exc}")
        with st.expander("Full traceback"):
            st.code(traceback.format_exc())

# ---------------------------------------------------------------- downloads
if st.session_state.get("pipeline_done"):
    st.divider()
    st.subheader("⬇️ Download Everything")
    cols = st.columns(3)
    for i, (name, path) in enumerate(st.session_state["outputs"].items()):
        with cols[i % 3]:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    st.download_button(
                        name, f, file_name=os.path.basename(path), key=f"dl_{i}"
                    )
            else:
                st.caption(f"{name} — not available")
