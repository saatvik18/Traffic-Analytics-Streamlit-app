"""
Day 6: Streamlit dashboard that chains Days 1-5 together.

Upload a video -> Detection -> Tracking -> ROI/Entry-Exit ->
Full Analytics (lanes + density + signal + CSV) -> Graphs -> Downloads.
"""

import os
import streamlit as st
from ultralytics import YOLO

from modules import detection, tracking, roi, analytics, graphs

st.set_page_config(page_title="Traffic Analytics Dashboard", layout="wide")

BASE_DIR = "run_output"
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
CSV_DIR = os.path.join(BASE_DIR, "csv")
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")

for d in (VIDEO_DIR, CSV_DIR, GRAPH_DIR):
    os.makedirs(d, exist_ok=True)


@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


st.title("🚦 Traffic Analytics Dashboard")
st.caption(
    "Vehicle Detection → Tracking → ROI/Entry-Exit → "
    "Density & Signal Analytics → Graphs"
)

uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    input_path = os.path.join(VIDEO_DIR, "input.mp4")
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(input_path)

    run_clicked = st.button("Run Complete Pipeline 🚀", type="primary")

    if run_clicked:
        model = load_model()

        # ---------------- Step 1: Detection ----------------
        st.subheader("Step 1 — Vehicle Detection")
        bar1 = st.progress(0.0)
        detection_out = os.path.join(VIDEO_DIR, "output.mp4")
        detection.run_detection(
            model, input_path, detection_out,
            progress_callback=lambda p: bar1.progress(p)
        )
        st.video(detection_out)

        # ---------------- Step 2: Tracking ----------------
        st.subheader("Step 2 — Vehicle Tracking (ByteTrack)")
        bar2 = st.progress(0.0)
        tracking_out = os.path.join(VIDEO_DIR, "tracked_output.mp4")
        tracking.run_tracking(
            model, input_path, tracking_out,
            progress_callback=lambda p: bar2.progress(p)
        )
        st.video(tracking_out)

        # ---------------- Step 3: ROI + Entry/Exit ----------------
        st.subheader("Step 3 — ROI + Entry/Exit")
        bar3 = st.progress(0.0)
        roi_out = os.path.join(VIDEO_DIR, "day3_output.mp4")
        roi.run_roi_analysis(
            model, input_path, roi_out,
            progress_callback=lambda p: bar3.progress(p)
        )
        st.video(roi_out)

        # ---------------- Step 4: Full analytics ----------------
        st.subheader("Step 4 — Smart Traffic Analytics")
        bar4 = st.progress(0.0)
        final_out = os.path.join(VIDEO_DIR, "final_output.mp4")
        csv_out = os.path.join(CSV_DIR, "traffic_log.csv")
        _, _, df = analytics.run_full_analytics(
            model, input_path, final_out, csv_out,
            progress_callback=lambda p: bar4.progress(p)
        )
        st.video(final_out)
        st.dataframe(df.head(20))

        # ---------------- Step 5: Graphs ----------------
        st.subheader("Step 5 — Graphs")
        graph_paths, summary = graphs.generate_graphs(csv_out, GRAPH_DIR)

        col1, col2 = st.columns(2)
        chart_keys = ["vehicle_count", "lane_graph", "density_chart", "signal_chart", "entry_exit"]
        for i, key in enumerate(chart_keys):
            with (col1 if i % 2 == 0 else col2):
                st.image(graph_paths[key])

        st.subheader("Summary")
        st.dataframe(summary)

        # Stash everything so downloads survive Streamlit reruns
        st.session_state["pipeline_done"] = True
        st.session_state["outputs"] = {
            "Detection Video": detection_out,
            "Tracked Video": tracking_out,
            "ROI Video": roi_out,
            "Final Analytics Video": final_out,
            "Traffic Log CSV": csv_out,
            "Summary CSV": graph_paths["summary"],
        }

    if st.session_state.get("pipeline_done"):
        st.divider()
        st.subheader("⬇️ Download Everything")
        cols = st.columns(3)
        for i, (name, path) in enumerate(st.session_state["outputs"].items()):
            with cols[i % 3]:
                with open(path, "rb") as f:
                    st.download_button(name, f, file_name=os.path.basename(path))
else:
    st.info("Upload a video above to get started.")
