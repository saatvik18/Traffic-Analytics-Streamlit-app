# 🚦 Traffic Analytics — YOLO + ByteTrack + Streamlit

A computer-vision pipeline that turns raw traffic camera footage into a full analytics dashboard — vehicle detection, multi-object tracking, lane-wise counting, traffic density classification, signal-timing recommendations, and auto-generated charts, all wrapped in a Streamlit web app.

Built as a 6-day progressive build: each stage (detection → tracking → ROI/lane counting → analytics → graphs → UI) was developed and validated independently before being assembled into one pipeline.

## Features

- **Vehicle Detection** — YOLO11 detects cars, trucks, buses, and motorcycles frame-by-frame
- **Multi-Object Tracking** — ByteTrack assigns persistent IDs so the same vehicle is tracked across frames
- **ROI & Lane Counting** — a region-of-interest polygon plus lane-divider lines split traffic into lanes and count vehicles entering/exiting
- **Smart Traffic Analytics** — per-frame traffic density (LOW/MEDIUM/HIGH) and a signal-timing recommendation, all logged to CSV
- **Auto-Generated Graphs** — vehicle count over time, lane-wise breakdown, density distribution, signal recommendation frequency, and entry/exit trends
- **Streamlit Dashboard** — upload a video, run the full pipeline, watch each stage's output video and chart appear live, and download the outputs

## Tech Stack

| Component | Tool |
|---|---|
| Object detection | [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) |
| Multi-object tracking | [Supervision](https://github.com/roboflow/supervision) (ByteTrack) |
| Video I/O & drawing | OpenCV |
| Data handling | Pandas |
| Charts | Matplotlib |
| Dashboard | Streamlit |

## Project Structure

```
TrafficAnalytics/
├── app.py                  # Streamlit dashboard — orchestrates the full pipeline
├── requirements.txt
├── yolo11n.pt               # YOLO weights (add locally — see Setup)
├── videos/                  # input + per-stage output videos
├── csv/                     # traffic_log.csv
├── graphs/                  # generated PNG charts + summary.csv
└── modules/
    ├── detection.py          # Step 1 — YOLO vehicle detection
    ├── tracking.py            # Step 2 — ByteTrack tracking
    ├── roi.py                  # Step 3 — ROI polygon + lane counting + entry/exit
    ├── analytics.py             # Step 4 — density, signal timing, CSV logging
    └── graphs.py                 # Step 5 — matplotlib charts + summary
```

## Pipeline

```
Upload video
     │
     ▼
Step 1 — Vehicle Detection ───────► output.mp4
     │
     ▼
Step 2 — Vehicle Tracking (ByteTrack) ───────► tracked_output.mp4
     │
     ▼
Step 3 — ROI + Lane Counting ───────► day3_output.mp4
     │
     ▼
Step 4 — Smart Traffic Analytics ───────► final_output.mp4 + traffic_log.csv
     │
     ▼
Step 5 — Graphs ───────► vehicle_count.png, lane_graph.png,
                          density_chart.png, signal_chart.png,
                          entry_exit.png, summary.csv
     │
     ▼
Download outputs
```

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/saatvik18/Traffic-Analytics-Streamlit-app.git
cd Traffic-Analytics-Streamlit-app
```

**2. Create a virtual environment**
```powershell
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add YOLO weights**
Place `yolo11n.pt` in the project root, or just run the app — `ultralytics` will auto-download it on first use if you're online.

**5. Run the app**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

## Usage

1. Upload a traffic video (`.mp4`, `.mov`, `.avi`, `.mkv`)
2. Click **Run Complete Pipeline**
3. Watch each stage's output video/chart appear as it's produced
4. Download the output videos, CSV log, and charts

## Notes

- The ROI polygon and lane-divider positions are currently hardcoded pixel coordinates tuned for 1920×1080 input. For a different resolution or camera angle, tune the coordinates in `roi.py` and `analytics.py`.
- Each pipeline step currently reruns detection + tracking independently on the full video. A future optimization is to run detection/tracking once and reuse the results across steps 3 and 4.

## License

Add a license of your choice (e.g. MIT) if you plan to share this publicly.
