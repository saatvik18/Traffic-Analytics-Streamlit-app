"""
Day 5 logic: build the analysis charts + summary CSV from traffic_log.csv.
"""

import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no GUI backend needed, works headless

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

CHART_KEYS = ["vehicle_count", "lane_graph", "density_chart", "signal_chart", "entry_exit"]


def _save(fig, output_dir, name, paths):
    path = os.path.join(output_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    paths[name.replace(".png", "")] = path


def generate_graphs(csv_path, output_dir):
    """
    Read the frame-by-frame log and write five PNG charts plus summary.csv.

    Returns (paths_dict, summary_dataframe). paths_dict has one entry per
    chart key plus "summary" for the CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(
            f"'{csv_path}' has no rows - the analytics step did not process "
            "any frames, so there is nothing to plot."
        )

    paths = {}

    # 1. Total vehicles per frame
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Frame"], df["Vehicles"], color="#1f77b4")
    ax.set_title("Vehicle Count")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Vehicles")
    ax.grid(alpha=0.3)
    _save(fig, output_dir, "vehicle_count.png", paths)

    # 2. Lane-wise breakdown
    fig, ax = plt.subplots(figsize=(12, 5))
    for lane in ("Lane1", "Lane2", "Lane3"):
        ax.plot(df["Frame"], df[lane], label=lane)
    ax.set_title("Lane Wise Vehicles")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Vehicles")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, output_dir, "lane_graph.png", paths)

    # 3. Share of frames at each density level
    density = df["Density"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(density.values, labels=list(density.index), autopct="%1.1f%%")
    ax.set_title("Traffic Density")
    _save(fig, output_dir, "density_chart.png", paths)

    # 4. Signal recommendation frequency
    signal = df["Signal"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(list(signal.index), signal.values, color="#2ca02c")
    ax.set_title("Signal Recommendation")
    ax.set_ylabel("Frames")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, output_dir, "signal_chart.png", paths)

    # 5. Cumulative entries vs exits
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Frame"], df["Entered"], label="Entered")
    ax.plot(df["Frame"], df["Exited"], label="Exited")
    ax.set_title("Entry vs Exit")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Vehicles (cumulative)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, output_dir, "entry_exit.png", paths)

    summary = pd.DataFrame({
        "Maximum Vehicles": [int(df["Vehicles"].max())],
        "Average Vehicles": [round(float(df["Vehicles"].mean()), 2)],
        "Minimum Vehicles": [int(df["Vehicles"].min())],
        "Maximum Lane1": [int(df["Lane1"].max())],
        "Maximum Lane2": [int(df["Lane2"].max())],
        "Maximum Lane3": [int(df["Lane3"].max())],
        "Total Entered": [int(df["Entered"].max())],
        "Total Exited": [int(df["Exited"].max())],
    })
    summary_path = os.path.join(output_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    paths["summary"] = summary_path

    return paths, summary
