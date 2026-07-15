"""
Day 5 logic: build the analysis charts + summary CSV from traffic_log.csv.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, works headless
import matplotlib.pyplot as plt


def generate_graphs(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    plt.figure(figsize=(12, 5))
    plt.plot(df["Frame"], df["Vehicles"])
    plt.title("Vehicle Count")
    plt.xlabel("Frame")
    plt.ylabel("Vehicles")
    plt.grid()
    p = os.path.join(output_dir, "vehicle_count.png")
    plt.savefig(p)
    plt.close()
    paths["vehicle_count"] = p

    plt.figure(figsize=(12, 5))
    plt.plot(df["Frame"], df["Lane1"], label="Lane1")
    plt.plot(df["Frame"], df["Lane2"], label="Lane2")
    plt.plot(df["Frame"], df["Lane3"], label="Lane3")
    plt.legend()
    plt.grid()
    plt.title("Lane Wise Vehicles")
    p = os.path.join(output_dir, "lane_graph.png")
    plt.savefig(p)
    plt.close()
    paths["lane_graph"] = p

    density = df["Density"].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(density, labels=density.index, autopct="%1.1f%%")
    plt.title("Traffic Density")
    p = os.path.join(output_dir, "density_chart.png")
    plt.savefig(p)
    plt.close()
    paths["density_chart"] = p

    signal = df["Signal"].value_counts()
    plt.figure(figsize=(6, 6))
    plt.bar(signal.index, signal.values)
    plt.title("Signal Recommendation")
    p = os.path.join(output_dir, "signal_chart.png")
    plt.savefig(p)
    plt.close()
    paths["signal_chart"] = p

    plt.figure(figsize=(10, 5))
    plt.plot(df["Frame"], df["Entered"], label="Entered")
    plt.plot(df["Frame"], df["Exited"], label="Exited")
    plt.legend()
    plt.grid()
    plt.title("Entry vs Exit")
    p = os.path.join(output_dir, "entry_exit.png")
    plt.savefig(p)
    plt.close()
    paths["entry_exit"] = p

    summary = pd.DataFrame({
        "Maximum Vehicles": [df["Vehicles"].max()],
        "Average Vehicles": [df["Vehicles"].mean()],
        "Minimum Vehicles": [df["Vehicles"].min()],
        "Maximum Lane1": [df["Lane1"].max()],
        "Maximum Lane2": [df["Lane2"].max()],
        "Maximum Lane3": [df["Lane3"].max()],
    })
    summary_path = os.path.join(output_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    paths["summary"] = summary_path

    return paths, summary
