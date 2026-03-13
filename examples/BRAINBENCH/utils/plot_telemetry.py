import glob
import json
import os
import sys
from datetime import timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def set_modern_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "figure.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.grid": True,
        }
    )


def format_time_axis(ax: plt.Axes, duration: timedelta) -> None:
    if duration < timedelta(minutes=5):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=5))
    elif duration < timedelta(hours=1):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def load_run_events(search_path: str) -> list[tuple[pd.Timestamp, str]]:
    events = []
    pattern = os.path.join(search_path, "*_run_data.json")

    for fpath in glob.glob(pattern):
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
                if "timestamps" in data:
                    for ts_entry in data["timestamps"]:
                        dt = pd.to_datetime(ts_entry[0])
                        label = ts_entry[1]
                        events.append((dt, label))
        except Exception as e:
            print(f"Warning: Could not parse JSON {fpath}: {e}")

    return sorted(events, key=lambda x: x[0])


def overlay_events(
    ax: plt.Axes,
    events: list[tuple[pd.Timestamp, str]],
    df_min: pd.Timestamp,
    df_max: pd.Timestamp,
) -> None:
    from collections import defaultdict

    valid_events = defaultdict(list)
    for dt, label in events:
        if df_min <= dt <= df_max:
            valid_events[dt].append(label)

    y_min, y_max = ax.get_ylim()
    text_y = y_max - (y_max - y_min) * 0.05 if y_max > y_min else y_max

    for dt, labels in valid_events.items():
        ax.axvline(
            dt,
            color="#444444",
            linestyle="--",
            alpha=0.35,
            linewidth=1.0,
            zorder=0,
        )

        full_label = " | ".join(labels)

        ax.text(
            dt,
            text_y,
            f"  {full_label}",
            rotation=90,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=8,
            color="#555555",
            alpha=0.9,
        )


def ensure_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def create_single_axis_plot(
    df: pd.DataFrame,
    x_col: str,
    y_process_col: str,
    y_system_col: str,
    title: str,
    y_label: str,
    output_path: str,
    duration: timedelta,
    events: list[tuple[pd.Timestamp, str]],
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
    y_limit_max: float | None = None,
) -> None:
    dynamic_width = min(max(14, len(df) * 0.15), 40)

    fig, ax = plt.subplots(figsize=(dynamic_width, 8))

    ax.plot(
        df[x_col],
        df[y_process_col],
        label="Process",
        linewidth=2.0,
        marker="o",
        markersize=3,
    )
    ax.plot(
        df[x_col],
        df[y_system_col],
        label="System",
        linewidth=2.0,
        linestyle="--",
        marker="s",
        markersize=3,
    )

    ax.set_title(title, pad=16)
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)

    if y_limit_max is not None:
        ax.set_ylim(0, y_limit_max)
    else:
        max_candidates = []
        if df[y_process_col].notna().any():
            max_candidates.append(df[y_process_col].max())
        if df[y_system_col].notna().any():
            max_candidates.append(df[y_system_col].max())

        ymax = max(max_candidates) * 1.1 if max_candidates else 1
        if ymax <= 0:
            ymax = 1
        ax.set_ylim(0, ymax)

    overlay_events(ax, events, t_min, t_max)
    format_time_axis(ax, duration)
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_gpu_plot(
    df: pd.DataFrame,
    output_path: str,
    duration: timedelta,
    events: list[tuple[pd.Timestamp, str]],
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
) -> None:
    dynamic_width = min(max(14, len(df) * 0.15), 40)

    fig, ax1 = plt.subplots(figsize=(dynamic_width, 8))

    # Left axis: GPU memory
    (line1,) = ax1.plot(
        df["timestamp"],
        df["gpu_MiB_process"],
        label="GPU Memory - Process",
        linewidth=2.0,
        marker="o",
        markersize=3,
    )
    (line2,) = ax1.plot(
        df["timestamp"],
        df["gpu_MiB_system"],
        label="GPU Memory - System",
        linewidth=2.0,
        linestyle="--",
        marker="s",
        markersize=3,
    )

    ax1.set_title("GPU: Memory vs System Utilization", pad=16)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("GPU Memory Usage (MiB)")

    mem_candidates = []
    if df["gpu_MiB_process"].notna().any():
        mem_candidates.append(df["gpu_MiB_process"].max())
    if df["gpu_MiB_system"].notna().any():
        mem_candidates.append(df["gpu_MiB_system"].max())

    mem_ymax = max(mem_candidates) * 1.1 if mem_candidates else 1
    if mem_ymax <= 0:
        mem_ymax = 1
    ax1.set_ylim(0, mem_ymax)

    overlay_events(ax1, events, t_min, t_max)
    format_time_axis(ax1, duration)

    # Right axis: system GPU utilization only
    ax2 = ax1.twinx()
    (line3,) = ax2.plot(
        df["timestamp"],
        df["gpu_util_system"],
        label="GPU Utilization - System",
        linewidth=2.0,
        linestyle="--",
        marker="s",
        markersize=3,
        color="green",
    )
    ax2.set_ylabel("GPU Utilization (%)")
    ax2.set_ylim(0, 100)

    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_visuals(csv_path: str, output_dir: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    required_cols = [
        "cpu_percent_process",
        "cpu_percent_system",
        "ram_MiB_process",
        "ram_MiB_system",
        "gpu_util_system",
        "gpu_MiB_process",
        "gpu_MiB_system",
    ]
    df = ensure_numeric_columns(df, required_cols)

    numeric_cols = [col for col in required_cols if col in df.columns]
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="linear", limit_direction="both"
    )

    t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
    duration = t_max - t_min

    events = load_run_events(output_dir)

    create_single_axis_plot(
        df=df,
        x_col="timestamp",
        y_process_col="cpu_percent_process",
        y_system_col="cpu_percent_system",
        title="CPU Usage: Process vs System",
        y_label="CPU Usage (%)",
        output_path=os.path.join(output_dir, "telemetry_cpu.png"),
        duration=duration,
        events=events,
        t_min=t_min,
        t_max=t_max,
    )

    create_single_axis_plot(
        df=df,
        x_col="timestamp",
        y_process_col="ram_MiB_process",
        y_system_col="ram_MiB_system",
        title="RAM Usage: Process vs System",
        y_label="RAM Usage (MiB)",
        output_path=os.path.join(output_dir, "telemetry_ram.png"),
        duration=duration,
        events=events,
        t_min=t_min,
        t_max=t_max,
    )

    create_gpu_plot(
        df=df,
        output_path=os.path.join(output_dir, "telemetry_gpu.png"),
        duration=duration,
        events=events,
        t_min=t_min,
        t_max=t_max,
    )

    stats = df.describe(include="all")
    stats.to_csv(os.path.join(output_dir, "telemetry_statistics.csv"))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_telemetry.py <telemetry_csv> <output_directory>")
        sys.exit(1)

    csv_path = sys.argv[1]
    save_path = sys.argv[2]

    set_modern_style()
    create_visuals(csv_path, save_path)
