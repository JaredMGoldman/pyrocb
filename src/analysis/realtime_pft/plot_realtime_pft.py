import os
import io
import sys
import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from tqdm import tqdm as timer

from utils.constants import CACHE_BASE_DIR
import analysis.mapping.config as config


def _worker_render_single_plot(fire_key: str, fire_name: str, subset_df: pd.DataFrame, fx_name: str, output_dir: Path) -> str:
    """
    Renders a PFT time-series log-scale plot for a single fire footprint and 
    saves it directly as a JPEG file in the output directory.
    """
    matplotlib.use('Agg')
    if subset_df.empty:
        return f"[-] Skipped {fire_key}: Empty footprint data."

    # Normalize column names from CSV to match render logic
    df = subset_df.copy()
    df = df.rename(columns={'timestamp': 'time', 'pft_value': 'value'})

    # Pivot spatial coordinates across time steps for each plume top temperature
    pivoted_df = df.pivot_table(
        index='time', 
        columns='plume_temp', 
        values='value', 
        aggfunc='mean'
    ).sort_index()

    fig, ax = plt.subplots(figsize=(4.2, 2.4), dpi=250)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    
    time_index = pd.to_datetime(pivoted_df.index)
    colors = plt.cm.tab10.colors

    plume_temps = getattr(config, 'MAX_PLUME_TOP_TS', [0.0, -10.0, -20.0])
    for i, temp in enumerate(plume_temps):
        if temp in pivoted_df.columns:
            ax.plot(
                time_index, 
                pivoted_df[temp].values, 
                color=colors[i % len(colors)], 
                linewidth=1.2, 
                marker='o', 
                markersize=2, 
                label=f"{temp}°C"
            )

    # Vertical day boundary markers
    vline_colors = ['green', 'orange', 'blue', 'purple', 'red', 'brown']
    unique_days = pd.Series(time_index.floor('D')).unique()

    for i, day in enumerate(unique_days):
        if time_index.min() <= day <= time_index.max():
            ax.axvline(
                x=day, 
                color=vline_colors[i % len(vline_colors)], 
                linestyle='--', 
                linewidth=1.0, 
                alpha=0.8
            )
    
    plot_freq = 6
    ax.set_title(f"{fire_name}: PFT {fx_name.upper()} Prediction", color='black', fontsize=8, fontweight='bold')
    ax.set_ylabel("PFT Value (GW)", color='black', fontsize=7)
    ax.set_yscale('log')
    ax.tick_params(colors='black', labelsize=6)
    # ax.xaxis.set_major_locator(mdates.HourLocator(interval=plot_freq))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10, interval_multiples=plot_freq))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.grid(True, color='#444444', linestyle='--', alpha=0.5)
    ax.legend(title="Plume Temp", fontsize=5, title_fontsize=6, loc='upper left')

    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()

    # Save directly to disk for active fire run
    safe_fire_id = str(fire_name).replace(" ", "_").replace("/", "_")
    out_img_path = output_dir / f"pft_plot_{safe_fire_id}.jpg"

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', pil_kwargs={'quality': 87, 'optimize': True})
    plt.close(fig)

    with open(out_img_path, 'wb') as f:
        f.write(buf.getvalue())

    return str(out_img_path)


def generate_realtime_pft_plots(csv_path: Path, fx_name: str = "RRFS", max_workers: int = 4) -> Path:
    """
    Loads generated CSV, creates a dedicated run directory, and processes 
    fire plots in parallel.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"PFT CSV file not found at: {csv_path}")

    df_pft = pd.read_csv(csv_path)
    if df_pft.empty:
        raise ValueError(f"PFT CSV at {csv_path} contains no records.")

    # Infer run tag or timestamp from CSV name
    run_tag = csv_path.stem.replace("pft_rrfs_", "")
    output_dir = Path(CACHE_BASE_DIR) / "pft_plots" / f"run_{run_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group records per fire footprint
    grouped = df_pft.groupby(['fire_key', 'fire_name'])
    print(f"[*] Generating PFT plots for {len(grouped)} active fires into: {output_dir}")

    tasks = []
    for (fire_key, fire_name), subset_df in grouped:
        tasks.append((fire_key, fire_name, subset_df, fx_name, output_dir))

    saved_plots = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_worker_render_single_plot, *task): task[0] for task in tasks
        }

        for future in timer(as_completed(futures), desc="Rendering Fire PFT Plots", total=len(futures)):
            try:
                res = future.result()
                saved_plots.append(res)
            except Exception as e:
                print(f"[-] Plotting failure: {e}", file=sys.stderr)

    print(f"[+] Successfully rendered {len(saved_plots)} plots to {output_dir}")
    return output_dir


if __name__ == "__main__":
    # Example execution picking up the latest output CSV
    pft_out_dir = Path(CACHE_BASE_DIR) / "pft_outputs"
    csv_files = sorted(pft_out_dir.glob("pft_rrfs_*.csv"))

    if csv_files:
        latest_csv = csv_files[-1]
        generate_realtime_pft_plots(latest_csv, fx_name="RRFS", max_workers=getattr(config, 'max_workers', 4))
    else:
        print("[-] No PFT output CSV files found to plot.")