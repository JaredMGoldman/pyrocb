from utils.constants import DATA_DIR, FEATURE_OUTPUT_DIR, PLOTS_DIR
from utils.feature_creation import process_features
from utils.utils import load_model

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from quantile_forest import RandomForestQuantileRegressor
from shapely import Point, intersects
import time

CP_POLY_PATH = os.path.join(DATA_DIR,"cp_poly.gpkg")
CP_IDX_PATH = os.path.join(DATA_DIR,"cp_na.csv")

def find_cp(start_day, location):
    cp_polys = gpd.read_file(CP_POLY_PATH)
    cp_idx = pd.read_csv(CP_IDX_PATH)
    merged_cps = pd.merge(cp_idx,cp_polys, "inner", "cp")
    merged_cps['dtime_min'] = pd.to_datetime(merged_cps['dtime_min'], format = 'mixed')
    merged_cps['dtime_max'] = pd.to_datetime(merged_cps['dtime_max'], format = 'mixed')

    cp_times = merged_cps[(merged_cps.dtime_min <= start_day) & (merged_cps.dtime_max >= start_day)]
    return np.squeeze(cp_times[intersects(location, cp_times['geometry'])].cp)

def plot_model_results(model, X, y, 
                       method_name = 'Quantire Regression Forest',
                       fire_label = ""):
    
    out_dir = os.path.join(PLOTS_DIR, "fire_comparisons", str(fire_label), method_name.replace(' ', '_'))

    if type(model) is RandomForestQuantileRegressor:
        intervals = model.predict(X[model.feature_names_in_], quantiles=[0.025, 0.5, 0.975])
        ci = np.mean(intervals[:, 2] - intervals[:, 0])

    preds = model.predict(X[model.feature_names_in_])
    persist = np.transpose(X['rave_FRP_MEAN_d0'])
    y_vals = np.squeeze(np.transpose(y))

    model_res = y_vals - preds
    persist_res = y_vals - persist

    days = pd.to_datetime(X['day'].astype(str), format ='mixed')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    ax1.plot(days, persist, label = 'persistence', color = 'blue')
    ax1.scatter(days, persist, color = 'blue')
    
    ax1.plot(days, y_vals, label = 'FRP Growth', color = 'black')
    ax1.scatter(days, y_vals, color = 'black')

    ax1.errorbar(days, preds, label = method_name, color = 'orange', linestyle='--', yerr=ci)
    ax1.scatter(days, preds, color = 'orange')

    ax1.set_title(f"{str(fire_label)} Daily FRP Growth Comparison")
    # plt.xticks(days, rotation = 45)
    # plt.xlabel('Day')
    ax1.set_ylabel('FRP Growth')
    ax1.legend()

    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5) # Zero line
    ax2.plot(days, model_res, label=f'{method_name} Residuals', color='orange', marker='o')
    ax2.plot(days, persist_res, label='Persistence Residuals', color='blue', linestyle=':', marker='x')
    
    ax2.set_title("Residuals (True - Predicted)")
    ax2.set_ylabel('Error')
    ax2.set_xlabel('Day')
    ax2.legend()

    fig.autofmt_xdate() 
    
    # Adjust layout so title/labels don't hit the edges
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fname = f"{method_name.replace(' ', '_')}_{time.strftime('%Y%m%d-%H%M')}.png"
    plt.savefig(os.path.join(out_dir,fname), bbox_inches='tight')
    print(f"saved plot to {os.path.join(out_dir,fname)}")
    plt.close()


if __name__ == "__main__":
    # sample cp parameters
    lat, lon = 59.33205266422446,-120.29917172952204
    point = Point(lon, lat)
    date = pd.Timestamp("2025-06-18")
    features_csv_path = os.path.join(FEATURE_OUTPUT_DIR, "cleaned_data.csv")
    model_fname = "rf_pred1_20260325-1003"

    model = load_model(model_fname)

    fire_cp = find_cp(date, point)

    X, y, _, _ = process_features(  features_csv = features_csv_path, 
                                    train_split = 1.0, keep_dtime=True) # single day growth features
    idxs = X.loc[X['idx'] == fire_cp].index

    plot_model_results(model, X.iloc[idxs], y.iloc[idxs], fire_label = str(fire_cp))