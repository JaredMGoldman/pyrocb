from utils.constants import DATA_DIR, FEATURE_OUTPUT_DIR, PLOTS_DIR, CP_POLY_PATH, CP_IDX_PATH, dry_run_cps
from utils.feature_creation import process_features
from utils.utils import load_model

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from quantile_forest import RandomForestQuantileRegressor
from shapely import intersects

def find_cp(start_day, location):
    cp_polys = gpd.read_file(CP_POLY_PATH)
    cp_idx = pd.read_csv(CP_IDX_PATH)
    merged_cps = pd.merge(cp_idx,cp_polys, "inner", "cp")
    merged_cps['dtime_min'] = pd.to_datetime(merged_cps['dtime_min'], format = 'mixed')
    merged_cps['dtime_max'] = pd.to_datetime(merged_cps['dtime_max'], format = 'mixed')

    cp_times = merged_cps[(merged_cps.dtime_min <= start_day) & (merged_cps.dtime_max >= start_day)]
    cps = np.squeeze(cp_times[intersects(location, cp_times['geometry'])].cp)
    if not type(cps) is np.int64:
        cps = cps.values
        return list(cps)
    return [cps]

def plot_model_results(model, X, y,
                       method_name = 'Quantire Regression Forest',
                       fire_label = "",
                       min_FRP = None, 
                       max_FRP = None,
                       pred_type = 'abs', 
                       pred_days = 1,
                       out_dir = PLOTS_DIR,
                       feature_names = []):
    res_offset = 0 if pred_type == 'abs' else 1
    fname = f"{fire_label}_{str(pred_days)}_{pred_type}_{method_name.replace(' ', '_')}.png"

    if type(model) is RandomForestQuantileRegressor:
        intervals = model.predict(X[feature_names], quantiles=[0.025, 0.5, 0.975])
        ci = np.mean(intervals[:, 2] - intervals[:, 0])

    preds = model.predict(X[feature_names])
    y_vals = np.squeeze(np.transpose(y))

    model_res = np.abs(y_vals - preds)
    persist_res = np.abs(y_vals - res_offset)

    days = pd.to_datetime(X['day'].astype(str), format ='mixed')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    ax1.plot(days, y_vals, label = 'FRP Growth', color = 'black', marker = 'o')

    ax1.errorbar(days, preds, label = method_name, color = 'orange', linestyle='--', yerr=ci, marker = 'o')

    ax1.set_title(f"{str(fire_label)} Daily FRP Growth Comparison")
    ax1.set_ylabel('FRP Growth')
    ax1.legend()

    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5) # Zero line
    ax2.plot(days, model_res, label=f'{method_name} Residuals', linestyle='--', color='orange', marker='o')
    ax2.plot(days,  persist_res, label='Persistence Residuals', color='blue', marker='o')
    
    ax2.set_title("Residuals")
    ax2.set_ylabel('Error')
    ax2.legend()
    
    # calculate FRP predictions based on training/evaluation
        # frp_0 * scale = frp_1
        # frp_x = (frp_raw_x - min_fpr)/(max_frp - min_frp) =>
        # (frp_raw_1 - min_fpr)/(max_frp - min_frp)  = frp_0 * scale =>
        # frp_raw_1 = frp_0 * scale * (max_frp - min_frp) + min_frp

    if pred_type == 'linear':
        pred_FRPS = X["rave_FRP_MEAN_d0"].values * preds * (max_FRP - min_FRP) + min_FRP
        FRP_vals = X["rave_FRP_MEAN_d0"].values * y["rave_FRP_MEAN"].values * (max_FRP - min_FRP) + min_FRP
    elif pred_type == 'abs':
        pred_FRPS = (X["rave_FRP_MEAN_d0"].values + preds) * (max_FRP + min_FRP) + min_FRP
        FRP_vals = (X["rave_FRP_MEAN_d0"].values + y["rave_FRP_MEAN"].values) * (max_FRP - min_FRP) + min_FRP
    abs_err = np.abs(pred_FRPS - FRP_vals)
        
    ax3.plot(days, pred_FRPS, label = 'Predicted FRP', color = 'orange', marker = 'o')
    ax3.plot(days, FRP_vals, label = 'Observed FRP', color = 'black', marker = 'o')
    ax3.plot(days, abs_err, label = 'Abs Error', color = 'red', linestyle = '--', marker = 'o')

    ax3.set_title("FRP Predictions")
    ax3.set_ylabel("FRP Predictions (MW)")
    ax3.set_xlabel('Day')
    ax3.legend()

    fig.autofmt_xdate() 
    
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir,fname), bbox_inches='tight')
    print(f"saved plot to {os.path.join(out_dir,fname)}")
    plt.close()

    return np.sqrt(np.mean((FRP_vals - pred_FRPS)**2))


if __name__ == "__main__":
    # sample cp parameters
    # min, max FRP for pred 1 (3/30/26): -1.0173882246017456, 17615.265753424657
    # min, max FRP for pred 2 (3/30/26): -1.0173882246017456, 17615.265753424657
    
    pred_days = 2
    scale = 'abs'
    
    features_csv_path = os.path.join(FEATURE_OUTPUT_DIR, "cleaned_data.csv")
    model_fname = f"rf_pred{pred_days}_dr_{scale}_20260330-1850"
    # model_fname = f"rf_pred{pred_days}_dr_{scale}_20260330-1847"

    min_FRP, max_FRP = (-1.0173882246017456, 17615.265753424657)

    model = load_model(model_fname)

    X, y, _, _ = process_features(  features_csv = features_csv_path, 
                                    train_split = 1.0, keep_dtime=True, 
                                    pred_days = pred_days, dry_run_cps = [], 
                                    min_FRP = min_FRP, max_FRP = max_FRP, 
                                    scale = scale)
    for cp in dry_run_cps:
        idxs = X.loc[X['idx'] == cp].index
        if len(idxs) == 0:
            continue

        plot_model_results(model, X.iloc[idxs], y.iloc[idxs], 
                           fire_label = str(cp), pred_days = pred_days,
                           min_FRP = min_FRP, max_FRP = max_FRP, 
                           pred_type = scale)