from utils.constants import FEATURE_OUTPUT_DIR, PLOTS_DIR, CP_POLY_PATH, CP_IDX_PATH, dry_run_cps, dry_run_map
from utils.feature_creation import process_features
from utils.io_utils import load_model

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from shapely import intersects
import warnings

def find_cp(start_day, location):
    warnings.filterwarnings("ignore")  
    cp_polys = gpd.read_file(CP_POLY_PATH)
    cp_idx = pd.read_csv(CP_IDX_PATH)
    merged_cps = pd.merge(cp_idx,cp_polys, "inner", "cp")

    merged_cps['dtime_min'] = pd.to_datetime(merged_cps['dtime_min'], format = 'mixed')
    merged_cps['dtime_max'] = pd.to_datetime(merged_cps['dtime_max'], format = 'mixed')
    cp_times = merged_cps[(merged_cps.dtime_min <= start_day) & (merged_cps.dtime_max >= start_day)]
    cp_times['dist'] = location.hausdorff_distance(cp_times['geometry'])

    closest_cp = cp_times.sort_values(['dist']).cp.values[0]
    cp_data = cp_times[cp_times.cp == closest_cp]
    closest_ll = (round(np.float64(cp_data.lon_mean.values), 2), 
                  round(np.float64(cp_data.lat_mean.values), 2))
    feature_df = pd.read_csv(f"/home/jaredgoldman/dev/pyrocb/outputs/features/frp_subdaily_cleaned.csv")
    print(f"{location} : closest cp is {closest_cp} with center at {closest_ll}")
    print(f"start/end: {cp_data.dtime_min.values} -> {cp_data.dtime_max.values}")
    print(f"is in features? {closest_cp in feature_df.cp.values}")
    return [closest_cp]

def dry_run_results(model, X, y, 
                    feature_names = [],
                    pred_type = 'abs',
                    cps = []):
    min_FRP, max_FRP =  (-1.0173882246017456, 17615.265753424657)
    rmses = []
    for cp in cps:
        idxs = X.loc[X['idx'] == cp].index
        X = X.iloc[idxs]
        y = y.iloc[idxs]
        preds = model.predict(X[feature_names])
        
        if pred_type == 'linear':
            pred_FRPS = X["rave_FRP_MEAN_d0"].values * preds * (max_FRP - min_FRP) + min_FRP
            FRP_vals = X["rave_FRP_MEAN_d0"].values * y["rave_FRP_MEAN"].values * (max_FRP - min_FRP) + min_FRP
        elif pred_type == 'abs':
            pred_FRPS = (X["rave_FRP_MEAN_d0"].values + preds) * (max_FRP + min_FRP) + min_FRP
            FRP_vals = (X["rave_FRP_MEAN_d0"].values + y["rave_FRP_MEAN"].values) * (max_FRP - min_FRP) + min_FRP
        rmses.append(np.sqrt(np.mean((FRP_vals - pred_FRPS)**2)))
    
    return np.mean(np.array(rmses))

def plot_model_results(model, X, y,
                       method_name = 'Quantire Regression Forest',
                       fire_label = "",
                       min_FRP = None, 
                       max_FRP = None,
                       pred_type = 'abs', 
                       pred_days = 1,
                       out_dir = PLOTS_DIR,
                       feature_names = [],
                       ci_info = {},
                       stratify_by = 'n_days', 
                       interval = 6,
                       last_day = None):
    fire_id = X['cp'].iloc[0]
    if fire_id in dry_run_cps:
        fire_id = dry_run_map[fire_id]
    fname = f"{fire_id.replace(' ', '_')}_{method_name.replace(' ', '_')}.png"

    times = pd.to_datetime(X['time'].values)
    
    actual_frp_phys = X['max_frp'].values * (max_FRP - min_FRP) + min_FRP
    
    frp_vals = X['frp_lag_d0_h0'].values * (max_FRP - min_FRP) + min_FRP
    preds = model.predict(X[feature_names])
    pred_growth = preds * actual_frp_phys
    pred_times = times + pd.Timedelta(1, 'd')

    iter_pred_df = X.copy()
    pred_frp_df = pd.DataFrame({'time' : pred_times, 'max_frp' : (pred_growth - min_FRP) / (max_FRP - min_FRP)})
    iter_pred_df['max_frp'] = iter_pred_df['time'].map(pred_frp_df.set_index('time')['max_frp'])
    iter_pred_df = iter_pred_df.dropna(subset = ['max_frp'])
    drop_cols = [col for col in iter_pred_df.columns if 'frp_lag_d0' in col]
    iter_pred_df = iter_pred_df.drop(drop_cols, axis = 1)
    
    iter_preds = model.predict(iter_pred_df[feature_names])
    iter_pred_growth =  iter_preds * \
                        iter_pred_df['max_frp'].values * \
                        (max_FRP - min_FRP) + min_FRP
    iter_pred_times = pd.to_datetime(iter_pred_df.time.values) + pd.Timedelta(1, 'd')

    if not last_day is None:
        pred_times_copy = pred_times.copy()
        pred_growth_copy = pred_growth.copy()
        iter_pred_times_copy = iter_pred_times.copy()
        iter_pred_growth_copy = iter_pred_growth.copy()
        times_copy = times.copy()
        frp_vals_copy = frp_vals.copy()
        actual_frp_phys_copy = actual_frp_phys.copy()

        last_day = pd.to_datetime(last_day)
        last_day_p1 = last_day + pd.Timedelta(1, 'd')
        last_day_p2 = last_day + pd.Timedelta(2, 'd')
        last_time_idx = times.get_loc(last_day)
        try:
            last_time_p2_idx = times.get_loc(last_day_p2)
        except:
            last_time_p2_idx = -1

        try:
            last_day_preds = pred_times.get_loc(last_day)
        except:
            last_day_preds = 0
        last_time_idx_p1 = pred_times.get_loc(last_day_p1)

        try:
            last_day_iter_preds = iter_pred_times.get_loc(last_day)
        except:
            last_day_iter_preds= 0
        try:
            iter_preds_day_p1 = iter_pred_times.get_loc(last_day_p1)
        except:
            iter_preds_day_p1 = 0

        try:
            last_time_idx_p2 = iter_pred_times.get_loc(last_day_p2)
        except:
            last_time_idx_p2 = -1


        pred_tail_times = list(pred_times_copy[last_day_preds:last_time_idx_p1]) + \
            list(iter_pred_times_copy[iter_preds_day_p1:last_time_idx_p2])
        pred_tail = list(pred_growth_copy[last_day_preds:last_time_idx_p1]) + \
                            list(iter_pred_growth_copy[iter_preds_day_p1:last_time_idx_p2])

        frp_val_times =     times_copy[:last_time_p2_idx]
        frp_vals =          frp_vals_copy[:last_time_p2_idx]

        times =             times_copy[:last_time_idx]
        actual_frp_phys =   actual_frp_phys_copy[:last_time_idx]

        pred_times =        pred_times_copy[:last_day_preds] 
        pred_growth =       pred_growth_copy[:last_day_preds]

        iter_pred_times =   iter_pred_times_copy[:last_day_iter_preds]
        iter_pred_growth =  iter_pred_growth_copy[:last_day_iter_preds]
    
    if fire_id == 'Line':
        import ipdb; ipdb.set_trace()

    plt.plot(times, actual_frp_phys, color='black', linewidth=3, label='Ground Truth Max FRP', zorder=5)
    plt.plot(frp_val_times, frp_vals, color='gray', linestyle='--', linewidth=1, label='Ground Truth FRP')
    plt.plot(pred_times, pred_growth, color = 'blue', label = 'Max FRP (One-Day Prediction)')
    plt.plot(iter_pred_times, iter_pred_growth, color = 'orange', label = 'Max FRP (Two-Day Prediction)')
    try:
        plt.plot(pred_tail_times, pred_tail, color = 'green', label = 'Forecasted Max FRP') if not last_day is None else None
    except:
        import ipdb; ipdb.set_trace()
    plt.title(f"{fire_id} FRP Predictions", fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("FRP (MW)", fontsize=12)
    # plt.legend(loc = 0)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.gcf().autofmt_xdate()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir,fname), bbox_inches='tight')
    print(f"saved plot to {os.path.join(out_dir,fname)}")
    # pd.DataFrame({'times' : times, 'mean_pred_frp' : pred_growth, 'frp_labels' : actual_frp_phys}) \
    #     .to_csv(os.path.join(out_dir, f"{fire_label}_{str(pred_days)}_{pred_type}_{method_name.replace(' ', '_')}_time_series.csv"), 
    #             index = False)
    plt.close()
    return True
    return np.sqrt(np.mean((actual_frp_phys - np.array(pred_growth))**2))


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