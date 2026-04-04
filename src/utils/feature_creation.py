import numpy as np
import pandas as pd
from random import sample, random
from scipy import stats

hrrr_features = [ "hrrr_dpt","hrrr_u","hrrr_v","hrrr_t", \
                "hrrr_rh","hrrr_tp","hrrr_mstav","hrrr_sdwe"]
rave_features = ["rave_FRP_MEAN","rave_FRP_SD"]

all_features = ["esi_DFPPM", "modis_MaxFRP","rave_FRP_MEAN","rave_FRP_SD",
                "hrrr_dpt","hrrr_u","hrrr_v","hrrr_t",
                "hrrr_rh","hrrr_tp","hrrr_mstav","hrrr_sdwe"]

def process_features(features_csv, seed = 42, target_name = 'rave_FRP_MEAN', 
                    train_split = 0.8, stratify_by = 'n_days', 
                    lookback_days = 1, idx_name = 'cp', end_pad = 1,
                    pred_growth = True, pred_days = 1, keep_dtime = False,
                    dry_run_cps = [], min_FRP = None, max_FRP = None, scale = "linear"):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(features_csv)
    good_cps = []
    for cp_idx in df.cp.unique():
        if not (df[hrrr_features].isna().all().all() or df[rave_features].isna().all().all()):
            good_cps.append(cp_idx)
    df_filtered = df[df.cp.isin(good_cps)]
    df_filtered = filter_outliers(df_filtered)
    df_filtered = add_fire_day_col(df_filtered, idx_name='cp')
    print(f"{len(good_cps)} fires selected")
    for col in df_filtered.columns:
        if col in ["cp", "day"]: 
            continue
        elif col in ["rave_FRP_MEAN", "rave_FRP_SD", "modis_MaxFRP"]:
            this_col = df_filtered[col]
            col_mean = this_col.mean(skipna = True)
            col_std = this_col.std(skipna = True)
            
            na_mask = this_col.isna()

            df_filtered.loc[na_mask, col] = np.zeros(na_mask.sum())
            if not col == "rave_FRP_MEAN":
                min_FRP = np.min(df_filtered[col])
                max_FRP = np.max(df_filtered[col])
            else:
                if min_FRP is None:
                    min_FRP = np.min(df_filtered[col])
                if max_FRP is None:
                    max_FRP = np.max(df_filtered[col])
                print(f"FRP min, max: ({min_FRP}, {max_FRP})")
            
            normalized = (df_filtered[col] - min_FRP)/(max_FRP - min_FRP)
            df_filtered[col] = normalized
        else:
            this_col = df_filtered[col]
            col_mean = this_col.mean(skipna = True)
            col_std = this_col.std(skipna = True)
            
            na_mask = this_col.isna()

            rand_vals = rng.normal(loc = col_mean, scale = col_std, size = na_mask.sum())
            df_filtered.loc[na_mask, col] = rand_vals

            normalized = (df_filtered[col] - np.min(df_filtered[col]))/(np.max(df_filtered[col])- np.min(df_filtered[col]))
            df_filtered[col] = normalized
    
    df_filtered['n_days'] = None
    for cp_idx in df_filtered.cp.unique():
        days = df_filtered[df_filtered.cp == cp_idx].day
        n_days = (pd.Timestamp(days.max()) - pd.Timestamp(days.min())).days + 1
        df_filtered["n_days"] = df_filtered["n_days"].where(df_filtered.cp != cp_idx, n_days)

    train_X, train_y, test_X, test_y = split_data(df_filtered, all_features, 
                                                    target_name = target_name, train_split = train_split,
                                                    stratify_by = stratify_by, lookback_days = lookback_days,
                                                    idx_name = idx_name, end_pad = end_pad, scale = scale, 
                                                    pred_growth = pred_growth, pred_days = pred_days,
                                                    keep_dtime = keep_dtime, dry_run_cps = dry_run_cps)
    print("completed data processing")
    return train_X, train_y, test_X, test_y

def filter_outliers(df, idx_col = 'cp', z_thresh = 3, 
                    drop_cols = ['cp', 'day'], 
                    cols = ['esi_DFPPM']):
    if cols is None:
        cols = list(set(df.columns) - set(drop_cols))

    idxed_df = df.groupby(idx_col)[cols].mean().reset_index()
    bad_idx = set()
    for col in cols:
        idxed_df[f'z_score_{col}'] = np.abs(stats.zscore(idxed_df[col], nan_policy = 'omit'))
        bad_idx = bad_idx | set(idxed_df[idxed_df[f'z_score_{col}'] > z_thresh][idx_col].unique())
    return df[~df[idx_col].isin(bad_idx)]

def split_data(df, feature_names, target_name = 'rave_FRP_MEAN', 
                    train_split = 0.8, stratify_by = 'n_days', scale = "linear",
                    lookback_days = 1, idx_name = 'cp', end_pad = 1,
                    pred_growth = True, pred_days = 1, keep_dtime = False,
                    dry_run_cps = []):
    # split data into train and test subsetted by fire duration
    # data are feature labels for previous two days (yesterday and today)
    train_idx = []
    for bucket in df[stratify_by].unique():
        bucket_idxs = df[df[stratify_by] == bucket][idx_name].unique()
        train_idx.extend(sample(list(bucket_idxs), k = int(len(bucket_idxs)*train_split)))
    
    train_data = []
    test_data = []

    train_labels = []
    test_labels = []
    for idx in df[idx_name].unique():
        this_data = df[df[idx_name] == idx]
        days = sorted(this_data.day.unique())
        for day_i in range(lookback_days, len(days)-end_pad-pred_days):
            day_dict = {'idx' : idx, stratify_by : np.squeeze(this_data[stratify_by].unique()), 
                        'day' : np.squeeze(this_data[this_data.day == days[day_i+pred_days]]['day'])}
            for day_j in range(lookback_days + 1):
                data_j = this_data[this_data.day == days[day_i - day_j]][feature_names]
                for name in feature_names:
                    day_dict[f"{name}_d{day_j}"] = np.squeeze(data_j[name])
            if pred_growth:
                label = np.squeeze(this_data[this_data.day == days[day_i+pred_days]][target_name]) / \
                            np.squeeze(this_data[this_data.day == days[day_i]][target_name])
                if scale == "log":
                    label = np.log(label)
                if scale == "abs":
                    label = np.squeeze(this_data[this_data.day == days[day_i+pred_days]][target_name]) - \
                        np.squeeze(this_data[this_data.day == days[day_i]][target_name])
            else:
                label = np.squeeze(this_data[this_data.day == days[day_i+pred_days]][target_name])
            if keep_dtime:
                day_dict["day"] = np.squeeze(days[day_i+pred_days])
            if idx in train_idx and not idx in dry_run_cps:
                train_data.append(day_dict)
                train_labels.append({target_name : label})
            else:
                test_data.append(day_dict)
                test_labels.append({target_name : label})

    train_data = pd.DataFrame(train_data)
    train_labels = pd.DataFrame(train_labels)
    test_data = pd.DataFrame(test_data)
    test_labels = pd.DataFrame(test_labels)
    return train_data, train_labels, test_data, test_labels

def add_fire_day_col(df : pd.DataFrame, idx_name = 'idx', date_name = 'day', day_count_name = 'day_num'):
    df_out = df.sort_values([idx_name, date_name]).copy()
    
    df_out[day_count_name] = df_out.groupby(idx_name).cumcount() + 1
    return df_out