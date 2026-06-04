import numpy as np
import pandas as pd
from random import sample, random
from scipy import stats
import warnings

hrrr_features = [ "hrrr_dpt","hrrr_u","hrrr_v","hrrr_t", \
                "hrrr_rh","hrrr_tp"] #,"hrrr_mstav","hrrr_sdwe"]
rave_features = ["rave_FRP_MEAN","rave_FRP_SD"]

all_features = ["esi_DFPPM","rave_FRP_MEAN","rave_FRP_SD",
                "hrrr_dpt","hrrr_u","hrrr_v","hrrr_t",
                "hrrr_rh","hrrr_tp"] # ,"hrrr_mstav","hrrr_sdwe", "modis_MaxFRP"

def sample_weights(y, factor = 1, method = 'prod'):
    calc = 0
    if method == 'prod':
        calc = np.abs(y)*factor
    elif method == 'id':
        calc = np.abs(y)
    elif method == 'sum':
        calc = np.abs(y) + factor
    elif method == 'exp':
        calc = factor**np.abs(y)
    return np.mean(calc.values, axis = 1)

def calculate_vpd(temp_k, rh):
    """
    Calculates Vapor Pressure Deficit (VPD) in kPa.
    temp_k: Temperature in Kelvin
    rh: Relative Humidity (0-100)
    """
    temp_c = temp_k - 273.15
    es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    ea = es * (rh / 100.0)
    vpd = es - ea
    
    return vpd

def calculate_hdw(temp_k, rh, wind_speed_ms):
    # (Using hPa for standard HDW units: 6.108 instead of 0.6108)
    temp_c = temp_k - 273.15
    es = 6.108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    
    ea = es * (rh / 100.0)
    vpd = es - ea
    
    hdw = vpd * wind_speed_ms
    
    return hdw

def generate_diurnal_features(df, target_col, n_future=1, idx_name='cp', daily_freq = 24, lookback = 2, model_type = 'ensemble'): 
    warnings.filterwarnings('ignore')
    df['wind_speed'] = np.sqrt(df['hrrr_u'] ** 2 + df['hrrr_v'] ** 2)
    df = df.drop(['hrrr_u', 'hrrr_v'], axis = 1)

    df['vpd'] = calculate_vpd(df['hrrr_t'], df['hrrr_rh'])
    df['hdw'] = calculate_hdw(df['hrrr_t'], df['hrrr_rh'], df['wind_speed'])

    df = data_normalization(df)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values([idx_name, 'time']).copy()
    
    # --- 1. Cyclical Time Encoding ---
    # Convert hour (0-23) into circular coordinates
    df['hour'] = df['time'].dt.hour
    df['hour_feat'] = np.sin(2 * np.pi * df['hour'] / 24)/ np.cos(2 * np.pi * df['hour'] / 24)
    
    X_features = ['hour_feat']
    daily_feats = list(set(hrrr_features) - set(['hrrr_u', 'hrrr_v'])) + ['esi_DFPPM', 'wind_speed', 'vpd', 'hdw']
    # --- 2. Historical FRP Lags ---
    print("processing frps")
    print("processing features")

    for d in range(lookback):
        for h in range(daily_freq):
            fname = f'frp_lag_d{d}_h{h}'
            df[fname] = df.groupby(idx_name)[target_col].shift(d * daily_freq + h)
            X_features.append(fname)
    if model_type == "ensemble":
        # 1. Create a key for 'Tomorrow'
        df['tomorrow_date'] = df['time'].dt.date + pd.Timedelta(days=1)

        # 2. Create a reference table of all weather values by Date and Hour
        weather_ref = df.pivot_table(
            index=[idx_name, df['time'].dt.date], 
            columns=df['time'].dt.hour, 
            values=daily_feats
        )

        # Flatten the multi-index columns: 'temp_0', 'temp_1'... 'rh_23'
        weather_ref.columns = [f'{col}_f{hour:02d}' for col, hour in weather_ref.columns]

        # 3. Merge this reference table back into the main DF using 'tomorrow_date'
        df = df.merge(
            weather_ref, 
            left_on=[idx_name, 'tomorrow_date'], 
            right_index=True, 
            how='left'
        )
        # Update X_features list
        X_features.extend([c for c in weather_ref.columns])
    else:
    # --- 3. Future Weather Forecasts (T+0 to T+23) ---
    # For a multi-output model, we need the weather for the ENTIRE forecast window
        for h in range(n_future):
            for w_col in daily_feats:
                fname = f'{w_col}_f{h}'
                # Shift backwards: row at T gets weather from T+h
                
                df[fname] = df.groupby(idx_name)[w_col].shift(-h)
                X_features.append(fname)

    window_size = 24 

    # 1. Get the current FRP (the denominator)
    current_frp = (
        df.groupby(idx_name)[target_col]
        .rolling(window=window_size, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        .clip(lower=1e-6)
    )
    df['max_frp'] = current_frp
    X_features.append('max_frp')

    # 2. Find the maximum FRP in the next 24 hours
    # We use a negative shift to look ahead, then a rolling max
    if model_type == 'ensemble':
        daily_max = df.groupby([idx_name, df['time'].dt.date])[target_col].max()
        tomorrow_date = df['time'].dt.date + pd.Timedelta(days=1)
        future_frp = pd.MultiIndex.from_arrays([df[idx_name], tomorrow_date]).map(daily_max)
        future_frp = future_frp.fillna(1e-6).clip(lower=1e-6)
    else:
        future_frp = (
            df.groupby(idx_name)[target_col]
            .shift(-window_size)                          # Start the window from T+1
            .rolling(window=window_size, min_periods=1)   # Look across the 24h block
            .max()
            .clip(lower=1e-6)
        )

    # 3. Create the new target: Max Growth over the next day
    df['y_max_growth_24h'] = future_frp / current_frp

    y_cols = ['y_max_growth_24h']

    # Drop NaNs and fill label NaNs with epsilon to preserve end-of-fire data
    df[y_cols] = df[y_cols].fillna(1e-6)
    df_clean = df.dropna(subset=X_features)
    
    return df_clean, X_features, y_cols

def process_features(features_csv, seed = 42, target_name = 'rave_FRP_MEAN', 
                    train_split = 0.8, stratify_by = 'n_days', 
                    lookback_days = 2, idx_name = 'cp', end_pad = 1,
                    pred_growth = True, pred_days = 1, keep_dtime = False, 
                    dry_run_cps = [], min_FRP = None, max_FRP = None, scale = "linear",
                    model_type = 'ensemble'):
    df = pd.read_csv(features_csv)
    good_cps = []
    for cp_idx in df.cp.unique():
        cp_df = df[df.cp == cp_idx]
        if not (cp_df[hrrr_features].isna().all().all() \
                or cp_df[rave_features].isna().all().all()):
            good_cps.append(cp_idx)
    df_filtered = df[df.cp.isin(good_cps)]
    train_X, train_y, test_X, test_y = \
            split_data_vectorized(
                    df_filtered, all_features, 
                    target_name = target_name, train_split = train_split,
                    stratify_by = stratify_by, lookback_days = lookback_days,
                    idx_name = idx_name, scale = scale, seed = seed,
                    pred_days = pred_days, dry_run_cps = dry_run_cps, 
                    model_type = model_type)
    
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

def get_fire_growth_metrics(df, idx_name='cp', target_name='rave_FRP_MEAN'):
    # Group by fire (cp) and find the peak vs. average
    fire_stats = df.groupby(idx_name)[target_name].agg(['max', 'mean', 'std']).reset_index()
    
    # Define 'Growth Rate' 
    df_sorted = df.sort_values([idx_name, 'time'])
    df_sorted['delta'] = df_sorted.groupby(idx_name)[target_name].diff()
    
    max_delta = df_sorted.groupby(idx_name)['delta'].max().reset_index()
    max_delta.columns = [idx_name, 'max_growth_rate']
    
    # Merge back to get a single row per fire
    fire_profile = pd.merge(fire_stats, max_delta, on=idx_name)
    zeros = fire_profile[fire_profile['max_growth_rate'] == 0].copy()
    actives = fire_profile[fire_profile['max_growth_rate'] > 0].copy()

    # Bin only the active ones
    zeros['growth_bucket'] = 'Zero'
    actives['growth_bucket'] = pd.qcut(actives['max_growth_rate'], q=3, labels=['Low', 'Medium', 'High'])

    # Combine back
    fire_profile = pd.concat([zeros, actives])
    
    return fire_profile

def data_normalization(df, idx_name = 'cp', seed = 42, epsilon =0.001, df_min_max = pd.DataFrame()):
    rng = np.random.default_rng(seed = seed)
    frp_cols = ["rave_FRP_MEAN", "rave_FRP_SD"]
    
    # Calculate global FRP min/max first (to ensure consistency across the 3 cols)
    # Using your logic: min/max defined by rave_FRP_MEAN if not already set
    if 'rave_FRP_MEAN' in df_min_max.columns:
        global_min_frp, global_max_frp = df_min_max['rave_FRP_MEAN']['min'], df_min_max['rave_FRP_MEAN']['max']
    else:
        global_min_frp = df['rave_FRP_MEAN'].min()
        global_max_frp = df['rave_FRP_MEAN'].max()
    print(f"global frp min/max: {global_min_frp}/{global_max_frp}")

    for col in df.columns:
        if col in [idx_name, "time", "day", "date"]:
            continue
        
        col_mean = df[col].mean(skipna=True)
        col_std = df[col].std(skipna=True)
        na_mask = df[col].isna()

        if col in frp_cols:
            # Impute FRP with 0 and use global FRP scaling
            df.loc[na_mask, col] = 0
            df[col] = (df[col] - global_min_frp) / (global_max_frp - global_min_frp) + epsilon
        else:
            # Impute weather with random Gaussian noise
            if na_mask.any():
                rand_vals = rng.normal(loc=col_mean, scale=col_std, size=na_mask.sum())
                df.loc[na_mask, col] = rand_vals
            
            # Standard Min-Max Scaling for the specific column
            if col in df_min_max.columns:
                c_min, c_max = df_min_max[col]['min'], df_min_max[col]['max']
            else:
                c_min, c_max = df[col].min(), df[col].max()
            if c_max == c_min:
                continue
            df[col] = (df[col] - c_min) / (c_max - c_min)
    return df

def split_data_vectorized(df, feature_names, target_name='rave_FRP_MEAN', 
                          train_split=0.8, stratify_by = 'growth_bucket', scale="linear",
                          lookback_days=2, idx_name='cp', pred_days=1, daily_hrs = 24,
                          dry_run_cps = [], seed = 42, epsilon = 0.001, model_type = 'ensemble'):
    df_clean, X_features, y_cols = generate_diurnal_features(df, target_name, n_future=24, idx_name=idx_name, 
                              daily_freq = daily_hrs, lookback = lookback_days, model_type = model_type)
    print(f"finished data cleaning and featurization")
    fire_profiles = get_fire_growth_metrics(df_clean)
    train_ids = []
    for bucket in fire_profiles[stratify_by].unique():
        subset = fire_profiles[fire_profiles[stratify_by] == bucket][idx_name].tolist()
        k = int(len(subset) * train_split)
        train_ids.extend(sample(subset, k=k))
    
    train_set = set(train_ids) - set(dry_run_cps)
    train_ids = list(train_set)

    train_mask = df_clean[idx_name].isin(train_ids)
    
    # Split into final dataframes
    cols_to_keep = X_features + [idx_name, 'time']
    
    train_data = df_clean.loc[train_mask, cols_to_keep]
    train_labels = df_clean.loc[train_mask, y_cols]
    
    test_data = df_clean.loc[~train_mask, cols_to_keep]
    test_labels = df_clean.loc[~train_mask, y_cols]
    print("labels created")
    return train_data, train_labels, test_data, test_labels

def add_fire_day_col(df : pd.DataFrame, idx_name = 'idx', date_name = 'day', day_count_name = 'day_num'):
    df_out = df.sort_values([idx_name, date_name]).copy()
    
    df_out[day_count_name] = df_out.groupby(idx_name).cumcount() + 1
    return df_out