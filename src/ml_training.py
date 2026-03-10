import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from plotting import plot_importances
from utils.utils import save_model
from random import sample

hrrr_features = [ "hrrr_dpt","hrrr_u","hrrr_v","hrrr_t", \
                "hrrr_rh","hrrr_tp","hrrr_mstav","hrrr_sdwe"]
rave_features = ["rave_FRP_MEAN","rave_FRP_SD"]

all_features = ["esi_DFPPM", "modis_MaxFRP","rave_FRP_MEAN","rave_FRP_SD",
                "hrrr_dpt","hrrr_u","hrrr_v","hrrr_t",
                "hrrr_rh","hrrr_tp","hrrr_mstav","hrrr_sdwe"]

thresh_inc = 0.18 #, scaling factor of 1.5
thresh_dec = -0.3 # scaling factor of 1.5
feature_set_names = list(feature_subsets.keys())

def train_regressor(X : pd.DataFrame, y : pd.DataFrame, 
                    X_test : pd.DataFrame, y_test : pd.DataFrame,
                    model_class = RandomForestRegressor,
                    model_kwargs = {'n_estimators' : 100,
                                    'random_state' : 42,
                                    'criterion' : 'squared_error'}, 
                    num_epochs = 10, num_splits = 5, shuffle = True, 
                    rand_state = 42):
    kf = KFold(n_splits=num_splits, shuffle = shuffle, random_state = rand_state)
    model = model_class(**model_kwargs)
    fold_metrics = []
    for epoch in range(num_epochs):
        for fold, (train_idxs, val_idxs) in enumerate(kf.split(X)):
            X_train = X.iloc[train_idxs]
            y_train = np.ravel(y.iloc[train_idxs])
            model = model.fit(X_train, y_train)
            
            X_val = X.iloc[val_idxs]
            y_val = y.iloc[val_idxs]

            preds = model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, preds))
            r2 = r2_score(y_val, preds)
            fold_metrics.append({
                'epoch' : epoch,
                'fold' : fold,
                'rmse' : rmse,
                'r2' : r2
            })
            print(f"Epoch {epoch} | Fold {fold} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

    fold_metrics = pd.DataFrame(fold_metrics)

    print("\nCross-validation summary")
    print(fold_metrics.describe())

    # train final model on all training data
    model.fit(X, np.ravel(y))

    test_preds = model.predict(X_test)
    y_test = np.ravel(y_test)

    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2 = r2_score(y_test, test_preds)

    test_metrics = {
        "rmse": test_rmse,
        "r2": test_r2
    }

    print("\nTest performance")
    print(test_metrics)

    return model, fold_metrics, test_metrics

def process_features(features_csv, seed = 42):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(features_csv)
    good_cps = []
    for cp_idx in df.cp.unique():
        if not (df[hrrr_features].isna().all().all() or df[rave_features].isna().all().all()):
            good_cps.append(cp_idx)
    df_filtered = df[df.cp.isin( good_cps)]
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

            normalized = (df_filtered[col] - np.min(df_filtered[col]))/(np.max(df_filtered[col])- np.min(df_filtered[col]))
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

    train_X, train_y, test_X, test_y = split_data(df_filtered, all_features)
    print("completed data processing")
    return train_X, train_y, test_X, test_y

def split_data(df, feature_names, target_name = 'rave_FRP_MEAN', 
                    train_split = 0.8, stratify_by = 'n_days', 
                    lookback_days = 1, idx_name = 'cp', end_pad = 1):
    # split data into train and test subsetted by fire duration
    # data are feature labels for previous two days (yesterday and today)
    train_idx = []
    for bucket in df[stratify_by].unique():
        bucket_idxs = df[df[stratify_by] == bucket][idx_name].unique()
        train_idx.extend(sample(list(bucket_idxs), k = int(len(bucket_idxs)*train_split)))
    
    # test_idx = set(list(df[idx_name].unique())) - set(train_idx)

    train_data = []
    test_data = []

    train_labels = []
    test_labels = []
    for idx in df[idx_name].unique():
        this_data = df[df[idx_name] == idx]
        days = sorted(this_data.day.unique())
        for day_i in range(lookback_days, len(days)-end_pad-1):
            day_dict = {}
            for day_j in range(lookback_days + 1):
                data_j = this_data[this_data.day == days[day_i - day_j]][feature_names]
                for name in feature_names:
                    day_dict[f"{name}_d{day_j}"] = np.squeeze(data_j[name])

            label = np.squeeze(this_data[this_data.day == days[day_i+1]][target_name])
            if idx in train_idx:
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

if __name__ == "__main__":
    features_fname = "/home/jaredgoldman/dev/pyrocb/outputs/features/data_gen_subset.csv"
    train_data, train_labels, test_data, test_labels = process_features(features_fname)
    model, fold_metrics, test_metrics = train_regressor(train_data,train_labels, 
                                                        test_data, test_labels)
    
    # feature_sets = ['features_no_persistence']
    # target = "log_Scaling_Factor"
    # for feature_set in feature_sets:
    #     model = train_model(feature_set, save_model=True, target = target)
    #     eval_model(model, feature_set, target = target)