from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import pandas as pd

from quantile_forest import RandomForestQuantileRegressor
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample

from tqdm import tqdm

from utils.io_utils import PLOTS_DIR

def plot_importances(model, exp_name, out_dir, feature_names):
    importance = None
    
    if not 'feature_importances_' in dir(model):
        importance = np.abs(model.coef_)
    else:
        importance = model.feature_importances_
        
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    df = df.sort_values("importance", ascending=False)

    plt.figure(figsize=(8,6))
    plt.barh(df["feature"], df["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Feature Importance")
    plt.title("Model Feature Importance")
    os.makedirs(f"{out_dir}", exist_ok = True)
    plt.savefig(f"{out_dir}/{exp_name}_feature_importances.png")
    plt.close()
    print(f"saved to {out_dir}/{exp_name}_feature_importances.png")

def plot_correlation(model, training_data, testing_data, exp_name = 'regression',  
                    target = "rave_FRP_MEAN", stratify_by = 'n_days',
                    out_dir = PLOTS_DIR, feature_names = []):
    X_train, y_train = training_data
    X_test, y_test = testing_data

    if type(X_train) is pd.DataFrame:
        X_train = X_train.reset_index()
        y_train = y_train.reset_index()
        X_test = X_test.reset_index()
        y_test = y_test.reset_index()

    train_pred = model.predict(X_train[feature_names])
    test_pred = model.predict(X_test[feature_names])
    import ipdb; ipdb.set_trace()
    bucket_stats = {str(bucket) : {'train_rmse' : 0 ,
                                    'test_rmse' : 0,
                                    'train_r2' : 0,
                                    'test_r2' : 0} for bucket in np.unique(X_train[stratify_by])}

    for bucket in np.unique(X_train[stratify_by]):
        bucket_locs = X_train.loc[X_train[stratify_by] == bucket].index
        bucket_idxs = [X_train.index.get_loc(loc) for loc in bucket_locs]
        y_train_bkt = np.squeeze(y_train.iloc[bucket_idxs][target])
        train_pred_bkt = train_pred[bucket_idxs]

        bucket_stats[str(bucket)]['train_rmse'] = np.sqrt(mean_squared_error(y_train_bkt, train_pred_bkt))
        bucket_stats[str(bucket)]['train_ci'] = bootstrap_intervals(model, X_train.iloc[bucket_locs], y_train_bkt, 
                                                                      X_test.loc[X_test[stratify_by] == bucket], 
                                                                      feature_names = feature_names)
        bucket_stats[str(bucket)]['train_r2'] = r2_score(y_train_bkt, train_pred_bkt)

    for bucket in np.unique(X_test[stratify_by]):
        if not str(bucket) in bucket_stats.keys():
            continue
        bucket_locs = X_test.loc[X_test[stratify_by] == bucket].index
        bucket_idxs = [X_test.index.get_loc(loc) for loc in bucket_locs]
        if len(bucket_idxs) == 0:
            continue
        y_test_bkt = np.squeeze(y_test.iloc[bucket_idxs][target])
        test_pred_bkt = test_pred[bucket_idxs]
        bucket_stats[str(bucket)]['test_rmse'] = np.sqrt(mean_squared_error(y_test_bkt, test_pred_bkt))
        bucket_stats[str(bucket)]['test_r2'] = r2_score(y_test_bkt, test_pred_bkt)

    
    buckets = sorted([int(buck) for buck in bucket_stats.keys()])

    rmse_test = [bucket_stats[str(bucket)]['test_rmse'] for bucket in buckets]
    rmse_train = [bucket_stats[str(bucket)]['train_rmse'] for bucket in buckets]

    r2_test = [bucket_stats[str(bucket)]['test_r2'] for bucket in buckets]
    r2_train = [bucket_stats[str(bucket)]['train_r2'] for bucket in buckets]

    ci_train = [bucket_stats[str(bucket)]['train_ci'] for bucket in buckets]

    if stratify_by == 'n_days':
        buckets = np.array(buckets) - 2

    plt.title("RMSE by Day")
    plt.xlabel("Duration")
    plt.ylabel("RMSE")
    plt.errorbar(buckets, rmse_train, label = 'train rmse', yerr=ci_train)
    plt.errorbar(buckets, rmse_test, label = 'test rmse', yerr=ci_train)
    plt.legend()
    plt.ylim((-0.5,1))
    os.makedirs(f"{out_dir}", exist_ok = True)
    plt.savefig(f"{out_dir}/{exp_name}_rmse.png")
    plt.close()
    
    print(f"rmse stats saved to {out_dir}/{exp_name}_rmse.png")

    plt.title(r"$R^2$ by Day")
    plt.xlabel("Duration")
    plt.ylabel(r"$R^2$")
    plt.plot(buckets, r2_train, label = r'train $r^2$')
    plt.plot(buckets, r2_test, label = r'test $r^2$')
    plt.ylim((0,1))
    plt.legend()

    plt.savefig(f"{out_dir}/{exp_name}_r2.png")
    plt.close()
    print(f"r2 stats saved to {out_dir}/{exp_name}_r2.png")

    return bucket_stats



def _bootstrap_worker(model, X, y, feature_names, X_test):
    """
    Worker function: Handles a single bootstrap iteration.
    Cloning inside the worker ensures a clean state for every process.
    """
    this_model = clone(model)
    
    # Resample the data
    X_resample, y_resample = resample(X[feature_names], y)
    
    # Fit and Predict
    this_model.fit(X_resample, y_resample)
    pred_mean = np.mean(this_model.predict(X_test[feature_names]))
    
    return pred_mean

def run_parallel_bootstrap(model, X_train, y_train, X_test, feature_names, n_iterations=1000, max_workers=None):
    # Use 'spawn' to avoid issues with library locks (like OpenBLAS/MKL)
    ctx = mp.get_context('spawn')
    
    results = []
    
    # Use ProcessPoolExecutor for CPU-bound tasks (model fitting)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        # Submit all tasks
        futures = [
            executor.submit(_bootstrap_worker, model, X_train, y_train, feature_names, X_test) 
            for _ in range(n_iterations)
        ]
        
        # Collect results as they complete
        for future in tqdm(as_completed(futures), total = len(futures), desc = "bootrapping model in parallel"):
            results.append(future.result())
            
    return np.array(results)

def bootstrap_intervals(model, X_train, y_train, X_test, n_iterations=1000, feature_names = [], max_workers = 40):
    all_preds = []
    if len(X_test) == 0:
        return 0

    if type(model) is RandomForestQuantileRegressor:
        X_test = X_test[feature_names]
        intervals = model.predict(X_test, quantiles=[0.025, 0.5, 0.975])

        lower_val = intervals[:, 0]
        upper_val = intervals[:, 2]
        return np.mean(upper_val - lower_val)
    
    all_preds = run_parallel_bootstrap(model, X_train, y_train, X_test, feature_names, n_iterations, max_workers)
    
    lower_bound = np.percentile(all_preds, 2.5, axis=0)
    upper_bound = np.percentile(all_preds, 97.5, axis=0)
    uncertainty = upper_bound - lower_bound
    
    return uncertainty