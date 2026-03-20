import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score

from utils.utils import PLOTS_DIR

def plot_importances(model, exp_name, out_dir):
    importance = None
    if not 'feature_importances_' in dir(model):
        importance = np.abs(model.coef_)
    else:
        importance = model.feature_importances_
    feature_names = model.feature_names_in_
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
                    target = "rave_FRP_MEAN", stratify_by = 'n_days', drop_vars = [],
                    out_dir = PLOTS_DIR):
    X_train, y_train = training_data
    X_test, y_test = testing_data

    if type(X_train) is pd.DataFrame:
        X_train = X_train.reset_index()
        y_train = y_train.reset_index()
        X_test = X_test.reset_index()
        y_test = y_test.reset_index()

    drop_vars.append('index')
    train_pred = model.predict(X_train.drop(drop_vars, axis = 1))
    test_pred = model.predict(X_test.drop(drop_vars, axis = 1))

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
        bucket_stats[str(bucket)]['train_r2'] = r2_score(y_train_bkt, train_pred_bkt)

    for bucket in np.unique(X_test[stratify_by]):
        bucket_locs = X_test.loc[X_train[stratify_by] == bucket].index
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
    
    if stratify_by == 'n_days':
        buckets = np.array(buckets) - 2

    plt.title("RMSE by Day")
    plt.xlabel("Duration")
    plt.ylabel("RMSE")
    plt.plot(buckets, rmse_train, label = 'train rmse')
    plt.plot(buckets, rmse_test, label = 'test rmse')
    plt.legend()

    os.makedirs(f"{out_dir}", exist_ok = True)
    plt.savefig(f"{out_dir}/{exp_name}_rmse.png")
    plt.close()
    
    print(f"rmse stats saved to {out_dir}/{exp_name}_rmse.png")

    plt.title(r"$R^2$ by Day")
    plt.xlabel("Duration")
    plt.ylabel(r"$R^2$")
    plt.plot(buckets, r2_train, label = r'train $r^2$')
    plt.plot(buckets, r2_test, label = r'test $r^2$')
    plt.legend()

    plt.savefig(f"{out_dir}/{exp_name}_r2.png")
    plt.close()
    print(f"r2 stats saved to {out_dir}/{exp_name}_r2.png")