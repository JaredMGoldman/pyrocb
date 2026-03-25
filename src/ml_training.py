import numpy as np
import os
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from utils.utils import save_model, load_model, \
                        FEATURE_OUTPUT_DIR, ML_FEATS_DIR, \
                        PLOTS_DIR

from utils.feature_creation import process_features
from ml_eval import plot_importances, plot_correlation
import argparse

model_dict = {
    'rand_forest' : RandomForestQuantileRegressor,
    'rf' : RandomForestQuantileRegressor,
    'ols' : LinearRegression,
    'least_sq' : LinearRegression
}

kwargs_dict  = {
    RandomForestQuantileRegressor : {'n_estimators' : 100,
                                    'random_state' : 42,
                                    'max_depth' : 10,
                                    'criterion' : 'squared_error'},
    LinearRegression : {}
}

parser = argparse.ArgumentParser(
                    prog='MachineLearning',
                    description='run ml training and eval script')

parser.add_argument('--pred', action = 'store_true', dest = 'pred_bool', default = False,
                    help = "predict FRP growth rather than straight FRP")
parser.add_argument('--pred_days', type = int, dest = 'pred_days', default = 1,
                    help = "number of days for prediction")
parser.add_argument('--model', type= str, choices = model_dict.keys(), default='rf',
                    help= "select from differnt sklearn model types available")
parser.add_argument('-n', '--name', dest='name', type = str, default='my_model',
                    help = "name of the model to save, informs file naming conventions")
parser.add_argument('-d', '--data', type = str, dest = 'data', 
                    default = "average_polygon_features_parallel.csv",
                    help = f"path to data file relative to {FEATURE_OUTPUT_DIR}")
parser.add_argument('-p', '--plot-dir', type = str, dest = 'plot', 
                    default = "",
                    help = f"path to plotting location relative to {PLOTS_DIR}")
parser.add_argument('--eval', action = 'store_true', dest = 'eval', default= False,
                    help = "evaluate the model")

def train_regressor(X : pd.DataFrame, y : pd.DataFrame, 
                    X_test : pd.DataFrame, y_test : pd.DataFrame,
                    model_class = RandomForestRegressor,
                    model_fname = "initial_regressor",
                    save_bool = True,
                    model_kwargs = {}, 
                    num_epochs = 1, num_splits = 3, shuffle = True, 
                    rand_state = 42, drop_vars = []):
    X = X.drop(drop_vars, axis = 1)
    X_test = X_test.drop(drop_vars, axis = 1)
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
                'epoch' : epoch+1,
                'fold' : fold+1,
                'rmse' : rmse,
                'r2' : r2
            })
            print(f"Epoch {epoch+1} | Fold {fold} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

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
    if save_bool:
        save_model(model, model_fname)
    return model, fold_metrics, test_metrics

def main(data_fname, model_name, 
        drop_vars = ["idx", "n_days"], 
        plot_dir = PLOTS_DIR,
        model_class = RandomForestRegressor, 
        pred_growth = True,
        pred_days = 1, eval_model = False):   
    train_data, train_labels, test_data, test_labels = process_features(data_fname, 
                                                                        pred_growth = pred_growth,
                                                                        pred_days = pred_days)
    [df.to_csv(os.path.join(ML_FEATS_DIR, fname), index = False) for df, fname in 
        zip([train_data, train_labels, test_data, test_labels],
            [f"train_data_{model_name}.csv", f"train_labels_{model_name}.csv", 
                f"test_data_{model_name}.csv", f"test_labels_{model_name}.csv"])]
    drop_vars = ["idx", "n_days"]
    if not eval_model:
        model, _, _ = train_regressor(train_data, train_labels, 
                                                            test_data, test_labels, 
                                                            model_fname = model_name, 
                                                            save_bool = True, 
                                                            drop_vars = drop_vars,
                                                            model_class = model_class)
    else:
        model = load_model(model_name)
    train_dataset = (train_data, train_labels)
    test_dataset = (test_data, test_labels)
    plot_importances(model, model_name, out_dir = plot_dir)
    plot_correlation(model, train_dataset, test_dataset, 
                        drop_vars = drop_vars, exp_name = model_name,
                        out_dir = plot_dir)

if __name__ == "__main__":
    args = parser.parse_args()

    features_fname = os.path.join(FEATURE_OUTPUT_DIR, args.data)
    model_name = args.name
    plot_dir = f"{PLOTS_DIR}/{args.plot}"
    model_class = model_dict[args.model]
    pred_days = args.pred_days
    pred_growth = args.pred_bool

    eval_model = args.eval

    main(features_fname, model_name, 
         plot_dir = plot_dir, 
         model_class = model_class,
         pred_growth = pred_growth,
         pred_days = pred_days,
         eval_model = eval_model)
    
    