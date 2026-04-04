
import argparse
from boruta import BorutaPy
import numpy as np
import os
import pandas as pd
import shap

from quantile_forest import RandomForestQuantileRegressor

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV

from tqdm import tqdm

from utils.utils import save_model, load_model, \
                        save_features, load_features, \
                        save_feature_names, load_feature_names
from utils.constants import FEATURE_OUTPUT_DIR, PLOTS_DIR, MODELS_DIR, dry_run_cps
from utils.feature_creation import process_features
from ml_eval import plot_importances, plot_correlation
from fire_comparison import plot_model_results

model_dict = {
    'quant_forest' : RandomForestQuantileRegressor,
    'qrf' : RandomForestQuantileRegressor,
    'rf'   : RandomForestRegressor,
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

scale_types = ['linear', 'log', 'abs']

elim_types = ['rfecv', 'shap', 'boruta', 'noise']

parser = argparse.ArgumentParser(
                    prog='MachineLearning',
                    description='run ml training and eval script')

parser.add_argument('--pred', action = 'store_true', dest = 'pred_bool', default = False,
                    help = "predict FRP growth rather than straight FRP")
parser.add_argument('-dr','--dry', action = 'store_true', dest = 'dry_run', default = False,
                    help = "run dry run analysis of fires")
parser.add_argument('--pred_days', type = int, dest = 'pred_days', default = 1,
                    help = "number of days for prediction")
parser.add_argument('--model', type= str, choices = model_dict.keys(), default='rf',
                    help= "select from differnt sklearn model types available")
parser.add_argument('--scale', type= str, choices = scale_types, default='linear',
                    help= f"type of scaling to use for growth value from {scale_types}")
parser.add_argument('--elim-type', type= str, dest = 'feat_elim', choices = elim_types, 
                    default='rfecv', help= f"type of feature elimination to use from {elim_types}")
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


def shap_feature_selection(X, y, model, threshold=0.01, weights_offset = 1):
    """
    Fits a model, calculates SHAP importance, and returns 
    a list of features that contribute more than the threshold.
    """
    model.fit(X, y, sample_weight = sample_weights(y, weights_offset))
    def _pred_model(X):
        return model.predict(X)
    
    explainer = shap.KernelExplainer(_pred_model, X)
    shap_values = explainer.shap_values(X)

    feature_importance = np.abs(shap_values).mean(axis=0)
    
    shap_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values(by='importance', ascending=False)

    selected_features = shap_df[shap_df['importance'] > threshold]['feature'].tolist()
    
    return selected_features, shap_df

def sample_weights(y, offset = 1):
    return np.abs(y)*offset

def train_regressor(X : pd.DataFrame, y : pd.DataFrame, 
                    X_test : pd.DataFrame, y_test : pd.DataFrame,
                    model_class = RandomForestRegressor,
                    out_dir = f"{MODELS_DIR}",
                    model_name = "initial_regressor",
                    model_kwargs = {}, 
                    num_epochs = 1, num_splits = 3, shuffle = True, 
                    rand_state = 42, drop_vars = [], feat_elim = 'shap', 
                    n_iters = 5, weights_offset = 2):
    X = X.drop(drop_vars, axis = 1)
    X_test = X_test.drop(drop_vars, axis = 1)
    kf = KFold(n_splits=num_splits, shuffle = shuffle, random_state = rand_state)
    fold_metrics = []
    feature_cols = []
    if feat_elim == 'rfecv':
        model = RFECV(model_class(**model_kwargs), step=1, cv=5, min_features_to_select=5)
        y = np.ravel(y)
        model = model.fit(X, y, sample_weight = sample_weights(y, weights_offset))
        print("Optimized Features:", X.columns[model.support_])
        feature_cols = X.columns[model.support_]
        model = model.estimator_
    elif feat_elim == 'noise':
        model = model_class(**model_kwargs)
        X['noise'] = np.random.rand(X.shape[0])
        feature_cols = set(X.columns)
        for i in tqdm(range(n_iters), total = num_splits, desc = "identifying features by noise threshold"):
            fold_model = clone(model)
            X_train = X[list(feature_cols)]
            y_train = np.ravel(y)
            fold_model.fit(X_train, y_train, sample_weight = sample_weights(y_train, weights_offset))
            feature_importances = sorted({feat_name : feat_val for feat_name, feat_val in zip(fold_model.feature_names_in_, 
                                                                                        fold_model.feature_importances_)})
            these_cols = set(feature_importances[:feature_importances.index('noise')])
            feature_cols = feature_cols & these_cols | set(['noise'])
            print(f"columns at fold {i+1}: {these_cols}\n\tall cols: {feature_cols}\n")
        feature_cols = list(feature_cols)
        X = X[feature_cols]
        for fold, (train_idxs, val_idxs) in tqdm(enumerate(kf.split(X)), total = num_splits,
                                                desc = "training noise feature reduction model"):
            X_train = X.iloc[train_idxs]
            y_train = np.ravel(y.iloc[train_idxs])
            model.fit(X_train, y_train, sample_weight=sample_weights(y_train, offset = weights_offset))
            
    elif feat_elim == 'boruta':
        print('starting boruta feature elimination')
        feat_selector = BorutaPy(model_class(**model_kwargs), 
                                 n_estimators = 'auto', 
                                 random_state = rand_state,
                                 max_iter = 10,
                                 verbose = 2)
        y = np.ravel(y)
        feat_selector.fit(X, y, sample_weight = sample_weights(y, weights_offset))
        feature_cols = X.columns[feat_selector.support_weak_]
        model = model_class(**model_kwargs)
        for fold, (train_idxs, val_idxs) in enumerate(kf.split(X)):
            X_train = X.iloc[train_idxs]
            y_train = np.ravel(y.iloc[train_idxs])
            model.fit(X_train[feature_cols], np.ravel(y_train), sample_weight = sample_weights(y_train, weights_offset))
    else:
        model = model_class(**model_kwargs)
        for epoch in range(num_epochs):
            fold_features = set(X.columns)
            for fold, (train_idxs, val_idxs) in enumerate(kf.split(X)):
                if feat_elim == 'shap':
                    model = model_class(**model_kwargs)

                    X_train = X.iloc[train_idxs]
                    y_train = np.ravel(y.iloc[train_idxs])

                    important_cols, _ = shap_feature_selection(X_train, y_train, model, threshold=0.01)
                    
                    fold_model = model_class(**model_kwargs)
                    fold_model.fit(X_train[important_cols], y_train, sample_weight = sample_weights(y_train, weights_offset))
                    
                    X_val = X.iloc[val_idxs][important_cols]
                    y_val = y.iloc[val_idxs]

                    preds = fold_model.predict(X_val)

                    rmse = np.sqrt(mean_squared_error(y_val, preds))
                    r2 = r2_score(y_val, preds)
                    fold_metrics.append({
                        'epoch' : epoch+1,
                        'fold' : fold+1,
                        'rmse' : rmse,
                        'r2' : r2
                    })
                    print(f"Epoch {epoch+1} | Fold {fold} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
                    print(f"important_cols: {important_cols}\n")
                    fold_features = fold_features & set(important_cols)
        
        feature_cols = list(fold_features)
        fold_metrics = pd.DataFrame(fold_metrics)
        model = model_class(**model_kwargs)

        print("\nCross-validation summary")
        print(fold_metrics.describe())

    model.fit(X[feature_cols], np.ravel(y), sample_weight = sample_weights(y, weights_offset))

    test_preds = model.predict(X_test[feature_cols])
    y_test = np.ravel(y_test)

    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2 = r2_score(y_test, test_preds)

    test_metrics = {
        "rmse": [test_rmse],
        "r2": [test_r2]
    }

    print("\nTest performance")
    print(test_metrics)

    save_model(model, out_dir, model_name)
    pd.DataFrame(test_metrics).to_csv(f"{out_dir}/test_metrics.csv", index = False)
    save_feature_names(feature_cols, out_dir, model_name)
    return model, feature_cols

def main(data_fname, model_name, 
        drop_vars = ["idx", "n_days"], 
        model_class = RandomForestRegressor, 
        pred_growth = True,
        pred_days = 1, 
        eval_model = False,
        scale = "linear",
        feat_elim = 'rfecv',
        dry_run_bool = True):
    
    out_dir = f"{MODELS_DIR}/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    drop_vars = ["idx", "n_days", "day"]

    if not eval_model:
        train_data, train_labels, test_data, test_labels = process_features(data_fname, 
                                                                            pred_growth = pred_growth,
                                                                            pred_days = pred_days,
                                                                            dry_run_cps = dry_run_cps, 
                                                                            scale = scale)

        model, feature_names = train_regressor( train_data, train_labels, 
                                                test_data, test_labels, 
                                                model_name = model_name, 
                                                drop_vars = drop_vars,
                                                model_class = model_class,
                                                feat_elim = feat_elim, 
                                                out_dir = out_dir )

        save_features(train_data, train_labels, 
                      test_data,  test_labels,
                      out_dir, model_name)
    else:
        model = load_model(f"{out_dir}/{model_name}")
        train_data, train_labels, test_data, test_labels = load_features(out_dir, model_name)
        feature_names = load_feature_names(f"{out_dir}/{model_name}_features.txt")
        train_data = train_data
        test_data = test_data

    train_dataset = (train_data, train_labels)
    test_dataset = (test_data, test_labels)

    plots_dir = f"{out_dir}/plots"
    os.makedirs(plots_dir, exist_ok = True)
    plot_importances(model, model_name, plots_dir, feature_names)
    plot_correlation(model, train_dataset, test_dataset, 
                        exp_name = model_name,
                        out_dir = plots_dir,
                        feature_names = feature_names)
    if dry_run_bool:
        X, y = test_dataset
        dry_run_plots_dir = os.path.join(out_dir, "dry_run_plots")
        os.makedirs(dry_run_plots_dir, exist_ok = True)
        dry_run_stats = {'rmse' : [],
                         'cp' : []}
        for cp in dry_run_cps:
            idxs = X.loc[X['idx'] == cp].index
            if len(idxs) == 0:
                continue
            
            rmse = plot_model_results(model, X = X.iloc[idxs], y = y.iloc[idxs], 
                               method_name = model_name, fire_label = str(cp), 
                               min_FRP = -1.0173882246017456, max_FRP = 17615.2, 
                               pred_type = scale, pred_days = pred_days, 
                               out_dir = dry_run_plots_dir, 
                               feature_names = feature_names)
            dry_run_stats['rmse'].append(rmse)
            dry_run_stats['cp'].append(cp)

        df = pd.DataFrame(dry_run_stats)
        df.to_csv(f"{out_dir}/dry_run_stats.csv", index = False)
        dr_rmse = np.mean(df['rmse'])
        with open(f"{out_dir}/dry_run_mean_rmse.txt", "w") as f:
            f.write(f"{dr_rmse}")
    

if __name__ == "__main__":
    args = parser.parse_args()

    features_fname = os.path.join(FEATURE_OUTPUT_DIR, args.data)
    model_name = args.name
    model_class = model_dict[args.model]
    pred_days = args.pred_days
    pred_growth = args.pred_bool

    eval_model = args.eval
    scale = args.scale
    feat_elim = args.feat_elim
    dry_run_bool = args.dry_run

    main(features_fname, model_name, 
         model_class = model_class,
         pred_growth = pred_growth,
         pred_days = pred_days,
         eval_model = eval_model,
         scale = scale,
         feat_elim = feat_elim,
         dry_run_bool = dry_run_bool)