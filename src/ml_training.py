
import argparse
from boruta import BorutaPy
import numpy as np
import optuna
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

from utils.io_utils import save_model, load_model, \
                        save_features, load_features, \
                        save_feature_names, load_feature_names
from utils.constants import FEATURE_OUTPUT_DIR, PLOTS_DIR, MODELS_DIR, dry_run_cps, dry_run_date
from utils.feature_creation import process_features, sample_weights
from analysis.ml_eval import plot_importances, plot_correlation
from analysis.fire_comparison import plot_model_results, dry_run_results   
from models.sequential_rf import SequentialRF 
from models.simple_rf import SimpleRF 
from models.simple_dnn import SimpleDNN
from models.simple_mdn import SimpleMDN
from models.simple_lstm import SimpleLSTM
from models.conditional_dnn import ConditionalDNN
from models.smart_lstm import SmartLSTM

import warnings

model_dict = {
    'quant_forest' : RandomForestQuantileRegressor,
    'qrf' : RandomForestQuantileRegressor,
    'rf'   : RandomForestRegressor,
    'ols' : LinearRegression,
    'least_sq' : LinearRegression,
    'seq' : SequentialRF,
    'srf' : SimpleRF,
    'dnn' : SimpleDNN,
    'mdn' : SimpleMDN,
    'lstm' : SimpleLSTM,
    'cdnn' : ConditionalDNN,
    'slstm' : SmartLSTM
}

kwargs_dict  = {
    RandomForestQuantileRegressor : {'n_estimators' : 100,
                                    'random_state' : 42,
                                    'max_depth' : 10,
                                    'criterion' : 'squared_error'},
    LinearRegression : {},

    SequentialRF : {'n_estimators' : 100,
                    'max_depth' : 10,
                    'n_jobs' : 40},

    SimpleRF : {'n_estimators' : 100,
                    'max_depth' : 10,
                    'n_jobs' : 40},
    
    SimpleDNN : {   'batch_size' : 164,
                    'epochs' : 10,
                    'lr' : 0.001},
    
    SimpleMDN : {   'batch_size' : 164,
                    'epochs' : 10,
                    'lr' : 0.001},
    
    SimpleLSTM : {   'batch_size' : 164,
                    'epochs' : 10,
                    'lr' : 0.001},

    SmartLSTM : {   'batch_size' : 164,
                    'epochs' : 10,
                    'lr' : 0.001},
    
    ConditionalDNN : {   'batch_size' : 164,
                    'epochs' : 10,
                    'lr' : 0.001},
}

kwargs_dict.update({model_class : {} for model_class in list(set(model_dict.values())) if not model_class in kwargs_dict.keys()})

scale_types = ['linear', 'log', 'abs']

elim_types = ['rfecv', 'shap', 'boruta', 'noise', 'none']

stratify_opts = ['n_days']

weighting_methods = ['prod', 'exp', 'id', 'sum']

parser = argparse.ArgumentParser(
                    prog='MachineLearning',
                    description='run ml training and eval script')

parser.add_argument('--pred', action = 'store_true', dest = 'pred_bool', default = False,
                    help = "predict FRP growth rather than straight FRP")
parser.add_argument('--optuna', action = 'store_true', dest = 'optuna_bool', default = False,
                    help = "run optuna hyperparameter optimization")
parser.add_argument('-dr','--dry', action = 'store_true', dest = 'dry_run', default = False,
                    help = "run dry run analysis of fires")
parser.add_argument('--pred_days', type = int, dest = 'pred_days', default = 1,
                    help = "number of days for prediction")
parser.add_argument('--model', type= str, choices = model_dict.keys(), default='qrf',
                    help= "select from differnt sklearn model types available")
parser.add_argument('--strat', type= str, choices = stratify_opts, default='n_days',
                    help= "stratify training data by given feature value distribution")
parser.add_argument('-wm', '--weight-method', dest = 'wm', type= str, choices = weighting_methods, default='prod',
                    help= "method for assigning weights to training labels")
parser.add_argument('-wf', '--weight-factor', dest = 'wf', type= float, default=1.0,
                    help= "factor for training label weighting")
parser.add_argument('--n_folds', type = int, dest = 'n_folds', default = 3,
                    help = "number of training folds for k-fold cross-validataion")
parser.add_argument('--scale', type= str, choices = scale_types, default='linear',
                    help= f"type of scaling to use for growth value from {scale_types}")
parser.add_argument('--elim-type', type= str, dest = 'feat_elim', choices = elim_types, 
                    default='none', help= f"type of feature elimination to use from {elim_types}")
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


def shap_feature_selection(X, y, model, threshold=0.01, weights_factor = 1, weighting_method = 'prod'):
    """
    Fits a model, calculates SHAP importance, and returns 
    a list of features that contribute more than the threshold.
    """
    model.fit(X, y, sample_weight = sample_weights(y, weights_factor, method = weighting_method))
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


def train_regressor(X : pd.DataFrame, y : pd.DataFrame, 
                    X_test : pd.DataFrame, y_test : pd.DataFrame,
                    model_class = RandomForestRegressor,
                    out_dir = f"{MODELS_DIR}",
                    model_name = "initial_regressor",
                    model_kwargs = {'n_jobs' : 20,
                                    'n_estimators' : 100}, 
                    num_epochs = 1, num_splits = 3, shuffle = True, 
                    rand_state = 42, drop_vars = [], feat_elim = 'shap', 
                    n_iters = 5, weights_factor = 2, weighting_method = 'prod',
                    min_features = 5):
    X = X.drop(drop_vars, axis = 1)
    X_test = X_test.drop(drop_vars, axis = 1)
    kf = KFold(n_splits=num_splits, shuffle = shuffle, random_state = rand_state)
    fold_metrics = []
    feature_cols = []
    if feat_elim == 'none':
        feature_cols = X.columns
        print("running normal training")
        for fold, (train_idxs, val_idxs) in enumerate(kf.split(X)):
            print(f"fold {fold}/{num_splits}")
            model = model_class(**model_kwargs)
            X_train = X.iloc[train_idxs]
            y_train = np.ravel(y.iloc[train_idxs])
            model = model.fit(X, y, sample_weight = sample_weights(y, weights_factor, method = weighting_method))
    elif feat_elim == 'rfecv':
        from sklearn import set_config
        set_config(enable_metadata_routing=True)
        model = model_class(**model_kwargs)
        model.set_fit_request(sample_weight=True)
        model = RFECV(model, step=1, cv=5, min_features_to_select=min_features)
        model = model.fit(X, y, sample_weight = sample_weights(y, weights_factor, method = weighting_method))
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
            fold_model.fit(X_train, y_train, sample_weight = sample_weights(y_train, weights_factor, method = weighting_method))
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
            model.fit(X_train, y_train, sample_weight=sample_weights(y_train, weights_factor, method = weighting_method))
            
    elif feat_elim == 'boruta':
        print('starting boruta feature elimination')
        feat_selector = BorutaPy(model_class(**model_kwargs), 
                                 n_estimators = 'auto', 
                                 random_state = rand_state,
                                 max_iter = 10,
                                 verbose = 2)
        y = np.ravel(y)
        feat_selector.fit(X, y, sample_weight = sample_weights(y, weights_factor, method = weighting_method))
        feature_cols = X.columns[feat_selector.support_weak_]
        model = model_class(**model_kwargs)
        for fold, (train_idxs, val_idxs) in enumerate(kf.split(X)):
            X_train = X.iloc[train_idxs]
            y_train = np.ravel(y.iloc[train_idxs])
            model.fit(X_train[feature_cols], np.ravel(y_train), sample_weight = sample_weights(y_train, weights_factor, method = weighting_method))
    else:
        model = model_class(**model_kwargs)
        for epoch in range(num_epochs):
            fold_features = set(X.columns)
            for fold, (train_idxs, val_idxs) in enumerate(kf.split(X)):
                if feat_elim == 'shap':
                    model = model_class(**model_kwargs)

                    X_train = X.iloc[train_idxs]
                    y_train = np.ravel(y.iloc[train_idxs])

                    important_cols, _ = shap_feature_selection(X_train, y_train, model, threshold = 0.01, 
                                                               weights_factor = weights_factor,
                                                               method = weighting_method)
                    
                    fold_model = model_class(**model_kwargs)
                    fold_model.fit(X_train[important_cols], y_train, sample_weight = sample_weights(y_train, weights_factor, method = weighting_method))
                    
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

    model.fit(X[feature_cols], y, sample_weight = sample_weights(y, weights_factor, method = weighting_method))

    test_preds = model.predict(X_test[feature_cols])

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
        model_kwargs = {},
        pred_growth = True,
        pred_days = 1, 
        eval_model = False,
        scale = "linear",
        feat_elim = 'rfecv',
        dry_run_bool = True,
        stratify_by = 'n_days', 
        weights_factor = 2, 
        weighting_method = 'prod',
        n_folds = 3, optuna_bool = True):
    
    out_dir = f"{MODELS_DIR}/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    drop_vars = ["cp", "time"]
    if optuna_bool:
        train_data, train_labels, test_data, test_labels = process_features(data_fname, 
                                                                        pred_growth = pred_growth,
                                                                        pred_days = pred_days,
                                                                        dry_run_cps = dry_run_cps, 
                                                                        scale = scale,
                                                                        stratify_by = stratify_by)
        best_err = 1000000
    def train_optuna(trial):
        nonlocal best_err
        drop_vars = ["cp", "time"]

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 32, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 20, log=True), 
            "weight_factor": trial.suggest_float("weight_factor", 5, 100),
            "min_features": trial.suggest_int("min_features", 5, 20)
        }

        model, feature_names = train_regressor( train_data, train_labels, 
                                                test_data, test_labels, 
                                                model_name = model_name, 
                                                drop_vars = drop_vars,
                                                model_class = model_class,
                                                model_kwargs= {"n_estimators" : params["n_estimators"],
                                                               "max_depth" : params["max_depth"], 
                                                               "min_samples_leaf": params["min_samples_leaf"]},
                                                feat_elim = feat_elim, 
                                                out_dir = out_dir,
                                                weights_factor = params['weight_factor'],
                                                weighting_method = weighting_method,
                                                n_iters = n_folds, min_features = params['min_features'])
        
        dr_mrmse = dry_run_results(model, test_data, test_labels, feature_names, cps = dry_run_cps)
        all_mrmse = dry_run_results(model, test_data, test_labels, feature_names, cps = train_data.idx.unique())

        score = 0.75 * dr_mrmse + 0.25 * all_mrmse
        if score < best_err:
            best_err = score
            save_model(model, out_dir, f"{model_name}_best")
            save_feature_names(feature_names, out_dir, f"{model_name}_best")
        return score
    
    if optuna_bool:
        study = optuna.create_study(direction="maximize")
        study.optimize(train_optuna, n_trials=50)
        print(f"Best Accuracy: {study.best_value:.4f}")
        print("Best Hyperparameters:", study.best_params)
        return study.best_value, study.best_params

    if not eval_model:
        if model_class in [SequentialRF, SimpleDNN, SimpleRF, 
                           SimpleMDN, SimpleLSTM, ConditionalDNN, 
                           SmartLSTM]:
            model = model_class(data_fname, n_folds, model_kwargs, 
                                out_dir = out_dir, model_name = model_name)
            model.train()
            model.save()
            
        else:
            train_data, train_labels, test_data, test_labels = process_features(data_fname, 
                                                                            pred_growth = pred_growth,
                                                                            pred_days = pred_days,
                                                                            dry_run_cps = dry_run_cps, 
                                                                            scale = scale,
                                                                            stratify_by = 'growth_bucket', 
                                                                            model_type = 'ensemble')
            print("training model...")
            model, feature_names = train_regressor( train_data, train_labels, 
                                                test_data, test_labels, 
                                                model_name = model_name, 
                                                drop_vars = drop_vars,
                                                model_class = model_class,
                                                feat_elim = feat_elim, 
                                                out_dir = out_dir,
                                                weights_factor = weights_factor,
                                                weighting_method = weighting_method,
                                                n_iters = n_folds)

            save_features(train_data, train_labels, 
                        test_data,  test_labels,
                        out_dir, model_name)
    else:
        model = load_model(f"{out_dir}/{model_name}")
        import ipdb; ipdb.set_trace()
        model.predict(pd.read_csv(data_fname))
        if not model_class is SequentialRF:
            train_data, train_labels, test_data, test_labels = load_features(out_dir, model_name)
            feature_names = load_feature_names(f"{out_dir}/{model_name}_features.txt")
            train_data = train_data
            test_data = test_data
    if not model_class is SequentialRF:
        train_dataset = (train_data, train_labels)
        test_dataset = (test_data, test_labels)

    plots_dir = f"{out_dir}/plots"
    os.makedirs(plots_dir, exist_ok = True)
    plot_importances(model, model_name, plots_dir, feature_names)
    # bucket_stats = plot_correlation(model, train_dataset, test_dataset, 
    #                     exp_name = model_name,
    #                     out_dir = plots_dir,
    #                     feature_names = feature_names,
    #                     stratify_by = 'growth_buckets')
    bucket_stats = {}
    if dry_run_bool:
        X, y = test_dataset
        dry_run_plots_dir = os.path.join(out_dir, "dry_run_plots")
        os.makedirs(dry_run_plots_dir, exist_ok = True)
        dry_run_stats = {'rmse' : [],
                         'cp' : []}
        for cp in dry_run_cps:
            idxs = X.loc[X['cp'] == cp].index
            if len(idxs) == 0:
                continue
            rmse = plot_model_results(model, X = X.loc[X.index.isin(idxs)], y = y.loc[y.index.isin(idxs)], 
                               method_name = model_name, fire_label = str(cp), 
                               min_FRP = 0.0, max_FRP = 453880.5, 
                               pred_type = scale, pred_days = pred_days, 
                               out_dir = dry_run_plots_dir, 
                               feature_names = feature_names,
                               ci_info = bucket_stats,
                               stratify_by = stratify_by, 
                               last_day = dry_run_date)
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

    stratify_by = args.strat

    weighting_method = args.wm
    weights_factor = args.wf

    n_folds = args.n_folds
    optuna_bool = args.optuna_bool


    main(features_fname, model_name, 
         model_class = model_class,
         model_kwargs = kwargs_dict[model_class],
         pred_growth = pred_growth,
         pred_days = pred_days,
         eval_model = eval_model,
         scale = scale,
         feat_elim = feat_elim,
         dry_run_bool = dry_run_bool,
         stratify_by = stratify_by,
         weighting_method = weighting_method,
         weights_factor = weights_factor,
         n_folds = n_folds,
         optuna_bool = optuna_bool)