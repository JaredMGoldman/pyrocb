import os
import warnings
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold

# Preserving your exact import framework
from utils.feature_creation import (
    calculate_hdw,
    calculate_vpd,
    data_normalization,
    get_fire_growth_metrics,
    hrrr_features,
    sample_weights,
)


# -----------------------------------------------------------------
# 1. Refactored MDN Network Architecture
# -----------------------------------------------------------------
class FireMDN(nn.Module):
    def __init__(self, input_dim, num_gaussians=2):
        super(FireMDN, self).__init__()
        self.num_gaussians = num_gaussians
        
        # Shared feature extractor backbone matching your exact density layer configuration
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # MDN Head 1: Mixing Coefficients (pi) -> Must sum to 1.0
        self.pi_head = nn.Linear(32, num_gaussians)
        # MDN Head 2: Cluster Means (mu) -> Continuous scaling factors
        self.mu_head = nn.Linear(32, num_gaussians)
        # MDN Head 3: Variances (sigma) -> Strictly positive standard deviations
        self.sigma_head = nn.Linear(32, num_gaussians)
        
    def forward(self, x):
        features = self.backbone(x)
        
        pi = torch.softmax(self.pi_head(features), dim=1)
        mu = self.mu_head(features)
        # Exponential activation with an stability floor prevents sigma from reaching zero
        sigma = torch.exp(self.sigma_head(features)) + 1e-6
        
        return pi, mu, sigma


# -----------------------------------------------------------------
# 2. Custom Negative Log-Likelihood (NLL) Loss Function
# -----------------------------------------------------------------
def mdn_loss_fn(pi, mu, sigma, target):
    """
    Computes Gaussian Mixture NLL while accommodating row-specific sample weight vectors.
    """
    target = target.expand_as(mu)
    variance = sigma ** 2
    
    # Standard Gaussian Probability Density Function
    gaussian_pdf = torch.exp(-((target - mu) ** 2) / (2 * variance)) / torch.sqrt(2 * np.pi * variance)
    
    # Aggregate weighted probability distributions across mixture components
    total_pdf = torch.sum(pi * gaussian_pdf, dim=1)
    total_pdf = torch.clamp(total_pdf, min=1e-10) # Protect against log(0)
    
    return -torch.log(total_pdf)


# -----------------------------------------------------------------
# 3. MDN Pipeline Class Execution Framework
# -----------------------------------------------------------------
class SimpleMDN:
    def __init__(self, dataset, n_folds, model_kwargs, 
                 lookback = 2, idx_name = 'cp', 
                 target_col = 'rave_FRP_MEAN', daily_freq = 24,
                 shuffle = True, rand_state = 42, out_dir = "my_path",
                 model_name = "demo_mdn_model", train_split = 0.8,
                 weight_factor = 0.5, weight_method = 'prod', num_gaussians = 2):
        
        self.log_bool = True
        self.dataset = pd.read_csv(dataset)
        self.dataset['time'] = pd.to_datetime(self.dataset['time'])
        self.lookback = lookback
        self.idx_name = idx_name
        self.target_col = target_col
        self.daily_freq = daily_freq

        self.rand_state = rand_state
        self.shuffle = shuffle
        self.n_folds = n_folds
        self.train_split = train_split
        self.weight_factor = weight_factor
        self.weight_method = weight_method
        self.num_gaussians = num_gaussians

        self.out_dir = out_dir
        self.model_name = model_name
        
        self.epochs = model_kwargs.get('epochs', 10)
        self.lr = model_kwargs.get('lr', 0.001)
        self.batch_size = model_kwargs.get('batch_size', 128)

        self.dnn_1day = None
        self.dnn_2day = None
        
        self.normalization_df = pd.DataFrame({col : {'min' : self.dataset[col].min(), 'max' : self.dataset[col].max()} for col in self.dataset.columns})
        self._split_data()
        self._define_growth_bins()

    def _split_data(self):
        fire_growth_df = get_fire_growth_metrics(self.dataset)
        self.train_ids = []
        for bucket in fire_growth_df['growth_bucket'].unique():
            subset = fire_growth_df[fire_growth_df['growth_bucket'] == bucket]['cp'].tolist()
            k = int(len(subset) * self.train_split)
            self.train_ids.extend(sample(subset, k=k))

        self.test_ids = np.array(list(set(fire_growth_df.cp.unique()) - set(self.train_ids)))
        self.train_ids = np.array(self.train_ids)     

    def _bin_growth(self, df, scale_col, target_col, thresh = 0.75):
        df[target_col] = 'stable'
        if self.log_bool:
            low_thresh = np.log(1.0 - thresh) if (1.0 - thresh) > 0 else -10.0
            hi_thresh = np.log(1.0 + thresh)
        else:
            low_thresh = 1.0 - thresh
            hi_thresh = 1.0 + thresh

        low_mask = df[scale_col] <= low_thresh
        hi_mask = df[scale_col] >= hi_thresh
        
        df.loc[low_mask, target_col] = 'decrease'
        df.loc[hi_mask, target_col] = 'increase'
        return df

    def _define_growth_bins(self):
        df = self.dataset.copy()[[self.idx_name, 'time', self.target_col]]
        daily_max = df.groupby([self.idx_name, df['time'].dt.date])[self.target_col].max().fillna(1e-6).clip(lower=1e-6)

        tmrw = df.time.dt.date + pd.Timedelta(1, 'day')
        two_days = df.time.dt.date + pd.Timedelta(2, 'day')
        if self.log_bool:
            one_day_scale = np.log(daily_max.shift(-1) / daily_max)
            two_day_scale = np.log(daily_max.shift(-2) / daily_max)
        else:
            one_day_scale = daily_max.shift(-1) / daily_max
            two_day_scale = daily_max.shift(-2) / daily_max

        df['one_day_scale'] = pd.MultiIndex.from_arrays([df[self.idx_name], tmrw]).map(one_day_scale)
        df['two_day_scale'] = pd.MultiIndex.from_arrays([df[self.idx_name], two_days]).map(two_day_scale)

        df = df.dropna(subset = ['one_day_scale', 'two_day_scale'])
        df = self._bin_growth(df, 'one_day_scale', 'one_day_scale_bin')
        df = self._bin_growth(df, 'two_day_scale', 'two_day_scale_bin')
        self.df_growth_bins = df[[self.idx_name, 'time', 'one_day_scale_bin', 'two_day_scale_bin']]

    def _pytorch_predict(self, model, X_matrix):
        """Computes the mathematically expected value mean E(x) = SUM(pi_k * mu_k) from MDN distributions."""
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_matrix).to(device)
            pi, mu, _ = model(X_tensor)
            
            pi = pi.cpu().numpy()
            mu = mu.cpu().numpy()
            # Dynamic calculation of the expected value across the multi-modal distribution
            expected_values = np.sum(pi * mu, axis=1).flatten()
        return expected_values

    def _pytorch_train_loop(self, model, X_df, y_df, weights_array):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        dataset = TensorDataset(
            torch.FloatTensor(X_df.values),
            torch.FloatTensor(y_df.values).view(-1, 1),
            torch.FloatTensor(weights_array).view(-1, 1)
        )
        
        train_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        for epoch in range(self.epochs):
            for batch_x, batch_y, batch_w in train_loader:
                batch_x, batch_y, batch_w = batch_x.to(device), batch_y.to(device), batch_w.to(device)
                
                optimizer.zero_grad()
                pi, mu, sigma = model(batch_x)
                
                # Compute Likelihood elements and cross-multiply by your sample training weights
                raw_losses = mdn_loss_fn(pi, mu, sigma, batch_y)
                loss = (batch_w * raw_losses.view(-1, 1)).mean()
                
                loss.backward()
                # Gradient clipping is vital to stabilize variance calculations in multi-headed MDNs
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

    def predict(self, X):
        rf_1day_data, rf_1d_X_cols, _ = self.process_1d_data(X)

        print('\nevaluating pred 1-day MDN model')
        rf_1day_data['preds'] = self._pytorch_predict(self.dnn_1day, rf_1day_data[rf_1d_X_cols].values)
        pred_df = pd.DataFrame({'cp' : rf_1day_data.cp, 'time' : rf_1day_data.time, 'preds' : rf_1day_data['preds'].values})

        print('\nevaluating pred 2-day MDN model')
        rf_2d_data, rf_2d_feats, _ = self.process_2d_data(X, pred_df)
        rf_2d_data['preds'] = self._pytorch_predict(self.dnn_2day, rf_2d_data[rf_2d_feats].values)
        
        idxed_1d_preds = pd.DataFrame({
            'cp' : rf_1day_data.cp, 
            'time' : rf_1day_data.time,
            'max_frp' : self._denormalize(self.target_col, rf_1day_data['max_frp'].values),
            '1day_pred' : rf_1day_data['preds'].values 
        })
        idxed_2d_preds = pd.DataFrame({
            'cp' : rf_2d_data.cp, 
            'time' : rf_2d_data.time,
            '2day_pred' : rf_2d_data.preds.values
        })
        results_df = pd.merge(idxed_1d_preds, idxed_2d_preds, how='outer', on = ['cp', 'time'])
        return results_df

    def fit(self, *args, **kwargs):
        self.train()

    def train(self):
        kf = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.rand_state)
        
        for fold, (train_idxs, val_idxs) in enumerate(kf.split(self.train_ids)):
            train_cps = self.train_ids[train_idxs]
            val_cps = self.train_ids[val_idxs]
            train_data = self.dataset[self.dataset.cp.isin(train_cps)]

            # --- 1-Day Model Pipeline ---
            rf_1day_data, rf_1d_X_cols, rf_1d_y_cols = self.process_1d_data(train_data)
            w_1day = sample_weights(rf_1day_data[rf_1d_y_cols], factor=self.weight_factor, method=self.weight_method)
            
            if self.dnn_1day is None:
                self.dnn_1day = FireMDN(input_dim=len(rf_1d_X_cols), num_gaussians=self.num_gaussians)
                
            print(f'\ntraining pred 1-day MDN model [Fold {fold}]')
            self._pytorch_train_loop(self.dnn_1day, rf_1day_data[rf_1d_X_cols], rf_1day_data[rf_1d_y_cols], w_1day)

            # Generate expected predictions as historical features for Day 2 pipeline
            predictions = self._pytorch_predict(self.dnn_1day, rf_1day_data[rf_1d_X_cols].values)
            pred_df = pd.DataFrame({'cp' : rf_1day_data.cp, 'time' : rf_1day_data.time, 'preds' : predictions})

            # --- 2-Day Model Pipeline ---
            rf_2d_data, rf_2d_feats, rf_2d_labels = self.process_2d_data(train_data, pred_df)
            w_2day = sample_weights(rf_2d_data[rf_2d_labels], factor=self.weight_factor, method=self.weight_method)
            
            if self.dnn_2day is None:
                self.dnn_2day = FireMDN(input_dim=len(rf_2d_feats), num_gaussians=self.num_gaussians)
                
            print(f'training pred 2-day MDN model [Fold {fold}]')
            self._pytorch_train_loop(self.dnn_2day, rf_2d_data[rf_2d_feats], rf_2d_data[rf_2d_labels], w_2day)

            self.eval(fold, val_cps)
            
        self.eval("test", self.test_ids)
    
    def eval(self, fold_num, val_cps):
        results_dict = {'fold_num' : [fold_num],
                        'rmse_1day' : [0.0],
                        'rmse_2day' : [0.0],
                        'r2_1day' : [0.0],
                        'r2_2day' : [0.0],
                        'adj_r2_1day' : [0.0],
                        'adj_r2_2day' : [0.0]}
        X = self.dataset[self.dataset.cp.isin(val_cps)]

        rf_1day_data, rf_1d_X_cols, _ = self.process_1d_data(X)
        print('\nevaluating pred 1-day MDN model')
        rf_1day_data['preds'] = self._pytorch_predict(self.dnn_1day, rf_1day_data[rf_1d_X_cols].values)
        pred_df = pd.DataFrame({'cp' : rf_1day_data.cp, 'time' : rf_1day_data.time, 'preds' : rf_1day_data['preds'].values})

        print('\nevaluating pred 2-day MDN model')
        rf_2d_data, rf_2d_feats, _ = self.process_2d_data(X, pred_df)
        rf_2d_data['preds'] = self._pytorch_predict(self.dnn_2day, rf_2d_data[rf_2d_feats].values)
        
        # Build categorical growth alignment summaries from MDN center expectations
        bucket_labels = self.df_growth_bins[self.df_growth_bins[self.idx_name].isin(val_cps)]
        bucket_preds_1day = self._bin_growth(rf_1day_data.copy(), 'preds', 'one_day_pred_bin')[[self.idx_name, 'time', 'one_day_pred_bin']]
        bucket_preds_2day = self._bin_growth(rf_2d_data.copy(), 'preds', 'two_day_pred_bin')[[self.idx_name, 'time', 'two_day_pred_bin']]
        cmp_1day_buckets = pd.merge(bucket_labels, bucket_preds_1day, on = [self.idx_name, 'time'], how = 'inner')
        cmp_2day_buckets = pd.merge(bucket_labels, bucket_preds_2day, on = [self.idx_name, 'time'], how = 'inner')
        
        labels = ['decrease', 'stable', 'increase']
        
        if not cmp_1day_buckets.empty:
            confusion_1day = confusion_matrix(cmp_1day_buckets['one_day_scale_bin'].values, cmp_1day_buckets['one_day_pred_bin'].values, labels = labels)
            self._plot_fire_confusion(confusion_1day, f'fold_{fold_num}_1_Day_MDN_Classification_Accuracy', labels)
        if not cmp_2day_buckets.empty:
            confusion_2day = confusion_matrix(cmp_2day_buckets['two_day_scale_bin'].values, cmp_2day_buckets['two_day_pred_bin'].values, labels = labels)
            self._plot_fire_confusion(confusion_2day, f'fold_{fold_num}_2_Day_MDN_Classification_Accuracy', labels)
            
        y_true_1d = rf_1day_data['scaling_label'].values.astype(np.float64)
        y_pred_1d = rf_1day_data['preds'].values.astype(np.float64)
        y_true_2d = rf_2d_data['scaling_label'].values.astype(np.float64)
        y_pred_2d = rf_2d_data['preds'].values.astype(np.float64)

        results_dict['rmse_1day'] = [root_mean_squared_error(y_true_1d, y_pred_1d)]
        results_dict['rmse_2day'] = [root_mean_squared_error(y_true_2d, y_pred_2d)]
        
        n_1day = len(y_true_1d)
        n_2day = len(y_true_2d)
        r2_1day = r2_score(y_true_1d, y_pred_1d)
        r2_2day = r2_score(y_true_2d, y_pred_2d)
        
        results_dict['r2_1day'] = [r2_1day]
        results_dict['r2_2day'] = [r2_2day]

        p_1day = len(rf_1d_X_cols)
        p_2day = len(rf_2d_feats)
        
        results_dict['adj_r2_1day'] = [1 - ((1 - r2_1day) * (n_1day - 1)) / (n_1day - p_1day - 1)] if (n_1day - p_1day - 1) > 0 else [0.0]
        results_dict['adj_r2_2day'] = [1 - ((1 - r2_2day) * (n_2day - 1)) / (n_2day - p_2day - 1)] if (n_2day - p_2day - 1) > 0 else [0.0]
        
        print(f"completed fold {fold_num} evaluation")
        
        results_df = pd.DataFrame(results_dict)
        os.makedirs(os.path.join(self.out_dir, "results"), exist_ok = True)
        results_df.to_csv(os.path.join(self.out_dir, "results", f'eval_results_fold_{fold_num}.csv'), index=False)
        return results_df
    
    def _plot_fire_confusion(self, matrix, title, labels):
        row_sums = matrix.sum(axis=1)[:, np.newaxis]
        matrix_perc = np.divide(matrix.astype('float'), row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums!=0)
        df_cm = pd.DataFrame(matrix_perc, index=labels, columns=labels)
        
        plt.figure(figsize=(8, 6))
        sns.set_context("talk")
        sns.heatmap(df_cm, annot=True, fmt='.2%', cmap="Blues")
        
        plt.title(title, pad=20)
        plt.ylabel('Actual Observation')
        plt.xlabel('Model Prediction')
        plt.tight_layout()
        os.makedirs(os.path.join(self.out_dir, "plots"), exist_ok = True)
        out_path = os.path.join(self.out_dir, "plots", title.lower().replace(" ", "_") + '.png')
        plt.savefig(out_path)
        plt.close()
    
    def save(self):
        """Saves weights for the PyTorch MDN network structures."""
        os.makedirs(self.out_dir, exist_ok=True)
        torch.save(self.dnn_1day.state_dict(), os.path.join(self.out_dir, f"{self.model_name}_1day.pt"))
        torch.save(self.dnn_2day.state_dict(), os.path.join(self.out_dir, f"{self.model_name}_2day.pt"))
        print(f"Successfully exported PyTorch MDN state tensors to {self.out_dir}")

    # Rest of preprocessing methods preserved natively
    def create_meta_features(self, df, forecast_days = 1):
        df['wind_speed'] = np.sqrt(df['hrrr_u'] ** 2 + df['hrrr_v'] ** 2)
        df = df.drop(['hrrr_u', 'hrrr_v'], axis = 1)

        df['vpd'] = calculate_vpd(df['hrrr_t'], df['hrrr_rh'])
        df['hdw'] = calculate_hdw(df['hrrr_t'], df['hrrr_rh'], df['wind_speed'])

        df = data_normalization(df, df_min_max = self.normalization_df)
        df = df.sort_values([self.idx_name, 'time']).copy()
        
        df['hour'] = df['time'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        X_features = ['hour_sin', 'hour_cos']
        weather_feats = list(set(hrrr_features) - set(['hrrr_u', 'hrrr_v'])) + ['esi_DFPPM', 'wind_speed', 'vpd', 'hdw']
        daily_feats = weather_feats + [self.target_col]

        for d in range(self.lookback):
            for h in range(self.daily_freq):
                for feat_name in daily_feats:
                    fname = f'{feat_name}_lag_d{d}_h{h}'
                    df[fname] = df.groupby(self.idx_name)[feat_name].shift(d * self.daily_freq + h)
                    X_features.append(fname)

        df['next_date'] = df['time'].dt.date + pd.Timedelta(days=forecast_days)

        weather_ref = df.pivot_table(
            index=[self.idx_name, df['time'].dt.date], 
            columns=df['time'].dt.hour, 
            values=weather_feats
        )
        weather_ref.columns = [f'{col}_f{hour:02d}' for col, hour in weather_ref.columns]

        df = df.merge(
            weather_ref, 
            left_on=[self.idx_name, 'next_date'], 
            right_index=True, 
            how='left'
        )
        X_features.extend([c for c in weather_ref.columns])
        return df, X_features
    
    def process_1d_data(self, X):
        warnings.filterwarnings('ignore')
        df, X_features = self.create_meta_features(X.copy())
        current_frp = (
            df.groupby(self.idx_name)[self.target_col]
            .rolling(window=self.daily_freq, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
            .clip(lower=1e-6)
        )
        df['max_frp'] = current_frp
        X_features.append('max_frp')

        daily_max = df.groupby([self.idx_name, df['time'].dt.date])[self.target_col].max().fillna(1e-6).clip(lower=1e-6)
        tomorrow_date = df['time'].dt.date + pd.Timedelta(days=1)
        future_frp = pd.MultiIndex.from_arrays([df[self.idx_name], tomorrow_date]).map(daily_max)
        
        if self.log_bool:
            df['scaling_label'] = np.log(future_frp / current_frp)
        else:
            df['scaling_label'] = future_frp / current_frp

        y_cols = ['scaling_label']
        df[y_cols] = df[y_cols].fillna(1e-6)
        df_clean = df.dropna(subset=X_features)
        return df_clean, sorted(X_features), y_cols

    def process_2d_data(self, X, predictions):
        warnings.filterwarnings('ignore')
        df, X_features = self.create_meta_features(X.copy(), forecast_days=2)

        df = pd.merge(df, predictions, on=['cp', 'time'], how='inner')
        df = df.dropna(subset=X_features)
        daily_max = df.groupby([self.idx_name, df['time'].dt.date])[self.target_col].max().fillna(1e-6).clip(lower=1e-6)
        two_date = df['time'].dt.date + pd.Timedelta(days=2)
        future_frp = pd.MultiIndex.from_arrays([df[self.idx_name], two_date]).map(daily_max)

        current_frp = (
            df.groupby(self.idx_name)[self.target_col]
            .rolling(window=self.daily_freq, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
            .clip(lower=1e-6)
        )
        df['yesterday_max_frp'] = current_frp
        
        if self.log_bool:
            df['pred_interim_max_frp'] = np.exp(df['preds']) * current_frp 
            df['scaling_label'] = np.log(future_frp / (np.exp(df['preds']) * current_frp)) 
        else:
            df['pred_interim_max_frp'] = df['preds'] * current_frp
            df['scaling_label'] = future_frp / (df['preds'] * current_frp)
        
        X_features.extend(['yesterday_max_frp', 'pred_interim_max_frp'])
        y_cols = ['scaling_label']

        df[y_cols] = df[y_cols].fillna(1e-6)
        return df, sorted(X_features), y_cols

    def _denormalize(self, col_name, values, epsilon = 0.001):
        max_val = np.float64(self.normalization_df[col_name]['max'])
        min_val = np.float64(self.normalization_df[col_name]['min'])
        return (values - epsilon) * (max_val - min_val) + min_val