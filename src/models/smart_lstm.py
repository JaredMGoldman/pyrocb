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
from torch.utils.data import Dataset, DataLoader
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
# 1. Defined Sequential LSTM Architecture
# -----------------------------------------------------------------
class FireLSTMRegressor(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, num_layers=2):
        super(FireLSTMRegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer expects input shape: (batch_size, seq_len, feature_dim)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # Fully connected head to map the final temporal state to a single prediction
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)  # Continuous scaling factor output
        )
        
    def forward(self, x, hidden=None):
        # Forward pass through LSTM with optional hidden state persistence per fire
        out, hidden = self.lstm(x, hidden)
        
        # Extract the hidden state output of the very last time-step row: out[:, -1, :]
        final_timestep_state = out[:, -1, :]
        
        # Map to regression scalar
        return self.fc_head(final_timestep_state), hidden


# -----------------------------------------------------------------
# 2. Custom Fire Sequence Dataset for Clean Batches
# -----------------------------------------------------------------
class FireSequenceDataset(Dataset):
    """
    Custom Dataset that yields structured 3D input sequences, targets, 
    sample weights, and group IDs (cp) for trackable state manipulation.
    """
    def __init__(self, X_3d, y_df, weights_array, cp_array):
        self.X_data = torch.FloatTensor(X_3d)
        self.y_data = torch.FloatTensor(y_df.values).view(-1, 1)
        self.weights = torch.FloatTensor(weights_array).view(-1, 1)
        self.cps = cp_array

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx], self.weights[idx], self.cps[idx]


# -----------------------------------------------------------------
# 3. Complete Group-Isolated LSTM Pipeline Class
# -----------------------------------------------------------------
class SmartLSTM:
    def __init__(self, dataset, n_folds, model_kwargs, 
                 lookback = 2, idx_name = 'cp', 
                 target_col = 'rave_FRP_MEAN', daily_freq = 24,
                 shuffle = True, rand_state = 42, out_dir = "my_path",
                 model_name = "demo_lstm_model", train_split = 0.8,
                 weight_factor = 5, weight_method = 'prod'):
        
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

    def _reshape_to_ltsm_tensor(self, X_df, X_cols):
        """
        Converts flattened structural lag lists into 3D LSTM Tensors.
        Ensures groupings remain chronologically isolated by unique fire IDs (cp).
        """
        base_features = [c for c in X_cols if '_lag_' not in c]
        lag_features = [c for c in X_cols if '_lag_' in c]
        unique_lagged_vars = sorted(list(set([c.split('_lag_')[0] for c in lag_features])))
        
        num_samples = len(X_df)
        seq_len = self.lookback * self.daily_freq 
        num_lag_vars = len(unique_lagged_vars)
        num_base_vars = len(base_features)
        
        X_3d = np.zeros((num_samples, seq_len, num_lag_vars + num_base_vars), dtype=np.float32)
        
        # Iterate over steps sequentially to parse time dimensions properly
        for step in range(seq_len):
            d = step // self.daily_freq
            h = step % self.daily_freq
            
            for var_idx, var_name in enumerate(unique_lagged_vars):
                col_name = f'{var_name}_lag_d{d}_h{h}'
                if col_name in X_df.columns:
                    X_3d[:, (seq_len - 1) - step, var_idx] = X_df[col_name].values
                    
            for b_idx, b_name in enumerate(base_features):
                X_3d[:, step, num_lag_vars + b_idx] = X_df[b_name].values
                
        return X_3d

    def _pytorch_predict(self, model, X_matrix_3d, cp_array):
        """Sequential inference handler that resets hidden memory when a new fire starts."""
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        preds_out = np.zeros(len(X_matrix_3d), dtype=np.float32)
        
        with torch.no_grad():
            hidden = None
            last_cp = None
            
            # Predict step-by-step or in mini-blocks to handle group sequence isolation
            for i in range(len(X_matrix_3d)):
                current_cp = cp_array[i]
                
                # Reset LSTM hidden memory when crossing boundaries into a new fire context
                if last_cp is None or current_cp != last_cp:
                    hidden = None
                
                # Prepare single-item slice tensor shape: (1, seq_len, features)
                x_tensor = torch.FloatTensor(X_matrix_3d[i:i+1]).to(device)
                pred, hidden = model(x_tensor, hidden)
                
                preds_out[i] = pred.cpu().item()
                last_cp = current_cp
                
        return preds_out

    def _pytorch_train_loop(self, model, X_matrix_3d, y_df, weights_array, cp_array):
        """Training loop that tracks group IDs and zeroes-out states between unique fires."""
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        # Sort data by fire event (cp) and time before constructing dataset to guarantee sequential rows
        sorted_indices = np.argsort(cp_array)
        
        dataset = FireSequenceDataset(
            X_matrix_3d[sorted_indices], 
            y_df.iloc[sorted_indices], 
            weights_array[sorted_indices],
            cp_array[sorted_indices]
        )
        
        # Shuffle = False is CRITICAL here; sequence isolation is managed within the batch sequence manually
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        for epoch in range(self.epochs):
            hidden = None
            last_cp = None
            
            for batch_x, batch_y, batch_w, batch_cps in train_loader:
                batch_x, batch_y, batch_w = batch_x.to(device), batch_y.to(device), batch_w.to(device)
                
                optimizer.zero_grad()
                
                # Check if the current batch contains a new fire boundary change
                current_first_cp = batch_cps[0].item()
                if last_cp is None or current_first_cp != last_cp:
                    # Clear out hidden context tensors completely to isolate states
                    hidden = None
                
                # Re-package hidden state to detach it from the history of previous updates
                if hidden is not None:
                    hidden = (hidden[0].detach(), hidden[1].detach())
                    
                    # If batch dimension changes due to a remainder batch, resize hidden arrays
                    if hidden[0].size(1) != batch_x.size(0):
                        hidden = None
                
                predictions, hidden = model(batch_x, hidden)
                
                loss = (batch_w * (predictions - batch_y) ** 2).mean()
                loss.backward()
                optimizer.step()
                
                last_cp = batch_cps[-1].item()

    def predict(self, X):
        # Enforce chronological ordering by group prior to calculations
        X = X.sort_values([self.idx_name, 'time']).reset_index(drop=True)
        
        rf_1day_data, rf_1d_X_cols, _ = self.process_1d_data(X)
        X_3d_1d = self._reshape_to_ltsm_tensor(rf_1day_data, rf_1d_X_cols)

        print('\nevaluating pred 1-day LSTM model')
        rf_1day_data['preds'] = self._pytorch_predict(self.dnn_1day, X_3d_1d, rf_1day_data[self.idx_name].values)
        pred_df = pd.DataFrame({'cp' : rf_1day_data.cp, 'time' : rf_1day_data.time, 'preds' : rf_1day_data['preds'].values})

        print('\nevaluating pred 2-day LSTM model')
        rf_2d_data, rf_2d_feats, _ = self.process_2d_data(X, pred_df)
        X_3d_2d = self._reshape_to_ltsm_tensor(rf_2d_data, rf_2d_feats)
        rf_2d_data['preds'] = self._pytorch_predict(self.dnn_2day, X_3d_2d, rf_2d_data[self.idx_name].values)
        
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
        # Group-Based Cross Validation Strategy: Splitting explicitly on unique fire IDs (cps)
        unique_train_cps = np.unique(self.train_ids)
        kf = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.rand_state)
        
        for fold, (train_group_idxs, val_group_idxs) in enumerate(kf.split(unique_train_cps)):
            train_cps = unique_train_cps[train_group_idxs]
            val_cps = unique_train_cps[val_group_idxs]
            
            # Extract data and enforce chronological ordering per fire
            train_data = self.dataset[self.dataset.cp.isin(train_cps)].sort_values([self.idx_name, 'time']).reset_index(drop=True)

            # --- 1-Day Model Pipeline ---
            rf_1day_data, rf_1d_X_cols, rf_1d_y_cols = self.process_1d_data(train_data)
            w_1day = sample_weights(rf_1day_data[rf_1d_y_cols], factor=self.weight_factor, method=self.weight_method)
            X_3d_1d = self._reshape_to_ltsm_tensor(rf_1day_data, rf_1d_X_cols)
            
            if self.dnn_1day is None:
                self.dnn_1day = FireLSTMRegressor(feature_dim=X_3d_1d.shape[2])
                
            print(f'\ntraining pred 1-day LSTM model [Fold {fold}]')
            self._pytorch_train_loop(self.dnn_1day, X_3d_1d, rf_1day_data[rf_1d_y_cols], w_1day, rf_1day_data[self.idx_name].values)

            # Generate sequential predictions for Day 2 pipeline features
            predictions = self._pytorch_predict(self.dnn_1day, X_3d_1d, rf_1day_data[self.idx_name].values)
            pred_df = pd.DataFrame({'cp' : rf_1day_data.cp, 'time' : rf_1day_data.time, 'preds' : predictions})

            # --- 2-Day Model Pipeline ---
            rf_2d_data, rf_2d_feats, rf_2d_labels = self.process_2d_data(train_data, pred_df)
            w_2day = sample_weights(rf_2d_data[rf_2d_labels], factor=self.weight_factor, method=self.weight_method)
            X_3d_2d = self._reshape_to_ltsm_tensor(rf_2d_data, rf_2d_feats)
            
            if self.dnn_2day is None:
                self.dnn_2day = FireLSTMRegressor(feature_dim=X_3d_2d.shape[2])
                
            print(f'training pred 2-day LSTM model [Fold {fold}]')
            self._pytorch_train_loop(self.dnn_2day, X_3d_2d, rf_2d_data[rf_2d_labels], w_2day, rf_2d_data[self.idx_name].values)

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
        
        # Extract evaluation data and enforce chronological ordering
        X = self.dataset[self.dataset.cp.isin(val_cps)].sort_values([self.idx_name, 'time']).reset_index(drop=True)

        rf_1day_data, rf_1d_X_cols, _ = self.process_1d_data(X)
        X_3d_1d = self._reshape_to_ltsm_tensor(rf_1day_data, rf_1d_X_cols)
        print('\nevaluating pred 1-day LSTM model')
        rf_1day_data['preds'] = self._pytorch_predict(self.dnn_1day, X_3d_1d, rf_1day_data[self.idx_name].values)
        pred_df = pd.DataFrame({'cp' : rf_1day_data.cp, 'time' : rf_1day_data.time, 'preds' : rf_1day_data['preds'].values})

        print('\nevaluating pred 2-day LSTM model')
        rf_2d_data, rf_2d_feats, _ = self.process_2d_data(X, pred_df)
        X_3d_2d = self._reshape_to_ltsm_tensor(rf_2d_data, rf_2d_feats)
        rf_2d_data['preds'] = self._pytorch_predict(self.dnn_2day, X_3d_2d, rf_2d_data[self.idx_name].values)
        
        # Prioritized confusion matrix generation framework remains unchanged
        bucket_labels = self.df_growth_bins[self.df_growth_bins[self.idx_name].isin(val_cps)]
        bucket_preds_1day = self._bin_growth(rf_1day_data.copy(), 'preds', 'one_day_pred_bin')[[self.idx_name, 'time', 'one_day_pred_bin']]
        bucket_preds_2day = self._bin_growth(rf_2d_data.copy(), 'preds', 'two_day_pred_bin')[[self.idx_name, 'time', 'two_day_pred_bin']]
        cmp_1day_buckets = pd.merge(bucket_labels, bucket_preds_1day, on = [self.idx_name, 'time'], how = 'inner')
        cmp_2day_buckets = pd.merge(bucket_labels, bucket_preds_2day, on = [self.idx_name, 'time'], how = 'inner')
        
        labels = ['decrease', 'stable', 'increase']
        
        if not cmp_1day_buckets.empty:
            confusion_1day = confusion_matrix(cmp_1day_buckets['one_day_scale_bin'].values, cmp_1day_buckets['one_day_pred_bin'].values, labels = labels)
            self._plot_fire_confusion(confusion_1day, f'fold_{fold_num}_1_Day_GroupIsolated_LSTM_Accuracy', labels)
        if not cmp_2day_buckets.empty:
            confusion_2day = confusion_matrix(cmp_2day_buckets['two_day_scale_bin'].values, cmp_2day_buckets['two_day_pred_bin'].values, labels = labels)
            self._plot_fire_confusion(confusion_2day, f'fold_{fold_num}_2_Day_GroupIsolated_LSTM_Accuracy', labels)
            
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

        p_1day = X_3d_1d.shape[2]
        p_2day = X_3d_2d.shape[2]
        
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
        os.makedirs(self.out_dir, exist_ok=True)
        torch.save(self.dnn_1day.state_dict(), os.path.join(self.out_dir, f"{self.model_name}_1day.pt"))
        torch.save(self.dnn_2day.state_dict(), os.path.join(self.out_dir, f"{self.model_name}_2day.pt"))
        print(f"Successfully exported PyTorch LSTM state tensors to {self.out_dir}")

    # Standard feature engineering processes retained below
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