import pandas as pd
import numpy as np

#LOAD IN THE FEATURE CSV WITH HDW500 (which now also has HRRR moisture added in)
features_no_outliers_weighted = pd.read_csv('/data/lthapa/data2restore/lthapa/ML_daily/features_no_outliers_weightedHDW500.csv',
                                           parse_dates=['datetime'])
features_no_outliers_weighted=features_no_outliers_weighted.drop(columns=['Unnamed: 0'])

# 2020 all fires, including August Complex
data_test_2020 = features_no_outliers_weighted[(features_no_outliers_weighted['datetime']>=np.datetime64('2020-01-01 00:00:00'))&
                                              (features_no_outliers_weighted['datetime']<=np.datetime64('2020-12-31 23:00:00'))] #time range

# 2019/2021 Training Set
data_train_1921 = features_no_outliers_weighted[(features_no_outliers_weighted['datetime']<np.datetime64('2020-01-01 00:00:00'))|
                                              (features_no_outliers_weighted['datetime']>np.datetime64('2020-12-31 23:00:00'))]


def load_data():
    return data_train_1921, data_test_2020