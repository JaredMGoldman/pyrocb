import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from features import feature_subsets
from data.load_datasets import load_data
from plotting import plot_importances
import utils
from helper_functions import summarize_error

thresh_inc = 0.18#0.1#, scaling factor of 1.5
thresh_dec = -0.3#-0.1# scaling factor of 1.5
train_data, test_data = load_data() 
feature_set_names = list(feature_subsets.keys())

def train_model(feature_set, 
                      min_samples_leaf = 5, 
                      random_state = 42,
                      save_model = False):
    training_variables = feature_subsets[feature_set]
    model = RandomForestRegressor(oob_score=True, random_state=random_state, min_samples_leaf=min_samples_leaf)
    model.fit(train_data.loc[:, training_variables], train_data.loc[:, ['Log_Scaling_Factor']])

    print(feature_set + ' OOB R2 score is: '+ str(model.oob_score_))
    plot_importances(model, training_variables, feature_set)
    if save_model:
        utils.save_model(model, f"{feature_set}_model_weights")
    return model

def eval_model(model, feature_set):
    results_dict = {}
    training_variables = feature_subsets[feature_set]

    results_dict['logsf'] = model.predict(test_data.loc[:,training_variables])

    results_dict['sf'] = 10**(results_dict['logsf'])
    results_dict['fre'] = test_data['FRE_1']*results_dict['sf']
    results_dict['logfre'] = np.log10(test_data['FRE'])
    results_dict['cat'] = pd.cut(   results_dict['logsf'], 
                                    bins=[-10, thresh_dec, thresh_inc, 10],
                                    labels=['decrease','no_change','increase'])#transform RF SF into categories
    return results_dict

if __name__ == "__main__":
    feature_sets = ['features_no_persistence']
    for feature_set in feature_sets:
        model = train_model(feature_set, save_model=True)
        eval_model(model, feature_set)