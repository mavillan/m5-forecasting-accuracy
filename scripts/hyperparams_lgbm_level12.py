import os
import gc
import pickle
import time
import copy
import numpy as np
import pandas as pd
import lightgbm as lgb
from tsforest.forecaster import LightGBMForecaster
import optuna 

# local modules
import sys
sys.path.append("./lib/")
from utils import compute_scaling, reduce_mem_usage
from evaluation import WRMSSEEvaluator

EPS = 1e-16

###########################################################################################
# logger setting
###########################################################################################

if os.path.exists(f"./results/lgbm_level12.csv"):
    logger = open(f"./results/lgbm_level12.csv", "a")
else:
    logger = open(f"./results/lgbm_level12.csv", "w")
    logger.write("trial;params;best_iterations;errors;best_iteration;error\n")

###########################################################################################
# data prep
###########################################################################################
weights_level12 = pd.read_parquet("./input/weights_level12.parquet")
scaling_input = pd.read_parquet("./input/scaling_input.parquet")
scales = compute_scaling(scaling_input, agg_columns=["store_id","item_id"]).rename({"q":"s"}, axis=1)

# validation periods
valid_periods = [(pd.to_datetime("2015-04-25"), pd.to_datetime("2015-05-22")),
                 (pd.to_datetime("2015-05-23"), pd.to_datetime("2015-06-19")),
                 #(pd.to_datetime("2016-02-29"), pd.to_datetime("2016-03-27")),
                 #(pd.to_datetime("2016-03-28"), pd.to_datetime("2016-04-24"))
                ]

# precomputed (features) models
precomputed_models = list()
for i in range(2):
    with open(f"./precomputed/model{i}.pickle", "rb") as handler: 
        model = pickle.load(handler)
        precomputed_models.append(model)
        handler.close()     

###########################################################################################
# definition of objective
###########################################################################################

default_model_params = {
    'objective':'tweedie',
    'boost_from_average':False,
    'metric':'None',
    'num_iterations':100000,
    'early_stopping_rounds':200,
    'num_leaves':2**10-1,
    'min_data_in_leaf':2**11-1,
    #'max_bin': 127,
    'bin_construct_sample_cnt':1000000,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction':0.7,
    'bagging_freq':1,
    'lambda_l2':0.1,
    'seed':7,
}

def objective(trial):
    sampled_params = {
        "tweedie_variance_power": trial.suggest_uniform("tweedie_variance_power", 1.0, 1.6),
        #"boost_from_average" : trial.suggest_categorical("boost_from_average", [True, False]),
        #"num_leaves": 2**(trial.suggest_int("num_leaves", 7, 12)) - 1,
        #"min_data_in_leaf": 2**(trial.suggest_int("min_data_in_leaf", 5, 13)) - 1,
        #"feature_fraction": trial.suggest_discrete_uniform("feature_fraction", 0.5, 1.0, 0.1),
        #"bagging_fraction": trial.suggest_discrete_uniform("bagging_fraction", 0.5, 1.0, 0.1),
    }
    model_params = {**default_model_params, **sampled_params}
    
    errors = list()
    best_iterations = list()
    
    for i,valid_period in enumerate(valid_periods):
        print(f" {i+1}/{len(valid_periods)} ".center(100, "#"))
        print(f" Validation period: {valid_period} ".center(100, "#"))
        print("#"*100)
        
        print(f"[INFO] preparing the model {i+1}")
        tic = time.time()
        fcaster = copy.deepcopy(precomputed_models[i])
        fcaster.set_params(model_params=model_params)
        tac = time.time()
        print(f"[INFO] time elapsed preparing the model {i+1}: {(tac-tic)/60.} min.\n")
        
        print(f"[INFO] fitting the model {i+1}")
        tic = time.time()
        evaluator = WRMSSEEvaluator(fcaster.valid_features.loc[:, ["ds"]+fcaster.ts_uid_columns+["y"]], 
                                    weights_level12, 
                                    scales,
                                    ts_uid_columns=fcaster.ts_uid_columns)
        fcaster.fit(fit_kwargs={"verbose_eval":25, "feval":evaluator.evaluate})
        tac = time.time()
        print(f"[INFO] time elapsed fitting the model {i+1}: {(tac-tic)/60.} min.\n")
        
        valid_error = fcaster.model.model.best_score["valid_0"]["wrmsse"]
        best_iteration = fcaster.best_iteration
        
        errors.append(valid_error)
        best_iterations.append(best_iteration)
        print(f"[INFO] validation error {i+1}: {valid_error}")
        print(f"[INFO] best iteration {i+1}: {best_iteration}")
        
        del fcaster,evaluator
        gc.collect()
            
    best_iteration = np.mean(best_iterations)
    error = np.mean(errors)     

    logger.write(f"{trial.number};{model_params};{best_iterations};{errors};{best_iteration};{error}\n")
    logger.flush()
    
    return error

###########################################################################################
# study definition
###########################################################################################
search_space = {'tweedie_variance_power': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                #'boost_from_average':[True, False], 
               }
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective, n_trials=6)
logger.close()
