import os
import gc
import copy
import time
import numpy as np; np.random.seed(42)
import pandas as pd
import lightgbm as lgb
import category_encoders as ce
import matplotlib.pyplot as plt
import optuna
import numba

from tsforest.forecaster import LightGBMForecaster
from tsforest.utils import make_time_range
from tsforest.metrics import compute_rmse

# local modules
import sys
sys.path.append("../lib/")
from utils import (compute_scaling, compute_weights, reduce_mem_usage, 
                   compute_scales_by_level, compute_weights_by_level)
from evaluation import _WRMSSEEvaluator, WRMSSEEvaluator, Evaluator, WRMSSEEvaluatorL12
from encoding import HierarchicalEncoder

###########################################################################################
# selection of fold
###########################################################################################

FOLD = 4

###########################################################################################
# logger setting
###########################################################################################

if os.path.exists(f"../results/lgbm_num_iterations-f{FOLD}.csv"):
    logger = open(f"../results/lgbm_num_iterations-f{FOLD}.csv", "a")
else:
    logger = open(f"../results/lgbm_num_iterations-f{FOLD}.csv", "w")
    logger.write("trial;params;wrmsse_by_level;wrmsse\n")
    
###########################################################################################
# data prep
###########################################################################################

# validation period to be used for test in all this notebook
valid_periods = {
    1: (pd.to_datetime("2015-04-25"), pd.to_datetime("2015-05-22")),
    2: (pd.to_datetime("2015-05-23"), pd.to_datetime("2015-06-19")), 
    3: (pd.to_datetime("2016-03-28"), pd.to_datetime("2016-04-24")),
    4: (pd.to_datetime("2016-04-25"), pd.to_datetime("2016-05-22")),
}
valid_period = valid_periods[FOLD]

data = (pd.read_parquet("../input/train_dataframe.parquet")
        .reset_index(drop=True)
        .rename({"q":"y"}, axis=1)
       )
data["sales"] = data.eval("y * sell_price")

scaling_input = pd.read_parquet("../input/scaling_input.parquet")
weighting_input = pd.read_parquet("../input/weighting_input.parquet")

scales_by_level = compute_scales_by_level(scaling_input, valid_period[0])
weights_by_level = compute_weights_by_level(weighting_input, valid_period[0])

###########################################################################################
# model config
###########################################################################################

default_model_params = {
    'objective':'tweedie',
    'tweedie_variance_power': 1.1,
    'metric':'None',
    #'num_iterations':500,
    #'early_stopping_rounds':300,
    'max_bin': 127,
    'bin_construct_sample_cnt':15000000,
    'num_leaves': 2**10-1,
    'min_data_in_leaf': 2**11-1,
    'learning_rate': 0.03, 
    'feature_fraction': 0.9,
    'bagging_fraction':0.66,
    'bagging_freq':1,
    'seed':7,
    'boost_from_average': False,
    'first_metric_only': True,
}

time_features = [
    "year",
    "month",
    #"year_week",
    #"year_day",
    "week_day",
    "month_progress",
    #"week_day_cos",
    #"week_day_sin",
    #"year_day_cos",
    #"year_day_sin",
    "year_week_cos",
    "year_week_sin",
    #"month_cos",
    #"month_sin"
]

exclude_features = [
    "ts_id",
    "no_stock_days",
    "sales",
]

categorical_features = {
    #"item_id": ("y", ce.GLMMEncoder, None),
    "item_id": "default",
    "store_id": "default",
    "state_id": "default",
    "dept_id": "default",
    "cat_id": "default",
    "event_name_1": "default",
    }

@numba.jit(nopython=True, nogil=True, fastmath=True)
def compute_nzeros(x):
    return np.sum(x==0)

@numba.jit(nopython=True, nogil=True, fastmath=True)
def compute_czeros(x):
    return np.sum(np.cumprod((x==0)[::-1]))/x.shape[0]

@numba.jit(nopython=True, nogil=True, fastmath=True)
def compute_sfreq(x):
    return np.sum(x!=0)/x.shape[0]

model_kwargs = {
    "time_features":time_features,
    "window_functions":{
        "mean":   (None, [1,7,28], [7,14,21,28]),
        "median": (None, [1,7,14,28], [7,]),
        "std":    (None, [1,7,28], [7,28]),
        "kurt":   (None, [1,7,28], [7,28]),
        "czeros": (compute_czeros, [1,], [7,14,21,28,56]),
        "sfreq":  (compute_sfreq, [1,],  [7,14,21,28]),
    },
    "exclude_features":exclude_features,
    "categorical_features":categorical_features,
    "ts_uid_columns":["item_id","store_id"],
}

lagged_features_to_dropna = list()
if "lags" in model_kwargs.keys():
    lag_features = [f"lag{lag}" for lag in model_kwargs["lags"]]
    lagged_features_to_dropna.extend(lag_features)
if "window_functions" in model_kwargs.keys():
    rw_features = list()
    for window_func,window_def in model_kwargs["window_functions"].items():
        _,window_shifts,window_sizes = window_def
        if window_func in ["mean","median","std","min","max"]:
            rw_features.extend([f"{window_func}{window_size}_shift{window_shift}"
                                for window_size in window_sizes
                                for window_shift in window_shifts])
    lagged_features_to_dropna.extend(rw_features)

###########################################################################################
# definition of objective
###########################################################################################

valid_start = valid_period[0]
valid_end = valid_period[1]
_train_data = data.query("ds < @valid_start").reset_index(drop=True)

print("Building the features")
tic = time.time()
model_level12_base = LightGBMForecaster(**model_kwargs)
model_level12_base.prepare_features(train_data=_train_data)
model_level12_base.train_features.dropna(subset=lagged_features_to_dropna, axis=0, inplace=True)
model_level12_base.train_features = reduce_mem_usage(model_level12_base.train_features)
tac = time.time()
print(f"Elapsed time: {(tac-tic)/60.} [min]")

ts_id_in_train = model_level12_base.train_features.ts_id.unique()
valid_data = data.query("@valid_start <= ds <= @valid_end & ts_id in @ts_id_in_train")
evaluator = Evaluator(valid_data, weights_by_level, scales_by_level)
    

def objective(trial):
    sampled_params = {"num_iterations": trial.suggest_int("num_iterations", 500, 1500)}
    model_params = {**default_model_params, **sampled_params}
    model_level12 = copy.deepcopy(model_level12_base)
    model_level12.set_params(model_params)

    print("Fitting the model")
    tic = time.time()
    model_level12.fit()
    tac = time.time()
    print(f"Elapsed time: {(tac-tic)/60.} [min]")

    print("Predicting")
    tic = time.time()
    forecast = model_level12.predict(valid_data.drop("y", axis=1), recursive=True)
    tac = time.time()
    print(f"Elapsed time: {(tac-tic)/60.} [min]")

    wrmsse = evaluator.eval1._evaluate(forecast.y_pred.values)
    print("wrmsse:", wrmsse)
    
    logger.write(f"{trial.number};{sampled_params};{evaluator.eval1.errors_by_level};{wrmsse}\n")
    logger.flush()
    
    del model_level12
    gc.collect()
    
    return wrmsse

###########################################################################################
# study definition
###########################################################################################
search_space = {
    'num_iterations': [500,750,1000,1250,1500],
    }
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective, n_trials=5)
logger.close()
