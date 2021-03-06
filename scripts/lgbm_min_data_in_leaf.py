import os
import gc
import time
import pickle
import numpy as np; np.random.seed(42)
import pandas as pd
from datetime import datetime
import lightgbm as lgb
import category_encoders as ce
import matplotlib.pyplot as plt
import optuna

from tsforest.trend import compute_trend_models
from tsforest.forecaster import LightGBMForecaster
from tsforest.utils import make_time_range
from tsforest.metrics import compute_rmse

# local modules
import sys
sys.path.append("../lib/")
from utils import compute_scaling, compute_weights, reduce_mem_usage
from evaluation import _WRMSSEEvaluator, WRMSSEEvaluator, Evaluator, WRMSSEEvaluatorL12

###########################################################################################
# logger setting
###########################################################################################

if os.path.exists(f"../results/lgbm_min_data_in_leaf.csv"):
    logger = open(f"../results/lgbm_min_data_in_leaf.csv", "a")
else:
    logger = open(f"../results/lgbm_min_data_in_leaf.csv", "w")
    logger.write("trial;params;wrmsse;wrmsseL12;error\n")

###########################################################################################
# data prep
###########################################################################################
data = (pd.read_parquet("../input/train_dataframe.parquet")
        .reset_index(drop=True)
        .rename({"q":"y"}, axis=1)
       )

scaling_input = pd.read_parquet("../input/scaling_input.parquet")
weighting_input = pd.read_parquet("../input/weighting_input.parquet")

data.info()

# periods used for validation
valid_periods = [
    (pd.to_datetime("2015-04-25"), pd.to_datetime("2015-05-22")),
    (pd.to_datetime("2015-05-23"), pd.to_datetime("2015-06-19")),
    (pd.to_datetime("2016-03-28"), pd.to_datetime("2016-04-24")),
    (pd.to_datetime("2016-04-25"), pd.to_datetime("2016-05-22")),
]

###########################################################################################
# definition of objective
###########################################################################################

default_model_params = {
    'objective':'tweedie',
    'tweedie_variance_power': 1.1,
    'metric':'None',
    'num_iterations':100000,
    'early_stopping_rounds':300,
    'max_bin': 127,
    'bin_construct_sample_cnt':6000000,
    'num_leaves': 511,
    #'min_data_in_leaf': 2**9-1,
    'learning_rate': 0.03, 
    'feature_fraction': 0.7,
    'bagging_fraction':0.9,
    'bagging_freq':1,
    'lambda_l2':0.1,
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
    "store_id",
    "state_id",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    
    "prev_newyear",
    "post_newyear",
    "prev_lw",
    "post_lw",
    "post_christmas",
    "post_thanksgiving",
    
    "no_stock_days",
    "sales",
]

model_kwargs = {
    "time_features":time_features,
    "window_shifts":[1, 7, 28, 56],
    "window_functions":["mean", "median", "std", "kurt",],
    "window_sizes":[7, 28],
    "exclude_features":exclude_features,
    "categorical_features":{
        "item_id": ("y", ce.GLMMEncoder, None),
        "dept_id": "default",
        "cat_id": "default",
        "event_name_1": "default",
    },
    "ts_uid_columns":["item_id",],
}

lagged_features = list()
if "lags" in model_kwargs.keys():
    lag_features = [f"lag{lag}" for lag in model_kwargs["lags"]]
    lagged_features.extend(lag_features)
if "window_functions" in model_kwargs.keys():
    rw_features = [f"{window_func}{window_size}_shift{window_shift}" 
                   for window_func in model_kwargs["window_functions"]
                   for window_size in model_kwargs["window_sizes"]
                   for window_shift in model_kwargs["window_shifts"]]
    lagged_features.extend(rw_features)
    
lagged_features_to_dropna = [feat for feat in lagged_features if feat not in ["skew", "kurt"]]


def objective(trial):
    sampled_params = {"min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 15, 2047)}
    model_params = {**default_model_params, **sampled_params}
    model_kwargs["model_params"] = model_params
    
    errors = list()
    errors_l12 = list()
    
    for i,valid_period in enumerate(valid_periods):

        print("#"*100)
        print(f" Validation period: {valid_period} ".center(100, "#"))
        print("#"*100)

        valid_start = valid_period[0]
        valid_end = valid_period[1]
        scales_level12 = compute_scaling(scaling_input, 
                                         cut_date=valid_start,
                                         agg_columns=["item_id","store_id"]).rename({"q":"s"}, axis=1)
        weights_level12 = compute_weights(weighting_input, 
                                          start_date=valid_start, level=12)

        stores_forecast = list()

        for store_id in range(1,11):
            print("-"*100)
            print(f" store_id: {store_id} ".center(100, "-"))
            print("-"*100)

            _train_data = data.query("ds <= @valid_end & store_id == @store_id").reset_index(drop=True)
            _valid_index = _train_data.query("@valid_start <= ds <= @valid_end").index

            if store_id in [1,2,3,4]:
                # CA store
                _train_data.drop(["snap_TX", "snap_TX_cum", "snap_WI", "snap_WI_cum"], axis=1, inplace=True)
            elif store_id in [5,6,7]:
                # TX store
                _train_data.drop(["snap_CA", "snap_CA_cum", "snap_WI", "snap_WI_cum"], axis=1, inplace=True)
            else:
                #WI store
                _train_data.drop(["snap_TX", "snap_TX_cum", "snap_CA", "snap_CA_cum"], axis=1, inplace=True)

            model_level12 = LightGBMForecaster(**model_kwargs)
            model_level12.prepare_features(train_data=_train_data, valid_index=_valid_index)
            model_level12.train_features.dropna(subset=lagged_features_to_dropna, axis=0, inplace=True)
            model_level12.train_features = reduce_mem_usage(model_level12.train_features)
            model_level12.valid_features = reduce_mem_usage(model_level12.valid_features)
            ts_id_in_train = model_level12.train_features.ts_id.unique()
            model_level12.valid_features = model_level12.valid_features.query("ts_id in @ts_id_in_train")

            print("Fitting the model")
            tic = time.time()
            evaluator = WRMSSEEvaluatorL12(model_level12.valid_features, weights_level12, scales_level12)
            model_level12.fit(fit_kwargs={"verbose_eval":25, "feval":evaluator.evaluate})
            tac = time.time()
            print(f"Elapsed time: {(tac-tic)/60.} [min]")

            print("Predicting with recursive approach")
            tic = time.time()
            valid_data = model_level12.valid_features.loc[:, model_level12.raw_train_columns].drop("y", axis=1)
            forecast_v1 = model_level12.predict(valid_data, recursive=True)
            tac = time.time()
            print(f"Elapsed time: {(tac-tic)/60.} [min]")

            wrmsse = evaluator._evaluate(forecast_v1.y_pred.values)
            print("wrmsse:", wrmsse)

            forecast_v1["store_id"] = store_id
            stores_forecast.append(forecast_v1)
            del model_level12, _train_data, _valid_index, evaluator
            gc.collect()

        fold_forecast = pd.concat(stores_forecast, ignore_index=True)
        mrg = pd.merge(data.loc[:, ["ds","item_id","dept_id","cat_id","store_id","state_id","y"]],
                       fold_forecast, how="inner", on=["ds","item_id","store_id"])
        evaluator = WRMSSEEvaluator(mrg.loc[:, ["ds","item_id","dept_id","cat_id","store_id","state_id","y"]])
        wrmsse = evaluator._evaluate(mrg.y_pred.values)
        
        errors.append(wrmsse)
        errors_l12.append(evaluator.errors_by_level[("item_id","store_id")])

    error = np.mean(errors_l12)
    logger.write(f"{trial.number};{model_params};{errors};{errors_l12};{error}\n")
    logger.flush()
    
    return error

###########################################################################################
# study definition
###########################################################################################
search_space = {
    'min_data_in_leaf': [15, 31, 63, 127, 255, 511, 1023, 2047],
    }
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective, n_trials=8)
logger.close()
