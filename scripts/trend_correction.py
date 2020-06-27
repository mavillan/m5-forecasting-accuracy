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
from trend import TrendEstimator

ts_id_columns_by_level = {
    1: [],
    2: ["state_id"],
    3: ["store_id"],
    4: ["cat_id"],
    5: ["dept_id"],
    6: ["state_id", "cat_id"],
    7: ["state_id", "dept_id"],
    8: ["store_id", "cat_id"],
    9: ["store_id", "dept_id"],
    10: ["item_id"],
    11: ["item_id", "state_id"],
    12: ["item_id", "store_id"]
}

def trend_correction(data, predict_data, weights, scales, level, kwargs1, kwargs2):
    ts_uid_columns = ts_id_columns_by_level[level]
    data_agg = data.groupby(["ds"]+ts_uid_columns)[["y","y_pred"]].sum().reset_index()
    start_date = predict_data.ds.min()
    
    errors = list()
    ts_uid_values = data_agg.loc[:, ts_uid_columns].drop_duplicates()

    for _,row in ts_uid_values.iterrows():
        query_string = " & ".join([f"{col} == {value}" for col,value in row.iteritems()])

        df = pd.concat([
            data_agg.query(query_string + "& ds < @start_date").loc[:, ["ds","y"]],
            data_agg.query(query_string + "& ds >= @start_date").loc[:, ["ds","y_pred"]].rename({"y_pred":"y"}, axis=1)],
            ignore_index=True)

        trend_model1 = TrendEstimator(**kwargs1)
        trend_model1.fit(df)
        trend1 = trend_model1.predict(predict_data)

        trend_model2 = TrendEstimator(**kwargs2)
        trend_model2.fit(df.query("ds < @start_date"))
        trend2 = trend_model2.predict(predict_data)

        _df = (data_agg.query(query_string)
               .merge(trend1, on="ds", how="inner")
               .merge(trend2, on="ds", how="inner"))

        _df["correction"] = _df.eval("trend_x - trend_y")

        scale = scales.query(query_string).s.values[0]
        weight = weights.query(query_string).weight.values[0]
        errors.append(weight*compute_rmse(_df.y.values, (_df.y_pred.values - _df.correction.values))/scale)
    
    return np.sum(errors)

###########################################################################################
# level selection
###########################################################################################

LEVEL = 3

###########################################################################################
# logger setting
###########################################################################################

if os.path.exists(f"../results/trend_correction_L{LEVEL}.csv"):
    logger = open(f"../results/trend_correction_L{LEVEL}.csv", "a")
else:
    logger = open(f"../results/trend_correction_L{LEVEL}.csv", "w")
    logger.write("trial;params;errors;error\n")

###########################################################################################
# data prep
###########################################################################################
data = (pd.read_parquet("../input/train_dataframe.parquet")
        .reset_index(drop=True)
        .rename({"q":"y"}, axis=1)
       )

scaling_input = pd.read_parquet("../input/scaling_input.parquet")
weighting_input = pd.read_parquet("../input/weighting_input.parquet")

# periods used for validation
valid_periods = [
    (pd.to_datetime("2015-04-25"), pd.to_datetime("2015-05-22")),
    (pd.to_datetime("2015-05-23"), pd.to_datetime("2015-06-19")),
    (pd.to_datetime("2016-03-28"), pd.to_datetime("2016-04-24")),
    (pd.to_datetime("2016-04-25"), pd.to_datetime("2016-05-22")),
]

# weights and scales for all the validation periods
all_scales = dict()
all_weights = dict()
for fold in range(4):
    all_scales[fold] = compute_scales_by_level(scaling_input, valid_periods[fold][0]) 
    all_weights[fold] = compute_weights_by_level(weighting_input, valid_periods[fold][0])

# forecast for each fold
fold_forecasts = [
    pd.read_csv("../output/forecast-f1.csv", parse_dates=["ds"]),
    pd.read_csv("../output/forecast-f2.csv", parse_dates=["ds"]),
    pd.read_csv("../output/forecast-f3.csv", parse_dates=["ds"]),
    pd.read_csv("../output/forecast-f4.csv", parse_dates=["ds"]), 
]

# input data by forecast
data_by_fold = list()
for fold in range(4):
    end_date = valid_periods[fold][1]
    mrg = pd.merge(data.query("ds <= @end_date").loc[:, ["ds","item_id","dept_id","cat_id","store_id","state_id","y"]],
                   fold_forecasts[fold],
                   how="left", on=["ds","item_id","store_id"])
    data_by_fold.append(mrg)

###########################################################################################
# model config
###########################################################################################

def objective(trial):
    pbw_min1 = trial.suggest_int("pbw_min1", 14, 42)
    pbw_max1 = trial.suggest_int("pbw_max1", pbw_min1+2, 56)
    mbw1 = trial.suggest_int("mbw1", 14, 56)
    fbw1 = trial.suggest_int("fbw1", 14, 56)
    alpha1 = trial.suggest_int("alpha1", 0, 10)
    drop_last_n = trial.suggest_int("drop_last_n", 0, 14)
    kwargs1 = {
        "primary_bandwidths" : np.arange(pbw_min1, pbw_max1),
        "middle_bandwidth": mbw1,
        "final_bandwidth": fbw1,
        "alpha": alpha1,
        "drop_last_n": drop_last_n,
    }
    
    pbw_min2 = trial.suggest_int("pbw_min2", 28, 112)
    pbw_max2 = trial.suggest_int("pbw_max2", pbw_min2+2, 168)
    mbw2 = trial.suggest_int("mbw2", 28, 168)
    fbw2 = trial.suggest_int("fbw2", 28, 168)
    alpha2 = trial.suggest_int("alpha2", 0, 10)
    kwargs2 = {
        "primary_bandwidths" : np.arange(pbw_min2, pbw_max2),
        "middle_bandwidth": mbw2,
        "final_bandwidth": fbw2,
        "alpha": alpha2,
    }

    errors = list()
    for fold in range(4):
        valid_period = valid_periods[fold]
        start_date,end_date = valid_period
        scales_by_level = all_scales[fold]
        weights_by_level = all_weights[fold]
        data = data_by_fold[fold]
        predict_data = make_time_range(start_date, end_date, "D")
        
        error = trend_correction(data, 
                                 predict_data, 
                                 weights_by_level[LEVEL], 
                                 scales_by_level[LEVEL], 
                                 level=LEVEL, 
                                 kwargs1=kwargs1, 
                                 kwargs2=kwargs2)
        errors.append(error)
            
    params = {
        "pbw_min1":pbw_min1,
        "pbw_max1":pbw_max1,
        "mbw1":mbw1,
        "fbw1":fbw1,
        "alpha1":alpha1,
        "drop_last_n":drop_last_n,
        "pbw_min2":pbw_min2,
        "pbw_max2":pbw_max2,
        "mbw2":mbw2,
        "fbw2":fbw2,
        "alpha2":alpha2,}
    mean_error = np.mean(errors)
    print(f"{params};{errors};{mean_error}")

    logger.write(f"{trial.number};{params};{errors};{mean_error}\n")
    logger.flush()
    
    return mean_error

###########################################################################################
# study definition
###########################################################################################

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000000)
logger.close()
