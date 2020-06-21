import numpy as np
import pandas as pd
import optuna
from tsforest.utils import make_time_range
from tsforest.metrics import compute_rmse

# local modules
import os
import sys
sys.path.append("../lib/")
from utils import compute_scaling, compute_weights, reduce_mem_usage
from evaluation import _WRMSSEEvaluator, WRMSSEEvaluator, Evaluator, WRMSSEEvaluatorL12
from trend import TrendEstimator, RobustTrendEstimator

###########################################################################################
# logger setting
###########################################################################################

if os.path.exists(f"../results/trend_correction_by_store.csv"):
    logger = open(f"../results/trend_correction_by_store.csv", "a")
else:
    logger = open(f"../results/trend_correction_by_store.csv", "w")
    logger.write("trial;params;errors;error\n")

###########################################################################################
# data loading
###########################################################################################

data = (pd.read_parquet("../input/train_dataframe.parquet")
        .reset_index(drop=True)
        .rename({"q":"y"}, axis=1))

scaling_input = pd.read_parquet("../input/scaling_input.parquet")
weighting_input = pd.read_parquet("../input/weighting_input.parquet")


###########################################################################################
# data preparation
###########################################################################################

forecast_folds = [pd.read_csv("../output/forecast-f1.csv", parse_dates=["ds"]),
                  pd.read_csv("../output/forecast-f2.csv", parse_dates=["ds"]),
                  pd.read_csv("../output/forecast-f3.csv", parse_dates=["ds"]),
                  pd.read_csv("../output/forecast-f4.csv", parse_dates=["ds"]),
                 ]

mrg_folds = list()
for forecast in forecast_folds:
    max_date = forecast.ds.max()
    mrg = pd.merge(data.query("ds <= @max_date").loc[:, ["ds","item_id","dept_id","cat_id","store_id","state_id","y"]],
                   forecast,
                   how="left", on=["ds","item_id","store_id"])
    mrg_folds.append(mrg)
    
mrg_level3_folds = list()
for mrg in mrg_folds:
    mrg_level3 = mrg.groupby(["ds","store_id"])[["y","y_pred"]].sum().reset_index()
    mrg_level3_folds.append(mrg_level3)
    
predict_folds = list()
for forecast in forecast_folds:
    min_date = forecast.ds.min()
    max_date = forecast.ds.max()
    predict_data = make_time_range(min_date, max_date, "D")
    predict_folds.append(predict_data)
    
weights_folds = list()
for forecast in forecast_folds:
    min_date = forecast.ds.min()
    weights = compute_weights(weighting_input, min_date, level=3)
    weights_folds.append(weights)
    
scales_folds = list()
for forecast in forecast_folds:
    min_date = forecast.ds.min()
    scales = compute_scaling(scaling_input, min_date, agg_columns=["store_id"]).rename({"q":"s"}, axis=1)
    scales_folds.append(scales)

    
###########################################################################################
# definition of objective
###########################################################################################

def objective(trial):
    win_min = trial.suggest_int("win_min", 7, 28)
    win_length_min = trial.suggest_int("win_length_min", 28, 55)
    win_length_max = trial.suggest_int("win_length_max", 56, 364)
    window_step = trial.suggest_int("window_step", 1, 14)

    errors_folds = list()

    for fold in range(4):
        mrg_level3 = mrg_level3_folds[fold]
        predict_data = predict_folds[fold]
        min_date = predict_data.ds.min()
        max_date = predict_data.ds.max()
        scales = scales_folds[fold]
        weights = weights_folds[fold]
        
        errors_cor = list()
        for store_id in mrg_level3.store_id.unique():
            df = pd.concat([
                mrg_level3.query("store_id == @store_id & ds < @min_date").loc[:, ["ds","y"]],
                mrg_level3.query("store_id == @store_id & ds >= @min_date").loc[:, ["ds","y_pred"]].rename({"y_pred":"y"}, axis=1)],
                ignore_index=True)

            trend_model1 = TrendEstimator(window_length=win_min, alpha=1)
            trend_model1.fit(df)
            trend1 = trend_model1.predict(predict_data)

            trend_model2 = RobustTrendEstimator(window_lengths=[win_length_min,win_length_max], window_step=window_step, alpha=1)
            trend_model2.fit(df.query("ds < @min_date"))
            trend2 = trend_model2.predict(predict_data)

            _df = (mrg_level3.query("store_id == @store_id")
                   .merge(trend1, on="ds", how="inner")
                   .merge(trend2, on="ds", how="inner"))

            _df["correction"] = _df.eval("trend_x - trend_y")

            scale = scales.query("store_id == @store_id").s.values[0]
            weight = weights.query("store_id == @store_id").weight.values[0]

            errors_cor.append( weight*compute_rmse(_df.y.values, (_df.y_pred.values - _df.correction.values))/scale )

        errors_folds.append(np.sum(errors_cor))
    
    mean_error = np.mean(errors_folds)
    params = {
        "win_min":win_min,
        "win_length_min":win_length_min, 
        "win_length_max":win_length_max, 
        "window_step":window_step}

    logger.write(f"{trial.number};{params};{errors_folds};{mean_error}\n")
    logger.flush()
    
    return mean_error


###########################################################################################
# study definition
###########################################################################################

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100000000000)
logger.close()
    
    