import os
import gc
import time
import numba
import numpy as np; np.random.seed(42)
import pandas as pd
import lightgbm as lgb
import category_encoders as ce
import click
from tsforest.forecaster import LightGBMForecaster
from scipy.stats import trim_mean

# local modules
import sys
sys.path.append("../lib/")
from utils import (compute_scaling, compute_weights, reduce_mem_usage, 
                   compute_scales_by_level, compute_weights_by_level)
from evaluation import _WRMSSEEvaluator, WRMSSEEvaluator, Evaluator, WRMSSEEvaluatorL12

@numba.jit(nopython=True, nogil=True, fastmath=True)
def compute_czeros(x):
    return np.sum(np.cumprod((x==0)[::-1]))/x.shape[0]

@numba.jit(nopython=True, nogil=True, fastmath=True)
def compute_sfreq(x):
    return np.sum(x!=0)/x.shape[0]

SEEDS = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 
         43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
NUM_ITER_RANGE = (500,701)


@click.command()
@click.option("-a", "--approach", required=True, type=int)
def main(approach):
    # loads training data
    data = (pd.read_parquet("../input/train_dataframe.parquet")
            .rename({"q":"y"}, axis=1))
    data["sales"] = data.eval("y * sell_price")

    # loads evaluation data
    eval_data = pd.read_parquet("../input/eval_dataframe.parquet")

    # model configuration
    model_params = {
        'objective':'tweedie',
        'tweedie_variance_power': 1.1,
        'metric':'None',
        'max_bin': 127,
        'bin_construct_sample_cnt':20000000,
        'num_leaves': 2**10-1,
        'min_data_in_leaf': 2**10-1,
        'learning_rate': 0.05,
        'feature_fraction':0.8,
        'bagging_fraction':0.8,
        'bagging_freq':1,
        'lambda_l2':0.1,
        'boost_from_average': False,
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
        "store_id": "default",
        "state_id": "default",
        "dept_id": "default",
        "cat_id": "default",
        "event_name_1": "default",}
    if approach == 1:
        categorical_features["item_id"] = "default"
    elif approach == 2:
        categorical_features["item_id"] = ("y", ce.GLMMEncoder, None)
    else:
        print("Invalid input.")
        sys.exit(0)
    
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

    print("Building features")
    tic = time.time()
    model_level12_base = LightGBMForecaster(**model_kwargs)
    model_level12_base.prepare_features(train_data=data)
    model_level12_base.train_features.dropna(subset=lagged_features_to_dropna, axis=0, inplace=True)
    model_level12_base.train_features = reduce_mem_usage(model_level12_base.train_features)
    gc.collect()
    tac = time.time()
    print(f"Elapsed time: {(tac-tic)/60.} [min]")

    for i,seed in enumerate(SEEDS):    
        num_iterations = np.random.randint(*NUM_ITER_RANGE)
        print("#"*100)
        print(f" model {i+1}/{len(SEEDS)} - seed: {seed} - num_iterations: {num_iterations} ".center(100, "#"))
        print("#"*100)
        
        model_level12 = copy.deepcopy(model_level12_base)
        model_params["seed"] = seed
        model_params["num_iterations"] = num_iterations
        model_level12.set_params(model_params)

        print("Fitting the model")
        tic = time.time()
        model_level12.fit()
        tac = time.time()
        print(f"Elapsed time: {(tac-tic)/60.} [min]")
        
        # thresh_value
        predict_data = model_level12.train_features.query("no_stock_days >= 28").loc[:, model_level12.input_features]
        predictions = model_level12.model.model.predict(predict_data)
        thresh_value = trim_mean(predictions, proportiontocut=0.05)
        def bias_corr_func(x, tv=thresh_value):
            x[x < tv] = 0
            return x

        print("Predicting")
        tic = time.time()
        forecast = model_level12.predict(eval_data, recursive=True, bias_corr_func=bias_corr_func)
        tac = time.time()
        print(f"Elapsed time: {(tac-tic)/60.} [min]")

        forecast.to_csv(f"../output/forecast_app{approach}_seed{seed}_niter{num_iterations}.csv", index=False)

        del model_level12, forecast, predict_data, predictions
        gc.collect()

if __name__ == "__main__":
    main()