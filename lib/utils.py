import numpy as np
import pandas as pd

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

def reduce_mem_usage(df, verbose=False):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #    df[col] = df[col].astype(np.float16)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def compute_scaling(scaling_input, start_date=None, level=12):
    if start_date is not None:
        scaling_input = scaling_input.query("date < @start_date")
    
    if level==1:
        _scaling_input = (
            scaling_input
            .groupby(["date"])["q"]
            .sum()
            .reset_index())
        return pd.DataFrame([np.sqrt(np.nanmean(_scaling_input.q.diff(1)**2))], columns=["s"])

    ts_id_columns = ts_id_columns_by_level[level]
    _scaling_input = (
        scaling_input
        .groupby(ts_id_columns+["date"])["q"]
        .sum()
        .reset_index())
    scaling = (
        _scaling_input.groupby(ts_id_columns)["q"]
        .apply(lambda x: np.sqrt(np.nanmean(x.diff(1)**2)))
        .reset_index()
        .rename({"q":"s"}, axis=1))
    return scaling

def compute_scales_by_level(scaling_input, start_date):
    scales_by_level = dict()
    for level in range(1,13):
        scales_by_level[level] = compute_scaling(scaling_input, start_date, level)
    return scales_by_level

def compute_weights(weighting_input, start_date, level=12):
    if level==1:
        return None
    if not isinstance(start_date, pd._libs.tslibs.timestamps.Timestamp):
        start_date = pd.to_datetime(start_date)
    
    left_date = start_date - pd.DateOffset(days=28)
    right_date = start_date - pd.DateOffset(days=1)

    ts_id_columns = ts_id_columns_by_level[level]
    weights_df = (weighting_input
                 .query("@left_date <= ds <= @right_date")
                 .groupby(ts_id_columns)["sales"]
                 .sum()
                 .reset_index())
    total = weights_df.sales.sum()
    weights_df["weight"] = weights_df.eval("sales/@total")
    weights_df.drop("sales", axis=1, inplace=True)
    
    return weights_df

def compute_weights_by_level(weighting_input, start_date):
    weights_by_level = dict()
    for level in range(1,13):
        weights_by_level[level] = compute_weights(weighting_input, start_date, level)
    return weights_by_level

def find_out_of_stock(df, threshold=28):
    df = df.copy()
    df["no_stock"] = 0
    zero_mask = (df.q == 0)
    transition_mask = (zero_mask != zero_mask.shift(1))
    zero_sequences = transition_mask.cumsum()[zero_mask]
    idx = zero_sequences[zero_sequences.map(zero_sequences.value_counts()) >= threshold].index 
    df.loc[idx, "no_stock"] = 1
    return df