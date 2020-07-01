import numpy as np
import pandas as pd
from tsforest.utils import make_time_range
from supersmoother import SuperSmoother

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

def trimean(array, axis=0):
    quantiles = np.percentile(array, [25, 50, 75], axis=axis)
    return (quantiles[0,:] + 2*quantiles[1,:] + quantiles[2,:])/4


class TrendEstimator():
    
    def __init__(self, primary_bandwidths, middle_bandwidth, final_bandwidth, alpha, drop_last_n=None):
        self.primary_bandwidths = primary_bandwidths
        self.middle_bandwidth = middle_bandwidth
        self.final_bandwidth = final_bandwidth
        self.alpha = alpha
        self.drop_last_n = drop_last_n
    
    def fit(self, data):
        if self.drop_last_n is not None: 
            if self.drop_last_n > 0:
                data = data.iloc[:-self.drop_last_n].copy() 
        min_date = data.ds.min()
        time_idx = (data.ds - min_date).apply(lambda x: x.days).values
        time_values = data.y.values
        
        # from bandwidths to spans
        kwargs = {
            "alpha":self.alpha,
            "primary_spans":self.primary_bandwidths/data.shape[0],
            "middle_span":self.middle_bandwidth/data.shape[0],
            "final_span":self.final_bandwidth/data.shape[0],   
        }
        trend_model = SuperSmoother(**kwargs)
        trend_model.fit(time_idx, time_values)
        
        self.min_date = min_date
        self.trend_model = trend_model
    
    def predict(self, data):
        time_idx = (data.ds - self.min_date).apply(lambda x: x.days).values
        trend = self.trend_model.predict(time_idx)
        return pd.DataFrame({"ds":data.ds, "trend":trend})


def apply_trend_correction(data, forecast, level, kwargs1, kwargs2):
    ts_uid_columns = ts_id_columns_by_level[level]
    start_date = forecast.ds.min()
    end_date = forecast.ds.max()
    predict_data = make_time_range(start_date, end_date, "D")
    
    mrg = pd.merge(data.query("ds <= @end_date").loc[:, ["ds","item_id","dept_id","cat_id","store_id","state_id","y"]],
                   forecast,
                   how="left", on=["ds","item_id","store_id"])
    mrg_agg = mrg.groupby(["ds"]+ts_uid_columns)[["y","y_pred"]].sum().reset_index()
    
    ts_uid_values = mrg_agg.loc[:, ts_uid_columns].drop_duplicates()
    corrected_dataframes = list()
    
    for _,row in ts_uid_values.iterrows():
        query_string = " & ".join([f"{col} == {value}" for col,value in row.iteritems()])

        df = pd.concat([
            mrg_agg.query(query_string + "& ds < @start_date").loc[:, ["ds","y"]],
            mrg_agg.query(query_string + "& ds >= @start_date").loc[:, ["ds","y_pred"]].rename({"y_pred":"y"}, axis=1)],
            ignore_index=True)

        trend_model1 = TrendEstimator(**kwargs1)
        trend_model1.fit(df)
        trend1 = trend_model1.predict(predict_data)

        trend_model2 = TrendEstimator(**kwargs2)
        trend_model2.fit(df.query("ds < @start_date"))
        trend2 = trend_model2.predict(predict_data)

        _df = (mrg_agg.query(query_string)
               .merge(trend1, on="ds", how="inner")
               .merge(trend2, on="ds", how="inner"))
        _df["y_pred"] -= _df.eval("trend_x - trend_y")
        corrected_dataframes.append(_df.loc[:, ["ds","store_id","y_pred"]])
        
    return pd.concat(corrected_dataframes, ignore_index=True)


def apply_robust_trend_correction(data, forecast, level, kwargs_list):
    ts_uid_columns = ts_id_columns_by_level[level]
    start_date = forecast.ds.min()
    end_date = forecast.ds.max()
    predict_data = make_time_range(start_date, end_date, "D")
        
    mrg_agg = pd.concat([data.query("ds < @start_date").groupby(["ds"]+ts_uid_columns)["y"].sum().reset_index(),
                         forecast.groupby(["ds"]+ts_uid_columns)["y_pred"].sum().reset_index().rename({"y_pred":"y"}, axis=1)],
                        ignore_index=False)
    
    ts_uid_values = mrg_agg.loc[:, ts_uid_columns].drop_duplicates()
    corrected_dataframes = list()
    
    for _,row in ts_uid_values.iterrows():
        query_string = " & ".join([f"{col} == {value}" for col,value in row.iteritems()])
        df = mrg_agg.query(query_string).copy()
        
        all_corrections = list()
        for kwargs1,kwargs2 in kwargs_list:
            trend_model1 = TrendEstimator(**kwargs1)
            trend_model1.fit(df)
            trend1 = trend_model1.predict(predict_data)

            trend_model2 = TrendEstimator(**kwargs2)
            trend_model2.fit(df.query("ds < @start_date"))
            trend2 = trend_model2.predict(predict_data)

            all_corrections.append(trend1.trend.values - trend2.trend.values)

        trend_correction = trimean(all_corrections, axis=0)
        _df = mrg_agg.query(query_string + "& @start_date <= ds <= @end_date").copy()
        _df["y"] -= trend_correction
        _df.rename({"y":"y_pred"}, axis=1, inplace=True)
        corrected_dataframes.append(_df.loc[:, ["ds"]+ts_uid_columns+["y_pred"]])
        
    return pd.concat(corrected_dataframes, ignore_index=True)
