import numpy as np
import pandas as pd
from supersmoother import SuperSmoother

def trimean(array, axis=0):
    quantiles = np.percentile(array, [25, 50, 75], axis=axis)
    return (quantiles[0,:] + 2*quantiles[1,:] + quantiles[2,:])/4


class TrendEstimator():
    
    def __init__(self, primary_bandwidths, middle_bandwidth, final_bandwidth, alpha):
        self.primary_bandwidths = primary_bandwidths
        self.middle_bandwidth = middle_bandwidth
        self.final_bandwidth = final_bandwidth
        self.alpha = alpha
    
    def fit(self, data):
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

    
#class RobustTrendEstimator():
#    
#     def __init__(self, window_lengths=[28,56], window_step=1, alpha=1):
#         self.window_lengths = window_lengths
#         self.window_step = window_step
#         self.alpha = 1
    
#     def fit(self, data):
#         min_window = self.window_lengths[0]
#         max_window = self.window_lengths[1]+1
#         trend_models = list()
#         for window_length in range(min_window, max_window, self.window_step):
#             trend_model = TrendEstimator(window_length=window_length, alpha=self.alpha)
#             trend_model.fit(data)
#             trend_models.append(trend_model)
#         self.trend_models = trend_models
    
#     def predict(self, data):
#         predictions = list()
#         for trend_model in self.trend_models:
#             trend_dataframe = trend_model.predict(data)
#             predictions.append(trend_dataframe.trend.values)
        
#         agg_predictions = trimean(predictions, axis=0)
#         return pd.DataFrame({"ds":data.ds, "trend":agg_predictions})
