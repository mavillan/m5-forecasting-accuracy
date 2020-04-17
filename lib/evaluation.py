import pandas as pd

class WRMSSEEvaluator(object):
    def __init__(self, valid_dataframe, weights_dataframe, scales_dataframe, ts_uid_columns):
        self.valid_dataframe = valid_dataframe
        self.weights_dataframe = weights_dataframe
        self.scales_dataframe = scales_dataframe
        self.ts_uid_columns = ts_uid_columns
    
    def _evaluate(self, predictions):
        valid_dataframe = self.valid_dataframe.copy()
        valid_dataframe["ypred"] = predictions
        valid_dataframe["sq_error"] = valid_dataframe.eval("(y-ypred)**2")
        return (valid_dataframe
                .groupby(self.ts_uid_columns)["sq_error"]
                .mean()
                .reset_index(name="mse")
                .merge(self.scales_dataframe, how="left", on=self.ts_uid_columns)
                .merge(self.weights_dataframe, how="left", on=self.ts_uid_columns)
                .assign(weight = lambda x: x.weight/x.weight.sum())
                .eval("weight * (sqrt(mse)/s)")
                .sum())
    
    def evaluate(self, ypred, dtrain):
        metric = self._evaluate(ypred)
        return "wrmsse", metric, False