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

class _WRMSSEEvaluator(object):
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
    
    
class WRMSSEEvaluatorL12(object):
    def __init__(self, valid_dataframe, weights_dataframe=None, scales_dataframe=None):
        self.valid_dataframe = valid_dataframe
        self.weights_dataframe = weights_by_level[12] if weights_dataframe is None else weights_dataframe
        self.scales_dataframe = scales_by_level[12] if scales_dataframe is None else scales_dataframe
        self.ts_uid_columns = ["item_id","store_id"]
    
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
        return "wrmsseL12", metric, False
    

class WRMSSEEvaluator(object):
    def __init__(self, valid_dataframe, weights_by_level=None, scales_by_level=None, single_store=False):
        self.valid_dataframe = valid_dataframe
        self.weights_by_level = weights_by_level if weights_by_level is None else weights_by_level
        self.scales_by_level = scales_by_level if scales_by_level is None else scales_by_level
        self.single_store = single_store
    
    def _evaluate(self, predictions):
        valid_dataframe = self.valid_dataframe.copy()
        valid_dataframe["ypred"] = predictions
        errors_by_level = dict()
        
        if not self.single_store: 
            levels_to_iterate = range(2,13)
            # computation for level1
            scales = self.scales_by_level[1]
            mse = (valid_dataframe
                .groupby(["ds"])[["y","ypred"]]
                .sum()
                .reset_index()
                .eval("(y-ypred)**2")
                .mean())
            errors_by_level["root"] = np.sqrt(mse)/scales.s[0]
        else:
            levels_to_iterate = [3,8,9,12]
        
        for level in levels_to_iterate:
            ts_id_columns = ts_id_columns_by_level[level]
            scales_dataframe = self.scales_by_level[level]
            weights_dataframe = self.weights_by_level[level]
            error = (valid_dataframe
                     .groupby(["ds"]+ts_id_columns)[["y","ypred"]]
                     .sum()
                     .reset_index()
                     .assign(sq_error = lambda x: x.eval("(y-ypred)**2"))
                     .groupby(ts_id_columns)["sq_error"]
                     .mean()
                     .reset_index(name="mse")
                     .merge(scales_dataframe, how="left", on=ts_id_columns)
                     .merge(weights_dataframe, how="left", on=ts_id_columns)
                     .assign(weight = lambda x: x.weight/x.weight.sum())
                     .eval("weight * (sqrt(mse)/s)")
                     .sum())
            errors_by_level[tuple(ts_id_columns)] = error
            
        self.errors_by_level = errors_by_level
        return np.mean(list(errors_by_level.values()))
    
    def evaluate(self, ypred, dtrain):
        metric = self._evaluate(ypred)
        return "wrmsse", metric, False
    

class Evaluator(object):
    def __init__(self, valid_dataframe, weights_by_level=None, scales_by_level=None, single_store=False):
        self.valid_dataframe = valid_dataframe
        self.weights_by_level = weights_by_level if weights_by_level is None else weights_by_level
        self.scales_by_level = scales_by_level if scales_by_level is None else scales_by_level
        self.single_store = single_store
        
        self.eval1 = WRMSSEEvaluator(valid_dataframe, weights_by_level, scales_by_level, single_store)
        self.eval2 = WRMSSEEvaluatorL12(valid_dataframe, weights_by_level[12], scales_by_level[12])
    
    def evaluate(self, ypred, dtrain):
        return [self.eval1.evaluate(ypred, dtrain), self.eval2.evaluate(ypred, dtrain)]
