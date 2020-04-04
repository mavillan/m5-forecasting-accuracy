import numpy as np
import pandas as pd

def compute_scaling(data, cut_date=None, agg_columns="ts_id"):
    if cut_date is not None:
        data = data.query("date < @cut_date")
    _data = (data
             .groupby(agg_columns+["date"])["q"]
             .sum()
             .reset_index())
    scaling = (_data.groupby(agg_columns)["q"]
               .apply(lambda x: np.nanmean(x.diff(1)**2))
               .reset_index())
    return scaling