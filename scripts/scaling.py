#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
import numpy as np 
import pandas as pd 
from category_encoders.ordinal import OrdinalEncoder
import matplotlib.pyplot as plt
from tsforest.utils import make_time_range

import matplotlib.pyplot as plt
import seaborn as sns

# local modules
import sys
sys.path.append("../lib/")
from utils import compute_scaling, reduce_mem_usage


# ***
# ## data loading

# In[3]:


sales_train = pd.read_csv("../input/sales_train_validation.csv")
sales_train.info()


# In[14]:


calendar = pd.read_csv("../input/calendar.csv", parse_dates=["date"])
calendar.info()


# ***
# ## hierarchy

# In[4]:


sales_train["id"] = sales_train.id.map(lambda x: x.replace("_validation", ""))
hierarchy = (sales_train.loc[:, ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]]
             .drop_duplicates())


# In[5]:


# hierarchy encoder
id_encoder = OrdinalEncoder()
id_encoder.fit(hierarchy.loc[:, ["id"]])
hierarchy["ts_id"]  = id_encoder.transform(hierarchy.loc[:, ["id"]])

item_encoder = OrdinalEncoder()
item_encoder.fit(hierarchy.loc[:, ["item_id"]])
hierarchy.loc[:, "item_id"]  = item_encoder.transform(hierarchy.loc[:, ["item_id"]])

dept_encoder = OrdinalEncoder()
dept_encoder.fit(hierarchy.loc[:, ["dept_id"]])
hierarchy.loc[:, "dept_id"]  = dept_encoder.transform(hierarchy.loc[:, ["dept_id"]])

cat_encoder = OrdinalEncoder()
cat_encoder.fit(hierarchy.loc[:, ["cat_id"]])
hierarchy.loc[:, "cat_id"]   = cat_encoder.transform(hierarchy.loc[:, ["cat_id"]])

store_encoder = OrdinalEncoder()
store_encoder.fit(hierarchy.loc[:, ["store_id"]])
hierarchy.loc[:, "store_id"] = store_encoder.transform(hierarchy.loc[:, ["store_id"]])

state_encoder = OrdinalEncoder()
state_encoder.fit(hierarchy.loc[:, ["state_id"]])
hierarchy.loc[:, "state_id"] = state_encoder.transform(hierarchy.loc[:, ["state_id"]])


# ***
# ## categorical encoding

# In[6]:


sales_train["ts_id"] = id_encoder.transform(sales_train.loc[:, ["id"]])
sales_train.loc[:, "item_id"]  = item_encoder.transform(sales_train.loc[:, ["item_id"]])
sales_train.loc[:, "dept_id"]  = dept_encoder.transform(sales_train.loc[:, ["dept_id"]])
sales_train.loc[:, "cat_id"]   = cat_encoder.transform(sales_train.loc[:, ["cat_id"]])
sales_train.loc[:, "store_id"] = store_encoder.transform(sales_train.loc[:, ["store_id"]])
sales_train.loc[:, "state_id"] = state_encoder.transform(sales_train.loc[:, ["state_id"]])


# ***
# ## data wrangling

# In[15]:


data = pd.melt(sales_train, 
               id_vars=["ts_id","item_id","dept_id","cat_id","store_id","state_id"],
               value_vars=[f"d_{i}" for i in range(1,1914)],
               var_name="d",
               value_name="q")
data = pd.merge(data, 
                calendar.loc[:, ["d","date"]],
                how="left",
                on="d")
data.drop("d", axis=1, inplace=True)


# In[18]:


data = reduce_mem_usage(data)
gc.collect()


# In[26]:


data.info()


# ***
# ## cleaning

# 
# ### removes zeros at the start of the time series

# In[27]:


def remove_starting_zeros(dataframe):
    idxmin = dataframe.query("q > 0").index.min()
    return dataframe.loc[idxmin:, :]


# In[28]:


data = (data
        .groupby(["item_id","store_id"])
        .apply(remove_starting_zeros)
        .reset_index(drop=True)
       )


# In[72]:


data.info()


# In[77]:


data.to_parquet("../input/scaling_input.parquet", index=False)


# ***
