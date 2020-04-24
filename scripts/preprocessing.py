#!/usr/bin/env python
# coding: utf-8

# In[33]:


import gc
import pickle
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


# In[4]:


calendar = pd.read_csv("../input/calendar.csv", parse_dates=["date"])
calendar.info()


# In[5]:


sell_prices = pd.read_csv("../input/sell_prices.csv")
sell_prices.info()


# ***
# ## hierarchy

# In[6]:


sales_train["id"] = sales_train.id.map(lambda x: x.replace("_validation", ""))
hierarchy = (sales_train.loc[:, ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]]
             .drop_duplicates())
encoders = dict()


# In[7]:


hierarchy.to_parquet("../input/hierarchy_raw.parquet", index=False)


# In[8]:


# hierarchy encoder
id_encoder = OrdinalEncoder()
id_encoder.fit(hierarchy.loc[:, ["id"]])
hierarchy["ts_id"]  = id_encoder.transform(hierarchy.loc[:, ["id"]])
encoders["id"] = id_encoder

item_encoder = OrdinalEncoder()
item_encoder.fit(hierarchy.loc[:, ["item_id"]])
hierarchy.loc[:, "item_id"]  = item_encoder.transform(hierarchy.loc[:, ["item_id"]])
encoders["item"] = item_encoder

dept_encoder = OrdinalEncoder()
dept_encoder.fit(hierarchy.loc[:, ["dept_id"]])
hierarchy.loc[:, "dept_id"]  = dept_encoder.transform(hierarchy.loc[:, ["dept_id"]])
encoders["dept"] = dept_encoder

cat_encoder = OrdinalEncoder()
cat_encoder.fit(hierarchy.loc[:, ["cat_id"]])
hierarchy.loc[:, "cat_id"]   = cat_encoder.transform(hierarchy.loc[:, ["cat_id"]])
encoders["cat"] = cat_encoder

store_encoder = OrdinalEncoder()
store_encoder.fit(hierarchy.loc[:, ["store_id"]])
hierarchy.loc[:, "store_id"] = store_encoder.transform(hierarchy.loc[:, ["store_id"]])
encoders["store"] = store_encoder

state_encoder = OrdinalEncoder()
state_encoder.fit(hierarchy.loc[:, ["state_id"]])
hierarchy.loc[:, "state_id"] = state_encoder.transform(hierarchy.loc[:, ["state_id"]])
encoders["state"] = state_encoder


# In[9]:


hierarchy.to_parquet("../input/hierarchy.parquet", index=False)


# In[10]:


outfile = open("../input/encoders.pkl", "wb")
pickle.dump(encoders, outfile)
outfile.close()


# ***
# ## calendar events encoding

# In[11]:


event_name_1_encoder = OrdinalEncoder()
event_name_1_encoder.fit(calendar.loc[:, ["event_name_1"]])
calendar.loc[:, "event_name_1"] = event_name_1_encoder.transform(calendar.loc[:, ["event_name_1"]])


# In[12]:


event_type_1_encoder = OrdinalEncoder()
event_type_1_encoder.fit(calendar.loc[:, ["event_type_1"]])
calendar.loc[:, "event_type_1"] = event_type_1_encoder.transform(calendar.loc[:, ["event_type_1"]])


# In[13]:


event_name_2_encoder = OrdinalEncoder()
event_name_2_encoder.fit(calendar.loc[:, ["event_name_2"]])
calendar.loc[:, "event_name_2"] = event_name_2_encoder.transform(calendar.loc[:, ["event_name_2"]])


# In[14]:


event_type_2_encoder = OrdinalEncoder()
event_type_2_encoder.fit(calendar.loc[:, ["event_type_2"]])
calendar.loc[:, "event_type_2"] = event_type_2_encoder.transform(calendar.loc[:, ["event_type_2"]])


# ***
# ## categorical encoding

# In[15]:


sales_train["ts_id"] = id_encoder.transform(sales_train.loc[:, ["id"]])
sales_train.loc[:, "item_id"]  = item_encoder.transform(sales_train.loc[:, ["item_id"]])
sales_train.loc[:, "dept_id"]  = dept_encoder.transform(sales_train.loc[:, ["dept_id"]])
sales_train.loc[:, "cat_id"]   = cat_encoder.transform(sales_train.loc[:, ["cat_id"]])
sales_train.loc[:, "store_id"] = store_encoder.transform(sales_train.loc[:, ["store_id"]])
sales_train.loc[:, "state_id"] = state_encoder.transform(sales_train.loc[:, ["state_id"]])


# In[16]:


sell_prices.loc[:, "store_id"] = store_encoder.transform(sell_prices.loc[:, ["store_id"]])
sell_prices.loc[:, "item_id"]  = item_encoder.transform(sell_prices.loc[:, ["item_id"]]) 


# ***
# ## building price features

# In[17]:


number_prices = (sell_prices
                 .groupby(["store_id", "item_id"])["sell_price"]
                 .apply(lambda x: len(np.unique(x)))
                 .reset_index(name="n_prices")
                )


# In[18]:


number_prices.n_prices.describe()


# In[19]:


sell_prices.wm_yr_wk.nunique()


# La mayoría de item-stores no tiene más de 4 precios diferentes, y el 50% no tiene más de 2 precios. El que más tiene, tiene 21 precios sobre un rango de 282 semanas.

# In[20]:


sell_prices.query("item_id == 1 & store_id == 1").sell_price.value_counts()


# In[21]:


regular_prices = (
    sell_prices
    .groupby(["store_id", "item_id"])["sell_price"]
    .apply(lambda x: x.value_counts().index[0])
    .reset_index(name="regular_price")
)


# In[22]:


sell_prices = (
    sell_prices
    .merge(regular_prices, how="left")
    .assign(discount = lambda x: x.regular_price - x.sell_price)
    .assign(discount_porc = lambda x: (x.regular_price - x.sell_price)/x.sell_price)
)


# In[23]:


sell_prices.head()


# ***
# ## data wrangling

# In[24]:


data = pd.melt(sales_train, 
               id_vars=["ts_id","item_id","dept_id","cat_id","store_id","state_id"],
               value_vars=[f"d_{i}" for i in range(1,1914)],
               var_name="d",
               value_name="q")


# In[25]:


calendar_columns = ["date", "wm_yr_wk", "d", "snap_CA", "snap_TX", "snap_WI",
                    "event_name_1", "event_type_1", "event_name_2", "event_type_2"]

data = pd.merge(data, 
                calendar.loc[:, calendar_columns],
                how="left",
                on="d")


# In[26]:


data = pd.merge(data, sell_prices,
                on=["store_id", "item_id", "wm_yr_wk"],
                how="left")


# In[27]:


data.sort_values(["item_id","store_id","date"], inplace=True, ignore_index=True)


# ***
# ## reduction of span features

# In[29]:


state_encoder.mapping[0]["mapping"]


# In[30]:


data["snap"] = 0

idx_snap_ca = data.query("state_id==1 & snap_CA==1").index
data.loc[idx_snap_ca, "snap"] = 1

idx_snap_tx = data.query("state_id==2 & snap_TX==1").index
data.loc[idx_snap_tx, "snap"] = 2

idx_snap_wi = data.query("state_id==3 & snap_WI==1").index
data.loc[idx_snap_wi, "snap"] = 3


# In[31]:


data.drop(["snap_CA", "snap_TX", "snap_WI"], axis=1, inplace=True)


# ***

# In[34]:


data = reduce_mem_usage(data)
gc.collect()


# In[35]:


data.info()


# In[36]:


data.isnull().sum(axis=0)


# ***
# ## cleaning

# 
# ### removes zeros at the start of the time series


# In[38]:


def remove_starting_zeros(dataframe):
    idxmin = dataframe.query("q > 0").index.min()
    return dataframe.loc[idxmin:, :]


# In[39]:


data = (data
        .groupby(["item_id","store_id"])
        .apply(remove_starting_zeros)
        .reset_index(drop=True)
       )


# In[40]:


data.info()


# In[41]:


data.isnull().sum(axis=0)


# ***

# In[43]:


data.drop(["d", "wm_yr_wk"], axis=1, inplace=True)
data.rename({"date":"ds"}, axis=1, inplace=True)


# ***
# ## validation and evaluation dataframes

# In[47]:


calendar_columns = ["date", "wm_yr_wk", "snap_CA", "snap_TX", "snap_WI",
                    "event_name_1", "event_type_1", "event_name_2", "event_type_2"]

valid_dataframe = (pd.concat([make_time_range("2016-04-25", "2016-05-22", "D").assign(**row)
                              for _,row in hierarchy.iterrows()], ignore_index=True)
                   .merge(calendar.loc[:, calendar_columns],
                          how="left", left_on="ds", right_on="date")
                   .merge(sell_prices, how="left")
                   .drop(["id","date","wm_yr_wk"], axis=1)
                  )


# In[48]:


valid_dataframe["snap"] = 0

idx_snap_ca = valid_dataframe.query("state_id==1 & snap_CA==1").index
valid_dataframe.loc[idx_snap_ca, "snap"] = 1

idx_snap_tx = valid_dataframe.query("state_id==2 & snap_TX==1").index
valid_dataframe.loc[idx_snap_tx, "snap"] = 2

idx_snap_wi = valid_dataframe.query("state_id==3 & snap_WI==1").index
valid_dataframe.loc[idx_snap_wi, "snap"] = 3

valid_dataframe.drop(["snap_CA", "snap_TX", "snap_WI"], axis=1, inplace=True)


# In[50]:


valid_dataframe = reduce_mem_usage(valid_dataframe)


# ***

# In[52]:


calendar_columns = ["date", "wm_yr_wk", "snap_CA", "snap_TX", "snap_WI",
                    "event_name_1", "event_type_1", "event_name_2", "event_type_2"]

eval_dataframe = (pd.concat([make_time_range("2016-05-23", "2016-06-19", "D").assign(**row)
                             for _,row in hierarchy.iterrows()], ignore_index=True)
                  .merge(calendar.loc[:, calendar_columns],
                         how="left", left_on="ds", right_on="date")
                  .merge(sell_prices, how="left")
                  .drop(["id","date","wm_yr_wk"], axis=1)
                 )


# In[53]:


eval_dataframe["snap"] = 0

idx_snap_ca = eval_dataframe.query("state_id==1 & snap_CA==1").index
eval_dataframe.loc[idx_snap_ca, "snap"] = 1

idx_snap_tx = eval_dataframe.query("state_id==2 & snap_TX==1").index
eval_dataframe.loc[idx_snap_tx, "snap"] = 2

idx_snap_wi = eval_dataframe.query("state_id==3 & snap_WI==1").index
eval_dataframe.loc[idx_snap_wi, "snap"] = 3

eval_dataframe.drop(["snap_CA", "snap_TX", "snap_WI"], axis=1, inplace=True)


# In[55]:


eval_dataframe = reduce_mem_usage(eval_dataframe)


# ***
# ### Saving the dataframes

# In[59]:


# training data
(data
 .to_parquet("../input/train_dataframe.parquet", index=False)
)


# In[60]:


# validation data
(valid_dataframe
 .to_parquet("../input/valid_dataframe.parquet", index=False)
)


# In[61]:


# evaluation data
(eval_dataframe
 .to_parquet("../input/eval_dataframe.parquet", index=False)
)


# ***
