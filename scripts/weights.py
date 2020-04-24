#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from category_encoders.ordinal import OrdinalEncoder


# In[2]:


sales_train = pd.read_csv("../input/sales_train_validation.csv")
weights_validation = pd.read_csv("../input/weights_validation.csv")


# ***

# In[3]:


hierarchy = (sales_train.loc[:, ["item_id", "dept_id", "cat_id", "store_id", "state_id"]]
             .drop_duplicates())


# In[4]:


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
# ### weights for level 12

# In[5]:


weights_level12 = (weights_validation.query("Level_id == 'Level12'")
                   .rename({"Agg_Level_1":"item_id", "Agg_Level_2":"store_id", "Weight":"weight"}, axis=1)
                   .drop("Level_id", axis=1)
                  )


# In[6]:


weights_level12


# In[7]:


weights_level12["item_id"] = item_encoder.transform(weights_level12.loc[:, ["item_id"]])
weights_level12["store_id"] = store_encoder.transform(weights_level12.loc[:, ["store_id"]])


# In[8]:


weights_level12.to_parquet("../input/weights_level12.parquet", index=False)


# ***
# ### weights for level 9

# In[9]:


weights_validation.query("Level_id == 'Level9'").head()


# In[10]:


weights_level9 = (weights_validation.query("Level_id == 'Level9'")
                  .rename({"Agg_Level_1":"store_id", "Agg_Level_2":"dept_id", "Weight":"weight"}, axis=1)
                  .drop("Level_id", axis=1)
                 )


# In[11]:


weights_level9.head()


# In[12]:


weights_level9["store_id"] = store_encoder.transform(weights_level9.loc[:, ["store_id"]])
weights_level9["dept_id"] = dept_encoder.transform(weights_level9.loc[:, ["dept_id"]])


# In[13]:


weights_level9.to_parquet("../input/weights_level9.parquet", index=False)


# ***
# ### weights for level 8: store_id,cat_id

# In[18]:


weights_validation.query("Level_id == 'Level8'").head()


# In[19]:


weights_level8 = (weights_validation.query("Level_id == 'Level8'")
                  .rename({"Agg_Level_1":"store_id", "Agg_Level_2":"cat_id", "Weight":"weight"}, axis=1)
                  .drop("Level_id", axis=1)
                 )
weights_level8.head()


# In[20]:


weights_level8["store_id"] = store_encoder.transform(weights_level8.loc[:, ["store_id"]])
weights_level8["cat_id"] = cat_encoder.transform(weights_level8.loc[:, ["cat_id"]])
weights_level8.head()


# In[21]:


weights_level8.to_parquet("../input/weights_level8.parquet", index=False)


# ***
# ### weights for level 5: dept_id

# In[35]:


weights_validation.query("Level_id == 'Level5'").head(10)


# In[36]:


weights_level5 = (weights_validation.query("Level_id == 'Level5'")
                  .rename({"Agg_Level_1":"dept_id", "Weight":"weight"}, axis=1)
                  .drop(["Level_id","Agg_Level_2"], axis=1)
                 )
weights_level5.head(10)


# In[37]:


weights_level5["dept_id"] = dept_encoder.transform(weights_level5.loc[:, ["dept_id"]])
weights_level5.head()


# In[38]:


weights_level5.to_parquet("../input/weights_level5.parquet", index=False)


# ***
# ### weights for level 4: cat_id

# In[22]:


weights_validation.query("Level_id == 'Level4'").head()


# In[23]:


weights_level4 = (weights_validation.query("Level_id == 'Level4'")
                  .rename({"Agg_Level_1":"cat_id", "Weight":"weight"}, axis=1)
                  .drop(["Level_id","Agg_Level_2"], axis=1)
                 )
weights_level4.head()


# In[24]:


weights_level4["cat_id"] = cat_encoder.transform(weights_level4.loc[:, ["cat_id"]])
weights_level4.head()


# In[27]:


weights_level4.to_parquet("../input/weights_level4.parquet", index=False)


# ***
# ### weights for level 3: store_id

# In[19]:


weights_validation.query("Level_id == 'Level3'").head()


# In[20]:


weights_level3 = (weights_validation.query("Level_id == 'Level3'")
                  .rename({"Agg_Level_1":"store_id", "Weight":"weight"}, axis=1)
                  .drop(["Level_id","Agg_Level_2"], axis=1)
                 )
weights_level3.head()


# In[21]:


weights_level3["store_id"] = store_encoder.transform(weights_level3.loc[:, ["store_id"]])
weights_level3.head()


# In[22]:


weights_level3.to_parquet("../input/weights_level3.parquet", index=False)


# ***
# ### weights for level 2: state_id

# In[15]:


weights_validation.query("Level_id == 'Level2'").head()


# In[18]:


weights_level2 = (weights_validation.query("Level_id == 'Level2'")
                  .rename({"Agg_Level_1":"state_id", "Weight":"weight"}, axis=1)
                  .drop(["Level_id","Agg_Level_2"], axis=1)
                 )
weights_level2.head()


# In[19]:


weights_level2["state_id"] = state_encoder.transform(weights_level2.loc[:, ["state_id"]])
weights_level2.head()


# In[20]:


weights_level2.to_parquet("../input/weights_level2.parquet", index=False)


# ***
