import pandas as pd
import category_encoders as ce


class HierarchicalEncoder(BaseEstimator):
    
    def __init__(self, cols, hierarchy, prior_level):
        self.cols = cols
        self.hierarchy = hierarchy
        self.prior_level = prior_level
    
    def fit(self, X, y):
        X["y"] = y
        X = X.merge(self.hierarchy, how="left")
        prior_level_values = X.loc[:, self.prior_level].drop_duplicates()
        encoders = dict()  
        for _,row in prior_level_values.iterrows():
            key = tuple([item for _,item in row.iteritems()])
            query_string = " & ".join([f"{col_name}=={value}" for col_name,value in row.iteritems()])
            _X = X.query(query_string).loc[:, self.cols]
            _y = X.query(query_string).loc[:, "y"].values
            encoder = ce.GLMMEncoder(cols=self.cols)
            encoder.fit(_X, _y)
            encoders[key] = encoder
        self.encoders = encoders

    def transform(self, X):
        X = X.merge(self.hierarchy, how="left")
        prior_level_values = X.loc[:, self.prior_level].drop_duplicates()
        encoded_dataframes = list()
        for _,row in prior_level_values.iterrows():
            key = tuple([item for _,item in row.iteritems()])
            query_string = " & ".join([f"{col_name}=={value}" for col_name,value in row.iteritems()])
            _X = X.query(query_string).loc[:, self.cols]
            encoded_dataframes.append(self.encoders[key].transform(_X))
        X_encoded = pd.concat(encoded_dataframes)
        return X_encoded.reindex(X.index)