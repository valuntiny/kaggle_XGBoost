'''
This file is used to run the XBGoost regressor because jupyter notebook cant run it through
'''

import pandas as pd
from sklearn.model_selection import train_test_split

# read data
X = pd.read_csv("train.csv", index_col='Id')
X_test_full = pd.read_csv("test.csv", index_col='Id')

# split as X and y
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(axis=0, columns=['SalePrice'], inplace=True)
X_train_full, X_val_full, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                      random_state=0)

# only choose the numercial column and low cardinality categorical column
low_cardinality_col = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10
                      and X_train_full[cname].dtype == 'object']
numerical_col = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
my_col = low_cardinality_col + numerical_col

# make a harder copy
X_train = X_train_full[my_col].copy()
X_val = X_val_full[my_col].copy()
X_test = X_test_full[my_col].copy()

# one-hot encode, and also use align() to make sure only the dummies in train exist
X_train = pd.get_dummies(X_train)
X_val = pd.get_dummies(X_val)
X_test = pd.get_dummies(X_test)
X_train, X_val = X_train.align(X_val, axis=1, join='left')
X_train, X_test = X_train.align(X_test, axis=1, join='left')

# start XBGoost
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

my_model_1 = XGBRegressor(random_state=0)
my_model_1.fit(X_train, y_train)
prediction_1 = my_model_1.predict(X_val)
mae_1 = mean_absolute_error(prediction_1, y_val)
print("MAE for model 1:", mae_1)

# improve the model
my_model_2 = XGBRegressor(random_state=0, n_estimators=1000, learning_rate=0.1)
my_model_2.fit(X_train, y_train)
prediction_2 = my_model_2.predict(X_val)
mae_2 = mean_absolute_error(prediction_2, y_val)
print("MAE for model 2:", mae_2)

# give a bad model
my_model_3 = XGBRegressor(random_state=0, n_estimators=1)
my_model_3.fit(X_train, y_train)
prediction_3 = my_model_3.predict(X_val)
mae_3 = mean_absolute_error(prediction_3, y_val)
print("MAE for model 3:", mae_3)