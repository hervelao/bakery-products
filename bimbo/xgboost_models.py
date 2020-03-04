import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import copy
"""
A shallow copy constructs a new compound object and then (to the extent possible) inserts references into it to the objects found in the original.

A deep copy constructs a new compound object and then, recursively, inserts copies into it of the objects found in the original.
===> deep copy copies everything but it may copy too much, such as data which is intended to be shared between copies.
    - keep a “memo” dictionary of objects
    - let user-defined classes override the copying operation or the set of components copied.
"""
from xgboost import XGBRegressor
import pickle

def build_model_1(val_X, val_y, name_model):
    """
    Build the first model
    """

    X_train, X_val, y_train, y_val = train_test_split(val_X, val_y,
        test_size=0.5, random_state=42)

    # model = XGBRegressor(
    #     max_depth=8,
    #     n_estimators=1000,
    #     min_child_weight=300,
    #     colsample_bytree=0.8,
    #     subsample=0.8,
    #     eta=0.3,
    #     seed=42)

    model = XGBRegressor(
                 base_score=0.5,
                 booster='gbtree',
                 colsample_bylevel=1,
                 colsample_bynode=1,
                 colsample_bytree=0.8,
                 eta=0.3,
                 gamma=0,
                 importance_type='gain',
                 learning_rate=0.3,
                 max_delta_step=0,
                 max_depth=7,
                 min_child_weight=300,
                 missing=None,
                 n_estimators=1000,
                 n_jobs=1,
                 nthread=None,
                 objective='reg:linear',
                 random_state=0,
                 reg_alpha=0,
                 reg_lambda=1,
                 scale_pos_weight=1,
                 seed=42,
                 silent=None,
                 subsample=0.7,
                 verbosity=1
             )

    model.fit(
        X_train,
        y_train,
        eval_metric="rmse",
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True,
        early_stopping_rounds = 1)

    # save model to file
    pickle.dump(model, open("../serialize-models/{name_model}.pickle.dat", "wb"))
    print("Saved model to: {name_model}.pickle.dat")

    y_pred = model.predict(processed_test_df)
    y_pred_exp = np.expm1(y_pred).round().astype(int)

    return y_pred, y_pred_exp

