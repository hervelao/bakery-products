import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle
from sklearn.externals import joblib

def build_model(X, y):
    """
    Build the first model
    """

    X_train, X_val, y_train, y_val = train_test_split(X, y,
        test_size=0.5, random_state=42)

    # We don't need Demanda_uni_equil in the features, so we can drop them now
    X_train.drop(['Demanda_uni_equil'], axis=1, inplace=True)
    X_val.drop(['Demanda_uni_equil'], axis=1, inplace=True)

    model = XGBRegressor(
                 n_estimators=1000, # Number of gradient boosted trees. Equivalent to number of boosting rounds.
                 max_depth=7, # Maximum tree depth for base learners.
                 learning_rate=0.3, #for model_1 & model_2, 0.1 Boosting learning rate (xgb’s “eta”)
                 verbosity=1, # The degree of verbosity. Valid values are 0 (silent) - 3 (debug).
                 objective='reg:linear', # Specify the learning task and the corresponding learning objective or a custom objective function to be used
                 booster='gbtree', # Specify which booster to use: gbtree, gblinear or dart.
                 n_jobs=-1, # check the nb of core (system_profiler SPHardwareDataType)
                 # because we need to set the parameter n_jobs equal to the number of cores on your machine
                 # it can be set to -1 to use all of the CPU cores on your system, which is good practice.
                 gamma=0, # Minimum loss reduction required to make a further partition on a leaf node of the tree.
                 min_child_weight=300, # Minimum sum of instance weight(hessian) needed in a child.
                 max_delta_step=0, # Maximum delta step we allow each tree’s weight estimation to be.
                 subsample=0.8, # Subsample ratio of the training instance.
                 colsample_bytree=0.8, # Subsample ratio of columns when constructing each tree.
                 colsample_bylevel=1, # Subsample ratio of columns for each level.
                 colsample_bynode=1, # Subsample ratio of columns for each split.
                 reg_alpha=0, # L1 regularization term on weights
                 reg_lambda=1, # L2 regularization term on weights
                 scale_pos_weight=1, # Balancing of positive and negative weights.
                 base_score=0.5, # The initial prediction score of all instances, global bias.
                 importance_type='gain',
                 missing=None, # Value in the data which needs to be present as a missing value.
                 random_state=0 # Random number seed.
             )

    model.fit(
        X_train,
        y_train,
        eval_metric=["mae", "rmse"], # # for model_1 & model_2, "rmse"
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True,
        early_stopping_rounds = 1) #for model_1 & model_2, only 1 ||| model_3, 3

    return model

def save_model(model, model_name):
    """
    Save the model using joblib
    """
    pickle.dump(model, open(f"../serialize-models/{model_name}.pickle.dat", "wb"))
    print(f"Saved model to: {model_name}.pickle.dat")
    joblib.dump(model, f"../serialize-models/{model_name}.joblib.dat")
    print(f"Saved model to: {model_name}.joblib.dat")

def load_model(model_name):
    """
    Load the model using joblib
    """
    # loaded_model = pickle.load(open(f"../serialize-models/{model_name}.pickle.dat", "rb"))
    # print(f"Loaded model from: {model_name}.pickle.dat")
    loaded_model = joblib.load(f"../serialize-models/{model_name}.joblib.dat")
    print(f"Loaded model from: {model_name}.joblib.dat")
    return loaded_model

def submit_model(processed_test_df, model, csv_name):
    """
    Create a dataframe usable for the submission in Kaggle
    """
    log_y_pred = model.predict(processed_test_df)
    final_predictions = np.expm1(log_y_pred).round().astype(int)
    # Create the CSV
    submit_df = pd.DataFrame( {'id':range(len(final_predictions)),
           'Demanda_uni_equil':final_predictions} )
    submit_df.to_csv(f"../data/{csv_name}.csv",index=False)
    return submit_df
