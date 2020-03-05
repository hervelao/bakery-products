import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle

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
                 base_score=0.5,
                 booster='gbtree',
                 colsample_bylevel=1,
                 colsample_bynode=1,
                 colsample_bytree=0.8,
                 eta=0.3,
                 gamma=0,
                 importance_type='gain',
                 learning_rate=0.1, #for model_2, 0.1
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
        eval_metric=["rmse", "rmsle", "mae", "logloss"], # only "rmse" previously
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True,
        early_stopping_rounds = 2) #for model_2, only 1

    return model

def save_model(model, model_name):
    """
    Save the model using pickle
    """
    pickle.dump(model, open(f"../serialize-models/{model_name}.pickle.dat", "wb"))
    print(f"Saved model to: {model_name}.pickle.dat")

def load_model(model_name):
    """
    Load the model using pickle
    """
    loaded_model = pickle.load(open(f"../serialize-models/{model_name}.pickle.dat", "rb"))
    print("Loaded model from: pima.pickle.dat")
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
