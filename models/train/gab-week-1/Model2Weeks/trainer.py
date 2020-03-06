import os
import datetime

import numpy as np
import pandas as pd

from google.cloud import storage

from sklearn.externals import joblib
from xgboost import XGBRegressor

PROJECT_ID='le-wagon-data-grupo-bimbo'            # gcp project id
BUCKET_NAME='wagon-data-grupo-bimbo-sales'        # gcp bucket name

BUCKET_DATA_PATH='data'                           # data folder
BUCKET_DATA_TRAIN_PATH='data/train.csv'      # train csv path

BUCKET_MODEL_PATH='models'                        # models folder
BUCKET_MODEL_NAME='gab_week_2'                    # model name
BUCKET_MODEL_VERSION='v_1'                        # will store model.joblib
BUCKET_MODEL_DUMP_NAME='model.joblib'             # required dump name (do not change this)

def get_data():
    '''retrieve train data from bucket'''
    df = pd.read_csv("gs://{}/{}".format(
            BUCKET_NAME,
            BUCKET_DATA_TRAIN_PATH))
    return df

def preprocess(df):
    '''process X and y from data and preprocess data'''

    '''PHASE1 - group data at the prediction level'''
    prediction_df = pd.DataFrame(df.groupby(['Cliente_ID', 'Producto_ID', 'Semana'])[['Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima',
       'Dev_proxima', 'Demanda_uni_equil']].agg(np.sum).reset_index())

    '''PHASE2 - Add feature: lagued demand, sales and returns'''
    pivot_df = df[['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID','Producto_ID','Semana']]
    lags = [1 ,2 ,3]
    lag_df = prediction_df.copy()
    col_list = ['Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima','Dev_proxima', 'Demanda_uni_equil']
    for lag in lags:
        #Create lag by changing week number in lag_df
        lag_df.Semana+=lag
        lag_df_bis = lag_df.copy()
        #Change col names
        for col in col_list:
            col_lag = col+'_lag_'+str(lag)
            lag_df_bis = lag_df_bis.rename(columns={col: col_lag})
        #Merge on prediction_df
        prediction_df = prediction_df.merge(lag_df_bis, on=['Cliente_ID', 'Producto_ID', 'Semana'], how='left')
        #Reinit lag_df week numbers for next iter
        lag_df.Semana-=lag

    '''PHASE3 - Add feature: avg product price per store'''
    price_df = pd.DataFrame(df.groupby(['Cliente_ID', 'Producto_ID']).sum()).reset_index()
    price_df = price_df[['Cliente_ID','Producto_ID', 'Venta_uni_hoy', 'Venta_hoy']]
    price_df = price_df[price_df.Venta_uni_hoy!=0]
    price_df['Product_price'] = price_df.apply(lambda x: x.Venta_hoy/x.Venta_uni_hoy, axis=1)
    price_df = price_df.drop(['Venta_uni_hoy', 'Venta_hoy'], axis=1)
    prediction_df = prediction_df.merge(price_df, on=['Cliente_ID', 'Producto_ID'], how='left')

    '''PHASE4 - Create mean encoding features and lag them'''
    groupcollist = ['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID','Producto_ID']
    aggregationlist = [\
                       #Demand sum and avg
                       ('Demanda_uni_equil',np.sum,'sum'),('Demanda_uni_equil',np.mean,'avg'),\
                       #Sales units sum and avg
                       ('Venta_uni_hoy',np.sum,'sum'),('Venta_uni_hoy',np.mean,'avg'),\
                       #Sales pesos sum and avg
                       ('Venta_hoy',np.sum,'sum'),('Venta_hoy',np.mean,'avg'),\
                       #Returns units sum and avg
                       ('Dev_uni_proxima',np.sum,'sum'),('Dev_uni_proxima',np.mean,'avg'),\
                       #Returns pesos sum and avg
                       ('Dev_proxima',np.sum,'sum'),('Dev_proxima',np.mean,'avg')]

    lags = [1 ,2 ,3]

    for type_id in groupcollist:
        for column_id,aggregator,aggtype in aggregationlist:
            # get numbers from df and set column names
            mean_df = df.groupby([type_id,'Semana']).agg(aggregator).\
                reset_index()[[column_id,type_id,'Semana']]
            new_column = type_id+'_'+aggtype+'_'+column_id
            mean_df.columns = [new_column,type_id,'Semana']
            # create corresponding lagued variables in mean_df dataframe
            lag_df = mean_df.copy()
            new_col_list = [new_column]
            for lag in lags:
                #List of lagued col names appended at each iter
                new_col_lag = new_column+'_lag_'+str(lag)
                new_col_list.append(new_col_lag)
                #Rename lagued feature
                lag_df.columns = [new_col_lag,type_id,'Semana']
                #Create lag by changing week number in lag_df
                lag_df.Semana+=lag
                #Merge on mean_df
                mean_df = mean_df.merge(lag_df, on=['Semana', type_id], how='left')
                #Reinit lag_df week numbers for next iter
                lag_df.Semana-=lag
            # merge new columns on df data
            merge_df = pd.merge(pivot_df,mean_df,on=['Semana',type_id],how='left')
            # aggregate at prediction level (product*store*week) with same aggregator
            merge_df = merge_df.groupby(['Cliente_ID','Producto_ID','Semana'])[new_col_list].agg(aggregator).reset_index()
            merge_df.columns = ['Cliente_ID','Producto_ID','Semana']+[col for col in new_col_list]
            prediction_df = pd.merge(prediction_df,merge_df,on=['Cliente_ID','Producto_ID','Semana'],\
                                     how='left')

    '''PHASE5 - Prepare data for modeling'''

    cols_to_drop_1 = ['Venta_uni_hoy','Venta_hoy','Dev_uni_proxima',
                'Dev_proxima',
                'Agencia_ID_sum_Demanda_uni_equil',
                'Agencia_ID_avg_Demanda_uni_equil',
                'Agencia_ID_sum_Venta_uni_hoy',
                'Agencia_ID_avg_Venta_uni_hoy',
                'Agencia_ID_sum_Venta_hoy',
                'Agencia_ID_avg_Venta_hoy',
                'Agencia_ID_sum_Dev_uni_proxima',
                'Agencia_ID_avg_Dev_uni_proxima',
                'Agencia_ID_avg_Dev_proxima',
                'Canal_ID_sum_Demanda_uni_equil',
                'Canal_ID_avg_Demanda_uni_equil',
                'Canal_ID_sum_Venta_uni_hoy',
                'Canal_ID_avg_Venta_uni_hoy',
                'Canal_ID_sum_Venta_hoy',
                'Canal_ID_avg_Venta_hoy',
                'Canal_ID_sum_Dev_uni_proxima',
                'Canal_ID_avg_Dev_uni_proxima',
                'Canal_ID_sum_Dev_proxima',
                'Canal_ID_avg_Dev_proxima',
                'Ruta_SAK_sum_Demanda_uni_equil',
                'Ruta_SAK_avg_Demanda_uni_equil',
                'Ruta_SAK_sum_Venta_uni_hoy',
                 'Ruta_SAK_avg_Venta_uni_hoy',
                 'Ruta_SAK_sum_Venta_hoy',
                 'Ruta_SAK_avg_Venta_hoy',
                 'Ruta_SAK_sum_Dev_uni_proxima',
                'Ruta_SAK_avg_Dev_uni_proxima',
                 'Ruta_SAK_sum_Dev_proxima',
                 'Ruta_SAK_avg_Dev_proxima',
                 'Cliente_ID_sum_Demanda_uni_equil',
                 'Cliente_ID_avg_Demanda_uni_equil',
                 'Cliente_ID_sum_Venta_uni_hoy',
                 'Cliente_ID_avg_Venta_uni_hoy',
                 'Cliente_ID_sum_Venta_hoy',
                 'Cliente_ID_avg_Venta_hoy',
                 'Cliente_ID_sum_Dev_uni_proxima',
                 'Cliente_ID_avg_Dev_uni_proxima',
                 'Cliente_ID_sum_Dev_proxima',
                 'Cliente_ID_avg_Dev_proxima',
                 'Producto_ID_sum_Demanda_uni_equil',
                 'Producto_ID_avg_Demanda_uni_equil',
                 'Producto_ID_sum_Venta_uni_hoy',
                 'Producto_ID_avg_Venta_uni_hoy',
                 'Producto_ID_sum_Venta_hoy',
                 'Producto_ID_avg_Venta_hoy',
                 'Producto_ID_sum_Dev_uni_proxima',
                 'Producto_ID_avg_Dev_uni_proxima',
                 'Producto_ID_sum_Dev_proxima',
                 'Producto_ID_avg_Dev_proxima']
    cols_to_drop_2 = [
                 'Venta_uni_hoy_lag_1',
                 'Venta_hoy_lag_1',
                 'Dev_uni_proxima_lag_1',
                 'Dev_proxima_lag_1',
                 'Demanda_uni_equil_lag_1',
                 'Agencia_ID_sum_Demanda_uni_equil_lag_1',
                 'Agencia_ID_avg_Demanda_uni_equil_lag_1',
                 'Agencia_ID_sum_Venta_uni_hoy_lag_1',
                 'Agencia_ID_avg_Venta_uni_hoy_lag_1',
                 'Agencia_ID_sum_Venta_hoy_lag_1',
                 'Agencia_ID_avg_Venta_hoy_lag_1',
                 'Agencia_ID_sum_Dev_uni_proxima_lag_1',
                 'Agencia_ID_avg_Dev_uni_proxima_lag_1',
                 'Agencia_ID_sum_Dev_proxima_lag_1',
                 'Agencia_ID_avg_Dev_proxima_lag_1',
                 'Canal_ID_sum_Demanda_uni_equil_lag_1',
                 'Canal_ID_avg_Demanda_uni_equil_lag_1',
                 'Canal_ID_sum_Venta_uni_hoy_lag_1',
                 'Canal_ID_avg_Venta_uni_hoy_lag_1',
                 'Canal_ID_sum_Venta_hoy_lag_1',
                 'Canal_ID_avg_Venta_hoy_lag_1',
                 'Canal_ID_sum_Dev_uni_proxima_lag_1',
                 'Canal_ID_avg_Dev_uni_proxima_lag_1',
                 'Canal_ID_sum_Dev_proxima_lag_1',
                 'Canal_ID_avg_Dev_proxima_lag_1',
                 'Ruta_SAK_sum_Demanda_uni_equil_lag_1',
                 'Ruta_SAK_avg_Demanda_uni_equil_lag_1',
                 'Ruta_SAK_sum_Venta_uni_hoy_lag_1',
                 'Ruta_SAK_avg_Venta_uni_hoy_lag_1',
                 'Ruta_SAK_sum_Venta_hoy_lag_1',
                 'Ruta_SAK_avg_Venta_hoy_lag_1',
                 'Ruta_SAK_sum_Dev_uni_proxima_lag_1',
                 'Ruta_SAK_avg_Dev_uni_proxima_lag_1',
                 'Ruta_SAK_sum_Dev_proxima_lag_1',
                 'Ruta_SAK_avg_Dev_proxima_lag_1',
                 'Cliente_ID_sum_Demanda_uni_equil_lag_1',
                 'Cliente_ID_avg_Demanda_uni_equil_lag_1',
                 'Cliente_ID_sum_Venta_uni_hoy_lag_1',
                 'Cliente_ID_avg_Venta_uni_hoy_lag_1',
                 'Cliente_ID_sum_Venta_hoy_lag_1',
                 'Cliente_ID_avg_Venta_hoy_lag_1',
                 'Cliente_ID_sum_Dev_uni_proxima_lag_1',
                 'Cliente_ID_avg_Dev_uni_proxima_lag_1',
                 'Cliente_ID_sum_Dev_proxima_lag_1',
                 'Cliente_ID_avg_Dev_proxima_lag_1',
                 'Producto_ID_sum_Demanda_uni_equil_lag_1',
                 'Producto_ID_avg_Demanda_uni_equil_lag_1',
                 'Producto_ID_sum_Venta_uni_hoy_lag_1',
                 'Producto_ID_avg_Venta_uni_hoy_lag_1',
                 'Producto_ID_sum_Venta_hoy_lag_1',
                 'Producto_ID_avg_Venta_hoy_lag_1',
                 'Producto_ID_sum_Dev_uni_proxima_lag_1',
                 'Producto_ID_avg_Dev_uni_proxima_lag_1',
                 'Producto_ID_sum_Dev_proxima_lag_1',
                 'Producto_ID_avg_Dev_proxima_lag_1']
    prediction_df = prediction_df.drop(cols_to_drop_1, axis=1)
    prediction_df = prediction_df.drop(cols_to_drop_2, axis=1)
    #Prepare X_train
    X_train = np.array(prediction_df.drop(['Demanda_uni_equil'], axis=1))
    #Prepare Y_train and take log
    y_train = np.log1p(prediction_df['Demanda_uni_equil'])
    return X_train, y_train

def train_model(X_train, y_train):
    '''train model'''
    model_2_week = XGBRegressor(base_score=0.5,
                     booster='gbtree',
                     colsample_bylevel=1,
                     colsample_bynode=1,
                     colsample_bytree=0.8,
                     eta=0.3,
                     gamma=0,
                     importance_type='gain',
                     learning_rate=0.3, #try smaller values later
                     max_delta_step=0,
                     max_depth=5,
                     min_child_weight=300,
                     missing=None,
                     n_estimators=1000,
                     n_jobs=1,
                     nthread=None,
                     objective='reg:linear', # for model_1 & model_2, reg:linear
                     random_state=0,
                     reg_alpha=0,
                     reg_lambda=1,
                     scale_pos_weight=1,
                     seed=42,
                     silent=None,
                     subsample=0.8,
                     verbosity=1)


    model_2_week.fit(
        X_train,
        y_train,
        eval_metric=["mae", "rmse"],
        eval_set=[(X_train, y_train)],
        verbose=True,
        early_stopping_rounds = 3)
    print('model trained')
    return model_2_week

def save_model(model):
    '''dump model and upload to gcp'''
    local_model_name = BUCKET_MODEL_DUMP_NAME
    joblib.dump(model, local_model_name)

    client = storage.Client().bucket(BUCKET_NAME)
    storage_location = '{}/{}/{}/{}'.format(
        BUCKET_MODEL_PATH,
        BUCKET_MODEL_NAME,
        BUCKET_MODEL_VERSION,
        local_model_name)
    blob = client.blob(storage_location)
    blob.upload_from_filename(local_model_name)
    print('model uploaded')

# retrieve data
df = get_data()

# get X and y
X_train, y_train = preprocess(df)

# retrieve model
model = train_model(X_train, y_train)

# dump model and upload to gcp
save_model(model)
