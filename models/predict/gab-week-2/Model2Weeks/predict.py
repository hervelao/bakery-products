import numpy as np
import pandas as pd

from google.cloud import storage
import googleapiclient.discovery

PROJECT_ID='le-wagon-data-grupo-bimbo'            # gcp project id
BUCKET_NAME='wagon-data-grupo-bimbo-sales'        # gcp bucket name

BUCKET_DATA_PATH='data'                           # data folder
BUCKET_DATA_TEST_PATH='data/test_10k.csv'         # test csv path
BUCKET_DATA_TRAIN_PATH='data/train.csv'      # train csv path

BUCKET_MODEL_NAME='static_baseline_fixed_response_4'        # model name
BUCKET_MODEL_VERSION='v_1'                        # will store model.joblib

def get_test_data():
    '''retrieve test data from bucket'''
    client = storage.Client()
    df = pd.read_csv("gs://{}/{}".format(
            BUCKET_NAME,
            BUCKET_DATA_TEST_PATH),
        nrows=1000)
    return df

def get_data():
    '''retrieve train data from bucket'''
    df = pd.read_csv("gs://{}/{}".format(
            BUCKET_NAME,
            BUCKET_DATA_TRAIN_PATH))
    return df

def preprocess(df_test, df_train):
    '''process X and y from data and preprocess data'''
    '''preprocess should be identical to the one used on train data'''
    '''actually preprocess is not identical'''
    df = pd.concat([df_train, df_test]).drop('id', axis=1)

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
    prediction_df = prediction_df[prediction_df.Semana==11]
    X_test = np.array(prediction_df.drop(['Demanda_uni_equil'], axis=1))
    #Prepare Y_train and take log
    y_test = None
    return X_test, y_test

def convert_to_json_instances(X_test):
    return X_test.values.tolist()

def predict_json(project, model, instances, version=None):
    '''call model for prediction'''
    service = googleapiclient.discovery.build('ml', 'v1') # google api endpoint /ml/v1
    name = 'projects/{}/models/{}'.format(project, model)
    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()
    if 'error' in response:
        raise RuntimeError(response['error'])
    return response['predictions']

# get data
df_test = get_test_data().head(100) # only predict for first 100 rows
df_train = get_data()
# apply preprocess
X_test, y_test = preprocess(df_test, df_train)

# # convert X_test to json
# instances = convert_to_json_instances(X_test)

# # send request and get response
# results = predict_json(project=PROJECT_ID,
#     model=BUCKET_MODEL_NAME,
#     version=BUCKET_MODEL_VERSION,
#     instances=instances)

# print(results)
