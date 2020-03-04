import numpy as np
import pandas as pd

from google.cloud import storage
import googleapiclient.discovery

PROJECT_ID='le-wagon-data-grupo-bimbo'            # gcp project id
BUCKET_NAME='wagon-data-grupo-bimbo-sales'        # gcp bucket name

BUCKET_DATA_PATH='data'                           # data folder
BUCKET_DATA_TEST_PATH='data/test_10k.csv'         # test csv path

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

def preprocess(df):
    '''process X and y from data and preprocess data'''
    '''preprocess should be identical to the one used on train data'''
    X_test = df[['Agencia_ID']]
    y_test = None
    # y_test = df['Demanda_uni_equil']
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

df = get_test_data().head(100) # only predict for first 100 rows
X_test, y_test = preprocess(df)
instances = convert_to_json_instances(X_test)
results = predict_json(project=PROJECT_ID,
    model=BUCKET_MODEL_NAME,
    version=BUCKET_MODEL_VERSION,
    instances=instances)

print(df)

#     id  Semana  Agencia_ID  Canal_ID  Ruta_SAK  Cliente_ID  Producto_ID
# 0    0      11        4037         1      2209     4639078        35305
# 1    1      11        2237         1      1226     4705135         1238
# 2    2      10        2045         1      2831     4549769        32940
# 3    3      11        1227         1      4448     4717855        43066

# old code sample
#
# df["fare_amount"] = results
# df[["key", "fare_amount"]].to_csv("predictions.csv", index=False)
