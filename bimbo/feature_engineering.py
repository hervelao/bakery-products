import numpy as np
import pandas as pd
from random import *

def change_type2cate(train_x):
    """
    Change the features [Agencia_ID' , 'Canal_ID' , 'Ruta_SAK' , 'Cliente_ID' ,
    'Producto_ID'] into categories
    """
    colname = ['Agencia_ID' , 'Canal_ID' , 'Ruta_SAK' ,
           'Cliente_ID' , 'Producto_ID'  ]
    for col in colname:
        train_x[col] = train_x[col].astype('category')

    return train_x

def feature_engineering(train_x):
    """
    This function does the feature engineering.
    Mostly we aggregate the existent features to get the mean.
    Indeed, we need XGBoost to capture the time series nature of the data,
    and this is the way.
    """
    mean_due_age = train_x.groupby(['Agencia_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_age':np.mean})
    mean_due_can = train_x.groupby(['Canal_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_can':np.mean})
    mean_due_rut = train_x.groupby(['Ruta_SAK'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_rut':np.mean})
    mean_due_cli = train_x.groupby(['Cliente_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_cli':np.mean})
    mean_due_pro = train_x.groupby(['Producto_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_pro':np.mean})

    mean_vh_age = train_x.groupby(['Agencia_ID'], as_index=False)['log_venta_hoy'].agg({'mean_vh_age':np.mean})

    mean_due_pa = train_x.groupby(['Producto_ID','Agencia_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_pa':np.mean})
    mean_due_pr = train_x.groupby(['Producto_ID','Ruta_SAK'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_pr':np.mean})
    mean_due_pcli = train_x.groupby(['Producto_ID','Cliente_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_pcli':np.mean})
    mean_due_pcan = train_x.groupby(['Producto_ID','Canal_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_pcan':np.mean})

    mean_due_pca = train_x.groupby(['Producto_ID','Cliente_ID','Agencia_ID'], as_index=False)
    mean_due_pca = mean_due_pca['log_demanda_uni_equil'].agg({'mean_due_pca':np.mean})

    mean_due_acrcp = train_x.groupby(['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'], as_index=False)
    mean_due_acrcp = mean_due_acrcp['log_demanda_uni_equil'].agg({'mean_due_acrcp':np.mean})
    sd_due_acrcp = train_x.groupby(['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'], as_index=False)
    sd_due_acrcp = sd_due_acrcp['log_demanda_uni_equil'].agg({'sd_due_acrcp':np.std})

    temp = [ mean_due_age,
            mean_due_can,
            mean_due_rut ,
            mean_due_cli ,
            mean_due_pro ,
            mean_due_pa,
            mean_due_pr,
            mean_due_pcli,
            mean_due_pcan,
            mean_due_pca,
            mean_vh_age,
            mean_due_acrcp,
            sd_due_acrcp]

    return temp

def merge_feature(y, temp, val_or_test):
    """
    Merging the different features created with the target
    """
    mean_due_age    = temp[0]
    mean_due_can    = temp[1]
    mean_due_rut    = temp[2]
    mean_due_cli    = temp[3]
    mean_due_pro    = temp[4]
    mean_due_pa     = temp[5]
    mean_due_pr     = temp[6]
    mean_due_pcli   = temp[7]
    mean_due_pcan   = temp[8]
    mean_due_pca    = temp[9]
    mean_vh_age     = temp[10]
    mean_due_acrcp  = temp[11]
    sd_due_acrcp    = temp[12]

    if val_or_test == 'val':
        seed(100)
        merged_df = y
        colname = ['Agencia_ID' , 'Canal_ID' , 'Ruta_SAK' ,
                   'Cliente_ID' , 'Producto_ID'  ,
                   'Demanda_uni_equil', 'log_demanda_uni_equil']
        merged_df = merged_df[colname]

    elif val_or_test == 'test':
        merged_df = y
        colname = ['Agencia_ID' , 'Canal_ID' , 'Ruta_SAK' ,
                   'Cliente_ID' , 'Producto_ID'  ]

    merged_df = pd.merge(merged_df, mean_due_age,how = 'left', on='Agencia_ID')

    merged_df = pd.merge(merged_df, mean_due_can,how = 'left', on='Canal_ID')
    merged_df = pd.merge(merged_df, mean_due_rut,how = 'left', on='Ruta_SAK')
    merged_df = pd.merge(merged_df, mean_due_cli,how = 'left', on='Cliente_ID')


    merged_df = pd.merge(merged_df , mean_due_pa, how = 'left', on = ["Producto_ID", "Agencia_ID"])
    merged_df = pd.merge(merged_df , mean_due_pr, how = 'left', on = ["Producto_ID", "Ruta_SAK"])
    merged_df = pd.merge(merged_df , mean_due_pcli, how = 'left', on = ["Producto_ID", "Cliente_ID"])
    merged_df = pd.merge(merged_df , mean_due_pcan, how = 'left', on = ["Producto_ID", "Canal_ID"])

    merged_df = pd.merge(merged_df , mean_due_pca, how = 'left',
                    on = ["Producto_ID", "Cliente_ID", "Agencia_ID"])

    merged_df = pd.merge(merged_df , mean_vh_age, how = 'left', on = ["Agencia_ID"])
    merged_df = pd.merge(merged_df , sd_due_acrcp, how = 'left',
                    on = ["Producto_ID", "Cliente_ID", "Agencia_ID",    "Canal_ID" , "Ruta_SAK"])

    merged_df = pd.merge(merged_df , mean_due_acrcp, how = 'left',
                    on = ["Producto_ID", "Cliente_ID", "Agencia_ID",    "Canal_ID" , "Ruta_SAK"])

    colname = ['Agencia_ID' , 'Canal_ID' , 'Ruta_SAK' , 'Cliente_ID' , 'Producto_ID']

    merged_df.drop(colname, axis=1, inplace=True)

    return merged_df
