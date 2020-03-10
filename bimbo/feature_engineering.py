import numpy as np
import pandas as pd

def change_type_to_categ(df_):
    """
    Change the features [Agencia_ID' ,'Canal_ID' ,'Ruta_SAK' ,'Cliente_ID' ,
    'Producto_ID'] into categories
    """
    colname = ['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID' ,'Producto_ID']
    for col in colname:
        df_[col] = df_[col].astype('category')

    return df_

def feature_engineering(df_):
    """
    This function does the feature engineering.

    """
    mean_due_agencia = df_.groupby(['Agencia_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_agencia':np.mean})
    mean_due_canal = df_.groupby(['Canal_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_canal':np.mean})
    mean_due_ruta = df_.groupby(['Ruta_SAK'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_ruta':np.mean})
    mean_due_cliente = df_.groupby(['Cliente_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_cliente':np.mean})
    mean_due_producto = df_.groupby(['Producto_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_producto':np.mean})

    mean_vh_agencia = df_.groupby(['Agencia_ID'], as_index=False)['log_venta_hoy'].agg({'mean_vh_agencia':np.mean})

    mean_due_prod_age = df_.groupby(['Producto_ID','Agencia_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_prod_age':np.mean})
    mean_due_prod_rut = df_.groupby(['Producto_ID','Ruta_SAK'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_prod_rut':np.mean})
    mean_due_prod_cli = df_.groupby(['Producto_ID','Cliente_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_prod_cli':np.mean})
    mean_due_prod_can = df_.groupby(['Producto_ID','Canal_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_prod_can':np.mean})

    mean_due_prod_cli_age = df_.groupby(['Producto_ID','Cliente_ID','Agencia_ID'], as_index=False)
    mean_due_prod_cli_age = mean_due_prod_cli_age['log_demanda_uni_equil'].agg({'mean_due_prod_cli_age':np.mean})

    mean_due_acrcp = df_.groupby(['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'], as_index=False)
    mean_due_acrcp = mean_due_acrcp['log_demanda_uni_equil'].agg({'mean_due_acrcp':np.mean})

    std_due_acrcp = df_.groupby(['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'], as_index=False)
    std_due_acrcp = std_due_acrcp['log_demanda_uni_equil'].agg({'std_due_acrcp':np.std})

    """
    ADD AS MANY FEATURES AS WANTED
    """

    # List of 13 new features, but more can be created
    temp = [mean_due_agencia,
            mean_due_canal,
            mean_due_ruta,
            mean_due_cliente,
            mean_due_producto,
            mean_vh_agencia,
            mean_due_prod_age,
            mean_due_prod_rut,
            mean_due_prod_cli,
            mean_due_prod_can,
            mean_due_prod_cli_age,
            mean_due_acrcp,
            std_due_acrcp]

    return temp

def merge_feature(df_, temp, val_or_test):
    """
    Merging the different features created with the target
    """
    mean_due_agencia = temp[0]
    mean_due_canal = temp[1]
    mean_due_ruta = temp[2]
    mean_due_cliente = temp[3]
    mean_due_producto = temp[4]
    mean_vh_agencia = temp[5]
    mean_due_prod_age = temp[6]
    mean_due_prod_rut = temp[7]
    mean_due_prod_cli = temp[8]
    mean_due_prod_can = temp[9]
    mean_due_prod_cli_age = temp[10]
    mean_due_acrcp = temp[11]
    std_due_acrcp = temp[12]

    # it should give an error if we don't have the following columns
    if val_or_test == 'val':
        merged_df = df_
        colname = ['Agencia_ID', 'Canal_ID' , 'Ruta_SAK',
                   'Cliente_ID', 'Producto_ID',
                   'Demanda_uni_equil', 'log_demanda_uni_equil']
        merged_df = merged_df[colname]

    # it should give an error if we don't have the following columns
    if val_or_test == 'test':
        merged_df = df_
        colname = ['Agencia_ID' , 'Canal_ID' , 'Ruta_SAK' ,
                   'Cliente_ID' , 'Producto_ID']
        merged_df = merged_df[colname]

    # Merge all the newly created features
    merged_df = pd.merge(merged_df, mean_due_agencia, how = 'left', on='Agencia_ID')
    merged_df = pd.merge(merged_df, mean_due_canal, how = 'left', on='Canal_ID')
    merged_df = pd.merge(merged_df, mean_due_ruta, how = 'left', on='Ruta_SAK')
    merged_df = pd.merge(merged_df, mean_due_cliente, how = 'left', on='Cliente_ID')
    merged_df = pd.merge(merged_df , mean_due_prod_age, how = 'left', on = ["Producto_ID", "Agencia_ID"])
    merged_df = pd.merge(merged_df , mean_due_prod_rut, how = 'left', on = ["Producto_ID", "Ruta_SAK"])
    merged_df = pd.merge(merged_df , mean_due_prod_cli, how = 'left', on = ["Producto_ID", "Cliente_ID"])
    merged_df = pd.merge(merged_df , mean_due_prod_can, how = 'left', on = ["Producto_ID", "Canal_ID"])
    merged_df = pd.merge(merged_df , mean_due_prod_cli_age, how = 'left',
                    on = ["Producto_ID", "Cliente_ID", "Agencia_ID"])
    merged_df = pd.merge(merged_df , mean_vh_agencia, how = 'left', on = ["Agencia_ID"])
    merged_df = pd.merge(merged_df , std_due_acrcp, how = 'left',
                    on = ["Producto_ID", "Cliente_ID", "Agencia_ID",    "Canal_ID" , "Ruta_SAK"])
    merged_df = pd.merge(merged_df , mean_due_acrcp, how = 'left',
                    on = ["Producto_ID", "Cliente_ID", "Agencia_ID",    "Canal_ID" , "Ruta_SAK"])

    # Now we can drop the categories. We no longer need them
    colname = ['Agencia_ID' , 'Canal_ID' , 'Ruta_SAK' , 'Cliente_ID' , 'Producto_ID']
    merged_df.drop(colname, axis=1, inplace=True)

    return merged_df


# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------
def change_type_to_categ2(df_):
    """
    Change the features [Agencia_ID' ,'Canal_ID' ,'Ruta_SAK' ,'Cliente_ID' ,
    'Producto_ID', 'short_name', 'brand', 'Town', 'State'] into categories
    """
    colname = ['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID' ,'Producto_ID', 'short_name', 'brand', 'Town', 'State']
    for col in colname:
        df_[col] = df_[col].astype('category')

    return df_


def feature_engineering2(df_):
    """
    This function does the feature engineering.

    """
    mean_due_agencia = df_.groupby(['Agencia_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_agencia':np.mean})
    mean_due_canal = df_.groupby(['Canal_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_canal':np.mean})
    mean_due_ruta = df_.groupby(['Ruta_SAK'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_ruta':np.mean})
    mean_due_cliente = df_.groupby(['Cliente_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_cliente':np.mean})
    mean_due_producto = df_.groupby(['Producto_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_producto':np.mean})

    mean_vh_agencia = df_.groupby(['Agencia_ID'], as_index=False)['log_venta_hoy'].agg({'mean_vh_agencia':np.mean})

    mean_due_prod_age = df_.groupby(['Producto_ID','Agencia_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_prod_age':np.mean})
    mean_due_prod_rut = df_.groupby(['Producto_ID','Ruta_SAK'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_prod_rut':np.mean})
    mean_due_prod_cli = df_.groupby(['Producto_ID','Cliente_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_prod_cli':np.mean})
    mean_due_prod_can = df_.groupby(['Producto_ID','Canal_ID'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_prod_can':np.mean})

    mean_due_prod_cli_age = df_.groupby(['Producto_ID','Cliente_ID','Agencia_ID'], as_index=False)
    mean_due_prod_cli_age = mean_due_prod_cli_age['log_demanda_uni_equil'].agg({'mean_due_prod_cli_age':np.mean})

    mean_due_acrcp = df_.groupby(['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'], as_index=False)
    mean_due_acrcp = mean_due_acrcp['log_demanda_uni_equil'].agg({'mean_due_acrcp':np.mean})

    std_due_acrcp = df_.groupby(['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'], as_index=False)
    std_due_acrcp = std_due_acrcp['log_demanda_uni_equil'].agg({'std_due_acrcp':np.std})

    """
    NEW FEATURES
    """
    # Add mean target value per client and product cluster
    mean_due_cli_clu = df_.groupby(['Cliente_ID','short_name'], as_index=False)['log_demanda_uni_equil'].agg({'mean_due_cli_clu':np.mean})

    # # Add max target value per client and product
    max_due_prod_cli = df_.groupby(['Producto_ID','Cliente_ID'], as_index=False)['log_demanda_uni_equil'].agg({'max_due_prod_cli':np.max})

    # List of 15 new features, but more can be created
    temp = [mean_due_agencia,
            mean_due_canal,
            mean_due_ruta,
            mean_due_cliente,
            mean_due_producto,
            mean_vh_agencia,
            mean_due_prod_age,
            mean_due_prod_rut,
            mean_due_prod_cli,
            mean_due_prod_can,
            mean_due_prod_cli_age,
            mean_due_acrcp,
            std_due_acrcp,
            mean_due_cli_clu,
            max_due_prod_cli
            ]

    return temp

def merge_feature2(df_, temp, val_or_test):
    """
    Merging the different features created with the target
    """
    mean_due_agencia = temp[0]
    mean_due_canal = temp[1]
    mean_due_ruta = temp[2]
    mean_due_cliente = temp[3]
    mean_due_producto = temp[4]
    mean_vh_agencia = temp[5]
    mean_due_prod_age = temp[6]
    mean_due_prod_rut = temp[7]
    mean_due_prod_cli = temp[8]
    mean_due_prod_can = temp[9]
    mean_due_prod_cli_age = temp[10]
    mean_due_acrcp = temp[11]
    std_due_acrcp = temp[12]
    mean_due_cli_clu = temp[13]
    max_due_prod_cli = temp[14]

    # it should give an error if we don't have the following columns
    if val_or_test == 'val':
        merged_df = df_
        colname = ['Agencia_ID', 'Canal_ID' , 'Ruta_SAK',
                   'Cliente_ID', 'Producto_ID',
                   'short_name', 'brand', 'Town', 'State',
                   'Demanda_uni_equil', 'log_demanda_uni_equil']
        merged_df = merged_df[colname]

    # it should give an error if we don't have the following columns
    if val_or_test == 'test':
        merged_df = df_
        colname = ['Agencia_ID' , 'Canal_ID' , 'Ruta_SAK' ,
                   'Cliente_ID' , 'Producto_ID',
                   'short_name', 'brand', 'Town', 'State']
        merged_df = merged_df[colname]

    # Merge all the newly created features
    merged_df = pd.merge(merged_df, mean_due_agencia, how = 'left', on='Agencia_ID')
    merged_df = pd.merge(merged_df, mean_due_canal, how = 'left', on='Canal_ID')
    merged_df = pd.merge(merged_df, mean_due_ruta, how = 'left', on='Ruta_SAK')
    merged_df = pd.merge(merged_df, mean_due_cliente, how = 'left', on='Cliente_ID')
    merged_df = pd.merge(merged_df , mean_due_prod_age, how = 'left', on = ["Producto_ID", "Agencia_ID"])
    merged_df = pd.merge(merged_df , mean_due_prod_rut, how = 'left', on = ["Producto_ID", "Ruta_SAK"])
    merged_df = pd.merge(merged_df , mean_due_prod_cli, how = 'left', on = ["Producto_ID", "Cliente_ID"])
    merged_df = pd.merge(merged_df , mean_due_prod_can, how = 'left', on = ["Producto_ID", "Canal_ID"])
    merged_df = pd.merge(merged_df , mean_due_prod_cli_age, how = 'left',
                    on = ["Producto_ID", "Cliente_ID", "Agencia_ID"])
    merged_df = pd.merge(merged_df , mean_vh_agencia, how = 'left', on = ["Agencia_ID"])
    merged_df = pd.merge(merged_df , std_due_acrcp, how = 'left',
                    on = ["Producto_ID", "Cliente_ID", "Agencia_ID",    "Canal_ID" , "Ruta_SAK"])
    merged_df = pd.merge(merged_df , mean_due_acrcp, how = 'left',
                    on = ["Producto_ID", "Cliente_ID", "Agencia_ID",    "Canal_ID" , "Ruta_SAK"])
    """
    NEW FEATURES
    """

    merged_df = pd.merge(merged_df , mean_due_cli_clu, how = 'left',
                    on = ["Cliente_ID", "short_name"])
    merged_df = pd.merge(merged_df , max_due_prod_cli, how = 'left',
                    on = ["Producto_ID", "Cliente_ID"])

    # Now we can drop the categories. We no longer need them
    colname = ['Agencia_ID' , 'Canal_ID' , 'Ruta_SAK' , 'Cliente_ID' , 'Producto_ID',
                'short_name', 'brand', 'Town', 'State']
    merged_df.drop(colname, axis=1, inplace=True)

    return merged_df
