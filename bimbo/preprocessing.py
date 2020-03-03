import numpy as np
import pandas as pd

def data_preprocess(df_):
    """
    Create a new column log_demanda_uni_equil on the val_X val_y
    log1p => Return the natural logarithm of one plus the input array
    Why log1p ? => log1p produces only positive values and removes the 'danger' of large negative numbers
    Indeed, if our dataset contains numbers much larger than zero, they can be distorted towards large negative numbers
    """
    log_venta_hoy = pd.DataFrame(np.log1p(df_['Venta_hoy']))
    log_venta_hoy.columns = ['log_venta_hoy']
    df_['log_venta_hoy'] = log_venta_hoy

    log_demanda_uni_equil = pd.DataFrame( np.log1p( df_['Demanda_uni_equil'] ) )
    log_demanda_uni_equil.columns = ['log_demanda_uni_equil']

    df_['log_demanda_uni_equil'] = log_demanda_uni_equil

    return df_
