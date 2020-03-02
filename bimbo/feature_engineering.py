from datetime import datetime


def feature_engineering(train_x):
    """
    This function does the feature engineering
    """
    #s = datetime.datetime.now()

    #mean_due_age =
    #mean.due.age = main.train.x[, .(mean.due.age = mean(log.due)), by = .(Agencia_ID)]
    #s = datetime.now()
    mean_due_age = train_x.groupby(['Agencia_ID'], as_index=False)['log_due'].agg({'mean_due_age':np.mean})
    mean_due_can = train_x.groupby(['Canal_ID'], as_index=False)['log_due'].agg({'mean_due_can':np.mean})
    mean_due_rut = train_x.groupby(['Ruta_SAK'], as_index=False)['log_due'].agg({'mean_due_rut':np.mean})
    mean_due_cli = train_x.groupby(['Cliente_ID'], as_index=False)['log_due'].agg({'mean_due_cli':np.mean})
    mean_due_pro = train_x.groupby(['Producto_ID'], as_index=False)['log_due'].agg({'mean_due_pro':np.mean})

    mean_vh_age = train_x.groupby(['Agencia_ID'], as_index=False)['log_vh'].agg({'mean_vh_age':np.mean})



    mean_due_pa = train_x.groupby(['Producto_ID','Agencia_ID'], as_index=False)['log_due'].agg({'mean_due_pa':np.mean})
    mean_due_pr = train_x.groupby(['Producto_ID','Ruta_SAK'], as_index=False)['log_due'].agg({'mean_due_pr':np.mean})
    mean_due_pcli = train_x.groupby(['Producto_ID','Cliente_ID'], as_index=False)['log_due'].agg({'mean_due_pcli':np.mean})
    mean_due_pcan = train_x.groupby(['Producto_ID','Canal_ID'], as_index=False)['log_due'].agg({'mean_due_pcan':np.mean})

    mean_due_pca = train_x.groupby(['Producto_ID','Cliente_ID','Agencia_ID'], as_index=False)
    mean_due_pca = mean_due_pca['log_due'].agg({'mean_due_pca':np.mean})



    #s = datetime.now()
    mean_due_acrcp = train_x.groupby(['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'], as_index=False)
    mean_due_acrcp = mean_due_acrcp['log_due'].agg({'mean_due_acrcp':np.mean})
    sd_due_acrcp = train_x.groupby(['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'], as_index=False)
    sd_due_acrcp = sd_due_acrcp['log_due'].agg({'sd_due_acrcp':np.std})
    #t = datetime.now() - s
    #print(t)
    # 0:00:04.097429

    tem = [ mean_due_age,
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

    return tem
