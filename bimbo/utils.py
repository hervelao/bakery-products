import numpy as np

def rmsle_eval(y, y0):
    """
    Compute the Root Mean Squared Logarithmic Error.
    RMSLE value (score) n is the total number of observations in the
    (public/private) data set, pi is your prediction of demand, and ai
    is the actual demand for i. log(x) is the natural logarithm of x
    Submission File For every row in the dataset, submission files should
    contain two columns: id and Demanda_uni_equi. The id corresponds to the
    column of that id in the test.csv
    """
    y0=y0.get_label()
    assert len(y) == len(y0)
    return 'rmsle',np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
