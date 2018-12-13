from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def read_input():
    data_train=pd.read_csv('seismic-bumps.data',header=None)

    X=data_train.ix[:,:17]
    y=data_train.ix[:,18]

    X=X.replace('a',0);
    X=X.replace('b',1);
    X=X.replace('c',2);
    X=X.replace('d',3);
    X=X.replace('N',0)  
    X=X.replace('W',1)  
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    print("Data loaded")
    return x_train, x_test, y_train, y_test
