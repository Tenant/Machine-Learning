from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def read_input():
    data=pd.read_csv('CNAE-9.data',header=None)

    X=np.array(data.ix[:,1:])
    y=np.array(data.ix[:,0])  
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    print("Data loaded")
    return x_train, x_test, y_train, y_test