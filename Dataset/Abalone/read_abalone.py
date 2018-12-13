from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def read_input():
    data=pd.read_csv("abalone.data",header=None)
    data=data.replace('M',0)
    data=data.replace('F',1)
    data=data.replace('I',2)
    X=np.array(data.ix[:,:7])
    y=np.array(data.ix[:,8])    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    print("Data loaded")
    return x_train, x_test, y_train, y_test