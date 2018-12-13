from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def read_input():
    data_train=pd.read_csv('dota2Train.csv',header=None)
    data_test=pd.read_csv('dota2Test.csv',header=None)

    x_train=data_train.ix[:,1:]
    y_train=data_train.ix[:,0]
    x_test=data_test.ix[:,1:]
    y_test=data_test.ix[:,0]
    print("Data loaded")
    return x_train, x_test, y_train, y_test
