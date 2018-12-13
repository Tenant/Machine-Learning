from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def read_input():
    data=pd.read_csv('krkopt.data',header=None)

    data=data.replace('a',1)
    data=data.replace('b',2)
    data=data.replace('c',3)
    data=data.replace('d',4)
    data=data.replace('e',5)
    data=data.replace('f',6)
    data=data.replace('g',7)
    data=data.replace('h',8)
    data=data.replace('draw',-1)
    data=data.replace('zero',0)
    data=data.replace('one',1)
    data=data.replace('two',2)
    data=data.replace('three',3)
    data=data.replace('four',4)
    data=data.replace('five',5)
    data=data.replace('six',6)
    data=data.replace('seven',7)
    data=data.replace('eight',8)
    data=data.replace('nine',9)
    data=data.replace('ten',10)
    data=data.replace('eleven',11)
    data=data.replace('twelve',12)
    data=data.replace('thirteen',13)
    data=data.replace('fourteen',14)
    data=data.replace('fifteen',15)
    data=data.replace('sixteen',16)

    X=np.array(data.ix[:,:5])
    y=np.array(data.ix[:,6])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    print("Data loaded")
    return x_train, x_test, y_train, y_test