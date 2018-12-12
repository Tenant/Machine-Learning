# -*- coding: utf-8 -*-
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from read_file import *


class Iterative_Bagging():
    def __init__(self, X, Y, estimator=DecisionTreeRegressor()):
        self.X = X
        self.Y = Y
        self.estimator = estimator
        self.estimators = []
        self.min_mean_square = 1e20
        self.train()


    def train(self):
        X = self.X
        Y = self.Y
        min_mean_square = self.min_mean_square
        while True:
            clf = BaggingRegressor(base_estimator=self.estimator, n_estimators=500)
            clf.fit(X, Y)
            Y_hat = clf.predict(X)
            Y = Y - Y_hat
            self.estimators.append(clf)
            if 1.1*np.sum(Y**2) > min_mean_square:
                break
            min_mean_square = min(min_mean_square, np.sum(Y**2))
            print("Epoch " + str(len(self.estimators)))
        print("Traning finish")


    def predict(self, X):
        print("=" * 40)
        print("Predict start")
        Y_hat = np.zeros(len(X))
        for estimator in self.estimators:
            Y_hat = Y_hat + estimator.predict(X)
        return Y_hat


    def check(self,Y_hat,Y):
        sstot = np.sum((Y - Y.mean()) ** 2)
        ssreg = np.sum((Y - Y_hat) ** 2)
        print("The accuracy is " + str(1 - ssreg / sstot))


x_train, x_test, y_train, y_test = read_input("airfoil_self_noise.dat")
clf = Iterative_Bagging(x_train, y_train)
y_hat = clf.predict(x_test)
clf.check(y_hat,y_test)

