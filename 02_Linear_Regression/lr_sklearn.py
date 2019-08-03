import numpy as np
import datagen as dg

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler

# hyperparams
alpha     = 0.00001
tolerance = 0.000001
max_iter  = 1000000

# get training data
add_ones = True
var = 0.5
x_all, y_all = dg.set_train_data(100, add_ones=add_ones, var=var)
if len(x_all.shape) == 1:
    x_all = x_all[:, np.newaxis]

x_train, x_test, y_train, y_test = train_test_split(x_all.T, y_all.T, random_state=0)

from sklearn.linear_model import LinearRegression

lr = LinearRegression(copy_X = True, fit_intercept = not add_ones)
lr.fit(x_train, y_train)

score = lr.score(x_test, y_test)

w = lr.coef_
if not add_ones:
    w = np.append(w, lr.intercept_)
w = w[:, np.newaxis]


y_err = np.square(y_test[:,  np.newaxis] - x_test.dot(w))
y_var = np.square(y_test[:,  np.newaxis] - np.mean(y_test))
r2 = 1. - np.sum(y_err)/np.sum(y_var)

dg.show_data(x_test, y_test, w)

pass


