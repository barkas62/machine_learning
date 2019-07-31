import numpy as np
import datagen as dg

def get_grad_and_error(w, x, y):
    h = w.dot(x)
    err = y - h
    norm = 1.0 #1.0/x.shape[1]
    grad = - norm * err.dot(x.T)
    err2 = (1.0/x.shape[1])*np.sum(err*err)
    return grad, err2

def get_err(w,x,y):
    h = w.dot(x)
    err = y - h
    err2 = (1.0 / x.shape[1]) * np.sum(err * err)
    return err2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler

# hyperparams
alpha     = 0.00001
tolerance = 0.000001
max_iter  = 1000000

# get training data
x_all, y_all = dg.set_train_data(10000)
x_train, x_test, y_train, y_test = train_test_split(x_all.T, y_all.T, random_state=0)
x_train = x_train.T
x_test = x_test.T

#dg.show_data(x_train, y_train)

# init variables
w = np.asarray([0.1, 0.1])

iter = 0
while iter < max_iter:
    grad, err = get_grad_and_error(w, x_train, y_train)
    new_w = w - alpha * grad

    if np.sum(abs(new_w - w)) < tolerance:
        err_test = get_err(w, x_test, y_test)
        print('Converged: Iteration: %d - Test Error: %.4f' % (iter, err_test))
        break

    if iter % 100 == 0:
        print("Iteration: %d - Error: %.4f" % (iter, err))

    iter += 1
    w = new_w

dg.show_data(x_test, y_test, w)

pass