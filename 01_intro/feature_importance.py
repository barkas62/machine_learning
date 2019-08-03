import numpy as np
from sklearn.linear_model import LinearRegression

M = 1000
n_feat = 3
x_train = np.random.rand(M,n_feat)
y_train = 2.0 * x_train[:,0] + 3.0 * x_train[:,1] + 4
y_train += np.random.uniform(-0.1, 0.1, M)

lr = LinearRegression(copy_X = True, fit_intercept = True)
lr.fit(x_train, y_train)

y_model = lr.predict(x_train)

err = np.sum( np.square(y_model - y_train) ) / M

mid = M//2
feature_importance = np.empty((n_feat,), dtype = float)
for i_feat in range(n_feat):
    # for i_feat column (feature) swap 0-mid and mid-M values
    x_swap = x_train[:,:].copy()
    x_swap[:mid, i_feat] = x_train[mid:, i_feat]
    x_swap[mid:, i_feat] = x_train[:mid, i_feat]
    # predictions for swapped features
    y_model_swap = lr.predict(x_swap)
    err_swap = np.sum( np.square(y_model_swap - y_train) ) / M
    feature_importance[i_feat] = err_swap / err

score = lr.score(x_train, y_train)

pass