import numpy as np

# two variables, 5 samples : equivalent to 2 features, 5 training examples
A = np.array([
    [0,4],
    [1,3],
    [2,2],
    [3,1],
    [4,0]
])

# numpy library function
cov1 = np.cov(A, rowvar = False) # rowvar : vars are in columns

mu = np.sum(A, axis=0) / A.shape[0]                  # mean

cov = np.dot((A - mu).T, (A - mu)) / (A.shape[0]-1) # unbiased, must = cov1

s2 = np.sum(np.square(A - mu), axis=0) / A.shape[0]  # variance
N = (A - mu) / np.sqrt(s2)
cov_normalized = np.dot(N.T, N) / (A.shape[0]-1)


pass