import numpy as np

def conv(data : np.array, filter : np.array, stride: int = 1)->np.array:
    '''
    2D confolution of array data and square array filter
    :param data:
    :param filter:
    :return:
    '''
    assert len(data.shape) == len(filter.shape) == 2
    assert filter.shape[0] == filter.shape[1]
    f = filter.shape[0]
    assert data.shape[0] >= f and data.shape[1] >= f

    m, n = data.shape[0], data.shape[1]
    m_out = np.int64(np.floor((m - f)/stride + 1))
    n_out = np.int64(np.floor((n - f)/stride + 1))

    result = np.zeros((m_out, n_out))

    i = 0
    i_out = 0
    while i + f <= m:
        j = 0
        j_out = 0
        while j + f <= n:
            slice = data[i:i+f, j:j+f]           # it's a copy, not a view
            mult = slice * filter                # element-wise multiplication
            result[i_out, j_out] = np.sum(mult)  # sum of all elements (no axis => sum them all)
            j_out += 1
            j += stride
        i_out += 1
        i += stride

    return result


data = np.array([[1,1,1,2,2,2,2],
                 [3,3,3,4,4,4,4],
                 [0,0,0,2,2,2,2],
                 [2,2,2,5,5,5,5],
                 [1,1,1,4,4,4,4],
                 [2,2,2,3,3,3,3],
                 [1,1,1,2,2,2,2],
                 ])

filter = np.array([
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
])

res = conv(data,filter, stride=2)

pass