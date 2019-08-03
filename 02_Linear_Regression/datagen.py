import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def set_train_data(data_size = 50, add_ones = True, var = 0.1):
    # Genrating random linear data
    # There will be 50 data points ranging from 0 to 50
    x = np.linspace(0, data_size, data_size)
    x /= np.max(x)
    x += np.random.uniform(-0.1, 0.1, data_size)
    if add_ones:
        x = np.vstack((x, np.ones(data_size)))
    y = np.linspace(0, data_size, data_size)
    y /= np.max(y)
    # Adding noise to the random linear data

    y += np.random.uniform(-var, +var, data_size)
    return x, y

# def set_train_data(data_size = 50, add_ones = True, var = 0.1):


def show_data(x, y, w = None):
    # Plot of Training Data
    if x.shape[1] > 1:
        x, _ = np.vsplit(x,2)
    plt.scatter(x, y)
    if w is not None:
        x_w = [[np.min(x), 1], [np.max(x), 1]]
        y_w = list(w.dot(x_w))
        x_w = [np.min(x), np.max(x)]
        plt.plot(x_w, y_w, color = 'C1')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Training Data")
    plt.show()