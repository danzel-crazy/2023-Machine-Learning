import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def cost(x, y, w, b):
    return 0

def linear_regression():
    return 0

def main():
    x_train, x_test, y_train, y_test = np.load('C:/Users/danzel/Hsu/課程/大三上/機器學習/HW1/regression_data.npy', allow_pickle=True)
    plt.plot(x_train, y_train, '.')
    plt.show()
    """y_pred = your_model(x_test)
    print(Mean_square_error(y_test, y_pred))"""

if __name__ == '__main__':
    main()