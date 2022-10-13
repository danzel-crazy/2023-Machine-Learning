import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def your_model(train, w, b):
    y = np.ndarray(train.shape)
    y.fill(0)
    for i in range(len(train)):
        y[i] = (train[i][0]*w[0][0]+b[0][0])
    return y

def Mean_square_error(test, pred, w, b):
    error = 0
    for i in range(len(test)):
        error += (test[i] - pred[i][0]) ** 2
    error = error/len(test)
    return error/2

def linear_regression(w, b, x, y, theta):
    error = 0
    temp_y = np.ndarray(x.shape)
    temp_y.fill(0)
    new_w = w
    new_b = b
    iterations = 100
    n = len(x)
    for j in range(iterations):
        gradient_b = 0
        gradient_w = 0 
        for i in range(n):
            gradient_b += -(2/n) * (y[i] - ((new_w[i][0] * x[i][0]) + new_b[i][0]))
            gradient_w += -(2/n) * x[i][0] * (y[i] - ((new_w[i][0] * x[i][0]) + new_b[i][0]))
            
        for i in range(n):
            new_w[i][0] -= theta * gradient_w
            new_b[i][0] -= theta * gradient_b
        temp_y = your_model(x, new_w, new_b)
        error = Mean_square_error(y, temp_y, new_w, new_b)

        plt.scatter(j, error)
        plt.pause(0.05)
    
    return new_w, new_b

def main():
    x_train, x_test, y_train, y_test = np.load('C:/Users/danzel/Hsu/課程/大三上/機器學習/HW1/regression_data.npy', allow_pickle=True)
    init_b = np.ndarray(x_train.shape)
    init_b.fill(0)
    init_w = np.ndarray(x_train.shape)
    init_w.fill(np.random.normal())
    learning_rate = 0.2
    new_w, new_b = linear_regression(init_w, init_b, x_train, y_train, learning_rate)
    print('weights:',new_w[0][0])
    print('intercepts:',new_b[0][0])
    y_pred = your_model(x_test, new_w, new_b)
    print('Mean_square_error:',Mean_square_error(y_test, y_pred, new_w, new_b))
    plt.show()
if __name__ == '__main__':
    main()