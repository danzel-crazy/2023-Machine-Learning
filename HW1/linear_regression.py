import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def your_model(train, w, b):
    y = np.ndarray(train.shape)
    y.fill(0)
    print(train[0])
    for i in range(len(train)):
        y[i] = (train[i]*w[i]+b[i])
    print(y[0]) 
    return y

def Mean_square_error(test, pred, w, b):
    error = 0
    print(test[0])
    # print(pred[0])
    for i in range(len(test)):
        error += (test[i] - (w[i]*pred[i] + b[i])) ** 2
        # print(error)
    return error/len(test)

def linear_regression(w, b, x, y, theta):
    new_w = w
    new_b = b
    iterations = 100
    n = len(x)
    for j in range(iterations):
        gradient_b = 0
        gradient_w = 0 
        for i in range(n):
            gradient_b += -(2/n) * (y[i] - ((new_w[i] * x[i]) + new_b[i]))
            gradient_w += -(2/n) * x[i] * (y[i] - ((new_w[i] * x[i]) + new_b[i]))
        
        for i in range(len(new_w)):
            new_w[i][0] -= theta * gradient_w
            new_b[i][0] -= theta * gradient_b
    
    return new_w, new_b

def main():
    x_train, x_test, y_train, y_test = np.load('C:/Users/danzel/Hsu/課程/大三上/機器學習/HW1/regression_data.npy', allow_pickle=True)
    #plt.plot(x_train, y_train, '.')
    #plt.show()
    init_b = np.ndarray(x_train.shape)
    init_b.fill(0)
    init_w = np.ndarray(x_train.shape)
    init_w.fill(np.random.standard_normal())
    #print(init_w)
    learning_rate = 0.0001
    new_w, new_b = linear_regression(init_w, init_b, x_train, y_train, learning_rate)
    y_pred = your_model(x_test, new_w, new_b)
    print(Mean_square_error(y_test, y_pred, new_w, new_b))

if __name__ == '__main__':
    main()