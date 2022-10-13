import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
def your_model(train, w, b):
    y = np.ndarray(train.shape)
    y.fill(0)
    for i in range(len(train)):
        y[i] = (1/(1 + math.exp(-(train[i][0]*w[0][0] + b[0][0]))))
    return y

def Cross_entropy_error(test, pred, w, b):
    error = 0
    n = len(test)
    for i in range(n):
        error -= (test[i]*math.log(pred[i]) + (1-test[i])*math.log((1-pred[i])))
    return error

def logistic_regression(w, b, x, y, theta):
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
        omega = your_model(x, new_w, new_b)
        for i in range(n):
            gradient_w += (omega[i]-y[i])*x[i][0]
            gradient_b += (omega[i]-y[i]) 
        
        for i in range(n):
            new_w[i][0] -= theta*gradient_w
            new_b[i][0] -= theta*gradient_b
        # temp_y = your_model(x, new_w, new_b)
        error = Cross_entropy_error(y, omega, new_w, new_b)

        plt.scatter(j, error)
        plt.pause(0.05)
    
    return new_w, new_b

def main():
    x_train, x_test, y_train, y_test = np.load('C:/Users/danzel/Hsu/課程/大三上/機器學習/HW1/classification_data.npy', allow_pickle=True)
    init_b = np.ndarray(x_train.shape)
    init_b.fill(0)
    init_w = np.ndarray(x_train.shape)
    init_w.fill(np.random.normal())
    learning_rate = 0.02
    new_w, new_b = logistic_regression(init_w, init_b, x_train, y_train, learning_rate)
    print('weights:',new_w[0][0])
    print('intercepts:',new_b[0][0])
    y_pred = your_model(x_test, new_w, new_b)
    print('Cross Entropy Error:',Cross_entropy_error(y_test, y_pred, new_w, new_b))
    plt.show()

if __name__ == '__main__':
    main()