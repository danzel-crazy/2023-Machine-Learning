import numpy as np
import matplotlib as plot

def mean_vectors():
    
    return m1, m2

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = np.load('classification_data.npy', allow_pickle=True)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    m1, m2 = mean_vectors(x_train)
    print(f"mean vector of class 1: {m1}", f"mean vector of class 2: {m2}")