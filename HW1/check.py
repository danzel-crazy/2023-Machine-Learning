import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

def main():
    x_train, x_test, y_train, y_test = np.load('C:/Users/danzel/Hsu/課程/大三上/機器學習/HW1/classification_data.npy', allow_pickle=True)
    model=linear_model.LogisticRegression()
    model.fit(x_train,y_train)
    model.predict(x_test)
    model.predict_proba(x_test)
    score = model.score(x_test,y_test)
    print(model.coef_)
    print(model.intercept_)
    print(score)


if __name__ == '__main__':
    main()