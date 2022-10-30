from turtle import register_shape
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics import accuracy_score

class fisher:
    def __init__(self, x, y, x_t, y_t):
        self.x = x
        self.y = y
        self.x_t = x_t
        self.y_t = y_t
        self.class_0 = []
        self.class_1 = []
        self.m1 = 0
        self.m2 = 0
        self.sw = 0
        self.sb = 0
        self.w = 0
        self.proj = 0
        self.proj_t = 0

    #compute the mean of two classes
    def mean_vectors(self):
        self.class_0, self.class_1 = [], []
        #count the number of class0 and class1
        for x, y in zip(self.x, self.y):
            if y == 0:
                self.class_0.append(x.flatten())
            else:
                self.class_1.append(x.flatten())
        
        self.class_0=np.asarray(self.class_0)
        self.class_1=np.asarray(self.class_1)
        #compute the mean
        self.m1 = np.mean(self.class_0, axis=0)
        self.m2 = np.mean(self.class_1, axis=0)

        print(f"mean vector of class 1: {self.m1}", f"mean vector of class 2: {self.m2}")

    #compute within class matrix 
    def within_class_matrix(self):
        temp = np.subtract(self.class_0, self.m1)
        # print(temp.shape)
        temp_0 = np.dot(temp.T, temp)
        temp = np.subtract(self.class_1, self.m2)
        temp_1 = np.dot(temp.T, temp)
        #add two classes' within class matrix
        self.sw = np.add(temp_0, temp_1)
        print(f"Within-class scatter matrix SW: {self.sw}")

    #compute between class matrix
    def between_class_matrix(self):
        temp = self.m2 - self.m1
        temp = np.array(np.reshape(temp, (2,1)))
        self.sb = np.dot(temp, temp.T)

        print(f"Between-class scatter matrix SB: {self.sb}")

    #compute the w by fisher linear discrimination
    def fisher_linear_discrimination(self):
        self.w = np.dot(np.linalg.inv(self.sw), (self.m2 - self.m1))
        temp = self.w
        self.w = temp / np.linalg.norm(temp)
        
        print(f" Fisher's linear discriminant: {self.w}")

    #compare test data and pjoted train data, find the accuracy 
    def compare(self):
        #compute projection
        proj = np.dot(self.w.T, self.x.T)
        proj_t = np.dot(self.w.T, self.x_t.T)
        self.proj = proj
        self.proj_t = proj_t
        
        #use k nearest neighbor to decide the class of test data
        for k in range(1,6) :
            y_pred = []
            for i in range(len(proj_t)):
                dist = np.abs(proj_t[i] - proj)
                nearest_neighbor_ids = dist.argsort()[:k]
                count_0 = 0
                count_1 = 0
                for idx in nearest_neighbor_ids:
                    if self.y[idx] == 0:
                        if k%2 == 0:
                            count_0 += 1/dist[idx]
                        else:
                            count_0 += 1            
                    else :
                        if k%2 == 0:
                            count_1 += 1/dist[idx]
                        else:
                            count_1 += 1

                if count_0 > count_1 :
                    y_pred.append(0)
                else :
                    y_pred.append(1)
            #use accuracy_score to compute the accuracy
            acc = accuracy_score(self.y_t, y_pred)
            print(f" k : {k} Accuracy of test-set {acc}")
        
    #plot the graph of test and train data and the line    
    def plot(self):
        fig, ph = plt.subplots(figsize=(10,8))
        #project the line
        plt.plot([-self.w[0] * 4, self.w[0] * 7], [-self.w[1] * 4, self.w[1] * 7], lw=3, color='green', alpha=.4)
        colors=['red','blue']
        color = ['r.:', 'b.:']

        #link train data with its projectd data
        proj = []
        for i in self.proj :
            proj.append(i * self.w)
        
        for idx, i in enumerate(self.y):
            if i == 0 :
                plt.plot([proj[idx][0], self.x[idx][0]], [proj[idx][1], self.x[idx][1]], 'r', alpha=.3)
            else :
                plt.plot([proj[idx][0], self.x[idx][0]], [proj[idx][1], self.x[idx][1]], 'b', alpha=.3)

        #link test data with its projectd data
        proj_t = []
        for i in self.proj_t :
            proj_t.append(i * self.w)
        for idx, i in enumerate(self.y_t):
            if i == 0 :
                plt.plot([proj_t[idx][0], self.x_t[idx][0]], [proj_t[idx][1], self.x_t[idx][1]], 'r', alpha=.3)
            else :
                plt.plot([proj_t[idx][0], self.x_t[idx][0]], [proj_t[idx][1], self.x_t[idx][1]], 'b', alpha=.3)

        #plot train data and test data and the graph
        plt.scatter(self.x[:, 0], self.x[:, 1], color=[colors[i] for i in self.y], s = 10)
        plt.scatter(self.x_t[:, 0], self.x_t[:, 1], color=[colors[i] for i in self.y_t], s = 10)
        plt.title(f"Projection Line: w= {self.w[1]/self.w[0]}, b=0")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.show()

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = np.load('classification_data.npy', allow_pickle=True)
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    
    #use the class fisher to represent all needed function and parameters
    fish=fisher(x_train, y_train, x_test, y_test)
    fish.mean_vectors()
    fish.within_class_matrix()
    fish.between_class_matrix()
    fish.fisher_linear_discrimination()
    fish.compare()
    fish.plot()