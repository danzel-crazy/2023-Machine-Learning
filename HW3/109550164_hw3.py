import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import cm
# Copy and paste your implementations right here to check your result
# (Of course you can add your classes not written here)

def gini(sequence):
    c1, c2 = 0, 0
    for i in sequence:
        if i == 1:
            c1 += 1
        else :
            c2 += 1
    
    return 1 - pow((c1/len(sequence)), 2) - pow((c2/len(sequence)), 2)

def entropy(sequence):
    c1, c2 = 0, 0
    for i in sequence:
        if i == 1:
            c1 += 1
        else :
            c2 += 1
    if c1 == 0 or c2 == 0: 
        entro = 0
    else :
        entro = - ((c1/len(sequence)) * math.log2(c1/len(sequence)) + (c2/len(sequence)) * math.log2(c2/len(sequence)))
    return entro

def partition(col, rows, cl, question):
    """Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_val, true_cl, false_val, false_cl = [], [], [], []
    
    for index in range(len(rows)):
        if rows[index][col] >= question:
            true_val.append(rows[index])
            true_cl.append(cl[index])
        else:
            false_val.append(rows[index])
            false_cl.append(cl[index])
    
    return true_val, true_cl, false_val, false_cl

def class_counts(y_train) :
    cl = 0
    c1, c2 = 0, 0
    for i in y_train:
        if i == 1:
            c1 += 1
        else :
            c2 += 1

    if c1 > c2 : 
        cl = 1
    else : 
        cl = 0

    return cl

def find_best_decision(x_data, y_data, y_train, criterion):
    best_gain = None  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    n_features = len(x_data)  # number of columns
    b_true_val = []
    b_true_cl = []
    b_false_val = [] 
    b_false_cl = []

    for col in range(n_features):
        temp = sorted(y_data, key = lambda s: s[col])
        # for index in y_data:
        for index in range(len(temp)):
            if(index+1 == len(temp)) : break
            # question = Question(col, val)
            val = (temp[index][col] + temp[index+1][col])/2
            # try splitting the dataset
            true_val, true_cl, false_val, false_cl = partition(col, y_data, y_train, val)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_cl) == 0 or len(false_cl) == 0:
                continue

            # Calculate the information gain from this split
            if criterion == 'gini' : 
                gain = ((len(true_cl)/len(y_data)) * gini(true_cl) + (len(false_cl)/len(y_data)) *gini(false_cl))/2
            elif criterion == 'entropy' :
                gain = ((len(true_cl)/len(y_data)) *entropy(true_cl) + (len(false_cl)/len(y_data)) *entropy(false_cl))/2
            # print(gain)
            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if best_gain == None or gain <= best_gain:
                best_gain, best_question = gain, [col, val, x_data[col]]
                b_true_val, b_true_cl, b_false_val, b_false_cl = true_val, true_cl, false_val, false_cl

    
    # print(best_gain)
    return best_gain, best_question, b_true_val, b_true_cl, b_false_val, b_false_cl

class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

class Leaf:
    def __init__(self, y_train):
        # print(len(y_train))
        self.cl = class_counts(y_train)
        # print(self.cl)
    
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.max_depth = max_depth
        self.acc = 0
        self.criterion = criterion
        self.features = {}

    def fit(self, x_data, y_data, y_train, depth):
        if depth == self.max_depth : return Leaf(y_train)
        if len(y_data) == 1 : return Leaf(y_train)

        gain, question, true_val, true_cl, false_val, false_cl = find_best_decision(x_data, y_data, y_train, self.criterion)

        # true_val, true_cl, false_val, false_cl = partition(question[0], y_data, y_train, question[1])
        
        true_branch = self.fit(x_data, true_val, true_cl, depth+1)

        false_branch = self.fit(x_data, false_val, false_cl, depth+1)

        print(question)
        if self.features.get(question[2]) == None :
            self.features[question[2]] = 1
        else :
            self.features[question[2]] += 1

        return Decision_Node(question, true_branch, false_branch) 


    def predict(self, my_tree, x_data, y_data, y_train, depth):
        if depth == self.max_depth :
            for i in y_train :
                # self.pred[i[0]] = my_tree.cl
                if i == my_tree.cl :
                    self.acc += 1
            return

        if type(my_tree) == (Leaf) : 
            # self.pred[y_train[0]] = my_tree.cl
            if y_train == my_tree.cl :
                self.acc += 1
            return

        true_val, true_cl, false_val, false_cl = partition(my_tree.question[0], y_data, y_train, my_tree.question[1])
        
        self.predict(my_tree.true_branch , x_data, true_val, true_cl, depth+1)

        self.predict(my_tree.false_branch, x_data, false_val, false_cl, depth+1)

        return 

if __name__ == '__main__':
    #Question 1
    data = np.array([1,2,1,1,1,1,2,2,1,1,2])
    print("Gini of data is ", gini(data))
    print("Entropy of data is ", entropy(data))

    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('val.csv')
    # print(train_df.shape)
    # print(val_df.shape)

    # print(train_df.head())
    # print(val_df.head())

    x_train = train_df.drop(labels=["price_range"], axis="columns")
    feature_names = x_train.columns.values
    x_train = x_train.values
    y_train = train_df['price_range'].values

    x_test = val_df.drop(labels=["price_range"], axis="columns")
    x_test = x_test.values
    y_test = val_df['price_range'].values
    # print(len(x_train))

    #Question 2-1
    clf_depth3 = DecisionTree(criterion='gini', max_depth=3)
    clf_depth10 = DecisionTree(criterion='gini', max_depth=10)

    my_tree3 = clf_depth3.fit(feature_names, x_train, y_train, 0)
    check_accuracy = clf_depth3.predict(my_tree3, feature_names, x_test, y_test, 0)
    print(clf_depth3.acc/len(y_test))

    my_tree = clf_depth10.fit(feature_names, x_train, y_train, 0)
    check_accuracy = clf_depth10.predict(my_tree, feature_names, x_test, y_test, 0)
    print(clf_depth10.acc/len(y_test))
    # print(clf_depth10.features)
    
    #Question 2-2
    clf_gini = DecisionTree(criterion='gini', max_depth=3)
    clf_entropy = DecisionTree(criterion='entropy', max_depth=3)

    my_tree = clf_gini.fit(feature_names, x_train, y_train, 0)
    
    check_accuracy = clf_gini.predict(my_tree, feature_names, x_test, y_test, 0)
    print(clf_gini.acc/len(y_test))
    # print(clf_gini.features)

    my_tree = clf_entropy.fit(feature_names, x_train, y_train, 0)
    check_accuracy = clf_entropy.predict(my_tree, feature_names, x_test, y_test, 0)
    print(clf_entropy.acc/len(y_test))
    # print(clf_entropy.features)

    #Question 3
    cmap = cm.jet(np.linspace(0, 1, len(clf_depth10.features)))
    plt.barh(*zip(*clf_depth10.features.items()), color=cmap)
    plt.ylabel('Features')
    plt.xlabel('Times')
    plt.title('Feature Importance')
    plt.show()

    # #Question 4
    # ada10 = AdaBoost(my_tree3, n_estimators=10)
    # ada100 = AdaBoost(my_tree3, n_estimators=100)

    # ada10.fit(x_train, y_train)
    # ada10.predict(x_test)
    # print(ada10.acc/len(y_test))


    # ada100.fit(feature_names, x_train, y_train)
    # ada100.predict(feature_names, x_test, y_test)
    # print(ada100.acc/len(y_test))
    # #Question 5-1
    # clf_10tree = RandomForest(n_estimators=10, max_features=np.sqrt(x_train.shape[1]))
    # clf_100tree = RandomForest(n_estimators=100, max_features=np.sqrt(x_train.shape[1]))