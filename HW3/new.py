import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from collections import Counter

def common_label(y_data) :
    counter = Counter(y_data)
    label = counter.most_common(1)[0][0]
    return label

def relabel(y_data) :
    for  i in range(len(y_data)) :
        if y_data[i] == 0 :
            y_data[i] = -1

    return y_data 

def compute_error(y, y_pred, w_i):
    
    return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)

def compute_alpha(error):
    
    return np.log((1 - error) / error) / 2

def update_weights(w_i, alpha, y, y_pred):
    # print(np.not_equal(y, y_pred))
    return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))

def gini_weight(sequence, col, w):
    c1, c2 = 0, 0
    for i, idx in zip(sequence, col) :
        if i == 1:
            c1 += 1 * w[idx]
        else :
            c2 += 1 * w[idx]
    
    return 1 - pow((c1/(c1+c2)), 2) - pow((c2/(c1+c2)), 2)

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
    if c1 == len(y_train) or c2 == len(y_train) :
        return 3
    if c1 > c2 : 
        cl = 1
    else : 
        cl = 0

    return cl

class Node:

    def __init__(self):
        self.col = None
        self.question = None
        self.cl = None
        self.depth = 0
        self.true_branch = None
        self.false_branch = None
        self.leaf = False

class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.max_depth = max_depth
        self.criterion = criterion
        self.x_data = None
        self.y_data = None
        self.tree = None
        self.features = []
        self.sample_weight = 1
        self.ada = False    
        self.max_features = 1
        self.boostrap = False

    def find_low_error(self, lists, x_data, y_data) :
        best_gain = None  
        best_question = None
        bestCol = None
        for col in lists: 
            val = []
            for i in x_data : 
                val.append(i[col])

            for i in val : 
                true_col, false_col = [], []
                true_branch, false_branch = [], []
                true_w, false_w = [], []
                question = i
                if self.ada == True :
                    true_branch = y_data[val >= question]
                    true_col = [ index for index, i in enumerate(y_data[val >= question])]
                    true_w = [self.sample_weight[index] for index, i in enumerate(y_data[val >= question])]

                    false_branch = y_data[val < question]
                    false_col = [ index for index, i in enumerate(y_data[val < question])]
                    false_w = [self.sample_weight[index] for index, i in enumerate(y_data[val < question])]
                else :
                    true_branch = y_data[val >= question]
                    false_branch = y_data[val < question]

                if len(true_branch) == 0 or len(false_branch) == 0:
                    continue

                if self.criterion == 'gini' : 
                    if self.ada == False :
                        gain = ((len(true_branch)/len(y_data)) * gini(true_branch) + (len(false_branch)/len(y_data)) *gini(false_branch))
                    else :
                        gain = (sum(true_w)/sum(self.sample_weight)) * gini_weight(true_branch, true_col, self.sample_weight) + (sum(false_w)/sum(self.sample_weight)) *gini_weight(false_branch, false_col, self.sample_weight)
                elif self.criterion == 'entropy' :
                    gain = (entropy(true_branch) + entropy(false_branch))
            
                if best_gain == None or gain <= best_gain:
                    bestCol = col
                    best_gain, best_question = gain, question
        
        return bestCol, best_gain, best_question

    def find_best_decision(self, x_data, y_data):
        best_gain = None  
        best_question = None
        bestCol = None  
        current_gini = gini(y_data)
        current_entropy = entropy(y_data)

        if self.boostrap == True :
            lists =  np.random.choice(x_data.shape[1], self.max_features, replace = True)
            bestCol, best_gain, best_question = self.find_low_error(lists, x_data, y_data)
        else :
            for col in range(len(x_data[0])):
                val = []
                for i in x_data : 
                    val.append(i[col])

                for i in val : 
                    true_col, false_col = [], []
                    true_branch, false_branch = [], []
                    true_w, false_w = [], []
                    question = i
                    if self.ada == True :
                        true_branch = y_data[val >= question]
                        true_col = [ index for index, i in enumerate(y_data[val >= question])]
                        true_w = [self.sample_weight[index] for index, i in enumerate(y_data[val >= question])]

                        false_branch = y_data[val < question]
                        false_col = [ index for index, i in enumerate(y_data[val < question])]
                        false_w = [self.sample_weight[index] for index, i in enumerate(y_data[val < question])]
                    else :
                        true_branch = y_data[val >= question]
                        false_branch = y_data[val < question]

                    if len(true_branch) == 0 or len(false_branch) == 0:
                        continue

                    if self.criterion == 'gini' : 
                        if self.ada == False :
                            gain = ((len(true_branch)/len(y_data)) * gini(true_branch) + (len(false_branch)/len(y_data)) *gini(false_branch))
                        else :
                            gain = (sum(true_w)/sum(self.sample_weight)) * gini_weight(true_branch, true_col, self.sample_weight) + (sum(false_w)/sum(self.sample_weight)) *gini_weight(false_branch, false_col, self.sample_weight)
                    elif self.criterion == 'entropy' :
                        gain = (entropy(true_branch) + entropy(false_branch))
                
                    if best_gain == None or gain <= best_gain:
                        bestCol = col
                        best_gain, best_question = gain, question
        # if best_gain == None :
        #     return None, None, None, None, None, None

        self.features.append(bestCol)
        # best_val = []
        # for i in x_data : 
        #     best_val.append(i[bestCol])
        best_val = [i[bestCol] for i in x_data]

        x_true, x_false = x_data[best_val >= best_question, :], x_data[best_val < best_question, :]
        y_true, y_false = y_data[best_val >= best_question], y_data[best_val < best_question]
        
        return bestCol, best_question, x_true, x_false, y_true, y_false

    def build_tree(self, x_data, y_data, node):
        if node.depth == self.max_depth : 
            # if self.ada == True :
            #     if class_counts(y_data) == 0:
            #         node.cl = -1
            #     else :
            #         node.cl = class_counts(y_data)
            # else : node.cl = class_counts(y_data)
            node.cl = class_counts(y_data)
            return

        if class_counts(y_data) == 3 :
            node.leaf = True
            node.cl = y_data[0]
            return

        bestCol, best_question, x_true, x_false, y_true, y_false = self.find_best_decision(x_data, y_data)
       
        # print(bestCol, best_question, len(x_true), len(x_false))

        node.col = bestCol
        node.question = best_question
        node.true_brach = Node()
        node.false_brach = Node()

        node.true_brach.depth = node.depth + 1
        node.false_brach.depth = node.depth + 1

        self.build_tree(x_true, y_true, node.true_brach)
        self.build_tree(x_false, y_false, node.false_brach)
    
    def prediction(self, x_data, node):
        if node.depth == self.max_depth:
            # print(node.cl)
            return node.cl

        if node.leaf == True :
            return node.cl

        if x_data[node.col] >= node.question :
            return self.prediction(x_data, node.true_brach)
        else:
            return self.prediction(x_data, node.false_brach)

    def fit(self, x_data, y_data, ada=False, boostrap=False, max_features=1, sample_weight=1):
        self.ada = ada
        self.boostrap = boostrap
        self.sample_weight = sample_weight
        self.max_features = max_features
        self.tree = Node()
        self.tree.depth = 0
        self.build_tree(x_data, y_data, self.tree)

    def predict(self,x_data):
        pred = []
        for x in x_data:
            y = self.prediction(x, self.tree)
            pred.append(y)
        pred = np.asarray(pred)

        return pred
    
class AdaBoost():
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.acc = 0
        self.tree = None
        self.alphas = []
        self.clf = []   
        
    def fit(self, x_data, y_data):
        # y_data = relabel(y_data)
        
        for m in tqdm(range(0, self.n_estimators)):
            # print(m)
            if m == 0:
                w_i = np.ones(len(y_data)) * 1 / len(y_data)
            else:
                w_i = update_weights(w_i, alpha, y_data, y_pred)
                w_i = w_i/sum(w_i)
            
            print(w_i)
            
            clf_depth3 = DecisionTree(criterion='gini', max_depth=1)
            clf_depth3.fit(x_data, y_data, ada = True , sample_weight = w_i)
            y_pred = clf_depth3.predict(x_data)
            
            self.clf.append(clf_depth3)

            error = compute_error(y_data, y_pred, w_i)

            alpha = compute_alpha(error)
            self.alphas.append(alpha)

    def predict(self, x_data):
        preds = pd.DataFrame(index = range(len(x_data)), columns = range(self.n_estimators)) 

        for m in range(self.n_estimators):
            y_pred_m = self.clf[m].predict(x_data) * self.alphas[m]
            preds[preds.columns[m]] = y_pred_m
        y_pred = (1 * np.sign(preds.T.sum())).astype(int)
        return y_pred

class RandomForest():
    def __init__(self, n_estimators, max_features, boostrap=True, criterion='gini', max_depth=None):
        self.n_estimators = n_estimators
        self.max_features = math.ceil(max_features)
        self.boostrap = boostrap
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = []

    def fit(self, x_data, y_data):

        for i in tqdm(range(self.n_estimators)) :
            # print(self.max_features)
            clf_depth3 = DecisionTree(criterion='gini', max_depth=3)
            idx = np.random.choice(len(x_data), len(x_data), replace = True)
            # print(chosen_idx)
            new_x, new_y = x_data[idx], y_data[idx]
            clf_depth3.fit(new_x, new_y, boostrap=self.boostrap, max_features=self.max_features)
            self.tree.append(clf_depth3)

    def predict(self, x_data):
        tree_predicts = np.array([tree.predict(x_data) for tree in self.tree])
        tree_predicts = np.swapaxes(tree_predicts, 0, 1)
        y_pred = [common_label(tree_pred) for tree_pred in tree_predicts]
        return np.array(y_pred)
    
def train_your_model(data):
    ## Define your model and training 
        return

if __name__ == '__main__':
    #Question 1
    data = np.array([1,2,1,1,1,1,2,2,1,1,2])
    print("Gini of data is ", gini(data))
    print("Entropy of data is ", entropy(data))

    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('val.csv')

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

    # clf_depth3.fit(x_train, y_train)
    # y_pred = clf_depth3.predict(x_test)
    # acc = 0
    # for i in range(len(y_test)) : 
    #     if y_test[i] == y_pred[i]:
    #         acc += 1
    # # print('Test-set accuarcy score: ', acc/len(y_test))
    # print('Test-set accuarcy score: ', accuracy_score(y_test, y_pred))

    # clf_depth10.fit(x_train, y_train)
    # y_pred = clf_depth10.predict(x_test)
    # print('Test-set accuarcy score: ', accuracy_score(y_test, y_pred))

    # #Question 2-2
    # clf_gini = DecisionTree(criterion='gini', max_depth=3)
    # clf_entropy = DecisionTree(criterion='entropy', max_depth=3)

    # clf_gini.fit(x_train, y_train)
    # y_pred = clf_gini.predict(x_test)
    # print('Test-set accuarcy score: ', accuracy_score(y_test, y_pred))

    # clf_entropy.fit(x_train, y_train)
    # y_pred = clf_entropy.predict(x_test)
    # print('Test-set accuarcy score: ', accuracy_score(y_test, y_pred))
    # print(clf_entropy.features)

    # #Question 3
    # features = {}
    # # print(len(clf_depth10.features))
    # for i in clf_depth10.features :
    #     if features.get(feature_names[i]) == None :
    #         features[feature_names[i]] = 1
    #     else :
    #         features[feature_names[i]] += 1
    # cmap = cm.jet(np.linspace(0, 1, len(features)))
    # plt.barh(*zip(*features.items()), color=cmap)
    # plt.ylabel('Features')
    # plt.xlabel('Times')
    # plt.title('Feature Importance')
    # plt.show()

    #Question 4
    # ada10 = AdaBoost(n_estimators=10)
    # ada100 = AdaBoost(n_estimators=100)

    # ada10.fit(x_train, y_train)
    # y_pred = ada10.predict(x_test)
    # print('Test-set accuarcy score: ', accuracy_score(y_test, y_pred))

    # ada100.fit(x_train, y_train)
    # y_pred = ada100.predict(x_test)
    # print('Test-set accuarcy score: ', accuracy_score(y_test, y_pred))

    #Question 5-1
    # clf_10tree = RandomForest(n_estimators=10, max_features=np.sqrt(x_train.shape[1]))
    # clf_100tree = RandomForest(n_estimators=100, max_features=np.sqrt(x_train.shape[1]))
    
    # clf_10tree.fit(x_train, y_train)
    # y_pred = clf_10tree.predict(x_test)
    # print('Test-set accuarcy score: ', accuracy_score(y_test, y_pred))

    # clf_100tree.fit(x_train, y_train)
    # y_pred = clf_100tree.predict(x_test)
    # print('Test-set accuarcy score: ', accuracy_score(y_test, y_pred))

    #Question 5-2
    clf_random_features = RandomForest(n_estimators=10, max_features=np.sqrt(x_train.shape[1]))
    clf_all_features = RandomForest(n_estimators=10, max_features=x_train.shape[1])

    clf_random_features.fit(x_train, y_train)
    y_pred = clf_random_features.predict(x_test)
    print('Test-set accuarcy score: ', accuracy_score(y_test, y_pred))

    clf_all_features.fit(x_train, y_train)
    y_pred = clf_all_features.predict(x_test)
    print('Test-set accuarcy score: ', accuracy_score(y_test, y_pred))

    #Question 6
    # my_model = train_your_model(train_df)
    # y_pred = my_model.predict(x_test)
    # print('Test-set accuarcy score: ', accuracy_score(y_test, y_pred))
    # assert y_pred.shape == (500, )