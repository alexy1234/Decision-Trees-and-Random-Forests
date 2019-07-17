
# coding: utf-8

# In[1]:

from collections import Counter
import math
import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats

from collections import defaultdict, OrderedDict

import random


# In[2]:

class Node:
    def __init__(self, split, left, right, label, leaf):
        self.split = split
        self.label = label
        self.left = left
        self.right = right
        self.leaf = leaf

class DecisionTree:
    def __init__(self, maxDepth = None, maxFeatures=None):
        """
        TODO: initialization of a decision tree
        """
        self.maxDepth = maxDepth
        self.maxFeatures = maxFeatures
        self.root = None
        
    @staticmethod
    def entropy(hist):
        total = sum(hist.values())
        return -sum([c/total * math.log(c/total, 2) if c > 0 
                     else 0 for c in hist.values()])

    @staticmethod
    def weighted_avg(left_hist, right_hist):
        
        left_entropy = DecisionTree.entropy(left_hist)
        right_entropy = DecisionTree.entropy(right_hist)
        
        left_count = sum(left_hist.values())
        right_count = sum(right_hist.values())
        
        
        total = left_count + right_count
        return (left_count * left_entropy + right_count * right_entropy) / total
    
    
    def split(self, data, labels, idx, thresh):
        """
        TODO: implement a method that return a split of the dataset given an index of the feature and
        a threshold for it
        """
        left = data[:, idx] <= thresh
        right = data[:, idx] > thresh
        left_data = data[left]
        right_data = data[right]
        left_labels = labels[left]
        right_labels = labels[right]
        return left_data, left_labels, right_data, right_labels
    
    """
    i minimized the weighted_avg instead 
    """
    def find_threshold(self, feature, labels):
        #TODO: Either one-hot encoding (hard coding) 
        #or determine split rules based on subsets of categorical variables
        
        splits = defaultdict(lambda: defaultdict(int))
        rightHist = defaultdict(int)
        leftHist = defaultdict(int)
        #all counts initially go to the right node in split
        #initialize set of all potential splits
        # feature value --> label --> count
        for i in range(len(feature)):
            splits[feature[i]][labels[i]] += 1
            rightHist[labels[i]] += 1
        

        min_split = math.inf
        best_thresh = None
        splits = OrderedDict(sorted(splits.items(), key=lambda k: k[0]))
        
        for thresh, label_cts in splits.items():
            for label, ct in label_cts.items():
                leftHist[label] += ct
                rightHist[label] -= ct
            temp_avg = DecisionTree.weighted_avg(leftHist, rightHist)
            #minimization
            if temp_avg < min_split:
                min_split = temp_avg
                best_thresh = thresh
                
        return best_thresh, min_split
        
        
    def segmenter(self, data, labels):
        """
        TODO: compute entropy gain for all single-dimension splits,
        return the feature and the threshold for the split that
        has maximum gain
        """
        #random selection of features & threshold for time's sake?
        if self.maxFeatures is not None:
            all_features = np.random.choice(range(data.shape[1]), 
                                            self.maxFeatures, replace = False)
        else:
            all_features = np.arange(data.shape[1])
        min_H = math.inf
        best_feature_ind = None
        best_thresh = None
        
        for i in all_features:
            threshold, H = self.find_threshold(data[:, i], labels)
            if H < min_H:
                min_H = H
                best_thresh = threshold
                best_feature_ind = i
        return best_feature_ind, best_thresh
    
    
    def grow_tree(self, data, labels, depth, prnt, classes, features):
        #recursively grows tree out, includes root case etc.
        tab = "\t"
        prnt_d = (depth-1)*tab
        
        if depth > self.maxDepth:
            result_label = Counter(labels).most_common(1)[0][0]
            if prnt == True:
                print(prnt_d + " |-class: " + classes[int(result_label)])
            return Node(None, None, None, result_label, leaf = True)
        else:
            if len(np.unique(labels)) == 1:
                return Node(None, None, None, 
                            np.unique(labels)[0], leaf = True)
            else:
                best_ind, best_thresh = self.segmenter(data, labels)
                left_data, left_labels, right_data, right_labels = self.split(data, labels,
                                                                              best_ind, best_thresh)
                #build leaf for most common if one side runs dry
                if len(left_labels) == 0  or len(right_labels) == 0:
                    #print("leaf")
                    result_label = Counter(labels).most_common(1)[0][0]
                    if prnt == True:
                        print(prnt_d + " |-class: " + classes[int(result_label)])
                    return Node(None, None, None, result_label, leaf = True)
                else:
                    #recurse down
                    if prnt == True:
                        print(prnt_d + " |- feature: " +features[best_ind] + ", threshold: " + repr(best_thresh))
                    left_node = self.grow_tree(left_data, left_labels, 
                                               depth + 1, prnt, classes, features)
                    right_node = self.grow_tree(right_data, right_labels, depth + 1, 
                                                prnt, classes, features)

                    return Node((best_ind, best_thresh), left_node, right_node, None, leaf = False)
    
    def fit(self, data, labels, prnt = False, classes = None, features = None):
        """
        TODO: fit the model to a training set. Think about what would be 
        your stopping criteria
        calls grow_tree
        """
        self.root = self.grow_tree(data, labels, 1, prnt, classes, features)
        #return 0

    def traverse(self, root, pt, prnt, classes, features):
        if root.leaf == True:
            if prnt == True:
                print("This is therefore: " + repr(classes[int(root.label)]))
                print("--------------------")
            return root.label
        else:
            idx, thresh = root.split[0], root.split[1]
            
            if pt[idx] <= thresh:
                if prnt == True:
                    print('feature: ',features[int(idx)], ' <= ',thresh )

                return self.traverse(root.left, pt, prnt, classes, features)
            else:
                if prnt == True:
                    print('feature: ', features[int(idx)], ' >', thresh)

                return self.traverse(root.right, pt, prnt, classes, features)
        
    def predict(self, data, prnt = False, classes = None, features = None):
        """
        TODO: predict the labels for input data 
        """
        results = []
        for pt in data:
            results.append(self.traverse(self.root, pt, prnt, classes, features))
        return results

    def __repr__(self):
        """
        TODO: one way to visualize the decision tree is to write out a __repr__ method
        that returns the string representation of a tree. Think about how to visualize 
        a tree structure. You might have seen this before in CS61A.
        """
        return 0

    def score(self, data, true_labels):
        labels = self.predict(data)
        N = len(labels)
        return np.sum(labels == true_labels) / float(N)


# \newpage

# In[66]:

from sklearn.utils import resample

class RandomForest():
    
    def __init__(self, n_trees, max_depth, max_features):
        """
        TODO: initialization of a random forest
        """
        self.trees = []
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features #sqrt(d) is recommended

    def fit(self, data, labels):
        """
        TODO: fit the model to a training set.
        """
        seed_no = 421
        #randomly select data - diff seed per each iteration!
        for i in range(self.n_trees):
            np.random.seed(seed_no)
            tree = DecisionTree(self.max_depth, self.max_features)
            tree_data, tree_labels = resample(data, labels)
            tree.fit(tree_data, tree_labels)
            self.trees.append(tree)
            tree.fit(tree_data, tree_labels)
            self.trees.append(tree)
            seed_no += 1
    
    def predict(self, data):
        """
        TODO: predict the labels for input data 
        """
        pred_total = []
        preds = []
        for tree in self.trees:
            pred_tree = tree.predict(data)
            pred_total.append(pred_tree)
        for i in range(data.shape[0]):
            pred_sample = [pred_total[j][i] for j in range(self.n_trees)]
            preds.append(int(Counter(pred_sample).most_common(1)[0][0]))
        return preds
 
    def score(self, data, true_labels):
        labels = self.predict(data)
        N = len(labels)
        return np.sum(labels == true_labels) / float(N)


# \newpage

# In[4]:


if __name__ == "__main__":
 #dataset = "titanic"
 dataset = "spam"

 if dataset == "spam":
     spam_features = [
         "pain", "private", "bank", "money", "drug", "spam", "prescription",
         "creative", "height", "featured", "differ", "width", "other",
         "energy", "business", "message", "volumes", "revision", "path",
         "meter", "memo", "planning", "pleased", "record", "out",
         "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
         "square_bracket", "ampersand"
     ]
     assert len(spam_features) == 32

     # Load spam data
     path_train = 'datasets/spam-dataset/spam_data.mat'
     data = scipy.io.loadmat(path_train)
     spam_train = data['training_data']
     spam_labels = np.squeeze(data['training_labels'])
     spam_test = data['test_data']
     spam_class_names = ["Ham", "Spam"]
      
 else:
     raise NotImplementedError("Dataset %s not handled" % dataset)
 
 """
 TODO: train decision tree/random forest on different datasets and perform the tasks 
 in the problem
 """


# ### SPAM SPAM SPAM

# In[5]:

spam_train.shape


# In[55]:

spam_test.shape


# In[6]:

eighty_20 = int(np.round(spam_train.shape[0]*0.2))


# In[7]:

eighty_20


# In[8]:

def validation_creation(train_data, train_labels, size):    
    np.random.seed(43)
    randomize = np.arange(train_data.shape[0])
    np.random.shuffle(randomize)
    train_data = train_data[randomize]
    train_labels = train_labels[randomize]
    indices = np.random.choice(train_data.shape[0], size, replace = False)
    validation_set = train_data[indices]
    validation_labels = train_labels[indices]
    training_set = np.delete(train_data, indices ,axis=0)
    training_labels = np.delete(train_labels, indices, axis=0)

    return training_set, training_labels, validation_set, validation_labels


# In[9]:

spam_tset, spam_tlabels, spam_vset, spam_vlabels = validation_creation(spam_train, spam_labels, eighty_20)


# In[10]:

np.random.seed(42)
t1 = DecisionTree(maxDepth = 15,maxFeatures= None)
t1.fit(spam_tset, spam_tlabels)


# \newpage

# #### 2.5.2: Given a data point of choosing from each class, state splits to classify it. 

# In[11]:

inds = np.array((0,2))
twoinds_data = spam_tset[inds]
twoinds_labels = spam_tlabels[inds]
twoinds_labels


# In[12]:

t1.predict(twoinds_data, prnt = True, classes = spam_class_names, features = spam_features)


# #### 2.5.3: With a 80/20 split, train decision trees from depth=1 to depth=40 and plot validation accuracies as a function of depth

# In[13]:

import matplotlib.pyplot as plt


# In[14]:

depths = range(1,41)
depths


# We already performed a random 80/20 training/validation split, so moving forward we can train the decision trees 

# In[15]:

scores = []
seed_no = 11
for d in depths:
    np.random.seed(seed_no)
    t = DecisionTree(maxDepth = d, maxFeatures = None) #test all features
    t.fit(spam_tset, spam_tlabels)
    score = t.score(spam_vset, spam_vlabels)
    scores.append(score)
    #print("seed: " + repr(seed_no) + " , score: " + repr(score) + " for depth: " + repr(d))
    seed_no += 1


# In[16]:

plt.plot(depths, scores)
plt.show()
plt.ylabel('Validation Accuracy')
plt.xlabel('Depth')


# Because only one tree is being used as opposed to a random forest, there are occasional spikes and dips in accuracy. Overall, however, the validation accuracy tends to converge at approximately a depth of 20.

# \newpage

# #### Decision Trees Training and Validation Accuracy (wrt Spam)

# In[17]:

train_score = t1.score(spam_tset, spam_tlabels)


# In[18]:

print("training score for Spam Decision Tree is: " + repr(train_score))


# In[19]:

valid_score = t1.score(spam_vset, spam_vlabels)
print("validation score for Spam Decision Tree is: " + repr(valid_score))


# #### Random Forest Training and Validation Accuracy (wrt Spam)

# In[20]:

r1 = RandomForest(n_trees = 15, max_depth = 20, max_features = 4)


# In[21]:

r1.fit(spam_tset, spam_tlabels)


# In[22]:

train_score = r1.score(spam_tset, spam_tlabels)
print("training score for Spam Random Forests is: " + repr(train_score))
valid_score = r1.score(spam_vset, spam_vlabels)
print("validation score for Spam random Forests is: " + repr(valid_score))


# In[23]:

np.sqrt(spam_vset.shape[1])


# ### Spam Kaggle

# In[24]:

depths = np.arange(15,35,5)
n_trees = np.arange(40,140,15)
n_trees
np.random.seed(53)
for d in depths:
    for n in n_trees:
        r = RandomForest(n_trees = n, max_depth = d, max_features = 6) # sqrt of no. of features
        #9 features, so 3 features selected
        r.fit(spam_tset, spam_tlabels)
        score = r.score(spam_vset, spam_vlabels)
        print("score: " + repr(score) + " for depth: " + repr(d) + " and trees: " +repr(n))


# Choosing the best classifier in this instance would be of depth 25, with 85 trees:

# In[52]:

kaggle_r = RandomForest(n_trees = 85, max_depth = 25, max_features = 6)
kaggle_r.fit(spam_train, spam_labels)


# In[56]:

predictions = kaggle_r.predict(spam_test)


# In[58]:

import pandas as pd
final_sub = pd.DataFrame(columns = ['Id', 'Category'])
final_sub['Id'] = list(range(1, spam_test.shape[0]+1))
final_sub['Category'] = predictions
final_sub.to_csv('spam_test.csv', index = False)


# My Kaggle submission name is: Alex Yi. The test accuracy turned out to be 0.79738.

# \newpage

# ### Titanic Cleaning

# In[26]:

titanic_data = pd.read_csv("datasets/titanic/titanic_training.csv")
titanic_data_test = pd.read_csv("datasets/titanic/titanic_testing_data.csv")


# In[27]:

titanic_data.head()


# In[28]:

titanic_data.isna().any()


# Because there is a column w/out a label, we drop it:

# In[29]:

titanic_data.drop([705], inplace = True)
titanic_data.shape


# In[30]:

titanic_data.isna().any()


# We drop useless columns (too many categorical labels to be significant):

# In[31]:

titanic_labels = titanic_data.pop('survived')
titanic_data['has_Cabin'] = titanic_data["cabin"].apply(lambda x: 1 if type(x) == str else 0)
titanic_data.drop(['ticket', 'cabin'], inplace=True, axis=1)


# In[32]:

titanic_train_labels = titanic_labels.values


# We then fill in the columns with missing values as follows:

# In[33]:

titanic_data.head()


# In[34]:

titanic_data.age.fillna(value=titanic_data.age.mean(), inplace=True)
titanic_data.fare.fillna(value=titanic_data.fare.mean(), inplace=True)
titanic_data.embarked.fillna(value=(titanic_data.embarked.value_counts().idxmax()), inplace=True)


# We can now map categorical values with a distinct number of labels to that of discrete labels as follows:

# In[35]:

titanic_data['embarked'] = titanic_data['embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
titanic_data['sex'] = titanic_data['sex'].map( {'female': 0, 'male': 1} ).astype(int)


# In[36]:

titanic_data_test.head()


# We repeat this for testing:

# In[37]:

titanic_data_test['has_Cabin'] = titanic_data_test["cabin"].apply(lambda x: 1 if type(x) == str else 0)
titanic_data_test.drop(['ticket', 'cabin'], inplace=True, axis=1)
titanic_data_test.age.fillna(value=titanic_data.age.mean(), inplace=True)
titanic_data_test.fare.fillna(value=titanic_data.fare.mean(), inplace=True)
titanic_data_test.embarked.fillna(value=(titanic_data.embarked.value_counts().idxmax()), inplace=True)
titanic_data_test['embarked'] = titanic_data_test['embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
titanic_data_test['sex'] = titanic_data_test['sex'].map( {'female': 0, 'male': 1} ).astype(int)


# In[38]:

titanic_train_vals = titanic_data.values


# In[39]:

titanic_test_vals = titanic_data_test.values


# In[59]:

titanic_test_vals


# In[40]:

titanic_train_vals.shape


# \newpage

# In[41]:

t_tset, t_tlabels, t_vset, t_vlabels = validation_creation(titanic_train_vals, titanic_train_labels, 
                                                                       int(np.round(titanic_train_vals.shape[0]*0.2)))


# #### Decision Trees Training and Validation Accuracy (wrt Titanic)

# In[42]:

np.random.seed(42)
titanic_tree =DecisionTree(maxDepth = 15,maxFeatures= None)
titanic_tree.fit(t_tset, t_tlabels)
train_score = titanic_tree.score(t_tset, t_tlabels)
print("training score for Titanic Decision Tree is: " + repr(train_score))
valid_score = titanic_tree.score(t_vset, t_vlabels)
print("validation score for Titanic Decision Tree is is: " + repr(valid_score))


# #### Random Forests Training and Validation Accuracy (wrt Titanic)

# In[45]:

np.random.seed(42)
titanic_f = RandomForest(n_trees = 15, max_depth = 15,max_features= 3)
titanic_f.fit(t_tset, t_tlabels)
train_score = titanic_f.score(t_tset, t_tlabels)
print("training score for Titanic Random Forest is: " + repr(train_score))
valid_score = titanic_f.score(t_vset, t_vlabels)
print("validation score for Titanic Random Forest is is: " + repr(valid_score))


# ### Titanic Kaggle

# In[46]:

depths = np.arange(5,25,5)
n_trees = np.arange(40,130,10)
n_trees


# In[47]:

np.random.seed(52)
for d in depths:
    for n in n_trees:
        r = RandomForest(n_trees = n, max_depth = d, max_features = 3) # sqrt of no. of features
        #9 features, so 3 features selected
        r.fit(t_tset, t_tlabels)
        score = r.score(t_vset, t_vlabels)
        print("score: " + repr(score) + " for depth: " + repr(d) + " and trees: " +repr(n))


# We can see that the best model would be a forest of depth 5, with 110 trees:

# In[67]:

kaggle_t = RandomForest(n_trees = 110, max_depth = 5, max_features = 3)
kaggle_t.fit(titanic_train_vals, titanic_train_labels)


# In[69]:

predictions = kaggle_t.predict(titanic_test_vals)


# In[70]:

final_sub = pd.DataFrame(columns = ['Id', 'Category'])
final_sub['Id'] = list(range(1, titanic_test_vals.shape[0]+1))
final_sub['Category'] = predictions
final_sub.to_csv('titanic_test.csv', index = False)


# This was submitted to Kaggle under the username: Alex Yi, and recieved a score of: 0.87096.

# \newpage

# ### 2.6 Titanic Visualizations for a Shallow Tree

# In[48]:

titanic_classes = ["Died", "Survived"]
titanic_features = titanic_data.columns.values
titanic_features


# In[49]:

t_tree =DecisionTree(maxDepth = 3,maxFeatures= None)


# In[50]:

t_tree.fit(t_tset, t_tlabels, prnt = True, classes = titanic_classes, features = titanic_features)


# In[ ]:



