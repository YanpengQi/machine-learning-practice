
# coding: utf-8

# In[148]:


import sklearn
import numpy as np
import os
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
from sklearn import svm


# In[27]:


def calc_f1(true_labels,predicted_labels):
    # true_labels - list of true labels (1/-1)
    # predicted_labels - list of predicted labels (1/-1)
    # return precision, recall and f1
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    length = len(true_labels)
    #actual positive and labelled positive
    APLP = 0
    #labelled positive
    LP = 0
    #actual positive
    AP = 0
    for i in range(0, length):
        if (true_labels[i] == 1 and predicted_labels[i] == 1):
            AP += 1
            LP += 1
            APLP += 1
        elif(true_labels[i] == 1 and predicted_labels[i] != 1):
            AP += 1
        elif(true_labels[i] != 1 and predicted_labels[i] == 1):
            LP += 1

    precision = APLP / LP
    recall = APLP / AP


    f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1


# In[ ]:



class Classifier(object):
    def __init__(self, algorithm, x_train, y_train, iterations=1, averaged = False, eta = 1, alpha = 1.1):
        # Algorithm values can be Perceptron, Winnow, Adagrad, Perceptron-Avg, Winnow-Avg, Adagrad-Avg, SVM
        # Get features from examples; this line figures out what features are present in
        # the training data, such as 'w-1=dog' or 'w+1=cat'
        self.features = {feature for xi in x_train for feature in xi.keys()}
        
        if algorithm == 'Perceptron':
            #Initialize w, bias
            self.w, self.w['bias'] = {feature:0.0 for feature in self.features}, 0.0
            #Iterate over the training data n times
            for j in range(iterations):
                #Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    
                    y_hat = self.predict(xi)
                    
                    #Update weights if there is a misclassification
                    if yi != y_hat:
                        for feature, value in xi.items():
                            self.w[feature] = self.w[feature] + eta*(yi*value)
                        self.w['bias'] = self.w['bias'] + eta*yi

        elif algorithm == 'Winnow':
            #Initialize w, bias
            self.w = {feature:1.0 for feature in self.features}
            self.w['bias'] = -len(self.w)
            #Iterate over the training data n times
            for j in range(iterations):
                #Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    #Update weights if there is a misclassification
                    if yi != y_hat:
                        for feature, value in xi.items():
                            if yi == 1:
                                self.w[feature] = self.w[feature] * alpha
                            else:
                                self.w[feature] = self.w[feature] / alpha
                        
        elif algorithm == 'Adagrad':
            #Initialize w, bias, gradient accumulator
            self.w, self.w['bias'] = {feature:0.0 for feature in self.features}, 0.0
            self.Gt, self.Gt['bias'] = {feature:0.0 for feature in self.features}, 0.0
            cur_bias = 0.0
            cur_feature = 0.0
            #Iterate over the training data n times
            for j in range(iterations):
                #Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    #calc (w_t^Tx + bias)
                    s = sum([self.w[feature]*value for feature, value in xi.items()]) + self.w['bias']
                    #calc yi * (w_t^Tx + bias) as condition
                    condition = yi * s
                    #Update weights if there is a misclassification
                    if condition <= 1:
                        #update gt
                        cur_bias = -yi
                        #update Gt
                        self.Gt['bias'] += cur_bias ** 2
                        for feature, value in xi.items():
                            cur_feature = -yi * value
                            self.Gt[feature] += cur_feature ** 2
                        #update w
                        self.w['bias'] = self.w['bias'] - eta * cur_bias / np.power(self.Gt['bias'], 1/2)
                        for feature, value in xi.items():
                            cur_feature = -yi * value
                            self.w[feature] = self.w[feature] - eta * cur_feature / np.power(self.Gt[feature], 1/2)

        elif algorithm == 'Perceptron-Avg':
            #Initialize w, bias
            self.w, self.w['bias'] = {feature:0.0 for feature in self.features}, 0.0
            #count
            c = 1
            #H to store average weight
            self.H, self.H['bias'] = {feature:0.0 for feature in self.features}, 0.0
            #Iterate over the training data n times
            for j in range(iterations):
                #Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    #Update weights if there is a misclassification
                    if yi != y_hat:
                        for feature, value in xi.items():
                            #update w and H
                            self.w[feature] = self.w[feature] + eta*(yi*value)
                            self.H[feature] = self.H[feature] + eta*(yi*value) * c
                        self.w['bias'] = self.w['bias'] + eta*yi
                        self.H['bias'] = self.H['bias'] + eta*yi*c
                    c = c + 1
            self.w['bias'] = self.w['bias'] - self.H['bias'] / c
            for feature in self.features:
                self.w[feature] = self.w[feature] - self.H[feature] / c

        elif algorithm == 'Winnow-Avg':
            #Initialize w, bias
            self.w = {feature:1.0 for feature in self.features}
            self.w['bias'] = -len(self.w)
            #count
            c = 1
            #H to store average weight
            self.H = {feature:1.0 for feature in self.features}
            #Iterate over the training data n times
            for j in range(iterations):
                #Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    #Update weights if there is a misclassification
                    if yi != y_hat:
                        for feature, value in xi.items():
                            if yi == 1:
                                self.H[feature] = self.H[feature] + self.w[feature] * (alpha - 1) * c
                                self.w[feature] = self.w[feature] * alpha
                            else:
                                self.H[feature] = self.H[feature] + c * self.w[feature] * (1 - alpha) / alpha
                                self.w[feature] = self.w[feature] / alpha
                    c = c + 1
            for feature in self.features:
                self.w[feature] = self.w[feature] - self.H[feature] / c

        elif algorithm == 'Adagrad-Avg':
            #Initialize w, bias, gradient accumulator
            self.w, self.w['bias'] = {feature:0.0 for feature in self.features}, 0.0
            self.Gt, self.Gt['bias'] = {feature:0.0 for feature in self.features}, 0.0
            self.H, self.H['bias'] = {feature:0.0 for feature in self.features}, 0.0
            cur_bias = 0.0
            cur_feature = 0.0
            c = 0
            #Iterate over the training data n times
            for j in range(iterations):
                #Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    #calc (w_t^Tx + bias)
                    s = sum([self.w[feature]*value for feature, value in xi.items()]) + self.w['bias']
                    #calc yi * (w_t^Tx + bias) as condition
                    condition = yi * s
                    #Update weights if there is a misclassification
                    if condition <= 1:
                        #update gt
                        cur_bias = -yi
                        #update Gt
                        self.Gt['bias'] += cur_bias ** 2
                        for feature, value in xi.items():
                            cur_feature = -yi * value
                            self.Gt[feature] += cur_feature ** 2
                        #update w
                        self.w['bias'] = self.w['bias'] - eta * cur_bias / np.power(self.Gt['bias'], 1/2)
                        self.H['bias'] = self.H['bias'] - c * eta * cur_bias / np.power(self.Gt['bias'], 1/2)
                        for feature, value in xi.items():
                            cur_feature = -yi * value
                            self.w[feature] = self.w[feature] - eta * cur_feature / np.power(self.Gt[feature], 1/2)
                            self.H[feature] = self.H[feature] - c * eta * cur_feature / np.power(self.Gt[feature], 1/2)
                    c = c + 1
            self.w['bias'] = self.w['bias'] - self.H['bias'] / c
            for feature in self.features:
                self.w[feature] = self.w[feature] - self.H[feature] / c

        elif algorithm == 'SVM':
            self.clf  = svm.LinearSVC(loss = 'hinge', penalty = 'l2')
            self.vectorizer = sklearn.feature_extraction.DictVectorizer();
            x = self.vectorizer.fit_transform(x_train)
            self.clf.fit(x, y_train)

            
        else:
            print('Unknown algorithm')
                
    def predict(self, x):
        s = sum([self.w[feature]*value for feature, value in x.items()]) + self.w['bias']
        return 1 if s > 0 else -1
    
    def predict_SVM(self, x):
        x = self.vectorizer.transform([x])
        return self.clf.predict(x)[0]


# Parse the real-world data to generate features, 
#Returns a list of tuple lists
def parse_real_data(path):
    #List of tuples for each sentence
    data = []
    for filename in os.listdir(path):
        with open(path+filename, 'r') as file:
            sentence = []
            for line in file:
                if line == '\n':
                    data.append(sentence)
                    sentence = []
                else:
                    sentence.append(tuple(line.split()))
    return data

In[30]:


#Returns a list of labels
def parse_synthetic_labels(path):
    #List of tuples for each sentence
    labels = []
    with open(path+'y.txt', 'rb') as file:
        for line in file:
            labels.append(int(line.strip()))
    return labels


# In[31]:


#Returns a list of features
def parse_synthetic_data(path):
    #List of tuples for each sentence
    data = []
    with open(path+'x.txt') as file:
        features = []
        for line in file:
            #print('Line:', line)
            for ch in line:
                if ch == '[' or ch.isspace():
                    continue
                elif ch == ']':
                    data.append(features)
                    features = []
                else:
                    features.append(int(ch))
    return data


# In[115]:


#extract features and x_train and y_train from news_train_data
def extract_features_train(news_train_data):
    news_train_y = []
    news_train_x = []
    train_features = set([])
    for sentence in news_train_data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(3,len(padded)-3):
            news_train_y.append(1 if padded[i][1]=='I' else -1)
            feat1 = 'w-1='+str(padded[i-1][0])
            feat2 = 'w+1='+str(padded[i+1][0])
            feat3 = 'w-2='+str(padded[i-2][0])
            feat4 = 'w+2='+str(padded[i+2][0])            
            feat5 = 'w-3='+str(padded[i-3][0])
            feat6 = 'w+3='+str(padded[i+3][0])
            feat7 = 'w-1&w-2='+str(padded[i-1][0])+' '+str(padded[i-2][0])
            feat8 = 'w+1&w+2='+str(padded[i+1][0])+' '+str(padded[i+2][0])
            feat9 = 'w-1&w+1='+str(padded[i-1][0])+' '+str(padded[i+1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9]
            train_features.update(feats)
            feats = {feature:1 for feature in feats}
            news_train_x.append(feats)
    return train_features, news_train_x, news_train_y


# In[114]:


#extract x_dev and y_dev from news_dev_data and train_features
def extract_features_dev(news_dev_data, train_features):
    news_dev_y = []
    news_dev_x = []
    for sentence in news_dev_data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(3,len(padded)-3):
            news_dev_y.append(1 if padded[i][1]=='I' else -1)
            feat1 = 'w-1='+str(padded[i-1][0])
            feat2 = 'w+1='+str(padded[i+1][0])
            feat3 = 'w-2='+str(padded[i-2][0])
            feat4 = 'w+2='+str(padded[i+2][0])            
            feat5 = 'w-3='+str(padded[i-3][0])
            feat6 = 'w+3='+str(padded[i+3][0])
            feat7 = 'w-1&w-2='+str(padded[i-1][0])+' '+str(padded[i-2][0])
            feat8 = 'w+1&w+2='+str(padded[i+1][0])+' '+str(padded[i+2][0])
            feat9 = 'w-1&w+1='+str(padded[i-1][0])+' '+str(padded[i+1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9]
            feats = {feature:1 for feature in feats if feature in train_features}
            news_dev_x.append(feats)
    return news_dev_x, news_dev_y


# In[113]:


#extract x_test and y_test from news_test_data, y_test is useless, so I set is as all 0
def extract_features_test(news_test_data, train_features):
    news_test_y = []
    news_test_x = []
    for sentence in news_test_data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(3,len(padded)-3):
            news_test_y.append(0)
            feat1 = 'w-1='+str(padded[i-1][0])
            feat2 = 'w+1='+str(padded[i+1][0])
            feat3 = 'w-2='+str(padded[i-2][0])
            feat4 = 'w+2='+str(padded[i+2][0])            
            feat5 = 'w-3='+str(padded[i-3][0])
            feat6 = 'w+3='+str(padded[i+3][0])
            feat7 = 'w-1&w-2='+str(padded[i-1][0])+' '+str(padded[i-2][0])
            feat8 = 'w+1&w+2='+str(padded[i+1][0])+' '+str(padded[i+2][0])
            feat9 = 'w-1&w+1='+str(padded[i-1][0])+' '+str(padded[i+1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9]
            feats = {feature:1 for feature in feats if feature in train_features}
            news_test_x.append(feats)
    return news_test_x, news_test_y


# # In[34]:


# print('Loading data...')
# #Load data from folders.
# #Real world data - lists of tuple lists
# news_train_data = parse_real_data('Data/Real-World/CoNLL/Train/')
# news_dev_data = parse_real_data('Data/Real-World/CoNLL/Dev/')
# news_test_data = parse_real_data('Data/Real-World/CoNLL/Test/')
# email_dev_data = parse_real_data('Data/Real-World/Enron/Dev/')
# email_test_data = parse_real_data('Data/Real-World/Enron/Test/')

# #Load dense synthetic data
# syn_dense_train_data = parse_synthetic_data('Data/Synthetic/Dense/Train/')
# syn_dense_train_labels = parse_synthetic_labels('Data/Synthetic/Dense/Train/')
# syn_dense_dev_data = parse_synthetic_data('Data/Synthetic/Dense/Dev/')
# syn_dense_dev_labels = parse_synthetic_labels('Data/Synthetic/Dense/Dev/')
# syn_dense_dev_no_noise_data = parse_synthetic_data('Data/Synthetic/Dense/Dev_no_noise/')
# syn_dense_dev_no_noise_labels = parse_synthetic_labels('Data/Synthetic/Dense/Dev_no_noise/')
# syn_dense_test_data = parse_synthetic_data('Data/Synthetic/Dense/Test/')
   
# #Load sparse synthetic data
# syn_sparse_train_data = parse_synthetic_data('Data/Synthetic/Sparse/Train/')
# syn_sparse_train_labels = parse_synthetic_labels('Data/Synthetic/Sparse/Train/')
# syn_sparse_dev_data = parse_synthetic_data('Data/Synthetic/Sparse/Dev/')
# syn_sparse_dev_labels = parse_synthetic_labels('Data/Synthetic/Sparse/Dev/')
# syn_sparse_test_data = parse_synthetic_data('Data/Synthetic/Sparse/Test/')

# print('Data Loaded.')


# # In[35]:


# # Convert to sparse dictionary representations.

# print('Converting Synthetic data...')
# syn_dense_train = zip(*[({'x'+str(i): syn_dense_train_data[j][i]
#     for i in range(len(syn_dense_train_data[j])) if syn_dense_train_data[j][i] == 1}, syn_dense_train_labels[j]) 
#         for j in range(len(syn_dense_train_data))])
# syn_dense_train_x, syn_dense_train_y = syn_dense_train
# syn_dense_dev = zip(*[({'x'+str(i): syn_dense_dev_data[j][i]
#     for i in range(len(syn_dense_dev_data[j])) if syn_dense_dev_data[j][i] == 1}, syn_dense_dev_labels[j]) 
#         for j in range(len(syn_dense_dev_data))])
# syn_dense_dev_x, syn_dense_dev_y = syn_dense_dev

# ## Similarly add code for the dev set with no noise and sparse data

# print('Done')


# # In[36]:


# # Feature extraction
# # Remember to add the other features mentioned in the handout
# print('Extracting features from real-world data...')
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# news_dev_x, news_dev_y = extract_features_dev(news_dev_data, train_features)
# news_test_x, news_test_y = extract_features_test(news_test_data, train_features)


# # In[125]:


# #accuracy of perceptron on dense dev
# print('\nPerceptron Accuracy')
# # Test Perceptron on Dense Synthetic
# p = Classifier('Perceptron',syn_dense_train_x , syn_dense_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)


# # In[38]:


# #accuracy of winnow on dense dev
# print('\nWinnow Accuracy')
# # Test Winnow on Dense Synthetic
# p = Classifier('Winnow', syn_dense_train_x, syn_dense_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)


# # In[39]:


# #accuracy of avergae perceptron on dense dev
# print('\nPerceptron-Avg Accuracy')
# # Test Perceptron-Avg on Dense Synthetic
# p = Classifier('Perceptron-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)


# # In[40]:


# #accuracy of avergae winnow on dense dev
# print('\nWinnow-Avg Accuracy')
# # Test Winnow-Avg on Dense Synthetic
# p = Classifier('Winnow-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)


# # In[128]:


# #accuracy of svm on dense dev
# print('\nSVM Accuracy')
# # Test Winnow-Avg on Dense Synthetic
# p = Classifier('SVM', syn_dense_train_x, syn_dense_train_y)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict_SVM(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)


# # In[41]:


# #accuracy of adagrad on dense dev
# print('\nAdagrad Accuracy')
# # Test Adagrad on Dense Synthetic
# p = Classifier('Adagrad', syn_dense_train_x, syn_dense_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)


# # In[42]:


# #accuracy of avergae adagrad on dense dev
# print('\nAdagrad_Avg Accuracy')
# # Test Adagrad on Dense Synthetic
# p = Classifier('Adagrad-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)


# # In[43]:


# #loading sparse data
# print('Converting Synthetic data...')
# syn_sparse_train = zip(*[({'x'+str(i): syn_sparse_train_data[j][i]
#     for i in range(len(syn_sparse_train_data[j])) if syn_sparse_train_data[j][i] == 1}, syn_sparse_train_labels[j]) 
#         for j in range(len(syn_sparse_train_data))])

# syn_sparse_train_x, syn_sparse_train_y = syn_sparse_train

# syn_sparse_dev = zip(*[({'x'+str(i): syn_sparse_dev_data[j][i]
#     for i in range(len(syn_sparse_dev_data[j])) if syn_sparse_dev_data[j][i] == 1}, syn_sparse_dev_labels[j]) 
#         for j in range(len(syn_sparse_dev_data))])

# syn_sparse_dev_x, syn_sparse_dev_y = syn_sparse_dev


# # In[44]:


# #print sparse accuracy on dev without parameter tunning
# print('\nPerceptron Accuracy')
# # Test Perceptron on Sparse Synthetic
# p = Classifier('Perceptron', syn_sparse_train_x, syn_sparse_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)

# print('\nWinnow Accuracy')
# # Test Winnow on Sparse Synthetic
# p = Classifier('Winnow', syn_sparse_train_x, syn_sparse_train_y, iterations=1)
# accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)

# print('\nAdagrad Accuracy')
# # Test Adagrad on Sparse Synthetic
# p = Classifier('Adagrad', syn_sparse_train_x, syn_sparse_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)

# print('\nPerceptron-Avg Accuracy')
# # Test Perceptron-Avg on Sparse Synthetic
# p = Classifier('Perceptron-Avg', syn_sparse_train_x, syn_sparse_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)

# print('\nWinnow-Avg Accuracy')
# # Test Winnow-Avg on Sparse Synthetic
# p = Classifier('Winnow-Avg', syn_sparse_train_x, syn_sparse_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)

# print('\nAdagrad_Avg Accuracy')
# # Test Adagrad on Sparse Synthetic
# p = Classifier('Adagrad-Avg', syn_sparse_train_x, syn_sparse_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)


# # In[45]:


# #tune parameters on dense data
# print('\nWinnow Accuracy')
# # tune parameters on winnow of dense data
# alphas = [1.1, 1.01, 1.005, 1.0005, 1.0001]
# accuracys = []
# for alpha in alphas:
#     p = Classifier('Winnow', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha = alpha)
#     accuracys.append(sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100)
# plt.figure()
# plt.plot(alphas, accuracys)
# plt.show()



# print('\nAdagrad Accuracy')
# # tune parameters on adagrad
# etas = [1.5, 0.25, 0.03, 0.005, 0.001]
# accuracys = []
# for eta in etas:
#     p = Classifier('Adagrad', syn_dense_train_x, syn_dense_train_y, iterations=10, eta = eta)
#     accuracys.append(sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100)
# plt.figure()
# plt.plot(etas, accuracys)
# plt.show()


# # In[46]:





# # In[47]:


# #tune parameters on sparse data
# print('\nWinnow Accuracy')
# # tune parameters on winnow of sparse data
# alphas = [1.1, 1.01, 1.005, 1.0005, 1.0001]
# accuracys = []
# for alpha in alphas:
#     p = Classifier('Winnow', syn_sparse_train_x, syn_sparse_train_y, iterations=10, alpha = alpha)
#     accuracys.append(sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100)
# plt.figure()
# plt.plot(alphas, accuracys)
# plt.show()



# print('\nAdagrad Accuracy')
# # tune parameters on adagrad
# etas = [1.5, 0.25, 0.03, 0.005, 0.001]
# accuracys = []
# for eta in etas:
#     p = Classifier('Adagrad', syn_sparse_train_x, syn_sparse_train_y, iterations=10, eta = eta)
#     accuracys.append(sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100)
# plt.figure()
# plt.plot(etas, accuracys)
# plt.show()


# # In[185]:


# #learning rate sparse data
# breakpoints = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 50000]
# clf_Per = []
# clf_Win = []
# clf_Ada = []
# clf_Per_ave = []
# clf_Win_ave = []
# clf_Ada_ave = []
# clf_SVM = []
# for breakpoint in breakpoints:
#     x_train = syn_sparse_train_x[:breakpoint - 1]
#     y_train = syn_sparse_train_y[:breakpoint - 1]
#     p = Classifier('Perceptron', x_train, y_train, iterations=10)
#     clf_Per.append(sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100)


#     p = Classifier('Winnow', x_train, y_train, iterations=10, alpha = 1.005)
#     clf_Win.append(sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100)


#     p = Classifier('Adagrad', x_train, y_train, iterations=10, eta = 1.5)
#     clf_Ada.append(sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100)


#     p = Classifier('Perceptron-Avg', x_train, y_train, iterations=10)
#     clf_Per_ave.append(sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100)


#     p = Classifier('Winnow-Avg', x_train, y_train, iterations=10, alpha = 1.005)
#     clf_Win_ave.append(sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100)


#     p = Classifier('Adagrad-Avg', x_train, y_train, iterations=10, eta = 1.5)
#     clf_Ada_ave.append(sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100)
    
#     p = Classifier('SVM', x_train, y_train)
#     clf_SVM.append(sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict_SVM(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100)
    
# plt.figure()
# f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w')   
# ax.set_xlim(0,5000)
# ax2.set_xlim(49000,51000)
# ax.spines['right'].set_visible(False)
# ax2.spines['left'].set_visible(False)
# ax.yaxis.tick_left()
# ax.tick_params(labelright='off')
# ax2.yaxis.tick_right()
# ax.plot(breakpoints, clf_Per, label = 'clf_Per')
# ax2.plot(breakpoints, clf_Per, label = 'clf_Per')

# ax.plot(breakpoints, clf_Win, label = 'clf_Win')
# ax2.plot(breakpoints, clf_Win, label = 'clf_Win')

# ax.plot(breakpoints, clf_Ada, label = 'clf_Ada')
# ax2.plot(breakpoints, clf_Ada, label = 'clf_Ada')

# ax.plot(breakpoints, clf_Per_ave, label = 'clf_Per_ave')
# ax2.plot(breakpoints, clf_Per_ave, label = 'clf_Per_ave')

# ax.plot(breakpoints, clf_Win_ave, label = 'clf_Win_ave')
# ax2.plot(breakpoints, clf_Win_ave, label = 'clf_Win_ave')

# ax.plot(breakpoints, clf_Ada_ave, label = 'clf_Ada_ave')
# ax2.plot(breakpoints, clf_Ada_ave, label = 'clf_Ada_ave')

# ax.plot(breakpoints, clf_SVM, label = 'clf_SVM')
# ax2.plot(breakpoints, clf_SVM, label = 'clf_SVM')

# plt.legend(loc='upper right')


# # In[184]:


# #learning rate of dense data
# breakpoints = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 50000]
# clf_Per = []
# clf_Win = []
# clf_Ada = []
# clf_Per_ave = []
# clf_Win_ave = []
# clf_Ada_ave = []
# clf_SVM = []
# for breakpoint in breakpoints:
#     x_train = syn_dense_train_x[:breakpoint - 1]
#     y_train = syn_dense_train_y[:breakpoint - 1]
#     p = Classifier('Perceptron', x_train, y_train, iterations=10)
#     clf_Per.append(sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100)


#     p = Classifier('Winnow', x_train, y_train, iterations=10, alpha = 1.0005)
#     clf_Win.append(sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100)


#     p = Classifier('Adagrad', x_train, y_train, iterations=10, eta = 1.5)
#     clf_Ada.append(sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100)


#     p = Classifier('Perceptron-Avg', x_train, y_train, iterations=10)
#     clf_Per_ave.append(sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100)


#     p = Classifier('Winnow-Avg', x_train, y_train, iterations=10, alpha = 1.0005)
#     clf_Win_ave.append(sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100)


#     p = Classifier('Adagrad-Avg', x_train, y_train, iterations=10, eta = 1.5)
#     clf_Ada_ave.append(sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100)
    
#     p = Classifier('SVM', x_train, y_train)
#     clf_SVM.append(sum([1 for i in range(len(syn_dense_dev_y)) if p.predict_SVM(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100)
    
# plt.figure()
# f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w')   
# ax.set_xlim(0,5000)
# ax2.set_xlim(49000,51000)
# ax.spines['right'].set_visible(False)
# ax2.spines['left'].set_visible(False)
# ax.yaxis.tick_left()
# ax.tick_params(labelright='off')
# ax2.yaxis.tick_right()
# ax.plot(breakpoints, clf_Per, label = 'clf_Per')
# ax2.plot(breakpoints, clf_Per, label = 'clf_Per')

# ax.plot(breakpoints, clf_Win, label = 'clf_Win')
# ax2.plot(breakpoints, clf_Win, label = 'clf_Win')

# ax.plot(breakpoints, clf_Ada, label = 'clf_Ada')
# ax2.plot(breakpoints, clf_Ada, label = 'clf_Ada')

# ax.plot(breakpoints, clf_Per_ave, label = 'clf_Per_ave')
# ax2.plot(breakpoints, clf_Per_ave, label = 'clf_Per_ave')

# ax.plot(breakpoints, clf_Win_ave, label = 'clf_Win_ave')
# ax2.plot(breakpoints, clf_Win_ave, label = 'clf_Win_ave')

# ax.plot(breakpoints, clf_Ada_ave, label = 'clf_Ada_ave')
# ax2.plot(breakpoints, clf_Ada_ave, label = 'clf_Ada_ave')

# ax.plot(breakpoints, clf_SVM, label = 'clf_SVM')
# ax2.plot(breakpoints, clf_SVM, label = 'clf_SVM')

# plt.legend(loc='upper right')


# # In[186]:


# print('Loading data...')
# #Load data from folders.
# #Real world data - lists of tuple lists
# news_train_data = parse_real_data('Data/Real-World/CoNLL/Train/')
# news_dev_data = parse_real_data('Data/Real-World/CoNLL/Dev/')
# news_test_data = parse_real_data('Data/Real-World/CoNLL/Test/')
# email_dev_data = parse_real_data('Data/Real-World/Enron/Dev/')
# email_test_data = parse_real_data('Data/Real-World/Enron/Test/')

# print('Data Loaded.')


# # In[74]:


# #test dense test and write file using perceptron-avg
# syn_dense_test_labels = np.zeros(len(syn_dense_test_data))
# syn_dense_test = zip(*[({'x'+str(i): syn_dense_test_data[j][i]
#     for i in range(len(syn_dense_test_data[j])) if syn_dense_test_data[j][i] == 1}, syn_dense_test_labels[j]) 
#         for j in range(len(syn_dense_test_data))])
# syn_dense_test_x, syn_dense_test_y = syn_dense_test

# print('\nPerceptron-Avg Accuracy')
# # Test Perceptron-Avg on Dense Synthetic
# p = Classifier('Perceptron-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10)
# syn_dense_test_y = [p.predict(syn_dense_test_x[i]) for i in range(len(syn_dense_test_x))]
# f = open('p-dense.txt','a')

# for i in range(len(syn_dense_test_y)):
#     f.write(str(syn_dense_test_y[i]) + '\n');

# f.close()


# # In[75]:


# #test sparse test and write file using perceptron-avg
# syn_sparse_test_labels = np.zeros(len(syn_sparse_test_data))
# syn_sparse_test = zip(*[({'x'+str(i): syn_sparse_test_data[j][i]
#     for i in range(len(syn_sparse_test_data[j])) if syn_sparse_test_data[j][i] == 1}, syn_sparse_test_labels[j]) 
#         for j in range(len(syn_sparse_test_data))])
# syn_sparse_test_x, syn_sparse_test_y = syn_sparse_test

# print('\nPerceptron-Avg Accuracy')
# # Test Perceptron-Avg on Dense Synthetic
# p = Classifier('Perceptron-Avg', syn_sparse_train_x, syn_sparse_train_y, iterations=10)
# syn_sparse_test_y = [p.predict(syn_sparse_test_x[i]) for i in range(len(syn_sparse_test_x))]
# f = open('p-sparse.txt','a')

# for i in range(len(syn_sparse_test_y)):
#     f.write(str(syn_sparse_test_y[i]) + '\n');

# f.close()


# # In[76]:


# #test dense test and write file using svm
# syn_dense_test_labels = np.zeros(len(syn_dense_test_data))
# syn_dense_test = zip(*[({'x'+str(i): syn_dense_test_data[j][i]
#     for i in range(len(syn_dense_test_data[j])) if syn_dense_test_data[j][i] == 1}, syn_dense_test_labels[j]) 
#         for j in range(len(syn_dense_test_data))])
# syn_dense_test_x, syn_dense_test_y = syn_dense_test

# print('\nPerceptron-Avg Accuracy')
# # Test Perceptron-Avg on Dense Synthetic
# p = Classifier('SVM', syn_dense_train_x, syn_dense_train_y, iterations=10)
# syn_dense_test_y = [p.predict_SVM(syn_dense_test_x[i]) for i in range(len(syn_dense_test_x))]
# f = open('svm-dense.txt','a')

# for i in range(len(syn_dense_test_y)):
#     f.write(str(syn_dense_test_y[i]) + '\n');

# f.close()


# # In[77]:


# #test sparse test and write file using svm
# syn_sparse_test_labels = np.zeros(len(syn_sparse_test_data))
# syn_sparse_test = zip(*[({'x'+str(i): syn_sparse_test_data[j][i]
#     for i in range(len(syn_sparse_test_data[j])) if syn_sparse_test_data[j][i] == 1}, syn_sparse_test_labels[j]) 
#         for j in range(len(syn_sparse_test_data))])
# syn_sparse_test_x, syn_sparse_test_y = syn_sparse_test

# print('\nPerceptron-Avg Accuracy')
# # Test Perceptron-Avg on Dense Synthetic
# p = Classifier('SVM', syn_sparse_train_x, syn_sparse_train_y, iterations=10)
# syn_sparse_test_y = [p.predict_SVM(syn_sparse_test_x[i]) for i in range(len(syn_sparse_test_x))]
# f = open('svm-sparse.txt','a')

# for i in range(len(syn_sparse_test_y)):
#     f.write(str(syn_sparse_test_y[i]) + '\n');

# f.close()


# # In[142]:


# # f1 on news_dev using PA
# print('Loading data...')
# #Load data from folders.
# #Real world data - lists of tuple lists
# news_train_data = parse_real_data('Data/Real-World/CoNLL/Train/')
# news_dev_data = parse_real_data('Data/Real-World/CoNLL/Dev/')
# news_test_data = parse_real_data('Data/Real-World/CoNLL/Test/')
# email_dev_data = parse_real_data('Data/Real-World/Enron/Dev/')
# email_test_data = parse_real_data('Data/Real-World/Enron/Test/')
# # Feature extraction
# # Remember to add the other features mentioned in the handout
# print('Extracting features from real-world data...')
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# news_dev_x, news_dev_y = extract_features_dev(news_dev_data, train_features)
# news_test_x, news_test_y = extract_features_test(news_test_data, train_features)
# print(len(news_train_x))
# print(len(news_train_x[0]))
# print(news_train_x[0:5])

# # news_train_data = zip(*[({'x'+str(i): news_train_x[j][i]
# #     for i in range(len(news_train_x[j])) if news_train_x[j][i] == 1}, news_train_y[j]) 
# #         for j in range(len(news_train_x))])
# # news_train_x, news_train_y = news_train_data

# p = Classifier('Perceptron-Avg', news_train_x, news_train_y, iterations=10)
# news_dev_predict_y = []
# for i in range(len(news_dev_x)):
#     news_dev_predict_y.append(p.predict(news_dev_x[i]))
# precision, recall, f1 = calc_f1(news_dev_y, news_dev_predict_y)
# print(precision)
# print(recall)
# print(f1)

# # f = open('p-conll.txt','a')
# # for i in range(len(syn_dense_test_y)):
# #     f.write(str(syn_dense_test_y[i]) + '\n');
# # f.close()


# # In[143]:


# # f1 on news_dev using svm
# print('Loading data...')
# #Load data from folders.
# #Real world data - lists of tuple lists
# news_train_data = parse_real_data('Data/Real-World/CoNLL/Train/')
# news_dev_data = parse_real_data('Data/Real-World/CoNLL/Dev/')
# news_test_data = parse_real_data('Data/Real-World/CoNLL/Test/')
# email_dev_data = parse_real_data('Data/Real-World/Enron/Dev/')
# email_test_data = parse_real_data('Data/Real-World/Enron/Test/')
# # Feature extraction
# # Remember to add the other features mentioned in the handout
# print('Extracting features from real-world data...')
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# news_dev_x, news_dev_y = extract_features_dev(news_dev_data, train_features)
# news_test_x, news_test_y = extract_features_test(news_test_data, train_features)
# print(len(news_train_x))
# print(len(news_train_x[0]))
# print(news_train_x[0:5])

# # news_train_data = zip(*[({'x'+str(i): news_train_x[j][i]
# #     for i in range(len(news_train_x[j])) if news_train_x[j][i] == 1}, news_train_y[j]) 
# #         for j in range(len(news_train_x))])
# # news_train_x, news_train_y = news_train_data

# p = Classifier('SVM', news_train_x, news_train_y, iterations=10)
# news_dev_predict_y = []
# for i in range(len(news_dev_x)):
#     news_dev_predict_y.append(p.predict_SVM(news_dev_x[i]))
# precision, recall, f1 = calc_f1(news_dev_y, news_dev_predict_y)
# print(precision)
# print(recall)
# print(f1)

# # f = open('p-conll.txt','a')
# # for i in range(len(syn_dense_test_y)):
# #     f.write(str(syn_dense_test_y[i]) + '\n');
# # f.close()


# # In[167]:


# #write test file p-conll.txt
# print('Loading data...')
# #Load data from folders.
# #Real world data - lists of tuple lists
# news_train_data = parse_real_data('Data/Real-World/CoNLL/Train/')
# news_dev_data = parse_real_data('Data/Real-World/CoNLL/Dev/')
# news_test_data = parse_real_data('Data/Real-World/CoNLL/Test/')
# email_dev_data = parse_real_data('Data/Real-World/Enron/Dev/')
# email_test_data = parse_real_data('Data/Real-World/Enron/Test/')
# # Feature extraction
# # Remember to add the other features mentioned in the handout
# print('Extracting features from real-world data...')
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# news_dev_x, news_dev_y = extract_features_dev(news_dev_data, train_features)
# news_test_x, news_test_y = extract_features_test(news_test_data, train_features)
# print(len(news_train_x))
# print(len(news_train_x[0]))
# print(news_train_x[0:5])

# # news_train_data = zip(*[({'x'+str(i): news_train_x[j][i]
# #     for i in range(len(news_train_x[j])) if news_train_x[j][i] == 1}, news_train_y[j]) 
# #         for j in range(len(news_train_x))])
# # news_train_x, news_train_y = news_train_data

# p = Classifier('Perceptron-Avg', news_train_x, news_train_y, iterations=10)
# news_test_y = []
# for i in range(len(news_test_x)):
#     news_test_y.append(p.predict(news_test_x[i]))

# f = open('p-conll.txt','w')
# for i in range(len(news_test_y)):
#     if news_test_y[i] == 1:
#         f.write('I' + '\n')
#     else:
#         f.write('O' + '\n')
# f.close()


# # In[166]:


# #write test file svm-conll.txt
# print('Loading data...')
# #Load data from folders.
# #Real world data - lists of tuple lists
# news_train_data = parse_real_data('Data/Real-World/CoNLL/Train/')
# news_dev_data = parse_real_data('Data/Real-World/CoNLL/Dev/')
# news_test_data = parse_real_data('Data/Real-World/CoNLL/Test/')
# email_dev_data = parse_real_data('Data/Real-World/Enron/Dev/')
# email_test_data = parse_real_data('Data/Real-World/Enron/Test/')
# # Feature extraction
# # Remember to add the other features mentioned in the handout
# print('Extracting features from real-world data...')
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# news_dev_x, news_dev_y = extract_features_dev(news_dev_data, train_features)
# news_test_x, news_test_y = extract_features_test(news_test_data, train_features)
# print(len(news_train_x))
# print(len(news_train_x[0]))
# print(news_train_x[0:5])

# # news_train_data = zip(*[({'x'+str(i): news_train_x[j][i]
# #     for i in range(len(news_train_x[j])) if news_train_x[j][i] == 1}, news_train_y[j]) 
# #         for j in range(len(news_train_x))])
# # news_train_x, news_train_y = news_train_data

# p = Classifier('SVM', news_train_x, news_train_y, iterations=10)
# news_test_y = []
# for i in range(len(news_test_x)):
#     news_test_y.append(p.predict_SVM(news_test_x[i]))

# f = open('svm-conll.txt','w')
# print(news_test_y[:5])
# for i in range(len(news_test_y)):
#     if news_test_y[i] == 1:
#         f.write('I' + '\n')
#     else:
#         f.write('O' + '\n')
# f.close()


# # In[165]:


# #write test file svm-enron.txt
# print('Loading data...')
# #Load data from folders.
# #Real world data - lists of tuple lists
# news_train_data = parse_real_data('Data/Real-World/CoNLL/Train/')
# email_dev_data = parse_real_data('Data/Real-World/Enron/Dev/')
# email_test_data = parse_real_data('Data/Real-World/Enron/Test/')
# # Feature extraction
# # Remember to add the other features mentioned in the handout
# print('Extracting features from real-world data...')
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# email_dev_x, email_dev_y = extract_features_dev(email_dev_data, train_features)
# email_test_x, email_test_y = extract_features_test(email_test_data, train_features)

# # news_train_data = zip(*[({'x'+str(i): news_train_x[j][i]
# #     for i in range(len(news_train_x[j])) if news_train_x[j][i] == 1}, news_train_y[j]) 
# #         for j in range(len(news_train_x))])
# # news_train_x, news_train_y = news_train_data

# p = Classifier('SVM', news_train_x, news_train_y, iterations=10)
# email_dev_predict_y = []
# for i in range(len(email_dev_x)):
#     email_dev_predict_y.append(p.predict_SVM(email_dev_x[i]))
# precision, recall, f1 = calc_f1(email_dev_y, email_dev_predict_y)
# print(precision)
# print(recall)
# print(f1)
# email_test_y = []
# for i in range(len(email_test_x)):
#     email_test_y.append(p.predict_SVM(email_test_x[i]))

# f = open('svm-enron.txt','w')
# for i in range(len(email_test_y)):
#     if email_test_y[i] == 1:
#         f.write('I' + '\n')
#     else:
#         f.write('O' + '\n')
# f.close()


# # In[164]:


# #write test file p-enron.txt
# print('Loading data...')
# #Load data from folders.
# #Real world data - lists of tuple lists
# news_train_data = parse_real_data('Data/Real-World/CoNLL/Train/')
# email_dev_data = parse_real_data('Data/Real-World/Enron/Dev/')
# email_test_data = parse_real_data('Data/Real-World/Enron/Test/')
# # Feature extraction
# # Remember to add the other features mentioned in the handout
# print('Extracting features from real-world data...')
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# email_dev_x, email_dev_y = extract_features_dev(email_dev_data, train_features)
# email_test_x, email_test_y = extract_features_test(email_test_data, train_features)

# # news_train_data = zip(*[({'x'+str(i): news_train_x[j][i]
# #     for i in range(len(news_train_x[j])) if news_train_x[j][i] == 1}, news_train_y[j]) 
# #         for j in range(len(news_train_x))])
# # news_train_x, news_train_y = news_train_data

# p = Classifier('Perceptron-Avg', news_train_x, news_train_y, iterations=10)
# email_dev_predict_y = []
# for i in range(len(email_dev_x)):
#     email_dev_predict_y.append(p.predict(email_dev_x[i]))
# precision, recall, f1 = calc_f1(email_dev_y, email_dev_predict_y)
# email_test_y = []
# for i in range(len(email_test_x)):
#     email_test_y.append(p.predict(email_test_x[i]))
# print(precision)
# print(recall)
# print(f1)
# f = open('p-enron.txt','w')
# for i in range(len(email_test_y)):
#     if email_test_y[i] == 1:
#         f.write('I' + '\n')
#     else:
#         f.write('O' + '\n')
# f.close()


# # In[179]:


# # test on training set with parameter selected in dense set
# print('\nPerceptron Accuracy')
# # Test Perceptron on Sparse Synthetic
# p = Classifier('Perceptron', syn_dense_train_x, syn_dense_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)

# print('\nWinnow Accuracy')
# # Test Winnow on Sparse Synthetic
# p = Classifier('Winnow', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha = 1.0005)
# accuracy = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)

# print('\nAdagrad Accuracy')
# # Test Adagrad on Sparse Synthetic
# p = Classifier('Adagrad', syn_dense_train_x, syn_dense_train_y, iterations=10, eta = 1.5)
# accuracy = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)

# print('\nPerceptron-Avg Accuracy')
# # Test Perceptron-Avg on Sparse Synthetic
# p = Classifier('Perceptron-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)

# print('\nWinnow-Avg Accuracy')
# # Test Winnow-Avg on Sparse Synthetic
# p = Classifier('Winnow-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha = 1.0005)
# accuracy = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)

# print('\nAdagrad_Avg Accuracy')
# # Test Adagrad on Sparse Synthetic
# p = Classifier('Adagrad-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10, eta = 1.5)
# accuracy = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)

# print('\nSVM Accuracy')
# # Test Winnow-Avg on Dense Synthetic
# p = Classifier('SVM', syn_dense_train_x, syn_dense_train_y)
# accuracy = sum([1 for i in range(len(syn_dense_train_y)) if p.predict_SVM(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)


# # In[172]:


# # test on training set with parameter selected in sparse set
# print('\nSVM Accuracy')
# # Test svm on sparse Synthetic
# p = Classifier('SVM', syn_sparse_train_x, syn_sparse_train_y)
# accuracy = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict_SVM(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# print('Syn sparse train Accuracy:', accuracy)


# # In[173]:


# # test on dev set with parameter selected in sparse set
# print('\nSVM Accuracy')
# # Test Winnow-Avg on Dense Synthetic
# p = Classifier('SVM', syn_sparse_train_x, syn_sparse_train_y)
# accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict_SVM(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
# print('Syn sparse Dev Accuracy:', accuracy)


# # In[180]:


# #expriement without noise
# syn_dense_dev_no_noise_data = parse_synthetic_data('Data/Synthetic/Dense/Dev_no_noise/')
# syn_dense_dev_no_noise_labels = parse_synthetic_labels('Data/Synthetic/Dense/Dev_no_noise/')

# syn_dense_dev = zip(*[({'x'+str(i): syn_dense_dev_no_noise_data[j][i]
#     for i in range(len(syn_dense_dev_no_noise_data[j])) if syn_dense_dev_no_noise_data[j][i] == 1}, syn_dense_dev_no_noise_labels[j]) 
#         for j in range(len(syn_dense_dev_no_noise_data))])
# syn_dense_dev_x, syn_dense_dev_y = syn_dense_dev

# print('\nPerceptron Accuracy')
# # Test Perceptron on Dense Synthetic
# p = Classifier('Perceptron', syn_dense_train_x, syn_dense_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)

# print('\nWinnow Accuracy')
# # Test Winnow on Dense Synthetic
# p = Classifier('Winnow', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha = 1.0005)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)

# print('\nAdagrad Accuracy')
# # Test Adagrad on Dense Synthetic
# p = Classifier('Adagrad', syn_dense_train_x, syn_dense_train_y, iterations=10, eta = 1.5)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)

# print('\nPerceptron-Avg Accuracy')
# # Test Perceptron-Avg on Dense Synthetic
# p = Classifier('Perceptron-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)

# print('\nWinnow-Avg Accuracy')
# # Test Winnow-Avg on Dense Synthetic
# p = Classifier('Winnow-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha = 1.0005)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)

# print('\nAdagrad_Avg Accuracy')
# # Test Adagrad on Dense Synthetic
# p = Classifier('Adagrad-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10, eta = 1.5)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)

# print('\nSVM Accuracy')
# # Test Winnow-Avg on Dense Synthetic
# p = Classifier('SVM', syn_dense_train_x, syn_dense_train_y)
# accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict_SVM(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
# print('Syn dense Dev Accuracy:', accuracy)


# # In[177]:


# #PA test on training set
# print('Loading data...')
# #Load data from folders.
# #Real world data - lists of tuple lists
# news_train_data = parse_real_data('Data/Real-World/CoNLL/Train/')
# news_dev_data = parse_real_data('Data/Real-World/CoNLL/Dev/')
# news_test_data = parse_real_data('Data/Real-World/CoNLL/Test/')
# email_dev_data = parse_real_data('Data/Real-World/Enron/Dev/')
# email_test_data = parse_real_data('Data/Real-World/Enron/Test/')
# # Feature extraction
# # Remember to add the other features mentioned in the handout
# print('Extracting features from real-world data...')
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# news_dev_x, news_dev_y = extract_features_dev(news_dev_data, train_features)
# news_test_x, news_test_y = extract_features_test(news_test_data, train_features)
# print(len(news_train_x))
# print(len(news_train_x[0]))
# print(news_train_x[0:5])

# # news_train_data = zip(*[({'x'+str(i): news_train_x[j][i]
# #     for i in range(len(news_train_x[j])) if news_train_x[j][i] == 1}, news_train_y[j]) 
# #         for j in range(len(news_train_x))])
# # news_train_x, news_train_y = news_train_data

# p = Classifier('Perceptron-Avg', news_train_x, news_train_y, iterations=10)
# news_predict_y = []
# for i in range(len(news_train_x)):
#     news_predict_y.append(p.predict(news_train_x[i]))
# precision, recall, f1 = calc_f1(news_train_y, news_predict_y)
# print(precision)
# print(recall)
# print(f1)

# # f = open('p-conll.txt','a')
# # for i in range(len(syn_dense_test_y)):
# #     f.write(str(syn_dense_test_y[i]) + '\n');
# # f.close()


# # In[178]:


# #test svm on training set in real data
# print('Loading data...')
# #Load data from folders.
# #Real world data - lists of tuple lists
# news_train_data = parse_real_data('Data/Real-World/CoNLL/Train/')
# news_dev_data = parse_real_data('Data/Real-World/CoNLL/Dev/')
# news_test_data = parse_real_data('Data/Real-World/CoNLL/Test/')
# email_dev_data = parse_real_data('Data/Real-World/Enron/Dev/')
# email_test_data = parse_real_data('Data/Real-World/Enron/Test/')
# # Feature extraction
# # Remember to add the other features mentioned in the handout
# print('Extracting features from real-world data...')
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# news_dev_x, news_dev_y = extract_features_dev(news_dev_data, train_features)
# news_test_x, news_test_y = extract_features_test(news_test_data, train_features)
# print(len(news_train_x))
# print(len(news_train_x[0]))
# print(news_train_x[0:5])

# # news_train_data = zip(*[({'x'+str(i): news_train_x[j][i]
# #     for i in range(len(news_train_x[j])) if news_train_x[j][i] == 1}, news_train_y[j]) 
# #         for j in range(len(news_train_x))])
# # news_train_x, news_train_y = news_train_data

# p = Classifier('SVM', news_train_x, news_train_y, iterations=10)
# news_predict_y = []
# for i in range(len(news_train_x)):
#     news_predict_y.append(p.predict_SVM(news_train_x[i]))
# precision, recall, f1 = calc_f1(news_train_y, news_predict_y)
# print(precision)
# print(recall)
# print(f1)

# # f = open('p-conll.txt','a')
# # for i in range(len(syn_dense_test_y)):
# #     f.write(str(syn_dense_test_y[i]) + '\n');
# # f.close()


# # In[183]:


# # test on training set with parameters selected
# print('\nPerceptron Accuracy')
# # Test Perceptron on Sparse Synthetic
# p = Classifier('Perceptron', syn_sparse_train_x, syn_sparse_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)

# print('\nWinnow Accuracy')
# # Test Winnow on Sparse Synthetic
# p = Classifier('Winnow', syn_sparse_train_x, syn_sparse_train_y, iterations=10, alpha = 1.005)
# accuracy = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)

# print('\nAdagrad Accuracy')
# # Test Adagrad on Sparse Synthetic
# p = Classifier('Adagrad', syn_sparse_train_x, syn_sparse_train_y, iterations=10, eta = 1.5)
# accuracy = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)

# print('\nPerceptron-Avg Accuracy')
# # Test Perceptron-Avg on Sparse Synthetic
# p = Classifier('Perceptron-Avg', syn_sparse_train_x, syn_sparse_train_y, iterations=10)
# accuracy = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)

# print('\nWinnow-Avg Accuracy')
# # Test Winnow-Avg on Sparse Synthetic
# p = Classifier('Winnow-Avg', syn_sparse_train_x, syn_sparse_train_y, iterations=10, alpha = 1.005)
# accuracy = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)

# print('\nAdagrad_Avg Accuracy')
# # Test Adagrad on Sparse Synthetic
# p = Classifier('Adagrad-Avg', syn_sparse_train_x, syn_sparse_train_y, iterations=10, eta = 1.5)
# accuracy = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# print('Syn Sparse Dev Accuracy:', accuracy)

# print('\nSVM Accuracy')
# # Test Winnow-Avg on Dense Synthetic
# p = Classifier('SVM', syn_sparse_train_x, syn_sparse_train_y)
# accuracy = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict_SVM(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# print('Syn Dense Dev Accuracy:', accuracy)

