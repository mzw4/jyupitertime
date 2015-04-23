
# coding: utf-8

# In[7]:

import collections
import heapq
import matplotlib.pyplot as plt
import numpy as np
import os, time, json, sys
import pickle
import itertools, sys, math, time
from scipy.spatial import distance as dist
from numpy import linalg
from matplotlib import offsetbox
from scipy.sparse import csr_matrix
from sklearn import cluster, datasets, decomposition, ensemble, lda, manifold, random_projection
from sklearn.decomposition import TruncatedSVD
from sklearn import svm, grid_search
from sklearn.decomposition import PCA, SparsePCA
from scipy import sparse as sp
from sklearn import cluster
from sklearn import svm, grid_search
from sklearn.decomposition import PCA, SparsePCA
from scipy import sparse as sp

def load_sparse_data(filename, num_lines):
    """
    Function to load sparse data.
    """
    inverted_index = collections.defaultdict(set)
    
    sparse_indptr = [0]
    sparse_indices = []
    sparse_data = []
    vocabulary = {}

    print 'Reading data.'
    for line_num, line in enumerate(open(filename)):
        new_row = [(idx,float(prob)) for idx, prob in enumerate(line.strip().split(',')) if float(prob) > 0.0]
        for i,p in new_row:
            sparse_indices.append(i)
            sparse_data.append(p)
            inverted_index[i].add(line_num)
        sparse_indptr.append(len(sparse_indices))
        sys.stdout.write("\r%d%%" % (100.0 * line_num / num_lines))
        sys.stdout.flush()
    print 100.0 * line_num / num_lines, '%'
    print 'Done reading data.'

    sparse_matrix = csr_matrix((sparse_data, sparse_indices, sparse_indptr), dtype=float)
    return sparse_matrix, inverted_index

NUM_SPEECHES = 2740
NUM_DEBATES = 38

print 'Loading speech vectors...'
speech_vectors, inverted_index = load_sparse_data('speech_vectors.csv', NUM_SPEECHES)
print 'Loading speech graph...'
speech_graph, inverted_graph = load_sparse_data('speech_graph.csv', NUM_SPEECHES)

# run PCA on data
truncated_svd = TruncatedSVD(n_components=10)
reduced_data = truncated_svd.fit_transform(sp.hstack((speech_graph.todense(), speech_vectors[:, :25000])))

# form the training data
X = reduced_data[[2, 13, 18, 24, 1, 3, 27, 177], :]
y = [0 for i in range(4)] + [1 for i in range(4)]

# fit SVM, searching over paramaters
print 'Fitting SVM...'
parameters = {'kernel':('linear', 'rbf'), 'C':np.arange(1, 10, 0.5)}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(X, y)
print clf

# make predictions
print 'Making predictions...'
predictions = []
ones = 0

for i in range(len(reduced_data)):
    prediction = clf.predict(reduced_data[i])[0]
    if prediction == 1:
        ones += 1
    predictions += [prediction]

clf = grid_search.GridSearchCV(svr, parameters)

print '%% ones: ' + str(float(ones)/len(predictions))

good_preds = np.loadtxt(open('preds_laplacian1.csv'),delimiter=",",skiprows=1)

print '%% match with 99.7%% accurate solution' + str((2740-sum(np.logical_xor(predictions, good_preds[:, 1])))/2740.0)
print 'done'



