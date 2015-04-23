
# coding: utf-8


# ## Imports

# In[1]:

import collections
import heapq
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from matplotlib import offsetbox
from scipy.sparse import csr_matrix
from sklearn import cluster, datasets, decomposition, ensemble, lda, manifold, random_projection, grid_search
from sklearn.decomposition import TruncatedSVD
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from scipy import sparse as sp

# To start you off, you are told that points (rows, where the first row is number 0) 2, 13, 18, 24 are examples of speeches that belong to category “Against" (=label 0) and speeches 1, 3, 27, 177 are examples of speeches that belong to the “For”(=label 1) category

# In[2]:

LABELS_AGAINST = set([2, 13, 18, 24])
LABELS_FOR = set([1, 3, 27, 177])

ORIG_LABELS = set([2, 13, 18, 24, 1, 3, 27, 177])


# In[3]:

# Skip parts that take a long time.
SKIP_LONG_PARTS = True

# Create a dense representation of the data.
CREATE_DENSE_ARRAY = False

NUM_SPEECHES = 2740

NUM_DEBATES = 38


# In[4]:

def write_evaluation(tuples, filename):
    f = open(filename, 'w+')
    f.write('Id,Prediction\n')
    for t in tuples:
        f.write(str(t[0]) + ',' + str(t[1]) + '\n')


# ## Import data

# In[5]:

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

        if line_num % 150 == 0:
            print 100.0 * line_num / num_lines, '%'
    print 100.0 * line_num / num_lines, '%'
    print 'Done reading data.'

    sparse_matrix = csr_matrix((sparse_data, sparse_indices, sparse_indptr), dtype=float)
    return sparse_matrix, inverted_index


# In[6]:

def load_sparse_data_01(filename, num_lines):
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
            sparse_data.append(1.0)
            inverted_index[i].add(line_num)
        sparse_indptr.append(len(sparse_indices))

        if line_num % 150 == 0:
            print 100.0 * line_num / num_lines, '%'
    print 100.0 * line_num / num_lines, '%'
    print 'Done reading data.'

    sparse_matrix = csr_matrix((sparse_data, sparse_indices, sparse_indptr), dtype=float)
    return sparse_matrix, inverted_index


# In[6]:

def sparse_where(sparse_matrix, num):
    """
    np.where() for a sparse matrix. Returns a set of indices.
    """
    return set(np.where(sparse_matrix[num,:].toarray())[1].tolist())


# ### Load Speech Vectors

# In[7]:

sparse_data, inverted_index = load_sparse_data('speech_vectors.csv', NUM_SPEECHES)
print sparse_data.shape


# #### Load Speech Graph

# In[8]:

speech_graph, inverted_graph = load_sparse_data('speech_graph.csv', NUM_SPEECHES)
print speech_graph.shape


#
speech_graph_copy, _ = load_sparse_data('speech_graph.csv', NUM_SPEECHES)

speech_row_sums = {i: len(sparse_where(speech_graph, i)) for i in range(NUM_SPEECHES)}

# Create affinity matrix.
print 'Creating affinity matrix.'
for row in range(NUM_SPEECHES):
    row_indices = sparse_where(speech_graph, row)
    for index in row_indices:
        if abs(speech_row_sums[index] - speech_row_sums[row]) <= 50.8851648322 :
            speech_graph_copy[row, index] = 0.0
    if row % 150 == 0:
        print row * 100.0 / NUM_SPEECHES, '%'

print 'Done'


# ### Their Laplacian

# In[298]:

print 'Calculating spectral clustering.'
spectral_labels = cluster.spectral_clustering(speech_graph_copy, n_clusters=2)
plt.hist(spectral_labels, bins=38)

print ''

print 'Minority percentage'
print 100.0 * min(sum(spectral_labels), NUM_SPEECHES-sum(spectral_labels)) / NUM_SPEECHES

print ''

print 'labels for original against', spectral_labels[2], spectral_labels[13], spectral_labels[18], spectral_labels[24]
print 'labels for original for' spectral_labels[1], spectral_labels[3], spectral_labels[27], spectral_labels[177]


# In[303]:

preds_laplacian_tuples = []
for i in range(NUM_SPEECHES):
    #
    # Important! Make sure the original points are in the right cluster because labeling is arbitrary!!
    #
    preds_laplacian_tuples.append( (i, abs(spectral_labels[i]-1)) )
write_evaluation(preds_laplacian_tuples, 'preds_laplacian.csv')
