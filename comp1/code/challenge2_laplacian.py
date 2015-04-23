# ## Imports

# In[1]:

import collections
import gensim
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



# ## Finding Clusters one by one..

# In[81]:

speech_graph_copy, _ = load_sparse_data('speech_graph.csv', NUM_SPEECHES)

speech_row_sums = {i: len(sparse_where(speech_graph, i)) for i in range(NUM_SPEECHES)}

# Create affinity matrix.
print 'Creating affinity matrix.'
for row in range(NUM_SPEECHES):
    row_indices = sparse_where(speech_graph, row)
    for index in row_indices:
        if speech_row_sums[index] <= 634 :
            speech_graph_copy[row, index] = 0.0
    if row % 150 == 0:
        print row * 100.0 / NUM_SPEECHES, '%'

print 'Done'


# In[37]:

speech_graph_copy_sums = {i: len(sparse_where(speech_graph_copy, i)) for i in range(NUM_SPEECHES)}


# In[39]:

plt.hist([speech_graph_copy_sums[i] for i in range(2740) if i in rows_over_635])


# In[43]:

print sum([1 for i in range(2740) if i in rows_over_635 and speech_graph_copy_sums[i] > 250])
rows_over_635_trimmed = [i for i in range(2740) if i in rows_over_635 and speech_graph_copy_sums[i] > 250]
print len(rows_over_635_trimmed)


# In[158]:

print [speech_row_sums[i] for i in range(2740) if i in rows_over_635 and speech_graph_copy_sums[i] < 250]


# In[51]:

speech_graph_copy_trimmed = speech_graph_copy[rows_over_635_trimmed,:]
print speech_graph_copy_trimmed.shape
speech_graph_copy_trimmed = speech_graph_copy_trimmed[:,rows_over_635_trimmed]
print speech_graph_copy_trimmed.shape


# In[55]:

spectral_labels = cluster.spectral_clustering(speech_graph_copy_trimmed, n_clusters=3)
plt.hist(spectral_labels, bins=10)
print [list(spectral_labels).count(i) for i in range(3)]


# In[59]:

big_3_labels = spectral_labels
orig_speech_graph_big_3 = speech_graph[rows_over_635_trimmed,:]
print orig_speech_graph_big_3.shape


# In[60]:

json.dump(zip(rows_over_635_trimmed,big_3_labels), open('big_3_clusters.json', 'w+'))


# In[451]:

zipped = json.load(open('big_3_clusters.json'))
print len(zipped)


# ### Estimate the probabilities

# In[76]:

# Load ground truth for for/against labels.
for_against_label = {}

for_against_label_file = 'preds_laplacian1.csv'
header = True # Skip the header.
for row in open(for_against_label_file):
    if header:
        header = False
        continue
    
    entries = row.strip().split(',')
    speech_num = int(entries[0])
    label = entries[1]
    
    for_against_label[speech_num] = label
    
for_against = for_against_label
print len(for_against)


# In[82]:

# Debate prob
p1s = []
# Yes/no prob
p2s = []
# Noise prob
p3s = []

rows_over_635_trimmed_list = list(rows_over_635_trimmed)
big_3_labels_list = list(big_3_labels)

for index1 in list(rows_over_635_trimmed):
    
    p1_est = 0.0
    p2_est = 0.0
    p3_est = 0.0
    
    for index2 in sparse_where(speech_graph, index):
        # calculate p1
        if index2 in rows_over_635_trimmed_list and                 big_3_labels[rows_over_635_trimmed_list.index(index1)] == big_3_labels[rows_over_635_trimmed_list.index(index2)]:
            p1_est += 1
        # calculate p2
        elif for_against[index1] == for_against[index2]:
            p2_est += 1
        # calculate p3
        else:
            p3_est += 1
     
    d1 = big_3_labels_list.count(big_3_labels_list[rows_over_635_trimmed_list.index(index1)])
    p1_est /= d1
    
    d2 = for_against.values().count(for_against[index1])
    d2 -= sum([1 for i in rows_over_635_trimmed_list if i != index1 and for_against[i] == for_against[index1]])
    p2_est /= d2
    
    d3 = 2740
    d3 -= d1
    d3 -= d2
    p3_est /= d3
    
    p1s.append(p1_est)
    p2s.append(p2_est)
    p3s.append(p3_est)
    


# In[83]:

print np.average(p1s)
print np.std(p1s)
plt.hist(p1s, bins=20)


# In[84]:

print np.average(p2s)
print np.std(p2s)
plt.hist(p2s, bins=20)


# In[85]:

print np.average(p3s)
print np.std(p3s)
plt.hist(p3s, bins=20)


# In[120]:

def expected_value(cluster_size):
    m1 = cluster_size * 0.19
    m2 =  (1370 - 0.5*cluster_size) * 0.27
    m3 = (2740 - cluster_size) - (1370 - 0.5 * cluster_size)
    m3 *= 0.16
    return m1+m2+m3

def calc_std_dev(cluster_size):
    # debate
    var1 = cluster_size * 0.19 * (1 - 0.91)
    # for against estimate
    var2 = (1370 - 0.5*cluster_size) * 0.27 * (1 - 0.27)
    # noise
    var3 = 2740 - cluster_size
    var3 -= var3 - (1370 - 0.5 * cluster_size)
    var3 *= 0.16 * (1 - 0.16)
    return np.sqrt(var1 + var2 + var3)


# In[122]:

n = 3

print expected_value(n)
print calc_std_dev(n)


# In[139]:

header = False
actual_debates = {}
for line in open('actual-debate-labels.csv'):
    if header:
        header = False
        continue
    parts = line.strip().split(',')
    actual_debates[int(parts[0])] = int(parts[1])
    
print len(actual_debates)


# ### Start

# In[702]:

#speech_graph_copy_sums = {i: len(sparse_where(speech_graph_copy, i)) for i in range(NUM_SPEECHES) if i in remaining_set}

t = [(k,actual_debates[k]) for k in actual_debates]
t.sort(key=lambda v: v[1], reverse=True)

print '#\tseed\tsize'
counter = 0
points_left = 0
for tup in t:
    if tup[0] in remaining_set:
        counter += 1
        points_left += sorted(actual_cluster_sizes)[-tup[1]-1]
        print tup[1], '\t', tup[0], '\t', sorted(actual_cluster_sizes)[-tup[1]-1]
        
print '==='
print counter ,'clusters left'
print points_left, 'points left'


# In[465]:

remaining = [i for i in range(NUM_SPEECHES) if i not in rows_over_635_trimmed]
remaining_set = set(remaining)

speech_row_sums = {i: len(sparse_where(speech_graph, i)) for i in range(NUM_SPEECHES)}

speech_graph_copy, _ = load_sparse_data('speech_graph.csv', NUM_SPEECHES)

# Create affinity matrix.
print 'Creating affinity matrix.'
for row in range(NUM_SPEECHES):
    row_indices = sparse_where(speech_graph, row)
    for index in row_indices:
        # Remove connections to points we have already labeled.
        # Remove connections to debate with the same label.
        if index not in remaining_set or for_against[row] == for_against[index]:
            speech_graph_copy[row, index] = 0.0
    if row % 450 == 0:
        print row * 100.0 / NUM_SPEECHES, '%'

print 'Done'


# In[458]:

remaining = [i[0] for i in json.load(open('big_3_clusters.json'))]
remaining_set = set(range(NUM_SPEECHES)).difference(set(remaining))
remaining = list(remaining_set)
print len(remaining)


# In[459]:

actual_debate_to_others = collections.defaultdict(list)


# ### Cluster 3, N = 18

# In[ ]:

N = 18
s = set(sparse_where(speech_graph, N))
ns = []
for i in range(NUM_SPEECHES):
    if i != N and i in remaining and i not in actual_debates:
        c = len(s.intersection(set(sparse_where(speech_graph_copy, i))))
        if c != 0:
            ns.append( c )

print sum([1 for i in ns if i > 55])        
plt.hist(ns, bins=20)


# In[275]:

for i in range(NUM_SPEECHES):
    if i != N:
        c = len(s.intersection(set(sparse_where(speech_graph_copy, i))))
        if c > 51:
            actual_debate_to_others[N].append(i)
            
print len(actual_debate_to_others[N])


# In[450]:

pos = []
votes = collections.defaultdict(int)
for i in range(NUM_SPEECHES):
    if i != N:
        d = set(sparse_where(speech_graph_copy, i))
        c = len(s.intersection(d))
        if c > 50:
            pos.append(i)
            for j in d:
                votes[j] += 1

print sum([1 for i in votes if votes[i] > 31])
plt.hist(votes.values())


# In[277]:

actual_debate_to_others[N].extend([i for i in votes if votes[i] > 31])
print len(actual_debate_to_others[N])


# In[278]:

print 'Before'

remaining_old = remaining
remaining_set_old = remaining_set
print len(remaining_old), len(remaining_set_old)


remaining_set = remaining_set.difference(set(actual_debate_to_others[N]))
remaining = sorted(list(remaining_set))

print 'After'
print len(remaining), len(remaining_set)


# ### Cluster 5, N = 13

# In[293]:

remaining_set.remove(N)
remaining = sorted(list(remaining_set))

print len(remaining), len(remaining_set)


# In[280]:

speech_graph_copy, _ = load_sparse_data('speech_graph.csv', NUM_SPEECHES)

# Create affinity matrix.
print 'Creating affinity matrix.'
for row in range(NUM_SPEECHES):
    row_indices = sparse_where(speech_graph, row)
    for index in row_indices:
        # Remove connections to points we have already labeled.
        # Remove connections to debate with the same label.
        if index not in remaining_set or for_against[row] == for_against[index]:
            speech_graph_copy[row, index] = 0.0
    if row % 450 == 0:
        print row * 100.0 / NUM_SPEECHES, '%'

print 'Done'


# In[295]:

N = 13
threshold = 65

s = set(sparse_where(speech_graph, N))
ns = []
for i in range(NUM_SPEECHES):
    if i != N:
        c = len(s.intersection(set(sparse_where(speech_graph_copy, i))))
        if c != 0:
            ns.append( c )

print sum([1 for i in ns if i > threshold])        
plt.hist(ns, bins=40)


# In[300]:

pos = []
votes = collections.defaultdict(int)
for i in range(NUM_SPEECHES):
    if i != N:
        d = set(sparse_where(speech_graph_copy, i))
        c = len(s.intersection(d))
        if c > threshold:
            pos.append(i)
            for j in d:
                votes[j] += 1

second_threshold = 15
                
print sum([1 for i in votes if votes[i] > second_threshold])
plt.hist(votes.values())


# In[299]:

for i in range(NUM_SPEECHES):
    if i != N:
        c = len(s.intersection(set(sparse_where(speech_graph_copy, i))))
        if c > threshold:
            actual_debate_to_others[N].append(i)
            
print len(actual_debate_to_others[N])

actual_debate_to_others[N].extend([i for i in votes if votes[i] > second_threshold])
print len(actual_debate_to_others[N])


# In[304]:

# undo next step
remaining = remaining_old
remaining_set = remaining_set_old


# In[305]:

print 'Before'

remaining_old = remaining
remaining_set_old = remaining_set
print len(remaining_old), len(remaining_set_old)


remaining_set = remaining_set.difference(set(actual_debate_to_others[N]))
remaining = sorted(list(remaining_set))

#if N in remaining_setremaining_set.remove(N)
#remaining = sorted(list(remaining_set))

print 'After'
print len(remaining), len(remaining_set)

print 'Difference'
print len(remaining_set_old) - len(remaining_set)


# ### Cluster 6, N = 30

# In[306]:

print len(remaining), len(remaining_set)


# In[307]:

speech_graph_copy, _ = load_sparse_data('speech_graph.csv', NUM_SPEECHES)

# Create affinity matrix.
print 'Creating affinity matrix.'
for row in range(NUM_SPEECHES):
    row_indices = sparse_where(speech_graph, row)
    for index in row_indices:
        # Remove connections to points we have already labeled.
        # Remove connections to debate with the same label.
        if index not in remaining_set or for_against[row] == for_against[index]:
            speech_graph_copy[row, index] = 0.0
    if row % 450 == 0:
        print row * 100.0 / NUM_SPEECHES, '%'

print 'Done'


# In[312]:

N = 30
threshold = 60

s = set(sparse_where(speech_graph, N))
ns = []
for i in range(NUM_SPEECHES):
    if i != N:
        c = len(s.intersection(set(sparse_where(speech_graph_copy, i))))
        if c != 0:
            ns.append( c )

print sum([1 for i in ns if i > threshold])        
plt.hist(ns, bins=40)


# In[316]:

pos = []
votes = collections.defaultdict(int)
for i in range(NUM_SPEECHES):
    if i != N:
        d = set(sparse_where(speech_graph_copy, i))
        c = len(s.intersection(d))
        if c > threshold:
            pos.append(i)
            for j in d:
                votes[j] += 1

second_threshold = 20
                
print sum([1 for i in votes if votes[i] > second_threshold])
plt.hist(votes.values())


# In[317]:

for i in range(NUM_SPEECHES):
    if i != N:
        c = len(s.intersection(set(sparse_where(speech_graph_copy, i))))
        if c > threshold:
            actual_debate_to_others[N].append(i)
            
print len(actual_debate_to_others[N])

actual_debate_to_others[N].extend([i for i in votes if votes[i] > second_threshold])
print len(actual_debate_to_others[N])


# In[ ]:

# undo next step
remaining = remaining_old
remaining_set = remaining_set_old


# In[318]:

print 'Before'

remaining_old = remaining
remaining_set_old = remaining_set
print len(remaining_old), len(remaining_set_old)


remaining_set = remaining_set.difference(set(actual_debate_to_others[N]))
remaining = sorted(list(remaining_set))

#if N in remaining_setremaining_set.remove(N)
#remaining = sorted(list(remaining_set))

print 'After'
print len(remaining), len(remaining_set)

print 'Difference'
print len(remaining_set_old) - len(remaining_set)


# ### Cluster 4, N = 2

# In[396]:

speech_graph_copy_trimmed = speech_graph[sorted(remaining),:]
print speech_graph_copy_trimmed.shape
speech_graph_copy_trimmed = speech_graph_copy_trimmed[:,sorted(remaining)]
print speech_graph_copy_trimmed.shape


# In[397]:

spectral_labels = cluster.spectral_clustering(speech_graph_copy_trimmed, n_clusters=2)
plt.hist(spectral_labels, bins=10)
print [list(spectral_labels).count(i) for i in range(2)]


# In[401]:

N = 2

min_label = 1
if list(spectral_labels).count(1) > list(spectral_labels).count(0):
    min_label = 0
    
for l,p in zip(spectral_labels, sorted(remaining)):
    if l == min_label:
        actual_debate_to_others[N].append(p)


# In[ ]:

# undo next step
remaining = remaining_old
remaining_set = remaining_set_old


# In[402]:

print 'Before'

remaining_old = remaining
remaining_set_old = remaining_set
print len(remaining_old), len(remaining_set_old)


remaining_set = remaining_set.difference(set(actual_debate_to_others[N]))
remaining = sorted(list(remaining_set))

#if N in remaining_setremaining_set.remove(N)
#remaining = sorted(list(remaining_set))

print 'After'
print len(remaining), len(remaining_set)

print 'Difference'
print len(remaining_set_old) - len(remaining_set)


# ### Cluster 10, N = 8

# In[403]:

print len(remaining), len(remaining_set)


# In[404]:

speech_graph_copy, _ = load_sparse_data('speech_graph.csv', NUM_SPEECHES)

# Create affinity matrix.
print 'Creating affinity matrix.'
for row in range(NUM_SPEECHES):
    row_indices = sparse_where(speech_graph, row)
    for index in row_indices:
        # Remove connections to points we have already labeled.
        # Remove connections to debate with the same label.
        if index not in remaining_set or for_against[row] == for_against[index]:
            speech_graph_copy[row, index] = 0.0
    if row % 450 == 0:
        print row * 100.0 / NUM_SPEECHES, '%'

print 'Done'


# In[424]:

N = 8
threshold = 46 # to 47

s = set(sparse_where(speech_graph, N))
ns = []
for i in range(NUM_SPEECHES):
    if i != N and i in remaining:
        c = len(s.intersection(set(sparse_where(speech_graph_copy, i))))
        if c != 0:
            ns.append( c )

print sum([1 for i in ns if i > threshold])        
plt.hist(ns, bins=50)


# In[425]:

pos = []
votes = collections.defaultdict(int)
for i in range(NUM_SPEECHES):
    if i != N and i in remaining:
        d = set(sparse_where(speech_graph_copy, i))
        c = len(s.intersection(d))
        if c > threshold:
            pos.append(i)
            for j in d:
                votes[j] += 1

second_threshold = 15
                
print sum([1 for i in votes if votes[i] > second_threshold])
plt.hist(votes.values())


# In[427]:

actual_debate_to_others[N] = []

for i in range(NUM_SPEECHES):
    if i != N and i in remaining:
        c = len(s.intersection(set(sparse_where(speech_graph_copy, i))))
        if c > threshold:
            actual_debate_to_others[N].append(i)
            
print len(actual_debate_to_others[N])

actual_debate_to_others[N].extend([i for i in votes if votes[i] > second_threshold])
print len(actual_debate_to_others[N])


# In[ ]:

# undo next step
remaining = remaining_old
remaining_set = remaining_set_old


# In[428]:

print 'Before'

remaining_old = remaining
remaining_set_old = remaining_set
print len(remaining_old), len(remaining_set_old)


remaining_set = remaining_set.difference(set(actual_debate_to_others[N]))
remaining = sorted(list(remaining_set))

#if N in remaining_setremaining_set.remove(N)
#remaining = sorted(list(remaining_set))

print 'After'
print len(remaining), len(remaining_set)

print 'Difference'
print len(remaining_set_old) - len(remaining_set)


# ### Cluster 7-8-9, N = 0, 11, 20

# In[481]:

print len(remaining), len(remaining_set)


# In[482]:

speech_graph_copy, _ = load_sparse_data('speech_graph.csv', NUM_SPEECHES)

# Create affinity matrix.
print 'Creating affinity matrix.'
for row in range(NUM_SPEECHES):
    row_indices = sparse_where(speech_graph, row)
    for index in row_indices:
        # Remove connections to points we have already labeled.
        # Remove connections to debate with the same label.
        if index not in remaining_set or for_against[row] == for_against[index]:
            speech_graph_copy[row, index] = 0.0
    if row % 450 == 0:
        print row * 100.0 / NUM_SPEECHES, '%'

print 'Done'


# In[484]:

speech_graph_copy_trimmed = speech_graph[sorted(remaining),:]
print speech_graph_copy_trimmed.shape
speech_graph_copy_trimmed = speech_graph_copy_trimmed[:,sorted(remaining)]
print speech_graph_copy_trimmed.shape


# In[485]:

n_clusters = 4
spectral_labels = cluster.spectral_clustering(speech_graph_copy_trimmed, n_clusters=n_clusters)
plt.hist(spectral_labels, bins=10)
print [list(spectral_labels).count(i) for i in range(n_clusters)]
print [spectral_labels[remaining.index(i)] for i in [0,11,20]]


# In[487]:

## Fix indices by hand..

for l,p in zip(spectral_labels, sorted(remaining)):
    if l == 1:
        actual_debate_to_others[0].append(p)
    elif l == 3:
        actual_debate_to_others[11].append(p)
    elif l == 2:
        actual_debate_to_others[20].append(p)


# In[ ]:

# undo next step
remaining = remaining_old
remaining_set = remaining_set_old


# In[488]:

print 'Before'

remaining_old = remaining
remaining_set_old = remaining_set
print len(remaining_old), len(remaining_set_old)


remaining_set = remaining_set.difference(set(actual_debate_to_others[0]))
remaining_set = remaining_set.difference(set(actual_debate_to_others[11]))
remaining_set = remaining_set.difference(set(actual_debate_to_others[20]))
remaining = sorted(list(remaining_set))

#if N in remaining_setremaining_set.remove(N)
#remaining = sorted(list(remaining_set))

print 'After'
print len(remaining), len(remaining_set)

print 'Difference'
print len(remaining_set_old) - len(remaining_set)


# ### Cluster 11, N = 96

# In[495]:

print len(remaining), len(remaining_set)


# In[496]:

speech_graph_copy, _ = load_sparse_data('speech_graph.csv', NUM_SPEECHES)

# Create affinity matrix.
print 'Creating affinity matrix.'
for row in range(NUM_SPEECHES):
    row_indices = sparse_where(speech_graph, row)
    for index in row_indices:
        # Remove connections to points we have already labeled.
        # Remove connections to debate with the same label.
        if index not in remaining_set or for_against[row] == for_against[index]:
            speech_graph_copy[row, index] = 0.0
    if row % 450 == 0:
        print row * 100.0 / NUM_SPEECHES, '%'

print 'Done'


# In[533]:

speech_graph_copy_trimmed = speech_graph[sorted(remaining),:]
print speech_graph_copy_trimmed.shape
speech_graph_copy_trimmed = speech_graph_copy_trimmed[:,sorted(remaining)]
print speech_graph_copy_trimmed.shape


# In[538]:

n_clusters = 3
spectral_labels = cluster.spectral_clustering(speech_graph_copy_trimmed, n_clusters=n_clusters)
plt.hist(spectral_labels, bins=10)
print [list(spectral_labels).count(i) for i in range(n_clusters)]
print [(spectral_labels[remaining.index(i)],i) for i in actual_debates if i in remaining_set]


# In[539]:

## Fix indices by hand..

for l,p in zip(spectral_labels, sorted(remaining)):
    if l == 2:
        actual_debate_to_others[96].append(p)


# In[ ]:

# undo next step
remaining = remaining_old
remaining_set = remaining_set_old


# In[541]:

print 'Before'

remaining_old = remaining
remaining_set_old = remaining_set
print len(remaining_old), len(remaining_set_old)


remaining_set = remaining_set.difference(set(actual_debate_to_others[96]))
remaining = sorted(list(remaining_set))

#if N in remaining_setremaining_set.remove(N)
#remaining = sorted(list(remaining_set))

print 'After'
print len(remaining), len(remaining_set)

print 'Difference'
print len(remaining_set_old) - len(remaining_set)


# ### Cluster 12-13-14-15, N = 12,125,59,4

# In[542]:

print len(remaining), len(remaining_set)


# In[543]:

speech_graph_copy, _ = load_sparse_data('speech_graph.csv', NUM_SPEECHES)

# Create affinity matrix.
print 'Creating affinity matrix.'
for row in range(NUM_SPEECHES):
    row_indices = sparse_where(speech_graph, row)
    for index in row_indices:
        # Remove connections to points we have already labeled.
        # Remove connections to debate with the same label.
        if index not in remaining_set or for_against[row] == for_against[index]:
            speech_graph_copy[row, index] = 0.0
    if row % 450 == 0:
        print row * 100.0 / NUM_SPEECHES, '%'

print 'Done'


# In[547]:

speech_graph_copy_trimmed = speech_graph[sorted(remaining),:]
print speech_graph_copy_trimmed.shape
speech_graph_copy_trimmed = speech_graph_copy_trimmed[:,sorted(remaining)]
print speech_graph_copy_trimmed.shape


# In[554]:

n_clusters = 6
spectral_labels = cluster.spectral_clustering(speech_graph_copy_trimmed, n_clusters=n_clusters)
plt.hist(spectral_labels, bins=10)
print [list(spectral_labels).count(i) for i in range(n_clusters)]
print [(spectral_labels[remaining.index(i)],i) for i in actual_debates if i in remaining_set]


# In[555]:

## Fix indices by hand..

for l,p in zip(spectral_labels, sorted(remaining)):
    if l == 5:
        actual_debate_to_others[12].append(p)
    elif l == 4:
        actual_debate_to_others[59].append(p)
    elif l == 3:
        actual_debate_to_others[125].append(p)
    elif l == 1:
        actual_debate_to_others[4].append(p)


# In[ ]:

# undo next step
remaining = remaining_old
remaining_set = remaining_set_old


# In[556]:

print 'Before'

remaining_old = remaining
remaining_set_old = remaining_set
print len(remaining_old), len(remaining_set_old)


remaining_set = remaining_set.difference(set(actual_debate_to_others[12]))
remaining_set = remaining_set.difference(set(actual_debate_to_others[125]))
remaining_set = remaining_set.difference(set(actual_debate_to_others[59]))
remaining_set = remaining_set.difference(set(actual_debate_to_others[4]))
remaining = sorted(list(remaining_set))

print 'After'
print len(remaining), len(remaining_set)

print 'Difference'
print len(remaining_set_old) - len(remaining_set)


# ### Cluster 17, N = 9

# In[561]:

print len(remaining), len(remaining_set)


# In[563]:

speech_graph_copy, _ = load_sparse_data('speech_graph.csv', NUM_SPEECHES)

# Create affinity matrix.
print 'Creating affinity matrix.'
for row in range(NUM_SPEECHES):
    row_indices = sparse_where(speech_graph, row)
    for index in row_indices:
        # Remove connections to points we have already labeled.
        # Remove connections to debate with the same label.
        if index not in remaining_set or for_against[row] == for_against[index]:
            speech_graph_copy[row, index] = 0.0
    if row % 450 == 0:
        print row * 100.0 / NUM_SPEECHES, '%'

print 'Done'


# In[564]:

speech_graph_copy_trimmed = speech_graph[sorted(remaining),:]
print speech_graph_copy_trimmed.shape
speech_graph_copy_trimmed = speech_graph_copy_trimmed[:,sorted(remaining)]
print speech_graph_copy_trimmed.shape


# In[571]:

n_clusters = 5
spectral_labels = cluster.spectral_clustering(speech_graph_copy_trimmed, n_clusters=n_clusters)
plt.hist(spectral_labels, bins=10)
print [list(spectral_labels).count(i) for i in range(n_clusters)]
print [(spectral_labels[remaining.index(i)],i) for i in actual_debates if i in remaining_set]


# In[572]:

## Fix indices by hand..

for l,p in zip(spectral_labels, sorted(remaining)):
    if l == 4:
        actual_debate_to_others[9].append(p)


# In[ ]:

# undo next step
remaining = remaining_old
remaining_set = remaining_set_old


# In[573]:

print 'Before'

remaining_old = remaining
remaining_set_old = remaining_set
print len(remaining_old), len(remaining_set_old)


remaining_set = remaining_set.difference(set(actual_debate_to_others[9]))
remaining = sorted(list(remaining_set))

print 'After'
print len(remaining), len(remaining_set)

print 'Difference'
print len(remaining_set_old) - len(remaining_set)


# ### Cluster 16-18, N = 160-28

# In[577]:

print len(remaining), len(remaining_set)


# In[576]:

speech_graph_copy, _ = load_sparse_data('speech_graph.csv', NUM_SPEECHES)

# Create affinity matrix.
print 'Creating affinity matrix.'
for row in range(NUM_SPEECHES):
    row_indices = sparse_where(speech_graph, row)
    for index in row_indices:
        # Remove connections to points we have already labeled.
        # Remove connections to debate with the same label.
        if index not in remaining_set or for_against[row] == for_against[index]:
            speech_graph_copy[row, index] = 0.0
    if row % 450 == 0:
        print row * 100.0 / NUM_SPEECHES, '%'

print 'Done'


# In[599]:

speech_graph_copy_trimmed = speech_graph[sorted(remaining),:]
print speech_graph_copy_trimmed.shape
speech_graph_copy_trimmed = speech_graph_copy_trimmed[:,sorted(remaining)]
print speech_graph_copy_trimmed.shape


# In[609]:

n_clusters = 4
spectral_labels = cluster.spectral_clustering(speech_graph_copy_trimmed, n_clusters=n_clusters)
plt.hist(spectral_labels, bins=10)
print [list(spectral_labels).count(i) for i in range(n_clusters)]
print [(spectral_labels[remaining.index(i)],i) for i in actual_debates if i in remaining_set]


# In[611]:

temp_s = [i for i in remaining if spectral_labels[remaining.index(i)] > 1]


# In[612]:

speech_graph_copy_trimmed1 = speech_graph[sorted(temp_s),:]
print speech_graph_copy_trimmed1.shape
speech_graph_copy_trimmed1 = speech_graph_copy_trimmed1[:,sorted(temp_s)]
print speech_graph_copy_trimmed1.shape


# In[617]:

n_clusters = 2
spectral_labels1 = cluster.spectral_clustering(speech_graph_copy_trimmed1, n_clusters=n_clusters)
plt.hist(spectral_labels1, bins=10)
print [list(spectral_labels1).count(i) for i in range(n_clusters)]
print [(spectral_labels1[temp_s.index(i)],i) for i in actual_debates if i in remaining_set and i in temp_s]


# In[618]:

## Fix indices by hand..

for l,p in zip(spectral_labels1, sorted(temp_s)):
    if l == 0:
        actual_debate_to_others[28].append(p)
    else:
        actual_debate_to_others[160].append(p)

print len(actual_debate_to_others[28])
print len(actual_debate_to_others[160])


# In[ ]:

# undo next step
remaining = remaining_old
remaining_set = remaining_set_old


# In[620]:

print 'Before'

remaining_old = remaining
remaining_set_old = remaining_set
print len(remaining_old), len(remaining_set_old)


remaining_set = remaining_set.difference(set(actual_debate_to_others[28]))
remaining_set = remaining_set.difference(set(actual_debate_to_others[160]))
remaining = sorted(list(remaining_set))

print 'After'
print len(remaining), len(remaining_set)

print 'Difference'
print len(remaining_set_old) - len(remaining_set)


# ### Cluster 19-20-21-23, N = 31-6-145-69

# In[622]:

print len(remaining), len(remaining_set)


# In[623]:

speech_graph_copy, _ = load_sparse_data('speech_graph.csv', NUM_SPEECHES)

# Create affinity matrix.
print 'Creating affinity matrix.'
for row in range(NUM_SPEECHES):
    row_indices = sparse_where(speech_graph, row)
    for index in row_indices:
        # Remove connections to points we have already labeled.
        # Remove connections to debate with the same label.
        if index not in remaining_set or for_against[row] == for_against[index]:
            speech_graph_copy[row, index] = 0.0
    if row % 450 == 0:
        print row * 100.0 / NUM_SPEECHES, '%'

print 'Done'


# In[626]:

speech_graph_copy_trimmed = speech_graph[sorted(remaining),:]
print speech_graph_copy_trimmed.shape
speech_graph_copy_trimmed = speech_graph_copy_trimmed[:,sorted(remaining)]
print speech_graph_copy_trimmed.shape


# In[633]:

n_clusters = 5
spectral_labels = cluster.spectral_clustering(speech_graph_copy_trimmed, n_clusters=n_clusters)
plt.hist(spectral_labels, bins=10)
print [list(spectral_labels).count(i) for i in range(n_clusters)]
print [(spectral_labels[remaining.index(i)],i) for i in actual_debates if i in remaining_set]


# In[635]:

## Fix indices by hand..

actual_debate_to_others[31] = []
actual_debate_to_others[6] = []
actual_debate_to_others[145] = []
actual_debate_to_others[69] = []

for l,p in zip(spectral_labels, sorted(remaining)):
    if l == 0:
        actual_debate_to_others[31].append(p)
    elif l == 2:
        actual_debate_to_others[6].append(p)
    elif l == 3:
        actual_debate_to_others[145].append(p)
    elif l == 4:
        actual_debate_to_others[69].append(p)

print len(actual_debate_to_others[31])
print len(actual_debate_to_others[6])
print len(actual_debate_to_others[145])
print len(actual_debate_to_others[69])


# In[ ]:

# undo next step
remaining = remaining_old
remaining_set = remaining_set_old


# In[636]:

print 'Before'

remaining_old = remaining
remaining_set_old = remaining_set
print len(remaining_old), len(remaining_set_old)


remaining_set = remaining_set.difference(set(actual_debate_to_others[31]))
remaining_set = remaining_set.difference(set(actual_debate_to_others[6]))
remaining_set = remaining_set.difference(set(actual_debate_to_others[145]))
remaining_set = remaining_set.difference(set(actual_debate_to_others[69]))
remaining = sorted(list(remaining_set))

print 'After'
print len(remaining), len(remaining_set)

print 'Difference'
print len(remaining_set_old) - len(remaining_set)


# ### Cluster 22-24-25, N = 55-46-22

# In[638]:

print len(remaining), len(remaining_set)


# In[640]:

speech_graph_copy, _ = load_sparse_data('speech_graph.csv', NUM_SPEECHES)

# Create affinity matrix.
print 'Creating affinity matrix.'
for row in range(NUM_SPEECHES):
    row_indices = sparse_where(speech_graph, row)
    for index in row_indices:
        # Remove connections to points we have already labeled.
        # Remove connections to debate with the same label.
        if index not in remaining_set or for_against[row] == for_against[index]:
            speech_graph_copy[row, index] = 0.0
    if row % 450 == 0:
        print row * 100.0 / NUM_SPEECHES, '%'

print 'Done'


# In[641]:

speech_graph_copy_trimmed = speech_graph[sorted(remaining),:]
print speech_graph_copy_trimmed.shape
speech_graph_copy_trimmed = speech_graph_copy_trimmed[:,sorted(remaining)]
print speech_graph_copy_trimmed.shape


# In[647]:

n_clusters = 4
spectral_labels = cluster.spectral_clustering(speech_graph_copy_trimmed, n_clusters=n_clusters)
plt.hist(spectral_labels, bins=10)
print [list(spectral_labels).count(i) for i in range(n_clusters)]
print [(spectral_labels[remaining.index(i)],i) for i in actual_debates if i in remaining_set]


# In[649]:

## Fix indices by hand..

actual_debate_to_others[22] = []
actual_debate_to_others[46] = []
actual_debate_to_others[55] = []

for l,p in zip(spectral_labels, sorted(remaining)):
    if l == 1:
        actual_debate_to_others[55].append(p)
    elif l == 2:
        actual_debate_to_others[46].append(p)
    elif l == 3:
        actual_debate_to_others[22].append(p)

print len(actual_debate_to_others[55])
print len(actual_debate_to_others[46])
print len(actual_debate_to_others[22])


# In[ ]:

# undo next step
remaining = remaining_old
remaining_set = remaining_set_old


# In[650]:

print 'Before'

remaining_old = remaining
remaining_set_old = remaining_set
print len(remaining_old), len(remaining_set_old)


remaining_set = remaining_set.difference(set(actual_debate_to_others[22]))
remaining_set = remaining_set.difference(set(actual_debate_to_others[46]))
remaining_set = remaining_set.difference(set(actual_debate_to_others[55]))
remaining = sorted(list(remaining_set))

print 'After'
print len(remaining), len(remaining_set)

print 'Difference'
print len(remaining_set_old) - len(remaining_set)


# ### Cluster 26-27, N = 53-130

# In[653]:

print len(remaining), len(remaining_set)


# In[654]:

speech_graph_copy, _ = load_sparse_data('speech_graph.csv', NUM_SPEECHES)

# Create affinity matrix.
print 'Creating affinity matrix.'
for row in range(NUM_SPEECHES):
    row_indices = sparse_where(speech_graph, row)
    for index in row_indices:
        # Remove connections to points we have already labeled.
        # Remove connections to debate with the same label.
        if index not in remaining_set or for_against[row] == for_against[index]:
            speech_graph_copy[row, index] = 0.0
    if row % 450 == 0:
        print row * 100.0 / NUM_SPEECHES, '%'

print 'Done'


# In[655]:

speech_graph_copy_trimmed = speech_graph[sorted(remaining),:]
print speech_graph_copy_trimmed.shape
speech_graph_copy_trimmed = speech_graph_copy_trimmed[:,sorted(remaining)]
print speech_graph_copy_trimmed.shape


# In[660]:

n_clusters = 3
spectral_labels = cluster.spectral_clustering(speech_graph_copy_trimmed, n_clusters=n_clusters)
plt.hist(spectral_labels, bins=10)
print [list(spectral_labels).count(i) for i in range(n_clusters)]
print [(spectral_labels[remaining.index(i)],i) for i in actual_debates if i in remaining_set]


# In[665]:

## Fix indices by hand..

actual_debate_to_others[130] = []
actual_debate_to_others[53] = []

for l,p in zip(spectral_labels, sorted(remaining)):
    if l == 0:
        actual_debate_to_others[130].append(p)
    elif l == 2:
        actual_debate_to_others[53].append(p)

print len(actual_debate_to_others[130])
print len(actual_debate_to_others[53])


# In[ ]:

# undo next step
remaining = remaining_old
remaining_set = remaining_set_old


# In[666]:

print 'Before'

remaining_old = remaining
remaining_set_old = remaining_set
print len(remaining_old), len(remaining_set_old)


remaining_set = remaining_set.difference(set(actual_debate_to_others[130]))
remaining_set = remaining_set.difference(set(actual_debate_to_others[53]))
remaining = sorted(list(remaining_set))

print 'After'
print len(remaining), len(remaining_set)

print 'Difference'
print len(remaining_set_old) - len(remaining_set)


# ### Cluster 28-30-31, N = 14-43-181

# In[669]:

print len(remaining), len(remaining_set)


# In[670]:

speech_graph_copy, _ = load_sparse_data('speech_graph.csv', NUM_SPEECHES)

# Create affinity matrix.
print 'Creating affinity matrix.'
for row in range(NUM_SPEECHES):
    row_indices = sparse_where(speech_graph, row)
    for index in row_indices:
        # Remove connections to points we have already labeled.
        # Remove connections to debate with the same label.
        if index not in remaining_set or for_against[row] == for_against[index]:
            speech_graph_copy[row, index] = 0.0
    if row % 450 == 0:
        print row * 100.0 / NUM_SPEECHES, '%'

print 'Done'


# In[671]:

speech_graph_copy_trimmed = speech_graph[sorted(remaining),:]
print speech_graph_copy_trimmed.shape
speech_graph_copy_trimmed = speech_graph_copy_trimmed[:,sorted(remaining)]
print speech_graph_copy_trimmed.shape


# In[686]:

n_clusters = 5
spectral_labels = cluster.spectral_clustering(speech_graph_copy_trimmed, n_clusters=n_clusters)
plt.hist(spectral_labels, bins=10)
print [list(spectral_labels).count(i) for i in range(n_clusters)]
print [(spectral_labels[remaining.index(i)],i) for i in actual_debates if i in remaining_set]


# In[687]:

## Fix indices by hand..

actual_debate_to_others[14] = []
actual_debate_to_others[43] = []
actual_debate_to_others[181] = []

for l,p in zip(spectral_labels, sorted(remaining)):
    if l == 0:
        actual_debate_to_others[14].append(p)
    elif l == 3:
        actual_debate_to_others[181].append(p)
    elif l == 4:
        actual_debate_to_others[43].append(p)

print len(actual_debate_to_others[14])
print len(actual_debate_to_others[181])
print len(actual_debate_to_others[43])


# In[688]:

print 'Before'

remaining_old = remaining
remaining_set_old = remaining_set
print len(remaining_old), len(remaining_set_old)


remaining_set = remaining_set.difference(set(actual_debate_to_others[14]))
remaining_set = remaining_set.difference(set(actual_debate_to_others[181]))
remaining_set = remaining_set.difference(set(actual_debate_to_others[43]))
remaining = sorted(list(remaining_set))

print 'After'
print len(remaining), len(remaining_set)

print 'Difference'
print len(remaining_set_old) - len(remaining_set)


# ### Cluster 29-32-33, N = 1-80-19

# In[692]:

print len(remaining), len(remaining_set)


# In[691]:

speech_graph_copy, _ = load_sparse_data('speech_graph.csv', NUM_SPEECHES)

# Create affinity matrix.
print 'Creating affinity matrix.'
for row in range(NUM_SPEECHES):
    row_indices = sparse_where(speech_graph, row)
    for index in row_indices:
        # Remove connections to points we have already labeled.
        # Remove connections to debate with the same label.
        if index not in remaining_set or for_against[row] == for_against[index]:
            speech_graph_copy[row, index] = 0.0
    if row % 450 == 0:
        print row * 100.0 / NUM_SPEECHES, '%'

print 'Done'


# In[693]:

speech_graph_copy_trimmed = speech_graph[sorted(remaining),:]
print speech_graph_copy_trimmed.shape
speech_graph_copy_trimmed = speech_graph_copy_trimmed[:,sorted(remaining)]
print speech_graph_copy_trimmed.shape


# In[698]:

n_clusters = 6
spectral_labels = cluster.spectral_clustering(speech_graph_copy_trimmed, n_clusters=n_clusters)
plt.hist(spectral_labels, bins=10)
print [list(spectral_labels).count(i) for i in range(n_clusters)]
print [(spectral_labels[remaining.index(i)],i) for i in actual_debates if i in remaining_set]


# In[699]:

## Fix indices by hand..

actual_debate_to_others[80] = []
actual_debate_to_others[19] = []
actual_debate_to_others[1] = []

for l,p in zip(spectral_labels, sorted(remaining)):
    if l == 0:
        actual_debate_to_others[80].append(p)
    elif l == 2:
        actual_debate_to_others[1].append(p)
    elif l == 3:
        actual_debate_to_others[19].append(p)

print len(actual_debate_to_others[80])
print len(actual_debate_to_others[19])
print len(actual_debate_to_others[1])


# In[700]:

print 'Before'

remaining_old = remaining
remaining_set_old = remaining_set
print len(remaining_old), len(remaining_set_old)


remaining_set = remaining_set.difference(set(actual_debate_to_others[80]))
remaining_set = remaining_set.difference(set(actual_debate_to_others[1]))
remaining_set = remaining_set.difference(set(actual_debate_to_others[19]))
remaining = sorted(list(remaining_set))

print 'After'
print len(remaining), len(remaining_set)

print 'Difference'
print len(remaining_set_old) - len(remaining_set)


# In[ ]:

### Cluster -, N = -


# In[703]:

print len(remaining), len(remaining_set)


# In[704]:

speech_graph_copy, _ = load_sparse_data('speech_graph.csv', NUM_SPEECHES)

# Create affinity matrix.
print 'Creating affinity matrix.'
for row in range(NUM_SPEECHES):
    row_indices = sparse_where(speech_graph, row)
    for index in row_indices:
        # Remove connections to points we have already labeled.
        # Remove connections to debate with the same label.
        if index not in remaining_set or for_against[row] == for_against[index]:
            speech_graph_copy[row, index] = 0.0
    if row % 450 == 0:
        print row * 100.0 / NUM_SPEECHES, '%'

print 'Done'


# In[705]:

speech_graph_copy_trimmed = speech_graph[sorted(remaining),:]
print speech_graph_copy_trimmed.shape
speech_graph_copy_trimmed = speech_graph_copy_trimmed[:,sorted(remaining)]
print speech_graph_copy_trimmed.shape


# In[713]:

n_clusters = 3
spectral_labels = cluster.spectral_clustering(speech_graph_copy_trimmed, n_clusters=n_clusters)
plt.hist(spectral_labels, bins=10)
print [list(spectral_labels).count(i) for i in range(n_clusters)]
print [(spectral_labels[remaining.index(i)],i) for i in actual_debates if i in remaining_set]


# In[714]:

temp_s = [i for i in remaining if spectral_labels[remaining.index(i)] != 1]


# In[715]:

speech_graph_copy_trimmed1 = speech_graph[sorted(temp_s),:]
print speech_graph_copy_trimmed1.shape
speech_graph_copy_trimmed1 = speech_graph_copy_trimmed1[:,sorted(temp_s)]
print speech_graph_copy_trimmed1.shape


# In[719]:

n_clusters = 3
spectral_labels1 = cluster.spectral_clustering(speech_graph_copy_trimmed1, n_clusters=n_clusters)
plt.hist(spectral_labels1, bins=10)
print [list(spectral_labels1).count(i) for i in range(n_clusters)]
print [(spectral_labels1[temp_s.index(i)],i) for i in actual_debates if i in remaining_set and i in temp_s]


# In[720]:

## Fix indices by hand..

actual_debate_to_others[66] = []
actual_debate_to_others[63] = []
actual_debate_to_others[498] = []
actual_debate_to_others[183] = []

for l,p in zip(spectral_labels, sorted(remaining)):
    if l == 1:
        actual_debate_to_others[66].append(p)


for l,p in zip(spectral_labels1, sorted(temp_s)):
    if l == 0:
        actual_debate_to_others[183].append(p)
    elif l == 1:
        actual_debate_to_others[63].append(p)
    elif l == 2:
        actual_debate_to_others[498].append(p)

print len(actual_debate_to_others[66])
print len(actual_debate_to_others[63])
print len(actual_debate_to_others[498])
print len(actual_debate_to_others[183])


# ### Create labeling

# In[721]:

create_labeling(rows_over_635_trimmed, big_3_labels, actual_debates, actual_debate_to_others, revert_dict)


# In[9]:

def create_labeling(rows_over_635_trimmed, big_3_labels, actual_debates, actual_debate_to_others, revert_dict):
    
    d = collections.defaultdict(lambda:-1)
    
    # label largest 3
    big_counts = [(i,list(big_3_labels).count(i)) for i in range(3)]
    big_counts.sort(key=lambda t: t[1], reverse=True)
    print big_counts
    for i in range(3):
        for r,l in zip(rows_over_635_trimmed,big_3_labels):
            if l == big_counts[i][0]:
                d[r] = i
    
    for k in actual_debate_to_others:
        for l in actual_debate_to_others[k]:
            d[l] = actual_debates[k]
    
    for k in revert_dict:
        d[k] = revert_dict[k]
    
    for k in actual_debates:
        d[k] = actual_debates[k]
    
    print len(d)
    
    write_evaluation([(i, d[i]) for i in range(NUM_SPEECHES)], 'hacky1.csv')


# In[478]:

##
# Revert
##

actual_debate_to_others = collections.defaultdict(list)

header = True
revert_dict = {}
for line in open('hacky1.csv'):
    if header:
        header = False
        continue
    parts = line.strip().split(',')
    if parts[1] != '-1':
        revert_dict[int(parts[0])] = int(parts[1])
    
print len(set(revert_dict.keys()))

finished = set([0,1,2,3,4,5,6,10,11,12,13,14,15])

remaining_set = set(range(NUM_SPEECHES))
remaining_set = remaining_set.difference(set(revert_dict.keys()))
remaining_set = remaining_set.union([k for k in actual_debates if actual_debates[k] not in finished])

remaining = sorted(list(remaining_set))

print len(remaining_set), len(remaining)


# ## Tweak labels

# In[ ]:

def load_clustering(fname):
    
    return cluster_dict, cluster_inv_idx


# In[ ]:




# In[ ]:



