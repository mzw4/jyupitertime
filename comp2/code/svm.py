import sys
import numpy as np
from sys import stdout
from scipy.spatial import distance
from numpy import linalg
from sklearn import svm, grid_search
import matplotlib.pyplot as plt
from sklearn import hmm

NUM_WORDS = 7
NUM_COEFFS = 13
NUM_WINDOWS = 83

_data = {}
_label_data = {}

# ============= Process Data =============

# create example -> info mapping
with open('competition_2/train.data', 'r') as data_file:
    for i, line in enumerate(data_file):
        _data[i] = {}
        _data[i]['data'] = []

        coeffs = line.split(',')
        for k in xrange(0, len(coeffs), NUM_COEFFS):
            _data[i]['data'].append(np.array([float(num) for num in coeffs[k:k+13]]))
    
with open('competition_2/train.labels', 'r') as label_file:
    for i, line in enumerate(label_file):
        if i == 0: continue
        vals = line.split(',')
        pt = int(vals[0])
        label = int(vals[1])
        _data[pt]['label'] = label
        
# create label -> data mapping
for i, val in _data.iteritems():
    label = val['label']
    data = val['data']
    if label not in _label_data:
        _label_data[label] = []
    _label_data[label].append(data)
        
NUM_EXAMPLES = len(_data)
print NUM_EXAMPLES, len(_label_data)
print 'done'

# ============= Perform SVM classification =============

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)

X = []
Y = []
for k, v in _data.iteritems():
    X.append(np.array(v['data']).flatten())
    Y.append(v['label'])
    
print 'fitting...'
clf.fit(X, Y)

# ============= Make predictions =============

test_vectors = []
with open('competition_2/test.data', 'r') as test_data:
    for i, line in enumerate(test_data):
        coeffs = line.split(',')
        test_vectors.append([float(val) for val in coeffs])

print len(test_vectors)
predictions = clf.predict(test_vectors)