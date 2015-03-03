# pcaEx2.py
# CS 4786, Profs Lillian Lee and Karthik Sridharan
# Feb 2015


'''Extended version of an example provided in class on Jan 29th for the steps of computing the PCA
 and understanding what information you (sometimes) get from the covariance matrix
 and the principal components.  Real-life data might not work out this nicely,
 but we just wanted to give you a flavor of the possibilities. '''


import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.linalg # http://stackoverflow.com/questions/8728732/what-is-wrong-with-importing-modules-in-scipy-is-it-a-bug

# generate a data set of two types of vectors, one where the odd-indexed
# items tend to be bigger than the even-indexed items, and vice versa.
halfn=3
numreps=2
baselength=2
mu1 = np.array([5,7]*numreps)
mu2 = np.array([7,5]*numreps)
cov = np.diagflat([.1]*(baselength*numreps))
odds = np.random.multivariate_normal(mu1, cov, halfn)
evens = np.random.multivariate_normal(mu2, cov, halfn)

data = np.concatenate((odds, evens)) # hard-to-see bug: input is a tuple
# np.random.shuffle(data)

assert data.shape == (2*halfn, baselength*numreps), 'unexpected shape for the matrix called data'

#@ trying to get a meaningful second component ...
# data = np.array([[ 4.95510801,  6.52139712,  5.25025749,  7.02239356,.1],
#  [ 4.4823309 , 6.84457706,  5.19244623,  6.5368952, .1],
#  [ 5.05132672, 6.97652322,  4.98007815,  6.9394447, .1 ],
#  [ 7.08887047, 4.91242484,  6.16726225,  5.08374745, 0],
#  [ 7.13911236, 5.29051298,  6.73452172,  5.06396644, 0],
#  [ 6.83552334, 4.43167221,  7.19899121,  4.73849215, 0]])


print 'data:\n', data, '\n..............'
raw_input("...pause for contemplation")
mean = data.mean(axis=0) # need axis or get single mean of all entries


# numpy doesn't need this to compute the covariance matrix;
# computing it here just so students can see it.
centered = data - mean
## students: uncomment the code below if you want to see the centered data
# print 'centered data:\n', centered
# raw_input("...pause for contemplation")

covmat = np.cov(data.T,bias=1)
# numpy wants the rows to be the features ("variables"),
# and the columns to be the observations ("instances")
# 'bias=1' means the normalization is by n.

print 'covariance matrix (in pcaEx2.covmat):\n', covmat
raw_input("...pause for contemplation")

evals,evecs = np.linalg.eig(covmat)

# Sort the eigenvectors by decreasing eigenvalues.
# One source: http://stackoverflow.com/questions/8092920/sort-eigenvalues-and-associated-eigenvectors-after-using-numpy-linalg-eig-in-pyt
indices = evals.argsort()[::-1]
evals = evals[indices]
evecs = evecs[:,indices]


print 'eigenvalues (in pcaEx2.evals):\n', evals
print 'eigenvectors (as columns) (in pcaEx2.evecs):\n', evecs
evec1 = evecs[:,0]

print 'length of claimed first eigenvector is:', np.linalg.norm(evec1)
#print 'length of first row is:', np.linalg.norm(evecs[0,:])
# print 'length of 2nd column is:', np.linalg.norm(evecs[:,1])
# print 'length of 2nd column is:', np.linalg.norm(evecs[1,:])
print "checking that the first and second eigenvector are orthogonal; here's their dotproduct: ", np.dot(evec1,evecs[:,1])

# np.dot(covmat,evecs[:,0])/evals[0] should
# give back the first eigenvector evecs[:,0]


W = evecs.T
print 'Y for 2-d projection:\n', np.dot(W[0:2,:], centered.T)


