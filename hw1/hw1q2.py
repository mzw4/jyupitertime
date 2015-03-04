import math, time
import numpy as np
import pylab
from numpy import linalg
from sklearn import random_projection

_output_dir = 'output_data/'

# ============================= Functions =============================

"""
Calculates the projection matrix W
given a matrix X and dimension K
"""
def get_W(X, K):
  # Calculate eigenvalues
  eig_vals, eig_vecs = np.linalg.eig(X)

  # Sort by eigenvalue and take top K eigenvectors
  indices = eig_vals.argsort()[::-1][:K]
  # take eigenvectors as rows of W, like in the cheatsheet
  W = eig_vecs[:,indices].T
  return W

"""
Principal component analysis
Arugments:
    X: numpy array (n x d)
    k : Number of eigenvalues to k (default is all)
Returns:
    W: The projection matrix (d x min(k, n_eigs))
    cov_mat: The covariance matrix (d x d)
"""
def PCA(X, K=float('inf')):
    mean = X.mean(axis=0) # need axis or get single mean of all entries
    centered = X - mean

    # numpy wants rows to be features and columns to be points
    X = X.T
    d, n = X.shape
    K = min(K, d)

    # Calculate covariance matrix
    cov_mat = np.cov(X, bias=1)

    W = get_W(cov_mat, K)
    Y = W.dot(X)

    assert W.shape == (K, d)
    assert Y.shape == (K, n)

    # return Y with rows as points and columns as features
    return Y.T, W

"""
CCA wooo
"""
def CCA(v1, v2, K):
  # want data matrix to be d x n
  X = np.hstack((v1, v2)).T
  v1_size = v1.shape[1]

  # parse joint covariance matrices
  joint_cov = np.cov(X, bias=1)
  c_11 = joint_cov[:v1_size, :v1_size]
  c_12 = joint_cov[:v1_size, v1_size:]
  c_21 = joint_cov[v1_size:, :v1_size]
  c_22 = joint_cov[v1_size:, v1_size:]

  inv_c_11 = linalg.inv(c_11)
  inv_c_22 = linalg.inv(c_22)

  # construct W
  W_joint = inv_c_11.dot(c_12).dot(inv_c_22).dot(c_21)
  W = get_W(W_joint, K)

  # construct V
  V_joint = inv_c_22.dot(c_21).dot(inv_c_11).dot(c_12)
  V = get_W(V_joint, K)

  Y_v1 = W.dot(v1.T)
  Y_v2 = V.dot(v2.T)

  assert Y_v1.shape == Y_v2.shape == (K, v1.shape[0])
  assert W.shape == (K, v1_size)
  assert V.shape == (K, X.shape[0] - v1_size)

  return Y_v1.T, W, Y_v2.T, V

"""
Guaussian random projection
Default K is automatically chosen so that epsilon = 0.1
"""
def rand_projection(X, K):
  transformer = random_projection.GaussianRandomProjection(n_components=K)
  Y = transformer.fit_transform(X)
  return Y

"""
Generate a 100x1000 dataset with norm = 1
"""
def generate_data(n, d, type='uniform', normalize=True, K=None):
  print '\nGenerating "' + type + '" dataset...'
  if type == 'uniform':
    X = np.random.rand(n, d)
  elif type == 'normal':
    X = np.random.normal(size=(n, d))
  elif type == 'custom_pca' and K:
    X = np.hstack((np.random.rand(n, K), np.zeros((n, d-K))))
  elif type == 'custom_pca_spread' and K:
    X = np.hstack((np.random.rand(n, K/2), np.zeros((n, d-K)), np.random.rand(n, K/2)))
  elif type == 'identity':
    X = np.hstack(( np.identity(n), np.zeros((n, d)) ))
  elif type == 'max_spread' and K:
    X = np.hstack((np.identity(K), np.zeros((K, d-K))))
    for i in range(4):
      X = np.vstack((X, (np.hstack((np.identity(K), np.zeros((K, d-K)))))))
  elif type == 'max_spread2':
    X = np.vstack((np.array([[1 if i == 0 else 0 for i in range(d)] for j in range(n/2)] ), np.array([[1 if i == 1 else 0 for i in range(d)] for j in range(n/2)] )))
  elif type == 'sparse':
    X = np.hstack((np.random.rand(n, 1), np.zeros((n, d-1))))
  elif type == 'half_covary':
    # generate vectors of n numbers with mean=0 and var=1
    # concatenate so half features increase monotonically and half decrease monotonically

    # increasing = np.sort(np.random.normal(loc=0, scale=1, size=(d/2, n))).T
    # decreasing = np.fliplr(np.sort(np.random.normal(loc=0, scale=1, size=(d/2, n)))).T
    
    # separated
    # increasing = np.random.normal(loc=0, scale=1, size=(d/2, n)).T
    # decreasing = np.random.normal(loc=0, scale=1, size=(d/2, n)).T


    increasing = np.random.normal(loc=0, scale=1, size=(d/2, n)).T
    decreasing = np.fliplr(increasing * -1)

    # wrong
    # increasing = np.sort(np.random.normal(loc=0, scale=1, size=(n, d/2)))
    # decreasing = np.fliplr(np.sort(np.random.normal(loc=0, scale=1, size=(n, d/2))))

    X = np.hstack((increasing, decreasing))
    # for i in range(0, X.shape[1]/2 + 2, 2):
    #   X[:, i] = X[:, (i+1)%50] + X[:, (i+3)%100] - X[:, (i+5)%100]

    # for i in range(X.shape[1]/2, X.shape[1], 2):
    #   X[:, i] = X[:, (i+1)%100] + X[:, (i+3)%100] - X[:, (i+5)%100]

    # for i in range(0, X.shape[1], 2):
    #   X[:, i] = sum(X[:, (i+j)%100] for j in range(50)) - sum(X[:, (i+j)%100] for j in range(50, 50, 1))

    X[:, :50] = np.sort(X[:, :50].T).T
    X[:, 50:] = np.fliplr(np.sort(X[:, 50:].T)).T

    print X
  else:
    print 'Invalid dataset type'
    return None

  if normalize == True:
    # Normalize and check (Err is ~0)
    X = np.array([x_i / linalg.norm(x_i) for x_i in X])
    assert(Err(X) < 1e-4)

  return X

"""
Finds the error by taking average vector norm difference from 1
Y is an n x K matrix
"""
def Err(Y):
  # formula given in writeup
  err = sum(abs(linalg.norm(y_i) - 1) for y_i in Y) / Y.shape[0]
  return err

"""
Run projections and compare errors for random projection and PCA
"""
def compare_pca_rp(X, K):
  Y_rp = rand_projection(X, K)
  Y_pca, _ = PCA(X, K)

  print 'Random projection err: ' + str(Err(Y_rp))
  print 'PCA err: ' + str(Err(Y_pca))

"""
Perform CCA and plot in 1 dimension
"""
def CCA_plot(v1, v2, K):
  Y_v1, W, Y_v2, V = CCA(v1, v2, K)

  # view 1
  pylab.plot(Y_v1[:500], [0] * (v1.shape[0]/2), 'g+', label='Projected view 1, first 500')
  pylab.plot(Y_v1[500:], [0] * (v1.shape[0]/2), 'r+', label='Projected view 1, last 500')
  pylab.legend(loc='upper right')
  pylab.show()

  pylab.plot(Y_v2[:500], [0] * (v2.shape[0]/2), 'g+', label='Projected view 2, first 500')
  pylab.plot(Y_v2[500:], [0] * (v2.shape[0]/2), 'r+', label='Projected view 2, second 500')
  pylab.legend(loc='upper right')
  pylab.show()

def output_data(X, fname):
  with open(fname, 'w') as datafile:
    content = ''
    for x_t in X:
      for i, f in enumerate(x_t):
        content += str(f) + (',' if i != len(x_t)-1 else '')
      content += '\n'
    datafile.write(content)

# ============================= Main =============================

# 1
def part1():
  n = 1000
  d = 100
  K = 1

  # this generates a dataset where, for each data vector, the first 50 features
  # are monotonically increasing and the last 50 are decreasing. This creates
  # the maximum inverse correlation between the first and last features.
  # The value of every feature is selected from a normal distribution with
  # mean=0 and var=1.
  # Each data vector is normalized for consistency.
  X = generate_data(n, d, 'half_covary', normalize=True)
  output_data(X, _output_dir + 'CcaPca.csv')

  # Take view 1 as first 50 coordinates and view 2 as second 50 coordinates,
  # shows a clear separation between views
  v1 = X[:, :d/2]
  v2 = X[:, d/2:]

  CCA_plot(v1, v2, K)

  # Take view 1 as odd coordinates and view 2 as even,
  # shows no clear separation of views
  v1 = X[:, ::2]  # even columns
  v2 = X[:, 1::2] # odd columns

  CCA_plot(v1, v2, K)

  # Perform PCA with K=2, shows no clear separation between
  # first 500 and last 500 points
  K = 2
  Y, W = PCA(X, K)
  Y1 = Y[:n/2]  # first 500 points
  Y2 = Y[n/2:]  # last 500 points

  Y1_x = [p[0] for p in Y1]
  Y1_y = [p[1] for p in Y1]
  pylab.plot(Y1_x, Y1_y, 'g+')

  Y2_x = [p[0] for p in Y2]
  Y2_y = [p[1] for p in Y2]
  pylab.plot(Y2_x, Y2_y, 'r+')
  pylab.show()

part1()

# --------------------------------------------------------------------
# 2

def part2():
  n = 100
  d = 1000
  K = 20

  # Generates a uniformly distributed, normalized dataset
  # PCA does poorly because each feature has nearly equal variance
  # RP does well because...the matrix is dense?
  X = generate_data(n, d, 'uniform')
  output_data(X, _output_dir + 'RpBeatsPCA.csv')
  compare_pca_rp(X, K)

  # Generates a dataset where the first K features are uniformly distributed
  # and normalized.
  # PCA does well because exactly K features have variance > 0 so all
  # information is essentially retained.
  # RP does poorly because... the matrix is more sparse?
  
  # X = generate_data(n, d, 'custom_pca_spread', K=K)
  X = generate_data(n, d, 'identity')

  output_data(X, _output_dir + 'PcaBeatsRp.csv')
  compare_pca_rp(X, K)

  # failed attempts

  # compare_pca_rp(generate_data(n, d, 'custom_pca', 2), K)
  # compare_pca_rp(generate_data(n, d, 'max_spread', K), K)
  # compare_pca_rp(generate_data(n, d, 'max_spread2'), K)

# part2()
