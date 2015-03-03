import math, time
import numpy as np
from numpy import linalg
from sklearn import random_projection

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
def CCA(X, v1_size, K):
  # want data matrix to be d x n
  X = X.T
  v1 = X[:, :v1_size]
  v2 = X[:, v1_size:]

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
  W = get_W(W_joint, K).T

  # construct V
  V_joint = inv_c_22.dot(c_21).dot(inv_c_11).dot(c_12)
  V = get_W(V_joint, K).T

  Y_v1 = W.dot(v1)
  Y_v2 = V.dot(v2)
  return Y_v1, W, Y_v2, V

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
  elif type == 'max_spread' and K:
    X = np.hstack((np.identity(K), np.zeros((K, d-K))))
    for i in range(4):
      X = np.vstack((X, (np.hstack((np.identity(K), np.zeros((K, d-K)))))))
  elif type == 'max_spread2':
    X = np.vstack((np.array([[1 if i == 0 else 0 for i in range(d)] for j in range(n/2)] ), np.array([[1 if i == 1 else 0 for i in range(d)] for j in range(n/2)] )))
  elif type == 'sparse':
    X = np.hstack((np.random.rand(n, 1), np.zeros((n, d-1))))
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
def project_data(X, K):
  Y_rp = rand_projection(X, K)
  Y_pca, _ = PCA(X, K)

  print 'Random projection err: ' + str(Err(Y_rp))
  print 'PCA err: ' + str(Err(Y_pca))

# ============================= Main =============================

K = 20
n = 100
d = 1000

# project_data(generate_data(n, d, 'uniform'), K)
project_data(generate_data(n, d, 'custom_pca', K=K), K)
# project_data(generate_data(n, d, 'custom_pca', 2), K)
# project_data(generate_data(n, d, 'max_spread', K), K)
# project_data(generate_data(n, d, 'max_spread2'), K)

X = generate_data(1000, 100, 'normal')
Y_v1, W, Y_v2, V = CCA(X, 50, 50)

