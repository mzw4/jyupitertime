import numpy as np
import os
import pylab
import random

DATA_DIR = 'smiley'

X1 = []
X2 = []

# Read CSV files
for line in open(os.path.join(DATA_DIR, '2d-gaussian.csv')):
    parts = line.strip().split(',')
    X1.append( [float(parts[0]), float(parts[1])] )
for line in open(os.path.join(DATA_DIR, '2d-gaussian-rotated.csv')):
    parts = line.strip().split(',')
    X2.append( [float(parts[0]), float(parts[1])] )


# Q1
# Scatter plot
X1_x = [p[0] for p in X1]
X1_y = [p[1] for p in X1]
X2_x= [p[0] for p in X2]
X2_y = [p[1] for p in X2]

# pylab.plot(X1_x, X1_y, 'gx')
# pylab.plot(X2_x, X2_y, 'rx')
# pylab.show()



# Q2
# PCA

def PCA(X, k=float('inf')):
    """
    Arugments:
        X: numpy array (n x d)
        k : Number of eigenvalues to k (default is all)

    Returns:
        W: The projection matrix (d x min(k, n_eigs))
        cov_mat: The covariance matrix (d x d)
    """
    n = X.shape[0]
    d = X.shape[1]
    
    # Calculate covariance matrix
    split_dims = [X[:,i] for i in range(d)]
    cov_mat = np.cov(split_dims)

    # Calculate eigenvalues
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    n_eigs = len(eig_vals)
    eig_pairs = [ (np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(n_eigs) ]
    eig_pairs.sort()
    eig_pairs.reverse()
    eig_pairs = eig_pairs[0:min(k,n_eigs)]

    W = np.hstack( (eig_pairs[i][1].reshape(d,1) for i in range(min(k,d))) )

    assert W.shape[0] == d
    assert W.shape[1] == min(k, n_eigs)

    return W, cov_mat

X1 = np.array(X1)
X2 = np.array(X2)

W1, X1_cov = PCA(X1)
print 'W1:\n', W1
print 'X1 cov:\n', X1_cov

W2, X2_cov = PCA(X2)
print 'W2:\n', W2
print 'X2 cov:\n', X2_cov

Y1 = X1.dot(W1)
Y2 = X2.dot(W2)

assert(np.allclose(np.absolute(Y1),np.absolute(Y2), rtol=1e-4))

A = [[0.7071068, 0.7071068],[0.56, 0.1]]
A = [[0,-1],[1,0]]
A = np.array(A)

# Tranpose because our points are rows, not columns.
X3 = X1.dot(A.T)
W3, _ = PCA(X3)
Y3 = X3.dot(W3)

assert(np.allclose(np.absolute(Y1),np.absolute(Y3), rtol=1e-4))
#assert(not np.allclose(np.absolute(Y1),np.absolute(Y3), rtol=1e-4))

Y1_x = [p[0] for p in Y1]
Y1_y = [p[1] for p in Y1]
Y3_x= [p[0] for p in Y3]
Y3_y = [p[1] for p in Y3]

X1_x = [p[0] for p in X1]
X1_y = [p[1] for p in X1]
X3_x= [p[0] for p in X3]
X3_y = [p[1] for p in X3]

pylab.plot(Y1_x, Y1_y, 'go')
pylab.plot(Y3_x, Y3_y, 'ro')

# pylab.plot(X1_x, X1_y, 'gx')
# pylab.plot(X3_x, X3_y, 'rx')

pylab.show()

# Q4
# Read cubist email
email_path = os.path.join('smiley', 'cubist_email.csv')

Y_tilde = []
for line in open(email_path):
    parts = line.strip().split(',')
    floats = [float(part.strip()) for part in parts]
    Y_tilde.append(floats)


