#from __future__ import print_function
from __future__ import division
from IPython import get_ipython
# coding: utf-8

# # CS 4786 HW 1

# In[3]:

import numpy as np
import math
#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import random
import scipy.linalg # http://stackoverflow.com/questions/8728732/what-is-wrong-with-importing-modules-in-scipy-is-it-a-bug


# In[4]:

from io import BytesIO
from IPython.core import display
from PIL import Image




# In[5]:

import os
print("Place data files in '"+os.getcwd()+"/'")


# ## Question 1

# ### Problem 1

# In[6]:

def csv_to_matrix(csv_file):
    return np.loadtxt(open(csv_file),delimiter=",")


# In[7]:

# Make the plot
XI = csv_to_matrix('a1_data/smiley/2d-gaussian.csv')
XII = csv_to_matrix('a1_data/smiley/2d-gaussian-rotated.csv')
plt.scatter(XI.T[0],XI.T[1], c='red', alpha=0.9, marker="+", label="X_I")
plt.scatter(XII.T[0],XII.T[1], c='green', alpha=0.9, marker="x", label="X_II")
legend = plt.legend(loc='upper right')
plt.show()


# ### Problem2

# In[42]:

def PCA(data, k=None):
    """
    !Modified from .py file from lecture page!
    Arugments:
        X: numpy array (n x d)
        k : Number of eigenvalues to k (default is all)
    Returns:
        Y: The modified data
        W: The projection matrix (d x min(k, n_eigs))
        centered: The centered data
    """
    covmat = np.cov(data.T,bias=1)
    # numpy wants the rows to be the features ("variables"),
    # and the columns to be the observations ("instances")
    # 'bias=1' means the normalization is by n.
    print 'covariance matrix:\n', covmat

    evals,evecs = np.linalg.eig(covmat)
    # Sort the eigenvectors by decreasing eigenvalues.
    indices = evals.argsort()[::-1]
    evals = evals[indices]
    evecs = evecs[:,indices]
    print 'eigenvalues:\n', evals
    print 'eigenvectors (as columns) :\n', evecs
    
    #testing
    evec1 = evecs[:,0]
    print 'length of claimed first eigenvector is:', np.linalg.norm(evec1)
    print "checking that the first and second eigenvector are orthogonal; here's their dotproduct: ", np.dot(evec1,evecs[:,1])
    #print "should give back the first eigenvector evecs[:,0]", np.dot(covmat,evecs[:,0])/evals[0]
    ###

    mean = data.mean(axis=0) #need axis or get single mean of all entries
    centered = data - mean
    k = min(len(evecs), k) if k is not None else None
    W = evecs.T[0:k]
    Y = np.dot(W, centered.T) 
    return Y, W, mean


# In[94]:

def PCA_SVD(data, k=None):
    """
    Arugments:
        X: numpy array (n x d)
        k : Number of eigenvalues to k (default is all)
    Returns:
        Y: The modified data
        W: The projection matrix (d x min(k, n_eigs))
        centered: The centered data
    """
    _, svals, Vt = linalg.svd(data, full_matrices=False)
    V = Vt.T
    
    print ("% info captured", sum(svals[:20]/sum(svals)) )

    mean = data.mean(axis=0) #need axis or get single mean of all entries
    centered = data - mean
    k = min(len(V), k) if k is not None else None
    W = V.T[0:k]
    Y = np.dot(W, centered.T) 
    return Y, W, mean


# In[146]:

YI,WI,UI = PCA(XI)
YII,WII,UII = PCA(XII)


# In[147]:

# Make the plot
plt.scatter(YI[0],YI[1], c='red', alpha=0.9, marker="+", label="Y_I")
plt.scatter(YII[0],YII[1], c='green', alpha=0.9, marker="x", label="Y_II")
plt.legend(loc='upper right')
plt.grid('on')
plt.show()


# In[9]:

A = np.array([[.7071068,.7071068],
              [.56,.1]])
XI_A = XI.dot(A.T) #check this


# In[10]:

# Make the plot
plt.scatter(XI.T[0],XI.T[1], c='red', alpha=0.9, marker="+", label="X_I")
plt.scatter(XII.T[0],XII.T[1], c='green', alpha=0.9, marker="x", label="X_II")
plt.scatter(XI_A.T[0],XI_A.T[1], c='blue', alpha=0.4, marker="o", label="XI_A")
legend = plt.legend(loc='upper right')
plt.grid('on')
plt.show()


# In[11]:

YIII, WIII, UIII = PCA(XI_A)


# In[12]:

# Make the plot
plt.scatter(YI[0],YI[1], c='red', alpha=0.9, marker="+", label="Y_I")
plt.scatter(YII[0],YII[1], c='green', alpha=0.9, marker="x", label="Y_II")
plt.scatter(YIII[0],YIII[1], c='blue', alpha=0.9, marker="o", label="Y_III")

plt.legend(loc='upper right')
plt.grid('on')
plt.show()


# ## Problem 1.4 : Making smileys

# In[140]:

#Loading Data
smiles  = csv_to_matrix('a1_data/smiley/X_smilie.csv')
Y_cubes = csv_to_matrix('a1_data/smiley/cubist_email.csv')


# In[141]:

def draw_smile(X):
    # Making an image from a matrix
    width = X.shape[1]
    height = X.shape[0]
    img = Image.new( 'RGB', (width,height), "black") # new black image
    pixels = img.load() # create the pixel map
    for i in range(width):    # for every pixel:
        for j in range(height):
            rgb_val = int(X[i][j])
            pixels[i,j] = (rgb_val, rgb_val, rgb_val) # set the colour
    img.show()


# In[142]:

# Building smile matrix
X_smiles = smiles.reshape(28,105,105)
draw_smile(X_smiles[0])


# In[145]:
# Run SVD PCA on all smiles
from sklearn import decomposition
skpca = decomposition.PCA(n_components=20)
skpca.fit(smiles)
Y_sci = skpca.transform(smiles)

# Build sign array
A = np.sign(Y_cubes[0])
B = np.sign(Y_sci[0])
S = np.multiply(A,B)

# Restore Cubed Smiles
Y_signed_cubes = [np.multiply(S, Y_cube) for Y_cube in Y_cubes]
Y_restored_cubes = [skpca.inverse_transform(Y_signed_cube).reshape((105,105)) for Y_signed_cube in Y_signed_cubes]
draw_smile(Y_restored_cubes[0])