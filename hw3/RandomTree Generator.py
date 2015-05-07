import random
import math
import numpy as np

# Set variables
pi_1 = .7 #pi_2 = 1-pi_1
mu_1 = (0,0)
mu_2 = (0,0)
sigma = 1
steps = 10

#create cov_mat
R = 1 #in R^2
sigma2 = sigma**2
cov = np.diag([sigma2,sigma2])


def pi():
    if pi_1 < random.random():
        return 'R'
    else:
        return 'G'
    
def X(mu):
    return np.random.multivariate_normal(mu, cov, R)[0]



tree_locs = []
tree_obs = []
prev_r_mu = mu_1
prev_g_mu = mu_2
for step in range(steps):
    tree_type = pi()
    tree_loc = X(prev_r_mu) if tree_type=='R' else X(prev_g_mu)
    prev_r_mu = tree_loc if tree_type=='R' else prev_r_mu
    prev_g_mu = tree_loc if tree_type=='G' else prev_g_mu
    tree_obs.append(tree_type)
    tree_locs.append(tree_loc)
print "Done"
print "locs: ", tree_locs
print "obs: ", tree_obs