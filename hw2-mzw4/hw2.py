import random, math, sys
import numpy as np
import pylab

from numpy import linalg
from sklearn.cluster import spectral_clustering
import matplotlib.pyplot as plt

# ======================= Kmeans =======================

def plot_kmeans(data, centroids, initial_centroids, assignments):
  # separate clusters
  clusters = [[] for i in range(len(centroids))]
  for (x, y), c in assignments.iteritems():
    clusters[c].append((x, y))

  # plot clusters
  for i, cluster in enumerate(clusters):
    color = 'g' if i == 0 else 'r'
    pylab.plot([x[0] for x in cluster], [x[1] for x in cluster],  color + 'o')
  
  # plot centroids
  pylab.plot([x for (x, y) in centroids], [y for (x, y) in centroids],  'b+', ms=10.0, label='Final centroids')

  # plot initial centroids
  pylab.plot([x for (x, y) in initial_centroids], [y for (x, y) in initial_centroids],  'g+', ms=10.0, label='Initial centroids')

  pylab.ylim([-2,7])
  pylab.xlim([-2,6])
  pylab.legend(loc='upper right')
  pylab.show()


def distance((x1, y1), (x2, y2)):
  return math.sqrt(abs(x1-x2)**2 + abs(y1-y2)**2)

"""
Perform k-means
Centroids given as (c, x, y)
"""
def kmeans(data, k = 2, centroids = []):
  _convergence_threshold = 0.01
  num_points = len(data)

  # if no given centroids, pick random ones from dataset
  if not centroids:
    centroids = []
    for i in range(k):
      index = int(random.random() * num_points)
      centroids.append((data[index][0], data[index][1]))
  print "initial centroids: " + str(centroids)

  initial_centroids = centroids

  assignments = {}
  converged = False
  while not converged:
    # assign step
    for i, (x, y) in enumerate(data):
      # get cluster with min distance to the point
      assignments[(x, y)] = min([(c, distance((x, y), (cx, cy))) for c, (cx, cy) in enumerate(centroids)], key=lambda a: a[1])[0]

    # move centroid step
    new_centroids = [0]*k
    new_centroids2 = [0]*k

    for c, (cx, cy) in enumerate(centroids):
      sumx = 0.0
      sumy = 0.0
      assign_count = 0
      for (x, y) in data:
        if assignments[(x, y)] == c:
          sumx += x
          sumy += y
          assign_count += 1
      new_centroids[c] = (sumx/assign_count, sumy/assign_count)

    # checked if algorithm converged by comparing distances of old centroids to new ones
    converged = all(distance(
      (centroids[i][0], centroids[i][1]), (new_centroids[i][0], new_centroids[i][1])
    ) < _convergence_threshold for i in range(k))
    centroids = new_centroids

  # show results
  plot_kmeans(data, centroids, initial_centroids, assignments)
  return assignments

# ======================= Single Link =======================

def plot_singlelink(clusters):
  # plot clusters
  for i, cluster in enumerate(clusters):
    color = 'g' if i == 0 else 'r'
    pylab.plot([x[0] for x in cluster], [x[1] for x in cluster],  color + 'o')
  pylab.show()

def singlelink(data, k = 2):
  def get_min_dist(c1, c2):
    min_dist = sys.maxint
    for p1 in c1:
      for p2 in c2:
        min_dist = min(distance(p1, p2), min_dist)
    return min_dist

  num_points = len(data)
  clusters = [[(x, y)] for (x, y) in data]

  while len(clusters) != k:
    # find min dist cluster pair
    min_dist = sys.maxint
    min_dist_pair = 0, 0
    for i in range(len(clusters)):
      c1 = clusters[i]
      for j in range(i+1, len(clusters)):
        c2 = clusters[j]
        dist = get_min_dist(c1, c2)
        if dist < min_dist:
          min_dist = dist
          min_dist_pair = i, j

    # combine clusters
    clusters[min_dist_pair[0]] = clusters[min_dist_pair[0]] + clusters[min_dist_pair[1]]
    del clusters[min_dist_pair[1]]

  plot_singlelink(clusters)
  
  assignments = {}
  for i, c in enumerate(clusters):
    for pt in c:
      assignments[pt] = i
  return assignments

def clustering_difference(assignments1, assignments2):
  assert len(assignments1) <= len(assignments2)
  diff = 0.0
  for pt, c in assignments2.iteritems():
    if pt in assignments1 and c != assignments1[pt]: diff += 1
  return diff/len(assignments1)

# ======================= Spectral Clustering =======================

def distance((x, y), (x2, y2)):
  return math.sqrt(abs(x-x2)**2 + abs(y-y2)**2)

def get_similarities(data):
  A = [[0 for i in range(len(data))] for i in range(len(data))]

  for i in range(len(data)):
    (x, y) = data[i]
    for j in range(i, len(data)):
      (x2, y2) = data[j]
      sim = math.exp(-distance((x, y), (x2, y2)))
      A[i][j] = sim
      A[j][i] = sim

  return A

def check_symmetrical(A):
  for i in range(len(A)):
    for j in range(len(A[0])):
      assert A[i][j] == A[j][i]

def plot_spectral(data, assignments, num_clusters):
  clusters = [[] for i in range(num_clusters)]
  for i, c in enumerate(assignments):
    clusters[c].append(data[i])

  for i, cluster in enumerate(clusters):
    color = 'r' if i == 0 else 'g'
    pylab.plot([x for (x, y) in cluster], [y for (x, y) in cluster],  color + 'o')
  
  pylab.ylim([-1,2])
  pylab.xlim([-1,16])
  pylab.show()

def spectral_label_diffs(l1, l2):
  assert len(l1) == len(l2)
  diffs = 0.0
  for i, label in enumerate(l1):
    if label != l2[i]: diffs += 1
  return diffs/len(l1)

# ====================================================
#                         Main
# ====================================================

# =============== Kmeans ===============

# generate k means datasets
centroids = [[2.2, 2.4], [1.8, 2.6]]

x_kmeans_i = [[i%5, i/5] for i in range(30)]
x_kmeans_ii = [[i%5, i/5] for i in range(30)]
x_kmeans_ii += [[1.9, 4.5], [2.1, 4.5]]

# test his files
# with open('../HW 2 - JAC/XkmeansI.csv') as datafile:
#   x_kmeans_i = []
#   for line in datafile:
#     x_kmeans_i.append( map(lambda num: float(num), line.split(',')) )
# with open('../HW 2 - JAC/XkmeansII.csv') as datafile:
#   x_kmeans_ii = []
#   for line in datafile:
#     x_kmeans_ii.append( map(lambda num: float(num), line.split(',')) )

# with open('../HW 2 - JAC/means.csv') as datafile:
#   centroids = []
#   for line in datafile:
#     centroids.append( map(lambda num: float(num), line.split(',')) )

# run kmeans
a1 = kmeans(x_kmeans_i, centroids=centroids)
a2 = kmeans(x_kmeans_ii, centroids=centroids)

print 'Clustering differences: %f' % clustering_difference(a1, a2)

# # =============== Singlelink ===============

x_sl_i = [[i**1.1, 0] for i in range(31)]
del x_sl_i[15]

x_sl_ii = [[i**1.1, 0] for i in range(31)]

# test his files
# with open('../HW 2 - JAC/XslinkI.csv') as datafile:
#   x_sl_i = []
#   for line in datafile:
#     x_sl_i.append( map(lambda num: float(num), line.split(',')) )

# with open('../HW 2 - JAC/XslinkII.csv') as datafile:
#   x_sl_ii = []
#   for line in datafile:
#     x_sl_ii.append( map(lambda num: float(num), line.split(',')) )

# run single link
a1 = singlelink(x_sl_i)
a2 = singlelink(x_sl_ii)

print 'Clustering differences: %f' % clustering_difference(a1, a2)

# =============== Spectral ===============

# Matrix 1 : Every point has two edges coming out of it except
# one point, which only has 1 edge
A_i = np.array([[0 for i in range(30)] for i in range(30)])
for i in range(len(A_i)):
  for k in range(1, 3):
    j = (i+k) % len(A_i)
    if i != 0 and j != 0:
      A_i[i][j] = 1
      A_i[j][i] = 1
  if i == 0:
    A_i[i][i+1] = 1
    A_i[i+1][i] = 1
check_symmetrical(A_i)

# Matrix 1 : Every point has two edges coming out of it
A_ii = np.array([[0 for i in range(30)] for i in range(30)])
for i in range(len(A_ii)):
  for k in range(1, 3):
    j = (i+k) % len(A_ii)
    A_ii[i][j] = 1
    A_ii[j][i] = 1
check_symmetrical(A_ii)

# test his files
# with open('../HW 2 - JAC/AspectralI.csv') as datafile:
#   A_i = []
#   for line in datafile:
#     A_i.append( map(lambda num: float(num), line.split(',')) )
# A_i = np.array(A_i)

# with open('../HW 2 - JAC/AspectralII.csv') as datafile:
#   A_ii = []
#   for line in datafile:
#     A_ii.append( map(lambda num: float(num), line.split(',')) )
# A_ii = np.array(A_ii)

# run spectral clustering
labels_i = spectral_clustering(A_i, n_clusters=2, eigen_solver='arpack')
print labels_i

labels_ii = spectral_clustering(A_ii, n_clusters=2, eigen_solver='arpack')
print labels_ii

print spectral_label_diffs(labels_i, labels_ii)


