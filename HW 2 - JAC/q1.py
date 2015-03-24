import copy
import math
import networkx as nx
import numpy as np
import json
import pylab
import random

'''
Helper methods.
'''

def dist(p, q):
    """
    Arguments:
        p,q: Points

    Returns:
        Returns distance between the two points.
    """
    dist = (p[0] - q[0])**2
    dist += (p[1] - q[1])**2
    return math.sqrt(dist)


assert dist([0,1], [0,0]) == 1
assert dist([0,-2], [0,0]) == 2
assert dist([1,0], [10,0]) == 9

def graph(A, c, title):
    """
    Arguments:
        A: An adjacency matrix.
        c: Cluster labeling.
        title: The title for the graph
    """

    colors = ['red', 'blue']

    graph = nx.Graph()
    for i in range(len(A)):
        graph.add_node(i, style='filled', fillcolor=colors[c[i]])
    for i in range(len(A)):
        row = A[i]
        for j in range(len(row)):
            if row[j] == 1:
                graph.add_edge(i,j)

    node_list = []
    node_colors = []
    for node_info in graph.nodes(data=True):
        node_list.append(node_info[0])
        node_colors.append(node_info[1]['fillcolor'])

    pos = nx.spring_layout(graph)
    pylab.clf()
    pylab.title(title)
    nx.draw_shell(graph, nodelist=node_list, node_color=node_colors)
    pylab.savefig(title + '.png')
    pylab.show()

def plot(ps, cs, title = '', us = None):
    """
    Arguments:
        ps: The points.
        cs: The cluster assignments.
        title: The title of the plot.
        us: The centroids.

    Returns:
    """
    c1_x = []
    c1_y = []

    c2_x = []
    c2_y = []

    for p,c in zip(ps, cs):
        if c == 1:
            c1_x.append(p[0])
            c1_y.append(p[1])
        else:
            c2_x.append(p[0])
            c2_y.append(p[1])

    pylab.plot(c1_x, c1_y, 'ro', label='Cluster 1')
    pylab.plot(c2_x, c2_y, 'bo', label='Cluster 2')
    if us:
        pylab.plot([u[0] for u in us], [u[1] for u in us], 'gs', label='Centroids')
    pylab.legend()
    pylab.title(title)
    pylab.show()
    pylab.clf()

def save_points(ps, filename):
    '''
    Arguments:
        ps: The points.
        filename: The file to save to.

    Results:

    '''
    f = open(filename, 'w+')
    for p in ps:
        f.write(json.dumps(p)[1:-1] + '\n')
    f.close()


def flip(c):
    '''
    Arguments:
        c: A 0-1 array, ex: [0,1,0,1]

    Returns:
        An array where 0 has been replaced by 1 and 1 has been replaced by 0.
    '''
    return [abs(i-1) for i in c]

def vary_amount(c1, c2, n=30):
    '''
    Calculates the percentage by which the two arrays vary.

    Arguments:
        c1: A 0-1 array
        c2: A 0-1 array
        n: The number of elements to compare in the arrays 
            (starting from the beginning)

    Returns:
        The percentage by which the two arrays vary, in the range (0,1).
    '''
    diff1 = sum([abs(a-b) for a,b in zip(c1[:n], c2[:n])])
    diff2 = sum([abs(a-b) for a,b in zip(flip(c1)[:n], c2[:n])])

    return min(diff1, diff2) / (1.0 * n)


assert flip([1,1,0,0,1]) == [0,0,1,1,0]
assert flip([1,1,1,1,1]) == [0,0,0,0,0]

assert vary_amount( [1,1,1,1], [0,0,0,1], 4 ) == 0.25
assert vary_amount( [1,0,0,1], [0,0,0,0], 4 ) == 0.50


'''
Q 1.1 - K-means
'''

def k_means(ps, us):
    """
    Arguments:
        ps: The points
        us: The centroids.

    Returns:

    """
    u1 = us[0]
    u2 = us[1]

    u1_old = float('inf')
    u2_old = float('inf')

    c1 = None
    c2 = None

    while (u1 != u1_old) or (u2 != u2_old):
        u1_old = u1
        u2_old = u2

        c1 = []
        c2 = []

        for p in ps:
            ud1 = dist(p, u1)
            ud2 = dist(p, u2)

            if ud1 < ud2:
                c1.append(p)
            else:
                c2.append(p)

        n1 = 1.0*len(c1)
        n2 = 1.0*len(c2)
        u1 = [sum([c[0] for c in c1])/n1, sum([c[1] for c in c1])/n1]
        u2 = [sum([c[0] for c in c2])/n2, sum([c[1] for c in c2])/n2]

    c = []
    for p in ps:
        if p in c1:
            c.append(1)
        else:
            c.append(0)
    return c, [u1,u2]

# Construct original matrices
# Two rotated, intersecting half circles.

epsilon = 1.0 / 31
km_1 = []

x0 = 1
y0 = 1
r = 1.0
tilt = 30.0/4
for i in range(0,15):
    x = x0 + r * math.cos(2 * math.pi * (i+tilt) / 30.0)
    y = y0 + r * math.sin(2 * math.pi * (i+tilt) / 30.0)
    km_1.append( [x,y] )

sep = -1.4
x1 = 1 + sep
y1 = 2*r + y0 + sep
for i in range(0,15):
    x = x1 + r * math.cos(2 * math.pi * (i-tilt) / 30.0)
    y = y1 + r * math.sin(2 * math.pi * (i-tilt) / 30.0)
    km_1.append( [x,y] )

# Add new points
uy_1 = y0+r/4

km_2 = copy.deepcopy(km_1)
km_2.append([0.1,0.1])
km_2.append([0.1,2.4])
km_2.append([0.1,0.11])

u1 = [x0-0.1,uy_1]
u2 = [x1+0.1,y1-r/4]
us = [u1, u2]

## 2nd attempt

c1, uk1 = k_means(km_1, us)
c2, uk2 = k_means(km_2,  us)

plot(km_1, c1, title='K-means for XkmeansI.csv', us=us)
plot(km_2, c2, title='K-means for XkmeansII.csv', us=us)

save_points(km_1, 'XkmeansI.csv')
save_points(km_2, 'XkmeansII.csv')
save_points([[c] for c in c1], 'ckmeansI.csv')
save_points([[c] for c in c2[:30]], 'ckmeansII.csv')


# Some assertions
assert len(c1) == 30
assert len(c2) <= 33
assert len(c2) > len(c1)
assert sum(c1) == 15, sum(c1)
assert km_1[:30] == km_2[:30]
assert vary_amount(c1,c2) > 0.30, vary_amount(c1, c2)


'''
Q 1.2 - Single-link
'''

def single_link(m):
    '''
    Computes the single clink clustering for the given points:

    
    Arguments:
        m: A list with shape (n,2)

    Returns:
        A 0-1 vector deliminating the clusters, in the same order as the input.
    '''
    clusters = [[i] for i in m]
    while len(clusters) > 2:
        # Find closest pair
        j1 = None
        j2 = None
        min_dist = float('inf')
        for c1 in clusters:
            for c2 in clusters:
                if c1 != c2:
                    for p in c1:
                        for q in c2:
                            d = dist(p,q)
                            
                            if d < min_dist:
                                j1 = c1
                                j2 = c2
                                min_dist = d

        # Update clustesr
        clusters.remove(j1)
        clusters.remove(j2)
        j1.extend(j2)
        clusters.append(j1)

    c = []
    for p in m:
        if p in clusters[0]:
            c.append(1)
        else:
            c.append(0)

    return c
    

# Construct original matrices
epsilon = 1.0 / 31
sl_1 = []

diff = 1
prev_x = 0
for i in range(0,15):
    next_x = prev_x + diff + i*epsilon
    prev_x = next_x
    sl_1.append( [next_x,1] )

prev_x = -2
for i in range(0,15):
    next_x = prev_x - diff
    next_x -= i*epsilon
    prev_x = next_x
    sl_1.append( [next_x,1] )

# Add new points
sl_2 = copy.deepcopy(sl_1)
sl_2.append([0,1])
sl_2.append([-1,1])
sl_2.append([-2,1])


c1 = single_link(sl_1)
c2 = single_link(sl_2)

plot(sl_1, c1, title='Single link for XslinkI.csv')
plot(sl_2, c2, title='Single link forXslinkII.csv')

save_points(sl_1, 'XslinkI.csv')
save_points(sl_2, 'XslinkII.csv')
save_points([[c] for c in c1], 'cslinkI.csv')
save_points([[c] for c in c2[:30]], 'cslinkII.csv')


# Some assertions
assert len(c1) == 30
assert len(c2) <= 33
assert len(c2) > len(c1)
assert sum(c1) == 15, sum(c1)
assert sl_1[:30] == sl_2[:30]
assert vary_amount(c1,c2) > 0.30, vary_amount(c1, c2)

'''
Q 1.3 - Spectral clustering
'''

def spectral_clustering(A, us, title):
    """
    Arguments:
        A: An adjacency matrix.

    Returns:
        A 0-1 vector deliminating the clusters, in the same order as the row
        in the input.
    """
    n = A.shape[0]

    L = np.eye(n)

    D = np.zeros(A.shape)
    for i in  range(n):
        D[i,i] = 1/np.sqrt(np.sum(A[i,:]))

    L -= np.dot(np.dot(D,A), D)

    eig_vals, eig_vecs = np.linalg.eig(L)
    n_eigs = len(eig_vals)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(n_eigs)]
    eig_pairs.sort(key=lambda t:t[0])

    ps =[ [x,y] for x,y in zip(eig_pairs[0][1], eig_pairs[1][1]) ]


    c, u_f = k_means(ps, us)
    
    plot(ps, c, us=us, title='Embeddings for ' + title)

    return c, u_f




A = [[0 for i in range(30)] for j in range(30)]
# Connect edges
for i in range(0,14):
    A[i][i+1] = 1
    A[i+1][i] = 1
for i in range(15,29):
    A[i][i+1] = 1
    A[i+1][i] = 1


us = None
us = [ [-0.25,0], [0, -0.25] ]

np_A = np.array(A)
sc_c1,_ = spectral_clustering(np_A, us, 'Spectral I Clustering')

A_new = copy.deepcopy(A)
# Add new edges
A_new[14][15] = 1
A_new[15][14] = 1
A_new[0][29] = 1
A_new[29][0] = 1
A_new[7][22] = 1
A_new[22][7] = 1


np_A_new = np.array(A_new)
sc_c2, _ = spectral_clustering(np_A_new, us, 'Spectral II Clustering')

graph(A, sc_c1, 'Spectral I Clustering')
graph(A_new, sc_c2, 'Spectral II Clustering')

save_points(A, 'AspectralI.csv')
save_points(A_new, 'AspectralII.csv')
save_points(us, 'spectralmeans.csv')
save_points([[c] for c in sc_c1], 'cspectralI.csv')
save_points([[c] for c in sc_c2[:30]], 'cspectralII.csv')


assert len(sc_c1) == 30
assert len(sc_c2) == 30
assert sum(sc_c1) == 15, sum(sc_c1)
assert np.linalg.norm(np_A_new) - np.linalg.norm(np_A) <= 3
assert vary_amount(sc_c1,sc_c2) > 0.30, vary_amount(sc_c1, sc_c2)

