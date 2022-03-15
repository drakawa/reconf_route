import math
import numpy as np
import itertools as it
import networkx as nx
import os
import csv
from validate_ecdg2_woacy_copy import gen_ud
from scipy.sparse.csgraph import shortest_path
from multiprocessing import Pool

NUM_POOLS = 16

def dst_transpose(src: int, num_nodes: int) -> int:
    """destination for transpose traffic.

    Example: 
        0b0010 -> 0b1000 (2 -> 8) for 16 nodes.

    Args:
        src (int): source node id
        num_nodes (int): # of nodes (must be 2^(2*i))

    Returns:
        (int): destination node id

    >>> dst_transpose(2, 16)
    8
    >>> dst_transpose(0b101010, 64)
    21
    """

    lg = int(round(math.log2(num_nodes)))

    if 1 << lg != num_nodes or lg % 2 != 0:
        print("Error: The number of nodes to be an even power of two!")
        exit(1)

    src_bit = "{:0{width}b}".format(src, width=lg)
    return int(src_bit[(lg//2):] + src_bit[:(lg//2)], 2)

def dst_reverse(src: int, num_nodes: int) -> int:
    """destination for reverse traffic.

    Example: 
        0b0010 -> 0b0100 (2 -> 4) for 16 nodes.

    Args:
        src (int): source node id
        num_nodes (int): # of nodes (must be 2^i)

    Returns:
        (int): destination node id

    >>> dst_reverse(2, 16)
    4
    >>> dst_reverse(0b001110, 64)
    28
    """

    lg = int(round(math.log2(num_nodes)))

    if 1 << lg != num_nodes:
        print("Error: The number of nodes to be an even power of two!")
        exit(1)

    src_bit = "{:0{width}b}".format(src, width=lg)
    return int(src_bit[::-1], 2)

def dst_shuffle(src: int, num_nodes: int) -> int:
    """destination for shuffle traffic.

    Example: 
        0b1110 -> 0b1101 (14 -> 13) for 16 nodes.

    Args:
        src (int): source node id
        num_nodes (int): # of nodes (must be 2^i)

    Returns:
        (int): destination node id

    >>> dst_shuffle(14, 16)
    13
    >>> dst_shuffle(0b101110, 64)
    29
    """

    lg = int(round(math.log2(num_nodes)))

    if 1 << lg != num_nodes:
        print("Error: The number of nodes to be an even power of two!")
        exit(1)

    src_bit = "{:0{width}b}".format(src, width=lg)
    return int(src_bit[1:] + src_bit[:1], 2)

def gen_TM_from_tf(tf, num_nodes):
    """generate Traffic Matrix from traffic function

    Args:
        tf (Callable): traffic function (src: int, num_nodes: int -> int)
        num_nodes (int): # of nodes (must be 2^i)

    Returns:
        (ndarray): num_nodes x num_nodes traffic matrix 
    """
    
    tm = np.zeros((num_nodes, num_nodes))
    for src in range(num_nodes):
        dst = tf(src, num_nodes)
        if src != dst:
            tm[src, dst] = 1

    return tm

def calc_coH_edge(inputs):
    n, H_i, H_j = inputs
    H_int = H_i.multiply(H_j)
    H_int_scsp = shortest_path(H_int)
    H_int_scsp_sd = H_int_scsp[0:n,n:2*n]
    num_posinf = sum(np.isposinf(H_int_scsp_sd))
    return sum(num_posinf) == 0

def Hs_iter(Hs, G):
    n = len(G)
    for i, j in it.combinations(sorted(G.nodes), 2):
        yield n, Hs[i], Hs[j]


class CompGraph:
    """compatibility graph

    Attributes:
        n (int): # of nodes
        d (int): degree
        s (int): random seed
        Hs (list of nx.DiGraph): channel dependency graphs
        coH (nx.Graph): compatibility graph among CDGs
    """
    
    def __init__(self, n, d, s):
        self.n, self.d, self.s = n, d, s
        self.Hs = None
        self.coH = None
    
    def comp_graph(self):
        n, d, s = self.n, self.d, self.s
        G = nx.random_regular_graph(d, n, s)
        G_dir = G.to_directed()

        Hs = [gen_ud(n, d, s, "hops", "./edgefiles", G, G_dir, bfs_root=i) for i in sorted(G.nodes)]

        self.Hs = Hs

        H_sorted_nodes = sorted(Hs[0].nodes, key=lambda x: (x[1],x[0]))
        Hs = [nx.to_scipy_sparse_matrix(H, nodelist=H_sorted_nodes) for H in Hs]
            
        coH = nx.Graph()
        coH.add_nodes_from(sorted(G.nodes))

        p = Pool(NUM_POOLS)
        result = p.map(calc_coH_edge, Hs_iter(Hs, G))
        coH_edges_p = [(i,j) for idx, (i, j) in enumerate(it.combinations(sorted(G.nodes), 2)) if result[idx]]

        coH.add_edges_from(coH_edges_p)
        self.coH = coH

        largest_cc = max(nx.connected_components(coH), key=len)
        print(largest_cc)
        print(self.Hs[0])

    
if __name__ == "__main__":

    import doctest
    doctest.testmod()

    print(gen_TM_from_tf(dst_shuffle, 64))
    print(np.nonzero(gen_TM_from_tf(dst_shuffle, 64)))

    compGraph = CompGraph(64, 3, 1)
    compGraph.comp_graph()
